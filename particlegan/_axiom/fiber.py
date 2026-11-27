"""Axiom L1 Law 3B / 5C conditional fiber bank and loss."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, arange, cat, einsum, eye, full, no_grad, ones, softmax, tensor, where, zeros, zeros_like
import torch.nn.functional as F

from .math_ops import (
    safe_cdist,
    smoothstep01,
    solve_spd_cholesky,
    sym_eigensplit_halley,
    sym_inv_sqrt_ns,
)

SCHEMA_VERSION = 2
MAX_STRATA = 64
MAX_RING_CAPACITY = 384
_F32, _F64, _BOOL, _LONG = torch.float32, torch.float64, torch.bool, torch.long

_PACKET_KEYS = frozenset(("schema", "d_c", "d_x", "strata"))
_SCALAR_KEYS = ("n_eff", "n_fast", "w_eff", "w_fast")
_PSD_KEYS = frozenset(("s_cc", "g_pp", "g_pp_fast", "s_xx", "s_yy"))
_RING_KEYS = ("ring_c", "ring_x", "ring_y")
_F64_KEYS = (
    "mu_c",
    "s_cc",
    "g_pp",
    "g_px",
    "g_py",
    "g_pp_fast",
    "g_px_fast",
    "g_py_fast",
    "s_xx",
    "s_yy",
) + _RING_KEYS
_STRATUM_KEYS = frozenset(("key", *_SCALAR_KEYS, *_F64_KEYS))
_BANK_DEFAULTS: dict[str, Any] = {
    "gamma_base": 0.96,
    "gamma_fast": 0.88,
    "gamma_min": 0.5,
    "innov_threshold": 2.5,
    "lin_ridge": 1e-3,
    "nl_ridge": 0.025,
    "ring_capacity": MAX_RING_CAPACITY,
    "knn_k": 20,
    "tail_leverage_gain": 0.9,
    "knn_res_weight": 0.3,
}


def _require(cond: bool, msg: str = "invalid fiber bank state") -> None:
    if not cond:
        raise ValueError(msg)


def _can_mutate_state(record, allow_update: bool) -> bool:
    in_probe = getattr(record, "_in_probe", False) or getattr(record, "_in_protected_probe", False)
    return allow_update and torch.is_grad_enabled() and not in_probe


def _clone_stratum(entry):
    return {
        **{k: float(entry[k]) for k in _SCALAR_KEYS},
        **{k: entry[k].detach().cpu().double().clone() for k in _F64_KEYS},
    }


def _whiten_cov(cov, floor: float = 1e-4):
    sym = 0.5 * (cov + cov.mT)
    diag_mean = sym.diagonal().mean().clamp_min(0).item()
    eps = max(1e-4 * max(diag_mean, floor), 1e-6) if floor > 0 else (1e-4 * diag_mean if diag_mean > 1e-12 else 1e-6)
    return sym_inv_sqrt_ns(sym, eps=eps)


def _eval_basis(c, mu_c, w_c, tail_gain: float):
    u = (c - mu_c[None]) @ w_c
    r_sq = u.square().mean(1)
    r = r_sq.clamp_min(1e-6).sqrt()
    h = (2 * (F.softplus(r - 1) - 0.3133))[:, None]
    q = (2 * (F.softplus(r - 0.8) - 0.3711) / (1 + 0.25 * r))[:, None]
    return cat((c.new_ones((c.shape[0], 1)), u, h, q * u), 1), 1 + tail_gain * r_sq.clamp_max(4.5), u


def _imq_kernel_3scale(a, b, eps):
    sq_dist = a.square().sum(-1, keepdim=True) + b.square().sum(-1)[None] - 2 * (a @ b.T)
    sq_dist = sq_dist + (sq_dist.detach().clamp_min(0) - sq_dist.detach())
    acc = sq_dist.new_zeros(sq_dist.shape)
    for mult in (0.35, 1.0, 2.8):
        e_m = eps * mult
        rad = (e_m.square() + sq_dist).sqrt()
        acc = acc + (e_m * sq_dist) / (rad * (e_m + rad)).clamp_min(1e-30)
    return acc * (1 / 3)


def _check_f64_tensor(val, shape=None):
    _require(isinstance(val, Tensor) and val.dtype == _F64 and (shape is None or tuple(val.shape) == shape))
    _require(bool(val.isfinite().all()))
    return val


def _validate_stratum(entry, d_c: int, d_x: int, p_dim: int, seen) -> None:
    _require(isinstance(entry, dict) and set(entry) == _STRATUM_KEYS)
    key = entry["key"]
    _require(isinstance(key, tuple) and bool(key) and all(type(v) is int for v in key) and key not in seen)
    seen.add(key)
    for name in _SCALAR_KEYS:
        val = entry[name]
        _require(not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(val) and val > 0)
    shapes = (
        (d_c,),
        (d_c, d_c),
        (p_dim, p_dim),
        (p_dim, d_x),
        (p_dim, d_x),
        (p_dim, p_dim),
        (p_dim, d_x),
        (p_dim, d_x),
        (d_x, d_x),
        (d_x, d_x),
    )
    for name, shp in zip(_F64_KEYS[:10], shapes):
        mat = _check_f64_tensor(entry[name], shp)
        _require(name not in _PSD_KEYS or not bool((mat.diagonal() < -1e-9).any()))
    ring_lens: set[int] = set()
    for name, cols in zip(_RING_KEYS, (d_c, d_x, d_x)):
        ring = _check_f64_tensor(entry[name])
        _require(ring.ndim == 2 and ring.shape[1] == cols and ring.shape[0] <= MAX_RING_CAPACITY)
        ring_lens.add(ring.shape[0])
    _require(len(ring_lens) == 1)


def validate_fiber_bank_state(state: Any) -> dict[str, Any]:
    """Validate a schema-2 conditional fiber bank checkpoint (fail-closed)."""
    _require(isinstance(state, dict) and set(state) == _PACKET_KEYS)
    _require(type(state["schema"]) is int and state["schema"] == SCHEMA_VERSION)
    d_c, d_x, strata = state["d_c"], state["d_x"], state["strata"]
    _require(isinstance(strata, list) and len(strata) <= MAX_STRATA)
    if d_c is None and d_x is None:
        _require(not strata)
        return state
    _require(type(d_c) is int and type(d_x) is int and d_c >= 1 and d_x >= 1)
    seen: set[tuple[int, ...]] = set()
    p_dim = AxiomCondFiberBank.basis_dim(d_c)
    for entry in strata:
        _validate_stratum(entry, d_c, d_x, p_dim, seen)
    return state


class AxiomCondFiberBank:
    """Schema-2 FP64 innovation-adaptive conditional fiber bank."""

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - set(_BANK_DEFAULTS)
        if unknown:
            raise TypeError(f"unexpected kwargs: {sorted(unknown)}")
        for name, default in _BANK_DEFAULTS.items():
            raw = kwargs.get(name, default)
            val = max(1, int(raw)) if name in ("ring_capacity", "knn_k") else float(raw)
            setattr(self, name, min(val, MAX_RING_CAPACITY) if name == "ring_capacity" else val)
        self.d_c: int | None = None
        self.d_x: int | None = None
        self.strata: dict[tuple[int, ...], dict[str, Any]] = {}

    @staticmethod
    def basis_dim(d_c: int) -> int:
        return 2 + 2 * int(d_c)

    def reset(self) -> None:
        self.d_c, self.d_x, self.strata = None, None, {}

    def copy(self) -> AxiomCondFiberBank:
        dup = AxiomCondFiberBank(**{k: getattr(self, k) for k in _BANK_DEFAULTS})
        dup.load_state_dict(self.state_dict())
        return dup

    def _ridge_vec(self, d_c: int):
        ridge = full((self.basis_dim(d_c),), self.nl_ridge * math.sqrt(d_c), dtype=_F64, device="cpu")
        ridge[: 1 + d_c] = self.lin_ridge
        return ridge

    def state_dict(self) -> dict[str, Any]:
        """Serialize bank state to a schema-2 CPU FP64 checkpoint dict."""
        strata = [{"key": tuple(int(v) for v in k), **_clone_stratum(s)} for k, s in sorted(self.strata.items())]
        return {"schema": SCHEMA_VERSION, "d_c": self.d_c, "d_x": self.d_x, "strata": strata}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate and restore a schema-2 checkpoint dict."""
        validate_fiber_bank_state(state)
        self.d_c, self.d_x = state["d_c"], state["d_x"]
        self.strata = {tuple(int(v) for v in entry["key"]): _clone_stratum(entry) for entry in state["strata"]}

    validate_state_dict = staticmethod(validate_fiber_bank_state)
    to_packet, load_packet, validate_packet = state_dict, load_state_dict, validate_state_dict

    @no_grad()
    def query_stratum(self, key: tuple[int, ...], c_g: Tensor, x_g: Tensor, y_g: Tensor) -> dict[str, Any] | None:
        """Query pre-update FP64 stratum drift, whitener, and local covariance operators."""
        d_c, d_x, dev = c_g.shape[1], x_g.shape[1], c_g.device
        st = self.strata.get(key) if (self.d_c == d_c and self.d_x == d_x) else None
        p_dim = self.basis_dim(d_c)
        if st is None or st["n_eff"] < max(p_dim + 2, 12) or st["ring_c"].shape[0] < 8:
            return None
        with torch.device("cpu"):
            n_eff, inv_we, inv_wf = st["n_eff"], 1 / max(st["w_eff"], 1), 1 / max(st["w_fast"], 1)
            mu_c, s_cc, ring_c, ring_x, ring_y = st["mu_c"], st["s_cc"], st["ring_c"], st["ring_x"], st["ring_y"]
            n_g, ring_len, ridge = c_g.shape[0], ring_c.shape[0], self._ridge_vec(d_c)
            c, x, y = c_g.detach().cpu().double(), x_g.detach().cpu().double(), y_g.detach().cpu().double()
            w_c = _whiten_cov(s_cc / max(n_eff, 1), floor=0)
            phi_q, omega, u_q = _eval_basis(c, mu_c, w_c, self.tail_leverage_gain)
            rhs_slow = cat((st["g_px"], st["g_py"]), 1) * inv_we
            sol, mat_inv = solve_spd_cholesky(st["g_pp"] * inv_we, rhs_slow, ridge=ridge, return_inv=True)
            b_x, b_y = sol[:, :d_x], sol[:, d_x:]
            b_pool = 0.5 * (b_x + b_y)
            m_x, m_y, m_pool = phi_q @ b_x, phi_q @ b_y, phi_q @ b_pool
            b_fast = solve_spd_cholesky(
                st["g_pp_fast"] * inv_wf, (st["g_px_fast"] - st["g_py_fast"]) * inv_wf, ridge=ridge
            )[0]
            delta_m_bank = 0.5 * (phi_q @ b_fast + m_x - m_y)
            if n_g >= 2 and mat_inv is not None:
                diff_xy, w_phi, proj = x - y, phi_q * omega[:, None], phi_q @ mat_inv
                loo_den = (omega.sum().item() - omega).clamp_min(1)[:, None]
                delta_m_loo = (proj @ (w_phi.T @ diff_xy) - (proj * w_phi).sum(1, keepdim=True) * diff_xy) / loo_den
            else:
                delta_m_loo = delta_m_bank
            w_mat = _whiten_cov(0.5 * (st["s_xx"] + st["s_yy"]) / n_eff)
            phi_r, _, u_r = _eval_basis(ring_c, mu_c, w_c, self.tail_leverage_gain)
            dist_qr = torch.cdist(u_q, u_r, p=2.0)
            k_eff = min(self.knn_k, max(6, ring_len // 5), ring_len)
            bw = dist_qr.topk(k_eff, 1, largest=False).values[:, -1].clamp_min(0.25 * math.sqrt(d_c))
            age = arange(ring_len - 1, -1, -1, dtype=_F64, device=u_q.device) / 128
            logits = -0.5 * (dist_qr / bw[:, None]).square() - age[None]
            alpha, m_pool_r = softmax(logits - logits.amax(1, keepdim=True), 1), phi_r @ b_pool
            res_x_r, res_y_r = ring_x - m_pool_r, ring_y - m_pool_r
            dcov_r = res_x_r[:, :, None] * res_x_r[:, None] - res_y_r[:, :, None] * res_y_r[:, None]
            delta_cov_loc = einsum("nm,mij->nij", alpha, dcov_r)
            knn_diff = alpha @ ((ring_x - phi_r @ b_x) - (ring_y - phi_r @ b_y))
            mean_err_sq = max(
                0.0, ((b_x - b_y) @ w_mat).square().sum().item() / max(1, d_x) - 2 * p_dim / max(n_eff, 1)
            )
            err_xy = max(((x - m_x).mean(0) @ w_mat).norm().item(), ((y - m_y).mean(0) @ w_mat).norm().item())
            err_c = ((c.mean(0) - mu_c) @ w_c).norm().item() * math.sqrt(n_g / max(1, d_c))
            z_innov = max(math.sqrt(n_g / max(1, d_x)) * err_xy, err_c)
            out: dict[str, Any] = {
                "n_eff": n_eff,
                "ring_len": ring_len,
                "omega": omega,
                "m_x": m_x + 0.5 * knn_diff,
                "m_pool": m_pool,
                "w_mat": w_mat,
                "delta_m_bank": delta_m_bank + self.knn_res_weight * knn_diff,
                "delta_m_loo": delta_m_loo,
                "delta_cov_loc": delta_cov_loc,
                "mean_err_sq": mean_err_sq,
                "z_innov": z_innov,
            }
        if dev.type != "cpu":
            for k, v in out.items():
                if isinstance(v, Tensor):
                    out[k] = v.to(dev)
        return out

    @no_grad()
    def update_stratum(
        self, key: tuple[int, ...], c_g: Tensor, x_g: Tensor, y_g: Tensor, *, z_innov: float = 0.0
    ) -> None:
        """Update innovation-adaptive stratum moments and FIFO ring."""
        with torch.device("cpu"):
            c, x, y = c_g.detach().cpu().double(), x_g.detach().cpu().double(), y_g.detach().cpu().double()
            n_g, d_c, d_x = c.shape[0], c.shape[1], x.shape[1]
            if n_g == 0 or (key not in self.strata and len(self.strata) >= MAX_STRATA):
                return
            if self.d_c != d_c or self.d_x != d_x:
                self.reset()
                self.d_c, self.d_x = d_c, d_x
            st, c_mean = self.strata.get(key), c.mean(0)
            c_cent = c - c_mean
            cov = c_cent.T @ c_cent
            if st is None:
                decay_mult, g_s, g_f, n_new, mu_c, s_cc = (
                    1.0,
                    0.0,
                    0.0,
                    float(n_g),
                    c_mean.clone(),
                    0.5 * (cov + cov.mT),
                )
            else:
                excess = max(0.0, float(z_innov) - self.innov_threshold)
                decay_mult = math.exp(-0.5 * excess * excess)
                g_s = self.gamma_min + (self.gamma_base - self.gamma_min) * decay_mult
                g_f = self.gamma_min + (self.gamma_fast - self.gamma_min) * decay_mult
                n_old = g_s * st["n_eff"]
                n_new, delta_c = n_old + n_g, c_mean - st["mu_c"]
                mu_c = st["mu_c"] + (n_g / n_new) * delta_c
                s_cc = g_s * st["s_cc"] + cov + (n_old * n_g / n_new) * torch.outer(delta_c, delta_c)
                s_cc = 0.5 * (s_cc + s_cc.mT)
            phi, omega, _ = _eval_basis(c, mu_c, _whiten_cov(s_cc / max(n_new, 1), floor=0), self.tail_leverage_gain)
            w_batch, w_phi, cap = omega.sum().item(), phi * omega[:, None], self.ring_capacity
            g_pp, g_px, g_py = w_phi.T @ phi, w_phi.T @ x, w_phi.T @ y
            if st is None:
                w_eff = w_fast = w_batch
                n_fast = n_new
                g_slow, g_fast = (g_pp, g_px, g_py), (g_pp.clone(), g_px.clone(), g_py.clone())
                ring = (c[-cap:].clone(), x[-cap:].clone(), y[-cap:].clone())
            else:
                w_eff, w_fast, n_fast = (
                    g_s * st["w_eff"] + w_batch,
                    g_f * st["w_fast"] + w_batch,
                    g_f * st["n_fast"] + n_g,
                )
                g_slow = (g_s * st["g_pp"] + g_pp, g_s * st["g_px"] + g_px, g_s * st["g_py"] + g_py)
                g_fast = (g_f * st["g_pp_fast"] + g_pp, g_f * st["g_px_fast"] + g_px, g_f * st["g_py_fast"] + g_py)
                cur = st["ring_c"].shape[0]
                keep = int(round(cur * max(0.25, decay_mult))) if decay_mult < 0.9 else cur
                sl = slice(-keep, None) if keep > 0 else slice(0, 0)
                ring = (
                    cat((st["ring_c"][sl], c))[-cap:],
                    cat((st["ring_x"][sl], x))[-cap:],
                    cat((st["ring_y"][sl], y))[-cap:],
                )
            inv_w = 1 / max(w_eff, 1)
            sol = solve_spd_cholesky(g_slow[0] * inv_w, cat(g_slow[1:], 1) * inv_w, ridge=self._ridge_vec(d_c))[0]
            res_x, res_y = x - phi @ sol[:, :d_x], y - phi @ sol[:, d_x:]
            s_xx, s_yy = res_x.T @ res_x, res_y.T @ res_y
            if st is not None:
                s_xx, s_yy = g_s * st["s_xx"] + s_xx, g_s * st["s_yy"] + s_yy
            s_xx, s_yy = 0.5 * (s_xx + s_xx.mT), 0.5 * (s_yy + s_yy.mT)
            vals = (n_new, n_fast, w_eff, w_fast, mu_c, s_cc, *g_slow, *g_fast, s_xx, s_yy, *ring)
            self.strata[key] = dict(zip(_SCALAR_KEYS + _F64_KEYS, vals))


def _partition_strata(batch_n: int, cond_disc):
    if cond_disc is None:
        return [((0,), arange(batch_n, dtype=_LONG))], ones((batch_n, batch_n), dtype=_BOOL)
    rows = cond_disc.detach().reshape(batch_n, -1).to("cpu", _LONG)
    buckets: dict[tuple[int, ...], list[int]] = {}
    for idx, row in enumerate(rows.tolist()):
        buckets.setdefault(tuple(int(v) for v in row), []).append(idx)
    return [(k, tensor(buckets[k], dtype=_LONG)) for k in sorted(buckets)], (rows[:, None] == rows[None]).all(-1)


def _fallback_midpoint_regression(c, z, groups):
    d_c, m_hat = c.shape[1], zeros_like(z)
    for _, idx in groups:
        z_g, z_mean = z[idx], z[idx].mean(0)
        if idx.numel() < max(4, d_c + 2):
            m_hat[idx] = z_mean
            continue
        c_cent = c[idx] - c[idx].mean(0)
        ridge = 1e-4 * c_cent.square().sum().clamp_min(1e-8).item() / max(1, d_c)
        m_hat[idx] = z_mean + c_cent @ solve_spd_cholesky(c_cent.T @ c_cent, c_cent.T @ (z_g - z_mean), ridge=ridge)[0]
    return m_hat


def _query_and_update_bank(bank, groups, c, x, y, mutate: bool, mean_blend: float):
    batch_n, d_x = x.shape
    has_op, w_op = zeros(batch_n, dtype=_BOOL), eye(d_x, dtype=_F64)[None].expand(batch_n, d_x, d_x).clone()
    delta_m, m_x, m_pool = x.new_zeros((3, batch_n, d_x)).unbind(0)
    err_cov, row_w, m_gate = x.new_zeros((batch_n, d_x, d_x)), x.new_zeros(batch_n), x.new_zeros((batch_n, 1))
    pending, max_mean_err = [], 0.0
    if bank is not None:
        for key, idx in groups:
            query, z_innov = bank.query_stratum(key, c[idx], x[idx], y[idx]), 0.0
            if query is not None:
                ring_len, omega, w_mat = query["ring_len"], query["omega"], query["w_mat"]
                gate = ring_len / (ring_len + 32)
                m_pool[idx], m_gate[idx], has_op[idx] = query["m_pool"], gate, True
                w_op[idx], m_x[idx] = w_mat[None], query["m_x"]
                dm_w = ((1 - mean_blend) * query["delta_m_loo"] + mean_blend * query["delta_m_bank"]) @ w_mat
                dm_n = dm_w.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
                delta_m[idx] = dm_w * ((3 * (dm_n / 3).tanh()) / dm_n)
                cov_w = einsum("ab,nbc,cd->nad", w_mat, query["delta_cov_loc"], w_mat)
                cov_s = 0.5 * (cov_w + cov_w.mT)
                cov_n = cov_s.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
                err_cov[idx] = cov_s * ((1.5 * (cov_n / 1.5).tanh()) / cov_n)
                row_w[idx] = gate * (omega / omega.mean().clamp_min(1e-6))
                max_mean_err, z_innov = max(max_mean_err, query["mean_err_sq"]), query["z_innov"]
            pending.append((key, idx, z_innov))
    any_op = bool(has_op.any())
    if bool(has_op.all()):
        m_hat = m_pool
    else:
        fb = _fallback_midpoint_regression(c, 0.5 * (x + y), groups)
        m_hat = (m_gate * m_pool + (1 - m_gate) * fb) if any_op else fb
    if bank is not None and mutate:
        for key, idx, z_innov in pending:
            bank.update_stratum(key, c[idx], x[idx], y[idx], z_innov=z_innov)
    err_pos, err_neg = sym_eigensplit_halley(err_cov) if any_op else (err_cov, err_cov)
    return m_hat, w_op, delta_m, m_x, err_pos, err_neg, row_w, any_op, max_mean_err


def _compute_imq_graph(c_w, y_w, same_stratum, mask_valid, valid_rows):
    dist = safe_cdist(c_w).to(c_w.dtype)
    d_max = dist.amax()
    d_unit = where(d_max > 0, dist / d_max.clamp_min(1e-30), dist)
    d_knn, idx_knn = d_unit.masked_fill(~mask_valid, float("inf")).topk(min(4, c_w.shape[0]), 1, largest=False)
    k_idx = mask_valid.sum(1).clamp(1, min(3, d_knn.shape[1])) - 1
    s_raw = d_knn.gather(1, k_idx[:, None])[:, 0]
    sigma = where(valid_rows, where(s_raw.isfinite(), s_raw, 0.0).clamp_min(0) + 1e-3 * d_unit.amax() + 1e-6, 1.0)
    k_ctx = where(mask_valid, (-d_unit.square() / (2 * sigma[:, None] * sigma[None]).clamp_min(1e-12)).exp(), 0.0)
    inv_sqrt_deg = (1 + k_ctx.sum(1)).rsqrt()
    weights = (k_ctx * inv_sqrt_deg[:, None] * inv_sqrt_deg[None]) / c_w.shape[0]
    full_mask = (same_stratum & valid_rows[None] & valid_rows[:, None]).to(y_w.dtype)
    nn_sq = 0.5 * (y_w - y_w[idx_knn[:, 0]]).square().sum(-1, keepdim=True)
    eps_sq = (full_mask @ nn_sq) / full_mask.sum(1, keepdim=True).clamp_min(1)
    y_norm_max = y_w.norm(dim=-1).amax()
    eps_floor = where(y_norm_max > 0, 1e-2 * y_norm_max, 1e-5).clamp_min(1e-12)
    return 0.5 * (weights + weights.mT), where(valid_rows[:, None], eps_sq.sqrt().clamp_min(eps_floor), 1.0)


def _resolve_fiber_bank(record, bank):
    if bank is not None or record is None:
        return bank
    cached = getattr(record, "_cond_bank_obj", None)
    if cached is not None:
        return cached
    new_bank, saved = AxiomCondFiberBank(), getattr(record, "cond_fiber_bank", None)
    if saved is not None:
        new_bank.load_state_dict(saved)
    record._cond_bank_obj = new_bank
    return new_bank


def _fiber_operator_loss(x_w, y_det, w_op, ops, mean_w: float, cov_w: float, out_dt):
    dev, dt, inv_d = x_w.device, x_w.dtype, 1 / max(1, x_w.shape[-1])
    delta_m, m_x, err_pos, err_neg, row_w = (t.to(dev, dt) for t in ops)
    loss = x_w.new_zeros((), dtype=out_dt)
    if mean_w > 0:
        proj_diff = einsum("nij,nj->ni", w_op, x_w - y_det)
        loss = loss + mean_w * (row_w * (proj_diff * delta_m).sum(-1) * inv_d).mean().to(out_dt)
    if cov_w > 0:
        res_x = einsum("nij,nj->ni", w_op, x_w - m_x)
        excess = F.relu(res_x.square().sum(-1, keepdim=True) * inv_d - (1 + 5 * math.sqrt(2 * inv_d)))
        res_soft = res_x * (1 + 0.25 * excess.square() / (1 + excess)).rsqrt()
        q_pos = einsum("ni,nij,nj->n", res_x, err_pos, res_x)
        q_neg = einsum("ni,nij,nj->n", res_soft, err_neg, res_soft)
        loss = loss + cov_w * (row_w * 0.5 * (q_pos + q_neg) * inv_d).mean().to(out_dt)
    return loss


def _update_cond_ema(record, weights, p_xy, p_xx, p_yy, max_mean_err: float) -> None:
    num = (weights * (2 * p_xy.detach() - p_xx.detach() - p_yy)).sum()
    rel_imq = max(-0.5, min(0.5, (num / (weights * p_yy).sum().clamp_min(1e-8)).item()))
    rel_cond = rel_imq + min(0.5, 0.11 * max(0.0, max_mean_err - 0.1))
    record.cond_ema = (
        0.99 * float(record.cond_ema) + 0.01 * rel_cond if record.cond_initialized else max(0.18, rel_cond)
    )
    record.cond_initialized = True
    if hasattr(record, "step_scale"):
        record.step_scale = float(1 - 0.85 * smoothstep01((0.05 - max(0.0, float(record.cond_ema))) / 0.024))


def conditional_fiber_loss(
    x_fake: Tensor,
    y_real: Tensor,
    c_cont: Tensor,
    *,
    cond_disc: Tensor | None = None,
    record: Any = None,
    bank: AxiomCondFiberBank | None = None,
    mean_op_weight: float = 0.36,
    cov_op_weight: float = 0.35,
    imq_active_scale: float = 0.40,
    bank_mean_blend: float = 0.75,
    allow_state_update: bool = True,
) -> Tensor:
    """Law 3B conditional fiber loss combining 3-scale IMQ and bank drift/covariance operators."""
    batch_n = x_fake.shape[0]
    if batch_n <= 1 or y_real.shape[0] != batch_n or c_cont.shape[0] != batch_n:
        return x_fake[:0].sum()
    x_flat, y_flat = x_fake.reshape(batch_n, -1), y_real.detach().reshape(batch_n, -1)
    dev, d_raw = x_flat.device, x_flat.shape[-1]
    dim, work_dt = max(1, d_raw), (_F64 if x_flat.dtype == _F64 else _F32)
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(dim + 1))
    mutate, resolved_bank = _can_mutate_state(record, allow_state_update), _resolve_fiber_bank(record, bank)
    with no_grad():
        groups, same_cpu = _partition_strata(batch_n, cond_disc)
        if not any(idx.numel() >= 2 for _, idx in groups):
            return x_fake[:0].sum()
        c_w, x_det, y_det = (
            c_cont.detach().reshape(batch_n, -1).to(work_dt),
            x_flat.detach().to(work_dt),
            y_flat.to(work_dt),
        )
        c_cpu, x_cpu, y_cpu = cat((c_w, x_det, y_det), 1).to("cpu", _F64).split((c_w.shape[1], d_raw, d_raw), 1)
        m_hat_c, w_op_c, *ops, any_op, max_err = _query_and_update_bank(
            resolved_bank, groups, c_cpu, x_cpu, y_cpu, mutate, float(bank_mean_blend)
        )
        m_hat, w_op = m_hat_c.to(dev, work_dt), w_op_c.to(dev, work_dt)
        w_imq = w_op / (w_op.square().sum((-2, -1), keepdim=True) / dim).sqrt().clamp_min(1e-8)
        same_stratum = same_cpu.to(dev)
        off_diag = same_stratum.clone().fill_diagonal_(False)
        valid_rows = off_diag.any(1)
        y_w = einsum("nij,nj->ni", w_imq, y_det - m_hat)
        weights, eps_imq = _compute_imq_graph(
            c_w, y_w, same_stratum, off_diag & valid_rows[:, None] & valid_rows[None], valid_rows
        )
        p_yy = _imq_kernel_3scale(y_w, y_w, eps_imq)
    x_work = x_flat.to(work_dt)
    x_w = einsum("nij,nj->ni", w_imq, x_work - m_hat)
    p_xy, p_xx = _imq_kernel_3scale(x_w, y_w, eps_imq), _imq_kernel_3scale(x_w, x_w, eps_imq)
    diff_imq = p_xy.clamp_max(fmax) - 0.5 * p_xx.clamp_max(fmax) - 0.5 * p_yy.clamp_max(fmax)
    loss_imq = (weights.to(x_flat.dtype) * diff_imq.to(x_flat.dtype)).sum()
    has_ops = any_op and (mean_op_weight > 0 or cov_op_weight > 0)
    loss_op = (
        _fiber_operator_loss(x_work, y_det, w_op, ops, float(mean_op_weight), float(cov_op_weight), x_flat.dtype)
        if has_ops
        else x_flat.new_zeros(())
    )
    if mutate and hasattr(record, "cond_initialized"):
        with no_grad():
            _update_cond_ema(record, weights, p_xy, p_xx, p_yy, max_err)
    return ((float(imq_active_scale) if has_ops else 1) * loss_imq + loss_op).clamp(-fmax, fmax).to(x_flat.dtype)


__all__ = ["AxiomCondFiberBank", "SCHEMA_VERSION", "conditional_fiber_loss", "validate_fiber_bank_state"]
