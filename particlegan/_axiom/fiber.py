"""Axiom L1 Law 3B / 5C Conditional Fiber Bank & Loss.

Dimension-agnostic FP64 conditional operator bank and paired-context fiber loss
built on ``math_ops`` spectral primitives (Halley eigensplit, coupled
Newton-Schulz inverse square root, and Cholesky SPD solves) with exact joint
``O(d_c) x O(d_x)`` equivariance, clock-freedom, and zero global RNG mutation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from .math_ops import (
    safe_cdist,
    smoothstep01,
    solve_spd_cholesky,
    sym_eigensplit_halley,
    sym_inv_sqrt_ns,
)

SCHEMA_VERSION: int = 2
_MAX_STRATA: int = 64
_MAX_RING: int = 384

_PACKET_KEYS = frozenset({"schema", "d_c", "d_x", "strata"})
_SCALAR_KEYS = ("n_eff", "n_fast", "w_eff", "w_fast")
_PSD_KEYS = frozenset({"s_cc", "g_pp", "g_pp_fast", "s_xx", "s_yy"})
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
    *_RING_KEYS,
)
_STRATUM_KEYS = frozenset({"key", *_SCALAR_KEYS, *_F64_KEYS})
_CFG_KEYS = (
    "gamma_base",
    "gamma_fast",
    "gamma_min",
    "innov_threshold",
    "lin_ridge",
    "nl_ridge",
    "ring_capacity",
    "knn_k",
    "tail_leverage_gain",
    "knn_res_weight",
)


class _BankStep(NamedTuple):
    m_hat: torch.Tensor
    op_w: torch.Tensor
    op_dm: torch.Tensor
    op_mx: torch.Tensor
    op_ep: torch.Tensor
    op_en: torch.Tensor
    op_rw: torch.Tensor
    any_op: bool
    r_worst: float


def _sym_whitener(cov_f64: torch.Tensor, *, scale_floor: float = 1e-4) -> torch.Tensor:
    """Dimension-agnostic FP64 whitener ``(sym(cov) + eps*I)^{-1/2}``."""
    sym = 0.5 * (cov_f64 + cov_f64.T)
    diag_mean = float(torch.diagonal(sym).mean().clamp_min(0.0).item())
    if scale_floor > 0.0:
        eps = max(1e-4 * max(diag_mean, scale_floor), 1e-6)
    else:
        eps = 1e-4 * diag_mean if diag_mean > 1e-12 else 1e-6
    return sym_inv_sqrt_ns(sym, eps=eps)


def _build_basis(
    c: torch.Tensor, mu: torch.Tensor, wc: torch.Tensor, gain: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``O(d_c)``-equivariant basis ``phi(c)``, leverage ``omega(c)``, and ``u``."""
    u = (c - mu.unsqueeze(0)) @ wc
    ones = torch.ones((c.shape[0], 1), dtype=torch.float64, device=c.device)
    r2 = u.square().mean(dim=1)
    r = r2.clamp_min(1e-6).sqrt()
    om = 1.0 + float(gain) * r2.clamp_max(4.5)
    h0 = (2.0 * (F.softplus(r - 1.0) - 0.3133)).unsqueeze(1)
    g1 = (2.0 * (F.softplus(r - 0.8) - 0.3711) / (1.0 + 0.25 * r)).unsqueeze(1)
    return torch.cat((ones, u, h0, g1 * u), dim=1), om, u


def _odd_sat(t: torch.Tensor, tau: float, dims: Tuple[int, ...]) -> torch.Tensor:
    """Smooth odd norm saturation ``(tau * tanh(||v|| / tau) / ||v||) * v``."""
    nrm = t.square().sum(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    return t * ((float(tau) * torch.tanh(nrm / float(tau))) / nrm)


def _imq_3scale(a: torch.Tensor, b: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    """Three-scale cancellation-free IMQ potential ``ell*||a-b||^2 / (r*(ell+r))``."""
    a2 = a.square().sum(dim=-1, keepdim=True)
    b2 = b.square().sum(dim=-1).unsqueeze(0)
    sq = a2 + b2 - 2.0 * (a @ b.T)
    sq = sq + (sq.detach().clamp_min(0.0) - sq.detach())
    out = sq.new_zeros(sq.shape)
    for mult in (0.35, 1.0, 2.8):
        es = ell * mult
        rad = torch.sqrt(es.square() + sq)
        out = out + (es * sq) / (rad * (es + rad)).clamp_min(1e-30)
    return out * (1.0 / 3.0)


def _check_f64(val: Any, shape: Optional[Tuple[int, ...]], name: str) -> torch.Tensor:
    if not isinstance(val, torch.Tensor) or val.dtype != torch.float64:
        raise ValueError(f"fiber bank tensor {name} must be float64")
    if shape is not None and tuple(val.shape) != shape:
        raise ValueError(
            f"fiber bank tensor {name} shape {tuple(val.shape)} != {shape}"
        )
    if not bool(torch.isfinite(val).all()):
        raise ValueError(f"fiber bank tensor {name} has non-finite values")
    return val


def _check_stratum_tensors(st: Dict[str, Any], dc: int, dx: int, p: int) -> None:
    shapes = (
        ("mu_c", (dc,)),
        ("s_cc", (dc, dc)),
        ("g_pp", (p, p)),
        ("g_px", (p, dx)),
        ("g_py", (p, dx)),
        ("g_pp_fast", (p, p)),
        ("g_px_fast", (p, dx)),
        ("g_py_fast", (p, dx)),
        ("s_xx", (dx, dx)),
        ("s_yy", (dx, dx)),
    )
    for name, shp in shapes:
        t = _check_f64(st[name], shp, name)
        if name in _PSD_KEYS and bool((torch.diagonal(t) < -1e-9).any()):
            raise ValueError(f"fiber bank PSD tensor {name} has negative diagonal")
    lens = set()
    for name, cols in (("ring_c", dc), ("ring_x", dx), ("ring_y", dx)):
        r = _check_f64(st[name], None, name)
        if r.ndim != 2 or r.shape[1] != cols or r.shape[0] > _MAX_RING:
            raise ValueError(f"invalid fiber bank ring {name}: {tuple(r.shape)}")
        lens.add(int(r.shape[0]))
    if len(lens) != 1:
        raise ValueError("mismatched fiber bank FIFO ring lengths")


def _check_stratum(
    st: Any, dc: int, dx: int, p: int, seen: set[Tuple[int, ...]]
) -> None:
    if not isinstance(st, dict) or set(st) != _STRATUM_KEYS:
        raise ValueError("invalid fiber bank stratum keys")
    key = st["key"]
    if (
        not isinstance(key, tuple)
        or not key
        or any(type(v) is not int for v in key)
        or key in seen
    ):
        raise ValueError("invalid or duplicate fiber bank stratum key")
    seen.add(key)
    for sk in _SCALAR_KEYS:
        sv = st[sk]
        if isinstance(sv, bool) or not isinstance(sv, (int, float)):
            raise ValueError(f"invalid fiber bank scalar {sk}: {sv!r}")
        if not math.isfinite(sv) or float(sv) <= 0.0:
            raise ValueError(f"non-positive fiber bank scalar {sk}: {sv!r}")
    _check_stratum_tensors(st, dc, dx, p)


def validate_fiber_bank_state(state: Any) -> Dict[str, Any]:
    """Strictly validate a schema-2 ``AxiomCondFiberBank`` checkpoint dictionary."""
    if not isinstance(state, dict) or set(state) != _PACKET_KEYS:
        raise ValueError("invalid fiber bank checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported fiber bank schema {state['schema']!r}")
    dc, dx, strata = state["d_c"], state["d_x"], state["strata"]
    if not isinstance(strata, list) or len(strata) > _MAX_STRATA:
        raise ValueError("invalid fiber bank strata list")
    if dc is None and dx is None:
        if strata:
            raise ValueError("uninitialized fiber bank cannot contain strata")
        return state
    if type(dc) is not int or type(dx) is not int or dc < 1 or dx < 1:
        raise ValueError(f"invalid fiber bank dimensions d_c={dc!r}, d_x={dx!r}")
    seen: set[Tuple[int, ...]] = set()
    p = AxiomCondFiberBank.basis_dim(dc)
    for entry in strata:
        _check_stratum(entry, dc, dx, p, seen)
    return state


class AxiomCondFiberBank:
    """Law 3B / 5C schema-2 FP64 conditional operator bank and FIFO ring."""

    def __init__(
        self,
        *,
        gamma_base: float = 0.96,
        gamma_fast: float = 0.88,
        gamma_min: float = 0.50,
        innov_threshold: float = 2.5,
        lin_ridge: float = 1e-3,
        nl_ridge: float = 0.025,
        ring_capacity: int = _MAX_RING,
        knn_k: int = 20,
        tail_leverage_gain: float = 0.90,
        knn_res_weight: float = 0.30,
    ) -> None:
        self.gamma_base = float(gamma_base)
        self.gamma_fast = float(gamma_fast)
        self.gamma_min = float(gamma_min)
        self.innov_threshold = float(innov_threshold)
        self.lin_ridge = float(lin_ridge)
        self.nl_ridge = float(nl_ridge)
        self.ring_capacity = max(1, min(int(ring_capacity), _MAX_RING))
        self.knn_k = max(1, int(knn_k))
        self.tail_leverage_gain = float(tail_leverage_gain)
        self.knn_res_weight = float(knn_res_weight)
        self.d_c: Optional[int] = None
        self.d_x: Optional[int] = None
        self.strata: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    @staticmethod
    def basis_dim(d_c: int) -> int:
        """Dimension-agnostic ``O(d_c)``-equivariant basis width ``2 + 2 * d_c``."""
        return 2 + 2 * int(d_c)

    def reset(self) -> None:
        """Clear all accumulated stratum moments and FIFO rings."""
        self.d_c, self.d_x, self.strata = None, None, {}

    def copy(self) -> "AxiomCondFiberBank":
        """Return an independent deep copy of this bank and its checkpointed state."""
        dup = AxiomCondFiberBank(**{k: getattr(self, k) for k in _CFG_KEYS})
        dup.load_state_dict(self.state_dict())
        return dup

    def _ridge(self, dc: int) -> torch.Tensor:
        r = torch.full(
            (self.basis_dim(dc),),
            self.nl_ridge * math.sqrt(float(dc)),
            dtype=torch.float64,
            device="cpu",
        )
        r[: 1 + dc] = self.lin_ridge
        return r

    def _knn_weights(self, uq: torch.Tensor, ur: torch.Tensor, dc: int) -> torch.Tensor:
        m = int(ur.shape[0])
        dqr = torch.cdist(uq, ur, p=2.0)
        k = min(self.knn_k, max(6, m // 5), m)
        hq = dqr.topk(k, dim=1, largest=False).values[:, -1]
        hq = hq.clamp_min(0.25 * math.sqrt(float(dc)))
        age = torch.arange(m - 1, -1, -1, dtype=torch.float64, device=uq.device)
        lg = -0.5 * (dqr / hq.unsqueeze(1)).square() - (age / 128.0).unsqueeze(0)
        return torch.softmax(lg - lg.amax(dim=1, keepdim=True), dim=1)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize bank state as a schema-2 CPU float64 checkpoint dictionary."""
        entries = []
        for key, st in sorted(self.strata.items()):
            rec: Dict[str, Any] = {"key": tuple(int(v) for v in key)}
            rec.update({sk: float(st[sk]) for sk in _SCALAR_KEYS})
            rec.update(
                {
                    tk: st[tk].detach().cpu().to(torch.float64).clone()
                    for tk in _F64_KEYS
                }
            )
            entries.append(rec)
        return {
            "schema": SCHEMA_VERSION,
            "d_c": self.d_c,
            "d_x": self.d_x,
            "strata": entries,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Atomically validate and restore a schema-2 checkpoint dictionary."""
        validate_fiber_bank_state(state)
        restored: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for e in state["strata"]:
            st: Dict[str, Any] = {sk: float(e[sk]) for sk in _SCALAR_KEYS}
            st.update(
                {tk: e[tk].detach().cpu().to(torch.float64).clone() for tk in _F64_KEYS}
            )
            restored[tuple(int(v) for v in e["key"])] = st
        self.d_c, self.d_x, self.strata = state["d_c"], state["d_x"], restored

    validate_state_dict = staticmethod(validate_fiber_bank_state)
    to_packet = state_dict
    load_packet = load_state_dict
    validate_packet = validate_state_dict

    def _decay_context(
        self, st: Optional[Dict[str, Any]], c: torch.Tensor, zi: float
    ) -> Tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
        ng = float(c.shape[0])
        cbar = c.mean(dim=0)
        dev = c - cbar.unsqueeze(0)
        cov = dev.T @ dev
        if st is None:
            return 1.0, 0.0, 0.0, ng, cbar.clone(), 0.5 * (cov + cov.T)
        ex = max(0.0, float(zi) - self.innov_threshold)
        dm = math.exp(-0.5 * ex * ex)
        gs = self.gamma_min + (self.gamma_base - self.gamma_min) * dm
        gf = self.gamma_min + (self.gamma_fast - self.gamma_min) * dm
        n_old = gs * float(st["n_eff"])
        n_new = n_old + ng
        dc = cbar - st["mu_c"]
        mu_new = st["mu_c"] + (ng / n_new) * dc
        scc = gs * st["s_cc"] + cov + (n_old * ng / n_new) * torch.outer(dc, dc)
        return dm, gs, gf, n_new, mu_new, 0.5 * (scc + scc.T)

    def _update_rings(
        self,
        st: Optional[Dict[str, Any]],
        c: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        dm: float,
    ) -> Dict[str, torch.Tensor]:
        cap = self.ring_capacity
        if st is None:
            return {k: t[-cap:].clone() for k, t in zip(_RING_KEYS, (c, x, y))}
        cur = int(st["ring_c"].shape[0])
        keep = int(round(cur * max(0.25, dm))) if dm < 0.9 else cur
        sl = slice(-keep, None) if keep > 0 else slice(0, 0)
        return {
            k: torch.cat((st[k][sl], t), dim=0)[-cap:]
            for k, t in zip(_RING_KEYS, (c, x, y))
        }

    @torch.no_grad()
    def query_stratum(
        self,
        key: Tuple[int, ...],
        c_g: torch.Tensor,
        x_g: torch.Tensor,
        y_g: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        """Query pre-update FP64 conditional operators for stratum ``key``."""
        dc, dx, out_dev = int(c_g.shape[1]), int(x_g.shape[1]), c_g.device
        st = self.strata.get(key) if (self.d_c == dc and self.d_x == dx) else None
        p = self.basis_dim(dc)
        if st is None or float(st["n_eff"]) < float(max(p + 2, 12)):
            return None
        rlen = int(st["ring_c"].shape[0])
        if rlen < 8:
            return None

        with torch.device("cpu"):
            n_eff, ng = float(st["n_eff"]), int(c_g.shape[0])
            c, x, y = (t.detach().cpu().to(torch.float64) for t in (c_g, x_g, y_g))
            mu = st["mu_c"]
            wc = _sym_whitener(st["s_cc"] / max(n_eff, 1.0), scale_floor=0.0)
            phi_q, om_q, u_q = _build_basis(c, mu, wc, self.tail_leverage_gain)
            iw = 1.0 / max(float(st["w_eff"]), 1.0)
            iwf = 1.0 / max(float(st["w_fast"]), 1.0)
            ridge = self._ridge(dc)

            rhs = torch.cat((st["g_px"], st["g_py"]), dim=1) * iw
            sol, mis = solve_spd_cholesky(
                st["g_pp"] * iw, rhs, ridge=ridge, return_inv=True
            )
            bxs, bys = sol[:, :dx], sol[:, dx:]
            bp = 0.5 * (bxs + bys)
            mxs, mys, mpq = phi_q @ bxs, phi_q @ bys, phi_q @ bp

            rhs_f = (st["g_px_fast"] - st["g_py_fast"]) * iwf
            dbf, _ = solve_spd_cholesky(st["g_pp_fast"] * iwf, rhs_f, ridge=ridge)
            dmb_p = 0.5 * (phi_q @ dbf) + 0.5 * (mxs - mys)
            if ng >= 2 and mis is not None:
                u_err, wphi, proj = x - y, phi_q * om_q.unsqueeze(1), phi_q @ mis
                h_ii = (proj * wphi).sum(dim=1, keepdim=True)
                den = (float(om_q.sum().item()) - om_q).clamp_min(1.0).unsqueeze(1)
                dml = (proj @ (wphi.T @ u_err) - h_ii * u_err) / den
            else:
                dml = dmb_p

            wm = _sym_whitener(0.5 * (st["s_xx"] + st["s_yy"]) / n_eff)
            rc, rxr, ryr = st["ring_c"], st["ring_x"], st["ring_y"]
            phi_r, _, u_r = _build_basis(rc, mu, wc, self.tail_leverage_gain)
            al = self._knn_weights(u_q, u_r, dc)
            mpr = phi_r @ bp
            rrx, rry = rxr - mpr, ryr - mpr
            dcov_r = rrx.unsqueeze(2) * rrx.unsqueeze(1) - rry.unsqueeze(
                2
            ) * rry.unsqueeze(1)
            dcov = torch.einsum("nm,mij->nij", al, dcov_r)
            kdiff = al @ ((rxr - phi_r @ bxs) - (ryr - phi_r @ bys))
            dmb, m_x = dmb_p + self.knn_res_weight * kdiff, mxs + 0.5 * kdiff

            raw_err = float(((bxs - bys) @ wm).square().sum().item()) / float(
                max(1, dx)
            )
            merr = max(0.0, raw_err - 2.0 * float(p) / max(n_eff, 1.0))
            ex = float(((x - mxs).mean(dim=0) @ wm).norm().item())
            ey = float(((y - mys).mean(dim=0) @ wm).norm().item())
            cj = float(((c.mean(dim=0) - mu) @ wc).norm().item())
            zi = max(
                math.sqrt(float(ng) / float(max(1, dx))) * max(ex, ey),
                cj * math.sqrt(float(ng) / float(max(1, dc))),
            )
            out: Dict[str, Any] = {
                "n_eff": n_eff,
                "ring_len": rlen,
                "omega": om_q,
                "m_x": m_x,
                "m_pool": mpq,
                "w_mat": wm,
                "delta_m_bank": dmb,
                "delta_m_loo": dml,
                "delta_cov_loc": dcov,
                "mean_err_sq": merr,
                "z_innov": zi,
            }
        if out_dev.type != "cpu":
            return {
                k: (v.to(device=out_dev) if isinstance(v, torch.Tensor) else v)
                for k, v in out.items()
            }
        return out

    @torch.no_grad()
    def update_stratum(
        self,
        key: Tuple[int, ...],
        c_g: torch.Tensor,
        x_g: torch.Tensor,
        y_g: torch.Tensor,
        *,
        z_innov: float = 0.0,
    ) -> None:
        """Update stratum moments and FIFO ring with clock-free innovation decay."""
        with torch.device("cpu"):
            c, x, y = (t.detach().cpu().to(torch.float64) for t in (c_g, x_g, y_g))
            ng, dc, dx = int(c.shape[0]), int(c.shape[1]), int(x.shape[1])
            if ng == 0 or (key not in self.strata and len(self.strata) >= _MAX_STRATA):
                return
            if self.d_c != dc or self.d_x != dx:
                self.reset()
                self.d_c, self.d_x = dc, dx

            st = self.strata.get(key)
            dm, gs, gf, n_new, mu_new, scc_new = self._decay_context(st, c, z_innov)
            wc = _sym_whitener(scc_new / max(n_new, 1.0), scale_floor=0.0)
            phi, om, _ = _build_basis(c, mu_new, wc, self.tail_leverage_gain)
            wb, wphi = float(om.sum().item()), phi * om.unsqueeze(1)
            bg = {"g_pp": wphi.T @ phi, "g_px": wphi.T @ x, "g_py": wphi.T @ y}

            if st is None:
                we, wf, nf = wb, wb, n_new
                upd = bg
                updf = {f"{k}_fast": v.clone() for k, v in bg.items()}
            else:
                we, wf = gs * float(st["w_eff"]) + wb, gf * float(st["w_fast"]) + wb
                nf = gf * float(st["n_fast"]) + float(ng)
                upd = {k: gs * st[k] + v for k, v in bg.items()}
                updf = {f"{k}_fast": gf * st[f"{k}_fast"] + v for k, v in bg.items()}

            rhs = torch.cat((upd["g_px"], upd["g_py"]), dim=1) / max(we, 1.0)
            sol, _ = solve_spd_cholesky(
                upd["g_pp"] / max(we, 1.0), rhs, ridge=self._ridge(dc)
            )
            rx, ry = x - phi @ sol[:, :dx], y - phi @ sol[:, dx:]
            rxc, ryc = rx.T @ rx, ry.T @ ry
            sxx = rxc if st is None else gs * st["s_xx"] + rxc
            syy = ryc if st is None else gs * st["s_yy"] + ryc
            rings = self._update_rings(st, c, x, y, dm)

            self.strata[key] = {
                "n_eff": n_new,
                "n_fast": nf,
                "w_eff": we,
                "w_fast": wf,
                "mu_c": mu_new,
                "s_cc": scc_new,
                **upd,
                **updf,
                "s_xx": 0.5 * (sxx + sxx.T),
                "s_yy": 0.5 * (syy + syy.T),
                **rings,
            }


def _partition_strata_cpu(
    bn: int, cond_disc: Optional[torch.Tensor]
) -> Tuple[List[Tuple[Tuple[int, ...], torch.Tensor]], torch.Tensor]:
    cpu = torch.device("cpu")
    if cond_disc is None:
        idx = torch.arange(bn, dtype=torch.long, device=cpu)
        return [((0,), idx)], torch.ones((bn, bn), dtype=torch.bool, device=cpu)
    rows = cond_disc.detach().reshape(bn, -1).to(device=cpu, dtype=torch.int64)
    bkt: Dict[Tuple[int, ...], List[int]] = {}
    for i, row in enumerate(rows.tolist()):
        bkt.setdefault(tuple(int(v) for v in row), []).append(i)
    groups = [
        (k, torch.tensor(bkt[k], dtype=torch.long, device=cpu)) for k in sorted(bkt)
    ]
    return groups, (rows.unsqueeze(1) == rows.unsqueeze(0)).all(dim=-1)


def _pooled_mean_fallback(
    c: torch.Tensor, z: torch.Tensor, groups: List[Tuple[Tuple[int, ...], torch.Tensor]]
) -> torch.Tensor:
    dc, mhat = int(c.shape[1]), torch.zeros_like(z)
    for _, idx in groups:
        zg, zm = z[idx], z[idx].mean(dim=0, keepdim=True)
        if int(idx.numel()) < max(4, dc + 2):
            mhat[idx] = zm
            continue
        dcv = c[idx] - c[idx].mean(dim=0, keepdim=True)
        rg = 1e-4 * float(dcv.square().sum().clamp_min(1e-8).item()) / float(max(1, dc))
        slope, _ = solve_spd_cholesky(dcv.T @ dcv, dcv.T @ (zg - zm), ridge=rg)
        mhat[idx] = zm + dcv @ slope
    return mhat


def _query_and_update_bank(
    bank: Optional[AxiomCondFiberBank],
    groups: List[Tuple[Tuple[int, ...], torch.Tensor]],
    c: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    can_mut: bool,
    blend: float,
) -> _BankStep:
    bn, d, cpu = int(x.shape[0]), int(x.shape[1]), torch.device("cpu")
    has_op = torch.zeros(bn, dtype=torch.bool, device=cpu)
    eye = torch.eye(d, dtype=torch.float64, device=cpu)
    op_w = eye.unsqueeze(0).expand(bn, d, d).clone()
    op_dm = torch.zeros((bn, d), dtype=torch.float64, device=cpu)
    op_mx = torch.zeros((bn, d), dtype=torch.float64, device=cpu)
    op_ec = torch.zeros((bn, d, d), dtype=torch.float64, device=cpu)
    op_rw = torch.zeros(bn, dtype=torch.float64, device=cpu)
    mp_cpu = torch.zeros((bn, d), dtype=torch.float64, device=cpu)
    mp_g = torch.zeros((bn, 1), dtype=torch.float64, device=cpu)
    pend: List[Tuple[Tuple[int, ...], torch.Tensor, float]] = []
    r_worst = 0.0

    if bank is not None:
        for key, ic in groups:
            q = bank.query_stratum(key, c[ic], x[ic], y[ic])
            zi = 0.0
            if q is not None:
                wg = float(q["ring_len"]) / (float(q["ring_len"]) + 32.0)
                wm, om = q["w_mat"], q["omega"]
                mp_cpu[ic], mp_g[ic] = q["m_pool"], wg
                dm = (1.0 - blend) * q["delta_m_loo"] + blend * q["delta_m_bank"]
                ec = torch.einsum("ab,nbc,cd->nad", wm, q["delta_cov_loc"], wm)
                has_op[ic], op_w[ic] = True, wm.unsqueeze(0)
                op_dm[ic] = _odd_sat(dm @ wm, 3.0, (-1,))
                op_mx[ic] = q["m_x"]
                op_ec[ic] = _odd_sat(0.5 * (ec + ec.transpose(-1, -2)), 1.5, (-2, -1))
                op_rw[ic] = wg * (om / om.mean().clamp_min(1e-6))
                zi, r_worst = float(q["z_innov"]), max(r_worst, float(q["mean_err_sq"]))
            pend.append((key, ic, zi))

    any_op = bool(has_op.any())
    if bool(has_op.all()):
        m_hat = mp_cpu
    else:
        fb = _pooled_mean_fallback(c, 0.5 * (x + y), groups)
        m_hat = (mp_g * mp_cpu + (1.0 - mp_g) * fb) if any_op else fb

    if bank is not None and can_mut:
        for key, ic, zi in pend:
            bank.update_stratum(key, c[ic], x[ic], y[ic], z_innov=zi)

    op_ep, op_en = sym_eigensplit_halley(op_ec) if any_op else (op_ec, op_ec)
    return _BankStep(m_hat, op_w, op_dm, op_mx, op_ep, op_en, op_rw, any_op, r_worst)


def _imq_geometry(
    cw: torch.Tensor,
    yw: torch.Tensor,
    same_t: torch.Tensor,
    m_val: torch.Tensor,
    vrow: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bn = int(cw.shape[0])
    dcm = safe_cdist(cw).to(cw.dtype)
    dmax = dcm.amax()
    du = torch.where(dmax > 0.0, dcm / dmax.clamp_min(1e-30), dcm)
    ds, si = du.masked_fill(~m_val, float("inf")).topk(
        k=min(4, bn), dim=1, largest=False
    )
    ni = m_val.sum(dim=1)
    ki = torch.minimum(torch.full_like(ni, min(3, ds.shape[1])), ni.clamp_min(1)) - 1
    sr = ds.gather(1, ki.unsqueeze(1)).squeeze(1)
    sraw = torch.where(torch.isfinite(sr), sr, torch.zeros_like(sr)).clamp_min(0.0)
    sig = torch.where(vrow, sraw + 1e-3 * du.amax() + 1e-6, torch.ones_like(sr))

    den = (2.0 * sig.unsqueeze(1) * sig.unsqueeze(0)).clamp_min(1e-12)
    kc = torch.where(m_val, torch.exp(-du.square() / den), 0.0)
    isd = torch.rsqrt(1.0 + kc.sum(dim=1))
    w0 = (kc * isd.unsqueeze(1) * isd.unsqueeze(0)) / float(bn)
    wi = 0.5 * (w0 + w0.T)

    fm = (same_t & vrow.unsqueeze(0) & vrow.unsqueeze(1)).to(yw.dtype)
    innov_sq = 0.5 * (yw - yw[si[:, 0]]).square().sum(dim=-1, keepdim=True)
    ell_sq = (fm @ innov_sq) / fm.sum(dim=1, keepdim=True).clamp_min(1.0)
    yl2 = yw.norm(dim=-1).amax()
    efloor = torch.where(yl2 > 0.0, 1e-2 * yl2, yl2.new_tensor(1e-5)).clamp_min(1e-12)
    ell_f = torch.where(
        vrow.unsqueeze(1), ell_sq.sqrt().clamp_min(efloor), torch.ones_like(ell_sq)
    )
    return wi, ell_f


def _resolve_bank(
    record: Any, bank: Optional[AxiomCondFiberBank]
) -> Optional[AxiomCondFiberBank]:
    if bank is not None:
        return bank
    if record is None:
        return None
    existing = getattr(record, "_cond_bank_obj", None)
    if existing is not None:
        return existing
    fresh = AxiomCondFiberBank()
    saved = getattr(record, "cond_fiber_bank", None)
    if saved is not None:
        fresh.load_state_dict(saved)
    record._cond_bank_obj = fresh
    return fresh


def _operator_terms(
    xw: torch.Tensor,
    yd: torch.Tensor,
    op_w: torch.Tensor,
    step: _BankStep,
    *,
    mean_w: float,
    cov_w: float,
    out_dt: torch.dtype,
) -> torch.Tensor:
    dev, dt = xw.device, xw.dtype
    inv_d = 1.0 / float(max(1, int(xw.shape[-1])))
    orw = step.op_rw.to(device=dev, dtype=dt)
    l_op = xw.new_zeros((), dtype=out_dt)
    if mean_w > 0.0:
        dxw = torch.einsum("nij,nj->ni", op_w, xw - yd)
        odm = step.op_dm.to(device=dev, dtype=dt)
        l_op = l_op + float(mean_w) * (orw * (dxw * odm).sum(-1) * inv_d).mean().to(
            out_dt
        )
    if cov_w > 0.0:
        rxw = torch.einsum("nij,nj->ni", op_w, xw - step.op_mx.to(device=dev, dtype=dt))
        u_sq = rxw.square().sum(dim=-1, keepdim=True) * inv_d
        ex_u = F.relu(u_sq - (1.0 + 5.0 * math.sqrt(2.0 * inv_d)))
        rxs = rxw * torch.rsqrt(1.0 + 0.25 * ex_u.square() / (1.0 + ex_u))
        qp = 0.5 * torch.einsum(
            "ni,nij,nj->n", rxw, step.op_ep.to(device=dev, dtype=dt), rxw
        )
        qn = 0.5 * torch.einsum(
            "ni,nij,nj->n", rxs, step.op_en.to(device=dev, dtype=dt), rxs
        )
        l_op = l_op + float(cov_w) * (orw * (qp + qn) * inv_d).mean().to(out_dt)
    return l_op


def _update_record_cond_ema(
    record: Any,
    wi: torch.Tensor,
    pxy: torch.Tensor,
    pxx: torch.Tensor,
    pyy: torch.Tensor,
    r_worst: float,
) -> None:
    num = (wi * (2.0 * pxy.detach() - pxx.detach() - pyy)).sum()
    den = (wi * pyy).sum().clamp_min(1e-8)
    r_imq = max(-0.5, min(0.5, float((num / den).item())))
    rc = r_imq + min(0.5, 0.11 * max(0.0, float(r_worst) - 0.10))
    if not record.cond_initialized:
        record.cond_ema = max(0.18, rc)
        record.cond_initialized = True
    else:
        record.cond_ema = 0.99 * float(record.cond_ema) + 0.01 * rc
    if hasattr(record, "step_scale"):
        uc = smoothstep01((0.050 - max(0.0, float(record.cond_ema))) / 0.024)
        record.step_scale = float(1.0 - 0.85 * uc)


def _can_mutate(record: Any, allow_state_update: bool) -> bool:
    if not (allow_state_update and torch.is_grad_enabled()):
        return False
    in_p = getattr(record, "_in_probe", False)
    in_pp = getattr(record, "_in_protected_probe", False)
    return not (in_p or in_pp)


def conditional_fiber_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
    record: Any = None,
    bank: Optional[AxiomCondFiberBank] = None,
    mean_op_weight: float = 0.36,
    cov_op_weight: float = 0.35,
    imq_active_scale: float = 0.40,
    bank_mean_blend: float = 0.75,
    allow_state_update: bool = True,
) -> torch.Tensor:
    """Law 3B conditional fiber discrepancy with paired-context midpoint residualization."""
    bn = int(x_fake.shape[0])
    if bn <= 1 or int(y_real.shape[0]) != bn or int(c_cont.shape[0]) != bn:
        return x_fake[:0].sum()

    xf, yf = x_fake.reshape(bn, -1), y_real.detach().reshape(bn, -1)
    dev, dim = xf.device, max(1, int(xf.shape[-1]))
    dt = torch.float64 if xf.dtype == torch.float64 else torch.float32
    fmax = torch.finfo(xf.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
    can_mut = _can_mutate(record, allow_state_update)
    active_bank = _resolve_bank(record, bank)

    with torch.no_grad():
        groups, same_t_cpu = _partition_strata_cpu(bn, cond_disc)
        if not any(int(idx.numel()) >= 2 for _, idx in groups):
            return x_fake[:0].sum()
        cw = c_cont.detach().reshape(bn, -1).to(dt)
        xd, yd = xf.detach().to(dt), yf.to(dt)
        dc = int(cw.shape[1])
        cxy = torch.cat((cw, xd, yd), dim=1).to(device="cpu", dtype=torch.float64)
        step = _query_and_update_bank(
            active_bank,
            groups,
            cxy[:, :dc],
            cxy[:, dc : dc + dim],
            cxy[:, dc + dim :],
            can_mut=can_mut,
            blend=float(bank_mean_blend),
        )
        m_hat = step.m_hat.to(device=dev, dtype=dt)
        op_w = step.op_w.to(device=dev, dtype=dt)
        w_rms = (op_w.square().sum(dim=(-2, -1), keepdim=True) / float(dim)).sqrt()
        op_w_imq = op_w / w_rms.clamp_min(1e-8)

        same_t = same_t_cpu.to(device=dev)
        mh = same_t.clone().fill_diagonal_(False)
        vrow = mh.any(dim=1)
        m_val = mh & vrow.unsqueeze(1) & vrow.unsqueeze(0)
        yw = torch.einsum("nij,nj->ni", op_w_imq, yd - m_hat)
        wi, ell_f = _imq_geometry(cw, yw, same_t, m_val, vrow)
        pyy = _imq_3scale(yw, yw, ell_f)

    x_work = xf.to(dt)
    xw = torch.einsum("nij,nj->ni", op_w_imq, x_work - m_hat)
    pxy, pxx, wi_d = (
        _imq_3scale(xw, yw, ell_f),
        _imq_3scale(xw, xw, ell_f),
        wi.to(xf.dtype),
    )
    l_imq = (
        (wi_d * pxy.clamp_max(fmax).to(xf.dtype)).sum()
        - 0.5 * (wi_d * pxx.clamp_max(fmax).to(xf.dtype)).sum()
        - 0.5 * (wi_d * pyy.clamp_max(fmax).to(xf.dtype)).sum()
    )
    has_op = step.any_op and (mean_op_weight > 0.0 or cov_op_weight > 0.0)
    l_op = (
        _operator_terms(
            x_work,
            yd,
            op_w,
            step,
            mean_w=mean_op_weight,
            cov_w=cov_op_weight,
            out_dt=xf.dtype,
        )
        if has_op
        else xf.new_zeros(())
    )
    if can_mut and hasattr(record, "cond_initialized"):
        with torch.no_grad():
            _update_record_cond_ema(record, wi, pxy, pxx, pyy, step.r_worst)
    scale = float(imq_active_scale) if has_op else 1.0
    return (scale * l_imq + l_op).clamp(-fmax, fmax).to(xf.dtype)


__all__ = [
    "AxiomCondFiberBank",
    "SCHEMA_VERSION",
    "conditional_fiber_loss",
    "validate_fiber_bank_state",
]
