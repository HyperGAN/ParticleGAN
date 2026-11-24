"""Axiom Module A (Law 1 — Representational Geometry): Pullback Prior & Laguerre Atlas.

Clean-slate Law-1 implementation (zero legacy sibling imports, zero eigh/svd/pinv/fwAD):
- Exact mass conservation across proto_w and w_fake_ema (including shock teleportation).
- Exact Pythagorean LTV on coalescence (2.2x guard), Gersho merge, and split (0.84*v).
- Central finite-offset pullback covariance isometry at sigma_z (O(sigma_z^2), zero RH).
- Per-cell relative-deficit Laguerre dual ascent with zero-droop deficits & exact gauge.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior

_PHI1 = 0.6180339887498949
_PRIMES = tuple(math.sqrt(p) % 1.0 for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53))
_Q_KEYS = ("proto_c", "proto_var", "proto_w", "w_fake_ema", "psi", "v_push")
_R_KEYS = ("row_m", "row_v", "row_n")
_STATE_KEYS = frozenset({"schema", "max_prototypes", "quantizer_initialized", *_Q_KEYS, "res_ema", "iso_ema", "iso_err_ema", "rows_initialized", *_R_KEYS})
_H8 = tuple(tuple(1 if bin(i & j).count("1") % 2 == 0 else -1 for j in range(8)) for i in range(8))

def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    u = x.clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)

def _box_muller_targets(rows: torch.Tensor, k_idx: torch.Tensor, comp_rep: torch.Tensor, k_proto: int, c_bar: torch.Tensor, v_mu_star: torch.Tensor, multicell: torch.Tensor, dim: int) -> torch.Tensor:
    n_p, r_f, pr = (dim + 1) // 2, (rows.to(torch.float64) + 0.5)[:, None], torch.tensor(_PRIMES, device=rows.device, dtype=torch.float64)
    j = torch.arange(n_p, device=rows.device)
    rad, ang = torch.sqrt(-2.0 * torch.log(torch.remainder(r_f * pr[(2 * j) % 16][None], 1.0).clamp(1e-4, 1.0 - 1e-4))).clamp_max(2.95), (2.0 * math.pi) * torch.remainder(r_f * pr[(2 * j + 1) % 16][None], 1.0)
    xi = torch.stack((rad * torch.cos(ang), rad * torch.sin(ang)), -1).reshape(len(rows), 2 * n_p)[:, :dim].to(device=c_bar.device, dtype=torch.float32)
    cid, cnt = comp_rep[k_idx], torch.bincount(comp_rep[k_idx], minlength=k_proto).float()
    xi_m = torch.where(cnt[:, None] >= 2.0, xi.new_zeros((k_proto, dim)).index_add_(0, cid, xi) / cnt.clamp_min(1.0)[:, None], 0.0)
    return torch.where(multicell[k_idx][:, None], c_bar[k_idx] + v_mu_star[k_idx].sqrt()[:, None] * (xi - xi_m[cid]), c_bar[k_idx])

@torch.no_grad()
def _principal_axis_from_points(pts: torch.Tensor, dim: int, seed_idx: int) -> torch.Tensor:
    u = None
    if pts.ndim == 2 and pts.shape[0] >= 2:
        xc = pts.float() - pts.float().mean(0, keepdim=True)
        sq = xc.square().sum(1)
        if float(sq.amax().item()) > 1e-12:
            cov, u = xc.T @ xc, xc[int(sq.argmax().item())]
            u = u / u.norm().clamp_min(1e-12)
            for _ in range(6):
                v = cov @ u
                u = v / v.norm() if float(v.norm().item()) > 1e-12 else u
    if u is None:
        u = torch.cos(torch.arange(1, dim + 1, device=pts.device, dtype=torch.float32) * ((seed_idx + 1.0) * _PHI1 * 2.0 * math.pi))
        u = u / u.norm().clamp_min(1e-12) if float(u.norm().item()) >= 1e-8 else torch.full((dim,), dim ** -0.5, device=pts.device)
    return -u if float(u[int(u.abs().argmax().item())].item()) < 0.0 else u

def _prior_kernel_spec(prior: nn.Module) -> Tuple[Any, float, bool]:
    base = unwrap_prior(prior)
    if getattr(base, "code_law", "table") != "table": return base, 0.0, False
    if not bool(getattr(prior, "transforms_codes", False)):
        return (base, base._cached_sigma_float() if bool(getattr(base, "_noise_enabled", False)) else 0.0, True) if isinstance(base, MoGParticlePrior) else (base, 0.0, hasattr(base, "z"))
    ms = tuple(getattr(prior, "manipulators", ()))
    return (base, float(ms[0].sigma), True) if (len(ms) == 1 and type(ms[0]).__name__ == "NormalNoiseManipulator" and float(getattr(ms[0], "kappa_radial", 1.0)) == 1.0) else (base, 0.0, False)

def has_positive_prior_sigma(prior: nn.Module) -> bool: return _prior_kernel_spec(prior)[1] > 0.0

@torch.no_grad()
def coalesce_pythagorean_ltv(proto_c: torch.Tensor, proto_var: torch.Tensor, proto_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Externally isolated sibling coalescence preserving exact scalar trace LTV & 2.2x width guard."""
    k, dim, pv = int(proto_c.shape[0]), int(proto_c.shape[1]), proto_var.clamp_min(1e-10)
    eye_k = torch.eye(k, device=proto_c.device, dtype=torch.float32)
    if k < 2: return proto_c, pv, eye_k, torch.zeros(k, device=proto_c.device, dtype=torch.bool)
    sig, sqrt_d, d_cc = pv.sqrt(), math.sqrt(float(dim)), torch.cdist(proto_c, proto_c, p=2.0)
    s_max, s_min = torch.maximum(sig[:, None], sig[None, :]), torch.minimum(sig[:, None], sig[None, :]).clamp_min(1e-8)
    adj = ((d_cc <= (2.6 * sqrt_d * s_max)) & (s_max <= (2.2 * s_min))).float().fill_diagonal_(1.0)
    for _ in range(3): adj = ((adj @ adj) > 0).float()
    cs = adj.sum(1)
    cand = (cs >= 1.0) & (cs <= 4.0)
    cw = torch.where(cand[:, None] & cand[None, :], adj, eye_k) * proto_w.clamp_min(1e-12)[None, :]
    cw = cw / cw.sum(1, keepdim=True).clamp_min(1e-12)
    cc = cw @ proto_c
    cv = (cw * (pv[None, :] + (proto_c[None, :] - cc[:, None]).square().sum(-1) / float(dim))).sum(1).clamp_min(1e-10)
    coal = cand & ((adj @ (~(d_cc.masked_fill(adj > 0.5, float("inf")).min(1).values >= (4.2 * sqrt_d * cv.sqrt()))).float()) < 0.5)
    return torch.where(coal[:, None], cc, proto_c), torch.where(coal, cv, pv), torch.where(coal[:, None], cw, eye_k), (cs > 1.5) & coal

@torch.no_grad()
def _sinkhorn_barycenter(L_xp: torch.Tensor, log_a: float, log_b: torch.Tensor, c_bar: torch.Tensor, b: int, out_dtype: torch.dtype) -> torch.Tensor:
    buf, u, v = torch.empty_like(L_xp), L_xp.new_zeros(b), L_xp.new_zeros(L_xp.shape[1])
    for _ in range(10):
        m0 = torch.add(L_xp, u[:, None], out=buf).amax(0, keepdim=True)
        v = log_b - (m0.squeeze(0) + buf.sub_(m0).exp_().sum(0).log_())
        m1 = torch.add(L_xp, v[None, :], out=buf).amax(1, keepdim=True)
        u = log_a - (m1.squeeze(1) + buf.sub_(m1).exp_().sum(1).log_())
    torch.add(L_xp, u[:, None], out=buf).add_(v[None, :]).exp_().mul_(float(b))
    return (buf.div_(buf.sum(1, keepdim=True).clamp_min_(1e-12)) @ c_bar).to(out_dtype)

@torch.no_grad()
def allocate_deficit_quotas(proto_w: torch.Tensor, w_fake_ema: torch.Tensor, n_slots: int) -> torch.Tensor:
    """Unbiased slot allocation via descending (proto_w - w_fake_ema, proto_w) + Weyl tie-break."""
    k, q = int(proto_w.numel()), torch.zeros(int(proto_w.numel()), device=proto_w.device, dtype=torch.long)
    if k == 0 or n_slots <= 0: return q
    w = proto_w.double().clamp_min(0.0)
    w, deficit = w / w.sum().clamp_min(1e-12), (proto_w - w_fake_ema).double()
    ord_i = torch.argsort(torch.remainder((torch.arange(k, device=proto_w.device, dtype=torch.float64) + 0.5) * _PHI1, 1.0), descending=True, stable=True)
    if n_slots < k:
        ord_i = ord_i[torch.argsort(w[ord_i], descending=True, stable=True)]
        q[ord_i[torch.argsort(deficit[ord_i], descending=True, stable=True)][:n_slots]] = 1
        return q
    exact = w * float(n_slots)
    q = torch.floor(exact).to(torch.long)
    rem = int(n_slots - int(q.sum().item()))
    if rem > 0:
        ord_i = ord_i[torch.argsort(deficit[ord_i], descending=True, stable=True)]
        q[ord_i[torch.argsort((exact - q.double())[ord_i], descending=True, stable=True)][:rem]] += 1
    return q

def central_pullback_cov(gen_fn: Callable[[torch.Tensor], torch.Tensor], mu: torch.Tensor, sigma_z: float) -> torch.Tensor:
    """Central finite-offset pushforward covariance at sigma_z (O(sigma_z^2), zero fwAD/eigh/svd)."""
    m, d_z = int(mu.shape[0]), int(mu.shape[1])
    if d_z <= 8: dirs, n_dirs, scale = torch.eye(d_z, device=mu.device, dtype=mu.dtype)[None].expand(m, d_z, d_z), d_z, 1.0
    else:
        n_dirs, scale, cols = 8, float(d_z) / 8.0, torch.arange(d_z, device=mu.device)
        sgn = torch.where(torch.remainder((torch.arange(m, device=mu.device, dtype=torch.float64)[:, None] + 0.5) * _PHI1 + (cols.to(torch.float64) + 1.0) * _PRIMES[0], 1.0) < 0.5, 1.0, -1.0).to(mu.dtype)
        dirs = (torch.tensor(_H8, device=mu.device, dtype=mu.dtype)[:, cols % 8][None] * sgn[:, None]) / math.sqrt(float(d_z))
    off = float(sigma_z) * dirs
    x_cat = gen_fn(torch.cat(((mu[:, None] + off).reshape(m * n_dirs, d_z), (mu[:, None] - off).reshape(m * n_dirs, d_z)), 0)).reshape(2, m, n_dirs, -1)
    x_f = x_cat.float() if x_cat.dtype in (torch.float16, torch.bfloat16) else x_cat
    j_cent = 0.5 * (x_f[0] - x_f[1])
    sp = torch.einsum("mad,mae->mde", j_cent, j_cent)
    return sp * scale if scale != 1.0 else sp

def validate_axiom_prior_state(state: Any) -> Dict[str, Any]:
    """Strict schema-1 checkpoint validator for AxiomPriorRegularizer."""
    def _chk(t: Any, nm: str, shape: Optional[Tuple[int, ...]] = None, ndim: Optional[int] = None, nonneg: bool = False) -> torch.Tensor:
        if not isinstance(t, torch.Tensor) or not t.is_floating_point() or not bool(torch.isfinite(t).all()) or (shape is not None and tuple(t.shape) != shape) or (ndim is not None and (t.ndim != ndim or any(s < 1 for s in t.shape))) or (nonneg and bool((t < 0.0).any())):
            raise ValueError(f"invalid AxiomPriorRegularizer tensor {nm!r}")
        return t
    if not isinstance(state, dict) or set(state.keys()) != _STATE_KEYS or type(state["schema"]) is not int or state["schema"] != 1 or type(state["max_prototypes"]) is not int or state["max_prototypes"] < 2 or type(state["quantizer_initialized"]) is not bool or type(state["rows_initialized"]) is not bool or any(type(state[k]) not in (int, float) or not math.isfinite(state[k]) or state[k] < 0.0 for k in ("res_ema", "iso_ema", "iso_err_ema")): raise ValueError("invalid AxiomPriorRegularizer checkpoint header")
    if state["quantizer_initialized"]:
        pc = _chk(state["proto_c"], "proto_c", ndim=2)
        if pc.shape[0] > state["max_prototypes"] or abs(float(_chk(state["proto_w"], "proto_w", (int(pc.shape[0]),), nonneg=True).double().sum().item()) - 1.0) > 1e-4 or abs(float(_chk(state["w_fake_ema"], "w_fake_ema", (int(pc.shape[0]),), nonneg=True).double().sum().item()) - 1.0) > 1e-4: raise ValueError("invalid AxiomPriorRegularizer quantizer size or mass conservation")
        for nm in ("proto_var", "psi", "v_push"): _chk(state[nm], nm, shape=(int(pc.shape[0]),), nonneg=(nm != "psi"))
    elif any(state[nm] is not None for nm in _Q_KEYS): raise ValueError("uninitialized AxiomPriorRegularizer must have None quantizer tensors")
    if state["rows_initialized"]:
        rm = _chk(state["row_m"], "row_m", ndim=2)
        for nm in ("row_v", "row_n"): _chk(state[nm], nm, shape=(int(rm.shape[0]), 1), nonneg=True)
    elif any(state[nm] is not None for nm in _R_KEYS): raise ValueError("uninitialized AxiomPriorRegularizer must have None row buffers")
    return state

class AxiomPriorRegularizer(nn.Module):
    """Law-1 standalone IMQ-repulsion regularizer + mass/LTV-exact GPU-resident prototype atlas."""
    def __init__(self, reference: Any = None, *, weight: float = 1.0, target_std: float = 1.0, eps: float = 1e-4, max_prototypes: int = 256, bound_device: Optional[torch.device] = None, bound_shape: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        del reference
        if type(max_prototypes) is not int or max_prototypes < 2 or any(type(v) not in (int, float) or not math.isfinite(v) or (v <= 0.0 if pos else v < 0.0) for v, pos in ((weight, False), (target_std, True), (eps, True))): raise ValueError("invalid AxiomPriorRegularizer hyperparameters")
        self.weight, self.target_std, self.eps, self.max_prototypes, self._bound_device, self._bound_shape = float(weight), float(target_std), float(eps), int(max_prototypes), bound_device, bound_shape
        self.record = self.axiom_record = self.thermal_source = self._last_raw_var_cache = None
        self.quantizer_initialized = self.rows_initialized = False
        self.proto_c = self.proto_var = self.proto_w = self.w_fake_ema = self.psi = self.v_push = self.row_m = self.row_v = self.row_n = None
        self.res_ema, self.iso_ema, self.iso_err_ema = 1.0, 1.0, 0.0

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2 or rows.shape[1] < 1: raise ValueError("rows must have shape (N, D) with D >= 1")
        n = int(rows.shape[0])
        if self.weight <= 0.0 or n < 2: return rows[:0].sum()
        m, z = min(n, 256), (rows[(torch.arange(min(n, 256), device=rows.device) * n) // min(n, 256)] if n > 256 else rows)
        with torch.no_grad():
            zf = z.detach().float()
            sq = (zf.square().sum(1, keepdim=True) + zf.square().sum(1, keepdim=True).T - 2.0 * (zf @ zf.T)).clamp_min(0.0)
            iu, ju = torch.triu_indices(m, m, offset=1, device=z.device)
            r_med_sq, knn_idx = torch.median(sq[iu, ju]).clamp_min(1e-12).to(z.dtype), torch.topk(sq.fill_diagonal_(float("inf")), k=min(8, m - 1), dim=1, largest=False).indices
        loss = self.weight * ((r_med_sq / (r_med_sq + (z[:, None] - z[knn_idx]).square().sum(-1))).mean() + 0.15 * (((z - z.mean(0, keepdim=True)).square().mean(0) + self.eps).sqrt() - self.target_std).square().mean())
        s = float(getattr(self.axiom_record or self.record or self.thermal_source, "thermal_scale", 1.0))
        return (loss * s + loss.detach() * (1.0 - s)) if (s < 1.0 and loss.requires_grad) else loss

    def state_dict_checkpoint(self) -> Dict[str, Any]:
        return {"schema": 1, "max_prototypes": int(self.max_prototypes), "quantizer_initialized": bool(self.quantizer_initialized), "res_ema": float(self.res_ema), "iso_ema": float(self.iso_ema), "iso_err_ema": float(self.iso_err_ema), "rows_initialized": bool(self.rows_initialized), **{nm: None if getattr(self, nm) is None else getattr(self, nm).detach().cpu().clone() for nm in (*_Q_KEYS, *_R_KEYS)}}

    def load_state_dict_checkpoint(self, state: Dict[str, Any], *, validate_only: bool = False) -> None:
        val = validate_axiom_prior_state(state)
        if val["max_prototypes"] != self.max_prototypes or (val["rows_initialized"] and self._bound_shape is not None and tuple(val["row_m"].shape) != tuple(self._bound_shape)): raise ValueError("AxiomPriorRegularizer configuration or shape mismatch")
        if validate_only: return
        self.quantizer_initialized, self.rows_initialized, self.res_ema, self.iso_ema, self.iso_err_ema, self._last_raw_var_cache = bool(val["quantizer_initialized"]), bool(val["rows_initialized"]), float(val["res_ema"]), float(val["iso_ema"]), float(val["iso_err_ema"]), None
        for nm in (*_Q_KEYS, *_R_KEYS):
            setattr(self, nm, None if val[nm] is None else (val[nm].to(self._bound_device).detach().clone() if self._bound_device is not None else val[nm].detach().clone()))

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: return self.state_dict_checkpoint()
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False) -> None: self.load_state_dict_checkpoint(state_dict)

    @torch.no_grad()
    def _merge_pair_ltv(self, a: int, b_idx: int) -> None:
        dim, wa, wb, ca, cb = float(self.proto_c.shape[1]), self.proto_w[a].double(), self.proto_w[b_idx].double(), self.proto_c[a].double(), self.proto_c[b_idx].double()
        wtot = (wa + wb).clamp_min(1e-12)
        c_ab = (wa * ca + wb * cb) / wtot
        self.proto_c[a], self.proto_var[a] = c_ab.float(), ((wa * (self.proto_var[a].double() + (ca - c_ab).square().sum() / dim) + wb * (self.proto_var[b_idx].double() + (cb - c_ab).square().sum() / dim)) / wtot).clamp_min(1e-10).float()
        self.v_push[a], self.psi[a], self.psi[b_idx], self.proto_w[a], self.proto_w[b_idx], self.w_fake_ema[a], self.w_fake_ema[b_idx] = ((wa * self.v_push[a].double() + wb * self.v_push[b_idx].double()) / wtot).clamp_min(1e-10).float(), ((wa * self.psi[a].double() + wb * self.psi[b_idx].double()) / wtot).float(), 0.0, wtot.float(), 0.0, self.w_fake_ema[a] + self.w_fake_ema[b_idx], 0.0

    @torch.no_grad()
    def _split_slot_ltv(self, p_best: int, dst: int, pts: torch.Tensor) -> None:
        dim, vp, c0 = int(self.proto_c.shape[1]), self.proto_var[p_best].clamp_min(1e-10), self.proto_c[p_best].clone()
        delta = (0.40 * math.sqrt(float(dim))) * vp.sqrt() * _principal_axis_from_points(pts, dim, p_best)
        self.proto_c[p_best], self.proto_c[dst] = c0 - delta, c0 + delta
        self.proto_var[p_best] = self.proto_var[dst] = (0.84 * vp).clamp_min(1e-10)
        self.v_push[p_best] = self.v_push[dst] = (0.84 * self.v_push[p_best]).clamp_min(1e-10)
        self.proto_w[p_best], self.w_fake_ema[p_best], self.psi[p_best] = 0.5 * (self.proto_w[p_best] + self.proto_w[dst]), 0.5 * (self.w_fake_ema[p_best] + self.w_fake_ema[dst]), 0.5 * (self.psi[p_best] + self.psi[dst])
        self.proto_w[dst], self.w_fake_ema[dst], self.psi[dst] = self.proto_w[p_best], self.w_fake_ema[p_best], self.psi[p_best]

    @torch.no_grad()
    def update_laguerre_dual(self, xu_det: torch.Tensor, c_bar: torch.Tensor, w_mat: torch.Tensor, r_macro_safe: torch.Tensor, laguerre_lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-cell relative-deficit Laguerre dual ascent with zero-droop deficits & exact gauge."""
        k_f, dim = float(len(c_bar)), int(c_bar.shape[1])
        k_star = (torch.cdist(xu_det, c_bar, p=2.0).square() - (w_mat @ self.psi)[None, :]).argmin(1)
        w_hard = w_mat[k_star].mean(0)
        w_fe = 0.88 * self.w_fake_ema + 0.12 * w_hard
        self.w_fake_ema = w_fe / w_fe.sum().clamp_min(1e-12)
        tv_now, tv_floor, d_max = 0.5 * (self.proto_w - self.w_fake_ema).abs().sum(), 0.35 * math.sqrt(k_f / max(float(len(xu_det)) * 8.0, 1.0)), ((self.proto_w - self.w_fake_ema).clamp_min(0.0) / self.proto_w.clamp_min(1.0 / k_f)).amax()
        g_psi = torch.maximum(_smoothstep01((tv_now - tv_floor) / max(0.08, tv_floor)), _smoothstep01((d_max - 0.15) / 0.20))
        eta = g_psi * min(3.0, max(0.35, float(laguerre_lr) * k_f)) * float(dim) * r_macro_safe.square()
        psi_next = w_mat @ (self.psi * (1.0 - 0.004 * g_psi * (1.0 - (self.proto_w > self.w_fake_ema).float())) + eta * (self.proto_w - (0.35 * w_hard + 0.65 * self.w_fake_ema)))
        psi_next = psi_next - psi_next.mean()
        self.psi = psi_next * ((2.5 * float(dim) * r_macro_safe.square() + 0.5) / psi_next.abs().amax().clamp_min(1e-12)).clamp_max(1.0)
        return k_star, g_psi

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        self._bound_device = yr.device if self._bound_device is None else self._bound_device
        for nm in (*_Q_KEYS, *_R_KEYS):
            t = getattr(self, nm)
            if isinstance(t, torch.Tensor) and t.device != yr.device: setattr(self, nm, t.to(yr.device))
        x, b, dim = yr.detach().float(), int(yr.shape[0]), int(yr.shape[1])
        if not self.quantizer_initialized or self.proto_c is None or self.proto_c.shape[1] != dim:
            k = min(self.max_prototypes, max(4, (3 * b) // 4) if b >= 8 else b)
            proj = torch.remainder(torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) * _PHI1, 1.0) - 0.5
            self.proto_c = x[torch.argsort(x @ (proj / proj.norm().clamp_min(1e-8)), stable=True)[((torch.arange(k, device=x.device) * b + b // 2) // k).clamp(0, b - 1)]].clone()
            d_nn = torch.cdist(self.proto_c, self.proto_c).fill_diagonal_(float("inf")).min(1).values if k >= 2 else x.new_empty(0)
            v0 = (0.25 * float(d_nn[d_nn > 1e-7].median().item()) ** 2 / float(dim + 2)) if (d_nn > 1e-7).any() else float(x.var(0, unbiased=False).mean().clamp_min(1e-6).item()) / float(max(1, k))
            self.proto_var, self.proto_w = torch.full((k,), max(v0, 1e-8), device=x.device), torch.full((k,), 1.0 / float(k), device=x.device)
            self.w_fake_ema, self.psi, self.v_push, self.quantizer_initialized = self.proto_w.clone(), torch.zeros(k, device=x.device), self.proto_var.clone(), True
            return
        k, c_prev, v_prev = len(self.proto_c), self.proto_c.clone(), self.proto_var.clone()
        d2_min, assign = torch.cdist(x, c_prev, p=2.0).square().min(1)
        med_v, raw_sp = v_prev.median().clamp_min(1e-10), 2.0 * x.var(0, unbiased=False).sum()
        sums = x.new_zeros((k, dim + 2)).index_add_(0, assign, torch.cat((x.new_ones((b, 1)), ((x - c_prev[assign]).square().sum(1) / float(dim))[:, None], x), 1))
        b_cnt, b_mean = sums[:, 0], sums[:, 2:] / sums[:, 0].clamp_min(1.0)[:, None]
        v_est = torch.where(b_cnt >= 2.0, x.new_zeros(k).index_add_(0, assign, (x - b_mean[assign]).square().sum(1) / float(dim)) / (b_cnt - 1.0).clamp_min(1.0), sums[:, 1]).clamp_min(1e-10)
        has_sp, pair_sp = raw_sp > 1e-10, raw_sp.clamp_min(1e-12)
        shock = ((torch.where(has_sp, torch.sqrt(d2_min.mean() / pair_sp), raw_sp.new_zeros(())) - 0.45) / 0.35).clamp(0.0, 1.0)
        aw, hit, ac = 0.02 + 0.48 * shock * shock, b_cnt > 0, (((0.06 + 0.38 * shock * shock) * (b_cnt / (b_cnt + 2.0))).clamp_max(0.50))[:, None]
        pw = (1.0 - aw) * self.proto_w + aw * (b_cnt / float(b))
        self.proto_w, self.proto_var, self.proto_c = pw / pw.sum().clamp_min(1e-12), torch.where(hit, 0.88 * v_prev + 0.12 * v_est, v_prev), torch.where(hit[:, None], (1.0 - ac) * c_prev + ac * b_mean, c_prev)
        mean_w, dead, gamma = 1.0 / float(k), self.proto_w < (0.04 / float(k)), float(min(dim, 4)) / float(min(dim, 4) + 2)
        max_ur, worst_j = (d2_min / torch.maximum(14.44 * float(dim) * torch.maximum(v_prev[assign], med_v), 0.08 * pair_sp)).max(0)
        ge = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
        max_e, pi_t = ge.max(0)
        do_uncov, need_g = ((max_ur > 1.0) & has_sp if k >= 4 else raw_sp.new_zeros((), dtype=torch.bool)), (((max_e > 10.0 * ge.median().clamp_min(1e-12)) & (self.proto_w[pi_t] > 0.25 * mean_w)) if k >= 16 else raw_sp.new_zeros((), dtype=torch.bool))
        u_flag, s_flag, dead_flag = torch.stack((do_uncov, (~do_uncov) & has_sp & (dead.any() | need_g) if k >= 2 else raw_sp.new_zeros((), dtype=torch.bool), dead.any())).tolist()
        if u_flag:
            new_w, y_far, slot = 1.0 / float(b), x[int(worst_j.item())], -1
            if dead_flag: slot = int(self.proto_w.argmin().item())
            elif k < self.max_prototypes:
                self.proto_c, self.proto_var, self.v_push, self.proto_w, self.w_fake_ema, self.psi = torch.cat((self.proto_c, y_far[None])), torch.cat((self.proto_var, med_v[None])), torch.cat((self.v_push, med_v[None])), torch.cat((self.proto_w, self.proto_w.new_full((1,), new_w))), torch.cat((self.w_fake_ema, self.w_fake_ema.new_full((1,), new_w))), torch.cat((self.psi, self.psi.new_zeros((1,))))
            else:
                flat = int(torch.cdist(self.proto_c, self.proto_c).square().fill_diagonal_(float("inf")).argmin().item())
                self._merge_pair_ltv(flat // k, flat % k)
                slot = flat % k
            if slot >= 0: self.proto_c[slot], self.proto_var[slot], self.v_push[slot], self.proto_w[slot], self.w_fake_ema[slot], self.psi[slot] = y_far, med_v, med_v, new_w, new_w, 0.0
        elif s_flag:
            slots = torch.nonzero(dead, as_tuple=False).squeeze(-1)[:4].tolist() if dead_flag else []
            if not dead_flag:
                d_cc = torch.cdist(self.proto_c, self.proto_c).square().fill_diagonal_(float("inf"))
                d_cc[int(pi_t.item()), :] = d_cc[:, int(pi_t.item())] = float("inf")
                if float(d_cc.min().item()) <= 8.0 * float(dim) * float(med_v.item()):
                    flat = int(d_cc.argmin().item())
                    self._merge_pair_ltv(flat // k, flat % k)
                    slots = [flat % k]
            for di in slots:
                ge_s = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
                ge_s[di] = 0.0
                pb = int(ge_s.argmax().item())
                if pb == di or float(self.proto_w[pb].item()) <= 0.20 * mean_w: break
                self._split_slot_ltv(pb, di, x[assign == pb])
        if u_flag or s_flag: self.proto_w, self.w_fake_ema, self.psi = self.proto_w / self.proto_w.sum().clamp_min(1e-12), self.w_fake_ema / self.w_fake_ema.sum().clamp_min(1e-12), self.psi - self.psi.mean()

def should_use_axiom_pullback(trainer: Any, real: torch.Tensor, latent: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> bool:
    """True on unconditioned low-D MoG priors (2 <= D <= 16, sigma_z > 0) or degenerate point-mass targets."""
    del latent, indices
    r = trainer.recipe
    if r.prior_regularizer != "axiom" or float(r.t1_pullback_weight) <= 0.0 or r.encoder_mode != "none" or r.conditioning != "scalar" or real.ndim != 2 or not (1 <= int(real.shape[1]) <= 16): return False
    if int(real.shape[1]) >= 2 and has_positive_prior_sigma(trainer.prior): return True
    raw_var = float(real.detach().float().var(0, unbiased=False).sum().item())
    if hasattr(getattr(trainer, "prior_regularizer", None), "_last_raw_var_cache"): trainer.prior_regularizer._last_raw_var_cache = (id(real), raw_var)
    return raw_var < 1e-11

def _finalize_gen_step(trainer: Any, base_prior: Any, loss_g: torch.Tensor, loss_gan: torch.Tensor, fake_g: torch.Tensor, real_g: torch.Tensor, fake_logits: torch.Tensor, latent: torch.Tensor, indices: torch.Tensor, loss_d: torch.Tensor, penalty: torch.Tensor, g_scale: float | torch.Tensor, wakeup_boost: float | torch.Tensor, prior_scale: float) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    r, zero, rec_d, opt_g = trainer.recipe, loss_gan.new_zeros(()), getattr(trainer.opt_d, "record", None), trainer.opt_g
    tr, tr_loc = (r.kinetic_transport_loss(fake_g, real_g) if r.kinetic_transport_weight else zero), (r.kinetic_transport_local_loss(fake_g, real_g) if r.kinetic_transport_local_weight else zero)
    pm = trainer.prior.regularize(base_prior.z if r.num_particles <= 1024 else base_prior.z[torch.unique(indices)], sampled_codes=latent, indices=indices, generated=fake_g, logits=fake_logits) if getattr(trainer, "_prior_penalty", False) else None
    loss_g = loss_g + tr + tr_loc + (pm if pm is not None else zero) + (r.prior_l2 * base_prior.z.square().mean() if (hasattr(base_prior, "z") and base_prior.z.requires_grad and r.prior_l2 > 0) else zero)
    if rec_d is not None: rec_d._disc_gate, rec_d.ema_push_res = 0.0, float(trainer.prior_regularizer.res_ema)
    opt_g.zero_grad()
    if getattr(opt_g, "direction_blend", False) and hasattr(opt_g, "bind_protected_from_grads"):
        loss_g.backward()
        opt_g.bind_protected_from_grads()
    else: constraint_geometry_backward(loss_g, opt_g, (loss_g,))
    if hasattr(base_prior, "z") and base_prior.z.requires_grad and base_prior.z.grad is None: base_prior.z.grad = torch.zeros_like(base_prior.z)
    trainer.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
    if hasattr(opt_g, "bind_axiom_modulation"):
        opt_g.bind_axiom_modulation(g_scale=g_scale, wakeup_boost=wakeup_boost, prior_scale=prior_scale)
        opt_g.step()
    else:
        gs, saved = (float(g_scale.item()) if isinstance(g_scale, torch.Tensor) else float(g_scale)), [g["lr"] for g in opt_g.param_groups]
        for grp in opt_g.param_groups: grp["lr"] *= gs
        try: opt_g.step()
        finally:
            for grp, lr0 in zip(opt_g.param_groups, saved): grp["lr"] = lr0
    trainer.policy.after_generator_step()
    return loss_g, pm, tr, tr_loc

def axiom_pullback_step(trainer: Any, real: torch.Tensor, generator_real: Any, latent: torch.Tensor, indices: torch.Tensor, *, sigma_out: Any, noise: torch.Generator, critic: nn.Module, loss_d: torch.Tensor, penalty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Execute Axiom Law-1 Pullback: Exact LTV + Central Pullback Isometry + Per-Cell Deficit Laguerre."""
    reg, recipe, b = trainer.prior_regularizer, trainer.recipe, len(real)
    base_prior, sigma_z, can_condense = _prior_kernel_spec(trainer.prior)
    has_sigma, rv_cache = sigma_z > 0.0, getattr(reg, "_last_raw_var_cache", None)
    reg._last_raw_var_cache = None
    is_case_a = (rv_cache[1] if (rv_cache is not None and rv_cache[0] == id(real)) else float(real.detach().float().var(0, unbiased=False).sum().item())) < 1e-11
    gen_clean = lambda z_in: trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_in, 0.0, noise))
    if is_case_a:
        mu_det, eps_g, fake_g = None, None, trainer._postprocess(trainer.prior, trainer._generate(trainer.G, latent, sigma_out, noise, rows=indices))
        x_ctr_flat = fake_flat = fake_g.reshape(b, -1)
    else:
        mu_g = base_prior.means()[indices] if (isinstance(base_prior, MoGParticlePrior) and not bool(getattr(trainer.prior, "transforms_codes", False))) else (base_prior.z[indices] if can_condense else latent.detach())
        mu_det, eps_g = mu_g.detach(), (((latent - mu_g) / sigma_z).detach() if has_sigma else None)
        fake_g = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, (mu_det + sigma_z * eps_g) if has_sigma else mu_det, sigma_out, noise, rows=indices))
        fake_flat, x_ctr_flat = fake_g.reshape(b, -1), (gen_clean(mu_det).reshape(b, -1) if has_sigma else fake_g.reshape(b, -1))
    if hasattr(trainer.opt_g, "bind_origin_sample"): trainer.opt_g.bind_origin_sample(None if is_case_a else fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:] or len(real_g) != len(real): raise ValueError("generator_real must match the real sample shape and batch size")
    fake_logits, real_logits, rec_d = critic(fake_g), critic(real_g), getattr(trainer.opt_d, "record", None)
    if rec_d is not None and hasattr(rec_d, "observe_margin_and_scale"): rec_d.observe_margin_and_scale(real_logits, fake_logits, real_g, fake_g)
    loss_gan, yr = (F.softplus(-fake_logits).mean() if recipe.loss == "non_saturating" else trainer.loss.g_loss(fake_logits, real_logits)), real_g.detach().reshape(b, -1)
    if is_case_a:
        with torch.no_grad():
            err = (fake_flat.detach().float() - yr.float()).norm(dim=-1)
            cur = max(float(err.mean().item()), 0.35 * float(err.amax().item()))
            reg.res_ema = min(cur, 10.0) if cur > max(3.5 * reg.res_ema, reg.res_ema + 0.12) else (0.88 * reg.res_ema + 0.12 * min(cur, 10.0))
            g_sc = max(0.015, min(1.0, reg.res_ema / 0.025))
        loss_pt = (torch.sqrt(1e-6 + (fake_flat - yr).square()) - 1e-3).mean() + (0.004 * base_prior.z.float().square().mean().to(fake_flat.dtype) if (hasattr(base_prior, "z") and base_prior.z.requires_grad) else 0.0)
        prior_diag = reg(base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]).detach() if (hasattr(base_prior, "z") and base_prior.z.requires_grad and recipe.prior_reg > 0) else loss_gan.new_zeros(())
        loss_g, pm, tr, tr_l = _finalize_gen_step(trainer, base_prior, (0.02 * min(1.0, reg.res_ema / 0.25)) * loss_gan + float(recipe.t1_pullback_weight) * loss_pt, loss_gan, fake_g, real_g, fake_logits, latent, indices, loss_d, penalty, g_sc, 1.0, g_sc)
        return loss_g, loss_gan, prior_diag, pm, tr, tr_l
    dim, was_init = int(yr.shape[1]), reg.quantizer_initialized
    with torch.no_grad():
        reg.update_quantizer(yr)
        yf, k_proto = yr.float(), len(reg.proto_c)
        r_macro_safe = yf.std(0, unbiased=False).mean().clamp_min(1e-4)
        c_bar_all, var_bar_all, w_mat_all, multicell_coal = coalesce_pythagorean_ltv(reg.proto_c, reg.proto_var, reg.proto_w)
        sigma_bar_all, sigma_cell_med, x_ctr_det, fake_det = var_bar_all.sqrt(), torch.quantile(var_bar_all.sqrt(), 0.35).clamp_min(1e-6), x_ctr_flat.detach().float(), fake_flat.detach().float()
        d_ctr = torch.cdist(x_ctr_det, c_bar_all, p=2.0)
        near_p = d_ctr.argmin(1)
        is_coal_i = multicell_coal[near_p]
        xff = torch.where(is_coal_i[:, None], x_ctr_det, fake_det)
        scale_ratio = xff.std(0, unbiased=False).mean().clamp_min(1e-8) / r_macro_safe
        wakeup_t, v_mu_star_all = torch.where(scale_ratio < 0.50, scale_ratio.new_full((), 5.0), torch.where(scale_ratio < 0.80, scale_ratio.new_full((), 2.0), scale_ratio.new_full((), 1.0))), (var_bar_all - (w_mat_all @ reg.v_push).clamp_min(0.0)).clamp_min(0.0)
        has_coal, any_coal_i, all_coal, need_whiten, scale_ge_065, any_v_mu = torch.stack((multicell_coal.any(), is_coal_i.any(), is_coal_i.all(), scale_ratio < 0.35, scale_ratio >= 0.65, multicell_coal.any() & (v_mu_star_all.amax() > 0.0))).tolist()
        iso_val = float((sigma_z * r_macro_safe / sigma_cell_med).clamp(1.0, 4.5).item()) if has_sigma else 1.0
        reg.iso_ema = (iso_val if not was_init else (0.95 * reg.iso_ema + 0.05 * iso_val)) if has_sigma else 1.0
        d_xp_phys, cb_m = (d_ctr if all_coal else torch.cdist(xff, c_bar_all, p=2.0)), (reg.proto_w[:, None] * c_bar_all).sum(0, keepdim=True)
        d_xp_route = torch.cdist((xff - xff.mean(0, keepdim=True)) / xff.std(0, unbiased=False, keepdim=True).clamp_min(1e-6), (c_bar_all - cb_m) / (reg.proto_w[:, None] * (c_bar_all - cb_m).square()).sum(0, keepdim=True).sqrt().clamp_min(1e-6), p=2.0) if (need_whiten and k_proto >= 4) else d_xp_phys
        y_ot = _sinkhorn_barycenter(-(d_xp_route / d_xp_route.mean().clamp_min(1e-5)) / max(0.010, min(0.06, 0.25 * reg.res_ema)), -math.log(float(b)), torch.log(reg.proto_w.clamp_min(1e-12)), c_bar_all, b, fake_flat.dtype)
        reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(float(((torch.maximum((reg.proto_w * d_xp_phys.min(0).values).sum(), d_xp_phys.min(1).values.median()) - 1.45 * math.sqrt(float(dim)) * sigma_cell_med).clamp_min(0.0) / r_macro_safe).item()), 10.0)
        comp_rep, v_iso_i = (w_mat_all > 0.0).float().argmax(1), var_bar_all[near_p].clone()
        cid_i = comp_rep[near_p]
        if any_coal_i:
            cnt_f = torch.bincount(cid_i, minlength=k_proto).float()
            mean_c = x_ctr_det.new_zeros((k_proto, dim)).index_add_(0, cid_i, x_ctr_det) / cnt_f.clamp_min(1.0)[:, None]
            if any_v_mu and scale_ge_065 and can_condense:
                halo = ((xff - c_bar_all[near_p]).norm(dim=-1) / (3.5 * math.sqrt(float(dim)) * sigma_bar_all[near_p]).clamp_min(1e-6)).clamp(0.0, 1.0).square()[:, None].to(fake_flat.dtype)
                y_ot = torch.where((is_coal_i & (v_mu_star_all[near_p] > 0.0))[:, None], halo * y_ot + (1.0 - halo) * _box_muller_targets(indices, near_p, comp_rep, k_proto, c_bar_all, v_mu_star_all, multicell_coal, dim).to(fake_flat.dtype), y_ot)
            valid_v = is_coal_i & (cnt_f[cid_i] >= 2.0)
            if bool(valid_v.any()):
                v_mu_c = x_ctr_det.new_zeros(k_proto).index_add_(0, cid_i, (x_ctr_det - mean_c[cid_i]).square().sum(-1) / float(dim)) / (cnt_f - 1.0).clamp_min(1.0)
                v_iso_i = (var_bar_all[near_p] - torch.where(valid_v, v_mu_c[cid_i].clamp_max(0.75 * var_bar_all[near_p]), 0.0)).clamp_min(1e-10)
        inv_w_i, w_sat, y_bound = (1.0 / (float(k_proto) * reg.proto_w[near_p]).clamp_min(1e-4)).sqrt().clamp(0.5, 4.0).to(fake_flat.dtype), torch.ones(b, device=yr.device, dtype=fake_flat.dtype), yf.abs().amax(0, keepdim=True) + 3.5 * sigma_cell_med
        if has_sigma and eps_g is not None:
            e_n = eps_g.float().norm(dim=-1)
            ve, s_obs = e_n >= 0.45, (fake_det - x_ctr_det).norm(dim=-1) / (sigma_z * e_n.clamp_min(0.45))
            s_ref = s_obs[ve].median().clamp_min(1e-6) if (has_coal and bool(ve.any())) else ((v_iso_i.sqrt() * math.sqrt(float(dim) / float(max(1, mu_det.shape[1])))) / max(sigma_z, 1e-4))
            rail = ve & (s_obs < (0.50 if has_coal else 0.40) * s_ref) & (True if has_coal else ((xff - y_ot.detach().float()).norm(dim=-1) > 1.35 * math.sqrt(float(dim)) * sigma_bar_all[near_p]))
            w_sat = torch.where(rail, (((1.0 if has_coal else 0.45) * s_ref) / s_obs.clamp_min(1e-6)).clamp(1.0, 8.0), 1.0).to(fake_flat.dtype)
    x_ot_src, sc_med_dt, loss_g_iso = (x_ctr_flat if all_coal else (fake_flat if not any_coal_i else torch.where(is_coal_i[:, None], x_ctr_flat, fake_flat))), sigma_cell_med.to(fake_flat.dtype), fake_flat.new_zeros(())
    loss_g_ot = (w_sat * (torch.sqrt(sc_med_dt.square() + (x_ot_src - y_ot).square().sum(-1)) - sc_med_dt)).mean() + 2.0 * F.relu(fake_flat.float().abs() - y_bound).square().mean().to(fake_flat.dtype)
    if has_sigma and float(recipe.t1_iso_weight) > 0.0:
        m_iso = b if (has_coal and mu_det.shape[1] <= 4) else min(b, 128)
        sigma_push = central_pullback_cov(gen_clean, mu_det[:m_iso], sigma_z)
        v_iso_sub, eye_d = v_iso_i[:m_iso].to(sigma_push.dtype), torch.eye(dim, device=yr.device, dtype=sigma_push.dtype)[None]
        with torch.no_grad():
            v_tr, (gid, idx_map) = sigma_push.detach().float().diagonal(dim1=-2, dim2=-1).sum(-1) / float(dim), ((cid_i[:m_iso], comp_rep) if has_coal else (near_p[:m_iso], torch.arange(k_proto, device=yr.device)))
            g_cnt = torch.bincount(gid, minlength=k_proto).float()
            reg.v_push, w_iso_sat = torch.where(g_cnt[idx_map] > 0, 0.90 * reg.v_push + 0.10 * (v_tr.new_zeros(k_proto).index_add_(0, gid, v_tr)[idx_map] / g_cnt[idx_map].clamp_min(1.0)), reg.v_push), (v_iso_i[:m_iso] / v_tr.clamp_min(0.15 * v_iso_i[:m_iso])).sqrt().clamp(1.0, 4.0).to(sigma_push.dtype)
            iso_frob = ((sigma_push.detach().float() / v_iso_i[:m_iso].view(m_iso, 1, 1).clamp_min(1e-10)) - eye_d.float()).square().sum((1, 2)).sqrt()
            reg.iso_err_ema = 0.92 * reg.iso_err_ema + 0.08 * min(float((0.60 * iso_frob.mean() + 0.40 * torch.quantile(iso_frob, 0.92)).item()), 5.0)
        loss_g_iso = (inv_w_i[:m_iso].to(sigma_push.dtype) * w_iso_sat * ((sigma_push - v_iso_sub.view(m_iso, 1, 1) * eye_d).square().sum((1, 2)) / ((float(dim) * v_iso_sub).square() + 1e-10))).mean().to(fake_flat.dtype)
    eq_g = max(0.0, min(1.0, (0.030 - reg.res_ema) / 0.020)) if has_coal else 0.0
    adv_w, g_floor, row_floor = (0.038 * (1.0 - eq_g) + 0.008 * eq_g, 0.022 * (1.0 - eq_g) + 0.012 * eq_g, 0.15 * (1.0 - eq_g) + 0.38 * eq_g) if has_coal else (0.15, 0.04, 0.15)
    g_scale_t = torch.where(scale_ratio < 0.60, scale_ratio.new_full((), 1.0), scale_ratio.new_full((), max(g_floor, min(1.0, max(reg.res_ema / 0.10, 0.28 * max(0.0, min(1.0, (reg.iso_err_ema - 0.22) / 0.45)))))))
    loss_g, pm, tr, tr_l = _finalize_gen_step(trainer, base_prior, adv_w * loss_gan + float(recipe.t1_pullback_weight) * loss_g_ot + float(recipe.t1_iso_weight) * loss_g_iso, loss_gan, fake_g, real_g, fake_logits, latent, indices, loss_d, penalty, g_scale_t, wakeup_t, 1.0)
    prior_diag = loss_gan.new_zeros(())
    if can_condense and hasattr(base_prior, "z") and base_prior.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0:
        u_rows, g_flags = torch.unique(indices), [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters(): p.requires_grad_(False)
            n_u, c_u = len(u_rows), (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[u_rows]
            x_u = gen_clean(c_u).reshape(n_u, -1)
            with torch.no_grad():
                xu_det = x_u.detach().float()
                k_star, g_psi = reg.update_laguerre_dual(xu_det, c_bar_all, w_mat_all, r_macro_safe, float(recipe.t1_laguerre_lr))
                sc_u, c_target, row_settle = sigma_bar_all[k_star].to(x_u.dtype), (_box_muller_targets(u_rows, k_star, comp_rep, k_proto, c_bar_all, v_mu_star_all, multicell_coal, dim) if any_v_mu else c_bar_all[k_star]).to(x_u.dtype), max(row_floor, min(1.0, max(reg.res_ema / 0.10, float(g_psi.item()))))
            sq_u, mu_raw_u = (x_u - c_target).square().sum(-1), base_prior.z[u_rows]
            loss_mu = (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u).mean() + (0.15 * F.relu(reg.iso_ema - ((mu_raw_u - mu_raw_u.mean(0, keepdim=True)).square().mean(0) + 1e-6).sqrt()).square().mean() if (has_sigma and n_u >= 2) else 0.0)
            prior_diag = loss_mu.detach()
            if base_prior.z.grad is not None: base_prior.z.grad.zero_()
            loss_mu.backward()
            if not reg.rows_initialized or reg.row_m is None or reg.row_m.shape != base_prior.z.shape: reg.row_m, reg.row_v, reg.row_n, reg.rows_initialized = torch.zeros_like(base_prior.z), torch.zeros((len(base_prior.z), 1), device=base_prior.z.device, dtype=base_prior.z.dtype), torch.zeros((len(base_prior.z), 1), device=base_prior.z.device, dtype=base_prior.z.dtype), True
            elif reg.row_m.device != base_prior.z.device: reg.row_m, reg.row_v, reg.row_n = reg.row_m.to(base_prior.z.device), reg.row_v.to(base_prior.z.device), reg.row_n.to(base_prior.z.device)
            with torch.no_grad():
                g_rows, dist_u = (base_prior.z.grad[u_rows] * float(n_u)).clone(), sq_u.detach().float().sqrt()
                base_prior.z.grad.zero_()
                part_scale, far_m, d_z_u = (dist_u.to(sq_u.dtype)[:, None] / (1.5 * sc_u[:, None])).clamp(0.02, 1.2), dist_u > (2.8 * math.sqrt(float(dim)) * sc_u.float()), int(c_u.shape[1])
                if d_z_u <= 8 and bool(far_m.any()):
                    fi, ii = torch.nonzero(far_m, as_tuple=False).squeeze(-1), torch.nonzero(~far_m, as_tuple=False).squeeze(-1)[:64]
                    pi_idx = torch.cat((fi, ii if ii.numel() >= 4 else torch.arange(min(64, n_u), device=c_u.device, dtype=torch.long)), 0)
                    cp, xp = c_u[pi_idx].detach(), xu_det[pi_idx]
                    c_pert = (cp[:, None] + (1e-3 * torch.where(cp >= 0.0, 1.0, -1.0))[:, :, None] * torch.eye(d_z_u, device=c_u.device, dtype=c_u.dtype)[None]).reshape(-1, d_z_u)
                    csen = (gen_clean(c_pert).float().reshape(len(pi_idx), d_z_u, dim) - xp[:, None]).norm(dim=-1) / 1e-3
                    dead_f = csen[: len(fi)] < (0.04 * torch.quantile(csen[len(fi) :], 0.5, dim=0, keepdim=True).clamp_min(1e-6))
                    if bool(dead_f.any()):
                        g_rows[fi] = torch.where(dead_f, (torch.sign(mu_raw_u[fi].detach() - c_u[pi_idx[len(fi) :]].detach().mean(0, keepdim=True)).float() * torch.maximum(g_rows[fi].float().norm(dim=-1, keepdim=True), torch.ones((len(fi), 1), device=c_u.device))).to(g_rows.dtype), g_rows[fi])
                        part_scale[fi] = torch.where(dead_f.any(1, keepdim=True), torch.ones_like(part_scale[fi]), part_scale[fi])
                m_r, v_r, n_r = reg.row_m[u_rows] * 0.80 + 0.20 * g_rows, reg.row_v[u_rows] * 0.80 + 0.20 * g_rows.square().sum(-1, keepdim=True), reg.row_n[u_rows] + 1.0
                reg.row_m[u_rows], reg.row_v[u_rows], reg.row_n[u_rows] = m_r, v_r, n_r
                base_prior.z.index_add_(0, u_rows, ((m_r / (1.0 - 0.80 ** n_r)) / ((v_r / (1.0 - 0.80 ** n_r)).sqrt() + 1e-5)) * part_scale, alpha=-float(recipe.t1_prior_row_lr) * row_settle)
        finally:
            for p, fl in zip(trainer.G.parameters(), g_flags): p.requires_grad_(fl)
            if rec_d is not None: rec_d._g_probe_cache = None
    return loss_g, loss_gan, prior_diag, pm, tr, tr_l

__all__ = ["AxiomPriorRegularizer", "allocate_deficit_quotas", "axiom_pullback_step", "central_pullback_cov", "coalesce_pythagorean_ltv", "has_positive_prior_sigma", "should_use_axiom_pullback", "validate_axiom_prior_state"]
