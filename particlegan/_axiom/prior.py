"""Axiom Module A (Law 1): Pullback Prior, Laguerre Atlas, and Unbiased 2-Sketch Isometry."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import NormalNoiseManipulator, unwrap_prior

_PHI1 = 0.6180339887498949


def _prime_irrationals(n: int) -> Tuple[float, ...]:
    out = []
    candidate = 2
    while len(out) < n:
        if all(candidate % p for p in range(2, int(candidate ** 0.5) + 1)):
            out.append(math.sqrt(candidate) % 1.0)
        candidate += 1
    return tuple(out)


_PRIMES = _prime_irrationals(128)
_Q_KEYS = ("proto_c", "proto_var", "proto_w", "proto_hits", "w_fake_ema", "psi", "v_push")
_R_KEYS = ("row_m", "row_v", "row_n")
_STATE_KEYS = frozenset(
    {
        "schema",
        "max_prototypes",
        "quantizer_initialized",
        *_Q_KEYS,
        "res_ema",
        "iso_ema",
        "iso_err_ema",
        "rows_initialized",
        *_R_KEYS,
    }
)
_H8 = tuple(tuple(1 if bin(i & j).count("1") % 2 == 0 else -1 for j in range(8)) for i in range(8))


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    u = x.clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _box_muller_targets(
    rows: torch.Tensor,
    k_idx: torch.Tensor,
    comp_rep: torch.Tensor,
    k_proto: int,
    c_bar: torch.Tensor,
    v_mu_star: torch.Tensor,
    multicell: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    n_p = (dim + 1) // 2
    r_f = (rows.to(torch.float64) + 0.5)[:, None]
    primes = _PRIMES[: 2 * n_p] if 2 * n_p <= len(_PRIMES) else _prime_irrationals(2 * n_p)
    pr = torch.tensor(primes, device=rows.device, dtype=torch.float64)
    u0 = torch.remainder(r_f * pr[0::2][None], 1.0).clamp(1e-4, 1.0 - 1e-4)
    rad = torch.sqrt(-2.0 * torch.log(u0)).clamp_max(2.95)
    ang = (2.0 * math.pi) * torch.remainder(r_f * pr[1::2][None], 1.0)
    xi = torch.stack((rad * torch.cos(ang), rad * torch.sin(ang)), -1).reshape(len(rows), 2 * n_p)[:, :dim]
    xi = xi.to(device=c_bar.device, dtype=torch.float32)
    cid = comp_rep[k_idx]
    cnt = torch.bincount(cid, minlength=k_proto).float()
    xi_sum = xi.new_zeros((k_proto, dim)).index_add_(0, cid, xi)
    xi_m = torch.where(cnt[:, None] >= 2.0, xi_sum / cnt.clamp_min(1.0)[:, None], 0.0)
    return torch.where(
        multicell[k_idx][:, None],
        c_bar[k_idx] + v_mu_star[k_idx].sqrt()[:, None] * (xi - xi_m[cid]),
        c_bar[k_idx],
    )


@torch.no_grad()
def _principal_axis_from_points(
    pts: torch.Tensor,
    dim: int,
    seed_idx: int | torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dev = pts.device
    if isinstance(seed_idx, torch.Tensor):
        s_f: float | torch.Tensor = seed_idx.to(device=dev, dtype=torch.float32) + 1.0
    else:
        s_f = float(seed_idx + 1.0)
    u_fb = torch.cos(torch.arange(1, dim + 1, device=dev, dtype=torch.float32) * (s_f * _PHI1 * 2.0 * math.pi))
    u_fb = torch.where(u_fb.norm() >= 1e-8, u_fb / u_fb.norm().clamp_min(1e-12), torch.full((dim,), dim ** -0.5, device=dev))
    if pts.ndim == 2 and pts.shape[0] >= 2:
        pf = pts.float()
        if mask is None:
            xc = pf - pf.mean(0, keepdim=True)
            sq = xc.square().sum(1)
            ok = sq.amax() > 1e-24
        else:
            mf = mask.to(device=dev, dtype=torch.float32)
            cnt = mf.sum()
            xc = (pf - (mf[:, None] * pf).sum(0, keepdim=True) / cnt.clamp_min(1.0)) * mf[:, None]
            sq = xc.square().sum(1)
            ok = (cnt >= 2.0) & (sq.amax() > 1e-24)
        cov = xc.T @ xc
        u = xc[sq.argmax()]
        u = u / u.norm().clamp_min(1e-20)
        for _ in range(6):
            v = cov @ u
            vn = v.norm()
            u = torch.where(vn > 1e-20, v / vn.clamp_min(1e-20), u)
        u = torch.where(ok, u, u_fb)
    else:
        u = u_fb
    return torch.where(u[u.abs().argmax()] < 0.0, -u, u)


def _prior_kernel_spec(prior: nn.Module) -> Tuple[Any, float, bool]:
    base = unwrap_prior(prior)
    if getattr(base, "code_law", "table") != "table":
        return base, 0.0, False
    if not bool(getattr(prior, "transforms_codes", False)):
        if isinstance(base, MoGParticlePrior):
            sigma = base._cached_sigma_float() if bool(getattr(base, "_noise_enabled", False)) else 0.0
            return base, sigma, True
        return base, 0.0, hasattr(base, "z")
    ms = tuple(getattr(prior, "manipulators", ()))
    if (
        len(ms) == 1
        and isinstance(ms[0], NormalNoiseManipulator)
        and float(getattr(ms[0], "kappa_radial", 1.0)) == 1.0
    ):
        return base, float(ms[0].sigma), True
    return base, 0.0, False


def has_positive_prior_sigma(prior: nn.Module) -> bool:
    return _prior_kernel_spec(prior)[1] > 0.0


@torch.no_grad()
def coalesce_pythagorean_ltv(
    proto_c: torch.Tensor,
    proto_var: torch.Tensor,
    proto_w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Externally isolated sibling coalescence preserving exact scalar trace LTV & 2.2x width guard."""
    k = int(proto_c.shape[0])
    dim = int(proto_c.shape[1])
    pv = proto_var.clamp_min(1e-24)
    eye_k = torch.eye(k, device=proto_c.device, dtype=torch.float32)
    if k < 2:
        return proto_c, pv, eye_k, torch.zeros(k, device=proto_c.device, dtype=torch.bool)
    sig = pv.sqrt()
    sqrt_d = math.sqrt(float(dim))
    d_cc = torch.cdist(proto_c, proto_c, p=2.0)
    s_max = torch.maximum(sig[:, None], sig[None, :])
    s_min = torch.minimum(sig[:, None], sig[None, :]).clamp_min(1e-12)
    adj = ((d_cc <= (2.6 * sqrt_d * s_max)) & (s_max <= (2.2 * s_min))).float().fill_diagonal_(1.0)
    for _ in range(4):
        adj = ((adj @ adj) > 0).float()
    cs = adj.sum(1)
    cand = (cs >= 1.0) & (cs <= float(max(10, 2 * dim))) & (cs < float(k))
    cw = torch.where(cand[:, None] & cand[None, :], adj, eye_k) * proto_w.clamp_min(1e-12)[None, :]
    cw = cw / cw.sum(1, keepdim=True).clamp_min(1e-12)
    cc = cw @ proto_c
    cv = (cw * (pv[None, :] + (proto_c[None, :] - cc[:, None]).square().sum(-1) / float(dim))).sum(1).clamp_min(1e-24)
    sep_r = 2.0 * sqrt_d + 3.11
    min_ext = d_cc.masked_fill(adj > 0.5, float("inf")).min(1).values
    isolated = (adj @ (~(min_ext >= (sep_r * cv.sqrt()))).float()) < 0.5
    coal = cand & ((d_cc * adj).amax(1) <= sep_r * cv.sqrt()) & isolated
    coal = coal & ((proto_w * coal.float()).sum() >= 0.70)
    return (
        torch.where(coal[:, None], cc, proto_c),
        torch.where(coal, cv, pv),
        torch.where(coal[:, None], cw, eye_k),
        (cs > 1.5) & coal,
    )


@torch.no_grad()
def _sinkhorn_barycenter(
    L_xp: torch.Tensor,
    log_a: float,
    log_b: torch.Tensor,
    c_bar: torch.Tensor,
    b: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    buf = torch.empty_like(L_xp)
    u = L_xp.new_zeros(b)
    v = L_xp.new_zeros(L_xp.shape[1])
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
    k = int(proto_w.numel())
    q = torch.zeros(k, device=proto_w.device, dtype=torch.long)
    if k == 0 or n_slots <= 0:
        return q
    w = proto_w.double().clamp_min(0.0)
    w = w / w.sum().clamp_min(1e-12)
    deficit = (proto_w - w_fake_ema).double()
    weyl = torch.remainder((torch.arange(k, device=proto_w.device, dtype=torch.float64) + 0.5) * _PHI1, 1.0)
    ord_i = torch.argsort(weyl, descending=True, stable=True)
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


def _sketch_dirs(
    m: int,
    d_z: int,
    phase: float,
    salt: int,
    dev: torch.device,
    dt: torch.dtype,
) -> torch.Tensor:
    cols = torch.arange(d_z, device=dev, dtype=torch.int64)
    idx = torch.arange(m, device=dev, dtype=torch.int64)[:, None] * d_z + cols[None, :]
    seed_mix = (int(phase * 1073741824.0) + salt * 7919 + 1337) & 0xFFFFFFFF
    h = (idx * 2654435761 + seed_mix * 3266489917 + 374761393) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    signs = torch.where(((h ^ (h >> 16)) & 1) == 0, 1.0, -1.0).to(dt)
    h8 = torch.tensor(_H8, device=dev, dtype=dt)[:, (cols + salt * 3) % 8][None]
    return (h8 * signs[:, None]) / math.sqrt(float(d_z))


def central_pullback_cov(
    gen_fn: Callable[[torch.Tensor], torch.Tensor],
    mu: torch.Tensor,
    sigma_z: float,
    *,
    phase: float = 0.0,
    salt: int = 0,
    return_pair: bool = False,
) -> Any:
    """Central finite-offset pushforward covariance at sigma_z (+ unbiased 2-sketch pair when d_z > 8)."""
    m = int(mu.shape[0])
    d_z = int(mu.shape[1])
    if d_z <= 8:
        dirs = torch.eye(d_z, device=mu.device, dtype=mu.dtype)[None].expand(m, d_z, d_z)
        n_dirs = d_z
        scale = 1.0
    elif return_pair:
        dirs = torch.cat(
            (
                _sketch_dirs(m, d_z, phase, 1, mu.device, mu.dtype),
                _sketch_dirs(m, d_z, phase, 2, mu.device, mu.dtype),
            ),
            dim=1,
        )
        n_dirs = 16
        scale = float(d_z) / 8.0
    else:
        dirs = _sketch_dirs(m, d_z, phase, salt, mu.device, mu.dtype)
        n_dirs = 8
        scale = float(d_z) / 8.0
    off = float(sigma_z) * dirs
    z_cat = torch.cat(
        (
            (mu[:, None] + off).reshape(m * n_dirs, d_z),
            (mu[:, None] - off).reshape(m * n_dirs, d_z),
        ),
        dim=0,
    )
    x_cat = gen_fn(z_cat).reshape(2, m, n_dirs, -1)
    x_work = x_cat.float() if x_cat.dtype in (torch.float16, torch.bfloat16) else x_cat
    jc = 0.5 * (x_work[0] - x_work[1])
    if return_pair and d_z > 8:
        s1 = torch.einsum("mad,mae->mde", jc[:, :8], jc[:, :8]) * scale
        s2 = torch.einsum("mad,mae->mde", jc[:, 8:], jc[:, 8:]) * scale
        return s1, s2
    sp = torch.einsum("mad,mae->mde", jc, jc) * scale
    return (sp, sp) if return_pair else sp


def validate_axiom_prior_state(state: Any) -> Dict[str, Any]:
    """Strict schema-1 checkpoint validator for AxiomPriorRegularizer."""

    def _chk(
        t: Any,
        nm: str,
        shape: Optional[Tuple[int, ...]] = None,
        ndim: Optional[int] = None,
        nonneg: bool = False,
    ) -> torch.Tensor:
        if (
            not isinstance(t, torch.Tensor)
            or not t.is_floating_point()
            or not bool(torch.isfinite(t).all())
            or (shape is not None and tuple(t.shape) != shape)
            or (ndim is not None and (t.ndim != ndim or any(s < 1 for s in t.shape)))
            or (nonneg and bool((t < 0.0).any()))
        ):
            raise ValueError(f"invalid AxiomPriorRegularizer tensor {nm!r}")
        return t

    if (
        not isinstance(state, dict)
        or set(state.keys()) != _STATE_KEYS
        or type(state["schema"]) is not int
        or state["schema"] != 1
        or type(state["max_prototypes"]) is not int
        or state["max_prototypes"] < 2
        or type(state["quantizer_initialized"]) is not bool
        or type(state["rows_initialized"]) is not bool
        or any(
            type(state[k]) not in (int, float) or not math.isfinite(state[k]) or state[k] < 0.0
            for k in ("res_ema", "iso_ema", "iso_err_ema")
        )
    ):
        raise ValueError("invalid AxiomPriorRegularizer checkpoint header")
    if state["quantizer_initialized"]:
        pc = _chk(state["proto_c"], "proto_c", ndim=2)
        k = int(pc.shape[0])
        pw = _chk(state["proto_w"], "proto_w", (k,), nonneg=True)
        wf = _chk(state["w_fake_ema"], "w_fake_ema", (k,), nonneg=True)
        if (
            k > state["max_prototypes"]
            or abs(float(pw.double().sum().item()) - 1.0) > 1e-4
            or abs(float(wf.double().sum().item()) - 1.0) > 1e-4
        ):
            raise ValueError("invalid AxiomPriorRegularizer quantizer size or mass conservation")
        for nm in ("proto_var", "proto_hits", "psi", "v_push"):
            _chk(state[nm], nm, shape=(k,), nonneg=(nm != "psi"))
    elif any(state[nm] is not None for nm in _Q_KEYS):
        raise ValueError("uninitialized AxiomPriorRegularizer must have None quantizer tensors")
    if state["rows_initialized"]:
        rm = _chk(state["row_m"], "row_m", ndim=2)
        n_rows = int(rm.shape[0])
        _chk(state["row_v"], "row_v", shape=(n_rows, 1), nonneg=True)
        rn = _chk(state["row_n"], "row_n", shape=(n_rows, 1), nonneg=True)
        if bool((rn > 1.0 + 1e-6).any()):
            raise ValueError("invalid AxiomPriorRegularizer tensor 'row_n'")
    elif any(state[nm] is not None for nm in _R_KEYS):
        raise ValueError("uninitialized AxiomPriorRegularizer must have None row buffers")
    return state


def _sc_prop(name: str) -> property:
    def _get(self: "AxiomPriorRegularizer") -> float:
        if name in self._ema_t:
            self._flush_emas()
        return self._ema[name]

    def _set(self: "AxiomPriorRegularizer", value: float) -> None:
        self._ema[name] = float(value)
        self._ema_t.pop(name, None)

    return property(_get, _set)


class AxiomPriorRegularizer(nn.Module):
    """Law-1 standalone IMQ-repulsion regularizer + mass/LTV-exact GPU-resident prototype atlas."""

    res_ema = _sc_prop("res_ema")
    iso_ema = _sc_prop("iso_ema")
    iso_err_ema = _sc_prop("iso_err_ema")

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
        max_prototypes: int = 256,
        bound_device: Optional[torch.device] = None,
        bound_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        del reference
        if type(max_prototypes) is not int or max_prototypes < 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) or (v <= 0.0 if pos else v < 0.0)
            for v, pos in ((weight, False), (target_std, True), (eps, True))
        ):
            raise ValueError("invalid AxiomPriorRegularizer hyperparameters")
        self.weight = float(weight)
        self.target_std = float(target_std)
        self.eps = float(eps)
        self.max_prototypes = int(max_prototypes)
        self._bound_device = bound_device
        self._bound_shape = bound_shape
        self.record: Any = None
        self.axiom_record: Any = None
        self.thermal_source: Any = None
        self._last_raw_var_cache: Any = None
        self.quantizer_initialized = False
        self.rows_initialized = False
        self.proto_c: Optional[torch.Tensor] = None
        self.proto_var: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.proto_hits: Optional[torch.Tensor] = None
        self.w_fake_ema: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.v_push: Optional[torch.Tensor] = None
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None
        self._ema: Dict[str, float] = {"res_ema": 1.0, "iso_ema": 1.0, "iso_err_ema": 0.0}
        self._ema_t: Dict[str, torch.Tensor] = {}

    def _flush_emas(self) -> None:
        if not self._ema_t:
            return
        ks = list(self._ema_t.keys())
        ts = list(self._ema_t.values())
        if all(t.device == ts[0].device for t in ts):
            vals = torch.stack([t.double() for t in ts]).tolist()
        else:
            vals = [float(t.item()) for t in ts]
        for k, v in zip(ks, vals):
            self._ema[k] = float(v)
        self._ema_t.clear()

    def _dev_ema(self, name: str, dev: torch.device) -> torch.Tensor:
        cur = self._ema_t.get(name)
        if cur is not None and cur.device == dev:
            return cur
        if cur is not None:
            self._flush_emas()
        return torch.full((), float(self._ema[name]), dtype=torch.float32, device=dev)

    def _ensure_hits(self, dev: torch.device) -> None:
        if self.proto_c is not None and (self.proto_hits is None or len(self.proto_hits) != len(self.proto_c)):
            self.proto_hits = torch.ones(len(self.proto_c), device=dev, dtype=torch.float32)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2 or rows.shape[1] < 1:
            raise ValueError("rows must have shape (N, D) with D >= 1")
        n = int(rows.shape[0])
        if self.weight <= 0.0 or n < 2:
            return rows[:0].sum()
        m = min(n, 256)
        z = rows[(torch.arange(m, device=rows.device) * n) // m] if n > 256 else rows
        with torch.no_grad():
            zf = z.detach().float()
            sq = (zf.square().sum(1, keepdim=True) + zf.square().sum(1, keepdim=True).T - 2.0 * (zf @ zf.T)).clamp_min(0.0)
            iu, ju = torch.triu_indices(m, m, offset=1, device=z.device)
            r_med_sq = torch.median(sq[iu, ju]).clamp_min(1e-12).to(z.dtype)
            knn_idx = torch.topk(sq.fill_diagonal_(float("inf")), k=min(8, m - 1), dim=1, largest=False).indices
        imq = (r_med_sq / (r_med_sq + (z[:, None] - z[knn_idx]).square().sum(-1))).mean()
        std_pen = (((z - z.mean(0, keepdim=True)).square().mean(0) + self.eps).sqrt() - self.target_std).square().mean()
        loss = self.weight * (imq + 0.15 * std_pen)
        s = float(getattr(self.axiom_record or self.record or self.thermal_source, "thermal_scale", 1.0))
        if s < 1.0 and loss.requires_grad:
            return loss * s + loss.detach() * (1.0 - s)
        return loss

    def state_dict_checkpoint(self) -> Dict[str, Any]:
        self._flush_emas()
        if self.quantizer_initialized and self.proto_c is not None:
            self._ensure_hits(self.proto_c.device)
        out: Dict[str, Any] = {
            "schema": 1,
            "max_prototypes": int(self.max_prototypes),
            "quantizer_initialized": bool(self.quantizer_initialized),
            "res_ema": float(self._ema["res_ema"]),
            "iso_ema": float(self._ema["iso_ema"]),
            "iso_err_ema": float(self._ema["iso_err_ema"]),
            "rows_initialized": bool(self.rows_initialized),
        }
        for nm in (*_Q_KEYS, *_R_KEYS):
            val = getattr(self, nm)
            out[nm] = None if val is None else val.detach().cpu().clone()
        return out

    def load_state_dict_checkpoint(self, state: Dict[str, Any], *, validate_only: bool = False) -> None:
        val = validate_axiom_prior_state(state)
        if val["max_prototypes"] != self.max_prototypes or (
            val["rows_initialized"]
            and self._bound_shape is not None
            and tuple(val["row_m"].shape) != tuple(self._bound_shape)
        ):
            raise ValueError("AxiomPriorRegularizer configuration or shape mismatch")
        if validate_only:
            return
        self.quantizer_initialized = bool(val["quantizer_initialized"])
        self.rows_initialized = bool(val["rows_initialized"])
        self._last_raw_var_cache = None
        self._ema = {
            "res_ema": float(val["res_ema"]),
            "iso_ema": float(val["iso_ema"]),
            "iso_err_ema": float(val["iso_err_ema"]),
        }
        self._ema_t = {}
        for nm in (*_Q_KEYS, *_R_KEYS):
            tensor = val[nm]
            if tensor is None:
                setattr(self, nm, None)
            elif self._bound_device is not None:
                setattr(self, nm, tensor.to(self._bound_device).detach().clone())
            else:
                setattr(self, nm, tensor.detach().clone())

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        return self.state_dict_checkpoint()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False) -> None:
        del strict, assign
        self.load_state_dict_checkpoint(state_dict)

    @torch.no_grad()
    def _merge_pair_ltv(self, a: int | torch.Tensor, b_idx: int | torch.Tensor) -> None:
        assert self.proto_c is not None and self.proto_var is not None and self.proto_w is not None
        assert self.v_push is not None and self.psi is not None and self.w_fake_ema is not None
        self._ensure_hits(self.proto_c.device)
        assert self.proto_hits is not None
        dim = float(self.proto_c.shape[1])
        wa = self.proto_w[a].double()
        wb = self.proto_w[b_idx].double()
        ca = self.proto_c[a].double()
        cb = self.proto_c[b_idx].double()
        wtot = (wa + wb).clamp_min(1e-12)
        c_ab = (wa * ca + wb * cb) / wtot
        var_ab = (
            wa * (self.proto_var[a].double() + (ca - c_ab).square().sum() / dim)
            + wb * (self.proto_var[b_idx].double() + (cb - c_ab).square().sum() / dim)
        ) / wtot
        self.proto_c[a] = c_ab.float()
        self.proto_var[a] = var_ab.clamp_min(1e-24).float()
        self.v_push[a] = ((wa * self.v_push[a].double() + wb * self.v_push[b_idx].double()) / wtot).clamp_min(1e-24).float()
        self.psi[a] = ((wa * self.psi[a].double() + wb * self.psi[b_idx].double()) / wtot).float()
        self.psi[b_idx] = 0.0
        self.proto_w[a] = wtot.float()
        self.proto_w[b_idx] = 0.0
        self.w_fake_ema[a] = self.w_fake_ema[a] + self.w_fake_ema[b_idx]
        self.w_fake_ema[b_idx] = 0.0
        self.proto_hits[a] = self.proto_hits[a] + self.proto_hits[b_idx]
        self.proto_hits[b_idx] = 0.0

    @torch.no_grad()
    def _split_slot_ltv(
        self,
        p_best: int | torch.Tensor,
        dst: int | torch.Tensor,
        pts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        assert self.proto_c is not None and self.proto_var is not None and self.proto_w is not None
        assert self.v_push is not None and self.psi is not None and self.w_fake_ema is not None
        self._ensure_hits(self.proto_c.device)
        assert self.proto_hits is not None
        dim = int(self.proto_c.shape[1])
        vp = self.proto_var[p_best].clamp_min(1e-24)
        c0 = self.proto_c[p_best].clone()
        delta = (0.40 * math.sqrt(float(dim))) * vp.sqrt() * _principal_axis_from_points(pts, dim, p_best, mask=mask)
        self.proto_c[p_best] = c0 - delta
        self.proto_c[dst] = c0 + delta
        v_split = (0.84 * vp).clamp_min(1e-24)
        self.proto_var[p_best] = v_split
        self.proto_var[dst] = v_split
        vp_split = (0.84 * self.v_push[p_best]).clamp_min(1e-24)
        self.v_push[p_best] = vp_split
        self.v_push[dst] = vp_split
        h_split = 0.5 * self.proto_hits[p_best]
        self.proto_hits[p_best] = h_split
        self.proto_hits[dst] = h_split
        w_split = 0.5 * (self.proto_w[p_best] + self.proto_w[dst])
        self.proto_w[p_best] = w_split
        self.proto_w[dst] = w_split
        wf_split = 0.5 * (self.w_fake_ema[p_best] + self.w_fake_ema[dst])
        self.w_fake_ema[p_best] = wf_split
        self.w_fake_ema[dst] = wf_split
        psi_split = 0.5 * (self.psi[p_best] + self.psi[dst])
        self.psi[p_best] = psi_split
        self.psi[dst] = psi_split

    @torch.no_grad()
    def update_laguerre_dual(
        self,
        xu_det: torch.Tensor,
        c_bar: torch.Tensor,
        w_mat: torch.Tensor,
        r_macro_safe: torch.Tensor,
        laguerre_lr: float,
        *,
        row_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale-equivariant per-cell relative-deficit Laguerre dual ascent with zero-droop deficits & exact gauge."""
        assert self.proto_w is not None and self.w_fake_ema is not None and self.psi is not None
        k = int(len(c_bar))
        dim = int(c_bar.shape[1])
        wm = w_mat
        n_eff_t = row_mask.float().sum().clamp_min(1.0) if row_mask is not None else xu_det.new_full((), float(len(xu_det)))
        sparse_t = (2.0 * n_eff_t) < float(k)
        if (
            self.proto_c is not None
            and self.proto_var is not None
            and len(self.proto_c) == k
            and (row_mask is not None or 2 * len(xu_det) < k)
        ):
            pv = self.proto_var.clamp_min(1e-24)
            sig = pv.sqrt()
            sqd = math.sqrt(float(dim))
            dcc = torch.cdist(self.proto_c, self.proto_c, p=2.0)
            s_max = torch.maximum(sig[:, None], sig[None, :])
            s_min = torch.minimum(sig[:, None], sig[None, :]).clamp_min(1e-12)
            adj = ((dcc <= (2.6 * sqd * s_max)) & (s_max <= (2.2 * s_min))).float().fill_diagonal_(1.0)
            for _ in range(4):
                adj = ((adj @ adj) > 0).float()
            cw = adj * self.proto_w.clamp_min(1e-12)[None, :]
            cw = cw / cw.sum(1, keepdim=True).clamp_min(1e-12)
            cc = cw @ self.proto_c
            cv = (cw * (pv[None, :] + (self.proto_c[None, :] - cc[:, None]).square().sum(-1) / float(dim))).sum(1).clamp_min(1e-24)
            ext_min = dcc.masked_fill(adj > 0.5, float("inf")).min(1).values
            iso_mask = (adj @ (~(ext_min >= (4.2 * sqd * cv.sqrt()))).float()) < 0.5
            wm = torch.where(sparse_t, torch.where(iso_mask[:, None], cw, wm), wm)
        k_f = torch.where(sparse_t, (1.0 / (wm > 0).float().sum(1).clamp_min(1.0)).sum(), xu_det.new_full((), float(k)))
        k_star = (torch.cdist(xu_det, c_bar, p=2.0).square() - (wm @ self.psi)[None, :]).argmin(1)
        if row_mask is not None:
            w_hard = (wm[k_star] * row_mask.float()[:, None]).sum(0) / n_eff_t
        else:
            w_hard = wm[k_star].mean(0)
        w_fe = 0.88 * self.w_fake_ema + 0.12 * w_hard
        self.w_fake_ema = w_fe / w_fe.sum().clamp_min(1e-12)
        pw_c = wm @ self.proto_w
        wf_c = wm @ self.w_fake_ema
        tv_sparse = 0.5 * ((pw_c - wf_c).abs() / (wm > 0).float().sum(1).clamp_min(1.0)).sum()
        tv_dense = 0.5 * (self.proto_w - self.w_fake_ema).abs().sum()
        tv_now = torch.where(sparse_t, tv_sparse, tv_dense)
        tv_floor = 0.35 * torch.sqrt(k_f / (n_eff_t * 8.0).clamp_min(1.0))
        d_max = ((pw_c - wf_c).clamp_min(0.0) / torch.maximum(pw_c, 1.0 / k_f.clamp_min(1.0))).amax()
        g_psi = torch.maximum(
            _smoothstep01((tv_now - tv_floor) / tv_floor.clamp_min(0.08)),
            _smoothstep01((d_max - 0.15) / 0.20),
        )
        eta = g_psi * (float(laguerre_lr) * k_f).clamp(0.35, 3.0) * float(dim) * r_macro_safe.square()
        psi_next = wm @ (
            self.psi * (1.0 - 0.004 * g_psi * (1.0 - (pw_c > wf_c).float()))
            + eta * (self.proto_w - (0.35 * w_hard + 0.65 * self.w_fake_ema))
        )
        psi_next = psi_next - psi_next.mean()
        cap_scale = ((2.5 * float(dim) * r_macro_safe.square()).clamp_min(1e-24) / psi_next.abs().amax().clamp_min(1e-24)).clamp_max(1.0)
        self.psi = psi_next * cap_scale
        return k_star, g_psi

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        if self._bound_device is None:
            self._bound_device = yr.device
        for nm in (*_Q_KEYS, *_R_KEYS):
            t = getattr(self, nm)
            if isinstance(t, torch.Tensor) and t.device != yr.device:
                setattr(self, nm, t.to(yr.device))
        x = yr.detach().float()
        b = int(yr.shape[0])
        dim = int(yr.shape[1])
        if not self.quantizer_initialized or self.proto_c is None or self.proto_c.shape[1] != dim:
            k = min(self.max_prototypes, max(4, (3 * b) // 4) if b >= 8 else b)
            proj = torch.remainder(torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) * _PHI1, 1.0) - 0.5
            order = torch.argsort(x @ (proj / proj.norm().clamp_min(1e-8)), stable=True)
            pick = ((torch.arange(k, device=x.device) * b + b // 2) // k).clamp(0, b - 1)
            self.proto_c = x[order[pick]].clone()
            d_nn = torch.cdist(self.proto_c, self.proto_c).fill_diagonal_(float("inf")).min(1).values if k >= 2 else x.new_empty(0)
            if (d_nn > 1e-12).any():
                v0 = 0.25 * float(d_nn[d_nn > 1e-12].median().item()) ** 2 / float(dim + 2)
            else:
                v0 = float(x.var(0, unbiased=False).mean().clamp_min(1e-24).item()) / float(max(1, k))
            self.proto_var = torch.full((k,), max(v0, 1e-24), device=x.device)
            self.proto_w = torch.full((k,), 1.0 / float(k), device=x.device)
            self.proto_hits = torch.ones(k, device=x.device)
            self.w_fake_ema = self.proto_w.clone()
            self.psi = torch.zeros(k, device=x.device)
            self.v_push = self.proto_var.clone()
            self.quantizer_initialized = True
            return

        assert self.proto_var is not None and self.proto_w is not None and self.v_push is not None
        assert self.w_fake_ema is not None and self.psi is not None
        k = len(self.proto_c)
        c_prev = self.proto_c.clone()
        v_prev = self.proto_var.clone()
        self._ensure_hits(x.device)
        assert self.proto_hits is not None
        d2_min, assign = torch.cdist(x, c_prev, p=2.0).square().min(1)
        med_v = v_prev.median().clamp_min(1e-24)
        raw_sp = 2.0 * x.var(0, unbiased=False).sum()
        payload = torch.cat((x.new_ones((b, 1)), ((x - c_prev[assign]).square().sum(1) / float(dim))[:, None], x), dim=1)
        sums = x.new_zeros((k, dim + 2)).index_add_(0, assign, payload)
        b_cnt = sums[:, 0]
        b_mean = sums[:, 2:] / b_cnt.clamp_min(1.0)[:, None]
        v_est = (x.new_zeros(k).index_add_(0, assign, (x - b_mean[assign]).square().sum(1) / float(dim)) / (b_cnt - 1.0).clamp_min(1.0)).clamp_min(1e-24)
        has_sp = raw_sp > 1e-24
        pair_sp = raw_sp.clamp_min(1e-24)
        shock = ((torch.where(has_sp, torch.sqrt(d2_min.mean() / pair_sp), raw_sp.new_zeros(())) - 0.45) / 0.35).clamp(0.0, 1.0)

        # Bounded, clock-free stationary occupancy EMA (converges to E[b_cnt], never climbs monotonically).
        h_base = torch.where(shock > 0.25, self.proto_hits * (1.0 - 0.8 * shock), self.proto_hits)
        self.proto_hits = (0.85 * h_base + 0.15 * b_cnt).clamp_min(0.0)

        aw = 0.02 + 0.48 * shock * shock
        hit = b_cnt > 0
        far_unhit = has_sp & (~hit) & ((c_prev - x.mean(0, keepdim=True)).square().sum(1) > 100.0 * pair_sp)
        aw_k = torch.where(far_unhit, aw.new_full((), 0.50), aw)
        ac_rate = (0.035 + 0.25 * torch.rsqrt(1.0 + self.proto_hits) + 0.38 * shock * shock) * (b_cnt / (b_cnt + 2.0))
        ac = ac_rate.clamp_max(0.50)[:, None]
        pw = (1.0 - aw_k) * self.proto_w + aw_k * (b_cnt / float(b))
        v_jump = (shock > 0.5) | (v_est < 0.04 * v_prev) | (v_est > 25.0 * v_prev)
        av_s = (0.65 + 0.25 * shock * shock).clamp_max(0.90)
        v_tot = (raw_sp / (2.0 * float(dim))).clamp_min(1e-24)
        v_col = has_sp & (v_prev > 4.0 * v_tot)
        v_col_g = torch.exp(
            0.15 * torch.log(v_prev.clamp_min(1e-24)) + 0.85 * torch.log((v_tot / float(max(1, k))).clamp_min(1e-24))
        ).clamp_min(1e-24)
        v_base = torch.where(v_col, v_col_g, v_prev)
        v_geo = torch.exp((1.0 - av_s) * torch.log(v_prev.clamp_min(1e-24)) + av_s * torch.log(v_est.clamp_min(1e-24))).clamp_min(1e-24)

        self.proto_w = pw / pw.sum().clamp_min(1e-12)
        self.proto_var = torch.where(b_cnt >= 2.0, torch.where(v_jump, v_geo, 0.88 * v_base + 0.12 * v_est), v_base)
        self.proto_c = torch.where(hit[:, None], (1.0 - ac) * c_prev + ac * b_mean, c_prev)
        self.v_push = torch.where((b_cnt >= 2.0) & v_jump, v_geo, torch.where(v_col, v_col_g, self.v_push))

        mean_w = 1.0 / float(k)
        dead = self.proto_w < (0.04 / float(k))
        gamma = float(min(dim, 4)) / float(min(dim, 4) + 2)
        max_ur, worst_j = (d2_min / torch.maximum(36.0 * float(dim) * torch.maximum(v_prev[assign], med_v), 0.08 * pair_sp)).max(0)
        ge = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
        max_e, pi_t = ge.max(0)
        do_uncov = (max_ur > 1.0) & has_sp if k >= 4 else raw_sp.new_zeros((), dtype=torch.bool)
        need_g = (
            (max_e > 10.0 * ge.median().clamp_min(1e-24)) & (self.proto_w[pi_t] > 0.25 * mean_w)
            if k >= 16
            else raw_sp.new_zeros((), dtype=torch.bool)
        )
        any_d = dead.any()
        do_sp = (~do_uncov) & has_sp & (any_d | need_g) if k >= 2 else raw_sp.new_zeros((), dtype=torch.bool)
        u_flag, s_flag, dead_flag = torch.stack((do_uncov, do_sp, any_d)).tolist()
        if u_flag:
            new_w_t = torch.maximum(aw / float(b), aw.new_full((), 0.05 * mean_w))
            y_far = x[worst_j]
            slot_t: Optional[torch.Tensor] = None
            if dead_flag:
                slot_t = self.proto_w.argmin()
            elif k < self.max_prototypes:
                self.proto_c = torch.cat((self.proto_c, y_far[None]))
                self.proto_var = torch.cat((self.proto_var, med_v[None]))
                self.v_push = torch.cat((self.v_push, med_v[None]))
                self.proto_w = torch.cat((self.proto_w, new_w_t[None]))
                self.proto_hits = torch.cat((self.proto_hits, self.proto_hits.new_zeros((1,))))
                self.w_fake_ema = torch.cat((self.w_fake_ema, new_w_t[None]))
                self.psi = torch.cat((self.psi, self.psi.new_zeros((1,))))
            else:
                flat = torch.cdist(self.proto_c, self.proto_c).square().fill_diagonal_(float("inf")).argmin()
                self._merge_pair_ltv(flat // k, flat % k)
                slot_t = flat % k
            if slot_t is not None:
                self.proto_c[slot_t] = y_far
                self.proto_var[slot_t] = med_v
                self.v_push[slot_t] = med_v
                self.proto_w[slot_t] = new_w_t
                self.proto_hits[slot_t] = 0.0
                self.w_fake_ema[slot_t] = new_w_t
                self.psi[slot_t] = 0.0
        elif s_flag:
            if not dead_flag:
                im = torch.arange(k, device=x.device) == pi_t
                dmat = torch.cdist(self.proto_c, self.proto_c).square().fill_diagonal_(float("inf"))
                d_min, flat = dmat.masked_fill(im[:, None] | im[None, :], float("inf")).view(-1).min(0)
                if bool((d_min <= 8.0 * float(dim) * med_v).item()):
                    a_t, b_t = flat // k, flat % k
                    self._merge_pair_ltv(a_t, b_t)
                    ge_new = ((self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var).masked_fill(
                        torch.arange(k, device=x.device) == b_t, 0.0
                    )
                    pb_t = ge_new.argmax()
                    if bool(((pb_t != b_t) & (self.proto_w[pb_t] > 0.20 * mean_w)).item()):
                        self._split_slot_ltv(pb_t, b_t, x, mask=(assign == pb_t))
            else:
                for di in torch.nonzero(dead, as_tuple=False).squeeze(-1)[:4]:
                    ge_new = ((self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var).masked_fill(
                        torch.arange(k, device=x.device) == di, 0.0
                    )
                    pb_t = ge_new.argmax()
                    if not bool(((pb_t != di) & (self.proto_w[pb_t] > 0.20 * mean_w)).item()):
                        break
                    self._split_slot_ltv(pb_t, di, x, mask=(assign == pb_t))
        if u_flag or s_flag:
            self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
            self.w_fake_ema = self.w_fake_ema / self.w_fake_ema.sum().clamp_min(1e-12)
            self.psi = self.psi - self.psi.mean()


def should_use_axiom_pullback(
    trainer: Any,
    real: torch.Tensor,
    latent: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
) -> bool:
    """True on unconditioned MoG priors (2 <= D <= 64, sigma_z > 0) or degenerate point-mass targets."""
    del latent, indices
    r = trainer.recipe
    if (
        r.prior_regularizer != "axiom"
        or float(r.t1_pullback_weight) <= 0.0
        or r.encoder_mode != "none"
        or r.conditioning != "scalar"
        or real.ndim != 2
        or not (1 <= int(real.shape[1]) <= 64)
    ):
        return False
    if int(real.shape[1]) >= 2 and has_positive_prior_sigma(trainer.prior):
        return True
    rf = real.detach().float()
    is_pt = bool((rf.var(0, unbiased=False).sum() < (1e-11 * rf.square().sum(1).mean()).clamp_min(1e-22)).item())
    if hasattr(getattr(trainer, "prior_regularizer", None), "_last_raw_var_cache"):
        trainer.prior_regularizer._last_raw_var_cache = (real, is_pt)
    return is_pt


def _finalize_gen_step(
    trainer: Any,
    base_prior: Any,
    loss_g: torch.Tensor,
    loss_gan: torch.Tensor,
    fake_g: torch.Tensor,
    real_g: torch.Tensor,
    fake_logits: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
    loss_d: torch.Tensor,
    penalty: torch.Tensor,
    g_scale: float | torch.Tensor,
    wakeup_boost: float | torch.Tensor,
    prior_scale: float,
    res_ema_t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    r = trainer.recipe
    zero = loss_gan.new_zeros(())
    rec_d = getattr(trainer.opt_d, "record", None)
    opt_g = trainer.opt_g
    tr = r.kinetic_transport_loss(fake_g, real_g) if r.kinetic_transport_weight else zero
    tr_loc = r.kinetic_transport_local_loss(fake_g, real_g) if r.kinetic_transport_local_weight else zero
    if getattr(trainer, "_prior_penalty", False):
        z_src = base_prior.z if r.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        pm = trainer.prior.regularize(z_src, sampled_codes=latent, indices=indices, generated=fake_g, logits=fake_logits)
    else:
        pm = None
    l2_term = (
        r.prior_l2 * base_prior.z.square().mean()
        if (hasattr(base_prior, "z") and base_prior.z.requires_grad and r.prior_l2 > 0)
        else zero
    )
    loss_g = loss_g + tr + tr_loc + (pm if pm is not None else zero) + l2_term
    if rec_d is not None:
        rec_d._disc_gate = 0.0
        if res_ema_t is not None and hasattr(rec_d, "_ema_t"):
            rec_d._ema_t["ema_push_res"] = res_ema_t.detach().double()
        else:
            rec_d.ema_push_res = float(trainer.prior_regularizer.res_ema)
    opt_g.zero_grad()
    if getattr(opt_g, "direction_blend", False) and hasattr(opt_g, "bind_protected_from_grads"):
        loss_g.backward()
        opt_g.bind_protected_from_grads()
    else:
        constraint_geometry_backward(loss_g, opt_g, (loss_g,))
    if hasattr(base_prior, "z") and base_prior.z.requires_grad and base_prior.z.grad is None:
        base_prior.z.grad = torch.zeros_like(base_prior.z)
    trainer.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
    if hasattr(opt_g, "bind_axiom_modulation"):
        opt_g.bind_axiom_modulation(g_scale=g_scale, wakeup_boost=wakeup_boost, prior_scale=prior_scale)
        opt_g.step()
    else:
        gs = float(g_scale.item()) if isinstance(g_scale, torch.Tensor) else float(g_scale)
        saved = [g["lr"] for g in opt_g.param_groups]
        for grp in opt_g.param_groups:
            grp["lr"] *= gs
        try:
            opt_g.step()
        finally:
            for grp, lr0 in zip(opt_g.param_groups, saved):
                grp["lr"] = lr0
    trainer.policy.after_generator_step()
    return loss_g, pm, tr, tr_loc


def axiom_pullback_step(
    trainer: Any,
    real: torch.Tensor,
    generator_real: Any,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    sigma_out: Any,
    noise: torch.Generator,
    critic: nn.Module,
    loss_d: torch.Tensor,
    penalty: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Execute Axiom Law-1 Pullback: Exact LTV + Unbiased 2-Sketch Isometry + Scale-Equivariant Laguerre."""
    reg = trainer.prior_regularizer
    recipe = trainer.recipe
    b = len(real)
    base_prior, sigma_z, can_condense = _prior_kernel_spec(trainer.prior)
    has_sigma = sigma_z > 0.0
    rv_cache = getattr(reg, "_last_raw_var_cache", None)
    reg._last_raw_var_cache = None
    if rv_cache is not None and rv_cache[0] is real:
        is_case_a = bool(rv_cache[1])
    else:
        rf_a = real.detach().float()
        is_case_a = bool((rf_a.var(0, unbiased=False).sum() < (1e-11 * rf_a.square().sum(1).mean()).clamp_min(1e-22)).item())

    def gen_clean(z_in: torch.Tensor) -> torch.Tensor:
        return trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_in, 0.0, noise))

    if is_case_a:
        mu_det = None
        eps_g = None
        fake_g = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, latent, sigma_out, noise, rows=indices))
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_flat = fake_flat
    else:
        if isinstance(base_prior, MoGParticlePrior) and not bool(getattr(trainer.prior, "transforms_codes", False)):
            mu_g = base_prior.means()[indices]
        elif can_condense:
            mu_g = base_prior.z[indices]
        else:
            mu_g = latent.detach()
        mu_det = mu_g.detach()
        eps_g = ((latent - mu_g) / sigma_z).detach() if has_sigma else None
        z_gen = (mu_det + sigma_z * eps_g) if (has_sigma and eps_g is not None) else mu_det
        fake_g = trainer._postprocess(trainer.prior, trainer._generate(trainer.G, z_gen, sigma_out, noise, rows=indices))
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_flat = gen_clean(mu_det).reshape(b, -1) if has_sigma else fake_flat

    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(None if is_case_a else fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:] or len(real_g) != len(real):
        raise ValueError("generator_real must match the real sample shape and batch size")
    fake_logits = critic(fake_g)
    real_logits = critic(real_g)
    rec_d = getattr(trainer.opt_d, "record", None)
    if rec_d is not None and hasattr(rec_d, "observe_margin_and_scale"):
        rec_d.observe_margin_and_scale(real_logits, fake_logits, real_g, fake_g)
    if recipe.loss == "non_saturating":
        loss_gan = F.softplus(-fake_logits).mean()
    else:
        loss_gan = trainer.loss.g_loss(fake_logits, real_logits)
    yr = real_g.detach().reshape(b, -1)

    if is_case_a:
        with torch.no_grad():
            err = (fake_flat.detach().float() - yr.float()).norm(dim=-1)
            cur = max(float(err.mean().item()), 0.35 * float(err.amax().item()))
            if cur > max(3.5 * reg.res_ema, reg.res_ema + 0.12):
                reg.res_ema = min(cur, 10.0)
            else:
                reg.res_ema = 0.88 * reg.res_ema + 0.12 * min(cur, 10.0)
            g_sc = max(0.015, min(1.0, reg.res_ema / 0.025))
            zd_pt = latent.detach()
            u_pt = torch.ones_like(zd_pt) / math.sqrt(float(max(1, zd_pt.shape[1])))
            x_pm = gen_clean(torch.cat((zd_pt + 1e-3 * u_pt, zd_pt - 1e-3 * u_pt), 0)).reshape(2, b, -1).float()
            s_c = ((x_pm[0] - x_pm[1]).abs() / 2e-3).clamp_min(1e-5)
            med_s = torch.maximum(
                torch.quantile(s_c, 0.5, dim=0, keepdim=True),
                s_c.new_full((1, int(fake_flat.shape[1])), max(0.15 * reg.res_ema, 1e-4)),
            )
            w_pt = (med_s / s_c).clamp(1.0, 12.0).to(fake_flat.dtype)
        z_l2 = (
            0.004 * base_prior.z.float().square().mean().to(fake_flat.dtype)
            if (hasattr(base_prior, "z") and base_prior.z.requires_grad)
            else 0.0
        )
        loss_pt = (torch.sqrt(1e-6 + w_pt * (fake_flat - yr).square()) - 1e-3).mean() + z_l2
        if hasattr(base_prior, "z") and base_prior.z.requires_grad and recipe.prior_reg > 0:
            z_sub = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
            prior_diag = reg(z_sub).detach()
        else:
            prior_diag = loss_gan.new_zeros(())
        comb_loss = (0.02 * min(1.0, reg.res_ema / 0.25)) * loss_gan + float(recipe.t1_pullback_weight) * loss_pt
        loss_g, pm, tr, tr_l = _finalize_gen_step(
            trainer,
            base_prior,
            comb_loss,
            loss_gan,
            fake_g,
            real_g,
            fake_logits,
            latent,
            indices,
            loss_d,
            penalty,
            g_sc,
            1.0,
            g_sc,
        )
        return loss_g, loss_gan, prior_diag, pm, tr, tr_l

    dim = int(yr.shape[1])
    was_init = reg.quantizer_initialized
    dev = yr.device
    with torch.no_grad():
        reg.update_quantizer(yr)
        yf = yr.float()
        k_proto = len(reg.proto_c)
        r_macro_safe = yf.std(0, unbiased=False).mean().clamp_min(1e-12)
        s_ot = (0.5 * yf.var(0, unbiased=False).mean().clamp_min(1e-24).sqrt()).clamp_min(1e-12)
        c_bar_all, var_bar_all, w_mat_all, multicell_coal = coalesce_pythagorean_ltv(
            reg.proto_c, reg.proto_var, reg.proto_w
        )
        sigma_bar_all = var_bar_all.sqrt()
        sigma_cell_med = torch.quantile(sigma_bar_all, 0.35).clamp_min(1e-12)
        x_ctr_det = x_ctr_flat.detach().float()
        fake_det = fake_flat.detach().float()
        near_p = torch.cdist(x_ctr_det, c_bar_all, p=2.0).argmin(1)
        is_coal_i = multicell_coal[near_p]
        xff = torch.where(is_coal_i[:, None], x_ctr_det, fake_det)
        scale_ratio = xff.std(0, unbiased=False).mean().clamp_min(1e-12) / r_macro_safe
        v_push_bar = (w_mat_all @ reg.v_push).clamp_min(0.0)
        wakeup_t = torch.where(
            scale_ratio < 0.50,
            scale_ratio.new_full((), 5.0),
            torch.where(scale_ratio < 0.80, scale_ratio.new_full((), 2.0), scale_ratio.new_full((), 1.0)),
        )
        v_mu_star_all = (var_bar_all - v_push_bar).clamp_min(0.0)
        has_coal_t = multicell_coal.any()
        iso_prev = reg._dev_ema("iso_ema", dev)
        res_prev = reg._dev_ema("res_ema", dev)
        iso_err_prev = reg._dev_ema("iso_err_ema", dev)
        iso_val_t = (sigma_z * r_macro_safe / sigma_cell_med).clamp(1.0, 4.5) if has_sigma else iso_prev.new_ones(())
        if has_sigma:
            iso_ema_t = iso_val_t if not was_init else (0.95 * iso_prev + 0.05 * iso_val_t)
        else:
            iso_ema_t = iso_prev.new_ones(())
        reg._ema_t["iso_ema"] = iso_ema_t
        d_xp_phys = torch.cdist(xff, c_bar_all, p=2.0)
        cb_m = (reg.proto_w[:, None] * c_bar_all).sum(0, keepdim=True)
        if k_proto >= 4:
            xff_w = (xff - xff.mean(0, keepdim=True)) / xff.std(0, unbiased=False, keepdim=True).clamp_min(1e-12)
            cb_std = (reg.proto_w[:, None] * (c_bar_all - cb_m).square()).sum(0, keepdim=True).sqrt().clamp_min(1e-12)
            d_xp_white = torch.cdist(xff_w, (c_bar_all - cb_m) / cb_std, p=2.0)
        else:
            d_xp_white = d_xp_phys
        d_xp_route = torch.where((scale_ratio < 0.35) & (k_proto >= 4), d_xp_white, d_xp_phys)
        y_ot = _sinkhorn_barycenter(
            -(d_xp_route / d_xp_route.mean().clamp_min(1e-12)) / (0.25 * res_prev).clamp(0.010, 0.06),
            -math.log(float(b)),
            torch.log(reg.proto_w.clamp_min(1e-12)),
            c_bar_all,
            b,
            fake_flat.dtype,
        )
        cov_dist = torch.maximum((reg.proto_w * d_xp_phys.min(0).values).sum(), d_xp_phys.min(1).values.median())
        res_cur = ((cov_dist - 1.45 * math.sqrt(float(dim)) * sigma_cell_med).clamp_min(0.0) / r_macro_safe).clamp_max(10.0)
        res_ema_t = 0.92 * res_prev + 0.08 * res_cur
        reg._ema_t["res_ema"] = res_ema_t
        comp_rep = (w_mat_all > 0.0).float().argmax(1)
        cid_i = comp_rep[near_p]
        cnt_f = torch.bincount(cid_i, minlength=k_proto).float()
        mean_c = x_ctr_det.new_zeros((k_proto, dim)).index_add_(0, cid_i, x_ctr_det) / cnt_f.clamp_min(1.0)[:, None]
        if can_condense:
            bm_t = _box_muller_targets(indices, near_p, comp_rep, k_proto, c_bar_all, v_mu_star_all, multicell_coal, dim).to(fake_flat.dtype)
            halo = ((xff - c_bar_all[near_p]).norm(dim=-1) / (3.5 * math.sqrt(float(dim)) * sigma_bar_all[near_p]).clamp_min(1e-12)).clamp(0.0, 1.0).square()[:, None].to(fake_flat.dtype)
            use_bm = is_coal_i & (v_mu_star_all[near_p] > 0.0) & (scale_ratio >= 0.65)
            y_ot = torch.where(use_bm[:, None], halo * y_ot + (1.0 - halo) * bm_t, y_ot)
        vv = (is_coal_i & (cnt_f[cid_i] >= 2.0)).float()
        nv = cnt_f[cid_i].clamp_min(2.0)
        rho_num = (((nv / (nv - 1.0)) * ((x_ctr_det - mean_c[cid_i]).square().sum(-1) / float(dim))) / var_bar_all[near_p].clamp_min(1e-24)).clamp(0.0, 0.75)
        rho_mu = (rho_num * vv).sum() / vv.sum().clamp_min(1.0)
        v_iso_i = torch.where(
            is_coal_i & (vv.sum() > 0.0),
            var_bar_all[near_p] * (1.0 - torch.minimum(rho_mu, (1.0 - float(dim) / nv).clamp(0.0, 0.75))).clamp_min(0.25),
            var_bar_all[near_p],
        ).clamp_min(1e-24)
        inv_w_i = (1.0 / (float(k_proto) * reg.proto_w[near_p]).clamp_min(1e-4)).sqrt().clamp(0.5, 4.0).to(fake_flat.dtype)
        w_sat = torch.ones(b, device=dev, dtype=fake_flat.dtype)
        y_bound = yf.abs().amax(0, keepdim=True) + 3.5 * sigma_cell_med
        if has_sigma and eps_g is not None and mu_det is not None:
            e_n = eps_g.float().norm(dim=-1)
            ve = e_n >= 0.45
            s_obs = (fake_det - x_ctr_det).norm(dim=-1) / (sigma_z * e_n.clamp_min(0.45))
            s_med_c = torch.where(ve, s_obs, s_obs.new_tensor(float("inf"))).sort().values[(ve.sum() // 2).clamp_max(b - 1)].clamp_min(1e-12)
            s_ref = torch.where(
                has_coal_t & ve.any(),
                s_med_c,
                (v_iso_i.sqrt() * math.sqrt(float(dim) / float(max(1, mu_det.shape[1])))) / max(sigma_z, 1e-4),
            )
            rail = (
                ve
                & (s_obs < torch.where(has_coal_t, 0.50, 0.40) * s_ref)
                & (has_coal_t | ((xff - y_ot.detach().float()).norm(dim=-1) > 1.35 * math.sqrt(float(dim)) * sigma_bar_all[near_p]))
            )
            w_sat = torch.where(
                rail,
                ((torch.where(has_coal_t, 1.0, 0.45) * s_ref) / s_obs.clamp_min(1e-12)).clamp(1.0, 8.0),
                1.0,
            ).to(fake_flat.dtype)

    x_ot_src = x_ctr_flat if has_sigma else fake_flat
    sc_med_dt = sigma_cell_med.to(fake_flat.dtype)
    s_ot_dt = s_ot.to(fake_flat.dtype)
    loss_g_iso = fake_flat.new_zeros(())
    loss_g_ot = (
        (w_sat * (torch.sqrt(sc_med_dt.square() + (x_ot_src - y_ot).square().sum(-1)) - sc_med_dt)).mean() / s_ot_dt
        + 2.0 * (F.relu(fake_flat.float().abs() - y_bound) / s_ot).square().mean().to(fake_flat.dtype)
    )

    if has_sigma and mu_det is not None and float(recipe.t1_iso_weight) > 0.0:
        d_z_m = int(mu_det.shape[1])
        ph_iso = float(getattr(rec_d, "chord_phase", 0.0)) if rec_d is not None else 0.0
        if d_z_m > 8 and d_z_m <= dim:
            m_iso = min(b, max(32, 2048 // d_z_m))
            eye_z = torch.eye(d_z_m, device=dev, dtype=mu_det.dtype)
            off = float(sigma_z) * eye_z[None]
            z_pm = torch.cat(
                (
                    (mu_det[:m_iso, None] + off).reshape(m_iso * d_z_m, d_z_m),
                    (mu_det[:m_iso, None] - off).reshape(m_iso * d_z_m, d_z_m),
                ),
                dim=0,
            )
            xc = gen_clean(z_pm).reshape(2, m_iso, d_z_m, -1).float()
            jc = 0.5 * (xc[0] - xc[1])
            gm = torch.einsum("mad,mbd->mab", jc, jc)
            v_dir = v_iso_i[:m_iso].float().clamp_min(1e-24) * (float(dim) / float(d_z_m))
            gn = gm / v_dir.view(m_iso, 1, 1)
            eye_zf = eye_z.float()[None]
            with torch.no_grad():
                v_tr = gm.detach().diagonal(dim1=-2, dim2=-1).clamp_min(1e-24).mean(-1) * (float(d_z_m) / float(dim))
                gid = torch.where(has_coal_t, cid_i[:m_iso], near_p[:m_iso])
                idx_map = torch.where(has_coal_t, comp_rep, torch.arange(k_proto, device=dev))
                g_cnt = torch.bincount(gid, minlength=k_proto).float()
                v_acc = v_tr.new_zeros(k_proto).index_add_(0, gid, v_tr)[idx_map] / g_cnt[idx_map].clamp_min(1.0)
                reg.v_push = torch.where(g_cnt[idx_map] > 0, 0.90 * reg.v_push + 0.10 * v_acc, reg.v_push)
                iso_frob = (gn.detach() - eye_zf).square().sum((1, 2)).sqrt() / math.sqrt(float(d_z_m))
                iso_err_ema_t = 0.92 * iso_err_prev + 0.08 * (0.60 * iso_frob.mean() + 0.40 * torch.quantile(iso_frob, 0.92)).clamp_max(5.0)
                reg._ema_t["iso_err_ema"] = iso_err_ema_t
            l_chol = torch.linalg.cholesky(0.5 * (gn + gn.transpose(-1, -2)) + 1e-3 * eye_zf)
            ld_div = (
                (
                    torch.diagonal(gn, dim1=-2, dim2=-1).sum(-1)
                    - 2.0 * torch.diagonal(l_chol, dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)
                    - float(d_z_m) * (1.0 - math.log(1.001))
                )
                / float(d_z_m)
            ).clamp_min(0.0)
            loss_g_iso = (4.5 * (inv_w_i[:m_iso].float() * ld_div).mean()).to(fake_flat.dtype)
        else:
            m_iso = b if d_z_m <= 4 else min(b, 256)
            sp1, sp2 = central_pullback_cov(gen_clean, mu_det[:m_iso], sigma_z, phase=ph_iso, return_pair=True)
            sigma_push = sp1 if d_z_m <= 8 else (0.5 * (sp1 + sp2))
            v_iso_sub = v_iso_i[:m_iso].to(sp1.dtype)
            eye_d = torch.eye(dim, device=dev, dtype=sp1.dtype)[None]
            with torch.no_grad():
                sp_d = sigma_push.detach().float()
                diag_p = sp_d.diagonal(dim1=-2, dim2=-1).clamp_min(1e-24)
                if dim == 2:
                    disc_det = torch.sqrt(((sp_d[:, 0, 0] - sp_d[:, 1, 1]).square() + 4.0 * sp_d[:, 0, 1].square()).clamp_min(1e-36))
                    lmin_det = (0.5 * (sp_d[:, 0, 0] + sp_d[:, 1, 1] - disc_det)).clamp_min(1e-24)
                else:
                    lmin_det = diag_p.amin(-1)
                v_tr = diag_p.sum(-1) / float(dim)
                gid = torch.where(has_coal_t, cid_i[:m_iso], near_p[:m_iso])
                idx_map = torch.where(has_coal_t, comp_rep, torch.arange(k_proto, device=dev))
                g_cnt = torch.bincount(gid, minlength=k_proto).float()
                sig_i_f = v_iso_i[:m_iso].sqrt()
                v_acc = v_tr.new_zeros(k_proto).index_add_(0, gid, v_tr)[idx_map] / g_cnt[idx_map].clamp_min(1.0)
                reg.v_push = torch.where(g_cnt[idx_map] > 0, 0.90 * reg.v_push + 0.10 * v_acc, reg.v_push)
                w_iso_sat = (sig_i_f / lmin_det.sqrt().clamp_min(0.15 * sig_i_f)).clamp(1.0, 4.0).to(sigma_push.dtype)
                iso_frob = ((sp_d / v_iso_i[:m_iso].view(m_iso, 1, 1).clamp_min(1e-24)) - eye_d.float()).square().sum((1, 2)).sqrt()
                iso_err_ema_t = 0.92 * iso_err_prev + 0.08 * (0.60 * iso_frob.mean() + 0.40 * torch.quantile(iso_frob, 0.92)).clamp_max(5.0)
                reg._ema_t["iso_err_ema"] = iso_err_ema_t
            w_iso = inv_w_i[:m_iso].to(sigma_push.dtype) * w_iso_sat
            sig_iso = v_iso_sub.sqrt()
            sc_iso = sc_med_dt.to(sigma_push.dtype).clamp_min(1e-12)
            if dim == 2:
                a2 = sigma_push[:, 0, 0]
                c2 = sigma_push[:, 1, 1]
                b2 = sigma_push[:, 0, 1]
                disc2 = torch.sqrt(((a2 - c2).square() + 4.0 * b2.square()).clamp_min(1e-36))
                r_lo = ((0.5 * (a2 + c2 - disc2)).clamp_min(1e-24).sqrt() - sig_iso) / sc_iso
                r_hi = ((0.5 * (a2 + c2 + disc2)).clamp_min(1e-24).sqrt() - sig_iso) / sc_iso
                iso_rad = (w_iso * (r_lo.square() + r_hi.square())).mean()
            elif d_z_m <= 8:
                iso_rad = (w_iso * ((torch.diagonal(sigma_push, dim1=-2, dim2=-1).clamp_min(1e-24).sqrt() - sig_iso[:, None]) / sc_iso).square().sum(-1)).mean()
            else:
                tr_rms = (torch.diagonal(sigma_push, dim1=-2, dim2=-1).sum(-1) / float(dim)).clamp_min(1e-24).sqrt()
                iso_rad = (w_iso * ((tr_rms - sig_iso) / sc_iso).square()).mean()
            dn_cov = (sig_iso * sc_iso).clamp_min(1e-24).view(m_iso, 1, 1)
            tgt_cov = v_iso_sub.view(m_iso, 1, 1) * eye_d
            u_cov = (w_iso * (((sp1 - tgt_cov) / dn_cov) * ((sp2 - tgt_cov) / dn_cov)).sum((1, 2))).mean() / float(dim)
            loss_g_iso = (0.45 * iso_rad + 0.55 * u_cov).to(fake_flat.dtype)
    else:
        iso_err_ema_t = iso_err_prev

    eq_g = torch.where(has_coal_t, ((0.030 - res_ema_t) / 0.020).clamp(0.0, 1.0), res_ema_t.new_zeros(()))
    adv_w = torch.where(has_coal_t, 0.038 * (1.0 - eq_g) + 0.008 * eq_g, eq_g.new_full((), 0.15)).to(loss_gan.dtype)
    g_floor = torch.where(has_coal_t, 0.022 * (1.0 - eq_g) + 0.012 * eq_g, eq_g.new_full((), 0.04))
    row_floor = torch.where(has_coal_t, 0.15 * (1.0 - eq_g) + 0.38 * eq_g, eq_g.new_full((), 0.15))
    iso_coef = 1.0 if (has_sigma and mu_det is not None and int(mu_det.shape[1]) > 8) else 0.28
    iso_gate = iso_coef * ((iso_err_ema_t - 0.22) / 0.45).clamp(0.0, 1.0)
    g_scale_t = torch.where(
        scale_ratio < 0.60,
        res_ema_t.new_ones(()),
        torch.maximum(g_floor, torch.minimum(res_ema_t.new_ones(()), torch.maximum(res_ema_t / 0.10, iso_gate))),
    )
    total_g_loss = adv_w * loss_gan + float(recipe.t1_pullback_weight) * loss_g_ot + float(recipe.t1_iso_weight) * loss_g_iso
    loss_g, pm, tr, tr_l = _finalize_gen_step(
        trainer,
        base_prior,
        total_g_loss,
        loss_gan,
        fake_g,
        real_g,
        fake_logits,
        latent,
        indices,
        loss_d,
        penalty,
        g_scale_t,
        wakeup_t,
        1.0,
        res_ema_t=res_ema_t,
    )

    prior_diag = loss_gan.new_zeros(())
    if can_condense and hasattr(base_prior, "z") and base_prior.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0:
        s_idx, _ = torch.sort(indices)
        is_f = torch.ones(len(s_idx), device=dev, dtype=torch.bool)
        if len(s_idx) > 1:
            is_f[1:] = s_idx[1:] != s_idx[:-1]
        w_f = is_f.float()
        n_u_t = w_f.sum().clamp_min(1.0)
        g_flags = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)
            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[s_idx]
            x_u = gen_clean(c_u).reshape(len(s_idx), -1)
            with torch.no_grad():
                xu_det = x_u.detach().float()
                k_star, g_psi = reg.update_laguerre_dual(
                    xu_det, c_bar_all, w_mat_all, r_macro_safe, float(recipe.t1_laguerre_lr), row_mask=is_f
                )
                sc_u = sigma_bar_all[k_star].to(x_u.dtype)
                bm_u = _box_muller_targets(s_idx, k_star, comp_rep, k_proto, c_bar_all, v_mu_star_all, multicell_coal, dim).to(x_u.dtype)
                c_target = torch.where(
                    (multicell_coal[k_star] & (v_mu_star_all[k_star] > 0.0))[:, None],
                    bm_u,
                    c_bar_all[k_star].to(x_u.dtype),
                )
                row_settle = torch.maximum(row_floor, torch.minimum(res_ema_t.new_ones(()), torch.maximum(res_ema_t / 0.10, g_psi)))
            sq_u = (x_u - c_target).square().sum(-1)
            mu_raw_u = base_prior.z[s_idx]
            mu_mean = (w_f[:, None] * mu_raw_u).sum(0, keepdim=True) / n_u_t
            z_std = ((w_f[:, None] * (mu_raw_u - mu_mean).square()).sum(0) / n_u_t + 1e-6).sqrt()
            spread_pen = (
                0.15 * F.relu(iso_ema_t.to(x_u.dtype) - z_std).square().mean() * (n_u_t >= 2.0).to(x_u.dtype)
                if has_sigma
                else 0.0
            )
            loss_mu = (w_f * (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u)).sum() / (n_u_t * s_ot_dt) + spread_pen
            prior_diag = loss_mu.detach()
            if base_prior.z.grad is not None:
                base_prior.z.grad.zero_()
            loss_mu.backward()
            if not reg.rows_initialized or reg.row_m is None or reg.row_m.shape != base_prior.z.shape:
                reg.row_m = torch.zeros(base_prior.z.shape, device=base_prior.z.device, dtype=torch.float32)
                reg.row_v = torch.zeros((len(base_prior.z), 1), device=base_prior.z.device, dtype=torch.float32)
                reg.row_n = torch.zeros((len(base_prior.z), 1), device=base_prior.z.device, dtype=torch.float32)
                reg.rows_initialized = True
            elif reg.row_m.device != base_prior.z.device or reg.row_m.dtype != torch.float32:
                reg.row_m = reg.row_m.to(device=base_prior.z.device, dtype=torch.float32)
                reg.row_v = reg.row_v.to(device=base_prior.z.device, dtype=torch.float32)
                reg.row_n = reg.row_n.to(device=base_prior.z.device, dtype=torch.float32)
            with torch.no_grad():
                g_rows = (base_prior.z.grad[s_idx] * n_u_t.to(base_prior.z.dtype)).clone()
                dist_u = sq_u.detach().float().sqrt()
                base_prior.z.grad.zero_()
                part_scale = (dist_u.to(sq_u.dtype)[:, None] / (1.5 * sc_u[:, None]).clamp_min(1e-12)).clamp(0.02, 1.2)
                far_m = is_f & (dist_u > (2.8 * math.sqrt(float(dim)) * sc_u.float()))
                d_z_u = int(c_u.shape[1])
                if d_z_u <= 8 and bool(far_m.any()):
                    n_s = len(s_idx)
                    k_f = min(64, n_s)
                    fi = torch.topk(dist_u.masked_fill(~far_m, -1.0), k=k_f, sorted=False).indices
                    fi_valid = far_m[fi]
                    ref_i = torch.topk((is_f & (~far_m)).to(torch.float32), k=k_f, sorted=False).indices
                    pi_idx = torch.cat((fi, ref_i), dim=0)
                    cp = c_u[pi_idx].detach()
                    xp = xu_det[pi_idx]
                    step_sign = torch.where(cp >= 0.0, 1.0, -1.0)
                    c_pert = (cp[:, None] + (1e-3 * step_sign)[:, :, None] * torch.eye(d_z_u, device=dev, dtype=c_u.dtype)[None]).reshape(-1, d_z_u)
                    csen = (gen_clean(c_pert).float().reshape(len(pi_idx), d_z_u, dim) - xp[:, None]).norm(dim=-1) / 1e-3
                    med_sen = torch.quantile(csen[k_f:], 0.5, dim=0, keepdim=True).clamp_min(1e-12)
                    dead_f = (csen[:k_f] < (0.04 * med_sen)) & fi_valid[:, None]
                    pull_dir = torch.sign(mu_raw_u[fi].detach() - c_u[ref_i].detach().mean(0, keepdim=True)).float()
                    pull_mag = torch.maximum(g_rows[fi].float().norm(dim=-1, keepdim=True), torch.ones((k_f, 1), device=dev))
                    g_rows[fi] = torch.where(dead_f, (pull_dir * pull_mag).to(g_rows.dtype), g_rows[fi])
                    part_scale[fi] = torch.where(dead_f.any(1, keepdim=True), torch.ones_like(part_scale[fi]), part_scale[fi])

                # Bounded, clock-free FP32 EMA weight w_r in [0, 1]: w_r = 0.80 * w_p + 0.20 (no monotone step counter).
                g_rows_f = g_rows.float()
                m_p = reg.row_m[s_idx]
                v_p = reg.row_v[s_idx]
                w_p = reg.row_n[s_idx]
                m_r = m_p * 0.80 + 0.20 * g_rows_f
                v_r = v_p * 0.80 + 0.20 * g_rows_f.square().sum(-1, keepdim=True)
                w_r = (w_p * 0.80 + 0.20).clamp_max(1.0)
                w_col = w_f.to(torch.float32)[:, None]
                reg.row_m.index_add_(0, s_idx, (m_r - m_p) * w_col)
                reg.row_v.index_add_(0, s_idx, (v_r - v_p) * w_col)
                reg.row_n.index_add_(0, s_idx, (w_r - w_p) * w_col)
                denom_w = w_r.clamp_min(1e-12)
                v_hat_s = (v_r / denom_w).sqrt()
                g_ref = (v_hat_s * w_col).sum().div(n_u_t.to(torch.float32)).clamp_min(1e-30)
                row_step = ((m_r / denom_w) / (v_hat_s + (1e-5 * g_ref).clamp_min(1e-30))) * part_scale.float() * w_col
                delta = row_step * (-float(recipe.t1_prior_row_lr) * row_settle.float())
                base_prior.z.index_add_(0, s_idx, delta.to(base_prior.z.dtype))
        finally:
            for p, fl in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(fl)
            if rec_d is not None:
                rec_d._g_probe_cache = None
    return loss_g, loss_gan, prior_diag, pm, tr, tr_l


__all__ = [
    "AxiomPriorRegularizer",
    "allocate_deficit_quotas",
    "axiom_pullback_step",
    "central_pullback_cov",
    "coalesce_pythagorean_ltv",
    "has_positive_prior_sigma",
    "should_use_axiom_pullback",
    "validate_axiom_prior_state",
]
