"""Axiom L2 Law 1 Prior & Pullback: Curvature-Compensated Whitened Burg Pullback & Clock-Free EMAs."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .atlas import (
    CoalescedAtlas,
    MAX_FULL_COV_DIM,
    PrototypeAtlas,
    SCHEMA_VERSION,
    coalesce_pythagorean_ltv,
    cov_from_scalar,
    cov_trace_mean,
    validate_atlas_state,
)
from .math_ops import (
    allocate_deficit_quotas,
    born_infeld_cost,
    log_sinkhorn,
    safe_cdist,
    smoothstep01,
    weyl_rademacher,
)
from .state import get_module_record, set_active_prior_rows

_VAR_FLOOR: float = 1e-24
_BURG_DELTA: float = 1e-3
_ROW_BETA: float = 0.80
_CURV_LO: float = 0.80
_CURV_HI: float = 1.35

_PRIOR_STATE_KEYS = frozenset(
    {
        "schema",
        "max_prototypes",
        "atlas",
        "v_real_ema",
        "v_tang_ema",
        "res_ema",
        "iso_ema",
        "iso_err_ema",
        "rows_initialized",
        "row_m",
        "row_v",
        "row_init",
    }
)


def _prime_irrationals(n: int) -> Tuple[float, ...]:
    out = []
    cand = 2
    while len(out) < n:
        if all(cand % p for p in range(2, int(cand**0.5) + 1)):
            out.append(math.sqrt(cand) % 1.0)
        cand += 1
    return tuple(out)


_PRIMES: Tuple[float, ...] = _prime_irrationals(128)


# ---------------------------------------------------------------------------
# Prior Kernel Inspection & Pullback Geometry Primitives
# ---------------------------------------------------------------------------


def _prior_kernel_spec(prior: nn.Module) -> Tuple[Any, float, bool]:
    """Return ``(base_prior, sigma_z, can_condense)`` without class-name string sniffing."""
    base = unwrap_prior(prior)
    if getattr(base, "code_law", "table") != "table":
        return base, 0.0, False
    if not bool(getattr(prior, "transforms_codes", False)):
        if isinstance(base, MoGParticlePrior):
            sig = (
                base._cached_sigma_float()
                if bool(getattr(base, "_noise_enabled", False))
                else 0.0
            )
            return base, sig, True
        return base, 0.0, hasattr(base, "z")
    ms = tuple(getattr(prior, "manipulators", ()))
    if (
        len(ms) == 1
        and bool(getattr(ms[0], "normal_noise", False))
        and float(getattr(ms[0], "kappa_radial", 1.0)) == 1.0
    ):
        return base, float(ms[0].sigma), True
    return base, 0.0, False


def has_positive_prior_sigma(prior: nn.Module) -> bool:
    """True when ``prior`` applies an isotropic additive Gaussian kernel with ``sigma > 0``."""
    return _prior_kernel_spec(prior)[1] > 0.0


@torch.no_grad()
def _cholesky_spd_safe(cov: torch.Tensor) -> torch.Tensor:
    """Zero-sync lower Cholesky factor of symmetric SPD ``[..., D, D]`` with isotropic fallback."""
    dim, dev = int(cov.shape[-1]), cov.device
    sym = 0.5 * (cov.float() + cov.float().transpose(-1, -2))
    v_tr = (
        sym.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(_VAR_FLOOR)
    )
    eye = torch.eye(dim, device=dev, dtype=torch.float32)
    jitter = (1e-6 * v_tr).unsqueeze(-1) * eye
    chol, info = torch.linalg.cholesky_ex(sym + jitter, check_errors=False)
    iso_chol = v_tr.sqrt().unsqueeze(-1) * eye
    return torch.where((info != 0).unsqueeze(-1).unsqueeze(-1), iso_chol, chol)


@torch.no_grad()
def _anisotropic_box_muller_targets(
    rows: torch.Tensor,
    comp_ids: torch.Tensor,
    comp_c: torch.Tensor,
    comp_chol: torch.Tensor,
    spread_scale: torch.Tensor,
) -> torch.Tensor:
    """Zero-mean quasi-random Box-Muller intra-component spread aligned to ``comp_chol``."""
    n_rows, dim = int(rows.shape[0]), int(comp_c.shape[1])
    k_c = int(comp_c.shape[0])
    n_p = (dim + 1) // 2
    r_f = (rows.to(torch.float64) + 0.5).unsqueeze(1)
    pr_list = (
        _PRIMES[: 2 * n_p] if 2 * n_p <= len(_PRIMES) else _prime_irrationals(2 * n_p)
    )
    pr = torch.tensor(pr_list, device=rows.device, dtype=torch.float64)
    u1 = torch.remainder(r_f * pr[0::2].unsqueeze(0), 1.0).clamp(1e-4, 1.0 - 1e-4)
    u2 = torch.remainder(r_f * pr[1::2].unsqueeze(0), 1.0)
    rad = torch.sqrt(-2.0 * torch.log(u1)).clamp_max(2.95)
    ang = (2.0 * math.pi) * u2
    xi = torch.stack((rad * torch.cos(ang), rad * torch.sin(ang)), dim=-1)
    xi = xi.reshape(n_rows, 2 * n_p)[:, :dim].to(
        device=comp_c.device, dtype=torch.float32
    )

    cnt = torch.bincount(comp_ids, minlength=k_c).float()
    xi_mean = xi.new_zeros((k_c, dim)).index_add_(0, comp_ids, xi) / cnt.clamp_min(
        1.0
    ).unsqueeze(1)
    xi_c = torch.where((cnt[comp_ids] >= 2.0).unsqueeze(1), xi - xi_mean[comp_ids], xi)
    offset = torch.bmm(comp_chol[comp_ids], xi_c.unsqueeze(-1)).squeeze(-1)
    return comp_c[comp_ids] + spread_scale[comp_ids].unsqueeze(1) * offset


def _eval_central_offsets(
    gen_fn: Callable[[torch.Tensor], torch.Tensor],
    mu: torch.Tensor,
    dirs: torch.Tensor,
    sigma_z: float,
) -> torch.Tensor:
    """Evaluate central finite offsets ``0.5 * (G(mu + sigma d) - G(mu - sigma d))`` -> ``[m, n_d, D]``."""
    m, n_dirs, d_z = int(dirs.shape[0]), int(dirs.shape[1]), int(dirs.shape[2])
    off = float(sigma_z) * dirs
    z_cat = torch.cat(
        (
            (mu.unsqueeze(1) + off).reshape(m * n_dirs, d_z),
            (mu.unsqueeze(1) - off).reshape(m * n_dirs, d_z),
        ),
        dim=0,
    )
    x_cat = gen_fn(z_cat).reshape(2, m, n_dirs, -1)
    if x_cat.dtype in (torch.float16, torch.bfloat16):
        x_cat = x_cat.float()
    return 0.5 * (x_cat[0] - x_cat[1])


def central_pullback_cov(
    gen_fn: Callable[[torch.Tensor], torch.Tensor],
    mu: torch.Tensor,
    sigma_z: float,
    *,
    phase: float = 0.0,
    salt: int = 0,
    return_pair: bool = False,
) -> Any:
    """Central finite-offset pushforward covariance at ``sigma_z`` (+ unbiased 2-sketch pair)."""
    m, d_z = int(mu.shape[0]), int(mu.shape[1])
    if d_z <= 8:
        dirs = (
            torch.eye(d_z, device=mu.device, dtype=mu.dtype)
            .unsqueeze(0)
            .expand(m, d_z, d_z)
        )
        jc = _eval_central_offsets(gen_fn, mu, dirs, sigma_z)
        sp = torch.einsum("mad,mae->mde", jc, jc)
        return (sp, sp) if return_pair else sp

    inv_sq = d_z**-0.5
    if return_pair:
        d1 = weyl_rademacher(
            (m, 8, d_z), phase=phase, salt=1, device=mu.device, dtype=mu.dtype
        )
        d2 = weyl_rademacher(
            (m, 8, d_z), phase=phase, salt=2, device=mu.device, dtype=mu.dtype
        )
        jc = _eval_central_offsets(
            gen_fn, mu, torch.cat((d1, d2), dim=1) * inv_sq, sigma_z
        )
        scale = float(d_z) / 8.0
        return (
            torch.einsum("mad,mae->mde", jc[:, :8], jc[:, :8]) * scale,
            torch.einsum("mad,mae->mde", jc[:, 8:], jc[:, 8:]) * scale,
        )
    d0 = weyl_rademacher(
        (m, 8, d_z), phase=phase, salt=salt, device=mu.device, dtype=mu.dtype
    )
    jc = _eval_central_offsets(gen_fn, mu, d0 * inv_sq, sigma_z)
    return torch.einsum("mad,mae->mde", jc, jc) * (float(d_z) / 8.0)


def _burg_divergence_exact(
    gen_fn: Callable[[torch.Tensor], torch.Tensor],
    mu_sub: torch.Tensor,
    sigma_z: float,
    chol_tgt: torch.Tensor,
    inv_w: torch.Tensor,
    eps_sub: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact canonical-basis whitened Burg (Log-Det) divergence for ``d_z <= max(D, 16)``."""
    m, d_z, dim = int(mu_sub.shape[0]), int(mu_sub.shape[1]), int(chol_tgt.shape[-1])
    dev = mu_sub.device
    dirs = (
        torch.eye(d_z, device=dev, dtype=mu_sub.dtype).unsqueeze(0).expand(m, d_z, d_z)
    )
    jc = _eval_central_offsets(gen_fn, mu_sub, dirs, sigma_z).float()  # [m, d_z, D]
    m_mat = jc.transpose(1, 2)  # [m, D, d_z]
    m_tilde = torch.linalg.solve_triangular(chol_tgt, m_mat, upper=False)

    if dim <= d_z:
        r_dim = dim
        s_mat = torch.bmm(m_tilde, m_tilde.transpose(1, 2))
    else:
        r_dim = d_z
        s_mat = (float(d_z) / float(dim)) * torch.bmm(m_tilde.transpose(1, 2), m_tilde)

    eye_r = torch.eye(r_dim, device=dev, dtype=torch.float32).unsqueeze(0)
    s_sym = 0.5 * (s_mat + s_mat.transpose(1, 2))
    l_chol = torch.linalg.cholesky(s_sym + _BURG_DELTA * eye_r)
    tr_s = s_sym.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    logdet = 2.0 * l_chol.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(dim=-1)
    burg = (tr_s - logdet - float(r_dim) * (1.0 - math.log(1.0 + _BURG_DELTA))) / float(
        r_dim
    )
    loss_iso = 4.5 * (inv_w.float() * burg.clamp_min(0.0)).mean()

    with torch.no_grad():
        if eps_sub is not None:
            tang_var = torch.bmm(m_mat.detach(), eps_sub.float().unsqueeze(-1)).squeeze(
                -1
            ).square().sum(dim=-1) / float(dim)
        else:
            tang_var = m_mat.detach().square().sum(dim=(-2, -1)) / float(dim)
        iso_frob = (s_sym.detach() - eye_r).square().sum(
            dim=(-2, -1)
        ).sqrt() / math.sqrt(float(r_dim))
    return loss_iso, tang_var, iso_frob


def _burg_2sketch_ustat(
    gen_fn: Callable[[torch.Tensor], torch.Tensor],
    mu_sub: torch.Tensor,
    sigma_z: float,
    chol_tgt: torch.Tensor,
    inv_w: torch.Tensor,
    eps_sub: Optional[torch.Tensor],
    phase: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unbiased 2-independent-sketch whitened U-statistic for wide latent ``d_z > max(D, 16)``."""
    m, d_z, dim = int(mu_sub.shape[0]), int(mu_sub.shape[1]), int(chol_tgt.shape[-1])
    dev = mu_sub.device
    inv_sq = d_z**-0.5
    d1 = weyl_rademacher(
        (m, 8, d_z), phase=phase, salt=1, device=dev, dtype=mu_sub.dtype
    )
    d2 = weyl_rademacher(
        (m, 8, d_z), phase=phase, salt=2, device=dev, dtype=mu_sub.dtype
    )
    jc = _eval_central_offsets(
        gen_fn, mu_sub, torch.cat((d1, d2), dim=1) * inv_sq, sigma_z
    ).float()
    g_raw = jc.transpose(1, 2)  # [m, D, 16]
    g_wh = torch.linalg.solve_triangular(chol_tgt, g_raw, upper=False)
    scale = float(d_z) / 8.0
    s1 = torch.bmm(g_wh[:, :, :8], g_wh[:, :, :8].transpose(1, 2)) * scale
    s2 = torch.bmm(g_wh[:, :, 8:], g_wh[:, :, 8:].transpose(1, 2)) * scale
    eye_d = torch.eye(dim, device=dev, dtype=torch.float32).unsqueeze(0)
    u_stat = ((s1 - eye_d) * (s2 - eye_d)).sum(dim=(-2, -1)) / float(dim)
    loss_iso = 2.25 * (inv_w.float() * u_stat).mean()

    with torch.no_grad():
        tr_raw = (
            g_raw.detach().square().sum(dim=(-2, -1)) * (float(d_z) / 16.0) / float(dim)
        )
        rad_fac = (
            eps_sub.float().square().sum(dim=-1) / float(d_z)
            if eps_sub is not None
            else tr_raw.new_ones(m)
        )
        tang_var = tr_raw * rad_fac
        s_avg = 0.5 * (s1.detach() + s2.detach())
        iso_frob = (s_avg - eye_d).square().sum(dim=(-2, -1)).sqrt() / math.sqrt(
            float(dim)
        )
    return loss_iso, tang_var, iso_frob


def whitened_burg_pullback_loss(
    gen_fn: Callable[[torch.Tensor], torch.Tensor],
    mu_det: torch.Tensor,
    sigma_z: float,
    target_cov: torch.Tensor,
    inv_w: torch.Tensor,
    *,
    eps_g: Optional[torch.Tensor] = None,
    phase: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GL(D)-invariant Whitened Burg Log-Det pullback (exact basis ``d_z<=max(D,16)`` else 2-sketch)."""
    b, d_z, dim = int(mu_det.shape[0]), int(mu_det.shape[1]), int(target_cov.shape[-1])
    use_exact = d_z <= max(dim, 16)
    m_iso = min(b, max(32, 2048 // max(1, d_z))) if use_exact else min(b, 256)
    chol_tgt = _cholesky_spd_safe(target_cov[:m_iso])
    eps_sub = None if eps_g is None else eps_g[:m_iso]
    if use_exact:
        return _burg_divergence_exact(
            gen_fn, mu_det[:m_iso], sigma_z, chol_tgt, inv_w[:m_iso], eps_sub
        )
    return _burg_2sketch_ustat(
        gen_fn, mu_det[:m_iso], sigma_z, chol_tgt, inv_w[:m_iso], eps_sub, phase
    )


# ---------------------------------------------------------------------------
# Strict Schema-2 Checkpoint Validation
# ---------------------------------------------------------------------------


def _check_nonneg_scalar(val: Any, name: str) -> float:
    if (
        type(val) not in (int, float)
        or not math.isfinite(float(val))
        or float(val) < 0.0
    ):
        raise ValueError(f"invalid AxiomPriorRegularizer.{name}: {val!r}")
    return float(val)


def _check_prior_tensor(
    t: Any, name: str, *, shape: Optional[Tuple[int, ...]] = None, nonneg: bool = False
) -> torch.Tensor:
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        raise ValueError(f"invalid AxiomPriorRegularizer tensor {name!r}")
    if shape is not None and tuple(t.shape) != shape:
        raise ValueError(f"{name!r} shape {tuple(t.shape)} != {shape}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"non-finite values in {name!r}")
    if nonneg and bool((t < 0.0).any()):
        raise ValueError(f"negative values in {name!r}")
    return t


def _validate_row_buffers(state: Dict[str, Any]) -> None:
    if not state["rows_initialized"]:
        if any(state[k] is not None for k in ("row_m", "row_v", "row_init")):
            raise ValueError("uninitialized row buffers must be None")
        return
    rm = _check_prior_tensor(state["row_m"], "row_m")
    if rm.ndim != 2 or int(rm.shape[0]) < 1 or int(rm.shape[1]) < 1:
        raise ValueError("row_m must be non-empty 2D float tensor")
    p_cnt = int(rm.shape[0])
    _check_prior_tensor(state["row_v"], "row_v", shape=(p_cnt, 1), nonneg=True)
    ri = state["row_init"]
    if (
        not isinstance(ri, torch.Tensor)
        or ri.dtype != torch.bool
        or tuple(ri.shape) != (p_cnt, 1)
    ):
        raise ValueError("row_init must be bool tensor matching (P, 1)")


def validate_axiom_prior_state(state: Any) -> Dict[str, Any]:
    """Fail-closed schema-2 checkpoint validator for :class:`AxiomPriorRegularizer`."""
    if not isinstance(state, dict) or set(state.keys()) != _PRIOR_STATE_KEYS:
        raise ValueError("invalid AxiomPriorRegularizer checkpoint keys")
    if type(state["schema"]) is not int or state["schema"] != SCHEMA_VERSION:
        raise ValueError(f"AxiomPriorRegularizer requires schema={SCHEMA_VERSION}")
    max_p, rows_init = state["max_prototypes"], state["rows_initialized"]
    if type(max_p) is not int or max_p < 2 or type(rows_init) is not bool:
        raise ValueError("invalid AxiomPriorRegularizer header fields")
    for key in ("res_ema", "iso_ema", "iso_err_ema"):
        _check_nonneg_scalar(state[key], key)
    atlas_st = validate_atlas_state(state["atlas"])
    if int(atlas_st["max_prototypes"]) != max_p:
        raise ValueError("atlas max_prototypes mismatch")
    if atlas_st["initialized"]:
        k = int(atlas_st["proto_c"].shape[0])
        _check_prior_tensor(state["v_real_ema"], "v_real_ema", shape=(k,), nonneg=True)
        _check_prior_tensor(state["v_tang_ema"], "v_tang_ema", shape=(k,), nonneg=True)
    elif state["v_real_ema"] is not None or state["v_tang_ema"] is not None:
        raise ValueError("uninitialized curvature EMAs must be None")
    _validate_row_buffers(state)
    return state


# ---------------------------------------------------------------------------
# AxiomPriorRegularizer (Schema = 2)
# ---------------------------------------------------------------------------


def _scalar_ema_prop(name: str) -> property:
    def _get(self: "AxiomPriorRegularizer") -> float:
        if name in self._ema_t:
            self._flush_emas()
        return self._ema[name]

    def _set(self: "AxiomPriorRegularizer", val: float) -> None:
        self._ema[name] = float(val)
        self._ema_t.pop(name, None)

    return property(_get, _set)


class AxiomPriorRegularizer(nn.Module):
    """Law-1 IMQ-repulsion regularizer owning anisotropic LTC PrototypeAtlas & clock-free row EMAs."""

    SCHEMA = SCHEMA_VERSION
    res_ema = _scalar_ema_prop("res_ema")
    iso_ema = _scalar_ema_prop("iso_ema")
    iso_err_ema = _scalar_ema_prop("iso_err_ema")

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
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be int >= 2")
        for val, pos in ((weight, False), (target_std, True), (eps, True)):
            if type(val) not in (int, float) or not math.isfinite(float(val)):
                raise ValueError("invalid AxiomPriorRegularizer hyperparameters")
            if float(val) <= 0.0 if pos else float(val) < 0.0:
                raise ValueError("invalid AxiomPriorRegularizer hyperparameter bounds")
        self.weight: float = float(weight)
        self.target_std: float = float(target_std)
        self.eps: float = float(eps)
        self.max_prototypes: int = int(max_prototypes)
        self._bound_device: Optional[torch.device] = bound_device
        self._bound_shape: Optional[Tuple[int, int]] = bound_shape
        self.record: Any = None
        self.axiom_record: Any = None
        self.thermal_source: Any = None
        self._last_raw_var_cache: Optional[Tuple[torch.Tensor, bool]] = None
        self._iso_var_override: Optional[torch.Tensor] = None

        self.atlas: PrototypeAtlas = PrototypeAtlas(
            max_prototypes=max_prototypes, device=bound_device
        )
        self.v_real_ema: Optional[torch.Tensor] = None
        self.v_tang_ema: Optional[torch.Tensor] = None
        self.rows_initialized: bool = False
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_init: Optional[torch.Tensor] = None
        self._ema: Dict[str, float] = {
            "res_ema": 1.0,
            "iso_ema": 1.0,
            "iso_err_ema": 0.0,
        }
        self._ema_t: Dict[str, torch.Tensor] = {}

    # Compatibility views delegating directly to self.atlas (incl. Suite 5B isotropic fixture shim).
    @property
    def quantizer_initialized(self) -> bool:
        return self.atlas.initialized

    @quantizer_initialized.setter
    def quantizer_initialized(self, val: bool) -> None:
        self.atlas.initialized = bool(val)

    @property
    def proto_c(self) -> Optional[torch.Tensor]:
        return self.atlas.proto_c

    @proto_c.setter
    def proto_c(self, val: Optional[torch.Tensor]) -> None:
        self.atlas.proto_c = val
        if val is not None and val.ndim == 2:
            self.atlas.dim = int(val.shape[1])

    @property
    def proto_cov(self) -> Optional[torch.Tensor]:
        return self.atlas.proto_cov

    @proto_cov.setter
    def proto_cov(self, val: Optional[torch.Tensor]) -> None:
        self._iso_var_override = None
        self.atlas.proto_cov = val

    @property
    def proto_var(self) -> Optional[torch.Tensor]:
        if self._iso_var_override is not None:
            return self._iso_var_override
        return self.atlas.proto_var

    @proto_var.setter
    def proto_var(self, val: Optional[torch.Tensor]) -> None:
        self._iso_var_override = val
        if val is not None and self.atlas.dim >= 1:
            self.atlas.proto_cov = cov_from_scalar(val, self.atlas.dim)

    @property
    def proto_w(self) -> Optional[torch.Tensor]:
        return self.atlas.proto_w

    @proto_w.setter
    def proto_w(self, val: Optional[torch.Tensor]) -> None:
        self.atlas.proto_w = val

    @property
    def w_fake_ema(self) -> Optional[torch.Tensor]:
        return self.atlas.w_fake_ema

    @w_fake_ema.setter
    def w_fake_ema(self, val: Optional[torch.Tensor]) -> None:
        self.atlas.w_fake_ema = val

    @property
    def proto_n(self) -> Optional[torch.Tensor]:
        return self.atlas.proto_n

    @proto_n.setter
    def proto_n(self, val: Optional[torch.Tensor]) -> None:
        self.atlas.proto_n = val

    @property
    def psi(self) -> Optional[torch.Tensor]:
        return self.atlas.psi

    @psi.setter
    def psi(self, val: Optional[torch.Tensor]) -> None:
        self.atlas.psi = val

    @property
    def v_push(self) -> Optional[torch.Tensor]:
        return self.atlas.v_push

    @v_push.setter
    def v_push(self, val: Optional[torch.Tensor]) -> None:
        self.atlas.v_push = val

    def _flush_emas(self) -> None:
        if not self._ema_t:
            return
        ks, ts = list(self._ema_t.keys()), list(self._ema_t.values())
        same_dev = all(t.device == ts[0].device for t in ts)
        vals = (
            torch.stack([t.double() for t in ts]).tolist()
            if same_dev
            else [float(t.item()) for t in ts]
        )
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

    def _sync_iso_and_curvature(self, dev: Optional[torch.device] = None) -> None:
        if self._iso_var_override is not None and self.atlas.proto_c is not None:
            dim = int(self.atlas.proto_c.shape[1])
            self.atlas.dim = dim
            self.atlas.proto_cov = cov_from_scalar(self._iso_var_override, dim)
            self._iso_var_override = None
        if not self.atlas.initialized or self.atlas.proto_c is None:
            self.v_real_ema = self.v_tang_ema = None
            return
        k = int(self.atlas.proto_c.shape[0])
        tgt_dev = dev if dev is not None else self.atlas.proto_c.device
        if self.atlas.proto_n is None or int(self.atlas.proto_n.numel()) != k:
            self.atlas.proto_n = torch.full(
                (k,), 25.0, device=tgt_dev, dtype=torch.float32
            )
        if self.atlas.v_push is None or int(self.atlas.v_push.numel()) != k:
            self.atlas.v_push = cov_trace_mean(self.atlas.proto_cov).clone()  # type: ignore[arg-type]
        ref_v = self.atlas.v_push.to(tgt_dev)
        if self.v_real_ema is None or int(self.v_real_ema.numel()) != k:
            if self.v_real_ema is not None and int(self.v_real_ema.numel()) < k:
                old_k = int(self.v_real_ema.numel())
                self.v_real_ema = torch.cat(
                    (self.v_real_ema.to(tgt_dev), ref_v[old_k:]), dim=0
                )
                self.v_tang_ema = torch.cat(
                    (self.v_tang_ema.to(tgt_dev), ref_v[old_k:]), dim=0
                )  # type: ignore[union-attr]
            else:
                self.v_real_ema = ref_v.clone()
                self.v_tang_ema = ref_v.clone()
        elif self.v_real_ema.device != tgt_dev:
            self.v_real_ema = self.v_real_ema.to(tgt_dev)
            self.v_tang_ema = self.v_tang_ema.to(tgt_dev)  # type: ignore[union-attr]

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """Scale-invariant IMQ-repulsion + coordinate std tether on prior rows ``(N, D)``."""
        if rows.ndim != 2 or int(rows.shape[1]) < 1:
            raise ValueError("rows must have shape (N, D) with D >= 1")
        n = int(rows.shape[0])
        if self.weight <= 0.0 or n < 2:
            return rows[:0].sum()
        m = min(n, 256)
        z = rows[(torch.arange(m, device=rows.device) * n) // m] if n > 256 else rows
        with torch.no_grad():
            zf = z.detach().float()
            sq = safe_cdist(zf).square()
            iu, ju = torch.triu_indices(m, m, offset=1, device=z.device)
            r_med_sq = torch.median(sq[iu, ju]).clamp_min(1e-12).to(z.dtype)
            knn_idx = torch.topk(
                sq.fill_diagonal_(float("inf")), k=min(8, m - 1), dim=1, largest=False
            ).indices
        imq = (
            r_med_sq / (r_med_sq + (z.unsqueeze(1) - z[knn_idx]).square().sum(dim=-1))
        ).mean()
        std_term = (
            (
                (
                    (z - z.mean(dim=0, keepdim=True)).square().mean(dim=0) + self.eps
                ).sqrt()
                - self.target_std
            )
            .square()
            .mean()
        )
        loss = self.weight * (imq + 0.15 * std_term)
        rec = (
            self.axiom_record
            or self.record
            or self.thermal_source
            or get_module_record(self)
        )
        s = float(getattr(rec, "thermal_scale", 1.0)) if rec is not None else 1.0
        return (
            (loss * s + loss.detach() * (1.0 - s))
            if (s < 1.0 and loss.requires_grad)
            else loss
        )

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        """Stream real batch ``yr`` into the underlying anisotropic LTC PrototypeAtlas."""
        if self._bound_device is None:
            self._bound_device = yr.device
        self._sync_iso_and_curvature(yr.device)
        self.atlas.update(yr)
        self._sync_iso_and_curvature(yr.device)

    @torch.no_grad()
    def update_curvature_ema(
        self,
        proto_ids: torch.Tensor,
        real_var_i: torch.Tensor,
        tang_var_i: torch.Tensor,
    ) -> None:
        """Track realized vs tangent intra-mode covariance trace EMA (zero extra forwards)."""
        self._sync_iso_and_curvature(proto_ids.device)
        if (
            self.v_real_ema is None
            or self.v_tang_ema is None
            or self.atlas.v_push is None
        ):
            return
        k = int(self.v_real_ema.shape[0])
        cnt = torch.bincount(proto_ids, minlength=k).float()
        hit = cnt > 0.0
        r_mean = self.v_real_ema.new_zeros(k).index_add_(
            0, proto_ids, real_var_i.float()
        ) / cnt.clamp_min(1.0)
        t_mean = self.v_tang_ema.new_zeros(k).index_add_(
            0, proto_ids, tang_var_i.float()
        ) / cnt.clamp_min(1.0)
        self.v_real_ema = torch.where(
            hit,
            0.90 * self.v_real_ema + 0.10 * r_mean.clamp_min(_VAR_FLOOR),
            self.v_real_ema,
        )
        self.v_tang_ema = torch.where(
            hit,
            0.90 * self.v_tang_ema + 0.10 * t_mean.clamp_min(_VAR_FLOOR),
            self.v_tang_ema,
        )
        self.atlas.v_push = torch.where(
            hit,
            0.90 * self.atlas.v_push + 0.10 * t_mean.clamp_min(_VAR_FLOOR),
            self.atlas.v_push,
        )

    @torch.no_grad()
    def curvature_gamma_per_component(self, coal: CoalescedAtlas) -> torch.Tensor:
        """Per-component curvature compensation ``gamma_C = clamp(tr(Sigma_tang)/tr(Sigma_real), 0.80, 1.35)``."""
        k_c = int(coal.comp_c.shape[0])
        if self.v_real_ema is None or self.v_tang_ema is None:
            return coal.comp_c.new_ones(k_c)
        w_p = (
            self.atlas.proto_w
            if self.atlas.proto_w is not None
            else coal.comp_c.new_ones(len(self.v_real_ema))
        ).float()
        cid = coal.proto_comp_id
        w_sum = coal.comp_c.new_zeros(k_c).index_add_(0, cid, w_p).clamp_min(1e-12)
        vr_c = (
            coal.comp_c.new_zeros(k_c).index_add_(0, cid, w_p * self.v_real_ema) / w_sum
        )
        vt_c = (
            coal.comp_c.new_zeros(k_c).index_add_(0, cid, w_p * self.v_tang_ema) / w_sum
        )
        return (vt_c / vr_c.clamp_min(_VAR_FLOOR)).clamp(_CURV_LO, _CURV_HI)

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
        """Scale-equivariant Laguerre dual ascent with exact zero-mean ``psi`` gauge (connected fallback)."""
        self._sync_iso_and_curvature(xu_det.device)
        k, dim = int(c_bar.shape[0]), int(c_bar.shape[1])
        pw, wf, psi = self.atlas.proto_w, self.atlas.w_fake_ema, self.atlas.psi
        assert pw is not None and wf is not None and psi is not None
        n_eff_t = (
            row_mask.float().sum().clamp_min(1.0)
            if row_mask is not None
            else xu_det.new_full((), float(len(xu_det)))
        )
        k_star = (
            safe_cdist(xu_det, c_bar).square() - (w_mat @ psi).unsqueeze(0)
        ).argmin(dim=1)
        w_hard = (
            (w_mat[k_star] * row_mask.float().unsqueeze(1)).sum(dim=0) / n_eff_t
            if row_mask is not None
            else w_mat[k_star].mean(dim=0)
        )
        w_fe = 0.88 * wf + 0.12 * w_hard
        self.atlas.w_fake_ema = w_fe / w_fe.sum().clamp_min(1e-12)
        pw_c, wf_c = w_mat @ pw, w_mat @ self.atlas.w_fake_ema
        tv_now = 0.5 * (pw - self.atlas.w_fake_ema).abs().sum()
        tv_floor = 0.35 * torch.sqrt(float(k) / (n_eff_t * 8.0).clamp_min(1.0))
        d_max = (
            (pw_c - wf_c).clamp_min(0.0)
            / torch.maximum(pw_c, pw_c.new_full((), 1.0 / float(k)))
        ).amax()
        g_psi = torch.maximum(
            smoothstep01((tv_now - tv_floor) / tv_floor.clamp_min(0.08)),
            smoothstep01((d_max - 0.15) / 0.20),
        )
        eta = (
            g_psi
            * min(3.0, max(0.35, float(laguerre_lr) * float(k)))
            * float(dim)
            * r_macro_safe.square()
        )
        psi_next = w_mat @ (
            psi * (1.0 - 0.004 * g_psi * (1.0 - (pw_c > wf_c).float()))
            + eta * (pw - (0.35 * w_hard + 0.65 * self.atlas.w_fake_ema))
        )
        psi_next = psi_next - psi_next.mean()
        cap = (2.5 * float(dim) * r_macro_safe.square()).clamp_min(_VAR_FLOOR)
        self.atlas.psi = psi_next * (
            cap / psi_next.abs().amax().clamp_min(_VAR_FLOOR)
        ).clamp_max(1.0)
        return k_star, g_psi

    @torch.no_grad()
    def apply_prior_row_update(
        self,
        z_param: torch.Tensor,
        s_idx: torch.Tensor,
        is_first: torch.Tensor,
        g_rows: torch.Tensor,
        part_scale: torch.Tensor,
        row_lr: float,
        row_settle: torch.Tensor,
    ) -> None:
        """Clock-free stationary EMA row update initialized on first touch via boolean ``row_init``."""
        dev = z_param.device
        p_cnt, d_z = int(z_param.shape[0]), int(z_param.shape[1])
        if (
            not self.rows_initialized
            or self.row_m is None
            or tuple(self.row_m.shape) != (p_cnt, d_z)
        ):
            self.row_m = torch.zeros((p_cnt, d_z), device=dev, dtype=torch.float32)
            self.row_v = torch.zeros((p_cnt, 1), device=dev, dtype=torch.float32)
            self.row_init = torch.zeros((p_cnt, 1), device=dev, dtype=torch.bool)
            self.rows_initialized = True
        elif self.row_m.device != dev:
            self.row_m = self.row_m.to(dev)
            self.row_v = self.row_v.to(dev)  # type: ignore[union-attr]
            self.row_init = self.row_init.to(dev)  # type: ignore[union-attr]

        assert self.row_v is not None and self.row_init is not None
        g_f = g_rows.detach().float()
        w_col = is_first.float().unsqueeze(1)
        n_u = w_col.sum().clamp_min(1.0)
        inited = self.row_init[s_idx]
        m_prev, v_prev = self.row_m[s_idx], self.row_v[s_idx]
        g2 = g_f.square().sum(dim=-1, keepdim=True)
        m_r = torch.where(inited, _ROW_BETA * m_prev + (1.0 - _ROW_BETA) * g_f, g_f)
        v_r = torch.where(inited, _ROW_BETA * v_prev + (1.0 - _ROW_BETA) * g2, g2)
        self.row_m.index_add_(0, s_idx, (m_r - m_prev) * w_col)
        self.row_v.index_add_(0, s_idx, (v_r - v_prev) * w_col)
        self.row_init[s_idx] = inited | is_first.unsqueeze(1)

        v_hat = v_r.clamp_min(0.0).sqrt()
        g_ref = (v_hat * w_col).sum().div(n_u).clamp_min(1e-30)
        step_dir = (
            (m_r / (v_hat + (1e-5 * g_ref).clamp_min(1e-30)))
            * part_scale.float()
            * w_col
        )
        z_param.index_add_(
            0,
            s_idx,
            (step_dir * (-float(row_lr) * row_settle.float())).to(z_param.dtype),
        )

    def state_dict_checkpoint(self) -> Dict[str, Any]:
        """Serialize schema-2 checkpoint state (atlas + curvature EMAs + clock-free row EMAs)."""
        self._flush_emas()
        self._sync_iso_and_curvature()
        return {
            "schema": SCHEMA_VERSION,
            "max_prototypes": int(self.max_prototypes),
            "atlas": self.atlas.state_dict(),
            "v_real_ema": None
            if self.v_real_ema is None
            else self.v_real_ema.detach().cpu().clone(),
            "v_tang_ema": None
            if self.v_tang_ema is None
            else self.v_tang_ema.detach().cpu().clone(),
            "res_ema": float(self._ema["res_ema"]),
            "iso_ema": float(self._ema["iso_ema"]),
            "iso_err_ema": float(self._ema["iso_err_ema"]),
            "rows_initialized": bool(self.rows_initialized),
            "row_m": None if self.row_m is None else self.row_m.detach().cpu().clone(),
            "row_v": None if self.row_v is None else self.row_v.detach().cpu().clone(),
            "row_init": None
            if self.row_init is None
            else self.row_init.detach().cpu().clone(),
        }

    def load_state_dict_checkpoint(
        self, state: Dict[str, Any], *, validate_only: bool = False
    ) -> None:
        """Fail-closed schema-2 restore preserving exact FP32/bool state on bound device."""
        val = validate_axiom_prior_state(state)
        if val["max_prototypes"] != self.max_prototypes:
            raise ValueError("AxiomPriorRegularizer max_prototypes mismatch")
        if val["rows_initialized"] and self._bound_shape is not None:
            if tuple(val["row_m"].shape) != tuple(self._bound_shape):
                raise ValueError("AxiomPriorRegularizer row_m shape mismatch")
        if validate_only:
            return
        dst = self._bound_device
        self._iso_var_override = None
        self._last_raw_var_cache = None
        self.atlas.load_state_dict(val["atlas"], device=dst)
        self._ema = {k: float(val[k]) for k in ("res_ema", "iso_ema", "iso_err_ema")}
        self._ema_t = {}
        self.rows_initialized = bool(val["rows_initialized"])
        for attr, dt in (
            ("v_real_ema", torch.float32),
            ("v_tang_ema", torch.float32),
            ("row_m", torch.float32),
            ("row_v", torch.float32),
            ("row_init", torch.bool),
        ):
            t = val[attr]
            if t is None:
                setattr(self, attr, None)
            else:
                out = (
                    t.detach().to(device=dst, dtype=dt)
                    if dst is not None
                    else t.detach().to(dtype=dt)
                )
                setattr(self, attr, out.clone())

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        return self.state_dict_checkpoint()

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        del strict, assign
        self.load_state_dict_checkpoint(state_dict)


# ---------------------------------------------------------------------------
# Rank-Free Pullback Eligibility & Step Execution
# ---------------------------------------------------------------------------


def _is_degenerate_point_mass(real: torch.Tensor) -> bool:
    rf = real.detach().reshape(int(real.shape[0]), -1).float()
    thresh = (1e-11 * rf.square().sum(dim=1).mean()).clamp_min(1e-22)
    return bool((rf.var(dim=0, unbiased=False).sum() < thresh).item())


def should_use_axiom_pullback(
    trainer: Any,
    real: torch.Tensor,
    latent: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
) -> bool:
    """Strictly rank-free pullback gate: ``1 <= per_sample_dim <= 64`` with ``sigma_z > 0`` or point-mass."""
    del latent, indices
    r = trainer.recipe
    if (
        r.prior_regularizer != "axiom"
        or float(r.t1_pullback_weight) <= 0.0
        or r.encoder_mode != "none"
        or r.conditioning != "scalar"
        or real.ndim < 2
        or int(real.shape[0]) < 1
    ):
        return False
    per_sample_dim = int(real[0].numel())
    if not (1 <= per_sample_dim <= MAX_FULL_COV_DIM):
        return False
    if per_sample_dim >= 2 and has_positive_prior_sigma(trainer.prior):
        return True
    is_pt = _is_degenerate_point_mass(real)
    reg = getattr(trainer, "prior_regularizer", None)
    if reg is not None and hasattr(reg, "_last_raw_var_cache"):
        reg._last_raw_var_cache = (real, is_pt)
    return is_pt


def _compute_aux_gen_losses(
    trainer: Any,
    base_prior: Any,
    fake_g: torch.Tensor,
    real_g: torch.Tensor,
    fake_logits: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
    zero: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    r = trainer.recipe
    tr = (
        r.kinetic_transport_loss(fake_g, real_g) if r.kinetic_transport_weight else zero
    )
    tr_loc = (
        r.kinetic_transport_local_loss(fake_g, real_g)
        if r.kinetic_transport_local_weight
        else zero
    )
    pm = None
    if getattr(trainer, "_prior_penalty", False):
        rows = (
            base_prior.z
            if r.num_particles <= 1024
            else base_prior.z[torch.unique(indices)]
        )
        pm = trainer.prior.regularize(
            rows,
            sampled_codes=latent,
            indices=indices,
            generated=fake_g,
            logits=fake_logits,
        )
    has_l2 = hasattr(base_prior, "z") and base_prior.z.requires_grad and r.prior_l2 > 0
    l2_term = r.prior_l2 * base_prior.z.square().mean() if has_l2 else zero
    return pm, tr, tr_loc, l2_term


def _step_gen_optimizer(
    opt_g: Any,
    g_scale: float | torch.Tensor,
    wakeup_boost: float | torch.Tensor,
    prior_scale: float,
) -> None:
    if hasattr(opt_g, "bind_axiom_modulation"):
        opt_g.bind_axiom_modulation(
            g_scale=g_scale, wakeup_boost=wakeup_boost, prior_scale=prior_scale
        )
        opt_g.step()
        return
    gs = float(g_scale.item()) if isinstance(g_scale, torch.Tensor) else float(g_scale)
    saved = [grp["lr"] for grp in opt_g.param_groups]
    for grp in opt_g.param_groups:
        grp["lr"] *= gs
    try:
        opt_g.step()
    finally:
        for grp, lr0 in zip(opt_g.param_groups, saved):
            grp["lr"] = lr0


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
    zero = loss_gan.new_zeros(())
    rec_d, opt_g = getattr(trainer.opt_d, "record", None), trainer.opt_g
    pm, tr, tr_loc, l2_term = _compute_aux_gen_losses(
        trainer, base_prior, fake_g, real_g, fake_logits, latent, indices, zero
    )
    loss_g = loss_g + tr + tr_loc + (pm if pm is not None else zero) + l2_term
    if rec_d is not None:
        rec_d._disc_gate = 0.0
        if res_ema_t is not None and hasattr(rec_d, "_ema_t"):
            rec_d._ema_t["ema_push_res"] = res_ema_t.detach().double()
        else:
            rec_d.ema_push_res = float(trainer.prior_regularizer.res_ema)
    opt_g.zero_grad()
    if getattr(opt_g, "direction_blend", False) and hasattr(
        opt_g, "bind_protected_from_grads"
    ):
        loss_g.backward()
        opt_g.bind_protected_from_grads()
    else:
        constraint_geometry_backward(loss_g, opt_g, (loss_g,))
    z_p = getattr(base_prior, "z", None)
    if isinstance(z_p, torch.Tensor) and z_p.requires_grad and z_p.grad is None:
        z_p.grad = torch.zeros_like(z_p)
    trainer.policy.after_generator_backward(
        loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach()
    )
    _step_gen_optimizer(opt_g, g_scale, wakeup_boost, prior_scale)
    trainer.policy.after_generator_step()
    return loss_g, pm, tr, tr_loc


def _run_point_mass_step(
    trainer: Any,
    reg: AxiomPriorRegularizer,
    base_prior: Any,
    gen_clean: Callable[[torch.Tensor], torch.Tensor],
    fake_g: torch.Tensor,
    real_g: torch.Tensor,
    yr: torch.Tensor,
    fake_logits: torch.Tensor,
    loss_gan: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
    loss_d: torch.Tensor,
    penalty: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    recipe, b = trainer.recipe, int(yr.shape[0])
    fake_flat = fake_g.reshape(b, -1)
    dim = int(fake_flat.shape[1])
    with torch.no_grad():
        err = (fake_flat.detach().float() - yr.float()).norm(dim=-1)
        cur = max(float(err.mean().item()), 0.35 * float(err.amax().item()))
        prev_r = float(reg.res_ema)
        reg.res_ema = (
            min(cur, 10.0)
            if cur > max(3.5 * prev_r, prev_r + 0.12)
            else 0.88 * prev_r + 0.12 * min(cur, 10.0)
        )
        g_sc = max(0.015, min(1.0, float(reg.res_ema) / 0.025))
        zd_pt = latent.detach()
        u_pt = torch.ones_like(zd_pt) / math.sqrt(float(max(1, int(zd_pt.shape[1]))))
        x_pm = gen_clean(torch.cat((zd_pt + 1e-3 * u_pt, zd_pt - 1e-3 * u_pt), dim=0))
        x_pm = x_pm.reshape(2, b, dim).float()
        s_c = ((x_pm[0] - x_pm[1]).abs() / 2e-3).clamp_min(1e-5)
        ref_s = torch.maximum(
            torch.quantile(s_c, 0.5, dim=0, keepdim=True),
            s_c.new_full((1, dim), max(0.15 * float(reg.res_ema), 1e-4)),
        )
        w_pt = (ref_s / s_c).clamp(1.0, 12.0).to(fake_flat.dtype)

    loss_pt = (
        torch.sqrt(1e-6 + w_pt * (fake_flat - yr.to(fake_flat.dtype)).square()) - 1e-3
    ).mean()
    if hasattr(base_prior, "z") and base_prior.z.requires_grad:
        loss_pt = loss_pt + 0.004 * base_prior.z.float().square().mean().to(
            fake_flat.dtype
        )
    prior_diag = (
        reg(
            base_prior.z
            if recipe.num_particles <= 1024
            else base_prior.z[torch.unique(indices)]
        ).detach()
        if (
            hasattr(base_prior, "z")
            and base_prior.z.requires_grad
            and recipe.prior_reg > 0
        )
        else loss_gan.new_zeros(())
    )
    adv_w = 0.02 * min(1.0, float(reg.res_ema) / 0.25)
    loss_g, pm, tr, tr_l = _finalize_gen_step(
        trainer,
        base_prior,
        adv_w * loss_gan + float(recipe.t1_pullback_weight) * loss_pt,
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


@torch.no_grad()
def _sinkhorn_barycenter(
    d_route: torch.Tensor,
    res_prev: torch.Tensor,
    proto_w: torch.Tensor,
    c_bar: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    b = int(d_route.shape[0])
    eps = (0.25 * res_prev).clamp(0.010, 0.06)
    log_k = -(d_route / d_route.mean().clamp_min(1e-12)) / eps
    log_a = d_route.new_full((b,), -math.log(float(max(1, b))))
    log_b = proto_w.clamp_min(1e-12).log()
    plan = log_sinkhorn(log_k, log_a, log_b, iters=10).exp_()
    return (plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12) @ c_bar).to(out_dtype)


@torch.no_grad()
def _compute_pullback_targets(
    reg: AxiomPriorRegularizer,
    coal: CoalescedAtlas,
    yr: torch.Tensor,
    x_ctr_det: torch.Tensor,
    fake_det: torch.Tensor,
    indices: torch.Tensor,
    base_prior: Any,
    can_condense: bool,
    sigma_z: float,
    was_init: bool,
) -> Dict[str, Any]:
    dim, dev = int(yr.shape[1]), yr.device
    yf = yr.float()
    k_proto, k_c = int(coal.proto_c_bar.shape[0]), int(coal.comp_c.shape[0])
    r_macro = yf.std(dim=0, unbiased=False).mean().clamp_min(1e-12)
    s_ot = (
        0.5 * yf.var(dim=0, unbiased=False).mean().clamp_min(_VAR_FLOOR).sqrt()
    ).clamp_min(1e-12)
    sigma_cell_med = torch.quantile(coal.comp_var.sqrt(), 0.35).clamp_min(1e-12)
    near_c = safe_cdist(x_ctr_det, coal.comp_c).argmin(dim=1)

    # Unified GPU Hamilton commitment ledger whenever all_iso & K_C >= 2 on finite learnable table.
    ledger_out = None
    if can_condense and hasattr(base_prior, "z"):
        ledger_out = reg.atlas.commit_particles(
            x_ctr_det,
            indices,
            int(base_prior.z.shape[0]),
            coalesced=coal,
            allow_update=True,
        )

    use_ledger = ledger_out is not None
    comp_i = ledger_out.batch_comp if ledger_out is not None else near_c
    all_iso_t = coal.all_iso
    xff = torch.where(all_iso_t, x_ctr_det, fake_det)
    scale_ratio = xff.std(dim=0, unbiased=False).mean().clamp_min(1e-12) / r_macro
    wakeup_t = torch.where(
        scale_ratio < 0.50,
        scale_ratio.new_full((), 5.0),
        torch.where(
            scale_ratio < 0.80,
            scale_ratio.new_full((), 2.0),
            scale_ratio.new_full((), 1.0),
        ),
    )

    iso_prev = reg._dev_ema("iso_ema", dev)
    res_prev = reg._dev_ema("res_ema", dev)
    iso_val = (
        (sigma_z * r_macro / sigma_cell_med).clamp(1.0, 4.5)
        if sigma_z > 0.0
        else iso_prev.new_ones(())
    )
    iso_ema_t = (
        (iso_val if not was_init else 0.95 * iso_prev + 0.05 * iso_val)
        if sigma_z > 0.0
        else iso_prev.new_ones(())
    )
    reg._ema_t["iso_ema"] = iso_ema_t

    d_xp_phys = safe_cdist(xff, coal.comp_c)
    res_cur = (
        (
            torch.maximum(
                (coal.comp_w * d_xp_phys.min(dim=0).values).sum(),
                d_xp_phys.min(dim=1).values.median(),
            )
            - 1.45 * math.sqrt(float(dim)) * sigma_cell_med
        ).clamp_min(0.0)
        / r_macro
    ).clamp_max(10.0)
    res_ema_t = 0.92 * res_prev + 0.08 * res_cur
    reg._ema_t["res_ema"] = res_ema_t

    # Curvature-compensated Marchenko-Pastur SPD target covariance per component.
    gamma_c = reg.curvature_gamma_per_component(coal)
    comp_cov_eff = coal.comp_cov_tgt * gamma_c.view(k_c, 1, 1)
    comp_chol = _cholesky_spd_safe(comp_cov_eff)

    if use_ledger and ledger_out is not None:
        y_ot = ledger_out.batch_target.clone()
    else:
        cb_m = (coal.comp_w.unsqueeze(1) * coal.comp_c).sum(dim=0, keepdim=True)
        d_route = d_xp_phys
        if k_c >= 4:
            x_w = (xff - xff.mean(dim=0, keepdim=True)) / xff.std(
                dim=0, unbiased=False, keepdim=True
            ).clamp_min(1e-12)
            c_std = (
                (coal.comp_w.unsqueeze(1) * (coal.comp_c - cb_m).square())
                .sum(dim=0, keepdim=True)
                .sqrt()
                .clamp_min(1e-12)
            )
            d_route = torch.where(
                scale_ratio < 0.35,
                safe_cdist(x_w, (coal.comp_c - cb_m) / c_std),
                d_xp_phys,
            )
        y_ot = _sinkhorn_barycenter(
            d_route, res_prev, coal.comp_w, coal.comp_c, torch.float32
        )

    # Anisotropic Box-Muller intra-component spread when pushforward variance < mode variance.
    v_push_c = coal.comp_c.new_zeros(k_c).index_add_(
        0,
        coal.proto_comp_id,
        (reg.atlas.proto_w * reg.atlas.v_push).float(),  # type: ignore[operator]
    ) / coal.comp_w.clamp_min(1e-12)
    spread_frac = (
        (1.0 - v_push_c / coal.comp_var.clamp_min(_VAR_FLOOR)).clamp(0.0, 0.75).sqrt()
    )
    if can_condense and sigma_z > 0.0:
        bm_t = _anisotropic_box_muller_targets(
            indices, comp_i, coal.comp_c, comp_chol, spread_frac
        )
        halo = (
            (
                (xff - coal.comp_c[comp_i]).norm(dim=-1)
                / (
                    3.5 * math.sqrt(float(dim)) * coal.comp_var[comp_i].sqrt()
                ).clamp_min(1e-12)
            )
            .clamp(0.0, 1.0)
            .square()
            .unsqueeze(1)
        )
        use_bm = all_iso_t & (spread_frac[comp_i] > 1e-3) & (scale_ratio >= 0.65)
        y_ot = torch.where(use_bm.unsqueeze(1), halo * y_ot + (1.0 - halo) * bm_t, y_ot)

    inv_w_i = (
        (1.0 / (float(k_c) * coal.comp_w[comp_i]).clamp_min(1e-4))
        .sqrt()
        .clamp(0.5, 4.0)
    )
    y_bound = yf.abs().amax(dim=0, keepdim=True) + 3.5 * sigma_cell_med
    return {
        "comp_i": comp_i,
        "comp_cov_eff": comp_cov_eff,
        "comp_chol": comp_chol,
        "spread_frac": spread_frac,
        "y_ot": y_ot,
        "s_ot": s_ot,
        "r_macro": r_macro,
        "sigma_cell_med": sigma_cell_med,
        "scale_ratio": scale_ratio,
        "wakeup_t": wakeup_t,
        "iso_ema_t": iso_ema_t,
        "res_ema_t": res_ema_t,
        "inv_w_i": inv_w_i,
        "y_bound": y_bound,
        "use_ledger": use_ledger,
        "ledger_out": ledger_out,
        "k_proto": k_proto,
    }


@torch.no_grad()
def _resolve_prior_row_targets(
    trainer: Any,
    reg: AxiomPriorRegularizer,
    coal: CoalescedAtlas,
    ctx: Dict[str, Any],
    s_idx: torch.Tensor,
    is_f: torch.Tensor,
    xu_det: torch.Tensor,
    out_dtype: torch.dtype,
    row_floor: torch.Tensor,
    has_sigma: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = s_idx.device
    if ctx["use_ledger"] and reg.atlas.ledger.row_comp is not None:
        rc = reg.atlas.ledger.row_comp.to(dev)
        c_sel = torch.where(
            rc[s_idx] >= 0,
            rc[s_idx].clamp_min(0),
            safe_cdist(xu_det, coal.comp_c).argmin(dim=1),
        )
        sc_u = coal.comp_var[c_sel].sqrt().to(out_dtype)
        g_psi = xu_det.new_zeros(())
        c_target = (
            _anisotropic_box_muller_targets(
                s_idx, c_sel, coal.comp_c, ctx["comp_chol"], ctx["spread_frac"]
            ).to(out_dtype)
            if has_sigma
            else coal.comp_c[c_sel].to(out_dtype)
        )
    else:
        k_star, g_psi = reg.update_laguerre_dual(
            xu_det,
            coal.proto_c_bar,
            coal.w_mat,
            ctx["r_macro"],
            float(trainer.recipe.t1_laguerre_lr),
            row_mask=is_f,
        )
        sc_u = coal.proto_var_bar[k_star].sqrt().to(out_dtype)
        c_target = coal.proto_c_bar[k_star].to(out_dtype)
    row_settle = torch.maximum(
        row_floor,
        torch.minimum(
            ctx["res_ema_t"].new_ones(()),
            torch.maximum(ctx["res_ema_t"] / 0.10, g_psi),
        ),
    )
    return c_target, sc_u, row_settle


def _update_prior_rows_step(
    trainer: Any,
    reg: AxiomPriorRegularizer,
    base_prior: Any,
    gen_clean: Callable[[torch.Tensor], torch.Tensor],
    coal: CoalescedAtlas,
    ctx: Dict[str, Any],
    indices: torch.Tensor,
    row_floor: torch.Tensor,
    has_sigma: bool,
) -> torch.Tensor:
    dev, dim = indices.device, int(coal.comp_c.shape[1])
    s_idx, _ = torch.sort(indices)
    is_f = torch.ones(int(s_idx.shape[0]), device=dev, dtype=torch.bool)
    if int(s_idx.shape[0]) > 1:
        is_f[1:] = s_idx[1:] != s_idx[:-1]
    w_f = is_f.float()
    n_u_t = w_f.sum().clamp_min(1.0)
    g_flags = [p.requires_grad for p in trainer.G.parameters()]
    try:
        for p in trainer.G.parameters():
            p.requires_grad_(False)
        c_u = (
            base_prior.means()
            if isinstance(base_prior, MoGParticlePrior)
            else base_prior.z
        )[s_idx]
        x_u = gen_clean(c_u).reshape(int(s_idx.shape[0]), dim)
        c_target, sc_u, row_settle = _resolve_prior_row_targets(
            trainer,
            reg,
            coal,
            ctx,
            s_idx,
            is_f,
            x_u.detach().float(),
            x_u.dtype,
            row_floor,
            has_sigma,
        )
        bi_u, _ = born_infeld_cost(
            x_u, c_target, (0.4 * sc_u).clamp_min(1e-12), aligned=True
        )
        loss_mu = (w_f.to(bi_u.dtype) * bi_u).sum() / (
            n_u_t.to(bi_u.dtype) * ctx["s_ot"].to(bi_u.dtype)
        )
        if has_sigma:
            mu_raw = base_prior.z[s_idx]
            z_mean = (w_f.unsqueeze(1).to(mu_raw.dtype) * mu_raw).sum(
                dim=0, keepdim=True
            ) / n_u_t.to(mu_raw.dtype)
            z_std = (
                (w_f.unsqueeze(1).to(mu_raw.dtype) * (mu_raw - z_mean).square()).sum(
                    dim=0
                )
                / n_u_t.to(mu_raw.dtype)
                + 1e-6
            ).sqrt()
            loss_mu = loss_mu + 0.15 * F.relu(
                ctx["iso_ema_t"].to(mu_raw.dtype) - z_std
            ).square().mean() * (n_u_t >= 2.0).to(mu_raw.dtype)
        if base_prior.z.grad is not None:
            base_prior.z.grad.zero_()
        loss_mu.backward()
        with torch.no_grad():
            g_rows = (base_prior.z.grad[s_idx] * n_u_t.to(base_prior.z.dtype)).clone()
            base_prior.z.grad.zero_()
            dist_u = (x_u.detach().float() - c_target.detach().float()).norm(dim=-1)
            part_scale = (
                (dist_u / (1.5 * sc_u.float()).clamp_min(1e-12))
                .clamp(0.02, 1.2)
                .unsqueeze(1)
            )
            reg.apply_prior_row_update(
                base_prior.z,
                s_idx,
                is_f,
                g_rows,
                part_scale,
                float(trainer.recipe.t1_prior_row_lr),
                row_settle,
            )
        return loss_mu.detach()
    finally:
        for p, fl in zip(trainer.G.parameters(), g_flags):
            p.requires_grad_(fl)
        rec_d = getattr(trainer.opt_d, "record", None)
        if rec_d is not None:
            rec_d._g_probe_cache = None


def _sample_pullback_fakes(
    trainer: Any,
    base_prior: Any,
    gen_clean: Callable[[torch.Tensor], torch.Tensor],
    b: int,
    latent: torch.Tensor,
    indices: torch.Tensor,
    sigma_out: Any,
    sigma_z: float,
    can_condense: bool,
    noise: torch.Generator,
    is_case_a: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if is_case_a:
        fake_g = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, latent, sigma_out, noise, rows=indices),
        )
        fake_flat = fake_g.reshape(b, -1)
        return fake_g, fake_flat, fake_flat, None, None

    use_means = isinstance(base_prior, MoGParticlePrior) and not bool(
        getattr(trainer.prior, "transforms_codes", False)
    )
    if use_means:
        mu_g = base_prior.means()[indices]
    elif can_condense:
        mu_g = base_prior.z[indices]
    else:
        mu_g = latent.detach()
    mu_det = mu_g.detach()
    if sigma_z > 0.0:
        eps_g: Optional[torch.Tensor] = ((latent - mu_g) / sigma_z).detach()
        z_in = mu_det + sigma_z * eps_g
    else:
        eps_g = None
        z_in = mu_det
    fake_g = trainer._postprocess(
        trainer.prior,
        trainer._generate(trainer.G, z_in, sigma_out, noise, rows=indices),
    )
    fake_flat = fake_g.reshape(b, -1)
    x_ctr_flat = gen_clean(mu_det).reshape(b, -1) if eps_g is not None else fake_flat
    return fake_g, fake_flat, x_ctr_flat, mu_det, eps_g


def _prepare_pullback_batch(
    trainer: Any,
    base_prior: Any,
    gen_clean: Callable[[torch.Tensor], torch.Tensor],
    real: torch.Tensor,
    generator_real: Any,
    latent: torch.Tensor,
    indices: torch.Tensor,
    sigma_out: Any,
    sigma_z: float,
    can_condense: bool,
    noise: torch.Generator,
    critic: nn.Module,
    is_case_a: bool,
) -> Dict[str, Any]:
    b = int(real.shape[0])
    fake_g, fake_flat, x_ctr_flat, mu_det, eps_g = _sample_pullback_fakes(
        trainer,
        base_prior,
        gen_clean,
        b,
        latent,
        indices,
        sigma_out,
        sigma_z,
        can_condense,
        noise,
        is_case_a,
    )
    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(None if is_case_a else fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:] or int(real_g.shape[0]) != b:
        raise ValueError(
            "generator_real must match the real sample shape and batch size"
        )
    fake_logits, real_logits = critic(fake_g), critic(real_g)
    rec_d = getattr(trainer.opt_d, "record", None)
    if rec_d is not None and hasattr(rec_d, "observe_margin_and_scale"):
        rec_d.observe_margin_and_scale(real_logits, fake_logits, real_g, fake_g)
    loss_gan = (
        F.softplus(-fake_logits).mean()
        if trainer.recipe.loss == "non_saturating"
        else trainer.loss.g_loss(fake_logits, real_logits)
    )
    return {
        "fake_g": fake_g,
        "fake_flat": fake_flat,
        "x_ctr_flat": x_ctr_flat,
        "real_g": real_g,
        "yr": real_g.detach().reshape(b, -1),
        "mu_det": mu_det,
        "eps_g": eps_g,
        "fake_logits": fake_logits,
        "loss_gan": loss_gan,
        "rec_d": rec_d,
    }


def _compute_burg_and_ema(
    reg: AxiomPriorRegularizer,
    gen_clean: Callable[[torch.Tensor], torch.Tensor],
    batch: Dict[str, Any],
    ctx: Dict[str, Any],
    sigma_z: float,
    iso_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_flat, x_ctr_flat, yr = batch["fake_flat"], batch["x_ctr_flat"], batch["yr"]
    mu_det, eps_g, rec_d = batch["mu_det"], batch["eps_g"], batch["rec_d"]
    iso_err_ema_t = reg._dev_ema("iso_err_ema", yr.device)
    if sigma_z <= 0.0 or mu_det is None or iso_weight <= 0.0:
        return fake_flat.new_zeros(()), iso_err_ema_t
    phase = float(getattr(rec_d, "chord_phase", 0.0)) if rec_d is not None else 0.0
    loss_iso_f, tang_var_i, iso_frob = whitened_burg_pullback_loss(
        gen_clean,
        mu_det,
        sigma_z,
        ctx["comp_cov_eff"][ctx["comp_i"]],
        ctx["inv_w_i"],
        eps_g=eps_g,
        phase=phase,
    )
    with torch.no_grad():
        m_sub = int(tang_var_i.shape[0])
        real_var_i = (
            fake_flat[:m_sub].detach().float() - x_ctr_flat[:m_sub].detach().float()
        ).square().sum(dim=-1) / float(yr.shape[1])
        near_p = safe_cdist(
            x_ctr_flat[:m_sub].detach().float(),
            reg.atlas.proto_c,  # type: ignore[arg-type]
        ).argmin(dim=1)
        reg.update_curvature_ema(near_p, real_var_i, tang_var_i)
        iso_cur = (
            0.60 * iso_frob.mean() + 0.40 * torch.quantile(iso_frob, 0.92)
        ).clamp_max(5.0)
        iso_err_ema_t = 0.92 * iso_err_ema_t + 0.08 * iso_cur
        reg._ema_t["iso_err_ema"] = iso_err_ema_t
    return loss_iso_f.to(fake_flat.dtype), iso_err_ema_t


def _compute_pullback_modulations(
    all_iso_t: torch.Tensor,
    res_ema_t: torch.Tensor,
    iso_err_ema_t: torch.Tensor,
    scale_ratio: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eq_g = torch.where(
        all_iso_t,
        ((0.030 - res_ema_t) / 0.020).clamp(0.0, 1.0),
        res_ema_t.new_zeros(()),
    )
    adv_w = torch.where(
        all_iso_t, 0.038 * (1.0 - eq_g) + 0.008 * eq_g, eq_g.new_full((), 0.15)
    ).to(out_dtype)
    g_floor = torch.where(
        all_iso_t, 0.022 * (1.0 - eq_g) + 0.012 * eq_g, eq_g.new_full((), 0.04)
    )
    row_floor = torch.where(
        all_iso_t, 0.15 * (1.0 - eq_g) + 0.38 * eq_g, eq_g.new_full((), 0.15)
    )
    iso_drive = ((iso_err_ema_t - 0.22) / 0.45).clamp(0.0, 1.0)
    g_scale_t = torch.where(
        scale_ratio < 0.60,
        res_ema_t.new_ones(()),
        torch.maximum(
            g_floor,
            torch.minimum(
                res_ema_t.new_ones(()), torch.maximum(res_ema_t / 0.10, iso_drive)
            ),
        ),
    )
    return adv_w, g_scale_t, row_floor


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
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Execute Axiom Law-1 Pullback: Anisotropic LTC + Hamilton Ledger + Curvature-Compensated Burg."""
    reg, recipe = trainer.prior_regularizer, trainer.recipe
    base_prior, sigma_z, can_condense = _prior_kernel_spec(trainer.prior)
    has_sigma = sigma_z > 0.0
    z_p = getattr(base_prior, "z", None)
    if isinstance(z_p, torch.Tensor):
        set_active_prior_rows(z_p, indices)

    rv_cache = getattr(reg, "_last_raw_var_cache", None)
    reg._last_raw_var_cache = None
    is_case_a = (
        bool(rv_cache[1])
        if (rv_cache is not None and rv_cache[0] is real)
        else _is_degenerate_point_mass(real)
    )
    gen_clean = lambda z_in: trainer._postprocess(
        trainer.prior, trainer._generate(trainer.G, z_in, 0.0, noise)
    )
    batch = _prepare_pullback_batch(
        trainer,
        base_prior,
        gen_clean,
        real,
        generator_real,
        latent,
        indices,
        sigma_out,
        sigma_z,
        can_condense,
        noise,
        critic,
        is_case_a,
    )
    if is_case_a:
        return _run_point_mass_step(
            trainer,
            reg,
            base_prior,
            gen_clean,
            batch["fake_g"],
            batch["real_g"],
            batch["yr"],
            batch["fake_logits"],
            batch["loss_gan"],
            latent,
            indices,
            loss_d,
            penalty,
        )

    was_init = reg.quantizer_initialized
    reg.update_quantizer(batch["yr"])
    coal = reg.atlas.coalesce(batch["yr"])
    ctx = _compute_pullback_targets(
        reg,
        coal,
        batch["yr"],
        batch["x_ctr_flat"].detach().float(),
        batch["fake_flat"].detach().float(),
        indices,
        base_prior,
        can_condense,
        sigma_z,
        was_init,
    )
    x_ot_src = batch["x_ctr_flat"] if has_sigma else batch["fake_flat"]
    bi_ot, _ = born_infeld_cost(
        x_ot_src,
        ctx["y_ot"].to(x_ot_src.dtype),
        ctx["sigma_cell_med"].to(x_ot_src.dtype),
        aligned=True,
    )
    loss_g_ot = bi_ot.mean() / ctx["s_ot"].to(x_ot_src.dtype) + 2.0 * (
        F.relu(batch["fake_flat"].float().abs() - ctx["y_bound"]) / ctx["s_ot"]
    ).square().mean().to(x_ot_src.dtype)
    loss_g_iso, iso_err_ema_t = _compute_burg_and_ema(
        reg, gen_clean, batch, ctx, sigma_z, float(recipe.t1_iso_weight)
    )
    adv_w, g_scale_t, row_floor = _compute_pullback_modulations(
        coal.all_iso,
        ctx["res_ema_t"],
        iso_err_ema_t,
        ctx["scale_ratio"],
        batch["loss_gan"].dtype,
    )
    loss_g, pm, tr, tr_l = _finalize_gen_step(
        trainer,
        base_prior,
        adv_w * batch["loss_gan"]
        + float(recipe.t1_pullback_weight) * loss_g_ot
        + float(recipe.t1_iso_weight) * loss_g_iso,
        batch["loss_gan"],
        batch["fake_g"],
        batch["real_g"],
        batch["fake_logits"],
        latent,
        indices,
        loss_d,
        penalty,
        g_scale_t,
        ctx["wakeup_t"],
        1.0,
        res_ema_t=ctx["res_ema_t"],
    )
    prior_diag = batch["loss_gan"].new_zeros(())
    can_row_upd = (
        can_condense
        and isinstance(z_p, torch.Tensor)
        and z_p.requires_grad
        and float(recipe.t1_prior_row_lr) > 0.0
    )
    if can_row_upd:
        prior_diag = _update_prior_rows_step(
            trainer,
            reg,
            base_prior,
            gen_clean,
            coal,
            ctx,
            indices,
            row_floor,
            has_sigma,
        )
    return loss_g, batch["loss_gan"], prior_diag, pm, tr, tr_l


__all__ = [
    "AxiomPriorRegularizer",
    "SCHEMA_VERSION",
    "allocate_deficit_quotas",
    "axiom_pullback_step",
    "central_pullback_cov",
    "coalesce_pythagorean_ltv",
    "has_positive_prior_sigma",
    "should_use_axiom_pullback",
    "validate_axiom_prior_state",
    "whitened_burg_pullback_loss",
]
