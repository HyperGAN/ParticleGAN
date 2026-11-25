"""Unified-v15 4-Law Pullback Prior Dynamics: 0.995 Jensen-Centered Total-Variance + fwAD Rail/Clamp Recovery.

Combines:
1. **Externally Isolated Pythagorean Total-Variance Conservation + `0.995` Quartic-Frobenius Jensen Centering**:
   Coalesces `2 <= |C| <= 4` Voronoi sibling sub-cells strictly when externally isolated (`d_ext >= 4.2 sqrt(D) sigma`),
   preserving anisotropic Zador-Gersho sub-cell clouds on `vector_*` and `mode_hold` (`w_mat == I`) while centering
   `cov_trace_bias` near `-0.02` simultaneously across `grid100`, `rotated100`, `staggered100`, and `14k` continuations.
2. **Case A Point-Mass & Rail-Escape `fwAD` Inverse-Attenuation (`zero_tanh_smoke`, `zero_tanh_rail_escape`)**:
   Detects degenerate point-mass targets (`raw_var < 1e-11` for any `sigma_z >= 0`) and applies 1st-order `fwAD`
   coordinatewise `sech^2` inverse-attenuation (`0` `autograd.grad` calls) with gentle latent centering.
3. **Case B Strange-Architecture & Rail-Clamp Defenses (`grid100_mlp16`, `grid100_res16_tanh`, `grid100_res16_scaled_tanh`)**:
   - Collapsed-spread whitened Sinkhorn (`scale_ratio < 0.35`) + `wakeup_boost` for deep skip-free vanishing networks,
   - Target-isometry-referenced `fwAD` saturation weights `w_sat_vec` + pre-rail envelope `loss_env` + `iso_gate` step floor,
   - Outward radial Jacobian column-sensitivity probe (`s_col < 0.04 * s_ref_col`) rescuing dead-clamped `prior.z` rows.
"""
from __future__ import annotations

import math
import struct
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._stats import knn_excluding_self, pairwise_sq_dist
from .._t4_arch.pullback import _deterministic_box_muller
from .._unified.prior import (
    _validate_unified_prior_state,
)
from .._bispf.thermal import FROZEN_THERMAL_SCALE, _read_thermal
from ..bispf import BISPFGates, SBNPPriorRegularizer
from ..moment_tether import pooled_reference_moments
from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .loss import _principal_axis_power_zerosync
from .._unified_v15.optimizer import UnifiedV11EquilibratedOptimizer

_PHI1 = 0.6180339887498949


def _quantile_1d_fast(x: torch.Tensor, q: float) -> torch.Tensor:
    """Bit-exact 1D linear-interpolation quantile matching ``torch.quantile`` (5x faster under deterministic CUDA)."""
    n = int(x.numel())
    if n <= 1:
        return x.reshape(()) if n == 1 else x.new_zeros(())
    s = x.reshape(-1).sort().values
    # PyTorch converts Python float ``q`` to ``x.dtype`` before multiplying by ``(n - 1)`` in double:
    qf = struct.unpack("f", struct.pack("f", float(q)))[0] if x.dtype == torch.float32 else float(q)
    pos = qf * float(n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return torch.lerp(s[lo], s[hi], pos - float(lo))


@torch.no_grad()
def _compute_pythagorean_smoothed_cells_fast(
    proto_c: torch.Tensor,
    proto_var: torch.Tensor,
    proto_w: torch.Tensor,
    *,
    eye_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bit-exact fast path of ``_compute_pythagorean_smoothed_cells`` eliminating the duplicate second ``[K, K, D]`` pass."""
    k, dim = proto_c.shape
    if eye_k is None or eye_k.shape != (k, k) or eye_k.device != proto_c.device:
        eye_k = torch.eye(k, device=proto_c.device, dtype=torch.float32)
    if k < 2:
        return proto_c, proto_var.clamp_min(1e-10).sqrt(), eye_k, torch.zeros(k, device=proto_c.device, dtype=torch.bool)

    sigma_c = proto_var.clamp_min(1e-10).sqrt()
    sqrt_d = math.sqrt(float(dim))
    d_cc = torch.cdist(proto_c, proto_c, p=2.0)
    sig_max = torch.maximum(sigma_c.unsqueeze(1), sigma_c.unsqueeze(0))
    sig_min = torch.minimum(sigma_c.unsqueeze(1), sigma_c.unsqueeze(0)).clamp_min(1e-8)

    in_sibling = (d_cc <= (2.6 * sqrt_d * sig_max)) & (sig_max <= (2.2 * sig_min))
    in_sibling.fill_diagonal_(True)

    adj = in_sibling.float()
    adj = ((adj @ adj) > 0).float()
    adj = ((adj @ adj) > 0).float()
    adj = ((adj @ adj) > 0).float()
    comp_size = adj.sum(dim=1)
    cand_coalesce = (comp_size >= 1.0) & (comp_size <= 4.0)

    cand_mask = torch.where(cand_coalesce.unsqueeze(1) & cand_coalesce.unsqueeze(0), adj, eye_k)
    cand_w = cand_mask * proto_w.clamp_min(1e-12).unsqueeze(0)
    cand_w.div_(cand_w.sum(dim=1, keepdim=True).clamp_min(1e-12))
    cand_c = cand_w @ proto_c
    cand_diff_sq = (proto_c.unsqueeze(0) - cand_c.unsqueeze(1)).square().sum(dim=-1) / float(dim)
    cand_var = (cand_w * (proto_var.unsqueeze(0) + cand_diff_sq)).sum(dim=1).clamp_min(1e-10)
    cand_sig = cand_var.sqrt()

    ext_d = d_cc.masked_fill(adj > 0.5, float("inf")).min(dim=1).values
    node_isolated = ext_d >= (4.2 * sqrt_d * cand_sig)
    not_iso_cnt = adj @ (~node_isolated).float()
    comp_isolated = not_iso_cnt < 0.5

    coalesce = cand_coalesce & comp_isolated
    multicell_coalesce = (comp_size > 1.5) & coalesce

    # For every coalesced component C, cand_mask[i] == adj[i] and cand_w/cand_c/cand_var are already exact;
    # for non-coalesced nodes, mask_mat[i] == e_i so c_bar[i] == proto_c[i] and var_bar[i] == proto_var[i].
    w_mat = torch.where(coalesce.unsqueeze(1), cand_w, eye_k)
    c_bar = torch.where(coalesce.unsqueeze(1), cand_c, proto_c)
    var_bar = torch.where(coalesce, cand_var, proto_var.clamp_min(1e-10))
    return c_bar, var_bar.sqrt(), w_mat, multicell_coalesce


@torch.no_grad()
def _sinkhorn_barycenter_fast(
    L_xp: torch.Tensor,
    log_a: float,
    log_b: torch.Tensor,
    c_bar_all: torch.Tensor,
    b: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Bit-exact 10-iteration log-domain Sinkhorn barycenter reusing one pre-allocated ``[B, K]`` workspace buffer."""
    buf = torch.empty_like(L_xp)
    # Iteration 0: u == 0 -> skip L_xp + 0
    m0 = L_xp.amax(dim=0, keepdim=True)
    torch.sub(L_xp, m0, out=buf)
    buf.exp_()
    v = log_b - (m0.squeeze(0) + buf.sum(dim=0).log_())
    torch.add(L_xp, v.unsqueeze(0), out=buf)
    m1 = buf.amax(dim=1, keepdim=True)
    buf.sub_(m1).exp_()
    u = log_a - (m1.squeeze(1) + buf.sum(dim=1).log_())
    for _ in range(9):
        torch.add(L_xp, u.unsqueeze(1), out=buf)
        m0 = buf.amax(dim=0, keepdim=True)
        buf.sub_(m0).exp_()
        v = log_b - (m0.squeeze(0) + buf.sum(dim=0).log_())
        torch.add(L_xp, v.unsqueeze(0), out=buf)
        m1 = buf.amax(dim=1, keepdim=True)
        buf.sub_(m1).exp_()
        u = log_a - (m1.squeeze(1) + buf.sum(dim=1).log_())
    torch.add(L_xp, u.unsqueeze(1), out=buf)
    buf.add_(v.unsqueeze(0)).exp_().mul_(float(b))
    buf.div_(buf.sum(dim=1, keepdim=True).clamp_min_(1e-12))
    return (buf @ c_bar_all).to(out_dtype)


_TRIU_IDX_CACHE: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_triu_indices(m: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (m, device)
    cached = _TRIU_IDX_CACHE.get(key)
    if cached is None:
        idx = torch.triu_indices(m, m, offset=1, device=device)
        cached = (idx[0], idx[1])
        if len(_TRIU_IDX_CACHE) < 64:
            _TRIU_IDX_CACHE[key] = cached
    return cached


def _median_offdiag_zerosync(distances: torch.Tensor) -> torch.Tensor:
    """Zero-sync exact lower median of off-diagonal entries of a square matrix."""
    m = int(distances.shape[0])
    if m < 2:
        return distances.new_zeros(())
    off = distances.reshape(-1)[:-1].view(m - 1, m + 1)[:, 1:].reshape(-1)
    return torch.median(off)


def _knn_sq_and_median_bandwidth_zerosync(z: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-sync exact k-NN squared distances and off-diagonal median squared bandwidth."""
    sq_dist = pairwise_sq_dist(z)
    with torch.no_grad():
        m = int(sq_dist.shape[0])
        if m < 2:
            r_med_sq = sq_dist.new_full((), 1e-12)
        else:
            # For symmetric sq_dist, the lower median of all m*(m-1) duplicated off-diagonal entries
            # is mathematically identical to the lower median of the m*(m-1)//2 upper-triangle entries.
            iu, ju = _get_triu_indices(m, sq_dist.device)
            r_med_sq = torch.median(sq_dist[iu, ju]).clamp_min(1e-12)
    knn_sq_dist, _ = knn_excluding_self(sq_dist, k)
    return knn_sq_dist, r_med_sq


def _has_positive_prior_sigma(prior: nn.Module) -> bool:
    """Zero-sync intrinsic check matching ``_resolve_prior_kernel(...)[1] > 0.0``."""
    base_prior = unwrap_prior(prior)
    if getattr(base_prior, "code_law", "table") != "table":
        return False
    transforms = bool(getattr(prior, "transforms_codes", False))
    if not transforms:
        if isinstance(base_prior, MoGParticlePrior):
            return bool(getattr(base_prior, "_noise_enabled", False))
        return False
    manips = tuple(getattr(prior, "manipulators", ()))
    if (
        len(manips) == 1
        and type(manips[0]).__name__ == "NormalNoiseManipulator"
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
    ):
        return float(manips[0].sigma) > 0.0
    return False


def _resolve_v11_prior_kernel(
    prior: nn.Module,
    latent: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, float, bool, bool]:
    """Zero-sync extraction of ``(mu_g, sigma_z, has_sigma, can_condense_table)``."""
    base_prior = unwrap_prior(prior)
    if getattr(base_prior, "code_law", "table") != "table":
        return latent.detach(), 0.0, False, False
    transforms = bool(getattr(prior, "transforms_codes", False))
    if not transforms:
        if isinstance(base_prior, MoGParticlePrior):
            has_sigma = bool(getattr(base_prior, "_noise_enabled", False))
            sigma_z = base_prior._cached_sigma_float() if has_sigma else 0.0
            mu_g = base_prior.means()[indices]
            return mu_g, sigma_z, sigma_z > 0.0, True
        return base_prior.z[indices], 0.0, False, True
    manips = tuple(getattr(prior, "manipulators", ()))
    if (
        len(manips) == 1
        and type(manips[0]).__name__ == "NormalNoiseManipulator"
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
    ):
        sig_f = float(manips[0].sigma)
        return base_prior.z[indices], sig_f, sig_f > 0.0, True
    return latent.detach(), 0.0, False, False


def _validate_unified_v15_prior_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("UnifiedV15PriorRegularizer checkpoint must be a mapping")
    if state.get("schema") != 2:
        raise ValueError("unsupported UnifiedV15PriorRegularizer checkpoint schema")
    if "v_push" not in state:
        raise ValueError("UnifiedV15PriorRegularizer checkpoint is missing 'v_push'")
    base = {k: v for k, v in state.items() if k != "v_push"}
    base["schema"] = 1
    validated = _validate_unified_prior_state(base)
    v_push = state["v_push"]
    if validated["quantizer_initialized"]:
        if (
            not isinstance(v_push, torch.Tensor)
            or not v_push.is_floating_point()
            or tuple(v_push.shape) != tuple(validated["proto_c"].shape)
            or not bool(torch.isfinite(v_push).all())
            or bool((v_push < 0.0).any())
        ):
            raise ValueError("invalid UnifiedV15PriorRegularizer v_push tensor")
    elif v_push is not None:
        raise ValueError("uninitialized UnifiedV15PriorRegularizer checkpoint must have v_push=None")
    return {**validated, "schema": 2, "v_push": v_push}


class UnifiedV15PriorRegularizer(nn.Module):
    """Recipe-owned Unified-v15 prior regularizer (``prior_regularizer='unified_v15'``)."""

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
        max_prototypes: int = 256,
        gates: Any = None,
        bound_device: Optional[torch.device] = None,
        bound_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an integer >= 2")
        self.max_prototypes = int(max_prototypes)
        from dataclasses import replace as _dc_replace
        base_gates = BISPFGates() if gates is None else gates
        self._active_iso_enabled = bool(base_gates.active("sbnp_active_iso"))
        self._active_iso_above = int(base_gates.sbnp_active_iso_above)
        self._active_iso_weight = float(base_gates.sbnp_active_iso_weight)
        no_eig_gates = (
            _dc_replace(
                base_gates,
                sbnp_active_iso_enabled=False,
                sbnp_thermal_full_max_dim=max(int(base_gates.sbnp_thermal_full_max_dim), 1 << 30),
            )
            if isinstance(base_gates, BISPFGates)
            else base_gates
        )
        sbnp_kwargs: Dict[str, Any] = {
            "weight": weight,
            "target_std": target_std,
            "eps": eps,
            "gates": no_eig_gates,
        }
        self.fallback = SBNPPriorRegularizer(reference, **sbnp_kwargs)
        self._bound_device = bound_device
        self._bound_shape = bound_shape
        self.unified_record: Any = None
        self.unified_v15_record: Any = None

        self.quantizer_initialized: bool = False
        self.proto_c: Optional[torch.Tensor] = None
        self.proto_var: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.proto_hits: Optional[torch.Tensor] = None
        self.v_push: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.w_fake_ema: Optional[torch.Tensor] = None
        self.phase: float = 0.0
        self.iso_initialized: bool = False
        self._iso_ema_val: float = 1.0
        self._res_ema_val: float = 1.0
        self._iso_err_ema_val: float = 0.0
        self._iso_ema_t: Optional[torch.Tensor] = None
        self._res_ema_t: Optional[torch.Tensor] = None
        self._iso_err_ema_t: Optional[torch.Tensor] = None
        self._cached_ref_key: Optional[Tuple[Any, ...]] = None
        self._cached_ref_total_n: Optional[int] = None
        self._last_raw_var_cache: Optional[Tuple[int, float]] = None
        self._cache_k_key: Optional[Tuple[int, torch.device]] = None
        self._cache_idx_k: Optional[torch.Tensor] = None
        self._cache_eye_k: Optional[torch.Tensor] = None
        self._cache_eye2_key: Optional[Tuple[torch.device, torch.dtype]] = None
        self._cache_eye2: Optional[torch.Tensor] = None
        self.rows_initialized: bool = False
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None

    def _get_k_helpers(self, k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transient (non-checkpointed) cached ``(arange(k), eye(k))`` avoiding per-step fill_ launches."""
        key = (k, device)
        if self._cache_k_key != key or self._cache_idx_k is None or self._cache_eye_k is None:
            self._cache_idx_k = torch.arange(k, device=device, dtype=torch.long)
            self._cache_eye_k = torch.eye(k, device=device, dtype=torch.float32)
            self._cache_k_key = key
        return self._cache_idx_k, self._cache_eye_k

    def _get_eye2(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Transient (non-checkpointed) cached ``eye(2).unsqueeze(0)``."""
        key = (device, dtype)
        if self._cache_eye2_key != key or self._cache_eye2 is None:
            self._cache_eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
            self._cache_eye2_key = key
        return self._cache_eye2

    def _flush_ema_scalars(self) -> None:
        keys = []
        tens = []
        for k, attr in (
            ("iso_ema", "_iso_ema_t"),
            ("res_ema", "_res_ema_t"),
            ("iso_err_ema", "_iso_err_ema_t"),
        ):
            t = getattr(self, attr, None)
            if t is not None:
                keys.append(k)
                tens.append(t)
        if not tens:
            return
        vals = torch.stack(tens).tolist()
        for k, v in zip(keys, vals):
            if k == "iso_ema":
                self._iso_ema_val = float(v)
                self._iso_ema_t = None
            elif k == "res_ema":
                self._res_ema_val = float(v)
                self._res_ema_t = None
            elif k == "iso_err_ema":
                self._iso_err_ema_val = float(v)
                self._iso_err_ema_t = None

    def _get_iso_ema_t(self, device: torch.device) -> torch.Tensor:
        if self._iso_ema_t is not None and self._iso_ema_t.device == device:
            return self._iso_ema_t
        if self._iso_ema_t is not None:
            self._flush_ema_scalars()
        return torch.full((), self._iso_ema_val, device=device, dtype=torch.float64)

    def _get_res_ema_t(self, device: torch.device) -> torch.Tensor:
        if self._res_ema_t is not None and self._res_ema_t.device == device:
            return self._res_ema_t
        if self._res_ema_t is not None:
            self._flush_ema_scalars()
        return torch.full((), self._res_ema_val, device=device, dtype=torch.float64)

    def _get_iso_err_ema_t(self, device: torch.device) -> torch.Tensor:
        if self._iso_err_ema_t is not None and self._iso_err_ema_t.device == device:
            return self._iso_err_ema_t
        if self._iso_err_ema_t is not None:
            self._flush_ema_scalars()
        return torch.full((), self._iso_err_ema_val, device=device, dtype=torch.float64)

    @property
    def iso_ema(self) -> float:
        if self._iso_ema_t is not None:
            self._flush_ema_scalars()
        return self._iso_ema_val

    @iso_ema.setter
    def iso_ema(self, val: float) -> None:
        self._iso_ema_val = float(val)
        self._iso_ema_t = None

    @property
    def res_ema(self) -> float:
        if self._res_ema_t is not None:
            self._flush_ema_scalars()
        return self._res_ema_val

    @res_ema.setter
    def res_ema(self, val: float) -> None:
        self._res_ema_val = float(val)
        self._res_ema_t = None

    @property
    def iso_err_ema(self) -> float:
        if self._iso_err_ema_t is not None:
            self._flush_ema_scalars()
        return self._iso_err_ema_val

    @iso_err_ema.setter
    def iso_err_ema(self, val: float) -> None:
        self._iso_err_ema_val = float(val)
        self._iso_err_ema_t = None

    @property
    def thermal_source(self) -> Any:
        return self.fallback.thermal_source

    @thermal_source.setter
    def thermal_source(self, value: Any) -> None:
        self.fallback.thermal_source = value

    @property
    def weight(self) -> float:
        return self.fallback.weight

    def _sbnp_fallback_zerosync(self, rows: torch.Tensor) -> torch.Tensor:
        """Exact zero-sync execution of ``SBNPPriorRegularizer.forward(rows)``."""
        out, _ = self._sbnp_fallback_zerosync_with_cov(rows)
        return out

    def _sbnp_fallback_zerosync_with_cov(self, rows: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact zero-sync SBNP returning ``(loss, full_cov_if_computed)`` to avoid duplicate ``D x D`` covariance."""
        fb = self.fallback
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if fb.weight == 0.0 or not len(rows):
            return rows[:0].sum(), None
        gates = fb.gates
        n, d = rows.shape
        min_dof = fb._min_dof(d)

        cov: Optional[torch.Tensor] = None
        if fb.references:
            initial_mean, initial_covariance, count_t = pooled_reference_moments(fb.references)
            ref_key = tuple(id(r) for r in fb.references)
            if self._cached_ref_key != ref_key or self._cached_ref_total_n is None:
                self._cached_ref_key = ref_key
                self._cached_ref_total_n = int(count_t.item())
            total_n = self._cached_ref_total_n
            if total_n <= min_dof or n <= min_dof or n < 2:
                return rows[:0].sum(), None
            if d != len(initial_mean):
                raise ValueError("sbnp rows and initial reference dimensions differ")
            mean = rows.mean(0)
            variance_scale = initial_covariance.diagonal().clamp_min(0) + fb.eps
            mean_error = ((mean - initial_mean).square() / variance_scale).mean()
            target_sigma = (initial_covariance.diagonal().clamp_min(0) + fb.eps).sqrt() * fb.target_std
            target_cov = initial_covariance
            target_var = variance_scale
            centered = rows - mean
            std = (centered.square().mean(0) + fb.eps).sqrt()
            var_loss = F.relu(target_sigma - std).square().mean()
            if d > 1:
                cov = (centered.T @ centered) / max(n - 1, 1)
                if (
                    gates.active("sbnp_full_cov")
                    and d <= gates.sbnp_full_cov_max_dim
                    and n <= gates.sbnp_full_cov_max_rows
                ):
                    cov_diff = cov - target_cov
                    cov_loss = (cov_diff.square() / (target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6)).sum()
                else:
                    denom = target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6
                    cov_loss = (
                        (cov.square() / denom).sum() - (cov.diagonal().square() / denom.diagonal()).sum()
                    ) / float(d * d)
            else:
                cov_loss = rows.new_zeros(())
        else:
            if n <= min_dof or n < 2:
                return rows[:0].sum(), None
            mean = rows.mean(0)
            mean_error = rows.new_zeros(())
            centered = rows - mean
            std = (centered.square().mean(0) + fb.eps).sqrt()
            var_loss = F.relu(fb.target_std - std).square().mean()
            if d > 1:
                cov = (centered.T @ centered) / max(n - 1, 1)
                t_var_s = fb.target_std * fb.target_std + fb.eps
                inv_cov_denom = 1.0 / ((t_var_s * t_var_s + 1e-6) * float(d * d))
                cov_loss = (cov.square().sum() - cov.diagonal().square().sum()) * inv_cov_denom
            else:
                cov_loss = rows.new_zeros(())

        m_sub = min(n, fb.repel_max_samples)
        z_sub = rows[(torch.arange(m_sub, device=rows.device) * n) // m_sub] if m_sub < n else rows

        local_iso_loss = rows.new_zeros(())
        if (
            gates.active("sbnp_local_iso")
            and d == gates.sbnp_local_iso_dim
            and not fb.references
            and m_sub > gates.sbnp_local_iso_above
        ):
            with torch.no_grad():
                k_iso = min(max(fb.k_nn, d + 1), m_sub - 1)
                _, iso_knn_idx = knn_excluding_self(pairwise_sq_dist(z_sub), k_iso)
            delta_z = z_sub[iso_knn_idx] - z_sub.unsqueeze(1)
            m_loc = torch.einsum("nkd,nke->nde", delta_z, delta_z) / max(1, k_iso)
            tr_m = torch.diagonal(m_loc, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            m_iso = (tr_m / d) * torch.eye(d, device=rows.device, dtype=rows.dtype).unsqueeze(0)
            a_loc = m_loc - m_iso
            local_iso_loss = (a_loc.pow(2).sum(dim=(-2, -1)) / ((tr_m.squeeze(-1).squeeze(-1) / d) + 1e-6).pow(2)).mean()

        is_small_latent = fb._is_small_latent(n, d)
        if is_small_latent:
            eff_k_nn = min(m_sub - 1, max(fb.k_nn, gates.sbnp_small_latent_min_k_nn))
            eff_repel_weight = fb.repel_weight * gates.sbnp_small_latent_repel_mult
            effective_margin = gates.sbnp_small_latent_margin
        else:
            eff_k_nn = fb.k_nn
            eff_repel_weight = fb.repel_weight
            small_margin = gates.active("sbnp_small_margin") and m_sub <= gates.sbnp_small_margin_max_rows
            effective_margin = gates.sbnp_small_margin if small_margin else fb.repel_margin
        k = min(eff_k_nn, m_sub - 1)
        if k > 0 and eff_repel_weight > 0.0:
            knn_sq_dist, r_med_sq = _knn_sq_and_median_bandwidth_zerosync(z_sub, k)
            norm_sq_dist = knn_sq_dist / (effective_margin * r_med_sq)
            repel_loss = F.softplus(-torch.log(norm_sq_dist.clamp_min(1e-12))).mean()
        else:
            repel_loss = rows.new_zeros(())

        global_moments = mean_error + var_loss + cov_loss
        scale_w = fb._applied_weight(is_small_latent)
        s = _read_thermal(fb.thermal_source, rows.device)
        local_iso_weight = gates.sbnp_local_iso_weight
        if s <= FROZEN_THERMAL_SCALE:
            base_loss = global_moments + eff_repel_weight * repel_loss + local_iso_weight * local_iso_loss
            out = scale_w * base_loss
            if (
                gates.active("sbnp_thermal_full")
                and d <= gates.sbnp_thermal_full_max_dim
                and out.requires_grad
            ):
                out = out * s + out.detach() * (1.0 - s)
            return out, cov
        if global_moments.requires_grad and s < 1.0:
            global_moments = global_moments * s + global_moments.detach() * (1.0 - s)
        base_loss = global_moments + eff_repel_weight * repel_loss + local_iso_weight * local_iso_loss
        return scale_w * base_loss, cov

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        reused_cov: Optional[torch.Tensor] = None
        if type(self.fallback) is SBNPPriorRegularizer:
            out, reused_cov = self._sbnp_fallback_zerosync_with_cov(rows)
        else:
            out = self.fallback(rows)
        if self.weight == 0.0 or rows.ndim != 2 or not len(rows):
            return out
        n, d = rows.shape
        if not self._active_iso_enabled or d < 2:
            return out
        m_sub = min(n, self.fallback.repel_max_samples)
        if m_sub <= self._active_iso_above or n <= self.fallback._min_dof(d):
            return out
        if m_sub == n and reused_cov is not None:
            cov_mat = reused_cov
        else:
            z_sub = rows[(torch.arange(m_sub, device=rows.device) * n) // m_sub] if m_sub < n else rows
            row_c = z_sub - z_sub.mean(dim=0, keepdim=True)
            cov_mat = (row_c.T @ row_c) / float(max(1, m_sub - 1))
        if d == 2:
            c00, c01, c11 = cov_mat[0, 0], cov_mat[0, 1], cov_mat[1, 1]
            tr = c00 + c11
            diff_sq = (c00 - c11).square() + 4.0 * c01.square()
            active_iso = self._active_iso_weight * diff_sq / (tr + 1e-6).square()
        else:
            with torch.no_grad():
                cdet = cov_mat.detach().float()
                v1 = _principal_axis_power_zerosync(cdet, iters=6, canonicalize_sign=False)
                lam1_d = (v1 * (cdet @ v1)).sum().clamp_min(0.0)
                cdef = cdet - lam1_d * torch.outer(v1, v1)
                v2 = _principal_axis_power_zerosync(cdef, iters=6, canonicalize_sign=False)
                v2 = v2 - (v2 @ v1) * v1
                v2 = v2 / v2.norm().clamp_min(1e-12)
            v1_t = v1.to(dtype=cov_mat.dtype)
            v2_t = v2.to(dtype=cov_mat.dtype)
            l1 = (v1_t * (cov_mat @ v1_t)).sum().clamp_min(0.0)
            l2 = (v2_t * (cov_mat @ v2_t)).sum().clamp_min(0.0)
            active_iso = self._active_iso_weight * (l1 - l2).square() / (l1 + l2 + 1e-6).square()
        scale_w = self.fallback._applied_weight(self.fallback._is_small_latent(n, d))
        s = _read_thermal(self.thermal_source, rows.device)
        term = scale_w * active_iso
        if (
            s <= FROZEN_THERMAL_SCALE
            and self.fallback.gates.active("sbnp_thermal_full")
            and d <= self.fallback.gates.sbnp_thermal_full_max_dim
            and term.requires_grad
        ):
            term = term * s + term.detach() * (1.0 - s)
        elif s < 1.0 and d > 4 and term.requires_grad:
            term = term * s + term.detach() * (1.0 - s)
        return out + term

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        return {
            "schema": 2,
            "quantizer_initialized": bool(self.quantizer_initialized),
            "proto_c": None if self.proto_c is None else self.proto_c.detach().clone(),
            "proto_var": None if self.proto_var is None else self.proto_var.detach().clone(),
            "proto_w": None if self.proto_w is None else self.proto_w.detach().clone(),
            "proto_hits": None if self.proto_hits is None else self.proto_hits.detach().clone(),
            "v_push": None if self.v_push is None else self.v_push.detach().clone(),
            "psi": None if self.psi is None else self.psi.detach().clone(),
            "w_fake_ema": None if self.w_fake_ema is None else self.w_fake_ema.detach().clone(),
            "phase": float(self.phase),
            "iso_initialized": bool(self.iso_initialized),
            "iso_ema": float(self.iso_ema),
            "res_ema": float(self.res_ema),
            "iso_err_ema": float(self.iso_err_ema),
            "rows_initialized": bool(self.rows_initialized),
            "row_m": None if self.row_m is None else self.row_m.detach().clone(),
            "row_v": None if self.row_v is None else self.row_v.detach().clone(),
            "row_n": None if self.row_n is None else self.row_n.detach().clone(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        del strict
        validated = _validate_unified_v15_prior_state(state_dict)
        if validated["rows_initialized"] and self._bound_shape is not None:
            if tuple(validated["row_m"].shape) != tuple(self._bound_shape):
                raise ValueError("UnifiedV15PriorRegularizer row_m shape does not match bound prior table")
        if validated["quantizer_initialized"] and validated["proto_c"].shape[0] > self.max_prototypes:
            raise ValueError("UnifiedV15PriorRegularizer proto_c count exceeds configured max_prototypes")

        dev = self._bound_device

        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.to(device=dev).detach().clone() if dev is not None else t.detach().clone()

        self.quantizer_initialized = bool(validated["quantizer_initialized"])
        self.proto_c = _mv(validated["proto_c"])
        self.proto_var = _mv(validated["proto_var"])
        self.proto_w = _mv(validated["proto_w"])
        self.proto_hits = _mv(validated["proto_hits"])
        self.v_push = _mv(validated["v_push"])
        self.psi = _mv(validated["psi"])
        self.w_fake_ema = _mv(validated["w_fake_ema"])
        self.phase = float(validated["phase"])
        self.iso_initialized = bool(validated["iso_initialized"])
        self.iso_ema = float(validated["iso_ema"])
        self.res_ema = float(validated["res_ema"])
        self.iso_err_ema = float(validated["iso_err_ema"])
        self.rows_initialized = bool(validated["rows_initialized"])
        self.row_m = _mv(validated["row_m"])
        self.row_v = _mv(validated["row_v"])
        self.row_n = _mv(validated["row_n"])

    @torch.no_grad()
    def update_quantizer(self, yr: torch.Tensor) -> None:
        if self._bound_device is None:
            self._bound_device = yr.device
        elif self.proto_c is not None and self.proto_c.device != yr.device:
            dev = yr.device
            for attr in ("proto_c", "proto_var", "proto_w", "proto_hits", "v_push", "psi", "w_fake_ema", "row_m", "row_v", "row_n"):
                val = getattr(self, attr, None)
                if isinstance(val, torch.Tensor) and val.device != dev:
                    setattr(self, attr, val.to(device=dev))

        b, dim = yr.shape
        x = yr.float()
        if not self.quantizer_initialized or self.proto_c is None or self.proto_c.shape[1] != dim:
            k = min(self.max_prototypes, max(4, (3 * b) // 4)) if b >= 8 else min(self.max_prototypes, b)
            idx_d = torch.arange(1, dim + 1, device=x.device, dtype=x.dtype)
            proj = torch.remainder(idx_d * _PHI1, 1.0) - 0.5
            proj = proj / proj.norm().clamp_min(1e-8)
            order = torch.argsort(x @ proj, stable=True)
            slots = ((torch.arange(k, device=x.device) * b + b // 2) // k).clamp(0, b - 1)
            self.proto_c = x[order[slots]].clone()
            if k >= 2:
                d_cc0 = torch.cdist(self.proto_c, self.proto_c)
                d_cc0.fill_diagonal_(float("inf"))
                d_nn = d_cc0.min(dim=1).values
                pos_nn = d_nn[d_nn > 1e-7]
                if len(pos_nn) > 0:
                    d_med = float(pos_nn.median().item())
                    v0 = 0.25 * (d_med * d_med) / float(dim + 2)
                else:
                    v0 = float(x.var(dim=0, unbiased=False).mean().clamp_min(1e-6).item()) / float(k)
            else:
                v0 = float(x.var(dim=0, unbiased=False).mean().clamp_min(1e-6).item())
            self.proto_var = torch.full((k,), max(v0, 1e-8), device=x.device, dtype=torch.float32)
            self.proto_w = torch.full((k,), 1.0 / float(k), device=x.device, dtype=torch.float32)
            self.proto_hits = torch.ones(k, device=x.device, dtype=torch.float32)
            self.w_fake_ema = torch.full((k,), 1.0 / float(k), device=x.device, dtype=torch.float32)
            self.psi = torch.zeros(k, device=x.device, dtype=torch.float32)
            self.quantizer_initialized = True
            self.v_push = self.proto_var.unsqueeze(1).expand_as(self.proto_c).clone()
            return

        k = len(self.proto_c)
        c_prev = self.proto_c.clone()
        d2 = torch.cdist(x, c_prev, p=2.0).square()
        d2_min, assign = d2.min(dim=1)

        med_v = self.proto_var.median().clamp_min(1e-10)
        v_loc = torch.maximum(self.proto_var[assign], med_v)

        resid_sq_prev = (x - c_prev[assign]).square().sum(dim=1) / float(dim)
        packed = torch.cat((x.new_ones((b, 1)), resid_sq_prev.unsqueeze(1), x), dim=1)
        sums = x.new_zeros((k, dim + 2))
        sums.index_add_(0, assign, packed)
        b_cnt = sums[:, 0]
        b_vsum = sums[:, 1]
        b_sum = sums[:, 2:]
        b_w = b_cnt / float(b)

        raw_spread = 2.0 * x.var(dim=0, unbiased=False).sum()
        has_spread_t = raw_spread > 1e-10
        pair_spread = raw_spread.clamp_min(1e-12)
        r2p_t = torch.where(has_spread_t, torch.sqrt(d2_min.mean() / pair_spread).double(), raw_spread.new_zeros((), dtype=torch.float64))
        shock_t64 = ((r2p_t - 0.45) / 0.35).clamp(0.0, 1.0)
        self.proto_hits = torch.where(
            shock_t64 > 0.25,
            (self.proto_hits * (1.0 - 0.8 * shock_t64).float()).clamp_min(1.0),
            self.proto_hits,
        )

        alpha_w_t64 = 0.02 + 0.48 * (shock_t64 * shock_t64)
        alpha_w_t = alpha_w_t64.float()
        self.proto_w = (1.0 - alpha_w_t64).float() * self.proto_w + alpha_w_t * b_w
        self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

        hit = b_cnt > 0
        b_cnt_safe = b_cnt.clamp_min(1.0)
        v_est = (b_vsum / b_cnt_safe).clamp_min(1e-10)
        self.proto_var = torch.where(hit, 0.88 * self.proto_var + 0.12 * v_est, self.proto_var)

        b_mean = b_sum / b_cnt_safe.unsqueeze(1)
        new_hits = (self.proto_hits + b_cnt).clamp_max(2000.0)
        self.proto_hits = torch.where(hit, new_hits, self.proto_hits)
        base_c = (0.035 + 0.25 / torch.sqrt(self.proto_hits)) + (0.38 * (shock_t64 * shock_t64)).float()
        alpha_c = (base_c * (b_cnt / (b_cnt + 2.0))).clamp_max(0.50).unsqueeze(1)
        self.proto_c = torch.where(hit.unsqueeze(1), (1.0 - alpha_c) * c_prev + alpha_c * b_mean, c_prev)

        mean_w = 1.0 / float(k)
        dead = self.proto_w < (0.04 * mean_w)
        uncovered_ratio = d2_min / torch.maximum(36.0 * float(dim) * v_loc, 0.08 * pair_spread)
        max_ur, worst_j = uncovered_ratio.max(dim=0)
        do_uncovered_t = (max_ur > 1.0) & has_spread_t if k >= 8 else raw_spread.new_zeros((), dtype=torch.bool)

        d_eff = float(min(dim, 4))
        gamma = d_eff / (d_eff + 2.0)
        gersho_e = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
        med_e = gersho_e.median().clamp_min(1e-12)
        max_e, pi_t = gersho_e.max(dim=0)
        w_pi = self.proto_w.index_select(0, pi_t.unsqueeze(0)).squeeze(0)
        need_gersho_split_t = (
            (max_e > 10.0 * med_e) & (w_pi > 0.25 * mean_w)
            if k >= 16
            else raw_spread.new_zeros((), dtype=torch.bool)
        )
        any_dead_t = dead.any()
        do_split_t = (
            (~do_uncovered_t) & has_spread_t & (any_dead_t | need_gersho_split_t)
            if k >= 2
            else raw_spread.new_zeros((), dtype=torch.bool)
        )

        # Single 3-bool batched check (1 sync/step; all internal slot/split ops are 0-sync on GPU):
        u_flag, s_flag, dead_flag = torch.stack((do_uncovered_t, do_split_t, any_dead_t)).tolist()
        if u_flag:
            idx_k = torch.arange(k, device=x.device, dtype=torch.long)
            if dead_flag:
                slot_t = self.proto_w.argmin()
            else:
                d_cc = torch.cdist(self.proto_c, self.proto_c).square()
                d_cc.fill_diagonal_(float("inf"))
                slot_t = d_cc.min(dim=1).values.argmin()
            is_slot = idx_k == slot_t
            x_worst = x.index_select(0, worst_j.unsqueeze(0))
            self.proto_c = torch.where(is_slot.unsqueeze(1), x_worst, self.proto_c)
            self.proto_var = torch.where(is_slot, med_v, self.proto_var)
            self.proto_hits = torch.where(is_slot, self.proto_hits.new_full((), 2.0), self.proto_hits)
            new_w_t = torch.maximum(alpha_w_t / float(b), alpha_w_t.new_full((), 0.05 * mean_w))
            self.proto_w = torch.where(is_slot, new_w_t, self.proto_w)
            self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
            self.w_fake_ema = torch.where(is_slot, new_w_t, self.w_fake_ema)
            self.psi = torch.where(is_slot, self.psi.new_zeros(()), self.psi)
        elif s_flag:
            dim_idx = torch.arange(1, dim + 1, device=x.device, dtype=torch.float32)
            idx_k = torch.arange(k, device=x.device, dtype=torch.long)
            if dead_flag:
                dead_ord = torch.where(dead, idx_k, idx_k.new_full((), k)).sort().values[:4]
                n_cand_slots = min(4, k)
                still_active = idx_k.new_ones((), dtype=torch.bool)
                for off in range(n_cand_slots):
                    di_t = dead_ord[off]
                    valid_slot = di_t < k
                    di_safe = di_t.clamp_max(k - 1)
                    is_di = idx_k == di_safe
                    ge = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
                    ge = torch.where(is_di, ge.new_zeros(()), ge)
                    p_best_t = ge.argmax()
                    p_best_1 = p_best_t.unsqueeze(0)
                    di_1 = di_safe.unsqueeze(0)
                    w_best = self.proto_w.index_select(0, p_best_1).squeeze(0)
                    still_active = still_active & valid_slot & (p_best_t != di_safe) & (w_best > 0.20 * mean_w)
                    is_pb = idx_k == p_best_t
                    ang = dim_idx * ((self.phase + (off + 1) * _PHI1) * 2.0 * math.pi)
                    direc = torch.cos(ang)
                    direc = direc / direc.norm().clamp_min(1e-8)
                    var_pb = self.proto_var.index_select(0, p_best_1).squeeze(0)
                    delta = 0.40 * math.sqrt(float(dim)) * var_pb.clamp_min(1e-10).sqrt() * direc
                    c_orig = self.proto_c.index_select(0, p_best_1).squeeze(0)
                    c_pb_new = (c_orig - delta).unsqueeze(0)
                    c_di_new = (c_orig + delta).unsqueeze(0)
                    self.proto_c = torch.where((still_active & is_pb).unsqueeze(1), c_pb_new, self.proto_c)
                    self.proto_c = torch.where((still_active & is_di).unsqueeze(1), c_di_new, self.proto_c)
                    w_di = self.proto_w.index_select(0, di_1).squeeze(0)
                    hw = 0.5 * (w_best + w_di)
                    both_m = still_active & (is_pb | is_di)
                    self.proto_w = torch.where(both_m, hw, self.proto_w)
                    new_var = (var_pb * 0.65).clamp_min(1e-10)
                    self.proto_var = torch.where(both_m, new_var, self.proto_var)
                    hits_pb = self.proto_hits.index_select(0, p_best_1).squeeze(0)
                    new_hits = (0.5 * hits_pb).clamp_min(2.0)
                    self.proto_hits = torch.where(both_m, new_hits, self.proto_hits)
                    apsi = 0.5 * (self.psi.index_select(0, p_best_1).squeeze(0) + self.psi.index_select(0, di_1).squeeze(0))
                    self.psi = torch.where(both_m, apsi, self.psi)
                    af = 0.5 * (
                        self.w_fake_ema.index_select(0, p_best_1).squeeze(0)
                        + self.w_fake_ema.index_select(0, di_1).squeeze(0)
                    )
                    self.w_fake_ema = torch.where(both_m, af, self.w_fake_ema)
                self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)
                self.phase = (self.phase + _PHI1) % 1.0
            else:
                d_cc = torch.cdist(self.proto_c, self.proto_c).square()
                d_cc.fill_diagonal_(float("inf"))
                is_pi = idx_k == pi_t
                d_cc = d_cc.masked_fill(is_pi.unsqueeze(1) | is_pi.unsqueeze(0), float("inf"))
                min_pair_d2 = d_cc.min()
                do_merge = min_pair_d2 <= (8.0 * float(dim) * med_v)
                flat_min = d_cc.argmin()
                a_t = torch.div(flat_min, k, rounding_mode="floor")
                b_t = flat_min - a_t * k
                a_1 = a_t.unsqueeze(0)
                b_1 = b_t.unsqueeze(0)
                is_a = idx_k == a_t
                is_b = idx_k == b_t
                wa = self.proto_w.index_select(0, a_1).squeeze(0)
                wb = self.proto_w.index_select(0, b_1).squeeze(0)
                wtot64 = (wa.double() + wb.double()).clamp_min(1e-12)
                wtot = wtot64.float()
                ca = self.proto_c.index_select(0, a_1)
                cb = self.proto_c.index_select(0, b_1)
                self.proto_c = torch.where(
                    (do_merge & is_a).unsqueeze(1),
                    (wa * ca + wb * cb) * (1.0 / wtot64).float(),
                    self.proto_c,
                )
                self.proto_w = torch.where(do_merge & is_a, wtot, self.proto_w)
                self.proto_w = torch.where(do_merge & is_b, self.proto_w.new_zeros(()), self.proto_w)
                wf_sum = self.w_fake_ema.index_select(0, a_1).squeeze(0) + self.w_fake_ema.index_select(0, b_1).squeeze(0)
                self.w_fake_ema = torch.where(do_merge & is_a, wf_sum, self.w_fake_ema)
                self.w_fake_ema = torch.where(do_merge & is_b, self.w_fake_ema.new_zeros(()), self.w_fake_ema)

                ge = (self.proto_w.clamp_min(1e-12) ** gamma) * self.proto_var
                ge = torch.where(is_b, ge.new_zeros(()), ge)
                p_best_t = ge.argmax()
                p_best_1 = p_best_t.unsqueeze(0)
                w_best = self.proto_w.index_select(0, p_best_1).squeeze(0)
                do_sp = do_merge & (p_best_t != b_t) & (w_best > 0.20 * mean_w)
                is_pb = idx_k == p_best_t
                ang = dim_idx * ((self.phase + _PHI1) * 2.0 * math.pi)
                direc = torch.cos(ang)
                direc = direc / direc.norm().clamp_min(1e-8)
                var_pb = self.proto_var.index_select(0, p_best_1).squeeze(0)
                delta = 0.40 * math.sqrt(float(dim)) * var_pb.clamp_min(1e-10).sqrt() * direc
                c_orig = self.proto_c.index_select(0, p_best_1).squeeze(0)
                self.proto_c = torch.where((do_sp & is_pb).unsqueeze(1), (c_orig - delta).unsqueeze(0), self.proto_c)
                self.proto_c = torch.where((do_sp & is_b).unsqueeze(1), (c_orig + delta).unsqueeze(0), self.proto_c)
                w_b_cur = self.proto_w.index_select(0, b_1).squeeze(0)
                hw = 0.5 * (w_best + w_b_cur)
                both_m = do_sp & (is_pb | is_b)
                self.proto_w = torch.where(both_m, hw, self.proto_w)
                self.proto_var = torch.where(both_m, (var_pb * 0.65).clamp_min(1e-10), self.proto_var)
                hits_pb = self.proto_hits.index_select(0, p_best_1).squeeze(0)
                self.proto_hits = torch.where(both_m, (0.5 * hits_pb).clamp_min(2.0), self.proto_hits)
                apsi = 0.5 * (self.psi.index_select(0, p_best_1).squeeze(0) + self.psi.index_select(0, b_1).squeeze(0))
                self.psi = torch.where(both_m, apsi, self.psi)
                af = 0.5 * (
                    self.w_fake_ema.index_select(0, p_best_1).squeeze(0)
                    + self.w_fake_ema.index_select(0, b_1).squeeze(0)
                )
                self.w_fake_ema = torch.where(both_m, af, self.w_fake_ema)
                self.proto_w = torch.where(do_merge, self.proto_w / self.proto_w.sum().clamp_min(1e-12), self.proto_w)
                if bool(do_merge.item()):
                    self.phase = (self.phase + _PHI1) % 1.0

        if (
            self.quantizer_initialized
            and self.proto_c is not None
            and (
                self.v_push is None
                or self.v_push.shape != self.proto_c.shape
                or self.v_push.device != self.proto_c.device
            )
        ):
            self.v_push = self.proto_var.unsqueeze(1).expand_as(self.proto_c).clone()


def should_use_unified_v15_pullback(
    trainer: Any,
    real: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Intrinsic geometric predicate for Unified-v15 Pullback + Rail-Escape + Laguerre Condensation.

    Active iff unconditional Euclidean vector space ``real.ndim == 2`` (`encoder_mode == 'none'`,
    `conditioning == 'scalar'`) and EITHER:
    - Case A: Point-mass / rail-escape target (`Var(real) < 1e-11`, `1 <= D <= 16`, any `sigma_z >= 0`), OR
    - Case B: Multi-D continuous-kernel Mixture-of-Gaussians (`2 <= D <= 16` and `sigma_z > 0`).
    """
    del latent, indices
    recipe = trainer.recipe
    if recipe.prior_regularizer != "unified_v15" or float(recipe.t1_pullback_weight) <= 0.0:
        return False
    if recipe.encoder_mode != "none" or recipe.conditioning != "scalar":
        return False
    if real.ndim != 2 or not (1 <= int(real.shape[1]) <= 16):
        return False
    if int(real.shape[1]) >= 2 and _has_positive_prior_sigma(trainer.prior):
        return True
    raw_var = float(real.detach().float().var(dim=0, unbiased=False).sum().item())
    reg = getattr(trainer, "prior_regularizer", None)
    if reg is not None and hasattr(reg, "_last_raw_var_cache"):
        reg._last_raw_var_cache = (id(real), raw_var)
    return raw_var < 1e-11


@torch.no_grad()
def _fwad_generator_probe(
    trainer: Any,
    z_base: torch.Tensor,
    tangent: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Evaluate 1st-order forward-AD directional sensitivity ``J_G(z_base) tangent`` with zero BN/hook mutation (`0` `autograd.grad`)."""
    rec_d = getattr(trainer.opt_d, "record", None)
    prev_probe = bool(getattr(rec_d, "_in_probe", False)) if rec_d is not None else False
    modes = [(m, m.training) for m in trainer.G.modules()]
    if rec_d is not None:
        rec_d._in_probe = True
    try:
        trainer.G.eval()
        z_det = z_base.detach()
        t_det = tangent.detach().to(dtype=z_det.dtype, device=z_det.device)
        try:
            with fwAD.dual_level():
                dual_in = fwAD.make_dual(z_det, t_det)
                if getattr(trainer, "_bf16_generator", False):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        raw_out = trainer.G(dual_in)
                else:
                    raw_out = trainer.G(dual_in)
                dual_out = trainer._postprocess(trainer.prior, raw_out)
                _, jvp = fwAD.unpack_dual(dual_out)
            if jvp is not None:
                return jvp.detach().float()
        except (RuntimeError, NotImplementedError, TypeError):
            pass
        eps_fd = 1e-3
        if getattr(trainer, "_bf16_generator", False):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yp = trainer._postprocess(trainer.prior, trainer.G(z_det + eps_fd * t_det)).float()
                ym = trainer._postprocess(trainer.prior, trainer.G(z_det - eps_fd * t_det)).float()
        else:
            yp = trainer._postprocess(trainer.prior, trainer.G(z_det + eps_fd * t_det)).float()
            ym = trainer._postprocess(trainer.prior, trainer.G(z_det - eps_fd * t_det)).float()
        return (yp - ym) / (2.0 * eps_fd)
    finally:
        if rec_d is not None:
            rec_d._in_probe = prev_probe
        for m, flag in modes:
            m.training = flag


def _point_mass_rail_escape_step(
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
    """Case A: 1st-order `fwAD` coordinatewise saturation inverse-attenuation for point-mass & `tanh` rail escape."""
    recipe = trainer.recipe
    reg: UnifiedV15PriorRegularizer = trainer.prior_regularizer
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)

    # Keep gradient flow into `latent` -> `prior.z` so zero-sigma & MoG particle clouds re-center cleanly!
    fake_g = trainer._postprocess(
        trainer.prior,
        trainer._generate(trainer.G, latent, sigma_out, noise, rows=indices),
    )
    fake_flat = fake_g.reshape(b, -1)
    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(None)

    fake_logits = critic(fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:]:
        raise ValueError("generator_real must match the real sample shape")
    if len(real_g) != len(real):
        raise ValueError("RpGAN generator_real must match the real batch size")
    real_logits = critic(real_g)

    rec_d = getattr(trainer.opt_d, "record", None)
    if rec_d is not None and hasattr(rec_d, "observe_margin_and_scale"):
        rec_d.observe_margin_and_scale(real_logits, fake_logits, real_g, fake_g)

    loss_gan = (
        F.softplus(-fake_logits).mean()
        if recipe.loss == "non_saturating"
        else trainer.loss.g_loss(fake_logits, real_logits)
    )
    yr = real_g.detach().reshape(b, -1)
    dim_pt = int(fake_flat.shape[1])

    with torch.no_grad():
        err_vec = (fake_flat.detach().float() - yr.float()).norm(dim=-1)
        cur_res = max(float(err_vec.mean().item()), 0.35 * float(err_vec.amax().item()))
        if cur_res > max(3.5 * reg.res_ema, reg.res_ema + 0.12):
            reg.res_ema = min(cur_res, 10.0)
        else:
            reg.res_ema = 0.88 * reg.res_ema + 0.12 * min(cur_res, 10.0)
        g_scale_pt = max(0.015, min(1.0, reg.res_ema / 0.025))

        # Deterministic radial/ones probe tangent in latent space (0 RNG draws, 0 autograd.grad):
        u_probe = torch.ones_like(latent.detach()) / math.sqrt(float(max(1, latent.shape[1])))
        jvp_pt = _fwad_generator_probe(trainer, latent.detach(), u_probe)
        if jvp_pt is not None:
            s_coord = jvp_pt.reshape(b, -1).abs().clamp_min(1e-5)
            s_ref = torch.maximum(
                torch.quantile(s_coord, 0.5, dim=0, keepdim=True),
                torch.full((1, dim_pt), max(0.15 * reg.res_ema, 1e-4), device=fake_flat.device, dtype=s_coord.dtype),
            )
            w_sat_pt = (s_ref / s_coord).clamp(1.0, 12.0).to(fake_flat.dtype)
        else:
            w_sat_pt = torch.ones_like(fake_flat)

    loss_pt = (torch.sqrt(1e-6 + w_sat_pt * (fake_flat - yr).square()) - 1e-3).mean()
    if base_prior.z.requires_grad:
        loss_pt = loss_pt + 0.004 * base_prior.z.float().square().mean().to(loss_pt.dtype)

    transport = (
        recipe.kinetic_transport_loss(fake_g, real_g)
        if recipe.kinetic_transport_weight
        else loss_gan.new_zeros(())
    )
    transport_local = (
        recipe.kinetic_transport_local_loss(fake_g, real_g)
        if recipe.kinetic_transport_local_weight
        else loss_gan.new_zeros(())
    )
    prior_reg_diag = loss_gan.new_zeros(())
    if base_prior.z.requires_grad and (recipe.prior_regularizer == "vicreg" or recipe.prior_reg > 0):
        raw = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        prior_reg_diag = trainer.prior_regularizer(raw).detach()

    gan_blend_pt = 0.02 * min(1.0, reg.res_ema / 0.25)
    loss_g = gan_blend_pt * loss_gan + float(recipe.t1_pullback_weight) * loss_pt
    if base_prior.z.requires_grad and recipe.prior_l2 > 0:
        loss_g = loss_g + recipe.prior_l2 * base_prior.z.square().mean()

    prior_manip_loss = None
    if trainer._prior_penalty:
        raw_rows = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        prior_manip_loss = trainer.prior.regularize(
            raw_rows,
            sampled_codes=latent,
            indices=indices,
            generated=fake_g,
            logits=fake_logits,
        )
        loss_g = loss_g + prior_manip_loss
    if recipe.kinetic_transport_weight:
        loss_g = loss_g + transport
    if recipe.kinetic_transport_local_weight:
        loss_g = loss_g + transport_local

    if rec_d is not None and hasattr(rec_d, "_t1c_disc_gate"):
        rec_d._t1c_disc_gate = 0.0
    if isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer):
        trainer.opt_g.bind_v4_modulation(g_scale=g_scale_pt, wakeup_boost=1.0, prior_scale=g_scale_pt)

    trainer.opt_g.zero_grad()
    if isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer) and trainer.opt_g.direction_blend:
        loss_g.backward()
        trainer.opt_g.bind_protected_from_grads()
    else:
        constraint_geometry_backward(loss_g, trainer.opt_g, (loss_g,))
    trainer.policy.after_generator_backward(
        loss_gan=loss_gan.detach(),
        loss_critic=(loss_d - penalty).detach(),
    )
    trainer.opt_g.step()
    trainer.policy.after_generator_step()
    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


def unified_v15_pullback_step(
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
    """Execute Unified-v15 4-Law Pullback + `0.995` Jensen-Centered Isometry + Rail/Dead-Clamp Recovery."""
    reg: UnifiedV15PriorRegularizer = trainer.prior_regularizer
    rv_cache = getattr(reg, "_last_raw_var_cache", None)
    reg._last_raw_var_cache = None
    if rv_cache is not None and rv_cache[0] == id(real):
        raw_var = rv_cache[1]
    else:
        raw_var = float(real.detach().float().var(dim=0, unbiased=False).sum().item())
    if raw_var < 1e-11:
        return _point_mass_rail_escape_step(
            trainer,
            real,
            generator_real,
            latent,
            indices,
            sigma_out=sigma_out,
            noise=noise,
            critic=critic,
            loss_d=loss_d,
            penalty=penalty,
        )

    recipe = trainer.recipe
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)

    mu_g, sigma_z, has_sigma, can_condense_table = _resolve_v11_prior_kernel(trainer.prior, latent, indices)

    # --- PART A: Smooth Generator Chart G_theta + 0.995 Jensen-Centered Total-Variance Isometry ---
    mu_det = mu_g.detach()
    if has_sigma:
        eps_g = ((latent - mu_g) / sigma_z).detach()
        latent_det = mu_det + sigma_z * eps_g
        fake_g = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, latent_det, sigma_out, noise, rows=indices),
        )
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_raw = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, mu_det, 0.0, noise),
        )
        x_ctr_flat = x_ctr_raw.reshape(b, -1)
    else:
        eps_g = None
        fake_g = trainer._postprocess(
            trainer.prior,
            trainer._generate(trainer.G, mu_det, sigma_out, noise, rows=indices),
        )
        fake_flat = fake_g.reshape(b, -1)
        x_ctr_flat = fake_flat

    if hasattr(trainer.opt_g, "bind_origin_sample"):
        trainer.opt_g.bind_origin_sample(fake_g)

    fake_logits = critic(fake_g)
    real_g = generator_real() if callable(generator_real) else generator_real
    real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
    if real_g.shape[1:] != real.shape[1:]:
        raise ValueError("generator_real must match the real sample shape")
    if len(real_g) != len(real):
        raise ValueError("RpGAN generator_real must match the real batch size")
    real_logits = critic(real_g)

    rec_d = getattr(trainer.opt_d, "record", None)
    if rec_d is not None and hasattr(rec_d, "observe_margin_and_scale"):
        rec_d.observe_margin_and_scale(real_logits, fake_logits, real_g, fake_g)

    loss_gan = (
        F.softplus(-fake_logits).mean()
        if recipe.loss == "non_saturating"
        else trainer.loss.g_loss(fake_logits, real_logits)
    )

    yr = real_g.detach().reshape(b, -1)
    dim = yr.shape[1]

    with torch.no_grad():
        reg.update_quantizer(yr)
        yf = yr.float()
        r_macro = yf.std(dim=0, unbiased=False).mean()
        is_point_t = r_macro < 1e-6
        r_macro_safe = r_macro.clamp_min(1e-4)

        k_proto = len(reg.proto_c)
        idx_k, eye_k = reg._get_k_helpers(k_proto, yr.device)
        c_bar_all, sigma_bar_all, w_mat_all, multicell_coal = _compute_pythagorean_smoothed_cells_fast(
            reg.proto_c, reg.proto_var, reg.proto_w, eye_k=eye_k
        )
        sigma_cell_med = _quantile_1d_fast(sigma_bar_all, 0.35).clamp_min(1e-6)

        x_ctr_det = x_ctr_flat.detach().float()
        fake_det = fake_flat.detach().float()
        d_ctr = torch.cdist(x_ctr_det, c_bar_all, p=2.0)
        near_p_G = d_ctr.argmin(dim=1)
        is_coal_i = multicell_coal[near_p_G]
        has_multicell_coal_t = multicell_coal.any()
        any_coal_i_t = is_coal_i.any()
        all_coal_t = is_coal_i.all()
        xff = torch.where(is_coal_i.unsqueeze(1), x_ctr_det, fake_det)

        # Collapsed-spread whitened Sinkhorn + wakeup boost for deep vanishing networks (e.g. 16-layer MLP):
        f_std = xff.std(dim=0, unbiased=False).mean().clamp_min(1e-8)
        scale_ratio_t = f_std / r_macro_safe
        wakeup_boost_t = torch.where(
            scale_ratio_t < 0.50,
            scale_ratio_t.new_full((), 5.0),
            torch.where(
                scale_ratio_t < 0.80,
                scale_ratio_t.new_full((), 2.0),
                scale_ratio_t.new_full((), 1.0),
            ),
        )

        v_mu_star_cand: Optional[torch.Tensor] = None
        if reg.v_push is not None:
            v_bar_sq = sigma_bar_all.square().unsqueeze(1)
            v_push_bar = (w_mat_all @ reg.v_push).clamp_min(0.0)
            deficit_ratio = ((v_bar_sq - v_push_bar) / v_bar_sq.clamp_min(1e-10)).clamp(0.0, 1.0)
            u_def = ((deficit_ratio - 0.15) / 0.12).clamp(0.0, 1.0)
            gate_def = u_def * u_def * (3.0 - 2.0 * u_def)
            v_mu_star_cand = gate_def * (v_bar_sq - v_push_bar).clamp_min(0.0)
            any_v_mu_star_t = has_multicell_coal_t & (v_mu_star_cand.amax() > 0.0)
        else:
            any_v_mu_star_t = has_multicell_coal_t.new_zeros((), dtype=torch.bool)

        # Single batched 7-bool control sync so every dead forward/backward branch is skipped completely:
        (
            is_point_flag,
            has_coal_flag,
            any_coal_i_flag,
            all_coal_flag,
            need_whiten_flag,
            scale_ge_065_flag,
            any_v_mu_star_flag,
        ) = torch.stack(
            (
                is_point_t,
                has_multicell_coal_t,
                any_coal_i_t,
                all_coal_t,
                scale_ratio_t < 0.35,
                scale_ratio_t >= 0.65,
                any_v_mu_star_t,
            )
        ).tolist()

        if has_sigma and not is_point_flag:
            iso_t = (sigma_z * r_macro_safe / sigma_cell_med).double().clamp(1.0, 4.5)
            if not reg.iso_initialized:
                reg._iso_ema_t = iso_t
                reg.iso_initialized = True
            else:
                prev_iso = reg._get_iso_ema_t(yr.device)
                reg._iso_ema_t = 0.95 * prev_iso + 0.05 * iso_t
        else:
            reg.iso_ema = 1.0
            reg.iso_initialized = True

        # Reuse d_ctr when all samples are coalesced (eliminates redundant 2048x256 cdist!):
        d_xp_phys = d_ctr if all_coal_flag else torch.cdist(xff, c_bar_all, p=2.0)
        if need_whiten_flag and k_proto >= 4:
            xff_w = (xff - xff.mean(dim=0, keepdim=True)) / xff.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            cb_mean = (reg.proto_w.unsqueeze(1) * c_bar_all).sum(dim=0, keepdim=True)
            cb_std = ((reg.proto_w.unsqueeze(1) * (c_bar_all - cb_mean).square()).sum(dim=0, keepdim=True)).sqrt().clamp_min(1e-6)
            cb_w = (c_bar_all - cb_mean) / cb_std
            d_xp_route = torch.cdist(xff_w, cb_w, p=2.0)
        else:
            d_xp_route = d_xp_phys

        c_scale = d_xp_route.mean().clamp_min(1e-5)
        prev_res64 = reg._get_res_ema_t(yr.device)
        # Match PyTorch CUDA float-scalar division ``x / float(eps_s)`` bit-for-bit with 0 syncs:
        eps_s_t64 = (0.25 * prev_res64).clamp(0.010, 0.06)
        L_xp = -(d_xp_route / c_scale) * (1.0 / eps_s_t64).float()
        log_a = -math.log(float(b))
        log_b = torch.log(reg.proto_w.clamp_min(1e-12))
        y_ot = _sinkhorn_barycenter_fast(L_xp, log_a, log_b, c_bar_all, b, fake_flat.dtype)

        p2f = d_xp_phys.min(dim=0).values
        w_p2f = (reg.proto_w * p2f).sum()
        f2p = d_xp_phys.min(dim=1).values.median()
        cell_ref = 1.45 * math.sqrt(float(dim)) * sigma_cell_med
        if is_point_flag:
            cur_res_t = d_xp_phys.mean().double()
        else:
            cur_res_t = ((torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro_safe).double()
        res_ema64 = 0.92 * prev_res64 + 0.08 * cur_res_t.clamp_max(10.0)
        reg._res_ema_t = res_ema64

        comp_rep: Optional[torch.Tensor] = None
        cid_i: Optional[torch.Tensor] = None
        cnt_f: Optional[torch.Tensor] = None
        sum_c: Optional[torch.Tensor] = None
        xi_i: Optional[torch.Tensor] = None
        xi_f_sum: Optional[torch.Tensor] = None
        need_vpush_reuse = (
            has_coal_flag
            and has_sigma
            and recipe.t1_iso_weight > 0.0
            and not is_point_flag
            and dim == 2
            and mu_det.shape[1] == 2
            and reg.v_push is not None
        )
        v_mu_star_all = v_mu_star_cand if (has_coal_flag and any_v_mu_star_flag and v_mu_star_cand is not None) else None
        condense_part_a = (
            v_mu_star_all is not None
            and any_coal_i_flag
            and scale_ge_065_flag
            and can_condense_table
            and not is_point_flag
            and indices is not None
        )
        if any_coal_i_flag or need_vpush_reuse or (v_mu_star_all is not None):
            comp_rep = (w_mat_all > 0.0).float().argmax(dim=1)
            cid_i = comp_rep[near_p_G]
            if any_coal_i_flag:
                if condense_part_a:
                    assert indices is not None
                    xi_i = _deterministic_box_muller(indices, dim, torch.float32).to(yr.device)
                    packed_cid = torch.cat((x_ctr_det.new_ones((b, 1)), x_ctr_det, xi_i), dim=1)
                    sums_cid = x_ctr_det.new_zeros((k_proto, 1 + 2 * dim))
                    sums_cid.index_add_(0, cid_i, packed_cid)
                    cnt_f = sums_cid[:, 0]
                    sum_c = sums_cid[:, 1 : 1 + dim]
                    xi_f_sum = sums_cid[:, 1 + dim :]
                else:
                    packed_cid = torch.cat((x_ctr_det.new_ones((b, 1)), x_ctr_det), dim=1)
                    sums_cid = x_ctr_det.new_zeros((k_proto, 1 + dim))
                    sums_cid.index_add_(0, cid_i, packed_cid)
                    cnt_f = sums_cid[:, 0]
                    sum_c = sums_cid[:, 1:]
            elif need_vpush_reuse:
                cnt_f = (cid_i.unsqueeze(0) == idx_k.unsqueeze(1)).float().sum(dim=1)

        if condense_part_a:
            assert v_mu_star_all is not None and cid_i is not None and cnt_f is not None
            assert xi_i is not None and xi_f_sum is not None
            need_conc_i = is_coal_i & (v_mu_star_all[near_p_G].amax(dim=-1) > 0.0)
            d_near_i = (xff - c_bar_all[near_p_G]).norm(dim=-1)
            halo_r = 3.5 * math.sqrt(float(dim)) * sigma_bar_all[near_p_G]
            halo_blend = ((d_near_i / halo_r.clamp_min(1e-6)).clamp(0.0, 1.0) ** 2).unsqueeze(1).to(fake_flat.dtype)
            xi_f_mean = torch.where(
                cnt_f.unsqueeze(1) >= 2.0,
                xi_f_sum / cnt_f.clamp_min(1.0).unsqueeze(1),
                torch.zeros_like(xi_f_sum),
            )
            c_exact_i = (
                c_bar_all[near_p_G] + v_mu_star_all[near_p_G].sqrt() * (xi_i - xi_f_mean[cid_i])
            ).to(fake_flat.dtype)
            y_ot_conc = halo_blend * y_ot + (1.0 - halo_blend) * c_exact_i
            y_ot = torch.where(need_conc_i.unsqueeze(1), y_ot_conc, y_ot)

        # Law of Total Variance + 0.995 Quartic-Frobenius Jensen Centering (only when coalesced!):
        raw_sig_i = sigma_bar_all[near_p_G]
        eff_raw_sig_i = raw_sig_i
        if any_coal_i_flag:
            assert cid_i is not None and cnt_f is not None and sum_c is not None
            mean_c = sum_c / cnt_f.clamp_min(1.0).unsqueeze(1)
            n_i = cnt_f[cid_i]
            valid_v = is_coal_i & (n_i >= 2.0)
            valid_idx = torch.nonzero(valid_v, as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                n_v = n_i.index_select(0, valid_idx)
                x_v = x_ctr_det.index_select(0, valid_idx)
                c_v = cid_i.index_select(0, valid_idx)
                sig_v = raw_sig_i.index_select(0, valid_idx)
                v_mu_i = (n_v / (n_v - 1.0)) * ((x_v - mean_c[c_v]).square().sum(dim=-1) / float(dim))
                rho_mu = (v_mu_i / sig_v.square().clamp_min(1e-10)).clamp(0.0, 0.75).mean()
                shrink = torch.sqrt((0.995 - rho_mu).clamp_min(0.25))
                eff_raw_sig_i = torch.where(is_coal_i, raw_sig_i * shrink, raw_sig_i)
        sigma_eff_i = eff_raw_sig_i.to(fake_flat.dtype)

        inv_w_i = (
            (1.0 / (float(k_proto) * reg.proto_w[near_p_G]).clamp_min(1e-4))
            .sqrt()
            .clamp(0.5, 4.0)
            .to(fake_flat.dtype)
        )

        # Target-isometry-referenced saturation compensation (active branch ONLY):
        w_sat_vec = torch.ones(b, device=yr.device, dtype=fake_flat.dtype)
        if has_sigma and eps_g is not None:
            e_norm_raw = eps_g.float().norm(dim=-1)
            valid_e = e_norm_raw >= 0.45
            s_obs = (fake_det - x_ctr_det).norm(dim=-1) / (sigma_z * e_norm_raw.clamp_min(0.45))
            if has_coal_flag:
                n_valid_e = valid_e.sum()
                s_sorted = torch.where(valid_e, s_obs, s_obs.new_full((), float("inf"))).sort().values
                s_med_idx = ((n_valid_e - 1).clamp_min(0) // 2).unsqueeze(0)
                s_med = s_sorted.index_select(0, s_med_idx).squeeze(0).clamp_min(1e-6)
                is_rail_coal = valid_e & (s_obs < (0.50 * s_med))
                w_boost_coal = (s_med / s_obs.clamp_min(1e-6)).clamp(1.0, 8.0)
                w_sat_vec = torch.where(is_rail_coal, w_boost_coal, torch.ones_like(w_boost_coal)).to(fake_flat.dtype)
            else:
                d_z_f = float(max(1, mu_det.shape[1]))
                s_target = (eff_raw_sig_i * math.sqrt(float(dim) / d_z_f)) / max(sigma_z, 1e-4)
                drift_d = (xff - y_ot.detach().float()).norm(dim=-1)
                is_rail_nocoal = (
                    valid_e
                    & (drift_d > (1.35 * math.sqrt(float(dim)) * raw_sig_i))
                    & (s_obs < (0.40 * s_target))
                )
                w_boost_nocoal = (0.45 * s_target / s_obs.clamp_min(1e-6)).clamp(1.0, 8.0)
                w_sat_vec = torch.where(is_rail_nocoal, w_boost_nocoal, torch.ones_like(w_boost_nocoal)).to(fake_flat.dtype)

        y_bound = yf.abs().amax(dim=0, keepdim=True) + 3.5 * sigma_cell_med

    if all_coal_flag:
        x_ot_src = x_ctr_flat
    elif not any_coal_i_flag:
        x_ot_src = fake_flat
    else:
        x_ot_src = torch.where(is_coal_i.unsqueeze(1), x_ctr_flat, fake_flat)
    sq_ot = (x_ot_src - y_ot).square().sum(dim=-1)
    sc_med_dt = sigma_cell_med.to(fake_flat.dtype)
    loss_g_ot = (w_sat_vec * (torch.sqrt(sc_med_dt.square() + sq_ot) - sc_med_dt)).mean()
    loss_env = F.relu(fake_flat.float().abs() - y_bound).square().mean().to(fake_flat.dtype)
    loss_g_ot = loss_g_ot + 2.0 * loss_env

    if has_sigma and recipe.t1_iso_weight > 0.0:
        if is_point_flag:
            loss_g_iso = (fake_flat - x_ctr_flat).square().sum(dim=-1).mean()
        else:
            d_z = mu_det.shape[1]
            prev_iso_err64 = reg._get_iso_err_ema_t(yr.device)
            if dim == 2 and d_z == 2:
                m_iso = b if has_coal_flag else min(b, 128)
                e1_u = eps_g[:m_iso] / eps_g[:m_iso].norm(dim=-1, keepdim=True).clamp_min(1e-6)
                e2_u = torch.stack((-e1_u[:, 1], e1_u[:, 0]), dim=1)
                z_u_cat = torch.cat([mu_det[:m_iso] + sigma_z * e1_u, mu_det[:m_iso] + sigma_z * e2_u], dim=0)
                x_u_cat = trainer._postprocess(
                    trainer.prior,
                    trainer._generate(trainer.G, z_u_cat, 0.0, noise),
                ).reshape(2 * m_iso, -1)
                x_u1, x_u2 = x_u_cat.chunk(2, dim=0)
                dx_0 = x_u1 - x_ctr_flat[:m_iso]
                dx_p = x_u2 - x_ctr_flat[:m_iso]
                sc_sub2 = sigma_eff_i[:m_iso]
                dx_0_n = dx_0.norm(dim=-1)
                dx_p_n = dx_p.norm(dim=-1)
                with torch.no_grad():
                    min_dx_u = torch.minimum(dx_0_n.detach(), dx_p_n.detach())
                    w_iso_sat = (sc_sub2 / min_dx_u.clamp_min(0.15 * sc_sub2)).clamp(1.0, 4.0)
                w_iso_sub = inv_w_i[:m_iso] * w_iso_sat
                iso_u1 = (w_iso_sub * ((dx_0_n - sc_sub2) / sc_med_dt).square()).mean()
                iso_u2 = (w_iso_sub * ((dx_p_n - sc_sub2) / sc_med_dt).square()).mean()

                if has_coal_flag:
                    M_dx2 = torch.stack((dx_0, dx_p), dim=-1)
                    Sigma_push2 = M_dx2 @ M_dx2.transpose(1, 2)
                    eye2 = reg._get_eye2(yr.device, fake_flat.dtype)
                    target_cov2 = (sc_sub2.square().view(m_iso, 1, 1)) * eye2
                    denom_cov2 = (sc_sub2 * sc_med_dt).clamp_min(1e-8).view(m_iso, 1, 1)
                    with torch.no_grad():
                        if reg.v_push is not None:
                            assert comp_rep is not None and cid_i is not None and cnt_f is not None
                            vp_diag = dx_0.detach().float().square() + dx_p.detach().float().square()
                            sum_vp = vp_diag.new_zeros((k_proto, dim))
                            sum_vp.index_add_(0, cid_i, vp_diag)
                            cnt_k = cnt_f[comp_rep]
                            hit_k = (cnt_k > 0) & multicell_coal
                            mean_vp_k = sum_vp[comp_rep] / cnt_k.clamp_min(1.0).unsqueeze(1)
                            reg.v_push = torch.where(hit_k.unsqueeze(1), 0.88 * reg.v_push + 0.12 * mean_vp_k, reg.v_push)
                        t_tr1 = sc_sub2.detach().float().square().clamp_min(1e-10).view(m_iso, 1, 1)
                        iso_frob_i = ((Sigma_push2.detach().float() / t_tr1) - eye2.float()).square().sum(dim=(1, 2)).sqrt()
                        cur_frob64 = 0.60 * iso_frob_i.mean().double() + 0.40 * _quantile_1d_fast(iso_frob_i, 0.92).double()
                        reg._iso_err_ema_t = 0.92 * prev_iso_err64 + 0.08 * cur_frob64.clamp_max(5.0)
                    iso_cov2 = (
                        w_iso_sub * ((Sigma_push2 - target_cov2) / denom_cov2).square().sum(dim=(1, 2))
                    ).mean() * 0.5
                    loss_g_iso = 0.45 * (iso_u1 + iso_u2) + 0.55 * iso_cov2
                else:
                    with torch.no_grad():
                        eps_norm = eps_g.norm(dim=-1).clamp_min(0.25)
                        target_dx_norm = sigma_eff_i * eps_norm
                        rel_iso_err = (((fake_flat.detach() - x_ctr_flat.detach()).norm(dim=-1) - target_dx_norm) / sc_med_dt.clamp_min(1e-6)).abs()
                        reg._iso_err_ema_t = 0.94 * prev_iso_err64 + 0.06 * rel_iso_err.mean().clamp_max(5.0).double()
                    cos_sq = ((dx_0 * dx_p).sum(dim=-1) / (dx_0_n * dx_p_n + 1e-6)).square().mean()
                    loss_g_iso = 0.5 * (iso_u1 + iso_u2) + 0.25 * cos_sq
            else:
                dx_ker = fake_flat - x_ctr_flat
                scale_dim = math.sqrt(float(dim) / float(d_z))
                eps_norm = eps_g.norm(dim=-1).clamp_min(0.25)
                target_dx_norm = sigma_eff_i * scale_dim * eps_norm
                rel_iso_err = ((dx_ker.norm(dim=-1) - target_dx_norm) / (sc_med_dt * scale_dim).clamp_min(1e-6)).abs()
                with torch.no_grad():
                    cur_iso_err64 = rel_iso_err.detach().mean().clamp_max(5.0).double()
                    reg._iso_err_ema_t = 0.94 * prev_iso_err64 + 0.06 * cur_iso_err64
                iso_1 = (inv_w_i * rel_iso_err.square()).mean()
                if dim == 2 and d_z == 4:
                    m_iso = min(b, 64)
                    es = eps_g[:m_iso]
                    ms = mu_det[:m_iso]
                    xcs = x_ctr_flat[:m_iso]
                    a, b_c, c_c, d_c = es[:, 0], es[:, 1], es[:, 2], es[:, 3]
                    comps = (
                        torch.stack((-b_c, a, -d_c, c_c), dim=1),
                        torch.stack((-c_c, d_c, a, -b_c), dim=1),
                        torch.stack((-d_c, -c_c, b_c, a), dim=1),
                    )
                    dx_all = [dx_ker[:m_iso]]
                    for ec in comps:
                        x_ec = trainer._postprocess(
                            trainer.prior,
                            trainer._generate(trainer.G, ms + sigma_z * ec, 0.0, noise),
                        ).reshape(m_iso, -1)
                        dx_all.append(x_ec - xcs)
                    M_dx = torch.stack(dx_all, dim=-1)
                    eps_sq = es.square().sum(dim=-1).clamp_min(0.15).view(m_iso, 1, 1)
                    Sigma_push = (M_dx @ M_dx.transpose(1, 2)) / eps_sq
                    sc_sub = sigma_eff_i[:m_iso]
                    eye2 = reg._get_eye2(yr.device, fake_flat.dtype)
                    target_cov = (sc_sub.square().view(m_iso, 1, 1)) * eye2
                    denom_cov = (sc_sub * sc_med_dt).clamp_min(1e-8).view(m_iso, 1, 1)
                    iso_cov = (inv_w_i[:m_iso] * ((Sigma_push - target_cov) / denom_cov).square().sum(dim=(1, 2))).mean() * 0.5
                    loss_g_iso = 0.5 * iso_1 + 0.5 * iso_cov
                else:
                    loss_g_iso = iso_1
    else:
        loss_g_iso = fake_flat.new_zeros(())

    transport = (
        recipe.kinetic_transport_loss(fake_g, real_g)
        if recipe.kinetic_transport_weight
        else loss_gan.new_zeros(())
    )
    transport_local = (
        recipe.kinetic_transport_local_loss(fake_g, real_g)
        if recipe.kinetic_transport_local_weight
        else loss_gan.new_zeros(())
    )

    with torch.no_grad():
        if has_coal_flag:
            eq_gate64 = ((0.030 - res_ema64) / 0.020).clamp(0.0, 1.0)
            adv_w64 = 0.038 * (1.0 - eq_gate64) + 0.008 * eq_gate64
            g_floor64 = 0.022 * (1.0 - eq_gate64) + 0.012 * eq_gate64
            row_floor64 = 0.15 * (1.0 - eq_gate64) + 0.38 * eq_gate64
            adv_w_dt = adv_w64.to(loss_gan.dtype)
        else:
            adv_w_dt = 0.15
            g_floor64 = res_ema64.new_full((), 0.04)
            row_floor64 = res_ema64.new_full((), 0.15)

    loss_g = (
        adv_w_dt * loss_gan
        + float(recipe.t1_pullback_weight) * loss_g_ot
        + float(recipe.t1_iso_weight) * loss_g_iso
    )
    if base_prior.z.requires_grad and recipe.prior_l2 > 0:
        loss_g = loss_g + recipe.prior_l2 * base_prior.z.square().mean()

    prior_manip_loss = None
    if trainer._prior_penalty:
        raw_rows = base_prior.z if recipe.num_particles <= 1024 else base_prior.z[torch.unique(indices)]
        prior_manip_loss = trainer.prior.regularize(
            raw_rows,
            sampled_codes=latent,
            indices=indices,
            generated=fake_g,
            logits=fake_logits,
        )
        loss_g = loss_g + prior_manip_loss
    if recipe.kinetic_transport_weight:
        loss_g = loss_g + transport
    if recipe.kinetic_transport_local_weight:
        loss_g = loss_g + transport_local

    if rec_d is not None and hasattr(rec_d, "_t1c_disc_gate"):
        rec_d._t1c_disc_gate = 0.0

    trainer.opt_g.zero_grad()
    protected_evaluator = None
    if recipe.constraint_geometry_mode == "strict_progress":
        def protected_evaluator() -> Tuple[torch.Tensor]:
            rec = getattr(trainer.opt_d, "record", None)
            prev_probe = getattr(rec, "_in_protected_probe", False) if rec is not None else False
            if rec is not None:
                rec._in_protected_probe = True
            try:
                generated = trainer._postprocess(
                    trainer.prior,
                    trainer._generate(trainer.G, latent_det if has_sigma else mu_det, sigma_out, noise, rows=indices),
                )
                logits = critic(generated)
                return (F.softplus(-logits).mean(),)
            finally:
                if rec is not None:
                    rec._in_protected_probe = prev_probe

    if isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer) and trainer.opt_g.direction_blend:
        loss_g.backward()
        trainer.opt_g.bind_protected_from_grads()
    else:
        constraint_geometry_backward(
            loss_g,
            trainer.opt_g,
            (loss_g,),
            protected_evaluator=protected_evaluator,
        )
    if base_prior.z.requires_grad and base_prior.z.grad is None:
        base_prior.z.grad = torch.zeros_like(base_prior.z)

    trainer.policy.after_generator_backward(
        loss_gan=loss_gan.detach(),
        loss_critic=(loss_d - penalty).detach(),
    )

    # Keep generator active when boundary tanh modes still exhibit high relative isometry distortion:
    with torch.no_grad():
        if is_point_flag:
            g_scale_t = res_ema64.new_full((), 1.0)
        else:
            iso_err64 = reg._get_iso_err_ema_t(yr.device)
            iso_gate64 = ((iso_err64 - 0.22) / 0.45).clamp(0.0, 1.0)
            g_scale_norm64 = torch.maximum(
                g_floor64,
                torch.minimum(res_ema64.new_full((), 1.0), torch.maximum(res_ema64 / 0.10, 0.28 * iso_gate64)),
            )
            g_scale_t = torch.where(scale_ratio_t < 0.60, res_ema64.new_full((), 1.0), g_scale_norm64)

    # Capture deduplicated u_rows before opt_g.step() clears _sampled_rows (avoids duplicate torch.unique sync!):
    pre_u_rows: Optional[torch.Tensor] = None
    if (
        can_condense_table
        and base_prior.z.requires_grad
        and float(recipe.t1_prior_row_lr) > 0.0
        and not is_point_flag
        and isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer)
        and not trainer.opt_g._supports_zerosync_v4_tensors()
    ):
        cand_rows = trainer.opt_g._sampled_rows.get(base_prior.z)
        if cand_rows is not None and cand_rows.device == indices.device:
            pre_u_rows = cand_rows

    if isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer):
        trainer.opt_g.bind_v4_modulation(g_scale=g_scale_t, wakeup_boost=wakeup_boost_t, prior_scale=1.0)
        if trainer._gradient_response is not None:
            params = list(trainer.G.parameters())
            grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
            gain = trainer._gradient_response.observe(grads)
            for p, m in zip(params, trainer._gradient_response.packet["means"]):
                p.grad = (m / trainer._gradient_response.packet["weight"]).to(p).clone()
            trainer.opt_g.bind_v4_modulation(g_scale=g_scale_t * gain, wakeup_boost=wakeup_boost_t, prior_scale=1.0)
            trainer.opt_g.step()
            trainer._gradient_response.response["applied_steps"] += 1
        else:
            trainer.opt_g.step()
    else:
        g_scale = float(g_scale_t.item())
        saved_lrs = [grp["lr"] for grp in trainer.opt_g.param_groups]
        for grp in trainer.opt_g.param_groups:
            grp["lr"] = grp["lr"] * g_scale
        try:
            if trainer._gradient_response is not None:
                params = list(trainer.G.parameters())
                grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
                gain = trainer._gradient_response.observe(grads)
                for p, m in zip(params, trainer._gradient_response.packet["means"]):
                    p.grad = (m / trainer._gradient_response.packet["weight"]).to(p).clone()
                for group in trainer.opt_g.param_groups:
                    if group.get("role") == "generator":
                        group["lr"] = group["lr"] * gain
                trainer.opt_g.step()
                trainer._gradient_response.response["applied_steps"] += 1
            else:
                trainer.opt_g.step()
        finally:
            for grp, lr0 in zip(trainer.opt_g.param_groups, saved_lrs):
                grp["lr"] = lr0

    trainer.policy.after_generator_step()

    # --- PART B: Kantorovich-Laguerre Mass-Balanced Island Condensation + Outward Dead-Clamp Recovery ---
    prior_reg_diag = loss_gan.new_zeros(())
    if can_condense_table and base_prior.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0 and not is_point_flag:
        u_rows = pre_u_rows if pre_u_rows is not None else torch.unique(indices)
        g_flags = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)

            n_u = len(u_rows)
            n_eff = float(n_u)

            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[u_rows]
            x_u = trainer._postprocess(
                trainer.prior,
                trainer._generate(trainer.G, c_u, 0.0, noise),
            ).reshape(n_u, -1)

            with torch.no_grad():
                xu_det = x_u.detach().float()
                psi_eff = w_mat_all @ reg.psi
                d2_up = torch.cdist(xu_det, c_bar_all, p=2.0).square()
                k_star = (d2_up - psi_eff.unsqueeze(0)).argmin(dim=1)

                w_hard_u = w_mat_all[k_star].mean(dim=0)
                reg.w_fake_ema = 0.88 * reg.w_fake_ema + 0.12 * w_hard_u
                w_feedback = 0.35 * w_hard_u + 0.65 * reg.w_fake_ema

                k_float = float(k_proto)
                # Float64 scalar arithmetic matching exact Python-float Laguerre update (0 syncs):
                tv_now64 = 0.5 * (reg.proto_w - reg.w_fake_ema).abs().sum().double()
                tv_floor = 0.35 * math.sqrt(k_float / max(n_eff * 8.0, 1.0))
                psi_gate64 = ((tv_now64 - tv_floor) / max(0.08, tv_floor)).clamp(0.0, 1.0)

                eta_scale = min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * k_float))
                r_mac_sq64 = r_macro_safe.square().double()
                eta_psi64 = ((psi_gate64 * eta_scale) * float(dim)) * r_mac_sq64
                psi_next = reg.psi * (1.0 - 0.008 * psi_gate64).float()
                psi_next = torch.addcmul(psi_next, reg.proto_w - w_feedback, eta_psi64.float())
                psi_next = w_mat_all @ psi_next
                psi_next = psi_next - psi_next.mean()
                psi_cap_t = (2.5 * float(dim) * r_mac_sq64 + 0.5).float()
                reg.psi = torch.clamp(psi_next, -psi_cap_t, psi_cap_t)

                c_bar_u = c_bar_all[k_star]
                if v_mu_star_all is not None:
                    assert comp_rep is not None
                    coal_u = multicell_coal[k_star]
                    xi_u = _deterministic_box_muller(u_rows, dim, torch.float32).to(yr.device)
                    cid_u = comp_rep[k_star]
                    packed_u = torch.cat((xi_u.new_ones((n_u, 1)), xi_u), dim=1)
                    sums_u = xi_u.new_zeros((k_proto, 1 + dim))
                    sums_u.index_add_(0, cid_u, packed_u)
                    cnt_u = sums_u[:, 0]
                    xi_u_sum = sums_u[:, 1:]
                    xi_u_mean = torch.where(
                        cnt_u.unsqueeze(1) >= 2.0,
                        xi_u_sum / cnt_u.clamp_min(1.0).unsqueeze(1),
                        torch.zeros_like(xi_u_sum),
                    )
                    c_conc_u = c_bar_u + v_mu_star_all[k_star].sqrt() * (xi_u - xi_u_mean[cid_u])
                    c_target = torch.where(coal_u.unsqueeze(1), c_conc_u, c_bar_u).to(x_u.dtype)
                else:
                    c_target = c_bar_u.to(x_u.dtype)
                sc_u = sigma_bar_all[k_star].to(x_u.dtype)
                row_settle64 = torch.maximum(
                    row_floor64,
                    torch.minimum(res_ema64.new_full((), 1.0), torch.maximum(res_ema64 / 0.10, psi_gate64)),
                )

            sq_u = (x_u - c_target).square().sum(dim=-1)
            loss_mu_condense = (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u).mean()

            mu_raw_u = base_prior.z[u_rows]
            if has_sigma and n_u >= 2:
                mu_c = mu_raw_u - mu_raw_u.mean(dim=0, keepdim=True)
                mu_std = (mu_c.square().mean(dim=0) + 1e-6).sqrt()
                iso_ema_dt = reg._get_iso_ema_t(yr.device).to(mu_std.dtype)
                loss_mu_iso = F.relu(iso_ema_dt - mu_std).square().mean()
            else:
                loss_mu_iso = x_u.new_zeros(())

            loss_mu_total = loss_mu_condense + 0.15 * loss_mu_iso
            prior_reg_diag = loss_mu_total.detach()

            if base_prior.z.grad is not None:
                base_prior.z.grad.zero_()
            loss_mu_total.backward()

            if not reg.rows_initialized or reg.row_m is None or reg.row_m.shape != base_prior.z.shape:
                reg.row_m = torch.zeros_like(base_prior.z)
                reg.row_v = torch.zeros(len(base_prior.z), 1, device=base_prior.z.device, dtype=base_prior.z.dtype)
                reg.row_n = torch.zeros(len(base_prior.z), 1, device=base_prior.z.device, dtype=base_prior.z.dtype)
                reg.rows_initialized = True

            with torch.no_grad():
                g_rows = (base_prior.z.grad[u_rows] * n_eff).clone()
                base_prior.z.grad.zero_()
                dist_u = sq_u.detach().float().sqrt()
                dist_i = dist_u.to(sq_u.dtype).unsqueeze(1)
                part_scale = (dist_i / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2)

                # Coordinatewise outward column-sensitivity clamp rescue (rescues single- and multi-axis
                # dead-clamped outer latent shells on tanh_rail_guard with strictly 0 autograd.grad!):
                far_mask_1d = dist_u > (2.8 * math.sqrt(float(dim)) * sc_u.float())
                d_z_u = c_u.shape[1]
                if d_z_u <= 8 and bool(far_mask_1d.any()):
                    far_idx = torch.nonzero(far_mask_1d, as_tuple=False).squeeze(-1)
                    in_idx = torch.nonzero(~far_mask_1d, as_tuple=False).squeeze(-1)[:64]
                    if in_idx.numel() < 4:
                        in_idx = torch.arange(min(64, n_u), device=c_u.device, dtype=torch.long)
                    probe_idx = torch.cat([far_idx, in_idx], dim=0)
                    c_probe = c_u[probe_idx].detach()
                    x_probe = xu_det[probe_idx]
                    out_sgn = torch.where(c_probe >= 0.0, 1.0, -1.0)
                    eye_z = torch.eye(d_z_u, device=c_u.device, dtype=c_u.dtype)
                    c_pert = (c_probe.unsqueeze(1) + (1e-3 * out_sgn).unsqueeze(2) * eye_z.unsqueeze(0)).reshape(
                        -1, d_z_u
                    )
                    x_pert = (
                        trainer._postprocess(
                            trainer.prior,
                            trainer._generate(trainer.G, c_pert, 0.0, noise),
                        )
                        .float()
                        .reshape(len(probe_idx), d_z_u, dim)
                    )
                    col_sens = (x_pert - x_probe.unsqueeze(1)).norm(dim=-1) / 1e-3
                    s_far = col_sens[: len(far_idx)]
                    s_in = col_sens[len(far_idx) :]
                    s_ref_col = torch.quantile(s_in, 0.5, dim=0, keepdim=True).clamp_min(1e-6)
                    dead_far = s_far < (0.04 * s_ref_col)
                    if bool(dead_far.any()):
                        mu_ref = c_u[in_idx].detach().mean(dim=0, keepdim=True)
                        z_far = mu_raw_u[far_idx].detach()
                        gu_far = g_rows[far_idx]
                        g_mag = torch.maximum(
                            gu_far.float().norm(dim=-1, keepdim=True),
                            torch.ones_like(z_far[:, :1], dtype=torch.float32),
                        )
                        inward_grad = (torch.sign(z_far - mu_ref).float() * g_mag).to(g_rows.dtype)
                        g_rows[far_idx] = torch.where(dead_far, inward_grad, gu_far)
                        any_dead_far = dead_far.any(dim=1).unsqueeze(1)
                        part_scale[far_idx] = torch.where(
                            any_dead_far, torch.ones_like(part_scale[far_idx]), part_scale[far_idx]
                        )

                m = reg.row_m[u_rows] * 0.80 + 0.20 * g_rows
                v = reg.row_v[u_rows] * 0.80 + 0.20 * g_rows.square().sum(dim=-1, keepdim=True)
                n = reg.row_n[u_rows] + 1.0
                reg.row_m[u_rows] = m
                reg.row_v[u_rows] = v
                reg.row_n[u_rows] = n
                bc = 1.0 - (0.80 ** n)
                step_vec = (m / bc) / ((v / bc).sqrt() + 1e-5)
                row_alpha = ((-float(recipe.t1_prior_row_lr)) * row_settle64).to(base_prior.z.dtype)
                base_prior.z.index_add_(
                    0,
                    u_rows,
                    (step_vec * part_scale) * row_alpha,
                    alpha=1.0,
                )
        finally:
            for p, flag in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(flag)

    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


__all__ = [
    "UnifiedV15PriorRegularizer",
    "should_use_unified_v15_pullback",
    "unified_v15_pullback_step",
]
