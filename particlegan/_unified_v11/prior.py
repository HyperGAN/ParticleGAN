"""Unified-v11 4-Law Pullback Prior Dynamics: 0.995 Jensen-Centered Total-Variance + fwAD Rail/Clamp Recovery.

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
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._stats import knn_excluding_self, pairwise_sq_dist
from .._t4_arch.pullback import _deterministic_box_muller
from .._unified.prior import (
    _compute_pythagorean_smoothed_cells,
    _validate_unified_prior_state,
)
from .._bispf.thermal import FROZEN_THERMAL_SCALE, _read_thermal
from ..bispf import BISPFGates, SBNPPriorRegularizer
from ..moment_tether import pooled_reference_moments
from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .loss import _principal_axis_power_zerosync
from .optimizer import UnifiedV11EquilibratedOptimizer

_PHI1 = 0.6180339887498949


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
        r_med_sq = _median_offdiag_zerosync(sq_dist).clamp_min(1e-12)
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
) -> Tuple[torch.Tensor, torch.Tensor, bool, bool]:
    """Zero-sync extraction of ``(mu_g, sigma_z_t, has_sigma, can_condense_table)``."""
    base_prior = unwrap_prior(prior)
    if getattr(base_prior, "code_law", "table") != "table":
        return latent.detach(), latent.new_zeros((), dtype=torch.float32), False, False
    transforms = bool(getattr(prior, "transforms_codes", False))
    if not transforms:
        if isinstance(base_prior, MoGParticlePrior):
            has_sigma = bool(getattr(base_prior, "_noise_enabled", False))
            sigma_t = base_prior.sigma.detach().float().reshape(())
            mu_g = base_prior.means()[indices]
            return mu_g, sigma_t, has_sigma, True
        return base_prior.z[indices], latent.new_zeros((), dtype=torch.float32), False, True
    manips = tuple(getattr(prior, "manipulators", ()))
    if (
        len(manips) == 1
        and type(manips[0]).__name__ == "NormalNoiseManipulator"
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
    ):
        sig_f = float(manips[0].sigma)
        return base_prior.z[indices], latent.new_full((), sig_f, dtype=torch.float32), sig_f > 0.0, True
    return latent.detach(), latent.new_zeros((), dtype=torch.float32), False, False


def _validate_unified_v11_prior_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("UnifiedV11PriorRegularizer checkpoint must be a mapping")
    if state.get("schema") != 2:
        raise ValueError("unsupported UnifiedV11PriorRegularizer checkpoint schema")
    if "v_push" not in state:
        raise ValueError("UnifiedV11PriorRegularizer checkpoint is missing 'v_push'")
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
            raise ValueError("invalid UnifiedV11PriorRegularizer v_push tensor")
    elif v_push is not None:
        raise ValueError("uninitialized UnifiedV11PriorRegularizer checkpoint must have v_push=None")
    return {**validated, "schema": 2, "v_push": v_push}


class UnifiedV11PriorRegularizer(nn.Module):
    """Recipe-owned Unified-v11 prior regularizer (``prior_regularizer='unified_v11'``)."""

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
        self.unified_v11_record: Any = None

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
        self.rows_initialized: bool = False
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None

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
        fb = self.fallback
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if fb.weight == 0.0 or not len(rows):
            return rows[:0].sum()
        gates = fb.gates
        n, d = rows.shape
        min_dof = fb._min_dof(d)

        if fb.references:
            initial_mean, initial_covariance, count_t = pooled_reference_moments(fb.references)
            ref_key = tuple(id(r) for r in fb.references)
            if self._cached_ref_key != ref_key or self._cached_ref_total_n is None:
                self._cached_ref_key = ref_key
                self._cached_ref_total_n = int(count_t.item())
            total_n = self._cached_ref_total_n
            if total_n <= min_dof or n <= min_dof or n < 2:
                return rows[:0].sum()
            if d != len(initial_mean):
                raise ValueError("sbnp rows and initial reference dimensions differ")
            mean = rows.mean(0)
            variance_scale = initial_covariance.diagonal().clamp_min(0) + fb.eps
            mean_error = ((mean - initial_mean).square() / variance_scale).mean()
            target_sigma = (initial_covariance.diagonal().clamp_min(0) + fb.eps).sqrt() * fb.target_std
            target_cov = initial_covariance
            target_var = variance_scale
        else:
            if n <= min_dof or n < 2:
                return rows[:0].sum()
            mean = rows.mean(0)
            mean_error = rows.new_zeros(())
            target_sigma = torch.full((d,), fb.target_std, device=rows.device, dtype=rows.dtype)
            target_var = target_sigma.pow(2) + fb.eps
            target_cov = torch.diag(target_var)

        centered = rows - mean
        std = (centered.pow(2).mean(0) + fb.eps).sqrt()
        var_loss = F.relu(target_sigma - std).pow(2).mean()

        if d > 1:
            cov = (centered.T @ centered) / max(n - 1, 1)
            if (
                fb.references
                and gates.active("sbnp_full_cov")
                and d <= gates.sbnp_full_cov_max_dim
                and n <= gates.sbnp_full_cov_max_rows
            ):
                cov_diff = cov - target_cov
                cov_loss = (cov_diff.pow(2) / (target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6)).sum()
            else:
                off_diag = cov - torch.diag(torch.diag(cov))
                cov_loss = (off_diag.pow(2) / (target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6)).sum() / (d * d)
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
            return out
        if global_moments.requires_grad and s < 1.0:
            global_moments = global_moments * s + global_moments.detach() * (1.0 - s)
        base_loss = global_moments + eff_repel_weight * repel_loss + local_iso_weight * local_iso_loss
        return scale_w * base_loss

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        out = (
            self._sbnp_fallback_zerosync(rows)
            if type(self.fallback) is SBNPPriorRegularizer
            else self.fallback(rows)
        )
        if self.weight == 0.0 or rows.ndim != 2 or not len(rows):
            return out
        n, d = rows.shape
        if not self._active_iso_enabled or d < 2:
            return out
        m_sub = min(n, self.fallback.repel_max_samples)
        if m_sub <= self._active_iso_above or n <= self.fallback._min_dof(d):
            return out
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
                v1 = _principal_axis_power_zerosync(cdet, iters=6)
                lam1_d = (v1 * (cdet @ v1)).sum().clamp_min(0.0)
                cdef = cdet - lam1_d * torch.outer(v1, v1)
                v2 = _principal_axis_power_zerosync(cdef, iters=6)
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
        validated = _validate_unified_v11_prior_state(state_dict)
        if validated["rows_initialized"] and self._bound_shape is not None:
            if tuple(validated["row_m"].shape) != tuple(self._bound_shape):
                raise ValueError("UnifiedV11PriorRegularizer row_m shape does not match bound prior table")
        if validated["quantizer_initialized"] and validated["proto_c"].shape[0] > self.max_prototypes:
            raise ValueError("UnifiedV11PriorRegularizer proto_c count exceeds configured max_prototypes")

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

        b_cnt = torch.zeros(k, device=x.device, dtype=torch.float32).scatter_add_(
            0, assign, torch.ones(b, device=x.device, dtype=torch.float32)
        )
        b_w = b_cnt / float(b)

        raw_spread = 2.0 * x.var(dim=0, unbiased=False).sum()
        has_spread_t = raw_spread > 1e-10
        pair_spread = raw_spread.clamp_min(1e-12)
        r2p_t = torch.where(has_spread_t, torch.sqrt(d2_min.mean() / pair_spread), raw_spread.new_zeros(()))
        shock_t = ((r2p_t - 0.45) / 0.35).clamp(0.0, 1.0)
        self.proto_hits = torch.where(
            shock_t > 0.25,
            (self.proto_hits * (1.0 - 0.8 * shock_t)).clamp_min(1.0),
            self.proto_hits,
        )

        alpha_w_t = 0.02 + 0.48 * (shock_t * shock_t)
        self.proto_w = (1.0 - alpha_w_t) * self.proto_w + alpha_w_t * b_w
        self.proto_w = self.proto_w / self.proto_w.sum().clamp_min(1e-12)

        hit = b_cnt > 0
        b_cnt_safe = b_cnt.clamp_min(1.0)
        resid_sq_prev = (x - c_prev[assign]).square().sum(dim=1) / float(dim)
        b_vsum = torch.zeros(k, device=x.device, dtype=torch.float32).scatter_add_(0, assign, resid_sq_prev)
        v_est = (b_vsum / b_cnt_safe).clamp_min(1e-10)
        self.proto_var = torch.where(hit, 0.88 * self.proto_var + 0.12 * v_est, self.proto_var)

        b_sum = torch.zeros((k, dim), device=x.device, dtype=torch.float32).index_add_(0, assign, x)
        b_mean = b_sum / b_cnt_safe.unsqueeze(1)
        new_hits = (self.proto_hits + b_cnt).clamp_max(2000.0)
        self.proto_hits = torch.where(hit, new_hits, self.proto_hits)
        base_c = (0.035 + 0.25 / torch.sqrt(self.proto_hits)) + 0.38 * (shock_t * shock_t)
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
                wtot = (wa + wb).clamp_min(1e-12)
                ca = self.proto_c.index_select(0, a_1)
                cb = self.proto_c.index_select(0, b_1)
                self.proto_c = torch.where((do_merge & is_a).unsqueeze(1), (wa * ca + wb * cb) / wtot, self.proto_c)
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


def should_use_unified_v11_pullback(
    trainer: Any,
    real: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Intrinsic geometric predicate for Unified-v11 Pullback + Rail-Escape + Laguerre Condensation.

    Active iff unconditional Euclidean vector space ``real.ndim == 2`` (`encoder_mode == 'none'`,
    `conditioning == 'scalar'`) and EITHER:
    - Case A: Point-mass / rail-escape target (`Var(real) < 1e-11`, `1 <= D <= 16`, any `sigma_z >= 0`), OR
    - Case B: Multi-D continuous-kernel Mixture-of-Gaussians (`2 <= D <= 16` and `sigma_z > 0`).
    """
    del latent, indices
    recipe = trainer.recipe
    if recipe.prior_regularizer != "unified_v11" or float(recipe.t1_pullback_weight) <= 0.0:
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
    reg: UnifiedV11PriorRegularizer = trainer.prior_regularizer
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


def unified_v11_pullback_step(
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
    """Execute Unified-v11 4-Law Pullback + `0.995` Jensen-Centered Isometry + Rail/Dead-Clamp Recovery."""
    reg: UnifiedV11PriorRegularizer = trainer.prior_regularizer
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

    mu_g, sigma_z_t, has_sigma, can_condense_table = _resolve_v11_prior_kernel(trainer.prior, latent, indices)
    sigma_z = float(sigma_z_t.item()) if has_sigma else 0.0
    has_sigma = sigma_z > 0.0

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
    gen_has_bn = bool(getattr(trainer.G, "_unified_v11_gen_has_bn", False))

    with torch.no_grad():
        reg.update_quantizer(yr)
        yf = yr.float()
        r_macro = yf.std(dim=0, unbiased=False).mean()
        is_point_t = r_macro < 1e-6
        r_macro_safe = r_macro.clamp_min(1e-4)

        c_bar_all, sigma_bar_all, w_mat_all, multicell_coal = _compute_pythagorean_smoothed_cells(
            reg.proto_c, reg.proto_var, reg.proto_w
        )
        k_proto = len(c_bar_all)
        sigma_cell_med = torch.quantile(sigma_bar_all, 0.35).clamp_min(1e-6)
        if has_sigma:
            iso_t = (sigma_z * r_macro_safe / sigma_cell_med).double().clamp(1.0, 4.5)
            one64 = iso_t.new_full((), 1.0)
            if not reg.iso_initialized:
                reg._iso_ema_t = torch.where(is_point_t, one64, iso_t)
                reg.iso_initialized = True
            else:
                prev_iso = reg._get_iso_ema_t(yr.device)
                reg._iso_ema_t = torch.where(is_point_t, one64, 0.95 * prev_iso + 0.05 * iso_t)
        else:
            reg.iso_ema = 1.0
            reg.iso_initialized = True

    with torch.no_grad():
        near_p_G = torch.cdist(x_ctr_flat.detach().float(), c_bar_all, p=2.0).argmin(dim=1)
        is_coal_i = multicell_coal[near_p_G]
        xff = torch.where(is_coal_i.unsqueeze(1), x_ctr_flat.detach().float(), fake_flat.detach().float())

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

        d_xp_phys = torch.cdist(xff, c_bar_all, p=2.0)
        if k_proto >= 4:
            xff_w = (xff - xff.mean(dim=0, keepdim=True)) / xff.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            cb_mean = (reg.proto_w.unsqueeze(1) * c_bar_all).sum(dim=0, keepdim=True)
            cb_std = ((reg.proto_w.unsqueeze(1) * (c_bar_all - cb_mean).square()).sum(dim=0, keepdim=True)).sqrt().clamp_min(1e-6)
            cb_w = (c_bar_all - cb_mean) / cb_std
            d_xp_w = torch.cdist(xff_w, cb_w, p=2.0)
            d_xp_route = torch.where(scale_ratio_t < 0.35, d_xp_w, d_xp_phys)
        else:
            d_xp_route = d_xp_phys

        c_scale = d_xp_route.mean().clamp_min(1e-5)
        prev_res64 = reg._get_res_ema_t(yr.device)
        eps_s_t = (0.25 * prev_res64.float()).clamp(0.010, 0.06)
        L_xp = -(d_xp_route / c_scale) / eps_s_t
        log_a = -math.log(float(b))
        log_b = torch.log(reg.proto_w.clamp_min(1e-12))
        u = torch.zeros(b, device=yr.device, dtype=torch.float32)
        v = torch.zeros(k_proto, device=yr.device, dtype=torch.float32)
        for _ in range(10):
            v = log_b - torch.logsumexp(L_xp + u.unsqueeze(1), dim=0)
            u = log_a - torch.logsumexp(L_xp + v.unsqueeze(0), dim=1)
        P_xp = torch.exp(L_xp + u.unsqueeze(1) + v.unsqueeze(0))
        row_p = (float(b) * P_xp).clamp_min(0.0)
        row_p = row_p / row_p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        y_ot = (row_p @ c_bar_all).to(fake_flat.dtype)

        p2f = d_xp_phys.min(dim=0).values
        w_p2f = (reg.proto_w * p2f).sum()
        f2p = d_xp_phys.min(dim=1).values.median()
        cell_ref = 1.45 * math.sqrt(float(dim)) * sigma_cell_med
        cur_res_t = torch.where(
            is_point_t,
            d_xp_phys.mean(),
            (torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro_safe,
        ).double()
        res_ema64 = 0.92 * prev_res64 + 0.08 * cur_res_t.clamp_max(10.0)
        reg._res_ema_t = res_ema64
        res_ema_f = res_ema64.float()

        has_multicell_coal_t = multicell_coal.any()
        comp_rep = (w_mat_all > 0.0).float().argmax(dim=1)
        ones_b = torch.ones(b, device=yr.device, dtype=torch.float32)
        cid_i = comp_rep[near_p_G]
        cnt_f = torch.zeros(k_proto, device=yr.device, dtype=torch.float32).scatter_add_(0, cid_i, ones_b)

        v_mu_star_all = torch.zeros_like(c_bar_all)
        if reg.v_push is not None:
            v_bar_sq = sigma_bar_all.square().unsqueeze(1)
            v_push_bar = (w_mat_all @ reg.v_push).clamp_min(0.0)
            deficit_ratio = ((v_bar_sq - v_push_bar) / v_bar_sq.clamp_min(1e-10)).clamp(0.0, 1.0)
            u_def = ((deficit_ratio - 0.15) / 0.12).clamp(0.0, 1.0)
            gate_def = u_def * u_def * (3.0 - 2.0 * u_def)
            v_mu_star_cand = gate_def * (v_bar_sq - v_push_bar).clamp_min(0.0)
            v_mu_star_all = torch.where(has_multicell_coal_t, v_mu_star_cand, v_mu_star_all)
            if can_condense_table and indices is not None:
                need_conc_i = (
                    has_multicell_coal_t
                    & is_coal_i
                    & (v_mu_star_all[near_p_G].amax(dim=-1) > 0.0)
                    & (scale_ratio_t >= 0.65)
                    & (~is_point_t)
                )
                d_near_i = (xff - c_bar_all[near_p_G]).norm(dim=-1)
                halo_r = 3.5 * math.sqrt(float(dim)) * sigma_bar_all[near_p_G]
                halo_blend = ((d_near_i / halo_r.clamp_min(1e-6)).clamp(0.0, 1.0) ** 2).unsqueeze(1).to(fake_flat.dtype)
                xi_i = _deterministic_box_muller(indices, dim, torch.float32).to(yr.device)
                xi_f_sum = torch.zeros((k_proto, dim), device=yr.device, dtype=torch.float32).index_add_(
                    0, cid_i, xi_i
                )
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

        # Law of Total Variance + 0.995 Quartic-Frobenius Jensen Centering (0 sync):
        raw_sig_i = sigma_bar_all[near_p_G]
        xcf_det = x_ctr_flat.detach().float()
        sum_c = torch.zeros((k_proto, dim), device=yr.device, dtype=torch.float32).index_add_(0, cid_i, xcf_det)
        mean_c = sum_c / cnt_f.clamp_min(1.0).unsqueeze(1)
        n_i = cnt_f[cid_i]
        valid_v = is_coal_i & (n_i >= 2.0)
        valid_cnt = valid_v.sum()
        n_safe = torch.where(valid_v, n_i, n_i.new_full((), 2.0))
        v_mu_all = (n_safe / (n_safe - 1.0)) * ((xcf_det - mean_c[cid_i]).square().sum(dim=-1) / float(dim))
        rho_i = (v_mu_all / raw_sig_i.square().clamp_min(1e-10)).clamp(0.0, 0.75)
        rho_mu = torch.where(valid_v, rho_i, rho_i.new_zeros(())).sum() / valid_cnt.clamp_min(1).float()
        shrink = torch.sqrt((0.995 - rho_mu).clamp_min(0.25))
        eff_raw_sig_i = torch.where(is_coal_i & (valid_cnt > 0), raw_sig_i * shrink, raw_sig_i)
        sigma_eff_i = eff_raw_sig_i.to(fake_flat.dtype)

        inv_w_i = (
            (1.0 / (float(len(reg.proto_c)) * reg.proto_w[near_p_G]).clamp_min(1e-4))
            .sqrt()
            .clamp(0.5, 4.0)
            .to(fake_flat.dtype)
        )

        # Target-isometry-referenced saturation compensation (0 sync via sort+index median):
        w_sat_vec = torch.ones(b, device=yr.device, dtype=fake_flat.dtype)
        if has_sigma and eps_g is not None:
            d_z_f = float(max(1, mu_det.shape[1]))
            e_norm_raw = eps_g.float().norm(dim=-1)
            valid_e = e_norm_raw >= 0.45
            s_obs = (fake_flat.detach().float() - x_ctr_flat.detach().float()).norm(dim=-1) / (
                sigma_z * e_norm_raw.clamp_min(0.45)
            )
            n_valid_e = valid_e.sum()
            s_sorted = torch.where(valid_e, s_obs, s_obs.new_full((), float("inf"))).sort().values
            s_med_idx = ((n_valid_e - 1).clamp_min(0) // 2).unsqueeze(0)
            s_med = s_sorted.index_select(0, s_med_idx).squeeze(0).clamp_min(1e-6)
            is_rail_coal = valid_e & (s_obs < (0.50 * s_med))
            w_boost_coal = (s_med / s_obs.clamp_min(1e-6)).clamp(1.0, 8.0)
            w_sat_coal = torch.where(is_rail_coal, w_boost_coal, torch.ones_like(w_boost_coal))

            s_target = (eff_raw_sig_i * math.sqrt(float(dim) / d_z_f)) / max(sigma_z, 1e-4)
            drift_d = (xff - y_ot.detach().float()).norm(dim=-1)
            is_rail_nocoal = (
                valid_e
                & (drift_d > (1.35 * math.sqrt(float(dim)) * raw_sig_i))
                & (s_obs < (0.40 * s_target))
            )
            w_boost_nocoal = (0.45 * s_target / s_obs.clamp_min(1e-6)).clamp(1.0, 8.0)
            w_sat_nocoal = torch.where(is_rail_nocoal, w_boost_nocoal, torch.ones_like(w_boost_nocoal))

            w_sat_vec = torch.where(has_multicell_coal_t, w_sat_coal, w_sat_nocoal).to(fake_flat.dtype)

        y_bound = yf.abs().amax(dim=0, keepdim=True) + 3.5 * sigma_cell_med

    x_ot_src = torch.where(is_coal_i.unsqueeze(1), x_ctr_flat, fake_flat)
    sq_ot = (x_ot_src - y_ot).square().sum(dim=-1)
    sc_med_dt = sigma_cell_med.to(fake_flat.dtype)
    loss_g_ot = (w_sat_vec * (torch.sqrt(sc_med_dt.square() + sq_ot) - sc_med_dt)).mean()
    loss_env = F.relu(fake_flat.float().abs() - y_bound).square().mean().to(fake_flat.dtype)
    loss_g_ot = loss_g_ot + 2.0 * loss_env

    if has_sigma and recipe.t1_iso_weight > 0.0:
        dx_ker = fake_flat - x_ctr_flat
        d_z = mu_det.shape[1]
        scale_dim = math.sqrt(float(dim) / float(d_z))
        eps_norm = eps_g.norm(dim=-1).clamp_min(0.25)
        target_dx_norm = sigma_eff_i * scale_dim * eps_norm
        rel_iso_err = ((dx_ker.norm(dim=-1) - target_dx_norm) / (sc_med_dt * scale_dim).clamp_min(1e-6)).abs()
        prev_iso_err64 = reg._get_iso_err_ema_t(yr.device)
        with torch.no_grad():
            cur_iso_err64 = rel_iso_err.detach().mean().clamp_max(5.0).double()
            iso_err_std64 = 0.94 * prev_iso_err64 + 0.06 * cur_iso_err64

        iso_1 = (inv_w_i * rel_iso_err.square()).mean()
        if dim == 2 and d_z == 2:
            if gen_has_bn and b > 128:
                has_coal_py = bool(has_multicell_coal_t.item())
                m_iso = b if has_coal_py else min(b, 128)
            else:
                has_coal_py = None
                m_iso = b
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
            with torch.no_grad():
                min_dx_u = torch.minimum(dx_0.detach().norm(dim=-1), dx_p.detach().norm(dim=-1))
                w_iso_sat = (sc_sub2 / min_dx_u.clamp_min(0.15 * sc_sub2)).clamp(1.0, 4.0)
            w_iso_sub = inv_w_i[:m_iso] * w_iso_sat
            u1_terms = w_iso_sub * ((dx_0.norm(dim=-1) - sc_sub2) / sc_med_dt).square()
            u2_terms = w_iso_sub * ((dx_p.norm(dim=-1) - sc_sub2) / sc_med_dt).square()
            iso_u1_full = u1_terms.mean()
            iso_u2_full = u2_terms.mean()

            m_nc = min(m_iso, 128)
            iso_u1_nc = u1_terms[:m_nc].mean() if m_nc < m_iso else iso_u1_full
            iso_u2_nc = u2_terms[:m_nc].mean() if m_nc < m_iso else iso_u2_full
            dx_0_nc = dx_0[:m_nc]
            dx_p_nc = dx_p[:m_nc]
            cos_sq_nc = ((dx_0_nc * dx_p_nc).sum(dim=-1) / (dx_0_nc.norm(dim=-1) * dx_p_nc.norm(dim=-1) + 1e-6)).square().mean()

            M_dx2 = torch.stack((dx_0, dx_p), dim=-1)
            Sigma_push2 = M_dx2 @ M_dx2.transpose(1, 2)
            eye2 = torch.eye(2, device=yr.device, dtype=fake_flat.dtype).unsqueeze(0)
            target_cov2 = (sc_sub2.square().view(m_iso, 1, 1)) * eye2
            denom_cov2 = (sc_sub2 * sc_med_dt).clamp_min(1e-8).view(m_iso, 1, 1)
            with torch.no_grad():
                if reg.v_push is not None:
                    vp_diag = dx_0.detach().float().square() + dx_p.detach().float().square()
                    comp_sub = comp_rep[near_p_G[:m_iso]]
                    cnt_vp = torch.zeros(k_proto, device=yr.device, dtype=torch.float32).scatter_add_(
                        0, comp_sub, ones_b[:m_iso]
                    )
                    sum_vp = torch.zeros((k_proto, dim), device=yr.device, dtype=torch.float32).index_add_(
                        0, comp_sub, vp_diag
                    )
                    cnt_k = cnt_vp[comp_rep]
                    hit_k = (cnt_k > 0) & multicell_coal & (~is_point_t)
                    mean_vp_k = sum_vp[comp_rep] / cnt_k.clamp_min(1.0).unsqueeze(1)
                    reg.v_push = torch.where(hit_k.unsqueeze(1), 0.88 * reg.v_push + 0.12 * mean_vp_k, reg.v_push)
                t_tr1 = sc_sub2.detach().float().square().clamp_min(1e-10).view(m_iso, 1, 1)
                iso_frob_i = ((Sigma_push2.detach().float() / t_tr1) - eye2.float()).square().sum(dim=(1, 2)).sqrt()
                cur_frob64 = (0.60 * iso_frob_i.mean() + 0.40 * torch.quantile(iso_frob_i, 0.92)).double()
                iso_err_coal64 = 0.92 * prev_iso_err64 + 0.08 * cur_frob64.clamp_max(5.0)
                iso_err_next64 = torch.where(has_multicell_coal_t, iso_err_coal64, iso_err_std64)
                reg._iso_err_ema_t = torch.where(is_point_t, prev_iso_err64, iso_err_next64)
            iso_cov2 = (
                w_iso_sub * ((Sigma_push2 - target_cov2) / denom_cov2).square().sum(dim=(1, 2))
            ).mean() * 0.5
            loss_coal = 0.45 * (iso_u1_full + iso_u2_full) + 0.55 * iso_cov2
            loss_nocoal = 0.5 * (iso_u1_nc + iso_u2_nc) + 0.25 * cos_sq_nc
            if has_coal_py is not None:
                loss_g_iso_norm = loss_coal if has_coal_py else loss_nocoal
            else:
                loss_g_iso_norm = torch.where(has_multicell_coal_t, loss_coal, loss_nocoal)
        elif dim == 2 and d_z == 4:
            reg._iso_err_ema_t = torch.where(is_point_t, prev_iso_err64, iso_err_std64)
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
            eye2 = torch.eye(2, device=yr.device, dtype=fake_flat.dtype).unsqueeze(0)
            target_cov = (sc_sub.square().view(m_iso, 1, 1)) * eye2
            denom_cov = (sc_sub * sc_med_dt).clamp_min(1e-8).view(m_iso, 1, 1)
            iso_cov = (inv_w_i[:m_iso] * ((Sigma_push - target_cov) / denom_cov).square().sum(dim=(1, 2))).mean() * 0.5
            loss_g_iso_norm = 0.5 * iso_1 + 0.5 * iso_cov
        else:
            reg._iso_err_ema_t = torch.where(is_point_t, prev_iso_err64, iso_err_std64)
            loss_g_iso_norm = iso_1
        loss_g_iso = torch.where(is_point_t, dx_ker.square().sum(dim=-1).mean(), loss_g_iso_norm)
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
        eq_gate_t = ((0.030 - res_ema_f) / 0.020).clamp(0.0, 1.0)
        adv_w_t = torch.where(
            has_multicell_coal_t,
            0.038 * (1.0 - eq_gate_t) + 0.008 * eq_gate_t,
            res_ema_f.new_full((), 0.15),
        )
        g_floor_t = torch.where(
            has_multicell_coal_t,
            0.022 * (1.0 - eq_gate_t) + 0.012 * eq_gate_t,
            res_ema_f.new_full((), 0.04),
        )
        row_floor_t = torch.where(
            has_multicell_coal_t,
            0.15 * (1.0 - eq_gate_t) + 0.38 * eq_gate_t,
            res_ema_f.new_full((), 0.15),
        )

    loss_g = (
        adv_w_t.to(loss_gan.dtype) * loss_gan
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
        iso_err_f = reg._get_iso_err_ema_t(yr.device).float()
        iso_gate_t = ((iso_err_f - 0.22) / 0.45).clamp(0.0, 1.0)
        g_scale_norm = torch.maximum(
            g_floor_t,
            torch.minimum(res_ema_f.new_full((), 1.0), torch.maximum(res_ema_f / 0.10, 0.28 * iso_gate_t)),
        )
        g_scale_t = torch.where(is_point_t | (scale_ratio_t < 0.60), res_ema_f.new_full((), 1.0), g_scale_norm)

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
    if can_condense_table and base_prior.z.requires_grad and float(recipe.t1_prior_row_lr) > 0.0:
        g_flags = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)

            if gen_has_bn:
                sorted_idx = torch.unique(indices)
                n_slots = len(sorted_idx)
                is_first = torch.ones(n_slots, device=indices.device, dtype=torch.bool)
            else:
                sorted_idx, _ = torch.sort(indices)
                n_slots = b
                is_first = torch.ones(n_slots, device=indices.device, dtype=torch.bool)
                if n_slots > 1:
                    is_first[1:] = sorted_idx[1:] != sorted_idx[:-1]
            w_u = is_first.float()
            n_u_t = is_first.sum()
            n_eff_t = n_u_t.clamp_min(1).float()
            active_part_b = ~is_point_t

            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[sorted_idx]
            x_u = trainer._postprocess(
                trainer.prior,
                trainer._generate(trainer.G, c_u, 0.0, noise),
            ).reshape(n_slots, -1)

            with torch.no_grad():
                xu_det = x_u.detach().float()
                psi_eff = w_mat_all @ reg.psi
                d2_up = torch.cdist(xu_det, c_bar_all, p=2.0).square()
                k_star = (d2_up - psi_eff.unsqueeze(0)).argmin(dim=1)

                w_hard_u = (w_u.unsqueeze(1) * w_mat_all[k_star]).sum(dim=0) / n_eff_t
                new_w_fake = 0.88 * reg.w_fake_ema + 0.12 * w_hard_u
                reg.w_fake_ema = torch.where(active_part_b, new_w_fake, reg.w_fake_ema)
                w_feedback = 0.35 * w_hard_u + 0.65 * reg.w_fake_ema

                k_float = float(len(reg.proto_c))
                tv_now_t = 0.5 * (reg.proto_w - reg.w_fake_ema).abs().sum()
                tv_floor_t = 0.35 * torch.sqrt(k_float / (n_eff_t * 8.0).clamp_min(1.0))
                psi_gate_t = ((tv_now_t - tv_floor_t) / tv_floor_t.clamp_min(0.08)).clamp(0.0, 1.0)

                eta_scale = min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * k_float))
                r_mac_sq = r_macro_safe.square()
                eta_psi_t = psi_gate_t * eta_scale * float(dim) * r_mac_sq
                psi_next = reg.psi * (1.0 - 0.008 * psi_gate_t) + eta_psi_t * (reg.proto_w - w_feedback)
                psi_next = w_mat_all @ psi_next
                psi_next = psi_next - psi_next.mean()
                psi_cap_t = 2.5 * float(dim) * r_mac_sq + 0.5
                psi_next = torch.clamp(psi_next, -psi_cap_t, psi_cap_t)
                reg.psi = torch.where(active_part_b, psi_next, reg.psi)

                c_bar_u = c_bar_all[k_star]
                coal_u = has_multicell_coal_t & multicell_coal[k_star]
                xi_u = _deterministic_box_muller(sorted_idx, dim, torch.float32).to(yr.device)
                cid_u = comp_rep[k_star]
                cnt_u = torch.zeros(k_proto, device=yr.device, dtype=torch.float32).scatter_add_(0, cid_u, w_u)
                xi_u_sum = torch.zeros((k_proto, dim), device=yr.device, dtype=torch.float32).index_add_(
                    0, cid_u, xi_u * w_u.unsqueeze(1)
                )
                xi_u_mean = torch.where(
                    cnt_u.unsqueeze(1) >= 2.0,
                    xi_u_sum / cnt_u.clamp_min(1.0).unsqueeze(1),
                    torch.zeros_like(xi_u_sum),
                )
                c_conc_u = c_bar_u + v_mu_star_all[k_star].sqrt() * (xi_u - xi_u_mean[cid_u])
                c_target = torch.where(coal_u.unsqueeze(1), c_conc_u, c_bar_u).to(x_u.dtype)
                sc_u = sigma_bar_all[k_star].to(x_u.dtype)
                dist_u = (xu_det - c_target.float()).norm(dim=-1)
                row_settle_t = torch.maximum(
                    row_floor_t,
                    torch.minimum(res_ema_f.new_full((), 1.0), torch.maximum(res_ema_f / 0.10, psi_gate_t)),
                )

            sq_u = (x_u - c_target).square().sum(dim=-1)
            w_u_dt = w_u.to(x_u.dtype)
            n_eff_dt = n_eff_t.to(x_u.dtype)
            loss_mu_condense = (w_u_dt * (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u)).sum() / n_eff_dt

            mu_raw_u = base_prior.z[sorted_idx]
            if has_sigma:
                w_u_col_z = w_u.to(mu_raw_u.dtype).unsqueeze(1)
                n_eff_z = n_eff_t.to(mu_raw_u.dtype)
                mu_mean = (w_u_col_z * mu_raw_u).sum(dim=0, keepdim=True) / n_eff_z
                mu_c = mu_raw_u - mu_mean
                mu_std = ((w_u_col_z * mu_c.square()).sum(dim=0) / n_eff_z + 1e-6).sqrt()
                iso_ema_dt = reg._get_iso_ema_t(yr.device).to(mu_std.dtype)
                loss_mu_iso = torch.where(
                    n_u_t >= 2,
                    F.relu(iso_ema_dt - mu_std).square().mean(),
                    x_u.new_zeros(()),
                )
            else:
                loss_mu_iso = x_u.new_zeros(())

            loss_mu_total = loss_mu_condense + 0.15 * loss_mu_iso
            prior_reg_diag = torch.where(active_part_b, loss_mu_total.detach(), prior_reg_diag)

            if base_prior.z.grad is not None:
                base_prior.z.grad.zero_()
            loss_mu_total.backward()

            if not reg.rows_initialized or reg.row_m is None or reg.row_m.shape != base_prior.z.shape:
                reg.row_m = torch.zeros_like(base_prior.z)
                reg.row_v = torch.zeros(len(base_prior.z), 1, device=base_prior.z.device, dtype=base_prior.z.dtype)
                reg.row_n = torch.zeros(len(base_prior.z), 1, device=base_prior.z.device, dtype=base_prior.z.dtype)
                reg.rows_initialized = True

            with torch.no_grad():
                g_rows = (base_prior.z.grad[sorted_idx] * n_eff_t.to(base_prior.z.dtype)).clone()
                base_prior.z.grad.zero_()
                dist_i = sq_u.detach().sqrt().unsqueeze(1)
                part_scale = (dist_i / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2)

                # Coordinatewise outward column-sensitivity clamp rescue (0 sync, 0 autograd.grad):
                far_mask_1d = is_first & (dist_u > (2.8 * math.sqrt(float(dim)) * sc_u.float()))
                d_z_u = c_u.shape[1]
                if d_z_u <= 8:
                    c_det = c_u.detach()
                    out_sgn = torch.where(c_det >= 0.0, 1.0, -1.0)
                    eye_z = torch.eye(d_z_u, device=c_u.device, dtype=c_u.dtype)
                    c_pert = (c_det.unsqueeze(1) + (1e-3 * out_sgn).unsqueeze(2) * eye_z.unsqueeze(0)).reshape(
                        -1, d_z_u
                    )
                    x_pert = (
                        trainer._postprocess(
                            trainer.prior,
                            trainer._generate(trainer.G, c_pert, 0.0, noise),
                        )
                        .float()
                        .reshape(n_slots, d_z_u, dim)
                    )
                    col_sens = (x_pert - xu_det.unsqueeze(1)).norm(dim=-1) / 1e-3
                    in_cand = is_first & (~far_mask_1d)
                    in_sel = in_cand & (in_cand.to(torch.int64).cumsum(0) <= 64)
                    fb_sel = is_first & (is_first.to(torch.int64).cumsum(0) <= 64)
                    eff_in = torch.where(in_sel.sum() < 4, fb_sel, in_sel)
                    n_eff_in = eff_in.sum().clamp_min(1)
                    s_for_q = torch.where(eff_in.unsqueeze(1), col_sens, col_sens.new_full((), float("inf")))
                    s_sorted, _ = torch.sort(s_for_q, dim=0)
                    q_idx = torch.stack(((n_eff_in - 1) // 2, n_eff_in // 2))
                    s_ref_col = s_sorted.index_select(0, q_idx).mean(dim=0, keepdim=True).clamp_min(1e-6)
                    dead_far = far_mask_1d.unsqueeze(1) & (col_sens < (0.04 * s_ref_col))
                    mu_ref = (eff_in.float().unsqueeze(1) * c_det.float()).sum(dim=0, keepdim=True) / n_eff_in.float()
                    z_far = mu_raw_u.detach()
                    g_mag = torch.maximum(
                        g_rows.float().norm(dim=-1, keepdim=True),
                        torch.ones_like(z_far[:, :1], dtype=torch.float32),
                    )
                    inward_grad = (torch.sign(z_far.float() - mu_ref) * g_mag).to(g_rows.dtype)
                    g_rows = torch.where(dead_far, inward_grad, g_rows)
                    any_dead_far = dead_far.any(dim=1, keepdim=True)
                    part_scale = torch.where(any_dead_far, torch.ones_like(part_scale), part_scale)

                w_u_col = (w_u * active_part_b.float()).to(base_prior.z.dtype).unsqueeze(1)
                m_prev = reg.row_m[sorted_idx]
                v_prev = reg.row_v[sorted_idx]
                n_prev = reg.row_n[sorted_idx]
                m = m_prev * 0.80 + 0.20 * g_rows
                v = v_prev * 0.80 + 0.20 * g_rows.square().sum(dim=-1, keepdim=True)
                n = n_prev + 1.0
                reg.row_m.index_add_(0, sorted_idx, (m - m_prev) * w_u_col)
                reg.row_v.index_add_(0, sorted_idx, (v - v_prev) * w_u_col)
                reg.row_n.index_add_(0, sorted_idx, w_u_col)
                bc = 1.0 - (0.80 ** n)
                step_vec = (m / bc) / ((v / bc).sqrt() + 1e-5)
                step_alpha = (-float(recipe.t1_prior_row_lr)) * row_settle_t.to(base_prior.z.dtype)
                base_prior.z.index_add_(0, sorted_idx, (step_vec * part_scale * w_u_col) * step_alpha)
        finally:
            for p, flag in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(flag)

    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


__all__ = [
    "UnifiedV11PriorRegularizer",
    "should_use_unified_v11_pullback",
    "unified_v11_pullback_step",
]
