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

from .._t4_arch.pullback import _deterministic_box_muller
from .._unified.prior import (
    _compute_pythagorean_smoothed_cells,
    _resolve_prior_kernel,
    _validate_unified_prior_state,
)
from .._bispf.thermal import FROZEN_THERMAL_SCALE, _read_thermal
from ..bispf import BISPFGates, SBNPPriorRegularizer
from .._unified_v4.loss import _principal_axis_power
from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .optimizer import UnifiedV11EquilibratedOptimizer

_PHI1 = 0.6180339887498949


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
            _dc_replace(base_gates, sbnp_active_iso_enabled=False)
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
        self.iso_ema: float = 1.0
        self.res_ema: float = 1.0
        self.iso_err_ema: float = 0.0
        self.rows_initialized: bool = False
        self.row_m: Optional[torch.Tensor] = None
        self.row_v: Optional[torch.Tensor] = None
        self.row_n: Optional[torch.Tensor] = None

    @property
    def thermal_source(self) -> Any:
        return self.fallback.thermal_source

    @thermal_source.setter
    def thermal_source(self, value: Any) -> None:
        self.fallback.thermal_source = value

    @property
    def weight(self) -> float:
        return self.fallback.weight

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        out = self.fallback(rows)
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
                v1 = _principal_axis_power(cdet, iters=6)
                lam1_d = (v1 * (cdet @ v1)).sum().clamp_min(0.0)
                cdef = cdet - lam1_d * torch.outer(v1, v1)
                v2 = _principal_axis_power(cdef, iters=6)
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
        from .._unified.prior import UnifiedPriorRegularizer

        if self._bound_device is None:
            self._bound_device = yr.device
        elif self.proto_c is not None and self.proto_c.device != yr.device:
            dev = yr.device
            for attr in ("proto_c", "proto_var", "proto_w", "proto_hits", "v_push", "psi", "w_fake_ema", "row_m", "row_v", "row_n"):
                val = getattr(self, attr, None)
                if isinstance(val, torch.Tensor) and val.device != dev:
                    setattr(self, attr, val.to(device=dev))
        UnifiedPriorRegularizer.update_quantizer(self, yr)
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
    recipe = trainer.recipe
    if recipe.prior_regularizer != "unified_v11" or float(recipe.t1_pullback_weight) <= 0.0:
        return False
    if recipe.encoder_mode != "none" or recipe.conditioning != "scalar":
        return False
    if real.ndim != 2 or not (1 <= int(real.shape[1]) <= 16):
        return False
    raw_var = float(real.detach().float().var(dim=0, unbiased=False).sum().item())
    if raw_var < 1e-11:
        return True
    if int(real.shape[1]) < 2:
        return False
    _, sigma_z, _ = _resolve_prior_kernel(trainer.prior, latent, indices)
    return sigma_z > 0.0


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
    reg: UnifiedV11PriorRegularizer = trainer.prior_regularizer
    base_prior = unwrap_prior(trainer.prior)
    b = len(real)

    mu_g, sigma_z, can_condense_table = _resolve_prior_kernel(trainer.prior, latent, indices)

    # --- PART A: Smooth Generator Chart G_theta + 0.995 Jensen-Centered Total-Variance Isometry ---
    mu_det = mu_g.detach()
    if sigma_z > 0.0:
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
        is_point_target = float(r_macro.item()) < 1e-6
        r_macro_safe = r_macro.clamp_min(1e-4)

        c_bar_all, sigma_bar_all, w_mat_all, multicell_coal = _compute_pythagorean_smoothed_cells(
            reg.proto_c, reg.proto_var, reg.proto_w
        )
        sigma_cell_med = torch.quantile(sigma_bar_all, 0.35).clamp_min(1e-6)
        if sigma_z > 0.0 and not is_point_target:
            iso_t = max(1.0, min(4.5, float((sigma_z * r_macro_safe / sigma_cell_med).item())))
            if not reg.iso_initialized:
                reg.iso_ema = iso_t
                reg.iso_initialized = True
            else:
                reg.iso_ema = 0.95 * reg.iso_ema + 0.05 * iso_t
        else:
            reg.iso_ema = 1.0
            reg.iso_initialized = True

    with torch.no_grad():
        near_p_G = torch.cdist(x_ctr_flat.detach().float(), c_bar_all, p=2.0).argmin(dim=1)
        is_coal_i = multicell_coal[near_p_G]
        xff = torch.where(is_coal_i.unsqueeze(1), x_ctr_flat.detach().float(), fake_flat.detach().float())

        # Collapsed-spread whitened Sinkhorn + wakeup boost for deep vanishing networks (e.g. 16-layer MLP):
        f_std = xff.std(dim=0, unbiased=False).mean().clamp_min(1e-8)
        scale_ratio = float((f_std / r_macro_safe).item())
        wakeup_boost = 5.0 if scale_ratio < 0.50 else (2.0 if scale_ratio < 0.80 else 1.0)

        d_xp_phys = torch.cdist(xff, c_bar_all, p=2.0)
        if scale_ratio < 0.35 and len(c_bar_all) >= 4:
            xff_w = (xff - xff.mean(dim=0, keepdim=True)) / xff.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            cb_mean = (reg.proto_w.unsqueeze(1) * c_bar_all).sum(dim=0, keepdim=True)
            cb_std = ((reg.proto_w.unsqueeze(1) * (c_bar_all - cb_mean).square()).sum(dim=0, keepdim=True)).sqrt().clamp_min(1e-6)
            cb_w = (c_bar_all - cb_mean) / cb_std
            d_xp_route = torch.cdist(xff_w, cb_w, p=2.0)
        else:
            d_xp_route = d_xp_phys

        c_scale = d_xp_route.mean().clamp_min(1e-5)
        eps_s = max(0.010, min(0.06, 0.25 * reg.res_ema))
        L_xp = -(d_xp_route / c_scale) / eps_s
        log_a = -math.log(float(b))
        log_b = torch.log(reg.proto_w.clamp_min(1e-12))
        u = torch.zeros(b, device=yr.device, dtype=torch.float32)
        v = torch.zeros(len(c_bar_all), device=yr.device, dtype=torch.float32)
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
        cur_res = (
            float(d_xp_phys.mean().item())
            if is_point_target
            else float(((torch.maximum(w_p2f, f2p) - cell_ref).clamp_min(0.0) / r_macro_safe).item())
        )
        reg.res_ema = 0.92 * reg.res_ema + 0.08 * min(cur_res, 10.0)

        has_multicell_coal = bool(multicell_coal.any().item())
        comp_rep = (w_mat_all > 0.0).float().argmax(dim=1)
        v_mu_star_all = torch.zeros_like(c_bar_all)
        if has_multicell_coal and reg.v_push is not None:
            v_bar_sq = sigma_bar_all.square().unsqueeze(1)
            v_push_bar = (w_mat_all @ reg.v_push).clamp_min(0.0)
            deficit_ratio = ((v_bar_sq - v_push_bar) / v_bar_sq.clamp_min(1e-10)).clamp(0.0, 1.0)
            u_def = ((deficit_ratio - 0.15) / 0.12).clamp(0.0, 1.0)
            gate_def = u_def * u_def * (3.0 - 2.0 * u_def)
            v_mu_star_all = gate_def * (v_bar_sq - v_push_bar).clamp_min(0.0)
            need_conc_i = is_coal_i & (v_mu_star_all[near_p_G].amax(dim=-1) > 0.0)
            if (
                bool(need_conc_i.any())
                and scale_ratio >= 0.65
                and can_condense_table
                and not is_point_target
                and indices is not None
            ):
                d_near_i = (xff - c_bar_all[near_p_G]).norm(dim=-1)
                halo_r = 3.5 * math.sqrt(float(dim)) * sigma_bar_all[near_p_G]
                halo_blend = ((d_near_i / halo_r.clamp_min(1e-6)).clamp(0.0, 1.0) ** 2).unsqueeze(1).to(fake_flat.dtype)
                xi_i = _deterministic_box_muller(indices, dim, torch.float32).to(yr.device)
                cid_i = comp_rep[near_p_G]
                cnt_f = torch.bincount(cid_i, minlength=len(c_bar_all)).float()
                xi_f_sum = torch.zeros((len(c_bar_all), dim), device=yr.device, dtype=torch.float32).index_add_(
                    0, cid_i, xi_i
                )
                xi_f_mean = torch.zeros((len(c_bar_all), dim), device=yr.device, dtype=torch.float32)
                cf2 = cnt_f >= 2
                xi_f_mean[cf2] = xi_f_sum[cf2] / cnt_f[cf2].unsqueeze(1)
                c_exact_i = (
                    c_bar_all[near_p_G] + v_mu_star_all[near_p_G].sqrt() * (xi_i - xi_f_mean[cid_i])
                ).to(fake_flat.dtype)
                y_ot_conc = halo_blend * y_ot + (1.0 - halo_blend) * c_exact_i
                y_ot = torch.where(need_conc_i.unsqueeze(1), y_ot_conc, y_ot)

        # Law of Total Variance + 0.995 Quartic-Frobenius Jensen Centering:
        # Unconstrained modes realize full 1.0*sigma^2 pushforward (v_mu_star == 0, zero center noise);
        # tanh-attenuated boundary modes automatically engage concentric residual (sigma^2 - v_push)_+.
        raw_sig_i = sigma_bar_all[near_p_G]
        eff_raw_sig_i = raw_sig_i
        if bool(is_coal_i.any()):
            xcf_det = x_ctr_flat.detach().float()
            cid_i = comp_rep[near_p_G]
            cnt_c = torch.bincount(cid_i, minlength=len(c_bar_all)).float()
            sum_c = torch.zeros((len(c_bar_all), dim), device=yr.device, dtype=torch.float32).index_add_(
                0, cid_i, xcf_det
            )
            mean_c = sum_c / cnt_c.clamp_min(1.0).unsqueeze(1)
            n_i = cnt_c[cid_i]
            valid_v = is_coal_i & (n_i >= 2.0)
            if bool(valid_v.any()):
                v_mu_i = (n_i[valid_v] / (n_i[valid_v] - 1.0)) * (
                    (xcf_det[valid_v] - mean_c[cid_i[valid_v]]).square().sum(dim=-1) / float(dim)
                )
                rho_mu = (v_mu_i / raw_sig_i[valid_v].square().clamp_min(1e-10)).clamp(0.0, 0.75).mean()
                shrink = torch.sqrt((0.995 - rho_mu).clamp_min(0.25))
                eff_raw_sig_i = torch.where(is_coal_i, raw_sig_i * shrink, raw_sig_i)
        sigma_eff_i = eff_raw_sig_i.to(fake_flat.dtype)

        inv_w_i = (
            (1.0 / (float(len(reg.proto_c)) * reg.proto_w[near_p_G]).clamp_min(1e-4))
            .sqrt()
            .clamp(0.5, 4.0)
            .to(fake_flat.dtype)
        )

        # Target-isometry-referenced saturation compensation (rescues boundary tanh attenuation
        # strictly on well-conditioned tangent draws, with zero false boosts on unsaturated tasks):
        w_sat_vec = torch.ones(b, device=yr.device, dtype=fake_flat.dtype)
        if sigma_z > 0.0 and eps_g is not None:
            d_z_f = float(max(1, mu_det.shape[1]))
            e_norm_raw = eps_g.float().norm(dim=-1)
            valid_e = e_norm_raw >= 0.45
            s_obs = (fake_flat.detach().float() - x_ctr_flat.detach().float()).norm(dim=-1) / (
                sigma_z * e_norm_raw.clamp_min(0.45)
            )
            if has_multicell_coal and bool(valid_e.any()):
                s_med = s_obs[valid_e].median().clamp_min(1e-6)
                is_rail_clamped = valid_e & (s_obs < (0.50 * s_med))
                if bool(is_rail_clamped.any()):
                    w_boost = (s_med / s_obs.clamp_min(1e-6)).clamp(1.0, 8.0)
                    w_sat_vec = torch.where(is_rail_clamped, w_boost, torch.ones_like(w_boost)).to(fake_flat.dtype)
            elif bool(valid_e.any()):
                s_target = (eff_raw_sig_i * math.sqrt(float(dim) / d_z_f)) / max(sigma_z, 1e-4)
                drift_d = (xff - y_ot.detach().float()).norm(dim=-1)
                is_rail_clamped = (
                    valid_e
                    & (drift_d > (1.35 * math.sqrt(float(dim)) * raw_sig_i))
                    & (s_obs < (0.40 * s_target))
                )
                if bool(is_rail_clamped.any()):
                    w_boost = (0.45 * s_target / s_obs.clamp_min(1e-6)).clamp(1.0, 8.0)
                    w_sat_vec = torch.where(is_rail_clamped, w_boost, torch.ones_like(w_boost)).to(fake_flat.dtype)

        y_bound = yf.abs().amax(dim=0, keepdim=True) + 3.5 * sigma_cell_med

    x_ot_src = torch.where(is_coal_i.unsqueeze(1), x_ctr_flat, fake_flat)
    sq_ot = (x_ot_src - y_ot).square().sum(dim=-1)
    sc_med_dt = sigma_cell_med.to(fake_flat.dtype)
    loss_g_ot = (w_sat_vec * (torch.sqrt(sc_med_dt.square() + sq_ot) - sc_med_dt)).mean()
    loss_env = F.relu(fake_flat.float().abs() - y_bound).square().mean().to(fake_flat.dtype)
    loss_g_ot = loss_g_ot + 2.0 * loss_env

    if sigma_z > 0.0 and recipe.t1_iso_weight > 0.0 and not is_point_target:
        dx_ker = fake_flat - x_ctr_flat
        d_z = mu_det.shape[1]
        scale_dim = math.sqrt(float(dim) / float(d_z))
        eps_norm = eps_g.norm(dim=-1).clamp_min(0.25)
        target_dx_norm = sigma_eff_i * scale_dim * eps_norm
        rel_iso_err = ((dx_ker.norm(dim=-1) - target_dx_norm) / (sc_med_dt * scale_dim).clamp_min(1e-6)).abs()
        if not (has_multicell_coal and dim == 2 and d_z == 2):
            with torch.no_grad():
                cur_iso_err = float(rel_iso_err.detach().mean().clamp_max(5.0).item())
                reg.iso_err_ema = 0.94 * reg.iso_err_ema + 0.06 * cur_iso_err

        iso_1 = (inv_w_i * rel_iso_err.square()).mean()
        if dim == 2 and d_z == 2:
            m_iso = b if has_multicell_coal else min(b, 128)
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
            iso_u1 = (w_iso_sub * ((dx_0.norm(dim=-1) - sc_sub2) / sc_med_dt).square()).mean()
            iso_u2 = (w_iso_sub * ((dx_p.norm(dim=-1) - sc_sub2) / sc_med_dt).square()).mean()
            cos_sq = ((dx_0 * dx_p).sum(dim=-1) / (dx_0.norm(dim=-1) * dx_p.norm(dim=-1) + 1e-6)).square().mean()
            if has_multicell_coal:
                M_dx2 = torch.stack((dx_0, dx_p), dim=-1)
                Sigma_push2 = M_dx2 @ M_dx2.transpose(1, 2)
                eye2 = torch.eye(2, device=yr.device, dtype=fake_flat.dtype).unsqueeze(0)
                target_cov2 = (sc_sub2.square().view(m_iso, 1, 1)) * eye2
                denom_cov2 = (sc_sub2 * sc_med_dt).clamp_min(1e-8).view(m_iso, 1, 1)
                with torch.no_grad():
                    if reg.v_push is not None:
                        vp_diag = dx_0.detach().float().square() + dx_p.detach().float().square()
                        comp_sub = comp_rep[near_p_G[:m_iso]]
                        cnt_vp = torch.bincount(comp_sub, minlength=len(c_bar_all)).float()
                        sum_vp = torch.zeros((len(c_bar_all), dim), device=yr.device, dtype=torch.float32).index_add_(
                            0, comp_sub, vp_diag
                        )
                        cnt_k = cnt_vp[comp_rep]
                        hit_k = (cnt_k > 0) & multicell_coal
                        if bool(hit_k.any()):
                            mean_vp_k = sum_vp[comp_rep[hit_k]] / cnt_k[hit_k].unsqueeze(1)
                            reg.v_push[hit_k] = 0.88 * reg.v_push[hit_k] + 0.12 * mean_vp_k
                    t_tr1 = sc_sub2.detach().float().square().clamp_min(1e-10).view(m_iso, 1, 1)
                    iso_frob_i = ((Sigma_push2.detach().float() / t_tr1) - eye2.float()).square().sum(dim=(1, 2)).sqrt()
                    cur_frob = 0.60 * float(iso_frob_i.mean().item()) + 0.40 * float(torch.quantile(iso_frob_i, 0.92).item())
                    reg.iso_err_ema = 0.92 * reg.iso_err_ema + 0.08 * min(cur_frob, 5.0)
                iso_cov2 = (
                    w_iso_sub * ((Sigma_push2 - target_cov2) / denom_cov2).square().sum(dim=(1, 2))
                ).mean() * 0.5
                loss_g_iso = 0.45 * (iso_u1 + iso_u2) + 0.55 * iso_cov2
            else:
                loss_g_iso = 0.5 * (iso_u1 + iso_u2) + 0.25 * cos_sq
        elif dim == 2 and d_z == 4:
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
            loss_g_iso = 0.5 * iso_1 + 0.5 * iso_cov
        else:
            loss_g_iso = iso_1
    elif sigma_z > 0.0 and recipe.t1_iso_weight > 0.0 and is_point_target:
        loss_g_iso = (fake_flat - x_ctr_flat).square().sum(dim=-1).mean()
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

    if has_multicell_coal:
        eq_gate = max(0.0, min(1.0, (0.030 - reg.res_ema) / 0.020))
        adv_w = 0.038 * (1.0 - eq_gate) + 0.008 * eq_gate
        g_floor = 0.022 * (1.0 - eq_gate) + 0.012 * eq_gate
        row_floor = 0.15 * (1.0 - eq_gate) + 0.38 * eq_gate
    else:
        adv_w = 0.15
        g_floor = 0.04
        row_floor = 0.15

    loss_g = (
        adv_w * loss_gan
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
                    trainer._generate(trainer.G, latent_det if sigma_z > 0.0 else mu_det, sigma_out, noise, rows=indices),
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
    iso_gate = max(0.0, min(1.0, (reg.iso_err_ema - 0.22) / 0.45))
    g_scale = (
        1.0
        if (is_point_target or scale_ratio < 0.60)
        else max(g_floor, min(1.0, max(reg.res_ema / 0.10, 0.28 * iso_gate)))
    )

    if isinstance(trainer.opt_g, UnifiedV11EquilibratedOptimizer):
        trainer.opt_g.bind_v4_modulation(g_scale=g_scale, wakeup_boost=wakeup_boost, prior_scale=1.0)
        if trainer._gradient_response is not None:
            params = list(trainer.G.parameters())
            grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
            gain = trainer._gradient_response.observe(grads)
            for p, m in zip(params, trainer._gradient_response.packet["means"]):
                p.grad = (m / trainer._gradient_response.packet["weight"]).to(p).clone()
            trainer.opt_g.bind_v4_modulation(g_scale=g_scale * gain, wakeup_boost=wakeup_boost, prior_scale=1.0)
            trainer.opt_g.step()
            trainer._gradient_response.response["applied_steps"] += 1
        else:
            trainer.opt_g.step()
    else:
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
    if (
        can_condense_table
        and base_prior.z.requires_grad
        and float(recipe.t1_prior_row_lr) > 0.0
        and not is_point_target
    ):
        u_rows = torch.unique(indices)
        g_flags = [p.requires_grad for p in trainer.G.parameters()]
        try:
            for p in trainer.G.parameters():
                p.requires_grad_(False)

            c_u = (base_prior.means() if isinstance(base_prior, MoGParticlePrior) else base_prior.z)[u_rows]
            x_u = trainer._postprocess(
                trainer.prior,
                trainer._generate(trainer.G, c_u, 0.0, noise),
            ).reshape(len(u_rows), -1)

            with torch.no_grad():
                xu_det = x_u.detach().float()
                psi_eff = w_mat_all @ reg.psi
                d2_up = torch.cdist(xu_det, c_bar_all, p=2.0).square()
                k_star = (d2_up - psi_eff.unsqueeze(0)).argmin(dim=1)

                w_hard_u = w_mat_all[k_star].mean(dim=0)
                reg.w_fake_ema = 0.88 * reg.w_fake_ema + 0.12 * w_hard_u
                w_feedback = 0.35 * w_hard_u + 0.65 * reg.w_fake_ema

                k_float = float(len(reg.proto_c))
                n_eff = float(len(u_rows))
                tv_now = 0.5 * float((reg.proto_w - reg.w_fake_ema).abs().sum().item())
                tv_floor = 0.35 * math.sqrt(k_float / max(n_eff * 8.0, 1.0))
                psi_gate = max(0.0, min(1.0, (tv_now - tv_floor) / max(0.08, tv_floor)))

                eta_scale = min(3.0, max(0.35, float(recipe.t1_laguerre_lr) * k_float))
                eta_psi = psi_gate * eta_scale * float(dim) * float((r_macro_safe ** 2).item())
                reg.psi.mul_(1.0 - 0.008 * psi_gate)
                reg.psi.add_(reg.proto_w - w_feedback, alpha=eta_psi)
                reg.psi = w_mat_all @ reg.psi
                reg.psi.sub_(reg.psi.mean())
                psi_cap = 2.5 * float(dim) * float((r_macro_safe ** 2).item()) + 0.5
                reg.psi.clamp_(-psi_cap, psi_cap)

                c_bar_u = c_bar_all[k_star]
                if has_multicell_coal:
                    coal_u = multicell_coal[k_star]
                    if bool(coal_u.any()):
                        xi_u = _deterministic_box_muller(u_rows, dim, torch.float32).to(yr.device)
                        cid_u = comp_rep[k_star]
                        cnt_u = torch.bincount(cid_u, minlength=len(c_bar_all)).float()
                        xi_u_sum = torch.zeros((len(c_bar_all), dim), device=yr.device, dtype=torch.float32).index_add_(
                            0, cid_u, xi_u
                        )
                        xi_u_mean = torch.zeros((len(c_bar_all), dim), device=yr.device, dtype=torch.float32)
                        cu2 = cnt_u >= 2
                        xi_u_mean[cu2] = xi_u_sum[cu2] / cnt_u[cu2].unsqueeze(1)
                        c_conc_u = c_bar_u + v_mu_star_all[k_star].sqrt() * (xi_u - xi_u_mean[cid_u])
                        c_target = torch.where(coal_u.unsqueeze(1), c_conc_u, c_bar_u).to(x_u.dtype)
                    else:
                        c_target = c_bar_u.to(x_u.dtype)
                else:
                    c_target = c_bar_u.to(x_u.dtype)
                sc_u = sigma_bar_all[k_star].to(x_u.dtype)
                dist_u = (xu_det - c_target.float()).norm(dim=-1)
                row_settle = max(row_floor, min(1.0, max(reg.res_ema / 0.10, psi_gate)))

            sq_u = (x_u - c_target).square().sum(dim=-1)
            loss_mu_condense = (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u).mean()

            if sigma_z > 0.0 and len(u_rows) >= 2:
                mu_raw_u = base_prior.z[u_rows]
                mu_c = mu_raw_u - mu_raw_u.mean(dim=0, keepdim=True)
                mu_std = (mu_c.square().mean(dim=0) + 1e-6).sqrt()
                loss_mu_iso = F.relu(reg.iso_ema - mu_std).square().mean()
            else:
                mu_raw_u = base_prior.z[u_rows]
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
                g_rows = (base_prior.z.grad[u_rows] * float(len(u_rows))).clone()
                base_prior.z.grad.zero_()
                dist_i = sq_u.detach().sqrt().unsqueeze(1)
                part_scale = (dist_i / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2)

                # Coordinatewise outward column-sensitivity clamp rescue (rescues single- and multi-axis
                # dead-clamped outer latent shells on tanh_rail_guard with strictly 0 autograd.grad!):
                far_mask_1d = dist_u > (2.8 * math.sqrt(float(dim)) * sc_u.float())
                d_z_u = c_u.shape[1]
                if d_z_u <= 8 and bool(far_mask_1d.any()):
                    far_idx = torch.nonzero(far_mask_1d, as_tuple=False).squeeze(-1)
                    in_idx = torch.nonzero(~far_mask_1d, as_tuple=False).squeeze(-1)[:64]
                    if in_idx.numel() < 4:
                        in_idx = torch.arange(min(64, len(c_u)), device=c_u.device, dtype=torch.long)
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
                base_prior.z.index_add_(
                    0, u_rows, step_vec * part_scale, alpha=-float(recipe.t1_prior_row_lr) * row_settle
                )
        finally:
            for p, flag in zip(trainer.G.parameters(), g_flags):
                p.requires_grad_(flag)

    return loss_g, loss_gan, prior_reg_diag, prior_manip_loss, transport, transport_local


__all__ = [
    "UnifiedV11PriorRegularizer",
    "should_use_unified_v11_pullback",
    "unified_v11_pullback_step",
]
