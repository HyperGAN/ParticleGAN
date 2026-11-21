"""Unified-v11 1st-Order Spatial-Frame Forward-Tangent & Multi-Probe Secant Born-Infeld Critic Penalty."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from .._t2_critic.penalty import (
    _format_logits_2d,
    _murmur32_signs,
    _separable_image_unit_perturbation,
)
from .._unified.penalty import (
    _GOLDEN_CONJ,
    _UnifiedPenaltyKernel,
    _adaptive_fd_delta,
    _eval_lowd_forward_grad_sq,
)
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .critic import (
    UnifiedV11CriticAdam,
    UnifiedV11StepRecord,
    _attach_unified_v11_critic_hook,
    _cond_signature,
)
from .optimizer import UnifiedV11EquilibratedOptimizer


def _universal_image_unit_tangent(x: torch.Tensor, phase: float, salt: int) -> torch.Tensor:
    """Zero-RNG isotropic unit tangent ``||u_i||_2 = 1`` for any spatial tensor rank >= 3."""
    b = int(x.shape[0])
    d = max(1, int(x[0].numel()))
    if d <= 8192:
        ranks = torch.arange(b, device=x.device, dtype=torch.int64)
        u = _murmur32_signs(ranks, d, phase, salt, x.device)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return u.to(dtype=x.dtype).reshape_as(x)
    return _separable_image_unit_perturbation(x, phase, salt=salt)


def _bn_dual_bf16_hook(
    _module: nn.Module,
    _inputs: Tuple[Any, ...],
    output: Any,
) -> Any:
    """Keep BatchNorm dual primal + tangent aligned in BF16 during autocast fwAD passes."""
    if isinstance(output, torch.Tensor):
        primal, tangent = fwAD.unpack_dual(output)
        if tangent is not None and (primal.dtype != torch.bfloat16 or tangent.dtype != torch.bfloat16):
            return fwAD.make_dual(primal.to(dtype=torch.bfloat16), tangent.to(dtype=torch.bfloat16))
    return output


class _UnifiedV11PenaltyKernel(_UnifiedPenaltyKernel):
    """Unified-v11 universal 1st-order forward-AD Born-Infeld critic kernel (`0` `autograd.grad`)."""

    def __init__(self, *, record: Any = None, **kwargs: Any) -> None:
        rec = UnifiedV11StepRecord() if record is None else record
        super().__init__(record=rec, **kwargs)
        self.arm = "unified_v11"

    def _eval_image_fwad_sq(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        x_probe: torch.Tensor,
        phi: float,
        *,
        salt: int,
        dim: int,
        has_batchnorm: bool,
        critic_module: Optional[nn.Module],
        quantized_fprop: bool,
    ) -> torch.Tensor:
        """Evaluate ``dim * ||J_D(x) u||_2^2`` via 1st-order forward-mode AD (0 `autograd.grad`)."""
        b_p = int(x_probe.shape[0])
        u_p = _universal_image_unit_tangent(x_probe, phi, salt=salt)
        bn_modes = []
        bn_hooks = []
        use_bf16_dual_hook = bool(
            has_batchnorm
            and x_probe.is_cuda
            and isinstance(critic_module, nn.Module)
            and (
                bool(getattr(critic_module, "bf16", False))
                or (
                    torch.is_autocast_enabled("cuda")
                    and torch.get_autocast_dtype("cuda") == torch.bfloat16
                )
            )
        )
        if has_batchnorm and isinstance(critic_module, nn.Module):
            for mod in critic_module.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    bn_modes.append((mod, mod.training))
                    mod.eval()
                    if use_bf16_dual_hook:
                        bn_hooks.append(mod.register_forward_hook(_bn_dual_bf16_hook))
        try:
            if not quantized_fprop:
                try:
                    if use_bf16_dual_hook:
                        x_in = x_probe.to(dtype=torch.bfloat16)
                        u_in = u_p.to(dtype=torch.bfloat16)
                    else:
                        x_in = x_probe
                        u_in = u_p
                    with fwAD.dual_level():
                        yd = live(fwAD.make_dual(x_in, u_in))
                        _, jvp = fwAD.unpack_dual(yd)
                    if jvp is None:
                        raise NotImplementedError("dual tangent dropped")
                    j2d = _format_logits_2d(jvp.float(), b_p)
                    return float(dim) * j2d.square().mean(dim=-1)
                except (RuntimeError, NotImplementedError, TypeError):
                    if use_bf16_dual_hook and isinstance(critic_module, nn.Module):
                        # Clean FP32 fwAD fallback if an exotic custom norm layer disagrees with BF16 duals.
                        for h in bn_hooks:
                            h.remove()
                        bn_hooks.clear()
                        prev_bf16 = bool(getattr(critic_module, "bf16", False))
                        if prev_bf16:
                            critic_module.bf16 = False
                        try:
                            with torch.autocast(device_type="cuda", enabled=False):
                                with fwAD.dual_level():
                                    yd = live(fwAD.make_dual(x_probe.float(), u_p.float()))
                                    _, jvp = fwAD.unpack_dual(yd)
                            if jvp is not None:
                                j2d = _format_logits_2d(jvp.float(), b_p)
                                return float(dim) * j2d.square().mean(dim=-1)
                        except (RuntimeError, NotImplementedError, TypeError):
                            pass
                        finally:
                            if prev_bf16:
                                critic_module.bf16 = True
            # Fallback strictly for custom quantized/non-jvp ops: scale-aware symmetric central diff.
            fd_delta = _adaptive_fd_delta(x_probe, quantized_fprop=True)
            xp_p = (x_probe.float() + fd_delta * u_p.float()).to(dtype=x_probe.dtype).detach()
            xp_m = (x_probe.float() - fd_delta * u_p.float()).to(dtype=x_probe.dtype).detach()
            yp = _format_logits_2d(live(xp_p).float(), b_p)
            ym = _format_logits_2d(live(xp_m).float(), b_p)
            step_sq = (xp_p.float() - xp_m.float()).flatten(1).square().sum(dim=1).clamp_min(1e-12)
            return float(dim) * ((yp - ym).square().mean(dim=-1) / step_sq)
        finally:
            for h in bn_hooks:
                h.remove()
            for mod, flag in bn_modes:
                mod.training = flag

    def penalty(
        self,
        live: Callable[[torch.Tensor], torch.Tensor],
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        is_conditioned: bool = False,
        cond_tensors: Sequence[torch.Tensor] = (),
        cond_sig: Tuple[Any, ...] = (),
        allow_live_reuse: bool = True,
        can_concat_batches: bool = False,
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        quantized_fprop: bool = False,
        has_batchnorm: bool = False,
        critic_module: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        is_image_tensor = x_real.ndim >= 3 and self.input_domain != "continuous"
        if not is_image_tensor:
            res = super().penalty(
                live,
                x_real,
                x_fake,
                step,
                collect_stats,
                is_conditioned=is_conditioned,
                cond_tensors=cond_tensors,
                cond_sig=cond_sig,
                allow_live_reuse=allow_live_reuse,
                can_concat_batches=can_concat_batches,
                live_rep=live_rep,
                quantized_fprop=quantized_fprop,
            )
            if hasattr(self.record, "observe_scale") and not getattr(self.record, "_capture_ring", None):
                self.record.observe_scale(x_real, x_fake)
            return res

        if (
            self.coeff == 0.0
            or (
                self.real_weight == 0.0
                and self.fake_weight == 0.0
                and self.gamma_gauge == 0.0
                and self.chord_weight == 0.0
            )
            or (self.lazy_k > 1 and step % self.lazy_k != 0)
        ):
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        self.record.calls += 1
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        b_r, b_f = int(x_real.shape[0]), int(x_fake.shape[0])
        dim = max(1, int(x_real[0].numel()))

        eff_beta = self.ebi_beta
        if self.settle_beta > 0.0 and bool(getattr(self.record, "t1c_thermal_initialized", False)):
            if self.settle_mode == "thermal":
                s_th = float(getattr(self.record, "_thermal_scale", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - s_th))
            else:
                res = float(getattr(self.record, "ema_residual", 1.0))
                settle_u = min(1.0, max(0.0, 1.0 - res / 0.22))
            eff_beta = eff_beta + self.settle_beta * (settle_u * settle_u)

        reused_endpoints = False
        try:
            phi = float(getattr(self.record, "t1c_chord_phase", 0.0))
            if hasattr(self.record, "t1c_chord_phase"):
                self.record.t1c_chord_phase = (phi + _GOLDEN_CONJ) % 1.0

            hit_r = self._lookup_live_endpoint(x_real, cond_sig) if allow_live_reuse else None
            hit_f = (
                self._lookup_live_endpoint(
                    x_fake,
                    cond_sig,
                    exclude_out=hit_r[1] if hit_r is not None else None,
                )
                if allow_live_reuse
                else None
            )
            reused_endpoints = bool(hit_r is not None and hit_f is not None)

            self.record._in_penalty = True
            try:
                if hit_r is not None:
                    z_r, d_r_raw = hit_r
                    z_r = z_r.detach().to(dtype=x_real.dtype)
                else:
                    self.record._last_probe_actual_input = None
                    x_r_det = x_real.detach()
                    d_r_raw = live(x_r_det)
                    act_r = getattr(self.record, "_last_probe_actual_input", None)
                    z_r = (
                        act_r.detach().to(dtype=x_real.dtype)
                        if isinstance(act_r, torch.Tensor) and act_r.shape == x_r_det.shape
                        else x_r_det
                    )

                if hit_f is not None:
                    z_f, d_f_raw = hit_f
                    z_f = z_f.detach().to(dtype=x_fake.dtype)
                else:
                    self.record._last_probe_actual_input = None
                    x_f_det = x_fake.detach()
                    d_f_raw = live(x_f_det)
                    act_f = getattr(self.record, "_last_probe_actual_input", None)
                    z_f = (
                        act_f.detach().to(dtype=x_fake.dtype)
                        if isinstance(act_f, torch.Tensor) and act_f.shape == x_f_det.shape
                        else x_f_det
                    )

                d_r_2d = _format_logits_2d(d_r_raw, b_r)
                d_f_2d = _format_logits_2d(d_f_raw, b_f)

                # Universal 1st-order forward-AD probe across ALL image spatial dimensions (8x8..256x256):
                # 1 interleaved half-real/half-fake batch (BN in eval -> zero cross-sample BN leakage).
                if not is_conditioned and b_r == b_f and b_r >= 2:
                    parity = ((torch.arange(b_f, device=x_real.device) + int(phi * 1024.0)) % 2) == 0
                    mask_nd = parity.reshape((b_f,) + (1,) * (x_real.ndim - 1))
                    x_probe = torch.where(mask_nd, z_r, z_f)
                    sq_mix = self._eval_image_fwad_sq(
                        live,
                        x_probe,
                        phi,
                        salt=3,
                        dim=dim,
                        has_batchnorm=has_batchnorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )
                    sq_r_raw = sq_mix[parity]
                    sq_f_raw = sq_mix[~parity]
                else:
                    sq_r_raw = self._eval_image_fwad_sq(
                        live,
                        z_r,
                        phi,
                        salt=3,
                        dim=dim,
                        has_batchnorm=has_batchnorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )
                    sq_f_raw = self._eval_image_fwad_sq(
                        live,
                        z_f,
                        phi,
                        salt=11,
                        dim=dim,
                        has_batchnorm=has_batchnorm,
                        critic_module=critic_module,
                        quantized_fprop=quantized_fprop,
                    )

                # Universal per-sample Born-Infeld cap on unbiased 1-probe forward-AD ||grad_x D||^2:
                norm_r = (sq_r_raw + 1e-12).sqrt()
                norm_f = (sq_f_raw + 1e-12).sqrt()
                total = x_real.new_zeros(())
                if self.real_weight > 0.0:
                    total = total + self.real_weight * F.softplus(norm_r - self.kappa, beta=eff_beta).square().mean()
                    if self.lambda_floor > 0.0:
                        total = total + self.real_weight * (0.0005 / float(dim)) * sq_r_raw.mean()
                if self.fake_weight > 0.0:
                    total = total + self.fake_weight * F.softplus(norm_f - self.kappa, beta=eff_beta).square().mean()
                    if self.lambda_floor > 0.0:
                        total = total + self.fake_weight * (0.0005 / float(dim)) * sq_f_raw.mean()
                if self.gamma_gauge > 0.0:
                    eff_gauge = min(self.gamma_gauge, 0.005)
                    mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                    gauge_pen = eff_gauge * mean_logit.square()
                else:
                    gauge_pen = x_real.new_zeros(())
                norm_stat = 0.5 * (norm_r.detach().mean() + norm_f.detach().mean())
            finally:
                self.record._in_penalty = False

            if hasattr(self.record, "observe_scale") and not getattr(self.record, "_capture_ring", None):
                self.record.observe_scale(z_r, z_f)

            pen_lip = 0.5 * coefficient * total
            pen_gauge = 0.5 * coefficient * gauge_pen
            s_therm = float(getattr(self.record, "_thermal_scale", 1.0))
            if s_therm < 1.0:
                if pen_lip.requires_grad:
                    pen_lip = pen_lip * s_therm + pen_lip.detach() * (1.0 - s_therm)
                if pen_gauge.requires_grad:
                    s_gauge = math.sqrt(max(0.0, s_therm))
                    pen_gauge = pen_gauge * s_gauge + pen_gauge.detach() * (1.0 - s_gauge)
            pen = pen_lip + pen_gauge

            if not collect_stats:
                return pen, {}

            pen_v, mean_norm = torch.stack((pen.detach().float(), norm_stat.float())).tolist()
            ell_v = float(getattr(self.record, "t1c_ell_ema", 0.0))
            return pen, {
                "applied": True,
                "pen": float(pen_v),
                "center": self.kappa,
                "phase": self.arm,
                "mean_grad_norm": float(mean_norm),
                "t1c_ell": float(ell_v),
                "s_thermal": float(s_therm),
                "live_endpoints_reused": bool(reused_endpoints),
            }
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()


class UnifiedV11CriticPenalty(_K3PCriticPenalty):
    """Recipe-facing Unified-v11 1st-order Forward-Tangent & Multi-Probe Secant critic penalty."""

    def __init__(
        self,
        recipe: Any,
        optimizer: Any,
        *,
        output: Any = None,
        collect_stats: bool = False,
        **penalty_overrides: Any,
    ) -> None:
        from ..discriminators import BatchDistanceDiscriminator
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (UnifiedV11CriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (
                isinstance(optimizer, (NormalizedOptimizer, UnifiedV11EquilibratedOptimizer))
                and hasattr(optimizer, "record")
            )
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_unified_v11_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _UnifiedV11PenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = isinstance(self.critic, nn.Module) and any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
        )
        self._has_batch_distance = isinstance(self.critic, nn.Module) and any(
            isinstance(m, BatchDistanceDiscriminator) for m in self.critic.modules()
        )
        self._has_quantized_fprop = isinstance(self.critic, nn.Module) and any(
            getattr(m, "format", None) in ("nvfp4", "mxfp8")
            or "Emulated" in type(m).__name__
            or "Quantized" in type(m).__name__
            for m in self.critic.modules()
        )

    def __call__(
        self,
        critic: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *condition: Any,
        **condition_kwargs: Any,
    ) -> torch.Tensor:
        from ..recipe_schedules import apply_penalty_schedule

        cid = id(critic)
        if cid not in self._verified_critics:
            self._ema_view(critic)
            self._verified_critics.add(cid)
        if getattr(self, "recipe", None) is not None:
            apply_penalty_schedule(self.optimizer.record.observed_steps, self.recipe, self)
        output = self.output

        def live(x: torch.Tensor) -> torch.Tensor:
            return output(critic(x, *condition, **condition_kwargs))

        base_b = int(x_real.shape[0])

        def live_rep(x: torch.Tensor, m: int) -> torch.Tensor:
            if m <= 1 or (not condition and not condition_kwargs) or x.shape[0] != m * base_b:
                return output(critic(x, *condition, **condition_kwargs))
            rep_args = tuple(
                t.repeat((m,) + (1,) * (t.ndim - 1))
                if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == base_b
                else t
                for t in condition
            )
            rep_kwargs = {
                k: (
                    v.repeat((m,) + (1,) * (v.ndim - 1))
                    if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == base_b
                    else v
                )
                for k, v in condition_kwargs.items()
            }
            return output(critic(x, *rep_args, **rep_kwargs))

        has_batch_state = isinstance(critic, torch.nn.Module) and any(
            isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == base_b
            for v in vars(critic).values()
        )
        uses_input_noise = float(getattr(critic, "std", 0.0)) > 0.0
        is_conditioned = bool(has_batch_state or condition or condition_kwargs)
        cond_tensors = [
            t.detach()
            for t in (*condition, *(condition_kwargs[k] for k in sorted(condition_kwargs)))
            if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == base_b
        ]
        cond_sig = _cond_signature((x_real, *condition), condition_kwargs)
        allow_live_reuse = (output is _first_output) and not has_batch_state
        can_concat_batches = (
            not self._has_batchnorm
            and not self._has_batch_distance
            and not has_batch_state
            and not uses_input_noise
            and (output is _first_output)
        )
        quantized_fprop = bool(
            self._has_quantized_fprop
            or (
                isinstance(critic, nn.Module)
                and any(
                    getattr(m, "format", None) in ("nvfp4", "mxfp8")
                    or "Emulated" in type(m).__name__
                    for m in critic.modules()
                )
            )
        )

        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(
            live,
            x_real,
            x_fake,
            step,
            self.collect_stats,
            is_conditioned=is_conditioned,
            cond_tensors=cond_tensors,
            cond_sig=cond_sig,
            allow_live_reuse=allow_live_reuse,
            can_concat_batches=can_concat_batches,
            live_rep=live_rep,
            quantized_fprop=quantized_fprop,
            has_batchnorm=self._has_batchnorm,
            critic_module=critic,
        )
        self.last_stats = stats
        if getattr(getattr(self, "recipe", None), "critic_step_mode", "none") == "finite_cap":
            real_c, fake_c = x_real.detach().clone(), x_fake.detach().clone()

            def finite_cap() -> torch.Tensor:
                _, sq_rc = _eval_lowd_forward_grad_sq(
                    live_rep, real_c, can_stack_basis=can_concat_batches, quantized_fprop=quantized_fprop
                )
                _, sq_fc = _eval_lowd_forward_grad_sq(
                    live_rep, fake_c, can_stack_basis=can_concat_batches, quantized_fprop=quantized_fprop
                )
                return torch.maximum((sq_rc + 1e-12).sqrt().max(), (sq_fc + 1e-12).sqrt().max())

            self.optimizer.bind_cap_panel(finite_cap, self.regularizer.kappa)
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        bank = getattr(record, "cond_fiber_bank", None)
        out.update(
            unified_v11_ell=float(getattr(record, "t1c_ell_ema", 0.0)),
            unified_v11_ell_initialized=bool(getattr(record, "t1c_ell_initialized", False)),
            unified_v11_chord_phase=float(getattr(record, "t1c_chord_phase", 0.0)),
            unified_v11_thermal_scale=float(getattr(record, "_thermal_scale", 1.0)),
            unified_v11_step_scale=float(getattr(record, "_step_scale", 1.0)),
            unified_v11_num_charts=0 if getattr(record, "chart_count", None) is None else int(record.chart_count.shape[0]),
            unified_v11_cond_strata=0 if bank is None else len(bank.strata),
        )
        return out


__all__ = [
    "UnifiedV11CriticPenalty",
    "_UnifiedV11PenaltyKernel",
]
