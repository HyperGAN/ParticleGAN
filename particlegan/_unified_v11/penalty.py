"""Unified-v11 1st-Order Spatial-Frame Forward-Tangent & Multi-Probe Secant Born-Infeld Critic Penalty."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .._unified.penalty import (
    _UnifiedPenaltyKernel,
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


class _UnifiedV11PenaltyKernel(_UnifiedPenaltyKernel):
    """Unified-v11 1st-order forward-tangent & multi-probe secant Born-Infeld kernel (`0` `autograd.grad`)."""

    def __init__(self, *, record: Any = None, **kwargs: Any) -> None:
        rec = UnifiedV11StepRecord() if record is None else record
        super().__init__(record=rec, **kwargs)
        self.arm = "unified_v11"


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
