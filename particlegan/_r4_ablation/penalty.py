"""Round-4 Occam's Razor `-fwad-critic` ablation: 2nd-order `torch.autograd.grad` EBI critic penalty.

Pairs `_T1CouplingPenaltyKernel` (exact 2nd-order `torch.autograd.grad(..., create_graph=True)`
Born-Infeld endpoint + geodesic chord penalty) with `UnifiedV4StepRecord` /
`UnifiedV4GANLoss` / `UnifiedV4PriorRegularizer` / `UnifiedV4EquilibratedOptimizer` so that
ONLY the critic derivative computation changes (1st-order `fwAD`/secant -> 2nd-order `autograd.grad`)
for a controlled quality-vs-energy (`ms/step`, VRAM) comparison.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .._t1_coupling.penalty import _T1CouplingPenaltyKernel
from .._unified_v4.critic import (
    UnifiedV4CriticAdam,
    _attach_unified_v4_critic_hook,
)
from .._unified_v4.optimizer import UnifiedV4EquilibratedOptimizer
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output


class R4AutogradCriticPenalty(_K3PCriticPenalty):
    """Exact 2nd-order `autograd.grad` EBI critic penalty on the `unified_v4` record & optimizer stack."""

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
            isinstance(optimizer, (UnifiedV4CriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (
                isinstance(optimizer, (NormalizedOptimizer, UnifiedV4EquilibratedOptimizer))
                and hasattr(optimizer, "record")
            )
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_unified_v4_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _T1CouplingPenaltyKernel(record=optimizer.record, **options)
        self.regularizer.arm = "r4_autograd_critic"
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
        has_batch_state = isinstance(critic, torch.nn.Module) and any(
            isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == base_b
            for v in vars(critic).values()
        )
        uses_input_noise = float(getattr(critic, "std", 0.0)) > 0.0
        is_conditioned = bool(has_batch_state or condition or condition_kwargs)
        allow_concat = (
            not self._has_batchnorm
            and not self._has_batch_distance
            and not has_batch_state
            and not uses_input_noise
            and not is_conditioned
            and (output is _first_output)
        )
        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(
            live,
            x_real,
            x_fake,
            step,
            self.collect_stats,
            allow_concat=allow_concat,
            is_conditioned=is_conditioned,
        )
        self.last_stats = stats
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            unified_v4_ell=float(getattr(record, "t1c_ell_ema", 0.0)),
            unified_v4_ell_initialized=bool(getattr(record, "t1c_ell_initialized", False)),
            unified_v4_chord_phase=float(getattr(record, "t1c_chord_phase", 0.0)),
            unified_v4_thermal_scale=float(getattr(record, "_thermal_scale", 1.0)),
            unified_v4_step_scale=float(getattr(record, "_step_scale", 1.0)),
            unified_v4_num_charts=0 if getattr(record, "chart_count", None) is None else int(record.chart_count.shape[0]),
            unified_v4_cond_strata=len(getattr(record, "cond_strata", {})),
        )
        return out


__all__ = [
    "R4AutogradCriticPenalty",
]
