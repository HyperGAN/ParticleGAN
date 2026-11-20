"""CondFiber critic penalty (composing cleanly on UnifiedCriticPenalty with zero autograd.grad)."""
from __future__ import annotations

from typing import Any

from .._unified.penalty import UnifiedCriticPenalty, _UnifiedPenaltyKernel
from .critic import CondFiberStepRecord


class _CondFiberPenaltyKernel(_UnifiedPenaltyKernel):
    """Unified zero-autograd.grad secant critic penalty bound to ``CondFiberStepRecord``."""

    def __init__(self, *args: Any, record: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, record=(CondFiberStepRecord() if record is None else record), **kwargs)
        self.arm = "cond_fiber"


class CondFiberCriticPenalty(UnifiedCriticPenalty):
    """CriticPenalty facade for ``critic_formulation='cond_fiber'`` (zero autograd.grad)."""

    def __init__(
        self,
        recipe: Any,
        optimizer: Any,
        *,
        output: Any = None,
        collect_stats: bool = False,
        **penalty_overrides: Any,
    ) -> None:
        super().__init__(recipe, optimizer, output=output, collect_stats=collect_stats, **penalty_overrides)
        self.regularizer.arm = "cond_fiber"


__all__ = [
    "CondFiberCriticPenalty",
    "_CondFiberPenaltyKernel",
]
