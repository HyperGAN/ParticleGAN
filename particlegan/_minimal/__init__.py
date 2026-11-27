"""Minimal-Principle Unified Core (2 Halves, 0 autograd.grad, 0 SVD/eigh/pinv, 0 Newton-Schulz)."""
from __future__ import annotations

from .dynamics import (
    MinimalConstraintGeneratorOptimizer,
    MinimalCriticOptimizer,
    MinimalGeneratorOptimizer,
    MinimalStepRecord,
    _attach_minimal_critic_hook,
    _attach_minimal_generator_hook,
    _collect_conv_layouts,
    bind_minimal_record,
)
from .geometry import MinimalCriticPenalty, MinimalLoss
from .prior import (
    MinimalPriorRegularizer,
    minimal_pullback_step,
    should_use_minimal_pullback,
)

__all__ = [
    "MinimalConstraintGeneratorOptimizer",
    "MinimalCriticOptimizer",
    "MinimalCriticPenalty",
    "MinimalGeneratorOptimizer",
    "MinimalLoss",
    "MinimalPriorRegularizer",
    "MinimalStepRecord",
    "_attach_minimal_critic_hook",
    "_attach_minimal_generator_hook",
    "_collect_conv_layouts",
    "bind_minimal_record",
    "minimal_pullback_step",
    "should_use_minimal_pullback",
]
