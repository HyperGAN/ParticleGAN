"""Public API for the Minimal-Principle Unified Core (`preset='minimal'`)."""
from ._minimal import (
    MinimalConstraintGeneratorOptimizer, MinimalCriticOptimizer, MinimalCriticPenalty,
    MinimalGeneratorOptimizer, MinimalLoss, MinimalPriorRegularizer, MinimalStepRecord,
    _attach_minimal_critic_hook, _attach_minimal_generator_hook, _collect_conv_layouts,
    bind_minimal_record, minimal_pullback_step, should_use_minimal_pullback,
)
__all__ = [
    "MinimalConstraintGeneratorOptimizer", "MinimalCriticOptimizer", "MinimalCriticPenalty",
    "MinimalGeneratorOptimizer", "MinimalLoss", "MinimalPriorRegularizer", "MinimalStepRecord",
    "_attach_minimal_critic_hook", "_attach_minimal_generator_hook", "_collect_conv_layouts",
    "bind_minimal_record", "minimal_pullback_step", "should_use_minimal_pullback",
]
