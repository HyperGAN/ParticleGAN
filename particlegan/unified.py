"""Public Unified Synthesis facade (Theories 1A + 1B + 2 + 3)."""
from __future__ import annotations

from ._unified import (
    UnifiedCriticAdam,
    UnifiedCriticPenalty,
    UnifiedGANLoss,
    UnifiedPriorRegularizer,
    UnifiedStepRecord,
    _UnifiedObserverStepRecord,
    _UnifiedPenaltyKernel,
    _attach_unified_critic_hook,
    _compute_pythagorean_smoothed_cells,
    _extract_t1c_inputs_parts,
    _lookup_t1c_capture_parts,
    _record_t1c_capture,
    _validate_unified_packet,
    _validate_unified_prior_state,
    bind_unified_record,
    should_use_unified_pullback,
    unified_pooled_conditional_loss,
    unified_pullback_step,
)

__all__ = [
    "UnifiedCriticAdam",
    "UnifiedCriticPenalty",
    "UnifiedGANLoss",
    "UnifiedPriorRegularizer",
    "UnifiedStepRecord",
    "_UnifiedObserverStepRecord",
    "_UnifiedPenaltyKernel",
    "_attach_unified_critic_hook",
    "_compute_pythagorean_smoothed_cells",
    "_extract_t1c_inputs_parts",
    "_lookup_t1c_capture_parts",
    "_record_t1c_capture",
    "_validate_unified_packet",
    "_validate_unified_prior_state",
    "bind_unified_record",
    "should_use_unified_pullback",
    "unified_pooled_conditional_loss",
    "unified_pullback_step",
]
