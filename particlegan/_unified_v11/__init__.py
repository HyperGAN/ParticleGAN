"""Unified-v11 Solution-Complete 4-Law + Zero-`eigh` CondFiber implementation package."""
from __future__ import annotations

from .bank import (
    UnifiedV11CondFiberBank,
    sym_eigensplit_zero_eigh,
    validate_v11_bank_packet,
    zero_eigh_symmetric,
)
from .critic import (
    UnifiedV11CriticAdam,
    UnifiedV11StepRecord,
    _UnifiedV11ObserverStepRecord,
    _attach_unified_v11_critic_hook,
    _attach_unified_v11_generator_hook,
    bind_unified_v11_record,
    record_unified_v11_generator_probe,
)
from .loss import (
    UnifiedV11CouplingLoss,
    UnifiedV11GANLoss,
    unified_v11_conditional_loss,
)
from .optimizer import (
    UnifiedV11EquilibratedOptimizer,
    collect_conv_layouts,
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
    zero_pinv_common_descent,
    zero_pinv_project_nonascent,
)
from .penalty import (
    UnifiedV11CriticPenalty,
    _UnifiedV11PenaltyKernel,
)
from .prior import (
    UnifiedV11PriorRegularizer,
    should_use_unified_v11_pullback,
    unified_v11_pullback_step,
)

__all__ = [
    "UnifiedV11CondFiberBank",
    "UnifiedV11CouplingLoss",
    "UnifiedV11CriticAdam",
    "UnifiedV11CriticPenalty",
    "UnifiedV11EquilibratedOptimizer",
    "UnifiedV11GANLoss",
    "UnifiedV11PriorRegularizer",
    "UnifiedV11StepRecord",
    "_UnifiedV11ObserverStepRecord",
    "_UnifiedV11PenaltyKernel",
    "_attach_unified_v11_critic_hook",
    "_attach_unified_v11_generator_hook",
    "bind_unified_v11_record",
    "collect_conv_layouts",
    "quintic_orthogonalize_2d",
    "quintic_orthogonalize_batched",
    "record_unified_v11_generator_probe",
    "should_use_unified_v11_pullback",
    "sym_eigensplit_zero_eigh",
    "unified_v11_conditional_loss",
    "unified_v11_pullback_step",
    "validate_v11_bank_packet",
    "zero_eigh_symmetric",
    "zero_pinv_common_descent",
    "zero_pinv_project_nonascent",
]
