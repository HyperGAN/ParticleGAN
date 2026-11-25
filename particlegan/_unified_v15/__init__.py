"""Unified-v15 4-Law implementation package."""
from __future__ import annotations

from .atomic import (
    compute_discrete_atomic_gate,
    match_inverse_atomic_targets,
    split_joint_table_pair,
    update_atomic_ledger,
)
from .charts import (
    compute_chart_valley_gate,
    deconflicted_chart_routing,
    deterministic_spatial_pool2d,
    project_chart_features,
    project_transport_features,
    update_streaming_charts,
)
from .critic import (
    UnifiedV15CriticAdam,
    UnifiedV15StepRecord,
    _UnifiedV15ObserverStepRecord,
    _attach_unified_v15_critic_hook,
    _attach_unified_v15_generator_hook,
    _eval_v15_probe_forward,
    bind_unified_v15_record,
    record_unified_v15_generator_probe,
)
from .loss import (
    UnifiedV15CouplingLoss,
    UnifiedV15GANLoss,
    unified_v15_conditional_loss,
)
from .optimizer import (
    UnifiedV15Optimizer,
    collect_conv_layouts,
)
from .penalty import (
    UnifiedV15CriticPenalty,
    _UnifiedV15PenaltyKernel,
)
from .prior import (
    UnifiedV15PriorRegularizer,
    should_use_unified_v15_pullback,
    unified_v15_pullback_step,
)

__all__ = [
    "UnifiedV15CouplingLoss",
    "UnifiedV15CriticAdam",
    "UnifiedV15CriticPenalty",
    "UnifiedV15GANLoss",
    "UnifiedV15Optimizer",
    "UnifiedV15PriorRegularizer",
    "UnifiedV15StepRecord",
    "_UnifiedV15ObserverStepRecord",
    "_UnifiedV15PenaltyKernel",
    "_attach_unified_v15_critic_hook",
    "_attach_unified_v15_generator_hook",
    "_eval_v15_probe_forward",
    "bind_unified_v15_record",
    "collect_conv_layouts",
    "compute_chart_valley_gate",
    "compute_discrete_atomic_gate",
    "deconflicted_chart_routing",
    "deterministic_spatial_pool2d",
    "match_inverse_atomic_targets",
    "project_chart_features",
    "project_transport_features",
    "record_unified_v15_generator_probe",
    "should_use_unified_v15_pullback",
    "split_joint_table_pair",
    "unified_v15_conditional_loss",
    "unified_v15_pullback_step",
    "update_atomic_ledger",
    "update_streaming_charts",
]
