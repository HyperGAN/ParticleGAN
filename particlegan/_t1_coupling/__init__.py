"""Track 1B Deconflicted Piecewise Coupling package."""
from __future__ import annotations

from .charts import (
    chart_effective_alpha,
    compute_chart_valley_gate,
    deconflicted_chart_routing,
    laguerre_hard_commitment,
    project_chart_features,
    project_transport_features,
    update_streaming_charts,
)
from .critic import (
    T1CouplingCriticAdam,
    T1CouplingStepRecord,
    _T1CouplingObserverStepRecord,
    _attach_t1_coupling_critic_hook,
    bind_t1_coupling_record,
)
from .deconflict import (
    attach_generator_deconflict_hooks,
    clear_generator_deconflict_captures,
    deconflicted_polar_step,
)
from .loss import T1CouplingGANLoss
from .penalty import T1CouplingCriticPenalty, _T1CouplingPenaltyKernel
from .primitive import T1CouplingLoss

__all__ = [
    "T1CouplingCriticAdam",
    "T1CouplingCriticPenalty",
    "T1CouplingGANLoss",
    "T1CouplingLoss",
    "T1CouplingStepRecord",
    "_T1CouplingObserverStepRecord",
    "_T1CouplingPenaltyKernel",
    "_attach_t1_coupling_critic_hook",
    "attach_generator_deconflict_hooks",
    "bind_t1_coupling_record",
    "chart_effective_alpha",
    "clear_generator_deconflict_captures",
    "compute_chart_valley_gate",
    "deconflicted_chart_routing",
    "deconflicted_polar_step",
    "laguerre_hard_commitment",
    "project_chart_features",
    "project_transport_features",
    "update_streaming_charts",
]
