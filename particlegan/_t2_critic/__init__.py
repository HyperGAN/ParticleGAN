"""Internal implementation package for Track 2 1st-order Secant Born-Infeld critic."""
from .critic import (
    FROZEN_THERMAL_SCALE,
    T2SecantCriticAdam,
    T2SecantStepRecord,
    _T2SecantObserverStepRecord,
    _attach_t2_critic_hook,
    bind_t2_record,
    compute_empf_step_scale,
    compute_thermal_scale,
    read_t2_thermal,
)
from .loss import T2SecantGANLoss
from .penalty import T2SecantCriticPenalty, _T2SecantPenaltyKernel

__all__ = [
    "FROZEN_THERMAL_SCALE",
    "T2SecantCriticAdam",
    "T2SecantCriticPenalty",
    "T2SecantGANLoss",
    "T2SecantStepRecord",
    "_T2SecantObserverStepRecord",
    "_T2SecantPenaltyKernel",
    "_attach_t2_critic_hook",
    "bind_t2_record",
    "compute_empf_step_scale",
    "compute_thermal_scale",
    "read_t2_thermal",
]
