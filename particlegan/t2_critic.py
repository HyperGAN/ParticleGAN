"""Public facade for Track 2 Pure 1st-Order Variational Multi-Scale Secant Born-Infeld Critic."""
from ._t2_critic import (
    FROZEN_THERMAL_SCALE,
    T2SecantCriticAdam,
    T2SecantCriticPenalty,
    T2SecantGANLoss,
    T2SecantStepRecord,
    _T2SecantObserverStepRecord,
    _T2SecantPenaltyKernel,
    _attach_t2_critic_hook,
    bind_t2_record,
    compute_empf_step_scale,
    compute_thermal_scale,
    read_t2_thermal,
)

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
