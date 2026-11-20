"""Theory 4A (T4Nash) internal implementation package."""
from .loss import T4NashLoss
from .optim import (
    T4NashConstraintGeneratorAdam,
    T4NashCriticAdam,
    T4NashGeneratorAdam,
    _op_norm_2d,
    _op_norm_batched_3d,
)
from .penalty import T4NashCriticPenalty, _T4NashPenaltyKernel
from .record import (
    FROZEN_THERMAL_SCALE,
    T4NashStepRecord,
    _T4NashObserverStepRecord,
    _attach_t4_critic_hook,
    _attach_t4_generator_hook,
    bind_t4_nash_record,
    compute_two_player_coupling,
    read_t4_thermal,
)
from .transport import (
    born_infeld_debiased_transport_loss,
    feydy_debiased_monge_displacement,
    intrinsic_sinkhorn_coupling,
    qmc_monge_displacement,
)

__all__ = [
    "FROZEN_THERMAL_SCALE",
    "T4NashConstraintGeneratorAdam",
    "T4NashCriticAdam",
    "T4NashCriticPenalty",
    "T4NashGeneratorAdam",
    "T4NashLoss",
    "T4NashStepRecord",
    "_T4NashObserverStepRecord",
    "_T4NashPenaltyKernel",
    "_attach_t4_critic_hook",
    "_attach_t4_generator_hook",
    "_op_norm_2d",
    "_op_norm_batched_3d",
    "bind_t4_nash_record",
    "born_infeld_debiased_transport_loss",
    "compute_two_player_coupling",
    "feydy_debiased_monge_displacement",
    "intrinsic_sinkhorn_coupling",
    "qmc_monge_displacement",
    "read_t4_thermal",
]
