"""Internal implementation package for Pure-Elementwise T4NashV3."""
from .loss import NashV3Loss, _protected_softplus_neg
from .optim import (
    NashV3ConstraintGeneratorAdam,
    NashV3CriticAdam,
    NashV3GeneratorAdam,
)
from .penalty import NashV3CriticPenalty, _NashV3PenaltyKernel
from .record import (
    FROZEN_THERMAL_SCALE,
    NashV3StepRecord,
    _NashV3ObserverStepRecord,
    _attach_nash_v3_critic_hook,
    _attach_nash_v3_generator_hook,
    bind_nash_v3_record,
    compute_two_player_coupling,
    read_nash_v3_thermal,
)
from .transport import (
    born_infeld_debiased_transport_loss,
    feydy_debiased_monge_displacement,
    intrinsic_sinkhorn_coupling,
    pullback_covariance_isometry_loss,
    qmc_monge_displacement,
    quantile_1d_monge_displacement,
)

__all__ = [
    "FROZEN_THERMAL_SCALE",
    "NashV3ConstraintGeneratorAdam",
    "NashV3CriticAdam",
    "NashV3CriticPenalty",
    "NashV3GeneratorAdam",
    "NashV3Loss",
    "NashV3StepRecord",
    "_NashV3ObserverStepRecord",
    "_NashV3PenaltyKernel",
    "_attach_nash_v3_critic_hook",
    "_attach_nash_v3_generator_hook",
    "_protected_softplus_neg",
    "bind_nash_v3_record",
    "born_infeld_debiased_transport_loss",
    "compute_two_player_coupling",
    "feydy_debiased_monge_displacement",
    "intrinsic_sinkhorn_coupling",
    "pullback_covariance_isometry_loss",
    "qmc_monge_displacement",
    "quantile_1d_monge_displacement",
    "read_nash_v3_thermal",
]
