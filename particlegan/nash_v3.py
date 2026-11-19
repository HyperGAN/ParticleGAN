"""Pure-Elementwise Imbalance-Invariant Nash Equilibrium (T4NashV3) public formulation surface.

Executes with strictly 0 SVD, 0 eigh, 0 pinv, 0 Newton-Schulz, and 0 torch.autograd.grad calls:
1. O(params) degree-0 scale-invariant peak-RMS settling + closed-form Marchenko-Pastur
   operator scaling + low-D realized pushforward & critic inversion-deficit secant bands.
2. Pure 1st-order forward-mode AD (fwAD) + live-endpoint reuse BF16 micro-secant critic.
3. Unbiased 1D quantile / real-batch Feydy-debiased Born-Infeld transport paired with
   O(BKD) debiased streaming-prototype pullback covariance isometry.
"""
from ._nash_v3 import (
    FROZEN_THERMAL_SCALE,
    NashV3ConstraintGeneratorAdam,
    NashV3CriticAdam,
    NashV3CriticPenalty,
    NashV3GeneratorAdam,
    NashV3Loss,
    NashV3StepRecord,
    _NashV3ObserverStepRecord,
    _NashV3PenaltyKernel,
    _attach_nash_v3_critic_hook,
    _attach_nash_v3_generator_hook,
    bind_nash_v3_record,
    born_infeld_debiased_transport_loss,
    compute_two_player_coupling,
    feydy_debiased_monge_displacement,
    intrinsic_sinkhorn_coupling,
    pullback_covariance_isometry_loss,
    qmc_monge_displacement,
    quantile_1d_monge_displacement,
    read_nash_v3_thermal,
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
