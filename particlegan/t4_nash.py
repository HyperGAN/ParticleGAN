"""Function-Space Impedance-Matched Nash Equilibrium (T4Nash) public formulation surface.

Addresses Problem #4A (Player Function-Space Impedance Mismatch & Non-Complementary
Initialization Deadlock) in minimal energy (``O(B)`` / ``O(params)``, zero SVD):
1. SVD-free 2-step deterministic power-iterated Adam direction normalization plus
   exact function-space pushforward secant trust-band regulation on ``(G, D)``.
2. Clock-free two-player relative-calibration coupling gating ``G`` vs ``D`` trust
   regions by realized critic margin and empirical transport residual.
3. Scale-protected relativistic rapidity loss paired with zero-bias Feydy-debiased
   relativistic Born-Infeld prototype transport for scale-free far-field attraction.
"""
from ._t4_nash import (
    FROZEN_THERMAL_SCALE,
    T4NashConstraintGeneratorAdam,
    T4NashCriticAdam,
    T4NashCriticPenalty,
    T4NashGeneratorAdam,
    T4NashLoss,
    T4NashStepRecord,
    _T4NashObserverStepRecord,
    _T4NashPenaltyKernel,
    _attach_t4_critic_hook,
    _attach_t4_generator_hook,
    bind_t4_nash_record,
    born_infeld_debiased_transport_loss,
    compute_two_player_coupling,
    feydy_debiased_monge_displacement,
    intrinsic_sinkhorn_coupling,
    qmc_monge_displacement,
    read_t4_thermal,
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
    "bind_t4_nash_record",
    "born_infeld_debiased_transport_loss",
    "compute_two_player_coupling",
    "feydy_debiased_monge_displacement",
    "intrinsic_sinkhorn_coupling",
    "qmc_monge_displacement",
    "read_t4_thermal",
]
