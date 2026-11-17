"""Internal implementation package for Scale- & Distortion-Aware Moving Prototype Flow (SMPF)."""
from .critic import (
    FROZEN_THERMAL_SCALE,
    SMPFCriticAdam,
    SMPFStepRecord,
    _SMPFObserverStepRecord,
    bind_smpf_record,
    compute_smpf_step_scale,
    compute_thermal_scale,
    read_smpf_thermal,
)
from .loss import SMPFGANLoss
from .penalty import SMPFCriticPenalty, _SMPFPenaltyKernel
from .prior import SMPFPriorRegularizer, smpf_prior_regularization_loss
from .prototypes import (
    d15_initial_prototypes,
    smeans_effective_alpha,
    tiled_smeans_assign,
    update_smeans_prototypes,
)
from .transport import (
    conditional_fiber_residual,
    smeans_conditional_fiber_loss,
    smeans_debiased_generator_loss,
    smeans_prototype_sinkhorn,
    smeans_simplicial_manifold_loss,
)

__all__ = [
    "FROZEN_THERMAL_SCALE",
    "SMPFCriticAdam",
    "SMPFCriticPenalty",
    "SMPFGANLoss",
    "SMPFPriorRegularizer",
    "SMPFStepRecord",
    "_SMPFObserverStepRecord",
    "_SMPFPenaltyKernel",
    "bind_smpf_record",
    "compute_smpf_step_scale",
    "compute_thermal_scale",
    "conditional_fiber_residual",
    "d15_initial_prototypes",
    "read_smpf_thermal",
    "smeans_conditional_fiber_loss",
    "smeans_debiased_generator_loss",
    "smeans_effective_alpha",
    "smeans_prototype_sinkhorn",
    "smeans_simplicial_manifold_loss",
    "smpf_prior_regularization_loss",
    "tiled_smeans_assign",
    "update_smeans_prototypes",
]
