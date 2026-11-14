"""Internal implementation modules for Exponential Moving Prototype Flow (EMPF)."""
from .critic import (
    EMPFCriticAdam,
    EMPFStepRecord,
    FROZEN_THERMAL_SCALE,
    _EMPFObserverStepRecord,
    bind_empf_record,
    compute_thermal_scale,
    read_empf_thermal,
)
from .loss import EMPFSinkhornRelativisticLoss
from .penalty import EMPFCriticPenalty, _EMPFPenaltyKernel
from .prior import EMPFPriorRegularizer, empf_prior_regularization_loss
from .prototypes import emeans_effective_alpha, update_emeans_prototypes
from .transport import conditional_fiber_residual, mass_weighted_sinkhorn

__all__ = [
    "EMPFCriticAdam",
    "EMPFCriticPenalty",
    "EMPFPriorRegularizer",
    "EMPFSinkhornRelativisticLoss",
    "EMPFStepRecord",
    "FROZEN_THERMAL_SCALE",
    "_EMPFObserverStepRecord",
    "_EMPFPenaltyKernel",
    "bind_empf_record",
    "compute_thermal_scale",
    "conditional_fiber_residual",
    "emeans_effective_alpha",
    "empf_prior_regularization_loss",
    "mass_weighted_sinkhorn",
    "read_empf_thermal",
    "update_emeans_prototypes",
]
