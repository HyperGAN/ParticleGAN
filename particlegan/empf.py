"""Exponential Moving Prototype Flow (EMPF) public formulation surface.

Inspired by e-means (Bessa et al., arXiv:2611.09501), EMPF replaces raw
minibatch optimal-transport targets with an online sufficient-statistics
prototype bank ``(S, N)``, clock-free ``(K, B)`` hit-rate EMA scaling,
deterministic proportional-split dead-prototype revival, mass-weighted
entropic Sinkhorn transport chords, and differentiable latent dead-particle
revival.
"""
from ._empf import (
    EMPFCriticAdam,
    EMPFCriticPenalty,
    EMPFPriorRegularizer,
    EMPFSinkhornRelativisticLoss,
    EMPFStepRecord,
    FROZEN_THERMAL_SCALE,
    _EMPFObserverStepRecord,
    _EMPFPenaltyKernel,
    bind_empf_record,
    compute_thermal_scale,
    conditional_fiber_residual,
    emeans_effective_alpha,
    empf_prior_regularization_loss,
    mass_weighted_sinkhorn,
    read_empf_thermal,
    update_emeans_prototypes,
)

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
