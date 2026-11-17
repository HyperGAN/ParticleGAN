"""Scale- & Distortion-Aware Moving Prototype Flow (SMPF) public formulation surface.

Inspired by s-means (Kocabay, Yuksel & Yalcin, 2026;
https://suayptalha.github.io/s-means-blog/) and building on ParticleGAN's
e-means-inspired EMPF, SMPF upgrades online prototype transport with:
- 4-round deterministic D^1.5 geometric coreset seeding,
- cache/SRAM-friendly tiled addmm nearest-prototype assignment across all d >= 1,
- online per-prototype distortion (sse_ema) and inter-cluster per-feature
  variance (feat_scale) driving distortion x usage^0.35 anisotropic RMS revival,
- clock-free state-settling power-cosine step decay plus progressive fast-window
  Empirical-Bayes centroid shrinkage (tau = 0.20),
- debiased physical Born-Infeld s-means prototype Sinkhorn generator transport
  with per-prototype RMS distortion shells, affine-detrended whitened conditional
  fiber transport, and connected-manifold simplicial chord alignment.
"""
from ._smpf import (
    FROZEN_THERMAL_SCALE,
    SMPFCriticAdam,
    SMPFCriticPenalty,
    SMPFGANLoss,
    SMPFPriorRegularizer,
    SMPFStepRecord,
    _SMPFObserverStepRecord,
    _SMPFPenaltyKernel,
    bind_smpf_record,
    compute_smpf_step_scale,
    compute_thermal_scale,
    conditional_fiber_residual,
    d15_initial_prototypes,
    read_smpf_thermal,
    smeans_conditional_fiber_loss,
    smeans_debiased_generator_loss,
    smeans_effective_alpha,
    smeans_prototype_sinkhorn,
    smeans_simplicial_manifold_loss,
    smpf_prior_regularization_loss,
    tiled_smeans_assign,
    update_smeans_prototypes,
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
