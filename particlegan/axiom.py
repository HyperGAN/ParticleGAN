"""Public entry point for ParticleGAN's Axiom 4-Law algorithm family."""

from __future__ import annotations

from ._axiom import (
    AxiomCondFiberBank,
    AxiomCriticPenalty,
    AxiomGANLoss,
    AxiomOptimizer,
    AxiomPenaltyKernel,
    AxiomPriorRegularizer,
    AxiomStepRecord,
    AxiomTransportLoss,
    HamiltonCommitmentLedger,
    PrototypeAtlas,
    _attach_axiom_critic_hook,
    _attach_axiom_generator_hook,
    axiom_pullback_step,
    bind_axiom_record,
    collect_conv_layouts,
    prepare_axiom_transport_step,
    should_use_axiom_pullback,
)

__all__ = [
    "AxiomCondFiberBank",
    "AxiomCriticPenalty",
    "AxiomGANLoss",
    "AxiomOptimizer",
    "AxiomPenaltyKernel",
    "AxiomPriorRegularizer",
    "AxiomStepRecord",
    "AxiomTransportLoss",
    "HamiltonCommitmentLedger",
    "PrototypeAtlas",
    "_attach_axiom_critic_hook",
    "_attach_axiom_generator_hook",
    "axiom_pullback_step",
    "bind_axiom_record",
    "collect_conv_layouts",
    "prepare_axiom_transport_step",
    "should_use_axiom_pullback",
]
