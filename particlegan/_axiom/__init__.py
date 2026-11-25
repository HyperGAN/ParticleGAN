"""Axiom 4-Law generative training algorithm family."""
from __future__ import annotations

from .critic import (
    AxiomCriticPenalty,
    AxiomPenaltyKernel,
    AxiomStepRecord,
    _attach_axiom_critic_hook,
    _attach_axiom_generator_hook,
    bind_axiom_record,
    eval_probe_forward,
)
from .loss import AxiomGANLoss
from .optimizer import (
    AxiomOptimizer,
    collect_conv_layouts,
    fp32_tikhonov_ns_2d,
    fp32_tikhonov_ns_batched,
    mantissa_exponent_normalize,
    zero_pinv_project_nonascent,
)
from .prior import (
    AxiomPriorRegularizer,
    allocate_deficit_quotas,
    axiom_pullback_step,
    central_pullback_cov,
    coalesce_pythagorean_ltv,
    has_positive_prior_sigma,
    should_use_axiom_pullback,
    validate_axiom_prior_state,
)
from .transport import (
    AxiomCondFiberBank,
    AxiomTransportLoss,
    compute_discrete_atomic_gate,
    project_transport_features,
    sym_eigensplit_zero_eigh,
    zero_eigh_symmetric,
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
    "_attach_axiom_critic_hook",
    "_attach_axiom_generator_hook",
    "allocate_deficit_quotas",
    "axiom_pullback_step",
    "bind_axiom_record",
    "central_pullback_cov",
    "coalesce_pythagorean_ltv",
    "collect_conv_layouts",
    "compute_discrete_atomic_gate",
    "eval_probe_forward",
    "fp32_tikhonov_ns_2d",
    "fp32_tikhonov_ns_batched",
    "has_positive_prior_sigma",
    "mantissa_exponent_normalize",
    "project_transport_features",
    "should_use_axiom_pullback",
    "sym_eigensplit_zero_eigh",
    "validate_axiom_prior_state",
    "zero_eigh_symmetric",
    "zero_pinv_project_nonascent",
]
