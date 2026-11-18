"""Riemannian Gauge-Deconflicted Transport (RGDT) public formulation module (Theory 3).

RGDT resolves Scale-Mass Gauge Non-Identifiability and Unbalanced Horizon Starvation
via three clock-free, RNG-free geometric operators:
1. Symmetric Target-Induced Conformal Riemannian Horizon ``ell_Y(u, v) = sqrt(ell_Y(u) ell_Y(v))``
   applied identically to cross-coupling and Feydy self-coupling.
2. Orphan-Gated Bipartite Monge Deficit-to-Surplus Macro-Transport coupling surplus
   fake row-slack to genuinely uncovered target regions via sharp Sinkhorn + Weyl
   inverse-CDF 1-to-1 assignment.
3. Exchangeability-Invariant Pooled-Tangent Residual Conditional Discrepancy removing
   base-manifold conditional drift ``m(c_i) - m(c_k)`` with exact zero finite-sample
   leverage bias at ``q_theta = p``.
"""
from __future__ import annotations

from ._t3_dynamics import (
    RGDTCriticAdam,
    RGDTCriticPenalty,
    RGDTGANLoss,
    RGDTPriorRegularizer,
    RGDTStepRecord,
    _RGDTObserverStepRecord,
    _RGDTPenaltyKernel,
    _attach_rgdt_critic_hook,
    _extract_rgdt_inputs_parts,
    _lookup_rgdt_capture_parts,
    _record_rgdt_capture,
    bind_rgdt_record,
    bipartite_deficit_macro_loss,
    born_infeld_aligned,
    born_infeld_pair,
    conformal_horizon_pair,
    pooled_tangent_conditional_loss,
    sinkhorn_slack,
)

__all__ = [
    "RGDTCriticAdam",
    "RGDTCriticPenalty",
    "RGDTGANLoss",
    "RGDTPriorRegularizer",
    "RGDTStepRecord",
    "_RGDTObserverStepRecord",
    "_RGDTPenaltyKernel",
    "_attach_rgdt_critic_hook",
    "_extract_rgdt_inputs_parts",
    "_lookup_rgdt_capture_parts",
    "_record_rgdt_capture",
    "bind_rgdt_record",
    "bipartite_deficit_macro_loss",
    "born_infeld_aligned",
    "born_infeld_pair",
    "conformal_horizon_pair",
    "pooled_tangent_conditional_loss",
    "sinkhorn_slack",
]
