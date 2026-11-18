"""Riemannian Gauge-Deconflicted Transport (RGDT) internal package."""
from __future__ import annotations

from .critic import (
    RGDTCriticAdam,
    RGDTStepRecord,
    _RGDTObserverStepRecord,
    _attach_rgdt_critic_hook,
    _extract_rgdt_inputs_parts,
    _lookup_rgdt_capture_parts,
    _record_rgdt_capture,
    bind_rgdt_record,
)
from .loss import RGDTGANLoss
from .metric import (
    bipartite_deficit_macro_loss,
    born_infeld_aligned,
    born_infeld_pair,
    conformal_horizon_pair,
    pooled_tangent_conditional_loss,
    sinkhorn_slack,
)
from .penalty import RGDTCriticPenalty, _RGDTPenaltyKernel
from .prior import RGDTPriorRegularizer

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
