"""Relativistic Hellinger-Born-Infeld Transport (RHBI) public module.

RHBI unifies nonlocal Born-Infeld optimal transport, hollow self-coupling
entropic debiasing, and local relativistic critic log-odds ascent into a single
slack-screened ``(B + 1) x (B + 1)`` Kantorovich-Hellinger Sinkhorn operator
requiring zero RNG draws, zero schedule annealing, and zero shape gates.
"""
from __future__ import annotations

from ._rhbi.critic import (
    RHBICriticAdam,
    RHBIStepRecord,
    _RHBIObserverStepRecord,  # noqa: F401
    _attach_rhbi_critic_hook,  # noqa: F401
    _extract_critic_inputs,  # noqa: F401
    _extract_critic_inputs_parts,  # noqa: F401
    _lookup_rhbi_capture,  # noqa: F401
    _record_rhbi_capture,  # noqa: F401
    _rhbi_critic_forward_hook,  # noqa: F401
    bind_rhbi_record,
)
from ._rhbi.loss import RHBIGANLoss
from ._rhbi.penalty import RHBICriticPenalty, _RHBIPenaltyKernel  # noqa: F401
from ._rhbi.primitive import RHBILoss

__all__ = [
    "RHBICriticAdam",
    "RHBICriticPenalty",
    "RHBIGANLoss",
    "RHBILoss",
    "RHBIStepRecord",
    "bind_rhbi_record",
]
