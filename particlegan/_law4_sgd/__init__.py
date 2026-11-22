"""Causal Law-4 SGD & Asymmetric G/D Optimizer Ladder on `unified_v11` (`particlegan/_law4_sgd`).

Holds Laws 1–3 (`critic_formulation='unified_v11'`, `prior_regularizer='unified_v11'`,
`UnifiedV11StepRecord`, `UnifiedV11CondFiberBank`) 100% identical to `unified-v11`,
varying ONLY `optimizer_family` across a 5-rung (6-arm) causal ladder:

1. `law4_pure_sgd` (`Law4PureSGDOptimizer`):
   Pure unnormalized SGD (`p.add_(g, alpha=-lr)`) on G, D, and sampled prior rows
   (with zero-`pinv` `direction_blend` support and sparse sampled-row masking on prior).
2. `law4_sgd_secant` (`Law4SGDSecantOptimizer`):
   Pure unnormalized SGD wrapped ONLY by Law 4's global player-level secant trust band
   (`delta_g_min/max`, `delta_d_min/max`), with ZERO per-layer normalization, ZERO MP,
   ZERO NS, and ZERO per-parameter tensor states.
3a. `law4_sgd_g_equil_d` (`Law4SGDGEquilDOptimizer`):
   Asymmetric hybrid — Pure unnormalized SGD on G, exact `unified_v11_equil` on D and
   row-normalized prior.
3b. `law4_sgd_g_secant_equil_d` (`Law4SGDGSecantEquilDOptimizer`):
   Asymmetric hybrid companion — Pure unnormalized SGD + Law 4 generator secant trust
   band on G, exact `unified_v11_equil` on D and row-normalized prior.
4. `law4_equil_g_sgd_d` (`Law4EquilGSGDDOptimizer`):
   Causal reverse control — exact `unified_v11_equil` on G + prior, Pure unnormalized
   SGD on D.
5. `law4_frob_sgd_mp` (`Law4FrobSGDMPOptimizer`):
   Zero-state layerwise Frobenius-normalized SGD + Marchenko-Pastur RMS-to-RMS operator
   scaling (`sqrt(fan_out / fan_in)`) + degree-0 `peak_rms` settling + Law 4 secant band
   (`delta_W = -lr_MP * G / hypot(||G||_F, s_eff)`, strictly `O(params)`, 0 tensor states
   per param, 0 matmuls, 0 NS, 0 Adam moments).
"""
from __future__ import annotations

from .optimizer import (
    LAW4_SGD_FAMILIES,
    Law4EquilGSGDDOptimizer,
    Law4FrobSGDMPOptimizer,
    Law4PureSGDOptimizer,
    Law4SGDGEquilDOptimizer,
    Law4SGDGSecantEquilDOptimizer,
    Law4SGDSecantOptimizer,
)

__all__ = [
    "LAW4_SGD_FAMILIES",
    "Law4EquilGSGDDOptimizer",
    "Law4FrobSGDMPOptimizer",
    "Law4PureSGDOptimizer",
    "Law4SGDGEquilDOptimizer",
    "Law4SGDGSecantEquilDOptimizer",
    "Law4SGDSecantOptimizer",
]
