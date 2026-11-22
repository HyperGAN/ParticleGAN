"""Energy-Optimal Law-4 Optimizer family (`law4_energy_v1`).

Combines Occam's Razor's causal discoveries on ``unified_v11`` without modifying
``particlegan/_unified_v11/``:

1. **Zero 4D Conv2d Newton-Schulz (Hard Invariant):** 4D convolutional kernels
   always use zero-state, zero-copy ``O(params)`` layerwise Frobenius-MP updates
   via fused ``torch._foreach_norm`` + in-place ``p.add_(g, alpha=a)`` (``0``
   matmuls, ``0`` tensor state per parameter, ``0`` temporary delta copies).
2. **Function-Space Stiffness-Gated 2D Quintic NS:** Well-conditioned 2D dense
   layers (``stiff_gate == 0.0``) use the same zero-state, zero-copy Frobenius-MP
   fast path; 5-step fused ``Z = [X, Y]`` Quintic NS activates dynamically ONLY
   on stiff 2D dense matrices (``stiff_gate > 0.0`` from ``ema_func_gain < 0.60``,
   zero architectural hardcoding).
3. **Critic Secant Probe Elimination:** Drops the proven-redundant critic logit
   secant probe while preserving the irreducible generator pushforward trust band
   (``delta_g_max`` / ``delta_g_min``) with zero-copy scalar rollback tracking.
4. **Strict Zero-Primitive & Checkpoint Contracts:** Strictly ``0`` SVD/``eigh``/
   ``pinv``/``autograd.grad``, full closed-form ``direction_blend``, and strict
   checkpoint validation + bit-exact fresh-process continuation.
"""
from .optimizer import (
    Law4EnergyOptimizer,
    Law4EnergyV2Optimizer,
    Law4EnergyV3Optimizer,
    Law4EnergyV4Optimizer,
    Law4EnergyV5Optimizer,
    Law4EnergyV6Optimizer,
    Law4EnergyV7Optimizer,
    Law4EnergyV8Optimizer,
    Law4EnergyV9Optimizer,
    Law4EnergyV10Optimizer,
    exact_polar_2xn,
)

__all__ = [
    "Law4EnergyOptimizer",
    "Law4EnergyV2Optimizer",
    "Law4EnergyV3Optimizer",
    "Law4EnergyV4Optimizer",
    "Law4EnergyV5Optimizer",
    "Law4EnergyV6Optimizer",
    "Law4EnergyV7Optimizer",
    "Law4EnergyV8Optimizer",
    "Law4EnergyV9Optimizer",
    "Law4EnergyV10Optimizer",
    "exact_polar_2xn",
]
