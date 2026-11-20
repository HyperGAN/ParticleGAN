"""Theory 1A: Gersho Total-Variance & Kantorovich-Laguerre Pullback Prior Dynamics."""
from .pullback import T1PullbackPriorRegularizer, t1_pullback_step

__all__ = [
    "T1PullbackPriorRegularizer",
    "t1_pullback_step",
]
