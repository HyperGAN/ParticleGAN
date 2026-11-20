"""Public re-export facade for Theory 4B ``t4_arch``."""
from ._t4_arch import (
    T4ArchPriorRegularizer,
    T4EquilibratedOptimizer,
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
    t4_arch_step,
)

__all__ = [
    "T4ArchPriorRegularizer",
    "T4EquilibratedOptimizer",
    "quintic_orthogonalize_2d",
    "quintic_orthogonalize_batched",
    "t4_arch_step",
]
