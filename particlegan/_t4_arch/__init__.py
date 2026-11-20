"""Internal implementation package for Theory 4B Strange Architectures & Rail Escape."""
from .optimizer import (
    T4EquilibratedOptimizer,
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
)
from .pullback import T4ArchPriorRegularizer, t4_arch_step

__all__ = [
    "T4ArchPriorRegularizer",
    "T4EquilibratedOptimizer",
    "quintic_orthogonalize_2d",
    "quintic_orthogonalize_batched",
    "t4_arch_step",
]
