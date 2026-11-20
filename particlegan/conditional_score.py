"""Sample-only proper scores for caller-owned conditional generators."""
from __future__ import annotations

import torch


def conditional_energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unbiased energy loss, averaged over a batch of conditional observations.

    ``samples`` has shape [batch, draws, *event] and ``target`` [batch, *event].
    Draws must be independent conditional on the same context. The off-diagonal
    U-statistic preserves the population score for any ensemble size >= 2;
    including diagonal pairs would incorrectly favor underdispersion. This is
    the negative of Gneiting/Raftery (2007), equation 22, with beta=1.
    The target is a fixed observation (detached). Nonfinite inputs propagate
    to the loss so the caller's finite-loss guard rejects the update.
    The caller owns sampling and every consumed RNG stream.
    """
    if samples.ndim < 3 or target.shape != samples.shape[:1] + samples.shape[2:]:
        raise ValueError("energy score expects [batch,draws,*event] and [batch,*event]")
    if samples.shape[0] == 0 or samples.shape[1] < 2 or target.numel() == 0:
        raise ValueError("energy score requires a nonempty batch/event and at least two independent draws")
    if not samples.is_floating_point() or target.device != samples.device or target.dtype != samples.dtype:
        raise ValueError("energy score requires matching floating-point device and dtype")
    x, y = samples.flatten(2), target.detach().flatten(1)
    attraction = torch.linalg.vector_norm(x - y[:, None], dim=-1).mean(1)
    separation = torch.linalg.vector_norm(x[:, :, None] - x[:, None, :], dim=-1)
    k = samples.shape[1]
    return (attraction - separation.sum((1, 2)) / (2 * k * (k - 1))).mean()
