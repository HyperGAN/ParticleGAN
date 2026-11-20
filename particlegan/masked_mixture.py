"""Observed-only mixture likelihood and an amortized component classifier.

No complete real examples, component labels or distances are needed at serving.
The synthetic classifier labels must come from the caller's own generator.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mixture_nll(observed, mask, components, *, sigma):
    """Uniform Gaussian-mixture NLL in observed coordinates, averaged per sensor.

Missing entries are removed with where before arithmetic, including NaNs.
Mixture locations retain gradients. Width is an explicit observation model,
not a change to the task's sampled MoG prior width.
"""
    if observed.ndim != 2 or mask.shape != observed.shape or components.ndim != 2 or components.shape[1] != observed.shape[1]:
        raise ValueError("expected observed/mask [B,D] and components [K,D]")
    if not isinstance(sigma, (float, int)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not bool(((mask == 0) | (mask == 1)).all()) or bool((mask.sum(1) == 0).any()):
        raise ValueError("mask must be binary with at least one observed sensor per row")
    clean = torch.where(mask.bool(), observed, 0.)
    residual = torch.where(mask[:, None].bool(), clean[:, None] - components[None], 0.)
    count = mask.sum(1)
    logp = -.5 * residual.square().sum(-1) / sigma**2 - count[:, None] * math.log(sigma * math.sqrt(2 * math.pi))
    return ((math.log(len(components)) - torch.logsumexp(logp, dim=1)) / count).mean()


class NeuralComponentImputer(nn.Module):
    """Neural conditional logits; serving draws never compute component distances."""
    def __init__(self, dimension, components, hidden=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2 * dimension, hidden), nn.LeakyReLU(.2),
                                     nn.Linear(hidden, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, components))

    def forward(self, observed, mask):
        return self.network(torch.cat((torch.where(mask.bool(), observed, 0.), mask), dim=-1))

    def synthetic_loss(self, generated, mask, component_ids):
        return F.cross_entropy(self(generated.detach(), mask), component_ids.detach())

    def draw_indices(self, observed, mask, *, draws, generator):
        return torch.multinomial(self(observed, mask).softmax(-1), draws,
                                 replacement=True, generator=generator).T


def masked_quantile_transport_loss(generated, observed, mask, *, projections=32):
    """Equal-mass projected quantile matching within each observed mask cohort.

The same generated batch and actual mask rows are reused. No complete real
data, mode labels, new noise, fitted mixture weights or altered sampling enter.
Single-row mask cohorts contribute zero because empirical spread is undefined.
"""
    from .kinetic_transport import kinetic_transport_loss
    if generated.ndim != 2 or generated.shape != observed.shape or mask.shape != observed.shape:
        raise ValueError("expected matching generated/observed/mask [B,D]")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("mask must be binary")
    _, groups = torch.unique(mask, dim=0, return_inverse=True)
    loss = generated[:0].sum()
    for group in groups.unique():
        rows = groups == group
        count = int(rows.sum())
        if count >= 2:
            keep = mask[rows].bool()
            fake = torch.where(keep, generated[rows], 0.)
            real = torch.where(keep, observed[rows], 0.)
            loss = loss + (count / len(generated)) * kinetic_transport_loss(fake, real, projections=projections)
    return loss
