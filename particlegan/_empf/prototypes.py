"""Online e-means sufficient-statistics prototype bank for EMPF.

Adapts the online sufficient-statistics EMA and proportional-split dead-unit
revival of e-means (Bessa et al., arXiv:2611.09501) to continuous adversarial
transport without dataset-size or epoch-horizon clocks.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .._stats import pairwise_sq_dist

# Golden-ratio and R2 plastic-constant low-discrepancy increments.
_PHI1 = 0.6180339887498949
_PHI2 = 0.7548776662466927


def emeans_effective_alpha(
    batch_size: int,
    num_prototypes: int,
    *,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.03,
    alpha_max: float = 0.35,
) -> float:
    """Clock-free (K, B) hit-rate EMA rate inspired by e-means scaling.

    In e-means, optimal EMA update rates grow sublinearly with the expected
    per-batch cluster hit count ``B / K``. Dropping dataset-size and horizon
    clocks yields ``alpha = clip(alpha_base * (B / K) ** hit_exponent,
    alpha_min, alpha_max)``.
    """
    if batch_size <= 0 or num_prototypes <= 0:
        raise ValueError("batch_size and num_prototypes must be positive")
    ratio = float(batch_size) / float(num_prototypes)
    scaled = float(alpha_base) * (ratio ** float(hit_exponent))
    return float(min(alpha_max, max(alpha_min, scaled)))


def _stratified_initial_prototypes(x: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic stratified selection of ``k`` rows from ``x`` (zero RNG)."""
    b_r, dim = x.shape
    if k >= b_r:
        return x.clone()
    idx_dim = torch.arange(1, dim + 1, device=x.device, dtype=x.dtype)
    proj_dir = torch.remainder(idx_dim * _PHI1, 1.0) - 0.5
    proj_norm = proj_dir.norm().clamp_min(1e-8)
    scores = x @ (proj_dir / proj_norm)
    order = torch.argsort(scores, stable=True)
    slots = (torch.arange(k, device=x.device, dtype=torch.long) * b_r + (b_r // 2)) // k
    slots = slots.clamp(0, b_r - 1)
    return x[order[slots]].clone()


@torch.no_grad()
def update_emeans_prototypes(
    x_real: torch.Tensor,
    proto_sum: torch.Tensor | None,
    proto_count: torch.Tensor | None,
    revival_phase: float,
    revivals: int,
    *,
    max_prototypes: int = 256,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.03,
    alpha_max: float = 0.35,
    revival_tau: float = 0.05,
    split_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """Update online (S, N) e-means prototypes and return ``(C, w, S, N, phase, revivals)``."""
    x = x_real.detach().flatten(1).to(dtype=torch.float32)
    b_r, dim = x.shape
    if b_r == 0 or dim == 0:
        raise ValueError("x_real must have positive batch size and feature dimension")

    if (
        proto_sum is None
        or proto_count is None
        or proto_sum.ndim != 2
        or proto_count.ndim != 1
        or proto_sum.shape[1] != dim
        or proto_sum.shape[0] != proto_count.shape[0]
        or proto_sum.shape[0] == 0
    ):
        k = min(int(max_prototypes), b_r)
        c_init = _stratified_initial_prototypes(x, k)
        init_count = float(b_r) / float(k)
        new_count = torch.full((k,), init_count, device=x.device, dtype=torch.float32)
        new_sum = c_init * init_count
        weights = torch.full((k,), 1.0 / float(k), device=x.device, dtype=x_real.dtype)
        return c_init.to(dtype=x_real.dtype), weights, new_sum, new_count, float(revival_phase), int(revivals)

    cur_sum = proto_sum.to(device=x.device, dtype=torch.float32)
    cur_count = proto_count.to(device=x.device, dtype=torch.float32)
    k = cur_sum.shape[0]
    centroids_prev = cur_sum / cur_count.clamp_min(1e-8).unsqueeze(1)

    d2 = pairwise_sq_dist(x, centroids_prev)
    d2_min, assign = d2.min(dim=1)

    batch_count = torch.zeros(k, device=x.device, dtype=torch.float32)
    batch_count.scatter_add_(0, assign, torch.ones(b_r, device=x.device, dtype=torch.float32))
    batch_sum = torch.zeros((k, dim), device=x.device, dtype=torch.float32)
    batch_sum.index_add_(0, assign, x)

    alpha = emeans_effective_alpha(
        b_r,
        k,
        alpha_base=alpha_base,
        hit_exponent=hit_exponent,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    # Clock-free innovation & starvation shock boost: normalized by total batch
    # pairwise spread 2 * sum(Var(X)) == E[||X - X'||^2] so 1-mode and 2-mode
    # stationary distributions stay at exact zero shock (alpha_eff == alpha) while
    # abrupt support shifts immediately accelerate prototype turnover.
    pair_spread = (2.0 * x.var(dim=0, unbiased=False).sum()).clamp_min(1e-12)
    r2p_rms_t = torch.sqrt(d2_min.mean() / pair_spread)
    if k >= 16 and b_r >= 16:
        cur_w = cur_count / cur_count.sum().clamp_min(1e-8)
        expected_zero_t = torch.exp(-float(b_r) * cur_w).mean()
        actual_zero_t = (batch_count == 0).to(dtype=torch.float32).mean()
        r2p_rms, expected_zero, actual_zero = torch.stack(
            (r2p_rms_t, expected_zero_t, actual_zero_t)
        ).tolist()
        shock_zero = max(0.0, min(1.0, (float(actual_zero) - float(expected_zero) - 0.35) / 0.25))
    else:
        r2p_rms = float(r2p_rms_t.item())
        shock_zero = 0.0
    shock_r2p = max(0.0, min(1.0, (float(r2p_rms) - 0.42) / 0.35))
    shock = max(shock_r2p, shock_zero)
    alpha_eff = float(min(0.50, alpha + (0.50 - alpha) * (shock * shock)))

    new_sum = (1.0 - alpha_eff) * cur_sum + alpha_eff * batch_sum
    new_count = (1.0 - alpha_eff) * cur_count + alpha_eff * batch_count

    mean_count = new_count.mean()
    dead_mask = new_count < (float(revival_tau) * mean_count)
    phase = float(revival_phase)
    total_revivals = int(revivals)

    if bool(dead_mask.any()) and k >= 2:
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(1).tolist()
        coord_std = x.std(dim=0, unbiased=False) + 1e-6
        dim_idx = torch.arange(1, dim + 1, device=x.device, dtype=torch.float32)

        for offset, d_idx in enumerate(dead_indices):
            donor_weights = (new_count - float(revival_tau) * mean_count).clamp_min(0.0)
            donor_weights[d_idx] = 0.0
            total_donor = float(donor_weights.sum().item())
            if total_donor <= 1e-12:
                donor_weights = new_count.clamp_min(1e-12).clone()
                donor_weights[d_idx] = 0.0
                total_donor = float(donor_weights.sum().item())
            if total_donor <= 1e-12:
                break
            donor_probs = donor_weights / total_donor
            donor_cdf = torch.cumsum(donor_probs, dim=0)
            q = (phase + (offset + 1) * _PHI2) % 1.0
            q_tensor = torch.tensor([q], device=x.device, dtype=torch.float32)
            p_idx = int(torch.searchsorted(donor_cdf, q_tensor).item())
            p_idx = max(0, min(k - 1, p_idx))
            if p_idx == d_idx:
                p_idx = (d_idx + 1) % k

            c_donor = new_sum[p_idx] / new_count[p_idx].clamp_min(1e-8)
            # Split donor + dead sufficient counts equally around donor centroid to conserve exact total mass.
            half_total = 0.5 * (new_count[p_idx] + new_count[d_idx])
            angles = dim_idx * ((phase + (offset + 1) * _PHI1) * (2.0 * math.pi))
            direction = torch.cos(angles)
            direction = direction / direction.norm().clamp_min(1e-8)
            delta = float(split_eps) * coord_std * direction
            new_count[p_idx] = half_total
            new_count[d_idx] = half_total
            new_sum[p_idx] = half_total * (c_donor - delta)
            new_sum[d_idx] = half_total * (c_donor + delta)

            total_revivals += 1

        phase = (phase + _PHI1) % 1.0

    centroids = (new_sum / new_count.clamp_min(1e-8).unsqueeze(1)).to(dtype=x_real.dtype)
    weights = (new_count / new_count.sum().clamp_min(1e-8)).to(dtype=x_real.dtype)
    return centroids, weights, new_sum, new_count, phase, total_revivals
