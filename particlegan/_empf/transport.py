"""Mass-weighted entropic optimal transport between e-means prototypes and generated samples."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch

from .._stats import pairwise_sq_dist


def effective_sinkhorn_eps(ema_residual: float, sinkhorn_eps: float = 0.15) -> float:
    """State-settling Sinkhorn temperature (zero clocks).

    Preserves full global coupling ``sinkhorn_eps`` while ``ema_residual >= 0.12``
    during mode acquisition, then sharpens quadratically down to ``0.015 * sinkhorn_eps``
    as modes populate so cross-mode Sinkhorn blur disappears on dense lattices.
    """
    ratio = max(0.0, min(1.0, float(ema_residual) / 0.12))
    return float(sinkhorn_eps) * max(0.015, ratio * ratio)


def _quantile_1d_sq_residual(
    p_1d: torch.Tensor,
    w: torch.Tensor,
    f_1d: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Trimmed 1D quantile W_2^2 / scale between weighted prototypes and fake samples."""
    b_f = f_1d.shape[0]
    if b_f == 0:
        return p_1d.new_zeros(())
    p_order = torch.argsort(p_1d)
    p_sorted = p_1d[p_order]
    w_sorted = w[p_order]
    f_sorted, _ = torch.sort(f_1d)
    cdf_p = torch.cumsum(w_sorted, dim=0) - 0.5 * w_sorted
    if b_f == 1:
        f_interp = f_sorted.expand_as(p_sorted)
    else:
        pos = (cdf_p * float(b_f - 1)).clamp(0.0, float(b_f - 1))
        idx_lo = pos.floor().to(dtype=torch.long).clamp(0, b_f - 1)
        idx_hi = (idx_lo + 1).clamp(0, b_f - 1)
        frac = pos - idx_lo.to(dtype=pos.dtype)
        f_interp = (1.0 - frac) * f_sorted[idx_lo] + frac * f_sorted[idx_hi]
    sq_err = (p_sorted - f_interp).square()
    if b_f >= 32 and p_sorted.shape[0] >= 16:
        keep = (cdf_p >= 0.04) & (cdf_p <= 0.96)
        if bool(keep.any()):
            w_keep = w_sorted * keep.to(dtype=w_sorted.dtype)
            w_keep = w_keep / w_keep.sum().clamp_min(1e-12)
            return (w_keep * sq_err).sum() / scale
    return (w_sorted * sq_err).sum() / scale


def mass_weighted_sinkhorn(
    prototypes: torch.Tensor,
    weights: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.15,
    sinkhorn_iters: int = 5,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-free log-domain Sinkhorn coupling with prototype row masses ``weights``.

    Row marginals equal the normalized e-means sufficient-count shares
    ``w_k = N_k / sum_j N_j``; column marginals are uniform ``1 / B_f``.
    Returns ``(Pi, norm_cost, residual, data_scale)`` with ``residual`` in
    dimensionless linear root-mean-square transport units.
    """
    c_flat = prototypes.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_proto, b_f = c_flat.shape[0], f_flat.shape[0]
    dim = c_flat.shape[1]

    w = weights.detach().reshape(-1).to(device=c_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    raw_cost = pairwise_sq_dist(c_flat, f_flat)
    with torch.no_grad():
        scale = raw_cost.detach().median().clamp_min(eps_scale)
        data_scale = scale.sqrt()

    norm_cost = raw_cost / scale
    log_a = torch.log(w)
    log_b = -math.log(max(1, b_f))
    eps_val = max(1e-4, float(sinkhorn_eps))
    iters = max(int(sinkhorn_iters), 7) if eps_val < 0.03 else int(sinkhorn_iters)
    log_kernel = -norm_cost / eps_val

    u = torch.zeros(k_proto, device=c_flat.device, dtype=torch.float32)
    v = torch.zeros(b_f, device=c_flat.device, dtype=torch.float32)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)

    pi = torch.exp(log_kernel + u.unsqueeze(1) + v.unsqueeze(0))
    pi = pi * (w.unsqueeze(1) / pi.sum(dim=1, keepdim=True).clamp_min(1e-12))

    with torch.no_grad():
        if dim == 1 and b_f >= 16:
            mean_sq = _quantile_1d_sq_residual(c_flat[:, 0], w, f_flat[:, 0], scale)
        else:
            # Keep 100% untrimmed prototype coverage so any missing mode prevents
            # premature cooling; in the compressed-bank regime (b_f >= 4 * k_proto,
            # e.g. 256 prototypes summarizing 2048 dense lattice draws), trim the
            # top 15% inter-mode MLP bridge outliers on fake-to-prototype distances.
            min_p2f = (w * norm_cost.detach().min(dim=1).values).sum()
            f2p_all = norm_cost.detach().min(dim=0).values
            if 2 <= dim <= 4 and b_f >= max(512, 3 * k_proto):
                k_keep = max(1, int(0.85 * b_f))
                min_f2p = torch.topk(f2p_all, k=k_keep, largest=False).values.mean()
            else:
                min_f2p = f2p_all.mean()
            mean_sq = 0.5 * (min_p2f + min_f2p)
        residual = torch.sqrt(mean_sq.clamp_min(0.0))

    return (
        pi.to(dtype=x_fake.dtype),
        norm_cost.to(dtype=x_fake.dtype),
        residual.to(dtype=x_fake.dtype),
        data_scale.to(dtype=x_fake.dtype),
    )


@torch.no_grad()
def live_batch_residual(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    eps_scale: float = 1e-12,
) -> torch.Tensor:
    """Instantaneous live-batch transport residual for 1-step target-shift detection."""
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    b_r, dim = r_flat.shape
    b_f = f_flat.shape[0]
    raw_cost = pairwise_sq_dist(r_flat, f_flat)
    scale = raw_cost.median().clamp_min(eps_scale)
    if dim == 1 and b_r >= 16 and b_f >= 16:
        w = torch.full((b_r,), 1.0 / float(b_r), device=r_flat.device, dtype=torch.float32)
        mean_sq = _quantile_1d_sq_residual(r_flat[:, 0], w, f_flat[:, 0], scale)
    else:
        norm_cost = raw_cost / scale
        min_r2f = norm_cost.min(dim=1).values.mean()
        f2r_all = norm_cost.min(dim=0).values
        if dim <= 4 and b_f >= 32:
            k_keep = max(1, int(0.80 * b_f))
            min_f2r = torch.topk(f2r_all, k=k_keep, largest=False).values.mean()
        else:
            min_f2r = f2r_all.mean()
        mean_sq = 0.5 * (min_r2f + min_f2r)
    return torch.sqrt(mean_sq.clamp_min(0.0)).to(dtype=x_fake.dtype)


@torch.no_grad()
def balanced_sinkhorn_coupling(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.15,
    sinkhorn_iters: int = 5,
    eps_scale: float = 1e-12,
) -> torch.Tensor:
    """Uniform-marginal log-domain Sinkhorn coupling ``Pi`` of shape ``[B_r, B_f]``."""
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    b_r, b_f = r_flat.shape[0], f_flat.shape[0]
    raw_cost = pairwise_sq_dist(r_flat, f_flat)
    scale = raw_cost.median().clamp_min(eps_scale)
    norm_cost = raw_cost / scale
    eps_val = max(1e-4, float(sinkhorn_eps))
    iters = max(int(sinkhorn_iters), 7) if eps_val < 0.03 else int(sinkhorn_iters)
    log_a = -math.log(max(1, b_r))
    log_b = -math.log(max(1, b_f))
    log_kernel = -norm_cost / eps_val
    u = torch.zeros(b_r, device=r_flat.device, dtype=torch.float32)
    v = torch.zeros(b_f, device=r_flat.device, dtype=torch.float32)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)
    pi = torch.exp(log_kernel + u.unsqueeze(1) + v.unsqueeze(0))
    pi = pi / (float(b_r) * pi.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return pi.to(dtype=x_fake.dtype)


@torch.no_grad()
def unconditioned_chamfer_residual(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale-free symmetric Chamfer residual without Sinkhorn matrix scaling."""
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    raw_cost = pairwise_sq_dist(r_flat, f_flat)
    scale = raw_cost.median().clamp_min(eps_scale)
    data_scale = scale.sqrt()
    mean_sq = 0.5 * (raw_cost.min(dim=1).values.mean() + raw_cost.min(dim=0).values.mean()) / scale
    residual = torch.sqrt(mean_sq.clamp_min(0.0))
    return residual.to(dtype=x_fake.dtype), data_scale.to(dtype=x_fake.dtype)


@torch.no_grad()
def conditional_fiber_residual(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale-free residual for row-aligned conditioned pairs ``(x_real[j], x_fake[j] | cond[j])``."""
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    raw_cost = pairwise_sq_dist(r_flat, f_flat)
    scale = raw_cost.median().clamp_min(eps_scale)
    data_scale = scale.sqrt()
    if r_flat.shape[0] == f_flat.shape[0]:
        fiber_sq = (r_flat - f_flat).square().sum(dim=1)
        mean_sq = (fiber_sq / scale).mean()
    else:
        mean_sq = 0.5 * (raw_cost.min(dim=1).values.mean() + raw_cost.min(dim=0).values.mean()) / scale
    residual = torch.sqrt(mean_sq.clamp_min(0.0))
    return residual.to(dtype=x_fake.dtype), data_scale.to(dtype=x_fake.dtype)


@torch.no_grad()
def conditional_joint_discrepancy(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    cond_tensors: Sequence[torch.Tensor],
    *,
    eps_scale: float = 1e-12,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Zero-floor off-diagonal conditional energy U-statistic and local fiber scale ratio.

    Returns ``(u_signed, data_scale, fiber_scale_ratio)`` where:
    - ``u_signed`` has exact zero expectation under the true conditional posterior
      even when transitions are stochastic (e.g. DDGAN reverse chains).
    - ``fiber_scale_ratio`` has shape ``[B_r]`` in ``[1.0, 3.5]``, measuring how
      much tighter each row's local conditional fiber is relative to global spread.
    """
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    b_r = r_flat.shape[0]
    raw_cost = pairwise_sq_dist(r_flat, f_flat)
    scale = raw_cost.median().clamp_min(eps_scale)
    data_scale = scale.sqrt()
    ones_ratio = torch.ones(b_r, device=r_flat.device, dtype=x_fake.dtype)

    if b_r != f_flat.shape[0] or b_r < 8 or not cond_tensors:
        res, _ = conditional_fiber_residual(x_real, x_fake, eps_scale=eps_scale)
        return float(res.item() ** 2), data_scale.to(dtype=x_fake.dtype), ones_ratio

    c_parts = []
    for t in cond_tensors:
        t_f = t.detach().reshape(b_r, -1).to(device=r_flat.device, dtype=torch.float32)
        t_std = t_f.std(dim=0, unbiased=False).clamp_min(1e-4)
        c_parts.append((t_f - t_f.mean(dim=0, keepdim=True)) / t_std)
    c_norm = torch.cat(c_parts, dim=1)

    r_std_scale = r_flat.std(dim=0, unbiased=False).clamp_min(1e-4)
    r_std = r_flat / r_std_scale
    f_std = f_flat / r_std_scale

    d_cc = (pairwise_sq_dist(c_norm) + 1e-12).sqrt()
    eye = torch.eye(b_r, device=r_flat.device, dtype=torch.bool)
    w_c_full = torch.exp(-d_cc / 0.65)
    w_c = w_c_full.masked_fill(eye, 0.0)
    w_sum = w_c.sum().clamp_min(1e-12)
    w_norm = w_c / w_sum

    d_rf = (pairwise_sq_dist(r_std, f_std) + 1e-12).sqrt()
    d_rr = (pairwise_sq_dist(r_std) + 1e-12).sqrt()
    d_ff = (pairwise_sq_dist(f_std) + 1e-12).sqrt()
    u_signed = float((w_norm * (2.0 * d_rf - d_rr - d_ff)).sum().item())

    # Context-smoothed empirical conditional fiber radius (includes self-weight = 1):
    # sharp low-noise transitions (small ||x_real - x_fake||) get moderately higher slope budget.
    pair_sq = (r_flat - f_flat).square().sum(dim=1)
    w_row_norm = w_c_full / w_c_full.sum(dim=1, keepdim=True).clamp_min(1e-12)
    sigma_fiber = (0.5 * (w_row_norm @ pair_sq)).clamp_min(1e-8).sqrt()
    raw_ratio = (data_scale / sigma_fiber).clamp_min(1.0)
    fiber_ratio = (1.0 + 0.35 * (raw_ratio - 1.0)).clamp(1.0, 1.55).to(dtype=x_fake.dtype)

    return u_signed, data_scale.to(dtype=x_fake.dtype), fiber_ratio
