"""Distribution fidelity (Feature-FID, unbiased KID), mode coverage, and critic diagnostics."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def feature_moments(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute float64 mean ``(D,)`` and sample covariance ``(D, D)``."""
    arr = features.detach().cpu().numpy().astype(np.float64)
    mu = np.mean(arr, axis=0)
    cov = np.cov(arr, rowvar=False)
    return mu, cov


def compute_fid(
    mu_real: np.ndarray,
    cov_real: np.ndarray,
    mu_fake: np.ndarray,
    cov_fake: np.ndarray,
) -> float:
    """Exact symmetric-PSD Fréchet distance between two multivariate Gaussians.

    Uses ``Tr((Sigma_r Sigma_f)^{1/2}) = sum(sqrt(eigvalsh(Sigma_r^{1/2} Sigma_f Sigma_r^{1/2})))``
    in float64, avoiding complex/singular warnings on rank-deficient covariances.
    """
    diff = mu_real - mu_fake
    sym_r = 0.5 * (cov_real + cov_real.T)
    sym_f = 0.5 * (cov_fake + cov_fake.T)
    eigvals_r, eigvecs_r = np.linalg.eigh(sym_r)
    sqrt_r = (eigvecs_r * np.sqrt(np.clip(eigvals_r, 0.0, None))) @ eigvecs_r.T
    prod = sqrt_r @ sym_f @ sqrt_r
    prod = 0.5 * (prod + prod.T)
    eigvals_prod = np.linalg.eigvalsh(prod)
    tr_covmean = float(np.sum(np.sqrt(np.clip(eigvals_prod, 0.0, None))))
    fid = float(diff @ diff + np.trace(sym_r) + np.trace(sym_f) - 2.0 * tr_covmean)
    return max(0.0, fid)


def _polynomial_mmd_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased polynomial-kernel MMD^2 (degree 3, coef0=1, gamma=1/d)."""
    m = x.shape[0]
    n = y.shape[0]
    d = float(x.shape[1])
    k_xx = (x @ x.T / d + 1.0) ** 3
    k_yy = (y @ y.T / d + 1.0) ** 3
    k_xy = (x @ y.T / d + 1.0) ** 3
    sum_xx = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1))
    sum_yy = (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1))
    sum_xy = np.sum(k_xy) / (m * n)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def compute_kid(
    feats_real: torch.Tensor,
    feats_fake: torch.Tensor,
    *,
    subset_size: int = 500,
    num_subsets: int = 20,
    seed: int = 0,
) -> Tuple[float, float]:
    """Unbiased Kernel Inception/Feature Distance (mean, std)."""
    x_all = feats_real.detach().cpu().numpy().astype(np.float64)
    y_all = feats_fake.detach().cpu().numpy().astype(np.float64)
    m = min(int(subset_size), x_all.shape[0], y_all.shape[0])
    if m < 2:
        raise ValueError("KID requires at least 2 samples per distribution")
    rng = np.random.default_rng(seed + 404)
    vals = []
    for _ in range(num_subsets):
        ix = rng.choice(x_all.shape[0], size=m, replace=False)
        iy = rng.choice(y_all.shape[0], size=m, replace=False)
        vals.append(_polynomial_mmd_unbiased(x_all[ix], y_all[iy]))
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def compute_class_coverage(
    feats_fake: torch.Tensor,
    class_centroids: torch.Tensor,
    *,
    min_mass_threshold: float = 0.01,
) -> Dict[str, float]:
    """Assign generated features to nearest real class centroid in ResNet-34 space."""
    f = feats_fake.detach().cpu().float()
    c = class_centroids.detach().cpu().float()
    num_classes = int(c.shape[0])
    dists = torch.cdist(f, c)
    assigned = dists.argmin(dim=1)
    counts = torch.bincount(assigned, minlength=num_classes).float()
    probs = counts / max(1.0, float(counts.sum().item()))
    uniform = 1.0 / float(num_classes)
    mass_tv = 0.5 * float((probs - uniform).abs().sum().item())
    covered = int((probs >= min_mass_threshold).sum().item())
    nonzero_p = probs[probs > 0]
    entropy = float(-(nonzero_p * nonzero_p.log()).sum().item())
    entropy_norm = entropy / math.log(float(num_classes)) if num_classes > 1 else 1.0
    return {
        "class_coverage": float(covered),
        "class_mass_tv": mass_tv,
        "class_entropy_norm": entropy_norm,
        "min_class_mass": float(probs.min().item()),
    }


def compute_pixel_diversity_ratio(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Ratio of generated-to-real across-sample pixel RMS (diversity indicator)."""
    r = real_images.detach().float()
    f = fake_images.detach().float()
    r_rms = float(r.std(dim=0, unbiased=False).square().mean().sqrt().item())
    f_rms = float(f.std(dim=0, unbiased=False).square().mean().sqrt().item())
    return f_rms / max(r_rms, 1e-8)


@torch.no_grad()
def compute_critic_diagnostics(
    critic: nn.Module,
    real_batch: torch.Tensor,
    fake_batch: torch.Tensor,
    *,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Side-effect-free critic logit & accuracy diagnostics under ``eval()`` + ``no_grad()``."""
    modes = [(m, m.training) for m in critic.modules()]
    device = next(critic.parameters()).device
    try:
        critic.eval()
        r_outs = []
        f_outs = []
        n = min(int(real_batch.shape[0]), int(fake_batch.shape[0]))
        for start in range(0, n, batch_size):
            r_chunk = real_batch[start : start + batch_size].to(device=device, dtype=torch.float32)
            f_chunk = fake_batch[start : start + batch_size].to(device=device, dtype=torch.float32)
            r_outs.append(critic(r_chunk).float().reshape(-1).cpu())
            f_outs.append(critic(f_chunk).float().reshape(-1).cpu())
        d_real = torch.cat(r_outs, dim=0)
        d_fake = torch.cat(f_outs, dim=0)
    finally:
        for m, flag in modes:
            m.training = flag

    real_mean = float(d_real.mean().item())
    fake_mean = float(d_fake.mean().item())
    margin = real_mean - fake_mean
    logit_std = float(torch.cat([d_real, d_fake], dim=0).std(unbiased=False).item())
    # Standard threshold accuracy (matching HyperGAN critic/accuracy) + pairwise accuracy.
    acc = 0.5 * (float((d_real > 0.0).float().mean().item()) + float((d_fake <= 0.0).float().mean().item()))
    pair_acc = float((d_real > d_fake).float().mean().item())
    return {
        "critic_real_mean": real_mean,
        "critic_fake_mean": fake_mean,
        "critic_margin": margin,
        "critic_logit_std": logit_std,
        "critic_accuracy": acc,
        "critic_pair_acc": pair_acc,
    }


def compute_grad_norm(parameters) -> float:
    sq = 0.0
    for p in parameters:
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach().float()
            sq += float(g.square().sum().item())
    return math.sqrt(sq)
