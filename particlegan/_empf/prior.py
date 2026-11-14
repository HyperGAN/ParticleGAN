"""EMPF Prior Regularizer: initial-moment envelope, k-NN blue-noise repulsion, and dead-particle revival."""
from __future__ import annotations

import math
from typing import Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .._stats import knn_excluding_self, knn_sq_and_median_bandwidth, pairwise_sq_dist
from ..moment_tether import REFERENCE_NAMES, pooled_reference_moments
from .critic import FROZEN_THERMAL_SCALE, read_empf_thermal


def empf_prior_regularization_loss(
    z: torch.Tensor,
    *,
    weight: float = 1.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
    k_nn: int = 5,
    repel_weight: float = 0.05,
    repel_margin: float = 0.50,
    isolation_weight: float = 0.02,
    isolation_ratio: float = 25.0,
    max_samples: int = 1024,
) -> torch.Tensor:
    """Standalone EMPF prior regularizer for unanchored latent particle tables."""
    reg = EMPFPriorRegularizer(
        None,
        weight=weight,
        target_std=target_std,
        eps=eps,
        k_nn=k_nn,
        repel_weight=repel_weight,
        repel_margin=repel_margin,
        isolation_weight=isolation_weight,
        isolation_ratio=isolation_ratio,
        max_samples=max_samples,
    )
    return reg(z)


class EMPFPriorRegularizer(nn.Module):
    """Recipe-owned EMPF latent-prior regularizer (``prior_regularizer='empf'``).

    Combines three clock-free geometric terms on learned latent locations:
    1. Initial-moment tether (mean, per-coordinate variance floor, covariance
       envelope relative to captured initial R2 moments when present).
    2. Scale-free k-NN log-repulsion preventing intra-mode particle collapse.
    3. Differentiable e-means dead-particle revival tether: a smooth hinge on
       ``d_1nn^2 / (isolation_ratio * r_med^2) - 1`` that leaves multi-particle
       clusters untouched while pulling stranded outlier particles back toward
       their nearest active cluster.

    Small unanchored codebooks with ``n <= 2 * k_nn + (d if d <= 4 else 0)``
    return exact zero; small reference-anchored discrete codebooks skip global
    moment/covariance distortion and apply only a short-range one-sided
    prototype non-collision hinge below ``sigma_ref * n ** (-1 / min(d, 4))``.
    """

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
        k_nn: int = 5,
        repel_weight: float = 0.05,
        repel_margin: float = 0.25,
        isolation_weight: float = 0.02,
        isolation_ratio: float = 25.0,
        max_samples: int = 1024,
        empf_record: Any = None,
    ) -> None:
        super().__init__()
        for name, value, strict in (
            ("weight", weight, False),
            ("target_std", target_std, True),
            ("eps", eps, True),
            ("repel_weight", repel_weight, False),
            ("repel_margin", repel_margin, True),
            ("isolation_weight", isolation_weight, False),
            ("isolation_ratio", isolation_ratio, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value <= 0 if strict else value < 0)
            ):
                raise ValueError(f"{name} must be finite and {'positive' if strict else 'nonnegative'}")
        if type(k_nn) is not int or not 1 <= k_nn <= 64:
            raise ValueError("k_nn must be an integer in [1, 64]")
        if type(max_samples) is not int or max_samples < 2:
            raise ValueError("max_samples must be an integer >= 2")

        self.weight = float(weight)
        self.target_std = float(target_std)
        self.eps = float(eps)
        self.k_nn = int(k_nn)
        self.repel_weight = float(repel_weight)
        self.repel_margin = float(repel_margin)
        self.isolation_weight = float(isolation_weight)
        self.isolation_ratio = float(isolation_ratio)
        self.max_samples = int(max_samples)
        self.empf_record = empf_record

        if reference is None:
            self.references: Tuple[Any, ...] = ()
        else:
            refs = tuple(reference) if isinstance(reference, (list, tuple)) else (reference,)
            if all(all(hasattr(prior, name) for name in REFERENCE_NAMES) for prior in refs):
                self.references = refs
            else:
                self.references = ()

    def _reference_moments(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        mean, covariance, count = pooled_reference_moments(self.references)
        return mean, covariance, int(count.item())

    def _min_dof(self, d: int) -> int:
        return 2 * self.k_nn + (d if d <= 4 else 0)

    def _is_small_high_dim(self, n: int, d: int) -> bool:
        return (
            n <= 64
            and d >= 8
            and (not self.references or sum(getattr(r, "moment_tether_count", 0) for r in self.references) <= 64)
        )

    def _applied_weight(self, n: int, small_high_dim: bool) -> float:
        if small_high_dim:
            return 1.0
        if not self.references and n > 64 and 0.0 < self.weight < 0.1:
            return 1.0
        return self.weight

    def effective_weight(self, rows: torch.Tensor) -> float:
        """Return the effective scalar multiplier applied by ``forward(rows)``."""
        if self.weight == 0.0:
            return 0.0
        n, d = rows.shape
        return self._applied_weight(n, self._is_small_high_dim(n, d))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if self.weight == 0.0 or not len(rows):
            return rows[:0].sum()

        n, d = rows.shape
        min_dof = self._min_dof(d)

        if self.references:
            if n < 2:
                return rows[:0].sum()
            if n <= min_dof:
                # Small anchored discrete codebook (e.g. 5-word BiGAN table):
                # skip global moment/covariance distortion and avoid CPU-GPU
                # count.item() syncs, applying only a short-range one-sided
                # prototype non-collision hinge so noisy warmup cannot merge
                # two sigma=0 rows onto an identical point.
                initial_mean, initial_covariance, _ = pooled_reference_moments(self.references)
                if d != len(initial_mean):
                    raise ValueError("empf prior rows and initial reference dimensions differ")
                ref_var = initial_covariance.diagonal().clamp_min(0.0).mean() + self.eps
                r_col_sq = ref_var * (float(n) ** (-2.0 / float(min(d, 4))))
                d2 = pairwise_sq_dist(rows)
                eye_mask = torch.eye(n, device=rows.device, dtype=torch.bool)
                d1_sq = d2.masked_fill(eye_mask, float("inf")).min(dim=1).values
                col_w = min(1.0, self.weight * 100.0)
                return col_w * F.relu(1.0 - d1_sq / r_col_sq.clamp_min(1e-8)).pow(2).mean()
            initial_mean, initial_covariance, total_n = self._reference_moments()
            if d != len(initial_mean):
                raise ValueError("empf prior rows and initial reference dimensions differ")
            if total_n <= min_dof:
                ref_var = initial_covariance.diagonal().clamp_min(0.0).mean() + self.eps
                r_col_sq = ref_var * (float(n) ** (-2.0 / float(min(d, 4))))
                d2 = pairwise_sq_dist(rows)
                eye_mask = torch.eye(n, device=rows.device, dtype=torch.bool)
                d1_sq = d2.masked_fill(eye_mask, float("inf")).min(dim=1).values
                col_w = min(1.0, self.weight * 100.0)
                return col_w * F.relu(1.0 - d1_sq / r_col_sq.clamp_min(1e-8)).pow(2).mean()
            mean = rows.mean(0)
            variance_scale = initial_covariance.diagonal().clamp_min(0) + self.eps
            mean_error = ((mean - initial_mean).square() / variance_scale).mean()
            target_sigma = variance_scale.sqrt() * self.target_std
            target_cov = initial_covariance
            target_var = variance_scale
        else:
            if n <= min_dof or n < 2:
                return rows[:0].sum()
            mean = rows.mean(0)
            mean_error = rows.new_zeros(())
            target_sigma = torch.full((d,), self.target_std, device=rows.device, dtype=rows.dtype)
            target_var = target_sigma.pow(2) + self.eps
            target_cov = torch.diag(target_var)

        centered = rows - mean
        std = (centered.pow(2).mean(0) + self.eps).sqrt()
        var_loss = F.relu(target_sigma - std).pow(2).mean()

        if d > 1:
            cov = (centered.T @ centered) / max(n - 1, 1)
            if self.references and d <= 4 and n <= 64:
                cov_diff = cov - target_cov
                cov_loss = (cov_diff.pow(2) / (target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6)).sum()
            else:
                off_diag = cov - torch.diag(torch.diag(cov))
                cov_loss = (off_diag.pow(2) / (target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6)).sum() / (d * d)
        else:
            cov_loss = rows.new_zeros(())

        m_sub = min(n, self.max_samples)
        z_sub = rows[(torch.arange(m_sub, device=rows.device) * n) // m_sub] if m_sub < n else rows

        local_iso_loss = rows.new_zeros(())
        if d == 2 and not self.references and m_sub > 64:
            with torch.no_grad():
                k_iso = min(max(self.k_nn, d + 1), m_sub - 1)
                _, iso_knn_idx = knn_excluding_self(pairwise_sq_dist(z_sub), k_iso)
            delta_z = z_sub[iso_knn_idx] - z_sub.unsqueeze(1)
            scatter = torch.einsum("nkd,nke->nde", delta_z, delta_z) / max(1, k_iso)
            tr_s = torch.diagonal(scatter, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            s_iso = (tr_s / d) * torch.eye(d, device=rows.device, dtype=rows.dtype).unsqueeze(0)
            aniso = scatter - s_iso
            local_iso_loss = 0.10 * (
                aniso.pow(2).sum(dim=(-2, -1)) / ((tr_s.squeeze(-1).squeeze(-1) / d) + 1e-6).pow(2)
            ).mean()

        active_iso_loss = rows.new_zeros(())
        if d >= 2 and m_sub > 64:
            row_centered = z_sub - z_sub.mean(dim=0, keepdim=True)
            cov_mat = (row_centered.T @ row_centered) / max(1, m_sub - 1)
            eigs = torch.linalg.eigvalsh(cov_mat)
            l1, l2 = eigs[-1], eigs[-2]
            active_iso_loss = 0.05 * (l1 - l2).pow(2) / (l1 + l2 + 1e-6).pow(2)

        is_small_high_dim = self._is_small_high_dim(n, d)
        eff_k_nn = min(m_sub - 1, max(self.k_nn, 10) if is_small_high_dim else self.k_nn)
        eff_repel_weight = self.repel_weight * (25.0 if is_small_high_dim else 1.0)
        scale_w = self._applied_weight(n, is_small_high_dim)

        repel_loss = rows.new_zeros(())
        isolation_loss = rows.new_zeros(())
        if eff_k_nn > 0 and (eff_repel_weight > 0.0 or self.isolation_weight > 0.0):
            knn_sq_dist, r_med_sq = knn_sq_and_median_bandwidth(z_sub, eff_k_nn)
            if eff_repel_weight > 0.0:
                margin = 1.0 if (is_small_high_dim or m_sub <= 64) else self.repel_margin
                norm_sq_dist = knn_sq_dist / (margin * r_med_sq)
                repel_loss = F.softplus(-torch.log(norm_sq_dist.clamp_min(1e-12))).mean()
            if self.isolation_weight > 0.0:
                # Local 1-NN median spacing isolates genuinely stranded dead particles
                # without penalizing well-separated multi-particle clusters.
                d1 = (knn_sq_dist[:, 0] + 1e-12).sqrt()
                d1_med = d1.detach().median().clamp_min(1e-6)
                ratio_thresh = math.sqrt(self.isolation_ratio)
                excess_iso = (d1 / (ratio_thresh * d1_med)) - 1.0
                isolation_loss = F.softplus(excess_iso, beta=4.0).mean()

        global_moments = mean_error + var_loss + cov_loss
        s = read_empf_thermal(self.empf_record)
        if s <= FROZEN_THERMAL_SCALE:
            out = scale_w * (
                global_moments
                + eff_repel_weight * repel_loss
                + self.isolation_weight * isolation_loss
                + local_iso_loss
                + active_iso_loss
            )
            if d <= 4 and out.requires_grad:
                out = out * s + out.detach() * (1.0 - s)
            return out
        if global_moments.requires_grad and s < 1.0:
            global_moments = global_moments * s + global_moments.detach() * (1.0 - s)

        return scale_w * (
            global_moments
            + eff_repel_weight * repel_loss
            + self.isolation_weight * isolation_loss
            + local_iso_loss
            + active_iso_loss
        )
