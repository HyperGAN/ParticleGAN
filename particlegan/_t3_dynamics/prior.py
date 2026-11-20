"""RGDT Prior Regularizer: kernel-variance-deconflicted k-NN blue-noise, moment envelope, and revival."""
from __future__ import annotations

import math
from typing import Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .._stats import knn_excluding_self, knn_sq_and_median_bandwidth, pairwise_sq_dist
from ..moment_tether import REFERENCE_NAMES, pooled_reference_moments

_FROZEN_THERMAL_SCALE = 1e-4


class RGDTPriorRegularizer(nn.Module):
    """Recipe-owned RGDT latent-prior regularizer (``prior_regularizer='rgdt'``).

    Deconflicts intrinsic MoG kernel entropy ``sigma_prior > 0`` from discrete
    center repulsion by softening k-NN log-repulsion at the prior's own kernel
    variance scale ``sigma_prior^2``:
    - For ``ParticlePrior`` (``sigma == 0``), preserves exact short-range k-NN
      non-collision repulsion across discrete codebooks and manifolds.
    - For ``MoGParticlePrior`` (``sigma > 0``), prevents singular ``1/r`` repulsion
      below ``sigma_prior`` from blowing co-located mode centers into inter-mode
      valleys on dense multi-mode lattices (e.g. ``grid100``).
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
        rgdt_record: Any = None,
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
        self.rgdt_record = rgdt_record

        self.prior_sigma: float = 0.0
        if reference is None:
            self.references: Tuple[Any, ...] = ()
        else:
            refs = tuple(reference) if isinstance(reference, (list, tuple)) else (reference,)
            sigmas = []
            for p in refs:
                s_attr = getattr(p, "sigma", None)
                if isinstance(s_attr, torch.Tensor) and s_attr.numel() == 1:
                    s_val = float(s_attr.detach().item())
                elif isinstance(s_attr, (int, float)) and not isinstance(s_attr, bool):
                    s_val = float(s_attr)
                else:
                    continue
                if math.isfinite(s_val) and s_val > 0.0:
                    sigmas.append(s_val)
            if sigmas:
                self.prior_sigma = max(sigmas)
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
        sigma_sq = float(self.prior_sigma) * float(self.prior_sigma)

        if self.references:
            if n < 2:
                return rows[:0].sum()
            if n <= min_dof:
                initial_mean, initial_covariance, _ = pooled_reference_moments(self.references)
                if d != len(initial_mean):
                    raise ValueError("rgdt prior rows and initial reference dimensions differ")
                ref_var = initial_covariance.diagonal().clamp_min(0.0).mean() + self.eps
                r_col_sq = ref_var * (float(n) ** (-2.0 / float(min(d, 4))))
                d2 = pairwise_sq_dist(rows)
                eye_mask = torch.eye(n, device=rows.device, dtype=torch.bool)
                d1_sq = d2.masked_fill(eye_mask, float("inf")).min(dim=1).values
                col_w = min(1.0, self.weight * 100.0)
                return col_w * F.relu(1.0 - (d1_sq + sigma_sq) / (r_col_sq + sigma_sq).clamp_min(1e-8)).pow(2).mean()
            initial_mean, initial_covariance, total_n = self._reference_moments()
            if d != len(initial_mean):
                raise ValueError("rgdt prior rows and initial reference dimensions differ")
            if total_n <= min_dof:
                ref_var = initial_covariance.diagonal().clamp_min(0.0).mean() + self.eps
                r_col_sq = ref_var * (float(n) ** (-2.0 / float(min(d, 4))))
                d2 = pairwise_sq_dist(rows)
                eye_mask = torch.eye(n, device=rows.device, dtype=torch.bool)
                d1_sq = d2.masked_fill(eye_mask, float("inf")).min(dim=1).values
                col_w = min(1.0, self.weight * 100.0)
                return col_w * F.relu(1.0 - (d1_sq + sigma_sq) / (r_col_sq + sigma_sq).clamp_min(1e-8)).pow(2).mean()
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
        std = (centered.pow(2).mean(0) + sigma_sq + self.eps).sqrt()
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
        if d == 2 and not self.references and m_sub > 64 and sigma_sq == 0.0:
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
                # Kernel-variance-softened k-NN repulsion: exact at sigma=0, non-singular inside MoG kernel balls!
                norm_sq_dist = (knn_sq_dist + float(d) * sigma_sq) / (margin * r_med_sq + float(d) * sigma_sq).clamp_min(1e-12)
                repel_loss = F.softplus(-torch.log(norm_sq_dist.clamp_min(1e-12))).mean()
            if self.isolation_weight > 0.0:
                d1 = (knn_sq_dist[:, 0] + 1e-12).sqrt()
                d1_med = d1.detach().median().clamp_min(1e-6)
                ratio_thresh = math.sqrt(self.isolation_ratio)
                excess_iso = (d1 / (ratio_thresh * d1_med)) - 1.0
                isolation_loss = F.softplus(excess_iso, beta=4.0).mean()

        global_moments = mean_error + var_loss + cov_loss
        s = float(getattr(self.rgdt_record, "_thermal_scale", 1.0)) if self.rgdt_record is not None else 1.0
        if s <= _FROZEN_THERMAL_SCALE:
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
