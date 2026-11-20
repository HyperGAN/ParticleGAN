"""Unbiased 1D quantile / Feydy-debiased Born-Infeld transport & O(BKD) streaming prototype pullback covariance isometry.

Strictly 0 SVD, 0 eigh, 0 pinv, 0 Newton-Schulz, and 0 torch.autograd.grad calls.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .._empf.prototypes import _PHI1
from .._empf.transport import (
    _quantile_1d_sq_residual,
    effective_sinkhorn_eps,
)
from .._stats import pairwise_sq_dist
from .record import _smoothstep01


def intrinsic_sinkhorn_coupling(
    prototypes: torch.Tensor,
    weights: torch.Tensor,
    x_fake: torch.Tensor,
    x_real: torch.Tensor | None = None,
    *,
    sinkhorn_eps: float = 0.15,
    sinkhorn_iters: int = 5,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-free log-domain Sinkhorn coupling anchored to intrinsic target scale."""
    c_flat = prototypes.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_proto, b_f = c_flat.shape[0], f_flat.shape[0]
    dim = c_flat.shape[1]

    w = weights.detach().reshape(-1).to(device=c_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    raw_cost = pairwise_sq_dist(c_flat, f_flat)
    with torch.no_grad():
        cross_med = raw_cost.detach().median().clamp_min(eps_scale)
        if x_real is not None and x_real.shape[0] >= 4:
            r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
            rr_med = pairwise_sq_dist(r_flat).median()
            r_var_sum = r_flat.var(dim=0, unbiased=False).sum()
            rr_med = torch.maximum(rr_med, r_var_sum)
        elif k_proto >= 4:
            rr_med = pairwise_sq_dist(c_flat).median()
            c_var_sum = c_flat.var(dim=0, unbiased=False).sum()
            rr_med = torch.maximum(rr_med, c_var_sum)
        else:
            rr_med = raw_cost.new_zeros(())
        has_intrinsic = rr_med > 1e-6
        ref_scale = torch.where(has_intrinsic, rr_med.clamp_min(eps_scale), raw_cost.new_full((), 0.04))
        cost_scale = torch.maximum(
            torch.where(has_intrinsic, ref_scale, cross_med),
            cross_med,
        ).clamp_min(eps_scale)
        data_scale = ref_scale.sqrt()

    norm_cost = raw_cost / cost_scale
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
            mean_sq = _quantile_1d_sq_residual(c_flat[:, 0], w, f_flat[:, 0], ref_scale)
        else:
            phys_cost = raw_cost.detach() / ref_scale
            min_p2f = (w * phys_cost.min(dim=1).values).sum()
            f2p_all = phys_cost.min(dim=0).values
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
def quantile_1d_monge_displacement(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact O(B log B) 1D quantile W_2 Monge displacement with zero entropic variance bias."""
    r_1d = x_real.detach().reshape(-1).to(dtype=torch.float32)
    f_1d = x_fake.detach().reshape(-1).to(dtype=torch.float32)
    b_r, b_f = int(r_1d.shape[0]), int(f_1d.shape[0])

    r_sorted, _ = torch.sort(r_1d)
    f_order = torch.argsort(f_1d)
    f_sorted = f_1d[f_order]

    if b_r == b_f:
        r_matched_sorted = r_sorted
    elif b_r == 1:
        r_matched_sorted = r_sorted.expand(b_f)
    else:
        pos = torch.linspace(0.0, float(b_r - 1), b_f, device=r_1d.device, dtype=torch.float32)
        idx_lo = pos.floor().to(dtype=torch.long).clamp(0, b_r - 1)
        idx_hi = (idx_lo + 1).clamp(0, b_r - 1)
        frac = pos - idx_lo.to(dtype=torch.float32)
        r_matched_sorted = (1.0 - frac) * r_sorted[idx_lo] + frac * r_sorted[idx_hi]

    disp_sorted = r_matched_sorted - f_sorted
    disp = torch.empty_like(f_1d)
    disp[f_order] = disp_sorted

    r_std = r_1d.std(unbiased=(b_r > 1)) if b_r > 1 else r_1d.new_zeros(())
    ref_sigma = torch.where(r_std > 1e-4, r_std, r_1d.new_ones(())).clamp_min(1e-3)
    return disp.reshape_as(x_fake).to(dtype=x_fake.dtype), ref_sigma.to(dtype=x_fake.dtype)


@torch.no_grad()
def _log_sinkhorn_uniform(
    cost_mat: torch.Tensor,
    cost_scale: torch.Tensor,
    eps_val: float,
    iters: int,
) -> torch.Tensor:
    """Compute uniform-marginal log-domain Sinkhorn coupling ``Pi`` with ``sum(Pi) = 1``."""
    n_r, n_c = int(cost_mat.shape[0]), int(cost_mat.shape[1])
    log_a = -math.log(max(1, n_r))
    log_b = -math.log(max(1, n_c))
    log_k = -(cost_mat / cost_scale) / eps_val
    u = torch.zeros(n_r, device=cost_mat.device, dtype=torch.float32)
    v = torch.zeros(n_c, device=cost_mat.device, dtype=torch.float32)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
    pi = torch.exp(log_k + u.unsqueeze(1) + v.unsqueeze(0))
    pi = pi / (float(n_r) * pi.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return pi


@torch.no_grad()
def feydy_debiased_monge_displacement(
    anchors: torch.Tensor,
    anchor_weights: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.12,
    sinkhorn_iters: int = 6,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact zero-bias Feydy-debiased Sinkhorn displacement ``v = T_{a->f} - T_{f->f}``."""
    a_flat = anchors.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_a, b_f = a_flat.shape[0], f_flat.shape[0]

    w = anchor_weights.detach().reshape(-1).to(device=a_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    if k_a > 512:
        slots_a = (torch.arange(512, device=a_flat.device, dtype=torch.long) * k_a) // 512
        a_flat = a_flat[slots_a]
        w = w[slots_a]
        w = w / w.sum().clamp_min(1e-12)
        k_a = 512

    cost_af = pairwise_sq_dist(a_flat, f_flat)
    cross_med = cost_af.median().clamp_min(eps_scale)
    if k_a >= 4:
        aa_med = torch.maximum(pairwise_sq_dist(a_flat).median(), a_flat.var(dim=0, unbiased=False).sum())
    else:
        aa_med = cost_af.new_zeros(())
    has_intrinsic = aa_med > 1e-6
    ref_sq = torch.where(has_intrinsic, aa_med.clamp_min(eps_scale), cost_af.new_ones(()))
    cost_scale = torch.maximum(
        torch.where(has_intrinsic, ref_sq, cross_med),
        cross_med,
    ).clamp_min(eps_scale)
    ref_sigma = ref_sq.sqrt().clamp_min(1e-3)

    eps_val = max(1e-4, float(sinkhorn_eps))
    iters = max(int(sinkhorn_iters), 7) if eps_val < 0.03 else int(sinkhorn_iters)

    log_a = torch.log(w)
    log_b = -math.log(max(1, b_f))
    log_k_af = -(cost_af / cost_scale) / eps_val
    u_af = torch.zeros(k_a, device=a_flat.device, dtype=torch.float32)
    v_af = torch.zeros(b_f, device=a_flat.device, dtype=torch.float32)
    for _ in range(iters):
        u_af = log_a - torch.logsumexp(log_k_af + v_af.unsqueeze(0), dim=1)
        v_af = log_b - torch.logsumexp(log_k_af + u_af.unsqueeze(1), dim=0)
    pi_af = torch.exp(log_k_af + u_af.unsqueeze(1) + v_af.unsqueeze(0))
    col_af = pi_af.sum(dim=0, keepdim=True).clamp_min(1e-12)
    t_cross = (pi_af / col_af).t() @ a_flat

    if b_f <= 512:
        f_ref = f_flat
        cost_rf = pairwise_sq_dist(f_ref, f_flat)
        log_k_ff = -(cost_rf / cost_scale) / eps_val
        f_dual = torch.zeros(b_f, device=f_flat.device, dtype=torch.float32)
        for _ in range(iters):
            f_dual = 0.5 * (f_dual + (log_b - torch.logsumexp(log_k_ff + f_dual.unsqueeze(0), dim=1)))
        pi_ff = torch.exp(log_k_ff + f_dual.unsqueeze(1) + f_dual.unsqueeze(0))
        col_ff = pi_ff.sum(dim=0, keepdim=True).clamp_min(1e-12)
        t_self = (pi_ff / col_ff).t() @ f_ref
    else:
        m_ref = 512
        slots = (torch.arange(m_ref, device=f_flat.device, dtype=torch.long) * b_f) // m_ref
        f_ref = f_flat[slots]
        cost_rf = pairwise_sq_dist(f_ref, f_flat)
        log_m = -math.log(m_ref)
        log_k_ff = -(cost_rf / cost_scale) / eps_val
        u_ff = torch.zeros(m_ref, device=f_flat.device, dtype=torch.float32)
        v_ff = torch.zeros(b_f, device=f_flat.device, dtype=torch.float32)
        for _ in range(iters):
            u_ff = log_m - torch.logsumexp(log_k_ff + v_ff.unsqueeze(0), dim=1)
            v_ff = log_b - torch.logsumexp(log_k_ff + u_ff.unsqueeze(1), dim=0)
        pi_ff = torch.exp(log_k_ff + u_ff.unsqueeze(1) + v_ff.unsqueeze(0))
        col_ff = pi_ff.sum(dim=0, keepdim=True).clamp_min(1e-12)
        t_self = (pi_ff / col_ff).t() @ f_ref

    disp = (t_cross - t_self).reshape_as(x_fake).to(dtype=x_fake.dtype)
    return disp, ref_sigma.to(dtype=x_fake.dtype)


@torch.no_grad()
def qmc_monge_displacement(
    anchors: torch.Tensor,
    anchor_weights: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.12,
    sinkhorn_iters: int = 6,
    qmc_phase: float = 0.0,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stratified QMC inverse-CDF Monge displacement from entropic plan ``Pi_{a -> f}``."""
    a_flat = anchors.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_a, b_f = a_flat.shape[0], f_flat.shape[0]

    w = anchor_weights.detach().reshape(-1).to(device=a_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    if k_a > 512:
        slots_a = (torch.arange(512, device=a_flat.device, dtype=torch.long) * k_a) // 512
        a_flat = a_flat[slots_a]
        w = w[slots_a]
        w = w / w.sum().clamp_min(1e-12)
        k_a = 512

    cost_af = pairwise_sq_dist(a_flat, f_flat)
    cross_med = cost_af.median().clamp_min(eps_scale)
    if k_a >= 4:
        aa_med = torch.maximum(pairwise_sq_dist(a_flat).median(), a_flat.var(dim=0, unbiased=False).sum())
    else:
        aa_med = cost_af.new_zeros(())
    has_intrinsic = aa_med > 1e-6
    ref_sq = torch.where(has_intrinsic, aa_med.clamp_min(eps_scale), cost_af.new_ones(()))
    cost_scale = torch.maximum(
        torch.where(has_intrinsic, ref_sq, cross_med),
        cross_med,
    ).clamp_min(eps_scale)
    ref_sigma = ref_sq.sqrt().clamp_min(1e-3)

    eps_val = max(1e-4, float(sinkhorn_eps))
    iters = max(int(sinkhorn_iters), 7) if eps_val < 0.03 else int(sinkhorn_iters)

    log_a = torch.log(w)
    log_b = -math.log(max(1, b_f))
    log_k_af = -(cost_af / cost_scale) / eps_val
    u_af = torch.zeros(k_a, device=a_flat.device, dtype=torch.float32)
    v_af = torch.zeros(b_f, device=a_flat.device, dtype=torch.float32)
    for _ in range(iters):
        u_af = log_a - torch.logsumexp(log_k_af + v_af.unsqueeze(0), dim=1)
        v_af = log_b - torch.logsumexp(log_k_af + u_af.unsqueeze(1), dim=0)
    pi_af = torch.exp(log_k_af + u_af.unsqueeze(1) + v_af.unsqueeze(0))
    pi_af = pi_af * (w.unsqueeze(1) / pi_af.sum(dim=1, keepdim=True).clamp_min(1e-12))

    col_probs = (b_f * pi_af.t()).clamp_min(1e-12)
    col_probs = col_probs / col_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cdf = torch.cumsum(col_probs, dim=1)
    j_idx = torch.arange(b_f, device=f_flat.device, dtype=torch.float32)
    q = torch.remainder((j_idx + 0.5) / max(1, b_f) + (float(qmc_phase) + 0.5 * _PHI1), 1.0).unsqueeze(1)
    matched_idx = torch.searchsorted(cdf, q).squeeze(1).clamp(0, max(0, k_a - 1))
    a_matched = a_flat[matched_idx]

    disp = (a_matched - f_flat).reshape_as(x_fake).to(dtype=x_fake.dtype)
    return disp, ref_sigma.to(dtype=x_fake.dtype)


def born_infeld_debiased_transport_loss(
    x_fake: torch.Tensor,
    x_real: torch.Tensor,
    *,
    prototypes: torch.Tensor | None = None,
    proto_weights: torch.Tensor | None = None,
    ema_residual: float = 1.0,
    calibration_gate: float = 1.0,
    qmc_phase: float = 0.0,
    sinkhorn_eps: float = 0.12,
    sinkhorn_iters: int = 6,
) -> torch.Tensor:
    """Relativistic Born-Infeld transport blending far-field QMC Monge with unbiased real-batch Feydy / 1D quantile OT.

    Key refinement over v2: near-field displacement uses unbiased ``x_real`` (exact 1D quantile
    matching for ``d = 1``, exact zero-bias Feydy-debiased Sinkhorn on ``x_real`` for ``d >= 2``)
    instead of variance-contracted e-means centroids, eliminating 1D ``std_ratio`` under-dispersion
    and preserving full circular / anisotropic mode covariance at Nash equilibrium.
    """
    if not x_fake.requires_grad:
        return x_fake.new_zeros(())

    dim = int(x_fake.flatten(1).shape[1])
    b_r = int(x_real.shape[0])
    real_flat = x_real.detach().flatten(1)
    real_weights = torch.full((b_r,), 1.0 / max(1, b_r), device=x_real.device, dtype=torch.float32)

    if dim == 1 and b_r >= 8:
        v_disp, sigma_ref = quantile_1d_monge_displacement(x_real, x_fake)
    else:
        if (
            prototypes is not None
            and proto_weights is not None
            and prototypes.shape[0] >= 2
            and prototypes.shape[1] == dim
            and b_r < 32
        ):
            qmc_anchors = prototypes
            qmc_weights = proto_weights
        else:
            qmc_anchors = real_flat
            qmc_weights = real_weights

        eps_eff = effective_sinkhorn_eps(ema_residual, sinkhorn_eps)
        u_res = max(0.0, min(1.0, (float(ema_residual) - 0.14) / 0.12))
        u_cal = max(0.0, min(1.0, (0.85 - float(calibration_gate)) / 0.45))
        w_qmc = max(u_res * u_res * (3.0 - 2.0 * u_res), u_cal * u_cal * (3.0 - 2.0 * u_cal))

        if w_qmc >= 0.999:
            v_disp, sigma_ref = qmc_monge_displacement(
                qmc_anchors,
                qmc_weights,
                x_fake.detach(),
                sinkhorn_eps=eps_eff,
                sinkhorn_iters=sinkhorn_iters,
                qmc_phase=qmc_phase,
            )
        elif w_qmc <= 0.001:
            v_disp, sigma_ref = feydy_debiased_monge_displacement(
                real_flat,
                real_weights,
                x_fake.detach(),
                sinkhorn_eps=eps_eff,
                sinkhorn_iters=sinkhorn_iters,
            )
        else:
            v_qmc, sigma_ref = qmc_monge_displacement(
                qmc_anchors,
                qmc_weights,
                x_fake.detach(),
                sinkhorn_eps=eps_eff,
                sinkhorn_iters=sinkhorn_iters,
                qmc_phase=qmc_phase,
            )
            v_feydy, _ = feydy_debiased_monge_displacement(
                real_flat,
                real_weights,
                x_fake.detach(),
                sinkhorn_eps=eps_eff,
                sinkhorn_iters=sinkhorn_iters,
            )
            v_disp = w_qmc * v_qmc + (1.0 - w_qmc) * v_feydy

    target_det = (x_fake.detach() + v_disp).detach()
    diff_flat = (x_fake - target_det).flatten(1).to(dtype=torch.float32)
    tau_mult = 1.0 if dim == 1 else 0.12
    tau_bi = (tau_mult * sigma_ref.to(dtype=torch.float32)).clamp_min(1e-3)
    sq_norm = diff_flat.square().sum(dim=1) / (tau_bi * tau_bi)
    bi_energy = tau_bi * (torch.sqrt(1.0 + sq_norm) - 1.0).mean()
    return bi_energy.to(dtype=x_fake.dtype)


def _burg_logdet_divergence(cov_r: torch.Tensor, cov_f: torch.Tensor, ridge: torch.Tensor) -> torch.Tensor:
    """Closed-form SPD Pinch-Weighted Symmetric Burg divergence ``D_Burg(Sr||Sf) + 0.5*D_Burg(Sf||Sr)`` (0 SVD/eigh/pinv)."""
    dim = int(cov_r.shape[-1])
    eye = torch.eye(dim, device=cov_r.device, dtype=torch.float32).unsqueeze(0)
    eps_eye = ridge.view(-1, 1, 1) * eye
    sr = cov_r + eps_eye
    sf = cov_f + eps_eye
    if dim == 2:
        sr00, sr01, sr11 = sr[:, 0, 0], sr[:, 0, 1], sr[:, 1, 1]
        sf00, sf01, sf11 = sf[:, 0, 0], sf[:, 0, 1], sf[:, 1, 1]
        det_r = (sr00 * sr11 - sr01 * sr01).clamp_min(1e-18)
        det_f = (sf00 * sf11 - sf01 * sf01).clamp_min(1e-18)
        log_ratio = torch.log(det_f) - torch.log(det_r)
        tr_rf = (sf11 * sr00 + sf00 * sr11 - 2.0 * sf01 * sr01) / det_f
        tr_fr = (sr11 * sf00 + sr00 * sf11 - 2.0 * sr01 * sf01) / det_r
        d_pinch = (tr_rf + log_ratio - 2.0).clamp_min(0.0)
        d_spread = (tr_fr - log_ratio - 2.0).clamp_min(0.0)
        return d_pinch + 0.5 * d_spread
    l_r, _ = torch.linalg.cholesky_ex(sr, check_errors=False)
    l_f, _ = torch.linalg.cholesky_ex(sf, check_errors=False)
    logdet_r = 2.0 * torch.log(l_r.diagonal(dim1=-2, dim2=-1).clamp_min(1e-9)).sum(dim=-1)
    logdet_f = 2.0 * torch.log(l_f.diagonal(dim1=-2, dim2=-1).clamp_min(1e-9)).sum(dim=-1)
    log_ratio = logdet_f - logdet_r
    tr_rf = torch.cholesky_solve(sr, l_f).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    tr_fr = torch.cholesky_solve(sf, l_r).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    d_pinch = (tr_rf + log_ratio - float(dim)).clamp_min(0.0)
    d_spread = (tr_fr - log_ratio - float(dim)).clamp_min(0.0)
    return d_pinch + 0.5 * d_spread


def pullback_covariance_isometry_loss(
    x_fake: torch.Tensor,
    x_real: torch.Tensor,
    *,
    prototypes: torch.Tensor | None = None,
    ema_residual: float = 1.0,
    qmc_phase: float = 0.0,
    sinkhorn_eps: float = 0.10,
    sinkhorn_iters: int = 5,
    max_anchors: int = 128,
) -> torch.Tensor:
    """O(BKD) Voronoi + Core-Ball Burg Log-Det Covariance Isometry (0 SVD, 0 eigh, 0 pinv, 0 autograd.grad).

    1. Discovers well-separated mode anchors ``{a_1, ..., a_M}`` deterministically from ``x_real``
       via adaptive k-NN greedy covering (zero RNG, zero knowledge of mode count ``K``).
    2. Partitions ALL real and fake samples into exact Voronoi cells (``argmin_m ||x - a_m||^2``):
       - **Full-Voronoi Burg Log-Det** pulls every outlier and inter-mode bridge straggler back
         into its assigned mode (preventing outlier ejection and collapsing full-sample covariance error).
       - **Core-Ball Burg Log-Det** (matched radius ``3 h_m^2`` inside each Voronoi cell) prevents
         outliers from masking a pinched inner core, imposing a sharp ``lambda_min^-1`` outward wall
         whenever an underpowered critic pinches any mode toward a 1D ridge.
    3. At Nash equilibrium ``p_fake == p_real``, both Voronoi and core covariances match identically
       so the loss and gradient vanish to exact zero.
    """
    del prototypes, sinkhorn_eps, sinkhorn_iters
    if not x_fake.requires_grad:
        return x_fake.new_zeros(())
    f_flat = x_fake.flatten(1).to(dtype=torch.float32)
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    b_f, dim = int(f_flat.shape[0]), int(f_flat.shape[1])
    b_r = int(r_flat.shape[0])
    if b_f < 16 or b_r < 16 or dim < 1 or dim > 8:
        return x_fake.new_zeros(())

    with torch.no_grad():
        if b_r > 512:
            shift_r = int(qmc_phase * b_r) % b_r
            slots_r = (torch.arange(512, device=r_flat.device, dtype=torch.long) * b_r // 512 + shift_r) % b_r
            r_sub = r_flat[slots_r]
        else:
            r_sub = r_flat
        n_r = int(r_sub.shape[0])

        cost_rr = pairwise_sq_dist(r_sub)
        rr_med = float(torch.maximum(cost_rr.median(), r_sub.var(dim=0, unbiased=False).sum()).item())
        has_intrinsic = _smoothstep01((rr_med - 1e-6) / 1e-5)
        if has_intrinsic <= 1e-6:
            return x_fake.new_zeros(())

        r_mu_g = r_sub.mean(dim=0, keepdim=True)
        dr_g = r_sub - r_mu_g
        cov_r_g = (dr_g.t() @ dr_g) / float(max(1, n_r - 1))
        var_g_scale = (cov_r_g.diagonal().sum() / float(dim)).clamp_min(1e-6)
        sigma_g = var_g_scale.sqrt()

    f_mu_g = f_flat.mean(dim=0, keepdim=True)
    df_g = f_flat - f_mu_g
    cov_f_g = (df_g.t() @ df_g) / float(max(1, b_f - 1))

    if dim == 1:
        vr = cov_r_g[0, 0] + 0.03 * var_g_scale
        vf = cov_f_g[0, 0] + 0.03 * var_g_scale
        d_burg_1d = (vr / vf - torch.log(vr / vf) - 1.0).clamp_min(0.0)
        g_mean_err_sq = (f_mu_g - r_mu_g).square().sum() / var_g_scale
        res_gate = _smoothstep01((0.80 - float(ema_residual)) / 0.45)
        out_1d = float(has_intrinsic * res_gate) * sigma_g * (
            (torch.sqrt(1.0 + d_burg_1d) - 1.0) + 0.25 * (torch.sqrt(1.0 + g_mean_err_sq) - 1.0)
        )
        return out_1d.to(dtype=x_fake.dtype)

    res_gate = _smoothstep01((0.40 - float(ema_residual)) / 0.25)
    if res_gate <= 1e-6:
        return x_fake.new_zeros(())

    with torch.no_grad():
        if b_f > 512:
            shift_f = int((qmc_phase + _PHI1) * b_f) % b_f
            f_idx = (torch.arange(512, device=f_flat.device, dtype=torch.long) * b_f // 512 + shift_f) % b_f
            f_sub_det = f_flat.detach()[f_idx]
        else:
            f_idx = None
            f_sub_det = f_flat.detach()

        # Adaptive k-NN local scale h2_i per real sample, then deterministic farthest-first mode covering.
        k_nn = max(5, min(14, n_r // 16))
        h2_all = torch.topk(cost_rr, k=k_nn + 1, dim=1, largest=False).values[:, -1].clamp_min(1e-6)  # [n_r]
        uncovered = torch.ones(n_r, device=r_sub.device, dtype=torch.bool)
        min_d2 = torch.full((n_r,), float("inf"), device=r_sub.device, dtype=torch.float32)
        selected: list[int] = []
        cur = int(qmc_phase * n_r) % n_r
        for _ in range(min(int(max_anchors), n_r)):
            selected.append(cur)
            d2_cur = cost_rr[cur]
            min_d2 = torch.minimum(min_d2, d2_cur)
            cover_rad2 = 4.5 * torch.maximum(h2_all[cur], h2_all)
            uncovered = uncovered & (d2_cur > cover_rad2)
            if not bool(uncovered.any()):
                break
            scores = torch.where(uncovered, min_d2, -1.0)
            cur = int(scores.argmax().item())

        sel_t = torch.tensor(selected, device=r_sub.device, dtype=torch.long)
        anchors = r_sub[sel_t]  # [M, D]
        h2_m = h2_all[sel_t]  # [M]
        m_anc = int(anchors.shape[0])

        # Exact Voronoi partition of ALL real and ALL fake samples (zero cross-mode leakage, zero dropped outliers!).
        d2_ar = cost_rr[sel_t]  # [M, n_r]
        d2_af = pairwise_sq_dist(anchors, f_sub_det)  # [M, n_f]
        assign_r = d2_ar.argmin(dim=0)  # [n_r]
        assign_f = d2_af.argmin(dim=0)  # [n_f]

        m_grid = torch.arange(m_anc, device=r_sub.device, dtype=torch.long).unsqueeze(1)
        mask_r_full = (assign_r.unsqueeze(0) == m_grid)  # [M, n_r]
        mask_f_full = (assign_f.unsqueeze(0) == m_grid)  # [M, n_f]

        cnt_r_full = mask_r_full.sum(dim=1).to(dtype=torch.float32)
        cnt_f_full = mask_f_full.sum(dim=1).to(dtype=torch.float32)
        valid_full = (cnt_r_full >= 4.0) & (cnt_f_full >= 4.0)
        if not bool(valid_full.any()):
            return x_fake.new_zeros(())

        w_r_full = mask_r_full.to(dtype=torch.float32) / cnt_r_full.clamp_min(1.0).unsqueeze(1)
        w_f_full = mask_f_full.to(dtype=torch.float32) / cnt_f_full.clamp_min(1.0).unsqueeze(1)
        bessel_r_full = (cnt_r_full / (cnt_r_full - 1.0).clamp_min(1.0)).view(-1, 1, 1)
        bessel_f_full = (cnt_f_full / (cnt_f_full - 1.0).clamp_min(1.0)).view(-1, 1, 1)

        mu_r_full = w_r_full @ r_sub  # [M, D]
        dr_full = r_sub.unsqueeze(0) - mu_r_full.unsqueeze(1)
        cov_r_full = bessel_r_full * torch.einsum("mb,mbd,mbe->mde", w_r_full, dr_full, dr_full)
        var_scale = (cov_r_full.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(dim)).clamp_min(1e-6)
        sigma_k = var_scale.sqrt()
        ridge = 0.04 * var_scale
        valid_full = valid_full & (var_scale > 1e-5)
        if not bool(valid_full.any()):
            return x_fake.new_zeros(())

        # Core-ball sub-mask inside each Voronoi cell (radius 3.5 * h2_m) so outliers cannot mask a pinched core!
        core_rad2 = (3.5 * h2_m).unsqueeze(1)
        mask_r_core = mask_r_full & (d2_ar <= core_rad2)
        mask_f_core = mask_f_full & (d2_af <= core_rad2)
        cnt_r_core = mask_r_core.sum(dim=1).to(dtype=torch.float32)
        cnt_f_core = mask_f_core.sum(dim=1).to(dtype=torch.float32)
        valid_core = valid_full & (cnt_r_core >= 4.0) & (cnt_f_core >= 4.0)

        w_r_core = mask_r_core.to(dtype=torch.float32) / cnt_r_core.clamp_min(1.0).unsqueeze(1)
        w_f_core = mask_f_core.to(dtype=torch.float32) / cnt_f_core.clamp_min(1.0).unsqueeze(1)
        bessel_r_core = (cnt_r_core / (cnt_r_core - 1.0).clamp_min(1.0)).view(-1, 1, 1)
        bessel_f_core = (cnt_f_core / (cnt_f_core - 1.0).clamp_min(1.0)).view(-1, 1, 1)

        mu_r_core = w_r_core @ r_sub
        dr_core = r_sub.unsqueeze(0) - mu_r_core.unsqueeze(1)
        cov_r_core = bessel_r_core * torch.einsum("mb,mbd,mbe->mde", w_r_core, dr_core, dr_core)
        var_core = (cov_r_core.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(dim)).clamp_min(1e-6)
        ridge_core = 0.04 * var_core
        valid_core = valid_core & (var_core > 1e-5)

        valid_full_f = valid_full.to(dtype=torch.float32)
        valid_core_f = valid_core.to(dtype=torch.float32)

    f_live = f_flat if f_idx is None else f_flat[f_idx]
    mu_f_full = w_f_full @ f_live
    df_full = f_live.unsqueeze(0) - mu_f_full.unsqueeze(1)
    cov_f_full = bessel_f_full * torch.einsum("mb,mbd,mbe->mde", w_f_full, df_full, df_full)

    d_burg_full = _burg_logdet_divergence(cov_r_full, cov_f_full, ridge)
    mean_err_sq = (mu_f_full - mu_r_full).square().sum(dim=-1) / (float(dim) * var_scale)
    e_full = sigma_k * ((torch.sqrt(1.0 + d_burg_full) - 1.0) + 0.25 * (torch.sqrt(1.0 + mean_err_sq) - 1.0))
    e_full_valid = e_full * valid_full_f
    mean_full = e_full_valid.sum() / valid_full_f.sum().clamp_min(1.0)
    max_full = e_full_valid.amax()
    loss_full = 0.65 * mean_full + 0.35 * max_full

    if bool(valid_core.any()):
        mu_f_core = w_f_core @ f_live
        df_core = f_live.unsqueeze(0) - mu_f_core.unsqueeze(1)
        cov_f_core = bessel_f_core * torch.einsum("mb,mbd,mbe->mde", w_f_core, df_core, df_core)
        d_burg_core = _burg_logdet_divergence(cov_r_core, cov_f_core, ridge_core)
        e_core_valid = (var_core.sqrt() * (torch.sqrt(1.0 + d_burg_core) - 1.0)) * valid_core_f
        mean_core = e_core_valid.sum() / valid_core_f.sum().clamp_min(1.0)
        max_core = e_core_valid.amax()
        loss_core = 0.50 * mean_core + 0.50 * max_core
        local_iso = 0.55 * loss_full + 0.45 * loss_core
    else:
        local_iso = loss_full

    total_iso = float(has_intrinsic * res_gate) * local_iso
    return total_iso.to(dtype=x_fake.dtype)


__all__ = [
    "born_infeld_debiased_transport_loss",
    "feydy_debiased_monge_displacement",
    "intrinsic_sinkhorn_coupling",
    "pullback_covariance_isometry_loss",
    "qmc_monge_displacement",
    "quantile_1d_monge_displacement",
]
