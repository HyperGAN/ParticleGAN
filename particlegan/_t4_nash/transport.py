"""Intrinsic-scale Feydy-debiased & QMC Monge relativistic Born-Infeld transport for T4Nash."""
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
    """Scale-free log-domain Sinkhorn coupling anchored to intrinsic target scale.

    Unlike pure cross-median normalization (which shrinks transport residuals
    when ``x_fake`` is initialized far off-manifold), ``intrinsic_sinkhorn_coupling``
    computes ``cost_scale = max(scale_rr, median_cross_cost)`` for numerically
    well-conditioned log-domain Sinkhorn scaling while measuring ``residual`` and
    ``data_scale`` relative to the target's intrinsic pairwise spread ``scale_rr``
    (falling back to unit coordinate scale ``1.0`` on degenerate zero-variance
    point-mass targets so ``residual = RMS(x_fake - x_0)`` contracts cleanly to zero).
    """
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
        elif k_proto >= 4:
            rr_med = pairwise_sq_dist(c_flat).median()
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
def feydy_debiased_monge_displacement(
    anchors: torch.Tensor,
    anchor_weights: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.12,
    sinkhorn_iters: int = 6,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact zero-bias Feydy-debiased Sinkhorn displacement ``v = T_{a->f} - T_{f->f}``.

    When ``x_fake`` equals ``anchors`` with matching uniform weights, ``T_{a->f} == T_{f->f}``
    identically for any ``sinkhorn_eps > 0``, eliminating entropic variance shrinkage.
    Returns ``(v, ref_sigma)`` where ``v`` matches ``x_fake`` shape and ``ref_sigma`` is scalar.
    """
    a_flat = anchors.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_a, b_f = a_flat.shape[0], f_flat.shape[0]

    w = anchor_weights.detach().reshape(-1).to(device=a_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    cost_af = pairwise_sq_dist(a_flat, f_flat)

    cross_med = cost_af.median().clamp_min(eps_scale)
    if k_a >= 4:
        aa_med = pairwise_sq_dist(a_flat).median()
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
    """Stratified QMC inverse-CDF Monge displacement from entropic plan ``Pi_{a -> f}``.

    Unlike barycentric projection ``E[A | X_f = x_j]`` (which averages symmetric
    modes into the convex-hull centroid when ``x_fake`` is collapsed or equidistant),
    deterministic Halton quantiles draw actual support atoms ``a_{pi(j)}`` in exact
    proportion to ``Pi_{. | j}``, breaking collapsed-origin symmetry without RNG draws.
    """
    a_flat = anchors.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_a, b_f = a_flat.shape[0], f_flat.shape[0]

    w = anchor_weights.detach().reshape(-1).to(device=a_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    cost_af = pairwise_sq_dist(a_flat, f_flat)
    cross_med = cost_af.median().clamp_min(eps_scale)
    if k_a >= 4:
        aa_med = pairwise_sq_dist(a_flat).median()
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
    """Relativistic Born-Infeld potential blending far-field QMC Monge & zero-bias Feydy OT.

    Far off-manifold or when the critic is uncalibrated (``ema_residual > 0.20`` or
    ``calibration_gate < 0.85``), stratified QMC Monge displacement splits collapsed
    or shifted clouds onto distinct target modes with unit Born-Infeld cone slope.
    As modes populate and the critic calibrates, the displacement transitions smoothly
    to exact zero-bias Feydy-debiased Sinkhorn ``T_{a->f} - T_{f->f}`` (which vanishes
    identically when ``p_fake == p_real`` so within-mode covariance never shrinks).
    """
    if not x_fake.requires_grad:
        return x_fake.new_zeros(())
    if (
        prototypes is not None
        and proto_weights is not None
        and prototypes.shape[0] >= 2
        and prototypes.shape[1] == x_fake.flatten(1).shape[1]
    ):
        anchors = prototypes
        weights = proto_weights
    else:
        b_r = x_real.shape[0]
        anchors = x_real.detach().flatten(1)
        weights = torch.full((b_r,), 1.0 / max(1, b_r), device=x_real.device, dtype=torch.float32)

    eps_eff = effective_sinkhorn_eps(ema_residual, sinkhorn_eps)
    u_res = max(0.0, min(1.0, (float(ema_residual) - 0.14) / 0.12))
    u_cal = max(0.0, min(1.0, (0.85 - float(calibration_gate)) / 0.45))
    w_qmc = max(u_res * u_res * (3.0 - 2.0 * u_res), u_cal * u_cal * (3.0 - 2.0 * u_cal))

    if w_qmc >= 0.999:
        v_disp, sigma_ref = qmc_monge_displacement(
            anchors,
            weights,
            x_fake.detach(),
            sinkhorn_eps=eps_eff,
            sinkhorn_iters=sinkhorn_iters,
            qmc_phase=qmc_phase,
        )
    elif w_qmc <= 0.001:
        v_disp, sigma_ref = feydy_debiased_monge_displacement(
            anchors,
            weights,
            x_fake.detach(),
            sinkhorn_eps=eps_eff,
            sinkhorn_iters=sinkhorn_iters,
        )
    else:
        v_qmc, sigma_ref = qmc_monge_displacement(
            anchors,
            weights,
            x_fake.detach(),
            sinkhorn_eps=eps_eff,
            sinkhorn_iters=sinkhorn_iters,
            qmc_phase=qmc_phase,
        )
        v_feydy, _ = feydy_debiased_monge_displacement(
            anchors,
            weights,
            x_fake.detach(),
            sinkhorn_eps=eps_eff,
            sinkhorn_iters=sinkhorn_iters,
        )
        v_disp = w_qmc * v_qmc + (1.0 - w_qmc) * v_feydy

    target_det = (x_fake.detach() + v_disp).detach()
    diff_flat = (x_fake - target_det).flatten(1).to(dtype=torch.float32)
    tau_bi = (0.12 * sigma_ref.to(dtype=torch.float32)).clamp_min(1e-3)
    sq_norm = diff_flat.square().sum(dim=1) / (tau_bi * tau_bi)
    bi_energy = tau_bi * (torch.sqrt(1.0 + sq_norm) - 1.0).mean()
    return bi_energy.to(dtype=x_fake.dtype)
