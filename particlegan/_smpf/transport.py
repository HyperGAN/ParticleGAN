"""Scale- & distortion-aware optimal transport between s-means prototypes and generated samples."""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._stats import knn_excluding_self, pairwise_sq_dist


def effective_sinkhorn_eps(ema_residual: float, sinkhorn_eps: float = 0.12) -> float:
    """State-settling Sinkhorn temperature (zero clocks)."""
    ratio = max(0.0, min(1.0, float(ema_residual) / 0.12))
    return float(sinkhorn_eps) * max(0.02, ratio * ratio)


def _quantile_1d_sq_residual(
    p_1d: torch.Tensor,
    w: torch.Tensor,
    f_1d: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Trimmed 1D quantile W_2^2 / scale between weighted EB prototypes and fake samples."""
    b_f = int(f_1d.shape[0])
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


def smeans_prototype_sinkhorn(
    prototypes: torch.Tensor,
    weights: torch.Tensor,
    ell_proto: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.12,
    sinkhorn_iters: int = 6,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-free log-domain Sinkhorn coupling with s-means prototype masses and local Born-Infeld horizon."""
    c_flat = prototypes.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_proto, b_f = int(c_flat.shape[0]), int(f_flat.shape[0])
    dim = int(c_flat.shape[1])

    w = weights.detach().reshape(-1).to(device=c_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)
    ell = ell_proto.detach().reshape(-1).to(device=c_flat.device, dtype=torch.float32).clamp_min(1e-6)

    raw_sq = pairwise_sq_dist(c_flat, f_flat)
    with torch.no_grad():
        scale_sq = torch.maximum(raw_sq.detach().median(), 0.5 * raw_sq.detach().mean()).clamp_min(eps_scale)
        data_scale = scale_sq.sqrt()

    ell_hat = (ell / data_scale.clamp_min(1e-6)).clamp(0.02, 0.50).unsqueeze(1)
    norm_sq = raw_sq / scale_sq
    bi_cost = torch.sqrt(ell_hat.square() + norm_sq) - ell_hat

    log_a = torch.log(w)
    log_b = -math.log(max(1, b_f))
    eps_val = max(1e-4, float(sinkhorn_eps))
    iters = max(int(sinkhorn_iters), 7) if eps_val < 0.03 else int(sinkhorn_iters)
    log_kernel = -bi_cost / eps_val

    u = torch.zeros(k_proto, device=c_flat.device, dtype=torch.float32)
    v = torch.zeros(b_f, device=c_flat.device, dtype=torch.float32)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)

    pi = torch.exp(log_kernel + u.unsqueeze(1) + v.unsqueeze(0))
    pi = pi * (w.unsqueeze(1) / pi.sum(dim=1, keepdim=True).clamp_min(1e-12))

    with torch.no_grad():
        if dim == 1 and b_f >= 16:
            mean_sq = _quantile_1d_sq_residual(c_flat[:, 0], w, f_flat[:, 0], scale_sq)
        else:
            min_p2f = (w * norm_sq.detach().min(dim=1).values).sum()
            f2p_all = norm_sq.detach().min(dim=0).values
            if 2 <= dim <= 4 and b_f >= max(512, 3 * k_proto):
                k_keep = max(1, int(0.85 * b_f))
                min_f2p = torch.topk(f2p_all, k=k_keep, largest=False).values.mean()
            else:
                min_f2p = f2p_all.mean()
            mean_sq = 0.5 * (min_p2f + min_f2p)
        residual = torch.sqrt(mean_sq.clamp_min(0.0))

    return (
        pi.to(dtype=x_fake.dtype),
        bi_cost.to(dtype=x_fake.dtype),
        residual.to(dtype=x_fake.dtype),
        data_scale.to(dtype=x_fake.dtype),
    )


def _born_infeld_physical_cost(
    a: torch.Tensor,
    b: torch.Tensor,
    ell_row: torch.Tensor,
) -> torch.Tensor:
    """Scale-normalized pure-FP32 overflow-safe physical Born-Infeld distance ``sqrt(ell^2 + ||a - b||^2) - ell``."""
    af = a.float()
    bf = b.float()
    ef = ell_row.float().clamp_min(1e-6)
    if ef.ndim == 1:
        ef = ef.unsqueeze(1)
    with torch.no_grad():
        s_max = torch.maximum(
            torch.maximum(af.detach().abs().amax(), bf.detach().abs().amax()),
            ef.detach().amax(),
        ).clamp_min(1.0)
    a_s = af / s_max
    b_s = bf / s_max
    e_s = (ef / s_max).clamp_min(1e-12)
    diff_sq = (
        a_s.square().sum(dim=-1, keepdim=True)
        + b_s.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (a_s @ b_s.t())
    )
    diff_sq = diff_sq + (diff_sq.detach().clamp_min(0.0) - diff_sq.detach())
    rad = torch.sqrt(e_s.square() + diff_sq)
    val = diff_sq / (rad + e_s).clamp_min(1e-24)
    # Exact analytical gradient of rad - e_s with rationalized forward precision at 0.
    out_s = (rad - e_s) + (val - (rad - e_s)).detach()
    return (out_s * s_max).to(dtype=a.dtype)


def smeans_debiased_generator_loss(
    x_fake: torch.Tensor,
    c_eb: torch.Tensor,
    weights: torch.Tensor,
    ell_proto: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.08,
    sinkhorn_iters: int = 7,
    shell_weight: float = 0.25,
    discrete_prior: bool = False,
    sub_batch_cap: int = 512,
    phase: float = 0.0,
) -> torch.Tensor:
    """Slack-screened local-horizon Born-Infeld s-means prototype Sinkhorn + Laguerre outlier snap + RMS shell."""
    f_all = x_fake.reshape(x_fake.shape[0], -1)
    b_total, dim = int(f_all.shape[0]), int(f_all.shape[1])
    if b_total < 8 or dim == 0 or c_eb.shape[0] < 2 or c_eb.shape[1] != dim:
        return x_fake[:0].sum()

    # Bound O(B_f^2) self-coupling workspace on large-batch native100 runs via Weyl striding.
    if b_total > int(sub_batch_cap):
        cap = int(sub_batch_cap)
        offset = int(float(phase) * float(b_total)) % b_total
        idx = ((torch.arange(cap, device=x_fake.device, dtype=torch.long) * b_total) // cap + offset) % b_total
        f_flat = f_all[idx]
    else:
        f_flat = f_all
    b_f = int(f_flat.shape[0])
    k_p = int(c_eb.shape[0])

    c_det = c_eb.detach().to(device=f_flat.device, dtype=f_flat.dtype)
    w = weights.detach().to(device=f_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)
    ell_raw = ell_proto.detach().to(device=f_flat.device, dtype=torch.float32).clamp_min(1e-5)

    with torch.no_grad():
        f_det = f_flat.detach().float()
        c_f32 = c_det.float()
        sq_pf_det = pairwise_sq_dist(c_f32, f_det)
        scale_sq = torch.maximum(sq_pf_det.median(), 0.5 * sq_pf_det.mean()).clamp_min(1e-10)
        scale_med = scale_sq.sqrt()
        sq_ff_det = pairwise_sq_dist(f_det)

        # Pure s-means local prototype radius per cluster.
        ell_eff = torch.clamp(ell_raw, min=0.015 * scale_med, max=0.50 * scale_med)
        ell_mean = (w * ell_eff).sum().clamp_min(1e-5)

        # 1. Kantorovich-Hellinger slack-screened local-scale Sinkhorn on (K + 1) x (B_f + 1).
        # Points inside ~3.2 * ell_k couple smoothly; valley/hole stragglers route into slack
        # rather than averaging symmetrically across opposite ring/corner modes.
        rho = 3.2
        eps_loc = max(0.06, float(sinkhorn_eps))
        c_pf_phys = torch.sqrt(ell_eff.unsqueeze(1).square() + sq_pf_det) - ell_eff.unsqueeze(1)
        c_pf_hat = c_pf_phys / ell_eff.unsqueeze(1).clamp_min(1e-5)

        border_val = -(0.5 * rho) / eps_loc
        log_k_aug = f_det.new_full((k_p + 1, b_f + 1), border_val)
        log_k_aug[:k_p, :b_f] = -c_pf_hat / eps_loc
        log_k_aug[k_p, b_f] = 0.0

        log_a_aug = torch.cat([torch.log(w), w.new_zeros((1,))], dim=0)
        log_b_aug = torch.cat([w.new_full((b_f,), -math.log(max(1, b_f))), w.new_zeros((1,))], dim=0)
        u_aug = torch.zeros_like(log_a_aug)
        v_aug = torch.zeros_like(log_b_aug)
        n_iters = max(7, int(sinkhorn_iters))
        for _ in range(n_iters):
            u_aug = log_a_aug - torch.logsumexp(log_k_aug + v_aug.unsqueeze(0), dim=1)
            v_aug = log_b_aug - torch.logsumexp(log_k_aug + u_aug.unsqueeze(1), dim=0)
        p_aug = torch.exp(log_k_aug + u_aug.unsqueeze(1) + v_aug.unsqueeze(0))
        pi_pf_act = p_aug[:k_p, :b_f]
        col_act = (float(b_f) * pi_pf_act.sum(dim=0)).clamp(0.0, 1.0)
        col_slack = (1.0 - col_act).clamp(0.0, 1.0)

        # 2. Laguerre power-diagram Monge assignment on slack-screened valley/acquisition points:
        # Dual potentials u_rem enforce underserved prototype mass conservation while hard-max
        # selects ONE target prototype per slack point (zero cross-ring cancellation at origin!).
        captured_w = pi_pf_act.sum(dim=1)
        w_rem = torch.maximum(w - captured_w, 0.02 * w)
        w_rem = w_rem / w_rem.sum().clamp_min(1e-12)
        b_rem = col_slack + 1e-4
        b_rem = b_rem / b_rem.sum().clamp_min(1e-12)

        eps_rem = 0.05
        c_pf_glob = c_pf_phys / scale_med
        log_k_rem = -c_pf_glob / eps_rem
        log_a_rem = torch.log(w_rem)
        log_b_rem = torch.log(b_rem)
        u_rem = torch.zeros_like(log_a_rem)
        v_rem = torch.zeros_like(log_b_rem)
        for _ in range(n_iters):
            u_rem = log_a_rem - torch.logsumexp(log_k_rem + v_rem.unsqueeze(0), dim=1)
            v_rem = log_b_rem - torch.logsumexp(log_k_rem + u_rem.unsqueeze(1), dim=0)
        laguerre_scores = log_k_rem + u_rem.unsqueeze(1)
        monge_idx = laguerre_scores.argmax(dim=0)
        pi_monge = torch.zeros_like(pi_pf_act)
        pi_monge.scatter_(0, monge_idx.unsqueeze(0), (col_slack / float(b_f)).unsqueeze(0))

        pi_pf_total = pi_pf_act + pi_monge

        # 3. Locally screened self-coupling pi_ff ONLY among active in-cluster samples (col_act):
        # Valley outliers have col_act -> 0 and slack escape at rho, eliminating inward valley traps.
        c_ff_hat = (torch.sqrt(ell_mean.square() + sq_ff_det) - ell_mean) / ell_mean
        eye_f = torch.eye(b_f, device=f_flat.device, dtype=torch.bool)
        if discrete_prior:
            clone_mask = sq_ff_det <= 1e-12
            row_has_nonclone = (~clone_mask).any(dim=1, keepdim=True)
            mask_ff = torch.where(row_has_nonclone, clone_mask, eye_f)
        else:
            mask_ff = eye_f

        log_k_ff_aug = f_det.new_full((b_f + 1, b_f + 1), border_val)
        log_k_ff_act = (-c_ff_hat / eps_loc).masked_fill(mask_ff, float("-inf"))
        log_k_ff_aug[:b_f, :b_f] = log_k_ff_act
        log_k_ff_aug[b_f, b_f] = 0.0

        log_m_ff = torch.cat(
            [torch.log((col_act / float(b_f)).clamp_min(1e-8)), w.new_zeros((1,))],
            dim=0,
        )
        u_ff = torch.zeros_like(log_m_ff)
        v_ff = torch.zeros_like(log_m_ff)
        for _ in range(n_iters):
            u_ff = log_m_ff - torch.logsumexp(log_k_ff_aug + v_ff.unsqueeze(0), dim=1)
            v_ff = log_m_ff - torch.logsumexp(log_k_ff_aug + u_ff.unsqueeze(1), dim=0)
        pi_ff = torch.exp(log_k_ff_aug[:b_f, :b_f] + u_ff[:b_f].unsqueeze(1) + v_ff[:b_f].unsqueeze(0))
        pi_ff = 0.5 * (pi_ff + pi_ff.t())
        pi_ff = pi_ff.masked_fill(mask_ff, 0.0)

        # Match total active repulsive mass to active cross-coupling mass so intra-mode variance is exact.
        act_mass_pf = pi_pf_act.sum()
        act_mass_ff = pi_ff.sum()
        ff_scale = torch.where(
            act_mass_ff > 1e-8,
            (act_mass_pf / act_mass_ff.clamp_min(1e-8)).clamp(0.0, 1.5),
            w.new_zeros(()),
        )
        pi_ff = pi_ff * ff_scale

        pi_pf_t = pi_pf_total.to(dtype=f_flat.dtype)
        pi_ff_t = pi_ff.to(dtype=f_flat.dtype)

    cost_pf = _born_infeld_physical_cost(c_det, f_flat, ell_eff.to(dtype=f_flat.dtype))
    cost_ff = _born_infeld_physical_cost(f_flat, f_flat, ell_mean.expand(b_f).to(dtype=f_flat.dtype))
    ot_debiased = (pi_pf_t * cost_pf).sum() - 0.5 * (pi_ff_t * cost_ff).sum()

    if float(shell_weight) <= 0.0 or discrete_prior or dim <= 1 or dim > 16:
        return ot_debiased

    # Per-prototype s-means RMS distortion shell on active in-mode couplings prevents rare-mode collapse.
    with torch.no_grad():
        row_mass = pi_pf_act.sum(dim=1)
        active_proto = (row_mass > (0.20 / float(max(1, k_p)))) & (ell_raw > 1e-5)
        cond_p = (pi_pf_act / row_mass.clamp_min(1e-12).unsqueeze(1)).to(dtype=f_flat.dtype)
        bary_det = cond_p @ f_flat.detach()
        act_float = active_proto.to(dtype=f_flat.dtype)
        w_act_raw = row_mass.to(dtype=f_flat.dtype) * act_float
        w_act_sum = w_act_raw.sum()
        w_act = torch.where(w_act_sum > 1e-10, w_act_raw / w_act_sum.clamp_min(1e-10), torch.zeros_like(w_act_raw))

    diff_cell = f_flat.unsqueeze(0) - bary_det.unsqueeze(1)
    sq_cell = diff_cell.square().sum(dim=-1)
    r_gen = torch.sqrt((cond_p * sq_cell).sum(dim=1) + 1e-8)
    r_target = ell_raw.to(dtype=f_flat.dtype)
    deficit = F.relu(0.85 * r_target - r_gen)
    shell_loss = (w_act * deficit).sum()
    return ot_debiased + float(shell_weight) * shell_loss


def smeans_conditional_fiber_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-FP32 affine-detrended + coordinate-whitened multi-scale IMQ conditional fiber transport."""
    b_n = int(x_fake.shape[0])
    if b_n < 8 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
        return x_fake[:0].sum()

    x_flat = x_fake.reshape(b_n, -1)
    y_flat = y_real.detach().reshape(b_n, -1)
    dim = int(x_flat.shape[1])
    if dim == 0 or dim > 32:
        return x_fake[:0].sum()

    with torch.no_grad():
        if cond_disc is not None and cond_disc.shape[0] == b_n:
            cd = cond_disc.detach().reshape(b_n, -1)
            same_t = (cd.unsqueeze(1) == cd.unsqueeze(0)).all(dim=-1)
        else:
            same_t = torch.ones((b_n, b_n), dtype=torch.bool, device=x_fake.device)

        m_hollow = same_t.clone()
        m_hollow.fill_diagonal_(False)
        valid_row = m_hollow.any(dim=1)
        m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)
        if not bool(m_valid.any()):
            return x_fake[:0].sum()

        cf = c_cont.detach().reshape(b_n, -1).float()
        yf = y_flat.float()
        c_scale = cf.abs().amax().clamp_min(1.0)
        c_unit = cf / c_scale

        d_c = torch.cdist(c_unit, c_unit, p=2.0)
        d_c_max = d_c.amax()
        d_c_masked = d_c.masked_fill(~m_valid, float("inf"))
        d_sorted, sort_idx = torch.sort(d_c_masked, dim=1)
        nn_idx = sort_idx[:, 0]

        n_i = m_valid.sum(dim=1)
        k_idx = torch.minimum(torch.full_like(n_i, 3), n_i.clamp_min(1)) - 1
        sigma_raw = d_sorted.gather(1, k_idx.unsqueeze(1)).squeeze(1)
        sigma_floor = 1e-3 * d_c_max + 1e-6
        sigma_i = torch.where(
            valid_row,
            torch.where(torch.isfinite(sigma_raw), sigma_raw, torch.zeros_like(sigma_raw)).clamp_min(0.0)
            + sigma_floor,
            torch.ones_like(sigma_raw),
        )

        # Fully vectorized GPU Nadaraya-Watson conditional trend removal (zero CPU .item() syncs).
        sig_nw2 = (2.0 * sigma_i.unsqueeze(1).square()).clamp_min(1e-12)
        w_nw_raw = torch.where(same_t, torch.exp(-d_c.square() / sig_nw2), 0.0)
        w_nw = w_nw_raw / w_nw_raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
        y_trend = w_nw @ yf
        y_res = yf - y_trend

        # Coordinate-wise residual noise scale from nearest context neighbor on detrended residuals.
        innov_d = (y_res - y_res[nn_idx]) / math.sqrt(2.0)
        same_valid = (same_t & valid_row.unsqueeze(0) & valid_row.unsqueeze(1)).float()
        f_cnt = same_valid.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        ell_d_sq_raw = (same_valid @ innov_d.square()) / f_cnt
        ell_iso_sq = ell_d_sq_raw.mean(dim=-1, keepdim=True)
        ell_d_sq = 0.50 * ell_iso_sq + 0.50 * ell_d_sq_raw
        y_sc = yf.abs().amax().clamp_min(1.0)
        ell_d_floor = (1e-2 * y_sc / math.sqrt(float(dim))) + 1e-5
        ell_d = torch.where(
            valid_row.unsqueeze(1),
            ell_d_sq.sqrt().clamp_min(ell_d_floor),
            torch.ones_like(ell_d_sq),
        )
        ell_rms = ell_d.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        w_coord = ell_rms / ell_d
        w_coord = w_coord / w_coord.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        ell_vec = ell_rms * math.sqrt(float(dim))

        denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
        k_c = torch.where(m_valid, torch.exp(-d_c.square() / denom_2s), 0.0)
        deg_c = 1.0 + k_c.sum(dim=1)
        inv_sqrt_deg = torch.rsqrt(deg_c)
        w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
        w_imq_sym = (0.5 * (w_imq + w_imq.t())).to(dtype=x_flat.dtype)

    xf = x_flat.float()
    xw = (xf - y_trend) * w_coord
    yw = y_res * w_coord

    with torch.no_grad():
        s_w = torch.maximum(
            torch.maximum(xw.detach().abs().amax(), yw.detach().abs().amax()),
            ell_vec.detach().amax(),
        ).clamp_min(1.0)

    xw_s = xw / s_w
    yw_s = yw / s_w
    ell_s0 = (ell_vec / s_w).clamp_min(1e-8)

    sq_w_xy = xw_s.square().sum(dim=-1, keepdim=True) + yw_s.square().sum(dim=-1).unsqueeze(0) - 2.0 * (xw_s @ yw_s.t())
    sq_w_xy = sq_w_xy + (sq_w_xy.detach().clamp_min(0.0) - sq_w_xy.detach())
    sq_w_xx = xw_s.square().sum(dim=-1, keepdim=True) + xw_s.square().sum(dim=-1).unsqueeze(0) - 2.0 * (xw_s @ xw_s.t())
    sq_w_xx = sq_w_xx + (sq_w_xx.detach().clamp_min(0.0) - sq_w_xx.detach())

    psi_xy = torch.zeros_like(sq_w_xy)
    psi_xx = torch.zeros_like(sq_w_xx)
    for s_mult in (0.25, 1.0, 4.0):
        ell_s = ell_s0 * s_mult
        ell_s2 = ell_s.square()
        rad_s_xy = torch.sqrt(ell_s2 + sq_w_xy)
        psi_xy = psi_xy + (ell_s * sq_w_xy) / (rad_s_xy * (ell_s + rad_s_xy)).clamp_min(1e-24)
        rad_s_xx = torch.sqrt(ell_s2 + sq_w_xx)
        psi_xx = psi_xx + (ell_s * sq_w_xx) / (rad_s_xx * (ell_s + rad_s_xx)).clamp_min(1e-24)
    psi_xy_t = (psi_xy * (1.0 / 3.0)).to(dtype=x_flat.dtype)
    psi_xx_t = (psi_xx * (1.0 / 3.0)).to(dtype=x_flat.dtype)

    # Diagonal exact same-context alignment term cancels residual bias on individual fibers.
    diag_diff_sq = (((xf - yf) * w_coord) / s_w).square().sum(dim=-1)
    ell_diag = ell_s0.squeeze(1)
    rad_diag = torch.sqrt(ell_diag.square() + diag_diff_sq)
    diag_bi = ((diag_diff_sq / (rad_diag + ell_diag).clamp_min(1e-24)) * s_w).to(dtype=x_flat.dtype)
    nn_self_sq = sq_w_xx.gather(1, nn_idx.unsqueeze(1)).squeeze(1)
    rad_nn = torch.sqrt(ell_diag.square() + nn_self_sq)
    nn_bi = ((nn_self_sq / (rad_nn + ell_diag).clamp_min(1e-24)) * s_w).to(dtype=x_flat.dtype)
    diag_debiased = (valid_row.to(dtype=x_flat.dtype) * (diag_bi - 0.5 * nn_bi)).mean()

    return (w_imq_sym * psi_xy_t).sum() - 0.5 * (w_imq_sym * psi_xx_t).sum() + 0.35 * diag_debiased


@torch.no_grad()
def _minimax_h0_ratio(points: torch.Tensor) -> float:
    """Dimensionless H_0 connectivity ratio on s-means EB prototypes."""
    k = int(points.shape[0])
    if k < 6:
        return 100.0
    p_u = points.float() / points.float().abs().amax().clamp_min(1.0)
    d_mat = torch.cdist(p_u, p_u, p=2.0)
    off = ~torch.eye(k, dtype=torch.bool, device=points.device)
    nn = d_mat.masked_fill(~off, float("inf")).amin(dim=-1)
    q90 = torch.quantile(nn, 0.90).clamp_min(1e-6)
    m_path = d_mat.clone()
    for _ in range(8):
        m_path = torch.minimum(m_path, torch.maximum(m_path.unsqueeze(2), m_path.unsqueeze(0)).amin(dim=1))
    span = torch.quantile(d_mat[off], 0.90) / q90
    if float(span.item()) < 1.25:
        return 100.0
    return float((m_path.amax() / q90).item())


def smeans_simplicial_manifold_loss(
    z_table: torch.Tensor,
    gen_fn: Any,
    record: Any,
    *,
    y_real: Optional[torch.Tensor] = None,
    k_neighbors: int = 4,
    max_queries: int = 48,
) -> torch.Tensor:
    """Align unseen latent k-NN chords, 2-simplex barycenters, and isometric geodesic charts onto s-means manifolds."""
    n_z, d_z = int(z_table.shape[0]), int(z_table.shape[1])
    if n_z < 4 or d_z == 0 or record is None:
        return z_table[:0].sum()
    eb = record.readout_eb_prototypes(dtype=z_table.dtype, device=z_table.device) if hasattr(record, "readout_eb_prototypes") else None
    if eb is None:
        return z_table[:0].sum()
    c_eb, _, ell_proto, _ = eb
    k_p = int(c_eb.shape[0])
    if k_p < 8:
        return z_table[:0].sum()

    with torch.no_grad():
        kappa_h0 = _minimax_h0_ratio(c_eb)
        u_conn = max(0.0, min(1.0, (8.5 - kappa_h0) / 3.0))
        if u_conn <= 1e-4:
            return z_table[:0].sum()
        g_conn = u_conn * u_conn * (3.0 - 2.0 * u_conn)

        if y_real is not None and isinstance(y_real, torch.Tensor) and y_real.shape[0] >= 8:
            yr_flat = y_real.detach().reshape(y_real.shape[0], -1).to(device=z_table.device, dtype=z_table.dtype)
            manifold_bank = torch.cat([c_eb.detach(), yr_flat], dim=0) if yr_flat.shape[1] == c_eb.shape[1] else c_eb.detach()
        else:
            manifold_bank = c_eb.detach()

        # Shortest-path graph geodesic metric on s-means EB prototypes so open arcs/curved sheets
        # never shortcut across missing gaps when unfolding the latent particle chart.
        c_f32 = c_eb.detach().float()
        d_pp = torch.cdist(c_f32, c_f32, p=2.0)
        eye_p = torch.eye(k_p, device=z_table.device, dtype=torch.bool)
        k_g = min(6, k_p - 1)
        _, g_knn = torch.topk(d_pp.masked_fill(eye_p, float("inf")), k=k_g, dim=1, largest=False)
        adj = torch.zeros((k_p, k_p), device=z_table.device, dtype=torch.bool)
        adj.scatter_(1, g_knn, True)
        adj = adj | adj.t()
        big_m = 12.0 * d_pp.amax().clamp_min(1.0)
        g_dist = torch.where(eye_p, torch.zeros_like(d_pp), torch.where(adj, d_pp, d_pp + big_m))
        for _ in range(6):
            g_dist = torch.minimum(g_dist, (g_dist.unsqueeze(2) + g_dist.unsqueeze(0)).amin(dim=1))

        k_eff = min(int(k_neighbors), n_z - 1)
        _, knn_idx = knn_excluding_self(pairwise_sq_dist(z_table.detach().float()), k_eff)
        k_chord = min(3, k_eff)
        src_idx = torch.arange(n_z, device=z_table.device).unsqueeze(1).expand(n_z, k_chord).reshape(-1)
        dst_idx = knn_idx[:, :k_chord].reshape(-1)
        lo = torch.minimum(src_idx, dst_idx)
        hi = torch.maximum(src_idx, dst_idx)
        edge_pairs = torch.unique(torch.stack((lo, hi), dim=1), dim=0)

        phase = float(getattr(record, "chord_phase", 0.0))
        n_edges = int(edge_pairs.shape[0])
        if n_edges > max_queries:
            sel = ((torch.arange(max_queries, device=z_table.device) * n_edges) // max_queries + int(phase * n_edges)) % n_edges
            edge_pairs = edge_pairs[sel]

    # Interior chord triad (0.25, 0.50, 0.75) + all C(k_eff, 2) 2-simplex triangle barycenters when d_z >= 2.
    z_a = z_table[edge_pairs[:, 0]]
    z_b = z_table[edge_pairs[:, 1]]
    e_cnt = int(z_a.shape[0])
    z_mid = 0.50 * (z_a + z_b)
    z_q25 = 0.75 * z_a + 0.25 * z_b
    z_q75 = 0.25 * z_a + 0.75 * z_b
    queries = [z_mid, z_q25, z_q75]

    bary_triples: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    if d_z >= 2 and k_eff >= 2 and n_z <= 64:
        tri_i, tri_a, tri_b = [], [], []
        idx_all = torch.arange(n_z, device=z_table.device)
        for a_i in range(k_eff):
            for b_i in range(a_i + 1, k_eff):
                tri_i.append(idx_all)
                tri_a.append(knn_idx[:, a_i])
                tri_b.append(knn_idx[:, b_i])
        ti = torch.cat(tri_i, dim=0)
        ta = torch.cat(tri_a, dim=0)
        tb = torch.cat(tri_b, dim=0)
        n_tri = int(ti.shape[0])
        max_bary = 64
        if n_tri > max_bary:
            b_sel = ((torch.arange(max_bary, device=z_table.device) * n_tri) // max_bary + int(phase * n_tri)) % n_tri
            ti, ta, tb = ti[b_sel], ta[b_sel], tb[b_sel]
        bary_triples = (ti, ta, tb)
        queries.append((z_table[ti] + z_table[ta] + z_table[tb]) / 3.0)

    z_q = torch.cat(queries, dim=0)
    y_q = gen_fn(z_q).reshape(z_q.shape[0], -1)
    y_end = gen_fn(z_table).reshape(n_z, -1)

    # 1. Pull unseen chord/barycenter outputs onto the nearest point in (c_eb U y_real)
    #    (crisp single-patch preservation on images + 3x denser surface coverage on curved sheets).
    ell_med = ell_proto.median().clamp_min(1e-4)
    c_qp = _born_infeld_physical_cost(y_q, manifold_bank, ell_med.expand(y_q.shape[0]))
    manifold_pull = c_qp.min(dim=1).values.mean()

    # 2. Novelty hinges on chord interiors & 2-simplex barycenters prevent Voronoi lookup step-snapping.
    y_mid = y_q[:e_cnt]
    y_25 = y_q[e_cnt : 2 * e_cnt]
    y_75 = y_q[2 * e_cnt : 3 * e_cnt]
    y_ea = y_end[edge_pairs[:, 0]].detach()
    y_eb_end = y_end[edge_pairs[:, 1]].detach()
    span_ab = (y_ea - y_eb_end).norm(dim=-1).clamp_min(1e-4)
    novelty_pen = (
        F.relu(0.32 * span_ab - (y_mid - y_ea).norm(dim=-1))
        + F.relu(0.32 * span_ab - (y_mid - y_eb_end).norm(dim=-1))
        + 0.4 * F.relu(0.16 * span_ab - (y_25 - y_ea).norm(dim=-1))
        + 0.4 * F.relu(0.16 * span_ab - (y_75 - y_eb_end).norm(dim=-1))
    ).mean()

    if bary_triples is not None:
        ti, ta, tb = bary_triples
        y_bary = y_q[3 * e_cnt :]
        yv0 = y_end[ti].detach()
        yv1 = y_end[ta].detach()
        yv2 = y_end[tb].detach()
        tri_span = torch.minimum(
            (yv0 - yv1).norm(dim=-1),
            torch.minimum((yv1 - yv2).norm(dim=-1), (yv0 - yv2).norm(dim=-1)),
        ).clamp_min(1e-4)
        min_vert_d = torch.minimum(
            (y_bary - yv0).norm(dim=-1),
            torch.minimum((y_bary - yv1).norm(dim=-1), (y_bary - yv2).norm(dim=-1)),
        )
        novelty_pen = novelty_pen + 0.4 * F.relu(0.22 * tri_span - min_vert_d).mean()

    # 3. Post-acquisition geodesic MDS chart unfolding on low-D charts (d_z <= 3) once endpoints settle.
    ema_res = float(getattr(record, "ema_residual", 1.0))
    u_acq = max(0.0, min(1.0, (0.42 - ema_res) / 0.22))
    g_acq = u_acq * u_acq * (3.0 - 2.0 * u_acq)
    chart_loss = z_table.new_zeros(())
    if d_z <= 3 and g_acq > 1e-4:
        with torch.no_grad():
            ye_f32 = y_end.detach().float()
            p_near = pairwise_sq_dist(ye_f32, c_f32).argmin(dim=1)
            d_euc_end = torch.cdist(ye_f32, ye_f32, p=2.0)
            d_geo_end = torch.maximum(d_euc_end, g_dist[p_near][:, p_near])
            off_z = ~torch.eye(n_z, dtype=torch.bool, device=z_table.device)
            med_geo = d_geo_end[off_z].median().clamp_min(1e-5)
            d_geo_norm = (d_geo_end / med_geo).to(dtype=z_table.dtype)
        d_z = torch.sqrt(pairwise_sq_dist(z_table) + 1e-8)
        med_z = d_z.detach()[off_z].median().clamp_min(1e-5)
        d_z_norm = d_z / med_z
        chart_loss = float(g_acq) * (d_z_norm[off_z] - d_geo_norm[off_z]).square().mean()

    s_th = float(getattr(record, "_thermal_scale", 1.0))
    raw_loss = float(g_conn) * (manifold_pull + 0.45 * novelty_pen + 0.30 * chart_loss)
    if s_th < 1.0 and raw_loss.requires_grad:
        raw_loss = raw_loss * s_th + raw_loss.detach() * (1.0 - s_th)
    return raw_loss


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
    scale = torch.maximum(raw_cost.median(), 0.5 * raw_cost.mean()).clamp_min(eps_scale)
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
    scale = torch.maximum(raw_cost.median(), 0.5 * raw_cost.mean()).clamp_min(eps_scale)
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
    """Zero-floor off-diagonal conditional energy U-statistic and local fiber scale ratio."""
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    b_r = int(r_flat.shape[0])
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
    u_tensor = (w_norm * (2.0 * d_rf - d_rr - d_ff)).sum()
    u_signed, ds_val = torch.stack((u_tensor, data_scale)).tolist()
    return float(u_signed), torch.tensor(ds_val, dtype=x_fake.dtype), ones_ratio
