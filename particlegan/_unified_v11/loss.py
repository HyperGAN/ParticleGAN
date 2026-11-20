"""Unified-v11 4-Law + Zero-`eigh` CondFiber GANLoss.

Combines:
1. Zero-`eigh` / zero-`svd` / zero-`pinv` ``UnifiedV11CondFiberBank`` cross-batch sufficient-statistics
   operator bank + midpoint-canceling paired-context covariance FIFO ring for continuous-conditioned
   reverse diffusion chains (``model_ddgan_*``, ``model_ae_ddgan_*``) and conditional mappings.
2. Zero-`eigvalsh` / zero-`svdvals` ``UnifiedV11CouplingLoss`` simplicial & connected-manifold
   deconfliction (reusing ``UnifiedV4CouplingLoss`` & ``zero_eigh_bipartite_deficit_macro_loss``).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .._rhbi.loss import _split_shared_conditional_parts
from .._t1_coupling.charts import project_transport_features
from .._t1_coupling.critic import _lookup_t1c_capture_parts
from .._t3_dynamics.metric import born_infeld_aligned, born_infeld_c_hat_detached, discrete_pair_mask, sinkhorn_slack
from .._unified.loss import _zero_shrinkage_pooled_mean
from .._unified_v4.loss import UnifiedV4CouplingLoss, _PHI1, _principal_axis_power
from ..gan_loss import GANLoss
from .bank import UnifiedV11CondFiberBank, sym_eigensplit_zero_eigh


class UnifiedV11CouplingLoss(UnifiedV4CouplingLoss):
    """Zero-`eigvalsh` / zero-`svdvals` coupling primitive for Unified-v11."""


def v11_zero_eigh_bipartite_deficit_macro_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_hat_xy: torch.Tensor,
    p_row_slack: torch.Tensor,
    p_col_slack: torch.Tensor,
    *,
    rho: float = 3.0,
    g_conn: float = 0.0,
) -> torch.Tensor:
    """Scale-invariant local-NN & Poisson-deficit-gated zero-`eigh` bipartite Monge macro-transport.

    Combines Born-Infeld proximity screening with the scale-invariant local-NN orphan ratio
    ``r_loc(y_j) = min_i ||x_i - y_j|| / NN_4(y_j; Y_real)`` so covered broad clusters next to
    narrow cores (``t3_iso_scale_mass_8x``, ``r_loc <= 1.0``) stay strictly zero while genuinely
    orphaned/shifted modes (``t3_iso_horizon_saddle_shift``, ``r_loc >> 3``) trigger full Monge rescue.
    """
    b_f = x_fake.shape[0]
    b_r = y_real.shape[0]
    zero = x_fake[:0].sum()
    if b_f < 2 or b_r < 2 or g_conn > 0.05:
        return zero

    with torch.no_grad():
        c_f = c_hat_xy.detach().float()
        col_min_c = c_f.amin(dim=0)
        u_orph = ((col_min_c - 2.80 * float(rho)) / max(1.20 * float(rho), 1e-6)).clamp(0.0, 1.0)
        g_prox = u_orph * u_orph * (3.0 - 2.0 * u_orph)

        xf = x_fake.detach().reshape(b_f, -1).float()
        yf = y_real.detach().reshape(b_r, -1).float()
        scale = torch.maximum(xf.abs().amax(), yf.abs().amax()).clamp_min(1.0)
        d_xy = torch.cdist(xf / scale, yf / scale, p=2.0) * scale

        g_orph = g_prox
        if b_r >= 8 and b_f >= 8:
            yu = yf / scale
            d_yy = torch.cdist(yu, yu, p=2.0) * scale
            d_yy_inf = d_yy.clone()
            d_yy_inf.fill_diagonal_(float("inf"))

            # Scale-invariant k-NN density-ratio orphan gate (k=4): zero on covered broad clusters
            # of arbitrary bandwidth (r_loc ~= 1.0), active on under-populated shifted modes until mass arrives:
            k_loc = min(4, b_r - 1, b_f)
            nn_loc = torch.topk(d_yy_inf, k=k_loc, dim=1, largest=False).values[:, -1].clamp_min(1e-6)
            d_xy_k = torch.topk(d_xy, k=k_loc, dim=0, largest=False).values[-1]
            r_loc = d_xy_k / nn_loc
            u_loc = ((r_loc - 3.52) / 1.30).clamp(0.0, 1.0)
            g_loc = u_loc * u_loc * (3.0 - 2.0 * u_loc)
            g_prox = g_prox * g_loc

            med_d1 = torch.quantile(d_yy_inf.amin(dim=1), 0.50).clamp_min(1e-6)
            k_far_m = min(b_r - 1, 2)
            j_far_m = torch.topk(d_yy, k=k_far_m, dim=-1, largest=True).indices[:, -1]
            y_mid = 0.5 * (yu + yu[j_far_m])
            d_my = torch.cdist(y_mid, yu, p=2.0) * scale
            idx_r = torch.arange(b_r, device=yf.device)
            d_my[idx_r, idx_r] = float("inf")
            d_my[idx_r, j_far_m] = float("inf")
            valley_q85 = torch.quantile(d_my.amin(dim=1), 0.85)
            u_val = ((valley_q85 / med_d1 - 3.20) / 1.80).clamp(0.0, 1.0)
            w_val = float((u_val * u_val * (3.0 - 2.0 * u_val)).item())
            if w_val > 1e-4:
                h_bas = (0.38 * valley_q85).clamp_min(1e-4)
                a_yy = torch.exp(-0.5 * (d_yy / h_bas).square())
                pi_y = a_yy.mean(dim=1).clamp(1e-4, 1.0)
                a_xy = torch.exp(-0.5 * (d_xy / h_bas).square())
                pi_x = a_xy.mean(dim=0)
                u_zd = ((float(b_f) * pi_y - 4.2) / 2.3).clamp(0.0, 1.0)
                s_zd = u_zd * u_zd * (3.0 - 2.0 * u_zd)
                u_starv = ((0.45 - pi_x / pi_y) / 0.20).clamp(0.0, 1.0)
                s_starv = u_starv * u_starv * (3.0 - 2.0 * u_starv)
                g_orph = g_prox * ((1.0 - w_val) + w_val * s_zd * s_starv)
            else:
                g_orph = g_prox

        s_col = (float(b_r) * p_col_slack.detach().float()).clamp(0.0, 1.0)
        q_raw = s_col * g_orph
        m_orph_val = float(q_raw.mean().item())
        if m_orph_val <= 1e-4:
            return zero

        s_row = (float(b_f) * p_row_slack.detach().float()).clamp(0.0, 1.0)
        med_d = torch.quantile(d_xy.reshape(-1), 0.50).clamp_min(1e-4)
        ell_macro = (0.25 * med_d).clamp_min(1e-5)

        idx_f = torch.arange(b_f, device=x_fake.device, dtype=torch.float32)
        det_tie = 1e-4 * torch.remainder((idx_f + 0.5) * _PHI1, 1.0)

        if m_orph_val >= 0.85:
            k_exp = b_f
            exp_idx = torch.arange(b_f, device=x_fake.device, dtype=torch.long)
        else:
            k_exp = min(b_f - 1, int(round(float(b_f) * m_orph_val)))
            if k_exp < 1:
                return zero
            orph_active = (g_orph > 0.05).float()
            if bool((orph_active > 0.0).any()):
                d_orph_min = (d_xy + (1.0 - orph_active).unsqueeze(0) * (1e6 * med_d)).amin(dim=1)
            else:
                d_orph_min = d_xy.amin(dim=1)
            score = 1.50 * s_row - 0.65 * (d_orph_min / med_d).clamp_max(4.0) + det_tie
            exp_idx = torch.sort(torch.topk(score, k=k_exp, largest=True).indices).values

        p_marg = torch.full((k_exp,), 1.0 / float(k_exp), device=x_fake.device, dtype=torch.float32)
        q_marg = q_raw + 1e-7
        q_marg = q_marg / q_marg.sum().clamp_min(1e-12)

        if yf.shape[1] >= 2 and b_r >= 2:
            y_mean_c = (q_marg.unsqueeze(1) * yf).sum(dim=0, keepdim=True)
            yc_c = (yf - y_mean_c) * q_marg.sqrt().unsqueeze(1)
            gram_c = (yc_c.T @ yc_c).double()
            axis_c = _principal_axis_power(gram_c, iters=6).float()
            col_order = torch.argsort(yf @ axis_c)
        else:
            col_order = torch.argsort(yf[:, 0])
        q_ord = q_marg[col_order]

        cost_norm = d_xy[exp_idx][:, col_order] / med_d
        L_mac = -cost_norm / 0.03
        log_p = p_marg.log()
        log_q = q_ord.clamp_min(1e-12).log()
        u = torch.zeros_like(log_p)
        v = torch.zeros_like(log_q)
        for _ in range(10):
            v = log_q - torch.logsumexp(L_mac + u.unsqueeze(1), dim=0)
            u = log_p - torch.logsumexp(L_mac + v.unsqueeze(0), dim=1)
        pi_mac = torch.exp(L_mac + u.unsqueeze(1) + v.unsqueeze(0))

        row_cond = pi_mac / pi_mac.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cdf = torch.cumsum(row_cond, dim=1)
        cdf[:, -1] = 1.0
        idx_e_long = torch.arange(k_exp, device=x_fake.device, dtype=torch.long)
        half = (k_exp + 1) // 2
        slot = torch.where(idx_e_long % 2 == 0, idx_e_long // 2, half + (idx_e_long // 2))
        q_u = ((slot.float() + 0.5) / float(k_exp)).clamp(1e-6, 1.0 - 1e-6).unsqueeze(1)
        j_ord = torch.searchsorted(cdf, q_u).squeeze(1).clamp(0, b_r - 1)
        y_matched = y_real.detach().reshape(b_r, -1)[col_order[j_ord]]

    chord_cost = born_infeld_aligned(x_fake.reshape(b_f, -1)[exp_idx], y_matched, ell_macro)
    return chord_cost.sum() / float(b_f)


def _odd_radial_saturate(v: torch.Tensor, tau: float) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (tau * torch.tanh(norm / tau)) / norm
    return v * scale


def _odd_frob_saturate(mat: torch.Tensor, tau: float) -> torch.Tensor:
    norm = mat.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    scale = (tau * torch.tanh(norm / tau)) / norm
    return mat * scale


def _group_strata_cpu(
    b_n: int,
    cond_disc: Optional[torch.Tensor],
) -> List[Tuple[Tuple[int, ...], torch.Tensor]]:
    cpu = torch.device("cpu")
    if cond_disc is None:
        return [((0,), torch.arange(b_n, dtype=torch.long, device=cpu))]
    cd_cpu = cond_disc.detach().reshape(b_n, -1).to(device=cpu, dtype=torch.int64)
    buckets: Dict[Tuple[int, ...], List[int]] = {}
    rows = cd_cpu.tolist()
    for i, row in enumerate(rows):
        key = tuple(int(v) for v in row)
        if key not in buckets:
            buckets[key] = [i]
        else:
            buckets[key].append(i)
    return [(key, torch.tensor(buckets[key], dtype=torch.long, device=cpu)) for key in sorted(buckets)]


def unified_v11_conditional_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
    record: Any = None,
    mean_op_weight: float = 0.36,
    cov_op_weight: float = 0.30,
    imq_active_scale: float = 0.40,
    bank_mean_blend: float = 0.85,
) -> torch.Tensor:
    """Zero-`eigh` cross-batch paired-ring + sufficient-statistics conditional fiber discrepancy."""
    b_n = x_fake.shape[0]
    if b_n <= 1 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
        return x_fake[:0].sum()

    x_flat = x_fake.reshape(b_n, -1)
    y_flat = y_real.detach().reshape(b_n, -1)
    orig_dev = x_flat.device
    dim = max(1, x_flat.shape[-1])
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))

    bank: Optional[UnifiedV11CondFiberBank] = (
        getattr(record, "cond_fiber_bank", None) if record is not None else None
    )
    allow_state_update = (
        record is not None
        and torch.is_grad_enabled()
        and not getattr(record, "_in_protected_probe", False)
        and not getattr(record, "_in_probe", False)
    )

    with torch.no_grad():
        same_t = discrete_pair_mask(cond_disc, cond_disc, b_n, b_n)
        if same_t is None:
            same_t = torch.ones((b_n, b_n), dtype=torch.bool, device=orig_dev)
        m_hollow = same_t.clone()
        m_hollow.fill_diagonal_(False)
        valid_row = m_hollow.any(dim=1)
        m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)
        if not bool(m_valid.any()):
            return x_fake[:0].sum()

        c_f32 = c_cont.detach().reshape(b_n, -1).float()
        x_det32 = x_flat.detach().float()
        y_det32 = y_flat.float()
        d_c = int(c_f32.shape[1])

        cpu = torch.device("cpu")
        strata_groups = _group_strata_cpu(b_n, cond_disc)
        cxy_cpu64 = torch.cat((c_f32, x_det32, y_det32), dim=1).to(device=cpu, dtype=torch.float64)
        c_cpu64 = cxy_cpu64[:, :d_c]
        x_cpu64 = cxy_cpu64[:, d_c : d_c + dim]
        y_cpu64 = cxy_cpu64[:, d_c + dim :]

        has_op_row = torch.zeros(b_n, dtype=torch.bool, device=cpu)
        op_w_mat = torch.eye(dim, dtype=torch.float32, device=cpu).unsqueeze(0).expand(b_n, dim, dim).clone()
        op_dm_w = torch.zeros((b_n, dim), dtype=torch.float32, device=cpu)
        op_mx = torch.zeros((b_n, dim), dtype=torch.float32, device=cpu)
        op_ecov_w = torch.zeros((b_n, dim, dim), dtype=torch.float32, device=cpu)
        op_row_w = torch.zeros(b_n, dtype=torch.float32, device=cpu)
        m_pool_cpu = torch.zeros((b_n, dim), dtype=torch.float32, device=cpu)
        m_pool_gate = torch.zeros((b_n, 1), dtype=torch.float32, device=cpu)

        pending_updates: List[Tuple[Tuple[int, ...], torch.Tensor, float]] = []
        r_op_worst = 0.0

        if bank is not None:
            for key, idx_cpu in strata_groups:
                n_g = int(idx_cpu.shape[0])
                if n_g == 0:
                    continue
                c_g = c_cpu64[idx_cpu]
                x_g = x_cpu64[idx_cpu]
                y_g = y_cpu64[idx_cpu]
                q = bank.query_stratum(key, c_g, x_g, y_g)
                z_innov = 0.0
                if q is not None:
                    ring_len = float(q["ring_len"])
                    w_mat_gate = ring_len / (ring_len + 32.0)
                    m_pool_cpu[idx_cpu] = q["m_pool"].float()
                    m_pool_gate[idx_cpu] = float(w_mat_gate)

                    w_mat = q["w_mat"]
                    dm_blend = (1.0 - bank_mean_blend) * q["delta_m_loo"] + bank_mean_blend * q["delta_m_bank"]
                    dm_w = _odd_radial_saturate(dm_blend @ w_mat, tau=3.0).float()

                    w_mat32 = w_mat.float()
                    dcov32 = q["delta_cov_loc"].float()
                    ecov_w = torch.matmul(w_mat32.unsqueeze(0), torch.matmul(dcov32, w_mat32.unsqueeze(0)))
                    ecov_w = _odd_frob_saturate(0.5 * (ecov_w + ecov_w.transpose(-1, -2)), tau=1.5)

                    omega_g = q["omega"].float()
                    omega_norm = omega_g / omega_g.mean().clamp_min(1e-6)

                    has_op_row[idx_cpu] = True
                    op_w_mat[idx_cpu] = w_mat32.unsqueeze(0)
                    op_dm_w[idx_cpu] = dm_w
                    op_mx[idx_cpu] = q["m_x"].float()
                    op_ecov_w[idx_cpu] = ecov_w
                    op_row_w[idx_cpu] = float(w_mat_gate) * omega_norm
                    z_innov = float(q["z_innov"])

                    r_stratum = float(q["mean_err_sq"])
                    if r_stratum > r_op_worst:
                        r_op_worst = r_stratum
                pending_updates.append((key, idx_cpu, z_innov))

        if bool(has_op_row.all()):
            m_hat32 = m_pool_cpu.to(device=orig_dev)
        else:
            z_pool32 = 0.5 * (x_det32 + y_det32)
            m_hat_fallback32 = _zero_shrinkage_pooled_mean(c_f32, z_pool32, cond_disc)
            if bool(has_op_row.any()):
                gate_dev = m_pool_gate.to(device=orig_dev)
                m_hat32 = gate_dev * m_pool_cpu.to(device=orig_dev) + (1.0 - gate_dev) * m_hat_fallback32
            else:
                m_hat32 = m_hat_fallback32

        if bank is not None and allow_state_update:
            for key, idx_cpu, z_innov in pending_updates:
                bank.update_stratum(
                    key,
                    c_cpu64[idx_cpu],
                    x_cpu64[idx_cpu],
                    y_cpu64[idx_cpu],
                    z_innov=z_innov,
                )

        c_scale = c_f32.abs().amax().clamp_min(1.0)
        d_c_mat = torch.cdist(c_f32 / c_scale, c_f32 / c_scale, p=2.0)
        d_c_max = d_c_mat.amax()
        d_c_masked = d_c_mat.masked_fill(~m_valid, float("inf"))
        k_top = min(4, b_n)
        d_sorted, sort_idx = torch.topk(d_c_masked, k=k_top, dim=1, largest=False)
        nn_idx = sort_idx[:, 0]

        n_i = m_valid.sum(dim=1)
        k_idx = torch.minimum(torch.full_like(n_i, min(3, k_top)), n_i.clamp_min(1)) - 1
        sigma_raw = d_sorted.gather(1, k_idx.unsqueeze(1)).squeeze(1)
        sigma_floor = 1e-3 * d_c_max + 1e-6
        sigma_i = torch.where(
            valid_row,
            torch.where(torch.isfinite(sigma_raw), sigma_raw, torch.zeros_like(sigma_raw)).clamp_min(0.0)
            + sigma_floor,
            torch.ones_like(sigma_raw),
        )

        y_res = y_det32 - m_hat32
        y_sc = y_res.abs().amax().clamp_min(1e-3)
        innov_d = (y_res - y_res[nn_idx]) * (1.0 / math.sqrt(2.0))
        same_valid = same_t & valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
        f_mask = same_valid.float()
        f_cnt = f_mask.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        ell_d_sq_raw = (f_mask @ innov_d.square()) / f_cnt
        ell_iso_sq = ell_d_sq_raw.mean(dim=-1, keepdim=True)
        ell_d_sq = 0.50 * ell_iso_sq + 0.50 * ell_d_sq_raw
        ell_d_floor = (1e-2 * y_sc / math.sqrt(float(dim))) + 1e-5
        ell_d = torch.where(
            valid_row.unsqueeze(1),
            ell_d_sq.sqrt().clamp_min(ell_d_floor),
            torch.ones_like(ell_d_sq),
        )
        ell_rms = ell_d.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        w_coord = ell_rms / ell_d
        w_coord = w_coord / w_coord.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        ell_f = ell_rms * math.sqrt(float(dim))

        denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
        k_c = torch.where(m_valid, torch.exp(-d_c_mat.square() / denom_2s), 0.0)
        deg_c = 1.0 + k_c.sum(dim=1)
        inv_sqrt_deg = torch.rsqrt(deg_c)
        w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
        w_imq_sym = 0.5 * (w_imq + w_imq.T)
        w_imq_dt = w_imq_sym.to(x_flat.dtype)

        has_any_op = bool(has_op_row.any())
        if has_any_op:
            if cov_op_weight > 0.0:
                op_ecov_dev = op_ecov_w.to(device=orig_dev)
                op_epos_dev, op_eneg_dev = sym_eigensplit_zero_eigh(op_ecov_dev)
            op_w_mat_dev = op_w_mat.to(device=orig_dev)
            op_dm_w_dev = op_dm_w.to(device=orig_dev)
            op_mx_dev = op_mx.to(device=orig_dev)
            op_row_w_dev = op_row_w.to(device=orig_dev)

    x_f32 = x_flat.float()
    xw = (x_f32 - m_hat32) * w_coord
    yw = (y_det32 - m_hat32) * w_coord
    sq_w_xy = (
        xw.square().sum(dim=-1, keepdim=True)
        + yw.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (xw @ yw.T)
    )
    sq_w_xy = sq_w_xy + (sq_w_xy.detach().clamp_min(0.0) - sq_w_xy.detach())
    sq_w_xx = (
        xw.square().sum(dim=-1, keepdim=True)
        + xw.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (xw @ xw.T)
    )
    sq_w_xx = sq_w_xx + (sq_w_xx.detach().clamp_min(0.0) - sq_w_xx.detach())

    psi_xy = torch.zeros_like(sq_w_xy)
    psi_xx = torch.zeros_like(sq_w_xx)
    for s_mult in (0.35, 1.0, 2.8):
        ell_s = ell_f * s_mult
        ell_s2 = ell_s.square()
        rad_s_xy = torch.sqrt(ell_s2 + sq_w_xy)
        psi_xy = psi_xy + (ell_s * sq_w_xy) / (rad_s_xy * (ell_s + rad_s_xy))
        rad_s_xx = torch.sqrt(ell_s2 + sq_w_xx)
        psi_xx = psi_xx + (ell_s * sq_w_xx) / (rad_s_xx * (ell_s + rad_s_xx))
    psi_xy_t = (psi_xy * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)
    psi_xx_t = (psi_xx * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)
    loss_imq = (w_imq_dt * psi_xy_t).sum() - 0.5 * (w_imq_dt * psi_xx_t).sum()

    loss_op = x_flat.new_zeros(())
    if has_any_op and (mean_op_weight > 0.0 or cov_op_weight > 0.0):
        inv_dim = 1.0 / float(dim)
        x_w = torch.einsum("nij,nj->ni", op_w_mat_dev, x_f32)
        if mean_op_weight > 0.0:
            mean_inner = (x_w * op_dm_w_dev).sum(dim=-1) * inv_dim
            loss_mean = (op_row_w_dev * mean_inner).mean()
            loss_op = loss_op + float(mean_op_weight) * loss_mean.to(dtype=x_flat.dtype)
        if cov_op_weight > 0.0:
            rx_w = torch.einsum("nij,nj->ni", op_w_mat_dev, x_f32 - op_mx_dev)
            cov_pos = 0.5 * torch.einsum("ni,nij,nj->n", rx_w, op_epos_dev, rx_w) * inv_dim
            rx_norm_sq = rx_w.square().sum(dim=-1, keepdim=True) / float(dim)
            rx_sat = rx_w * torch.rsqrt(1.0 + 0.25 * rx_norm_sq)
            cov_neg = 0.5 * torch.einsum("ni,nij,nj->n", rx_sat, op_eneg_dev, rx_sat) * inv_dim
            loss_cov = (op_row_w_dev * (cov_pos + cov_neg)).mean()
            loss_op = loss_op + float(cov_op_weight) * loss_cov.to(dtype=x_flat.dtype)

    if (
        record is not None
        and hasattr(record, "t1c_cond_initialized")
        and allow_state_update
    ):
        with torch.no_grad():
            yw_det = yw.detach()
            sq_w_yy = (
                yw_det.square().sum(dim=-1, keepdim=True)
                + yw_det.square().sum(dim=-1).unsqueeze(0)
                - 2.0 * (yw_det @ yw_det.T)
            ).clamp_min(0.0)
            psi_yy = torch.zeros_like(sq_w_yy)
            for s_mult in (0.35, 1.0, 2.8):
                ell_s = ell_f * s_mult
                ell_s2 = ell_s.square()
                rad_s_yy = torch.sqrt(ell_s2 + sq_w_yy)
                psi_yy = psi_yy + (ell_s * sq_w_yy) / (rad_s_yy * (rad_s_yy + ell_s))
            psi_yy = psi_yy * (1.0 / 3.0)
            psi_xy_d = psi_xy.detach() * (1.0 / 3.0)
            psi_xx_d = psi_xx.detach() * (1.0 / 3.0)
            num_c = (w_imq_sym * (2.0 * psi_xy_d - psi_xx_d - psi_yy)).sum()
            den_c = (w_imq_sym * psi_yy).sum().clamp_min(1e-8)
            r_imq = max(-0.5, min(0.5, float((num_c / den_c).item())))
            r_cond = r_imq + min(0.5, 0.11 * max(0.0, float(r_op_worst) - 0.10))
            if not record.t1c_cond_initialized:
                record.t1c_cond_ema = max(0.18, r_cond)
                record.t1c_cond_initialized = True
            else:
                record.t1c_cond_ema = 0.99 * float(record.t1c_cond_ema) + 0.01 * r_cond
            u_cond = min(1.0, max(0.0, (0.050 - max(0.0, float(record.t1c_cond_ema))) / 0.024))
            s_cond = u_cond * u_cond * (3.0 - 2.0 * u_cond)
            if hasattr(record, "_step_scale"):
                record._step_scale = float(1.0 - 0.85 * s_cond)

    imq_scale = float(imq_active_scale) if (has_any_op and (mean_op_weight > 0.0 or cov_op_weight > 0.0)) else 1.0
    return (imq_scale * loss_imq + loss_op).clamp(-fmax, fmax).to(x_flat.dtype)


class UnifiedV11GANLoss(GANLoss):
    """Unified-v11 4-Law + Zero-`eigh` CondFiber GANLoss."""

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        weight: float = 1.0,
        rho: float = 3.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        critic_weight: float = 0.0,
        self_coupling: str = "feydy_discrete",
        relativistic_weight: float = 0.0,
        rapidity_screening: bool = False,
        scale_log_odds: bool = False,
        conditional_fibre: str = "slack",
        akt_horizon: bool = True,
        akt_mode: str = "weibull",
        slack_gain: float = 0.35,
        thermal_tau: float = 0.084,
        thermal_shock_mu: float = 0.35,
        fibre_rel_blend: float = 0.50,
        cond_ot_weight: float = 1.5,
        cond_ot_mode: str = "diag_imq_cmmd",
        cvx_mu_gain: float = 0.58,
        tail_spill_gain: float = 0.95,
        simplex_macro_settle: bool = True,
        simplex_curve_settle: bool = True,
        support_completion: bool = True,
        chart_capacity: int = 256,
        chart_weight: float = 1.4,
        halo_mult: float = 2.1,
        multiscale_horizon: bool = True,
        cond_drift_deconflict: bool = True,
        macro_slack_weight: float = 1.0,
        mean_op_weight: float = 0.36,
        cov_op_weight: float = 0.30,
        imq_active_scale: float = 0.40,
        bank_mean_blend: float = 0.85,
        record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        for name, val in (
            ("critic_weight", critic_weight),
            ("cond_ot_weight", cond_ot_weight),
            ("macro_slack_weight", macro_slack_weight),
            ("mean_op_weight", mean_op_weight),
            ("cov_op_weight", cov_op_weight),
            ("imq_active_scale", imq_active_scale),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, val in (("fibre_rel_blend", fibre_rel_blend), ("bank_mean_blend", bank_mean_blend)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0 or val > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if type(cond_ot_mode) is not str or cond_ot_mode not in ("imq_cmmd", "diag_imq_cmmd"):
            raise ValueError("cond_ot_mode must be 'imq_cmmd' or 'diag_imq_cmmd'")
        if type(conditional_fibre) is not str or conditional_fibre not in ("joint", "slack", "calibrated"):
            raise ValueError("conditional_fibre must be 'joint', 'slack', or 'calibrated'")

        self.critic_weight = float(critic_weight)
        self.conditional_fibre = conditional_fibre
        self.fibre_rel_blend = float(fibre_rel_blend)
        self.cond_ot_weight = float(cond_ot_weight)
        self.cond_ot_mode = cond_ot_mode
        self.macro_slack_weight = float(macro_slack_weight)
        self.mean_op_weight = float(mean_op_weight)
        self.cov_op_weight = float(cov_op_weight)
        self.imq_active_scale = float(imq_active_scale)
        self.bank_mean_blend = float(bank_mean_blend)
        self.record = record
        self.t1c = UnifiedV11CouplingLoss(
            weight=weight,
            rho=rho,
            tau=tau,
            eps=eps,
            iters=iters,
            c_max=c_max,
            beta_ell=beta_ell,
            nn_quantile=nn_quantile,
            rapidity_screening=rapidity_screening,
            self_coupling=self_coupling,
            relativistic_weight=relativistic_weight,
            scale_log_odds=scale_log_odds,
            akt_horizon=akt_horizon,
            akt_mode=akt_mode,
            slack_gain=slack_gain,
            thermal_tau=thermal_tau,
            thermal_shock_mu=thermal_shock_mu,
            cvx_mu_gain=cvx_mu_gain,
            tail_spill_gain=tail_spill_gain,
            simplex_macro_settle=simplex_macro_settle,
            simplex_curve_settle=simplex_curve_settle,
            support_completion=support_completion,
            chart_capacity=chart_capacity,
            chart_weight=chart_weight,
            halo_mult=halo_mult,
            multiscale_horizon=multiscale_horizon,
            cond_drift_deconflict=cond_drift_deconflict,
        )
        self.last_metrics: Dict[str, float] = {}

    def _resolve_pair_parts(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Any,
        Any,
        Any,
        bool,
        Optional[Tuple[torch.Tensor, ...]],
        Optional[Tuple[torch.Tensor, ...]],
        bool,
    ]:
        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_t1c_capture_parts(fake_logits, self.record)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_t1c_capture_parts(
            real_logits, rec_f if rec_f is not None else self.record
        )
        rec = rec_f if rec_f is not None else (rec_r if rec_r is not None else self.record)
        has_cont_cond = bool(cc_f or cc_r)
        captured = x_f is not None and x_r is not None
        if not captured:
            x_f = fake_logits.reshape(fake_logits.shape[0], -1)
            x_r = real_logits.reshape(real_logits.shape[0], -1)
            cond_f = None
            cond_r = None
            has_cont_cond = False
            parts_f = None
            parts_r = None
        return x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r, captured

    @staticmethod
    def _scalar_pair_relativistic(
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        *,
        detach_neg: bool = False,
    ) -> Optional[torch.Tensor]:
        b_p, b_n = int(pos_logits.shape[0]), int(neg_logits.shape[0])
        if b_p != b_n or b_p < 1:
            return None
        p_flat = pos_logits.reshape(b_p, -1).mean(dim=1) if pos_logits.ndim > 1 else pos_logits.reshape(b_p)
        n_src = neg_logits.detach() if detach_neg else neg_logits
        n_flat = n_src.reshape(b_n, -1).mean(dim=1) if n_src.ndim > 1 else n_src.reshape(b_n)
        return F.softplus(n_flat - p_flat).mean()

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        base = super().d_loss(real_logits, fake_logits)
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, _, _, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if rec is not None and hasattr(rec, "observe_margin_and_scale") and not has_cont_cond:
            if captured:
                rec.observe_margin_and_scale(real_logits, fake_logits, x_r, x_f)
            else:
                rec.observe_margin_and_scale(real_logits, fake_logits)
        if self.critic_weight == 0.0 and self.fibre_rel_blend == 0.0:
            return base
        if not captured:
            if self.loss == "non_saturating" and self.t1c.weight > 0.0 and self.fibre_rel_blend > 0.0:
                pair_d = self._scalar_pair_relativistic(real_logits, fake_logits)
                if pair_d is not None:
                    return (1.0 - self.fibre_rel_blend) * base + self.fibre_rel_blend * pair_d
            return base
        if self.conditional_fibre == "slack" and has_cont_cond:
            if self.fibre_rel_blend > 0.0:
                b_r = real_logits.shape[0]
                b_f = fake_logits.shape[0]
                if b_r == b_f:
                    r_flat = (
                        real_logits.reshape(b_r, -1).mean(dim=1)
                        if real_logits.ndim > 1
                        else real_logits.reshape(b_r)
                    )
                    f_flat = (
                        fake_logits.reshape(b_f, -1).mean(dim=1)
                        if fake_logits.ndim > 1
                        else fake_logits.reshape(b_f)
                    )
                    pair_d = F.softplus(f_flat - r_flat).mean()
                    return (1.0 - self.fibre_rel_blend) * base + self.fibre_rel_blend * pair_d
            return base
        if self.conditional_fibre == "calibrated" and has_cont_cond:
            return base
        if self.critic_weight == 0.0:
            return base
        slack_reg = self.t1c.critic_slack_loss(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            critic_weight=self.critic_weight,
            record=rec,
            cond_fake=cond_f,
            cond_real=cond_r,
        )
        return base + slack_reg

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        if real_logits is None:
            if self.t1c.weight == 0.0:
                return super().g_loss(fake_logits, None)
            raise ValueError("UnifiedV11GANLoss requires real_logits in g_loss when t1c_weight > 0")
        if self.t1c.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)

        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if not captured:
            base_g = super().g_loss(fake_logits, real_logits)
            if self.loss == "non_saturating" and self.fibre_rel_blend > 0.0:
                pair_g = self._scalar_pair_relativistic(fake_logits, real_logits, detach_neg=True)
                if pair_g is not None:
                    return self.t1c.weight * ((1.0 - self.fibre_rel_blend) * base_g + self.fibre_rel_blend * pair_g)
            return self.t1c.weight * base_g

        b_f = fake_logits.shape[0]
        b_r = real_logits.shape[0]

        # 1. Continuous-conditioned fibers (DDGAN / AE-DDGAN / residual_student):
        if self.conditional_fibre == "calibrated" and has_cont_cond:
            return self.t1c.weight * super().g_loss(fake_logits, real_logits)
        if self.conditional_fibre == "slack" and has_cont_cond:
            if b_f == b_r:
                f_flat = fake_logits.reshape(b_f, -1).mean(dim=1) if fake_logits.ndim > 1 else fake_logits.reshape(b_f)
                r_flat = (
                    real_logits.detach().reshape(b_r, -1).mean(dim=1)
                    if real_logits.ndim > 1
                    else real_logits.detach().reshape(b_r)
                )
                rel_term = F.softplus(r_flat - f_flat).mean()
                if self.cond_ot_weight > 0.0:
                    same_disc = (cond_f is None and cond_r is None) or (
                        cond_f is not None
                        and cond_r is not None
                        and cond_f.shape == cond_r.shape
                        and torch.equal(cond_f, cond_r)
                    )
                    if same_disc:
                        x_sf, x_sr, c_cont = _split_shared_conditional_parts(parts_f, parts_r)
                        if x_sf is not None and x_sr is not None and c_cont is not None:
                            cond_ot = unified_v11_conditional_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
                                record=rec,
                                mean_op_weight=self.mean_op_weight,
                                cov_op_weight=self.cov_op_weight,
                                imq_active_scale=self.imq_active_scale,
                                bank_mean_blend=self.bank_mean_blend,
                            )
                            if rec is not None and hasattr(rec, "_simplicial_step"):
                                rec._simplicial_step = False
                            s_th = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
                            total_cond = rel_term + self.cond_ot_weight * cond_ot
                            if s_th < 1.0 and total_cond.requires_grad:
                                if getattr(fake_logits, "_t1c_thermally_scaled", False):
                                    cond_scaled = cond_ot * s_th + cond_ot.detach() * (1.0 - s_th)
                                    total_cond = rel_term + self.cond_ot_weight * cond_scaled
                                else:
                                    total_cond = total_cond * s_th + total_cond.detach() * (1.0 - s_th)
                            return self.t1c.weight * total_cond
                return self.t1c.weight * rel_term
            return self.t1c.weight * super().g_loss(fake_logits, real_logits)

        # 2. High-resolution RGB image domains (dim > 1024, e.g. 32x32..256x256 DCGAN):
        dim = max(1, x_r[0].numel()) if b_r > 0 else 0
        if x_r.ndim >= 4 and dim > 1024:
            self.t1c._discrete_prior_step = False
            self.t1c._simplicial_active_step = False
            if rec is not None and hasattr(rec, "_t1c_disc_gate"):
                rec._t1c_disc_gate = 0.0
            if rec is not None and hasattr(rec, "_simplicial_step"):
                rec._simplicial_step = False
            base_g = super().g_loss(fake_logits, real_logits)
            if self.loss == "non_saturating" and self.fibre_rel_blend > 0.0:
                pair_g = self._scalar_pair_relativistic(fake_logits, real_logits, detach_neg=True)
                if pair_g is not None:
                    return self.t1c.weight * ((1.0 - self.fibre_rel_blend) * base_g + self.fibre_rel_blend * pair_g)
            return base_g

        # 3. H_0-connectedness-gated chart coupling + zero-eigh orphan macro-transport:
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        is_discrete = bool(self.t1c._discrete_prior_step)
        is_simplicial = bool(self.t1c._simplicial_active_step)
        if rec is not None and hasattr(rec, "_simplicial_step"):
            rec._simplicial_step = is_simplicial
        saved_cw = self.t1c.chart_weight

        g_conn_fwd = 0.0
        if (is_simplicial and saved_cw > 0.0) or (
            self.macro_slack_weight > 0.0
            and is_discrete
            and not is_simplicial
            and cond_f is None
            and cond_r is None
            and x_f.ndim == 2
            and 2 <= dim <= 16
        ):
            if b_r >= 4:
                y_feat = project_transport_features(x_r.detach(), max_spatial=self.t1c.max_spatial)
                g_conn_fwd, _, _, _ = self.t1c._compute_manifold_gates(y_feat)

        if is_simplicial and saved_cw > 0.0:
            self.t1c.chart_weight = saved_cw * max(0.0, 1.0 - float(g_conn_fwd))

        try:
            loss_val, _ = self.t1c(
                x_f,
                x_r,
                fake_logits,
                real_logits,
                record=rec,
                symmetric=False,
                allow_ell_update=allow_ell,
                cond_fake=cond_f,
                cond_real=cond_r,
                return_metrics=False,
            )
            if is_simplicial and rec is not None and hasattr(rec, "_t1c_disc_gate"):
                rec._t1c_disc_gate = 0.0
        finally:
            self.t1c.chart_weight = saved_cw

        if (
            self.macro_slack_weight > 0.0
            and is_discrete
            and not is_simplicial
            and cond_f is None
            and cond_r is None
            and x_f.ndim == 2
            and 2 <= dim <= 16
            and g_conn_fwd <= 0.05
            and b_f >= 4
            and b_r >= 4
        ):
            with torch.no_grad():
                ell_m = self.t1c._update_ell(x_r, record=rec, allow_update=False)
                c_hat_m = born_infeld_c_hat_detached(x_f, x_r, ell_m)
                f_1d = fake_logits.detach().reshape(b_f, -1).mean(dim=1)
                r_1d = real_logits.detach().reshape(b_r, -1).mean(dim=1)
                c_mid = 0.5 * (f_1d.mean() + r_1d.mean())
                d_f_tilde = (f_1d - c_mid).clamp(-12.0, 12.0)
                d_r_tilde = (r_1d - c_mid).clamp(-12.0, 12.0)
                P_m = sinkhorn_slack(
                    c_hat_m,
                    d_f_tilde,
                    d_r_tilde,
                    rho=self.t1c.rho,
                    tau=self.t1c.tau,
                    eps=self.t1c.eps,
                    iters=self.t1c.iters,
                )
            macro_loss = v11_zero_eigh_bipartite_deficit_macro_loss(
                x_f,
                x_r,
                c_hat_m,
                P_m[:b_f, b_r],
                P_m[b_f, :b_r],
                rho=self.t1c.rho,
                g_conn=g_conn_fwd,
            )
            s_th = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            if s_th < 1.0 and macro_loss.requires_grad:
                macro_loss = macro_loss * s_th + macro_loss.detach() * (1.0 - s_th)
            loss_val = loss_val + (self.t1c.weight * self.macro_slack_weight) * macro_loss

        return loss_val

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        if self.t1c.weight == 0.0:
            return super().joint_g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, _, _, _, captured = self._resolve_pair_parts(fake_logits, real_logits)
        if not captured:
            return super().joint_g_loss(fake_logits, real_logits)
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        loss_val, _ = self.t1c.joint_relativistic_loss(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            record=rec,
            allow_ell_update=allow_ell,
            cond_fake=cond_f,
            cond_real=cond_r,
            return_metrics=False,
        )
        return loss_val


__all__ = [
    "UnifiedV11CouplingLoss",
    "UnifiedV11GANLoss",
    "unified_v11_conditional_loss",
]
