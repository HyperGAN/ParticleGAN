"""Axiom L2 Law 3A Born-Infeld OT, chart routing, simplicial, and joint transport."""

from __future__ import annotations

import math
from typing import Any, Callable
import weakref

import torch
from torch import (
    Tensor,
    arange,
    bincount,
    cat,
    eye,
    full,
    nn,
    no_grad,
    ones,
    quantile,
    searchsorted,
    softmax,
    stack,
    where,
    zeros_like,
)
import torch.nn.functional as F

from .atlas import (
    MAX_LEDGER_PARTICLES,
    PrototypeAtlas,
    _group_rank,
    batch_annular_gap_ratio,
    cov_trace_mean,
    manifold_connectivity_stats,
)
from .fiber import AxiomCondFiberBank, conditional_fiber_loss
from .math_ops import (
    PHI1,
    PHI2,
    born_infeld_cost,
    hamilton_quotas,
    log_sinkhorn,
    power_principal_axis,
    safe_cdist,
    sinkhorn_with_slack,
    smoothstep01,
    sym_eigensplit_halley,
)
from .state import recover_active_prior_rows, set_active_prior_rows

_INF = float("inf")
_F32, _F64, _BOOL, _LONG = torch.float32, torch.float64, torch.bool, torch.long
_ATLAS_EXPORT_KEYS = (
    "max_prototypes",
    "beta_n",
    "initialized",
    "dim",
    "proto_c",
    "proto_cov",
    "proto_w",
    "w_fake_ema",
    "proto_n",
    "psi",
    "v_push",
)
_LEDGER_EXPORT_KEYS = ("num_particles", "dim", "row_comp", "row_target", "row_dist")
_INT_CONFIG_KEYS = frozenset(("iters", "chart_capacity", "max_spatial", "max_flat_dim"))
_FIBER_CONFIG_KEYS = ("mean_op_weight", "cov_op_weight", "imq_active_scale", "bank_mean_blend")
_TRANSPORT_DEFAULTS: dict[str, Any] = {
    "weight": 1.0,
    "rho": 3.0,
    "tau": 2.0,
    "eps": 0.1,
    "iters": 8,
    "chart_capacity": 256,
    "chart_weight": 1.4,
    "halo_mult": 2.1,
    "max_spatial": 8,
    "max_flat_dim": 256,
    "mean_op_weight": 0.36,
    "cov_op_weight": 0.35,
    "imq_active_scale": 0.4,
    "bank_mean_blend": 0.85,
}


def _scale_grad(tensor, scale: float):
    return (tensor * scale + tensor.detach() * (1 - scale)) if (scale < 1 and tensor.requires_grad) else tensor


def _can_mutate_state(record, allow_update: bool = True) -> bool:
    in_probe = getattr(record, "_in_probe", False) or getattr(record, "_in_protected_probe", False)
    return allow_update and torch.is_grad_enabled() and not in_probe


def _flatten_logits(logits):
    return logits.reshape(logits.shape[0], -1).mean(1) if logits.ndim > 1 else logits.reshape(-1)


def _compute_log_odds(d_fake, d_real, cap: float = 20.0):
    fake_flat, real_flat = _flatten_logits(d_fake), _flatten_logits(d_real.detach())
    mid = 0.5 * (fake_flat.detach().double().mean() + real_flat.double().mean())
    return (
        cap * ((fake_flat.double() - mid).to(fake_flat.dtype) / cap).tanh(),
        cap * ((real_flat.double() - mid).to(real_flat.dtype) / cap).tanh(),
    )


def _discrete_cond_mask(cond_a, cond_b, m: int, n: int):
    if cond_a is None or cond_b is None or cond_a.shape[0] != m or cond_b.shape[0] != n:
        return None
    u, v = cond_a.detach().reshape(m, -1), cond_b.detach().reshape(n, -1)
    return (u[:, None] == v[None]).all(-1) if u.shape[1] == v.shape[1] else None


def _valid_prior_binding(prior_table, batch_rows, batch_n: int) -> bool:
    return (
        prior_table is not None
        and batch_rows is not None
        and prior_table.ndim == 2
        and batch_rows.ndim == 1
        and batch_rows.shape[0] == batch_n
        and 1 <= prior_table.shape[0] <= MAX_LEDGER_PARTICLES
    )


def _positive_quantile(dist, mask, divisor: int):
    vals = where(mask, dist, _INF).reshape(-1).sort().values
    return vals[(mask.sum() // divisor).clamp_max(vals.numel() - 1)]


def _ema_blend(prev, cur):
    ratio = (cur / prev.clamp_min(1e-30)).maximum(prev / cur.clamp_min(1e-30))
    return prev.lerp(cur, 0.1 + 0.7 * (ratio > 8).to(cur.dtype))


def project_features(x: Tensor, max_spatial: int = 8, max_flat_dim: int = 256) -> Tensor:
    """Energy-preserving spatial/flat feature projection for transport."""
    batch_n = x.shape[0]
    if x.ndim >= 4 and (x.shape[-2] > max_spatial or x.shape[-1] > max_spatial):
        h_in, w_in = x.shape[-2], x.shape[-1]
        h_out, w_out = min(max_spatial, h_in), min(max_spatial, w_in)
        z = x if x.ndim == 4 else x.reshape(-1, x.shape[-3], h_in, w_in)
        if z.dtype == torch.float16:
            z = z.float()
        pad_h, pad_w = (-h_in) % h_out, (-w_in) % w_out
        if pad_h or pad_w:
            z = F.pad(z, (0, pad_w, 0, pad_h), "replicate")
        kh, kw = (h_in + pad_h) // h_out, (w_in + pad_w) // w_out
        z = F.avg_pool2d(z, (kh, kw), (kh, kw)).float().reshape(batch_n, -1)
        z = z * math.sqrt((h_in * w_in) / max(1, h_out * w_out))
    else:
        z = x.reshape(batch_n, -1).float()
    dim, cap = z.shape[1], max(1, max_flat_dim)
    if dim > cap:
        groups = (dim + cap - 1) // cap
        if groups * cap > dim:
            z = F.pad(z, (0, groups * cap - dim), "replicate")
        z = z.reshape(batch_n, groups, cap).mean(1) * math.sqrt(dim / cap)
    return z.to(x.dtype)


project_transport_features = project_features


@no_grad()
def compute_contrast_confidence(y_feat: Tensor, *, max_b: int = 64) -> Tensor:
    """Sync-free continuous distance-contrast confidence kappa_struct(Y)."""
    z = y_feat.reshape(y_feat.shape[0], -1).float()
    batch_n = z.shape[0]
    if batch_n < 4:
        return z.new_ones(())
    if batch_n > max_b:
        z, batch_n = z.index_select(0, (arange(max_b, device=z.device) * batch_n) // max_b), max_b
    dist = safe_cdist(z)
    sorted_all, n_off = dist.reshape(-1).sort().values, batch_n * (batch_n - 1)
    sorted_nn = dist.clone().fill_diagonal_(_INF).amin(1).sort().values
    med_all = (0.5 * (sorted_all[batch_n + (n_off - 1) // 2] + sorted_all[batch_n + n_off // 2])).clamp_min(1e-30)
    med_nn = 0.5 * (sorted_nn[(batch_n - 1) // 2] + sorted_nn[batch_n // 2])
    return where(sorted_all[-1] <= 1e-24, 1.0, smoothstep01((0.50 - med_nn / med_all) / 0.30))


@no_grad()
def compute_discrete_atomic_gate(
    y_cf: Tensor, *, charts: Tensor | None = None, radii: Tensor | None = None, record: Any = None
) -> float:
    """Scale-invariant annular-gap atomic purity gate."""
    y = y_cf.reshape(y_cf.shape[0], -1).float()
    if y.shape[0] < 8:
        return 0.0
    if charts is not None and charts.shape[0] >= 4:
        c = charts.to(y.device, _F32)
        sep = quantile(safe_cdist(c, diag_inf=True).amin(1), 0.5)
        q_rad = quantile(radii.float(), 0.75) if radii is not None else y.new_zeros(())
        if ((sep > 1e-12) & (quantile(safe_cdist(y, c).amin(1), 0.75) < 1e-3 * sep) & (q_rad < 1e-3 * sep)).item():
            return 1.0
    cs = getattr(record, "chart_state", None)
    if isinstance(cs, dict) and cs.get("initialized") and "rho_atom_ema" in cs:
        rho_raw = cs["rho_atom_ema"]
        if (rho_raw.item() if isinstance(rho_raw, Tensor) else float(rho_raw)) < 1e-3:
            return 1.0
    dist = safe_cdist(y)
    d_max = dist.amax()
    tol, nn_dist = 1e-6 * d_max, dist.clone().fill_diagonal_(_INF).amin(1)
    n_atom, sep_mask = (nn_dist <= tol).float().sum(), dist > tol
    ring_hi = (10 * tol).maximum(0.25 * _positive_quantile(dist, sep_mask, 20))
    n_ring = ((nn_dist > tol) & (nn_dist < ring_hi)).float().sum()
    gate = smoothstep01((n_atom / (n_atom + n_ring).clamp_min(1) - 0.7) / 0.2)
    return where((d_max > 1e-24) & (n_atom >= 2) & sep_mask.any(), gate, 0.0).item()


@no_grad()
def _batch_ell(y, q: float = 0.25):
    batch_n = y.shape[0]
    if batch_n <= 1:
        return y.new_ones((), dtype=_F32)
    a_exp = max(0.0, 1 / max(1, y.shape[-1]) - 0.5)
    a_fac = ((2 * math.e * batch_n) / max(1e-6, -math.log(max(1e-6, 1 - q)))) ** a_exp if a_exp > 0 else 1.0
    dist = safe_cdist(y)
    d_max = dist.amax()
    tol = 1e-5 * d_max
    pos_min = where(dist > tol, dist, _INF).amin()
    raw_q = quantile(dist.fill_diagonal_(_INF).amin(-1), q)
    val = where(raw_q <= tol, where(pos_min.isfinite(), 0.15 * pos_min, 0.05 * d_max), raw_q * a_fac)
    return where(d_max <= 0, 1e-3, val.clamp_min(tol))


@no_grad()
def _valley_radius(y, d_yy):
    far_idx = d_yy.topk(min(y.shape[0] - 1, 2), -1).indices[:, -1]
    mid_dist = safe_cdist(0.5 * (y + y[far_idx]), y)
    rows = arange(y.shape[0], device=y.device)
    mid_dist[rows, rows] = mid_dist[rows, far_idx] = _INF
    return quantile(mid_dist.amin(-1), 0.85)


@no_grad()
def _conformal_horizon(x, y, ell, g_conn: float, has_cond: bool):
    if g_conn > 0.5 or has_cond or x.shape[0] < 12 or y.shape[0] < 12:
        return ell, ell, None, None, None
    ell_f = ell.float().clamp_min(1e-30)
    floor, d_yy = 1e-6 * ell_f, safe_cdist(y)
    nn1, nn2, nn3 = d_yy.clone().fill_diagonal_(_INF).topk(3, 1, largest=False).values.clamp_min(floor).unbind(1)
    r_loc = where((nn2 <= 2.8 * nn1) & (nn3 <= 2.8 * nn2), nn2, nn1)
    k_dist, k_idx = d_yy.topk(8, 1, largest=False)
    k_mask = (k_dist <= (3.2 * nn2)[:, None]).float()
    k_mask[:, 0] = 1
    r_smooth = (r_loc[k_idx] * k_mask).sum(1) / k_mask.sum(1).clamp_min(1)
    spread_w = smoothstep01((quantile(r_smooth, 0.85) / quantile(r_smooth, 0.15).clamp_min(floor) - 2.85) / 0.85)
    if spread_w.item() <= 1e-4:
        return ell, ell, None, None, d_yy
    r_scaled = (ell_f * (r_smooth / quantile(r_smooth, 0.25).clamp_min(floor))).clamp(0.7 * ell_f, 4.5 * ell_f)
    scale_y = ell_f.lerp(r_scaled, spread_w)
    bw = (2 * (2.4 * scale_y).square()).clamp_min(1e-30)[None]
    ell_y = (softmax(-d_yy.square() / bw, 1) @ scale_y).clamp_min(floor)
    ell_x = (softmax(-safe_cdist(x, y).square() / bw, 1) @ scale_y).clamp_min(floor)
    ell_xy, ell_xx = (
        (ell_x[:, None] * ell_y[None]).sqrt().to(ell.dtype),
        (ell_x[:, None] * ell_x[None]).sqrt().to(ell.dtype),
    )
    return ell_xy, ell_xx, ell_x, ell_y, d_yy


@no_grad()
def _hetero_horizon(x, y, ell, ell_x_c, ell_y_c, d_yy_cache, w_cluster, g_conn: float, record):
    d_yy = d_yy_cache.clone() if d_yy_cache is not None else safe_cdist(y)
    d_zero, ell_b = d_yy.clone().fill_diagonal_(0), ell.float().clamp_min(1e-30)
    valley_w = smoothstep01((_valley_radius(y.float(), d_zero) / (2 * ell_b) - 2.55) / 1.5) * (1 - g_conn)
    w_sep, pos_mask = w_cluster.maximum(valley_w), d_zero > 0
    q25_dist = where(pos_mask.any(), _positive_quantile(d_zero, pos_mask, 4).clamp_min(1e-30), ell_b)
    bw = (1.15 * q25_dist * (ell_y_c / ell_b)).clamp_min(1e-30)[:, None] if ell_y_c is not None else (1.15 * q25_dist)
    density = (-d_zero.square() / (2 * bw.square()).clamp_min(1e-30)).exp().sum(1)
    ratio = (w_sep * (quantile(density, 0.85) / quantile(density, 0.03).clamp_min(1))).clamp(0, 20)
    het_val = ratio
    if hasattr(record, "het_initialized"):
        het_val = (0.92 * float(record.het_ema) + 0.08 * ratio) if record.het_initialized else ratio.clamp_max(3)
    g_het, d_inf = smoothstep01(het_val - 3.4), d_zero.clone().fill_diagonal_(_INF)
    nn1, nn2 = d_inf.amin(1), d_inf.topk(min(2, y.shape[0] - 1), 1, largest=False).values[:, -1]
    sparse_g = g_het * smoothstep01((nn2 / ell_b - 2.8) / 1.4)
    ell_y_0 = ell_y_c if ell_y_c is not None else ell_b.expand(y.shape[0])
    ell_y = ell_y_0.maximum(ell_b + sparse_g * (0.64 * nn2 - ell_b).clamp(0, 3.4 * ell_b))
    nn_dist, nn_idx = safe_cdist(x, y).min(1)
    nn_ell = ell_y[nn_idx]
    sparse_near = sparse_g[nn_idx] * smoothstep01((2.8 * nn_ell - nn_dist) / (1.2 * nn_ell).clamp_min(1e-30))
    ell_x_0 = ell_x_c if ell_x_c is not None else ell_b.expand(x.shape[0])
    ell_x, ell_y_out = ell_x_0.maximum(ell_b + sparse_near * (nn_ell - ell_b)).to(ell.dtype), ell_y.to(ell.dtype)
    return (
        (ell_x[:, None] * ell_y_out[None]).sqrt(),
        (ell_x[:, None] * ell_x[None]).sqrt(),
        ell_y_out,
        g_het,
        nn1,
        sparse_near,
        w_sep,
        het_val,
        d_yy,
    )


def _outlier_pull_loss(cost_xy, c_hat_xy, ell_y, nn1, d_fake, slack_frac, g_het, p_align, rho: float):
    with no_grad():
        ey, cf = ell_y.float().clamp_min(1e-30), c_hat_xy.float()
        row_min = cf.amin(1)
        t_gap = (2.1 * quantile(((ey.square() + nn1.square()).sqrt() - ey) / ey, 0.98)).clamp_min(4.2 * rho)
        out_w = (
            g_het
            * smoothstep01((p_align.float().sum() - 0.72) / 0.12)
            * smoothstep01((row_min - t_gap) / (0.4 * t_gap))
        )
        out_w = out_w * ((-d_fake.float() - 0.35) / 0.2).sigmoid() * slack_frac.float().pow(4)
        soft_w = softmax(-(cf - row_min[:, None]) / 0.25, 1).to(cost_xy.dtype)
    return 2.2 * ((out_w.to(cost_xy.dtype) / cost_xy.shape[0])[:, None] * soft_w * cost_xy).sum()


def _orphan_macro_loss(x, y, c_hat_xy, p_row_slack, p_col_slack, *, rho: float, g_conn: float):
    n_fake, n_real = x.shape[0], y.shape[0]
    if n_fake < 2 or n_real < 2 or g_conn > 0.05:
        return x[:0].sum()
    with no_grad():
        orphan_g = smoothstep01((c_hat_xy.float().amin(0) - 2.8 * rho) / max(1.2 * rho, 1e-6))
        d_xy, y_f = safe_cdist(x, y), y.float()
        if n_real >= 8 and n_fake >= 8:
            d_yy = safe_cdist(y_f)
            d_yy_inf, k_nn = d_yy.clone().fill_diagonal_(_INF), min(4, n_real - 1, n_fake)
            d_fk = d_xy.topk(k_nn, 0, largest=False).values[-1]
            d_rk = d_yy_inf.topk(k_nn, 1, largest=False).values[:, -1].clamp_min(1e-30)
            orphan_g = orphan_g * smoothstep01((d_fk / d_rk - 3.52) / 1.3)
            valley = _valley_radius(y_f, d_yy)
            val_w = smoothstep01((valley / quantile(d_yy_inf.amin(1), 0.5).clamp_min(1e-30) - 3.2) / 1.8)
            kde_h = (0.38 * valley).clamp_min(1e-30)
            p_y = (-0.5 * (d_yy / kde_h).square()).exp().mean(1).clamp(1e-4, 1)
            p_x = (-0.5 * (d_xy / kde_h).square()).exp().mean(0)
            g_kde = smoothstep01((n_fake * p_y - 4.2) / 2.3) * smoothstep01((0.45 - p_x / p_y) / 0.2)
            orphan_g = orphan_g * ((1 - val_w) + val_w * g_kde)
        q_mass = (n_real * p_col_slack.float()).clamp(0, 1) * orphan_g
        mean_q = q_mass.mean().item()
        k_sel = n_fake if mean_q >= 0.85 else min(n_fake - 1, int(round(n_fake * mean_q)))
        if mean_q <= 1e-4 or k_sel < 1:
            return x[:0].sum()
        med_d, active = quantile(d_xy.reshape(-1), 0.5).clamp_min(1e-30), (orphan_g > 0.05).float()
        d_orph = where((active > 0).any(), (d_xy + (1 - active)[None] * (1e6 * med_d)).amin(1), d_xy.amin(1))
        if k_sel == n_fake:
            sel_idx = arange(n_fake, device=x.device)
        else:
            tie = 1e-4 * (((arange(n_fake, device=x.device, dtype=_F32) + 0.5) * PHI1) % 1)
            score = 1.5 * (n_fake * p_row_slack.float()).clamp(0, 1) - 0.65 * (d_orph / med_d).clamp_max(4) + tie
            sel_idx = score.topk(k_sel).indices.sort().values
        q_prob = (q_mass + 1e-7) / (q_mass + 1e-7).sum().clamp_min(1e-12)
        y_cent = (y_f - (q_prob[:, None] * y_f).sum(0)) * q_prob.sqrt()[:, None]
        if y_f.shape[1] >= 2:
            order = (y_f @ power_principal_axis((y_cent.T @ y_cent).double(), iters=6, seed=0)[1].float()).argsort()
        else:
            order = y_f[:, 0].argsort()
        log_u, log_q = d_xy.new_full((k_sel,), -math.log(k_sel)), q_prob[order].clamp_min(1e-12).log()
        plan = log_sinkhorn(-(d_xy[sel_idx][:, order] / med_d) / 0.03, log_u, log_q, iters=10).exp_()
        cdf = (plan / plan.sum(1, keepdim=True).clamp_min(1e-12)).cumsum(1)
        cdf[:, -1] = 1
        j = arange(k_sel, device=x.device)
        u_quant = (((j // 2 + (j % 2) * ((k_sel + 1) // 2)).float() + 0.5) / k_sel).clamp(1e-6, 1 - 1e-6)
        y_match = y.detach()[order[searchsorted(cdf, u_quant[:, None])[:, 0].clamp(0, n_real - 1)]]
    return born_infeld_cost(x[sel_idx], y_match, (0.25 * med_d).clamp_min(1e-30), aligned=True)[0].sum() / n_fake


@no_grad()
def _batch_residuals(x, y, ell, cap_gain: float = 0.58):
    x_f, y_f = x.reshape(x.shape[0], -1).float(), y.reshape(y.shape[0], -1).float()
    n_x, n_y = x_f.shape[0], y_f.shape[0]
    d_xy, d_xx, d_yy = safe_cdist(x_f, y_f), safe_cdist(x_f), safe_cdist(y_f)
    m_xy, m_xx, m_yy = d_xy.mean(), d_xx.sum() / max(1, n_x * (n_x - 1)), d_yy.sum() / max(1, n_y * (n_y - 1))
    denom = (0.5 * (m_xy + 0.5 * (m_xx + m_yy))).clamp_min(1e-30)
    r_energy, r_spread, n_min = (2 * m_xy - m_xx - m_yy) / denom, (m_xx - m_yy) / denom, min(n_x, n_y)
    var_x, var_y = (x_f.var(0).sum() / n_x if n_x > 1 else 0.0), (y_f.var(0).sum() / n_y if n_y > 1 else 0.0)
    mean_sq = ((x_f.mean(0) - y_f.mean(0)).square().sum() - var_x - var_y) / denom.square()
    if n_min < 4:
        return r_energy, 0.16 * mean_sq, r_spread, r_energy, r_energy
    d_xy_off = d_xy[:n_min, :n_min].clone().fill_diagonal_(_INF)
    y_nn = d_yy[:n_min, :n_min].clone().fill_diagonal_(_INF).amin(1)
    ell_ratio = ell.float().clamp_min(1e-30) / denom
    r_base = r_mean = (0.15 + 5.5 * ell_ratio).clamp(0.15, 0.4).square() * mean_sq
    if cap_gain != 0.4 and n_y >= 4:
        nn_mean = d_yy.clone().fill_diagonal_(_INF).amin(1).mean()
        mid_dist = safe_cdist(0.5 * (y_f + y_f.roll(1, 0)), y_f)
        rows = arange(n_y, device=y_f.device)
        mid_dist[rows, rows] = mid_dist[rows, (rows + 1) % n_y] = _INF
        w_conn = where(
            nn_mean > 1e-30, smoothstep01((1.15 - mid_dist.amin(1).mean() / nn_mean.clamp_min(1e-30)) / 0.2), 0.0
        )
        r_conn = (0.15 + 5.5 * ell_ratio).clamp(0.15, 0.4 + w_conn * (cap_gain - 0.4)).square() * mean_sq
        r_mean = where(nn_mean > 1e-30, where(mean_sq >= 0, r_base.maximum(r_conn.clamp_max(0.045)), r_conn), r_base)
    q_fg = 10.2 * (quantile(d_xy_off.amin(1), 0.75) - quantile(y_nn, 0.75)) / denom
    q_cov = (quantile(d_xy_off.amin(0), 0.9) - quantile(y_nn, 0.9)) / denom
    return r_energy, r_mean, r_spread, q_fg, q_cov


def _split_joint_prior_batch(x_fake, y_real, prior_table, batch_rows):
    n_fake, n_real = x_fake.shape[0], y_real.shape[0]
    if not _valid_prior_binding(prior_table, batch_rows, n_fake) or n_fake < 4 or n_real < 4:
        return None
    z_dim, xf, yf = prior_table.shape[1], x_fake.reshape(n_fake, -1), y_real.reshape(n_real, -1)
    if xf.shape[1] <= z_dim or yf.shape[1] != xf.shape[1]:
        return None
    rows_l = batch_rows.to(prior_table.device, _LONG)
    z_ref = prior_table.detach()[rows_l.clamp(0, prior_table.shape[0] - 1)].to(xf.device, _F32)
    oob = ((rows_l < 0).any() | (rows_l >= prior_table.shape[0]).any()).float().to(xf.device)
    tail_err = (xf[:, -z_dim:].detach().float() - z_ref).abs().amax()
    head_err = (xf[:, :z_dim].detach().float() - z_ref).abs().amax()
    oob_v, tail_v, head_v = stack((oob, tail_err, head_err)).tolist()
    if oob_v > 0.5:
        return None
    if tail_v <= 1e-5:
        return xf[:, :-z_dim], xf[:, -z_dim:], yf[:, :-z_dim], yf[:, -z_dim:]
    return (xf[:, z_dim:], xf[:, :z_dim], yf[:, z_dim:], yf[:, :z_dim]) if head_v <= 1e-5 else None


def _halo_born_infeld(x, target, r_halo, ell, weights=None):
    ell_c = ell.detach().float().clamp_min(1e-30)
    sq_d = (x.float() - target.detach().float()).square().sum(-1)
    excess = F.relu((sq_d + (1e-12 * ell_c).square().clamp_min(1e-36)).sqrt() - r_halo.detach().float())
    per_row = excess.square() / ((ell_c.square() + excess.square()).sqrt() + ell_c).clamp_min(1e-30)
    if weights is not None:
        per_row = weights.detach().float() * per_row
    return per_row.mean().to(x.dtype)


@no_grad()
def _gpu_quota_assign(scores, quotas):
    n_rows, n_cols, dev = scores.shape[0], scores.shape[1], scores.device
    weyl = ((arange(n_rows, device=dev, dtype=_F32) + 0.5) * PHI1).remainder(1.0)
    win, rem = full((n_rows,), -1, dtype=_LONG, device=dev), quotas.clone()
    unassigned = ones(n_rows, dtype=_BOOL, device=dev)
    for _ in range(4):
        best_s, best_c = scores.masked_fill((rem <= 0)[None], -_INF).max(1)
        group_key = where(unassigned & best_s.isfinite(), best_c, n_cols)
        order, sorted_g, rank = _group_rank(group_key, -best_s.clamp(-1e6, 1e6), weyl, n_cols)
        keep = (sorted_g < n_cols) & (rank < rem[sorted_g.clamp_max(n_cols - 1)])
        accept = zeros_like(unassigned).scatter_(0, order, keep)
        win, unassigned = where(accept, best_c, win), unassigned & (~accept)
        rem = (rem - bincount(where(accept, best_c, n_cols), minlength=n_cols + 1)[:n_cols]).clamp_min(0)
    comp_order = (-scores.masked_fill(~unassigned[:, None], -_INF).amax(0)).clamp(-1e6, 1e6).argsort(stable=True)
    slot_ends, ranks = rem[comp_order].cumsum(0), arange(n_rows, device=dev, dtype=_LONG)
    slot_comp = comp_order[searchsorted(slot_ends, ranks, right=True).clamp_max(n_cols - 1)]
    rem_s = scores.masked_fill((rem <= 0)[None], -_INF).amax(1)
    row_order = where(unassigned, -rem_s.clamp(-1e6, 1e6) + 1e-6 * weyl, 1e12).argsort(stable=True)
    fb = full((n_rows,), -1, dtype=_LONG, device=dev).scatter_(
        0, row_order, where(unassigned[row_order] & (ranks < slot_ends[-1]), slot_comp, -1)
    )
    return where(unassigned & (fb >= 0), fb.clamp_min(0), win)


@no_grad()
def _continuous_quota_route(
    x_proj, atlas, coal, r_halo, nn_dist, batch_hits, ell_c, w_cluster, halo_mult, s_th, mutate
):
    bf, k_c = x_proj.shape[0], coal.comp_c.shape[0]
    wr = coal.comp_w.float()
    rad_k = (coal.comp_c.shape[1] * cov_trace_mean(coal.comp_cov)).clamp_min(0).sqrt()
    r_c = r_halo.clamp_min(0.02 * halo_mult * nn_dist).minimum(0.35 * nn_dist)
    r_bas = (1.35 * r_c).maximum(2.5 * rad_k)
    near_d, near_i = safe_cdist(x_proj.detach(), coal.comp_c).min(1)
    cur_fm = F.one_hot(coal.proto_comp_id, k_c).float().T @ atlas.w_fake_ema.float()
    bf_frac = (F.one_hot(near_i, k_c).float() * (near_d <= r_bas[near_i]).float()[:, None]).sum(0) / max(1, bf)
    al_f = min(0.35, max(0.025, 0.08 * ((bf / max(1, k_c)) ** 0.45)))
    next_fm = ((1.0 - al_f) * cur_fm + al_f * bf_frac) if mutate else cur_fm
    if mutate:
        wf_raw = next_fm[coal.proto_comp_id] * (atlas.proto_w.float() / wr[coal.proto_comp_id].clamp_min(1e-12))
        wf_sum = wf_raw.sum()
        atlas.w_fake_ema.copy_(where(wf_sum > 1e-12, wf_raw / wf_sum.clamp_min(1e-12), atlas.proto_w.float()))
    se_k = ((2.0 * (al_f / max(1e-6, 2.0 - al_f)) / max(1, bf)) * wr).clamp_min(1e-8).sqrt()
    def_k, sur_k = F.relu(wr - next_fm - 1.60 * se_k), F.relu(next_fm - wr - 1.60 * se_k)
    ab_sh = (
        w_cluster
        * smoothstep01((r_c[near_i] - near_d) / (0.35 * r_c[near_i]).clamp_min(1e-30))
        * (batch_hits[near_i] <= 0).float()
        * (sur_k[near_i] <= 1e-6).float()
        if k_c > 4
        else x_proj.new_zeros(bf, dtype=_F32)
    )
    if k_c < 3:
        return None, None, None, ab_sh
    wt = wr + 2.0 * def_k
    wt = wt / wt.sum().clamp_min(1e-8)
    bi = born_infeld_cost(x_proj.detach(), coal.comp_c, ell_c)[0].float()
    logits = -(bi / bi.median().clamp_min(1e-30)).clamp_max(200.0) / 0.08
    scores = log_sinkhorn(logits, bi.new_full((bf,), -math.log(max(1, bf))), wt.clamp_min(1e-12).log(), iters=8)
    quotas = hamilton_quotas(bf, wt, min_one_when_feasible=False)
    win, sth_g = _gpu_quota_assign(scores, quotas), smoothstep01((s_th - 0.02) / 0.18)
    reb_mask = (win != near_i) & (def_k[win] > 0) & (sur_k[near_i] > 0) & (def_k.sum() > 1e-4) & (w_cluster > 1e-4)
    g_reb = (
        where(reb_mask, (w_cluster * sth_g) * (def_k[win] / wr[win].clamp_min(1e-6)).clamp(0, 1), 0.0)
        if sth_g > 1e-4
        else x_proj.new_zeros(bf, dtype=_F32)
    )
    tgt_i = where((g_reb > 0) | ((sth_g > 0.5) & (quotas[near_i] >= 1)), win, near_i)
    route_w = where(w_cluster > 1e-4, 2.0 * (w_cluster * (near_d > r_c[near_i]).float()).maximum(g_reb), 0.0)
    return coal.comp_c[tgt_i], r_c[tgt_i], route_w, ab_sh


@no_grad()
def _curve_gate(y, g_conn: float) -> float:
    if g_conn <= 1e-4 or y.shape[0] < 12:
        return 0.0
    if y.shape[1] < 2:
        return g_conn
    nb_idx = safe_cdist(y).topk(min(9, y.shape[0]), -1, largest=False).indices
    cent = y.float()[nb_idx].double()
    cent = cent - cent.mean(1, keepdim=True)
    gram = (cent.mT @ cent) if cent.shape[2] < cent.shape[1] else (cent @ cent.mT)
    lam1, _ = power_principal_axis(gram, iters=6, seed=0)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
    resid = ((trace - lam1).clamp_min(0) / lam1.clamp_min(1e-30)).median()
    return g_conn * smoothstep01((0.11 - resid) / 0.06).item()


@no_grad()
def _simplicial_gates(y, chart_weight: float):
    if chart_weight <= 0 or y.shape[0] < 4:
        return 0.0, 0.0, 0.1, y.float().abs().amax().clamp_min(1e-12), 0.0
    g_raw, m_raw, q90_t, y_scale = manifold_connectivity_stats(y)
    ann = smoothstep01((batch_annular_gap_ratio(y) - 0.15) / 0.2)
    g_conn, g_man, q90 = stack((g_raw * ann, m_raw * ann, q90_t)).tolist()
    return g_conn, g_man, q90, y_scale, _curve_gate(y, g_conn)


@no_grad()
def _barycentric_samples(z_table, phase: float, n_query: int, g_curve: float):
    n_pts, dev, two_pi = z_table.shape[0], z_table.device, 2 * math.pi
    d_z = (safe_cdist(z_table.double()).double() + 1e-12 * arange(n_pts, device=dev, dtype=_F64)[None]).fill_diagonal_(
        _INF
    )
    k_nn = min(4, n_pts - 1)
    nb = d_z.topk(k_nn, -1, largest=False).indices
    base_edges = (n_query // 2) if (n_pts >= 3 and k_nn >= 2 and n_query >= 4) else n_query
    n_tri = int(round((1 - g_curve) * (n_query - base_edges)))
    n_edge, r_edge, k_edge = n_query - n_tri, arange(n_query - n_tri, device=dev), min(3, k_nn)
    if g_curve > 0.5:
        denom = max(1, n_pts * k_edge)
        pair_idx = (r_edge + int(phase * denom)) % denom
        v_a, v_b = pair_idx // k_edge, nb[pair_idx // k_edge, pair_idx % k_edge]
        u_edge = (((r_edge // denom).float() + 0.5) / max(1, (n_edge + denom - 1) // denom) + phase) % 1
    else:
        v_a = (r_edge + int(phase * n_pts)) % n_pts
        v_b, u_edge = nb[v_a, r_edge % k_edge], ((r_edge.float() + 0.5) / max(1, n_edge) + phase) % 1
    t_edge = (0.2 + 0.6 * u_edge).to(z_table.dtype)
    z_edge = z_table[v_a].lerp(z_table[v_b], t_edge[:, None])
    if n_tri <= 0:
        return z_edge, n_edge, v_a, v_b, t_edge, None
    r_tri, ph_tri = arange(n_tri, device=dev), (phase * PHI2) % 1
    v0, col1 = (r_tri + int(ph_tri * n_pts)) % n_pts, r_tri % k_nn
    v1, v2 = nb[v0, col1], nb[v0, (col1 + 1 + ((r_tri // k_nn) % max(1, k_nn - 1))) % k_nn]
    angle, rad_w = two_pi * (((r_tri.float() + 0.5) / max(1, n_tri) + ph_tri) % 1), 0.12 * (r_tri % 2).float()
    w0 = (1 / 3 + rad_w * angle.cos()).to(z_table.dtype)[:, None]
    w1 = (1 / 3 + rad_w * (angle + two_pi / 3).cos()).to(z_table.dtype)[:, None]
    z_tri = w0 * z_table[v0] + w1 * z_table[v1] + (1 - w0 - w1) * z_table[v2]
    return cat((z_edge, z_tri)), n_edge, v_a, v_b, t_edge, (v0, v1, v2, z_tri)


def _simplicial_isometry_loss(x_samples, x_verts, z_det, ell, g_man: float, samples, use_f64: bool):
    _, n_edge, v_a, v_b, t_edge, tri_info = samples
    d0 = born_infeld_cost(x_samples[:n_edge], x_verts[v_a], ell, aligned=True, fp64=use_f64)[0]
    d1 = born_infeld_cost(x_samples[:n_edge], x_verts[v_b], ell, aligned=True, fp64=use_f64)[0]
    t_cast = t_edge.to(x_samples.dtype)
    iso = ((1 - t_cast) * d0 - t_cast * d1).abs().mean()
    if tri_info is None:
        return iso
    v0, v1, v2, z_tri = tri_info
    verts = (v0, v1, v2)
    tri_c = stack([born_infeld_cost(x_samples[n_edge:], x_verts[v], ell, aligned=True, fp64=use_f64)[0] for v in verts])
    with no_grad():
        p0, p1, p2 = x_verts[v0].double(), x_verts[v1].double(), x_verts[v2].double()
        edges = stack(((p0 - p1).norm(dim=-1), (p1 - p2).norm(dim=-1), (p2 - p0).norm(dim=-1)))
        semi_p = 0.5 * edges.sum(0)
        area = (semi_p * (semi_p - edges).clamp_min(0).prod(0)).clamp_min(0).sqrt()
        ell_d = ell.clamp_min(1e-30).double()
        min_rad = 0.7 * ((ell_d.square() + (area / semi_p.clamp_min(1e-8)).square()).sqrt() - ell_d)
        if g_man > 1e-4:
            q0, q1, q2, qz = z_det[v0].double(), z_det[v1].double(), z_det[v2].double(), z_tri.double()
            z_e = stack(((q0 - q1).norm(dim=-1), (q1 - q2).norm(dim=-1), (q2 - q0).norm(dim=-1)))
            z_min = stack(((qz - q0).norm(dim=-1), (qz - q1).norm(dim=-1), (qz - q2).norm(dim=-1))).amin(0)
            min_rad = (1 - g_man + g_man * smoothstep01((z_min / z_e.amax(0).clamp_min(1e-8) - 0.12) / 0.1)) * min_rad
    return 0.5 * (iso + F.relu(min_rad.to(tri_c.dtype) - tri_c).mean())


def _simplicial_chart_loss(
    z_table, x_live, x_verts, y_flat, ell, g_conn, g_curve, q90, y_scale, phase, chart_w, use_f64
):
    n_pts, n_real = z_table.shape[0], y_flat.shape[0]
    y_unit, loss = y_flat.float() / y_scale, ell.new_zeros((), dtype=x_verts.dtype)
    if x_live is not None:
        cost_xy, _ = born_infeld_cost(x_live, y_flat, ell, fp64=use_f64)
        with no_grad():
            logits = -(cost_xy.float() / cost_xy.float().mean().clamp_min(1e-12)) / 0.08
            log_m, log_n = logits.new_full((n_pts,), -math.log(n_pts)), logits.new_full((n_real,), -math.log(n_real))
            plan = log_sinkhorn(logits, log_m, log_n, iters=10).exp_().to(cost_xy.dtype)
        loss = loss + (1.1 * g_conn) * (plan * cost_xy).sum()
        if g_curve > 1e-4:
            tgt = y_flat[(plan.float() @ safe_cdist(y_unit)).argmin(1)]
            loss = (
                loss
                + (0.65 * g_conn * g_curve)
                * born_infeld_cost(x_live, tgt, 0.5 * ell, aligned=True, fp64=use_f64)[0].mean()
            )
    if not z_table.requires_grad:
        return loss
    with no_grad():
        if (n_pts + n_real) <= 512:
            y_chart = y_unit
        else:
            n_sub = min(n_real, 128)
            y_chart = y_unit[((arange(n_sub, device=y_unit.device) * n_real) // n_sub + int(phase * n_real)) % n_real]
        verts_all = cat((x_verts.float() / y_scale, y_chart))
        d_all = safe_cdist(verts_all.double() if use_f64 else verts_all).double()
        geo = d_all * (1 + (d_all / max(1e-4, 1.35 * q90)).square())
        for _ in range(6):
            geo = geo.minimum((geo[:, :, None] + geo[None]).amin(1))
        geo_p, off = geo[:n_pts, :n_pts], ~eye(n_pts, dtype=_BOOL, device=z_table.device)
        geo_hat = geo_p / geo_p[off].square().mean().clamp_min(1e-12).sqrt()
        w_loc = where(off, (0.2 + geo_hat.square()).pow(-2), 0.0)
        w_loc = w_loc / w_loc.sum().clamp_min(1e-12)
        w_glb = where(off, 1 + 2 * geo_hat.square(), 0.0)
        w_glb = w_glb / w_glb.sum().clamp_min(1e-12)
        w_mds = w_loc if g_curve <= 1e-4 else w_loc.lerp(w_glb, g_curve)
    z_d = z_table.double()
    z_sq = z_d.square().sum(-1)
    d_z_sq = z_sq[:, None] + z_sq[None] - 2 * (z_d @ z_d.T)
    d_z_sq = d_z_sq + (d_z_sq.detach().clamp_min(0) - d_z_sq.detach())
    d_z_hat = (d_z_sq + 1e-12).sqrt() / d_z_sq[off].mean().clamp_min(1e-12).sqrt()
    ell_det = ell.detach().to(loss.dtype)
    mds_term = (w_mds * (d_z_hat - geo_hat).square()).sum().to(loss.dtype)
    loss = loss + (g_conn * chart_w * (1 + 18 * g_curve)) * ell_det * mds_term
    if g_curve > 1e-4 and z_table.shape[1] >= 2:
        z_cent = z_d - z_d.mean(0)
        frob_sq = z_cent.square().sum().clamp_min(1e-12)
        with no_grad():
            _, axis = power_principal_axis(z_cent.T @ z_cent, iters=6, seed=0)
        orth_frac = ((frob_sq - (z_cent @ axis).square().sum()).clamp_min(0) / frob_sq).to(loss.dtype)
        loss = loss + (3.2 * g_conn * g_curve * chart_w) * ell_det * orth_frac
    return loss


def _observe_transport_thermal(x_proj, y_proj, ell, aux, disc: bool, simp: bool, has_cond: bool, record) -> None:
    w_sep, het_val, d_yy, r_tail, has_ledger = aux
    sim_on = simp or bool(getattr(record, "_simplicial_step", False))
    min_b, has_het = min(x_proj.shape[0], y_proj.shape[0]), het_val is not None and hasattr(record, "het_initialized")
    if not sim_on and not has_cond and min_b >= 4 and hasattr(record, "observe_thermal"):
        with no_grad():
            r5 = _batch_residuals(x_proj, y_proj, ell, cap_gain=0.58)
            if not disc and not has_ledger and min_b >= 8 and d_yy is not None:
                off = ~eye(y_proj.shape[0], dtype=_BOOL, device=y_proj.device)
                q_x = quantile(safe_cdist(x_proj, y_proj).amin(1), 0.96)
                q_y = quantile(d_yy.masked_fill(~off, _INF).amin(1), 0.96)
                r_tail = r_tail.maximum(w_sep * 1.2 * (q_x - q_y) / d_yy[off].mean().clamp_min(1e-30))
            packed = stack([*r5, r_tail, het_val] if has_het else [*r5, r_tail]).tolist()
            if has_het:
                record.het_ema, record.het_initialized = packed[6], True
            record.observe_thermal(*packed[:5], r_tail=packed[5], shock_mu_ratio=0.35)
    elif has_het:
        record.het_ema, record.het_initialized = het_val.item(), True


def _observe_simplicial_thermal(x_verts, x_samples, y_flat, ell, g_man: float, g_curve: float, record) -> None:
    if g_man <= 1e-4 or y_flat.shape[0] < 4 or not _can_mutate_state(record) or not hasattr(record, "observe_thermal"):
        return
    with no_grad():
        res = stack(_batch_residuals(cat((x_verts, x_samples.detach())), y_flat, ell)).tolist()
        kw = dict(
            tau=0.084 * (1 + 0.6 * g_curve), shock_mu_ratio=0.35 + 0.55 * g_curve, shock_ed_floor=0.3 + 0.5 * g_curve
        )
        record.observe_thermal(*res[:3], (1 - 0.5 * g_curve) * res[3], res[4], lock_hysteresis=0.22 * g_curve, **kw)


class AxiomTransportLoss(nn.Module):
    """Unified Law 3A + Law 3B transport operator."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        unknown = set(kwargs) - set(_TRANSPORT_DEFAULTS)
        if unknown:
            raise TypeError(f"unexpected kwargs: {sorted(unknown)}")
        for name, default in _TRANSPORT_DEFAULTS.items():
            raw = kwargs.get(name, default)
            val = (
                max(2, int(raw)) if name == "chart_capacity" else (int(raw) if name in _INT_CONFIG_KEYS else float(raw))
            )
            setattr(self, name, val)
        self._discrete_step = self._simplicial_step = False
        self._ell_local: Tensor | None = None
        self.last_p_xy_slack: Tensor | None = None
        self._atlas_local: PrototypeAtlas | None = None
        self._rec_atlases: Any = weakref.WeakKeyDictionary()
        self._contrast_cache: tuple[Any, ...] | None = None

    def bind_discrete_prior_step(self, active: bool, *, simplicial_active: bool = False) -> None:
        self._discrete_step, self._simplicial_step = bool(active), bool(simplicial_active)

    @no_grad()
    def _cached_contrast(self, y_raw: Tensor) -> tuple[Tensor, Tensor]:
        key = (
            y_raw.untyped_storage().data_ptr(),
            y_raw.storage_offset(),
            y_raw.shape,
            y_raw.stride(),
            y_raw.dtype,
            y_raw.device,
            y_raw._version,
            self.max_spatial,
            self.max_flat_dim,
        )
        cached = self._contrast_cache
        if cached is not None:
            prev = cached[0]()
            if prev is not None and cached[1] == key and prev._version == key[6]:
                return cached[2], cached[3]
        y_proj = project_features(y_raw.detach(), self.max_spatial, self.max_flat_dim)
        kappa = compute_contrast_confidence(y_proj)
        self._contrast_cache = (weakref.ref(y_raw), key, y_proj, kappa)
        return y_proj, kappa

    def _sinkhorn_slack(self, cost: Tensor, df: Tensor, dr: Tensor, mask: Tensor | None) -> Tensor:
        return sinkhorn_with_slack(cost, df, dr, rho=self.rho, tau=self.tau, eps=self.eps, iters=self.iters, mask=mask)

    def _update_ell(self, y_proj, record, mutate: bool):
        with no_grad():
            ell_b = _batch_ell(y_proj, q=0.25)
            if hasattr(record, "ell_initialized"):
                has_t = hasattr(record, "dev_ema") and hasattr(record, "_ema_t")
                prev = (
                    record.dev_ema("ell_ema", ell_b.device).float()
                    if has_t
                    else ell_b.new_full((), float(record.ell_ema))
                )
                if not mutate:
                    return (prev if record.ell_initialized else ell_b).to(y_proj.dtype)
                cur = _ema_blend(prev, ell_b) if record.ell_initialized else ell_b
                record.ell_initialized = True
                if has_t:
                    record._ema_t["ell_ema"] = cur.double()
                else:
                    record.ell_ema = cur.item()
                return cur.to(y_proj.dtype)
            if mutate:
                self._ell_local = (
                    ell_b.clone() if self._ell_local is None else _ema_blend(self._ell_local.to(y_proj.device), ell_b)
                )
            return (self._ell_local.to(y_proj.device) if self._ell_local is not None else ell_b).to(y_proj.dtype)

    def _resolve_atlas(self, y_proj: Tensor, record: Any, mutate: bool):
        dev = y_proj.device
        if record is None:
            atlas = self._atlas_local = self._atlas_local or PrototypeAtlas(
                max_prototypes=self.chart_capacity, device=dev
            )
        else:
            cached, saved = self._rec_atlases.get(record), getattr(record, "chart_state", None)
            if cached and cached[1] is saved:
                atlas = cached[0]
            else:
                ok = isinstance(saved, dict) and saved.get("schema") == 2 and "proto_c" in saved
                max_p = int(saved.get("max_prototypes", self.chart_capacity)) if ok else self.chart_capacity
                atlas = PrototypeAtlas(max_prototypes=max_p, device=dev)
                if ok:
                    scalars = {
                        k: (saved[k].item() if isinstance(saved[k], Tensor) else float(saved[k]))
                        for k in ("rho_atom_ema", "g_conn_ema")
                        if k in saved
                    }
                    atlas.load_state_dict({**saved, **scalars}, device=dev)
                self._rec_atlases[record] = (atlas, saved)
        if mutate or not atlas.initialized:
            atlas.update(y_proj)
        return atlas, atlas.coalesce(y_proj if mutate else None)

    def _commit_atlas(self, x_proj, y_proj, prior_table, batch_rows, min_k: int, record: Any, mutate: bool):
        atlas, coal = self._resolve_atlas(y_proj, record, mutate)
        r_halo = nn_dist = assign = None
        k_c, dim = coal.comp_c.shape
        if k_c >= min_k:
            nn_dist = (
                safe_cdist(coal.comp_c, diag_inf=True).amin(1).clamp_min(1e-30)
                if k_c >= 2
                else coal.comp_c.new_ones(k_c)
            )
            r_halo = (self.halo_mult * (dim * cov_trace_mean(coal.comp_cov)).clamp_min(0).sqrt()).minimum(
                0.35 * nn_dist
            )
            if _valid_prior_binding(prior_table, batch_rows, x_proj.shape[0]):
                assign = atlas.commit_particles(
                    x_proj, batch_rows, prior_table.shape[0], coalesced=coal, allow_update=mutate
                )
        if mutate and hasattr(record, "chart_state"):
            led = atlas.ledger
            led_d = (
                {"schema": 2, "initialized": True, **{k: getattr(led, k) for k in _LEDGER_EXPORT_KEYS}}
                if led.initialized
                else None
            )
            rho_v = atlas._rho_atom_t if atlas._rho_atom_t is not None else float(atlas._rho_atom_f)
            conn_v = atlas._g_conn_t if atlas._g_conn_t is not None else float(atlas._g_conn_f)
            state = {
                "schema": 2,
                **{k: getattr(atlas, k) for k in _ATLAS_EXPORT_KEYS},
                "rho_atom_ema": rho_v,
                "g_conn_ema": conn_v,
                "ledger": led_d,
            }
            record.chart_state = state
            if hasattr(record, "atomic_ledger"):
                record.atomic_ledger = led_d
            self._rec_atlases[record] = (atlas, state)
        return atlas, coal, assign, r_halo, nn_dist

    def _route_charts(self, x_proj, y_proj, x_fake, ell, g_conn: float, record, mutate: bool):
        zero = ell.new_zeros((), dtype=_F32)
        prior_table, batch_rows = recover_active_prior_rows(x_fake, record)
        atlas, coal, assign, r_halo, nn_dist = self._commit_atlas(
            x_proj, y_proj, prior_table, batch_rows, 2, record, mutate
        )
        if r_halo is None:
            return None, None, zero, zero, None
        with no_grad():
            ell_c = ell.float().clamp_min(1e-30)
            d_yc_min, d_yc_arg = safe_cdist(y_proj, coal.comp_c).min(1)
            q_mid = quantile(safe_cdist(0.5 * (y_proj + y_proj.roll(1, 0)), coal.comp_c).amin(1), 0.75)
            q_real = quantile(d_yc_min, 0.75).maximum(ell_c)
            w_cluster = smoothstep01((q_mid / q_real - 0.85 * self.rho) / 1.5) * (1 - g_conn)
            batch_hits = bincount(d_yc_arg, minlength=coal.comp_c.shape[0])
        if assign is not None:
            comp_i = assign.batch_comp
            rho_t = atlas._rho_atom_t if atlas._rho_atom_t is not None else zero.new_full((), atlas._rho_atom_f)
            is_atom = smoothstep01((0.005 - rho_t) / 0.004)
            ell_r = where(is_atom > 0.5, (0.15 * nn_dist.median()).clamp_min(1e-30), ell_c)
            route_loss = _halo_born_infeld(x_proj, assign.batch_target, r_halo[comp_i], ell_r)
            with no_grad():
                r_eff = r_halo[comp_i].clamp_min(0.05 * nn_dist[comp_i])
                dist_t = (x_proj.float() - assign.batch_target).norm(dim=-1)
                g_rad = smoothstep01((r_eff - dist_t) / (0.35 * r_eff))
                absent = (batch_hits[comp_i] <= 0).float() * (is_atom + (1.0 - is_atom) * g_rad)
            return route_loss, absent, is_atom * assign.tail_residual, w_cluster, assign
        if (1 - g_conn) <= 1e-4:
            return None, None, zero, w_cluster, None
        s_th = float(getattr(record, "thermal_scale", 1.0)) if record is not None else 1.0
        tgt_c, tgt_r, route_w, ab_sh = _continuous_quota_route(
            x_proj, atlas, coal, r_halo, nn_dist, batch_hits, ell_c, w_cluster, self.halo_mult, s_th, mutate
        )
        if tgt_c is None:
            return None, ab_sh, zero, w_cluster, None
        return _halo_born_infeld(x_proj, tgt_c, tgt_r, ell, weights=route_w), ab_sh, zero, w_cluster, None

    @no_grad()
    def _solve_plans(self, c_hat_xy, c_hat_xx, df, dr, cond_fake, cond_real, disc: bool, sparse_near, absent):
        n_f, n_r = c_hat_xy.shape
        plan_xy = self._sinkhorn_slack(c_hat_xy, df, dr, _discrete_cond_mask(cond_fake, cond_real, n_f, n_r))
        mask_xx = _discrete_cond_mask(cond_fake, cond_fake, n_f, n_f)
        if disc and n_f > 1:
            unique_x = (c_hat_xx > 1e-12) | eye(n_f, dtype=_BOOL, device=c_hat_xx.device)
            mask_xx = unique_x if mask_xx is None else (mask_xx & unique_x)
        zeros = df.new_zeros(n_f)
        plan_xx = self._sinkhorn_slack(c_hat_xx, zeros, zeros, mask_xx)[:n_f, :n_f].fill_diagonal_(0)
        p_align, p_row_slack, p_col_slack = plan_xy[:n_f, :n_r], plan_xy[:n_f, n_r], plan_xy[n_f, :n_r]
        self.last_p_xy_slack = p_row_slack.detach()
        align_frac = (n_f * p_align.sum(1)).clamp(0, 1)
        if sparse_near is not None:
            align_frac = (align_frac + 0.5 * sparse_near.to(align_frac.dtype)).clamp(0, 1)
        slack_frac = (n_f * p_row_slack).clamp(0, 1)
        abs_fac = (1 - absent.to(p_row_slack.dtype)) if absent is not None else 1
        return (
            p_align,
            p_row_slack,
            p_col_slack,
            plan_xx,
            align_frac,
            slack_frac,
            p_row_slack * (1 + 0.35 * slack_frac) * abs_fac,
        )

    def _assemble_transport(
        self, x_proj, y_proj, x_fake, df, dr, ell, g_conn, cond_fake, cond_real, disc, mutate, record
    ):
        n_f, n_r, has_cond = x_proj.shape[0], y_proj.shape[0], (cond_fake is not None or cond_real is not None)
        zero = ell.new_zeros((), dtype=_F32)
        ell_xy, ell_xx, ell_x_c, ell_y_c, d_yy = _conformal_horizon(x_proj, y_proj, ell, g_conn, has_cond)
        if not has_cond and self.chart_weight > 0 and n_r >= 4 and n_f >= 2:
            route_loss, absent, r_tail, w_cluster, assign = self._route_charts(
                x_proj, y_proj, x_fake, ell, g_conn, record, mutate
            )
        else:
            route_loss, absent, r_tail, w_cluster, assign = None, None, zero, zero, None
        if not has_cond and assign is None and n_f >= 8 and n_r >= 8:
            ell_xy, ell_xx, ell_y, g_het, nn1, sparse_near, w_sep, het_val, d_yy = _hetero_horizon(
                x_proj, y_proj, ell, ell_x_c, ell_y_c, d_yy, w_cluster, g_conn, record
            )
        else:
            ell_y = ell_y_c.to(ell.dtype) if ell_y_c is not None else ell
            g_het, nn1, sparse_near, w_sep, het_val = zero, None, None, w_cluster, None
        (cost_xy, c_hat_xy), (cost_xx, c_hat_xx) = (
            born_infeld_cost(x_proj, y_proj, ell_xy),
            born_infeld_cost(x_proj, x_proj, ell_xx),
        )
        p_align, p_r_slack, p_c_slack, plan_xx, align_f, slack_f, p_slack_w = self._solve_plans(
            c_hat_xy, c_hat_xx, df, dr, cond_fake, cond_real, disc, sparse_near, absent
        )
        coord = (p_align * cost_xy).sum() - 0.5 * (plan_xx * (align_f[:, None] * align_f[None]).sqrt() * cost_xx).sum()
        if route_loss is not None:
            coord = coord + self.chart_weight * route_loss.to(coord.dtype)
        if nn1 is not None:
            coord = coord + _outlier_pull_loss(cost_xy, c_hat_xy, ell_y, nn1, df, slack_f, g_het, p_align, self.rho)
        if not has_cond:
            coord = coord + _orphan_macro_loss(
                x_proj, y_proj, c_hat_xy, p_r_slack, p_c_slack, rho=self.rho, g_conn=g_conn
            )
        return coord, p_align, p_slack_w, (w_sep, het_val, d_yy, r_tail, assign is not None)

    def forward(
        self,
        x_fake: Tensor,
        y_real: Tensor,
        d_fake: Tensor,
        d_real: Tensor,
        *,
        record: Any = None,
        cond_fake: Tensor | None = None,
        cond_real: Tensor | None = None,
        allow_state_update: bool = True,
        thermally_scaled: bool = False,
    ) -> Tensor:
        """Compute Law 3A Born-Infeld OT and critic transport generator loss."""
        if self.weight == 0:
            return x_fake[:0].sum()
        disc, simp = self._discrete_step, self._simplicial_step
        self._discrete_step = self._simplicial_step = False
        n_f, n_r, mutate = x_fake.shape[0], y_real.shape[0], _can_mutate_state(record, allow_state_update)
        has_cond = cond_fake is not None or cond_real is not None
        y_proj, kappa_t = self._cached_contrast(y_real)
        df, dr = _compute_log_odds(d_fake, d_real)
        kappa = kappa_t.item()
        rel_diff = (dr - df) if n_f == n_r else (dr[None] - df[:, None])
        c_rel = 0.5 * self.tau * (F.softplus(rel_diff).mean() - math.log(2))
        if kappa <= 1e-4 and getattr(record, "atomic_ledger", None) is None:
            if hasattr(record, "_disc_gate"):
                record._disc_gate = 0.0
            return self.weight * c_rel
        x_proj = project_features(x_fake, self.max_spatial, self.max_flat_dim)
        ell, g_conn = self._update_ell(y_proj, record, mutate), manifold_connectivity_stats(y_proj)[0].item()
        if hasattr(record, "_disc_gate"):
            record._disc_gate = 1 - g_conn
        coord, p_align, p_slack_w, aux = self._assemble_transport(
            x_proj, y_proj, x_fake, df, dr, ell, g_conn, cond_fake, cond_real, disc, mutate, record
        )
        if aux[4]:
            kappa = 1.0
        x_f, y_f = x_proj.detach().float(), y_proj.float()
        rad_y = (y_f - y_f.mean(0)).square().sum(-1).mean().sqrt()
        rad_x = (x_f - x_f.mean(0)).square().sum(-1).mean().sqrt()
        scale = rad_y.maximum(0.25 * rad_x).maximum((x_f.mean(0) - y_f.mean(0)).norm(dim=-1)).maximum(ell.float())
        c_norm = coord / where(scale > 0, scale.clamp_min(1e-30), 1.0).to(x_proj.dtype)
        w_rel, rel_mat = 0.03 * kappa + 0.5 * (1 - kappa), F.softplus(dr[None] - df[:, None]) - math.log(2)
        c_ot = -(p_slack_w * (0.5 * self.tau * df)).sum() + (w_rel * self.tau) * (p_align * rel_mat).sum()
        total = kappa * (c_norm + c_ot) + (1 - kappa) * c_rel
        if record is not None:
            if mutate and kappa >= 0.5:
                _observe_transport_thermal(x_proj, y_proj, ell, aux, disc, simp, has_cond, record)
            t_s = float(getattr(record, "thermal_scale", 1.0))
            total = (
                (kappa * (_scale_grad(c_norm, t_s) + c_ot) + (1 - kappa) * c_rel)
                if thermally_scaled
                else _scale_grad(total, t_s)
            )
        return self.weight * total

    def simplicial_manifold_loss(
        self,
        z_table: Tensor,
        gen_fn: Callable[[Tensor], Tensor],
        y_real: Tensor,
        *,
        record: Any = None,
        iso_weight: float = 0.45,
        chart_weight: float = 4.0,
    ) -> Tensor:
        """Simplicial manifold interpolation and chart alignment gated by g_conn."""
        n_pts, n_real = z_table.shape[0], y_real.shape[0]
        if self.weight == 0 or n_pts < 2 or n_real < 1:
            return y_real[:0].sum()
        phase = float(getattr(record, "chord_phase", 0.0))
        if n_pts > 512:
            z_table, n_pts = (
                z_table[((arange(128, device=z_table.device) * n_pts) // 128 + int(phase * n_pts)) % n_pts],
                128,
            )
        y_flat, use_f64 = y_real.detach().reshape(n_real, -1), (n_pts <= 256)
        with no_grad():
            g_conn, g_man, q90, y_scale, g_curve = _simplicial_gates(y_flat, chart_weight)
            z_det, ell = z_table.detach(), self._update_ell(y_flat, record, False)
            n_query = max(4, n_real if n_real <= 512 else 128)
            samples = _barycentric_samples(z_det, phase, n_query, g_curve)
        chart_on = g_conn > 1e-4 and chart_weight > 0
        x_live = gen_fn(z_table).reshape(n_pts, -1) if chart_on else None
        x_verts = x_live.detach() if x_live is not None else gen_fn(z_det).detach().reshape(n_pts, -1)
        x_samples = gen_fn(samples[0]).reshape(n_query, -1)
        _observe_simplicial_thermal(x_verts, x_samples, y_flat, ell, g_man, g_curve, record)
        cost_p, c_hat = born_infeld_cost(x_samples, y_flat, ell, fp64=use_f64)
        total = (softmax(-c_hat / self.eps, 1).to(cost_p.dtype) * cost_p).sum(1).mean()
        with no_grad():
            y_bar = softmax(-c_hat / (2.5 * self.eps), 1).to(y_flat.dtype) @ y_flat
        total = total + born_infeld_cost(x_samples, y_bar, 0.5 * ell, aligned=True, fp64=use_f64)[0].mean()
        eff_iso = ((1 - g_man) * iso_weight + g_man * 0.25) if g_man > 1e-4 else iso_weight
        if eff_iso > 0:
            total = total + eff_iso * _simplicial_isometry_loss(x_samples, x_verts, z_det, ell, g_man, samples, use_f64)
        if chart_on:
            total = total + _simplicial_chart_loss(
                z_table, x_live, x_verts, y_flat, ell, g_conn, g_curve, q90, y_scale, phase, chart_weight, use_f64
            )
        return self.weight * _scale_grad(total, float(getattr(record, "thermal_scale", 1.0)))

    def _bigan_codebook_loss(self, fake_x, real_x, real_z, prior_table, batch_rows, record, mutate: bool):
        _, coal, assign, r_halo, nn_dist = self._commit_atlas(
            fake_x, real_x.detach(), prior_table, batch_rows, 3, record, mutate
        )
        if assign is None or r_halo is None:
            return None
        ell_f = (0.15 * nn_dist.median()).clamp_min(1e-30)
        loss_fwd = _halo_born_infeld(fake_x, assign.batch_target, r_halo[assign.batch_comp], ell_f)
        with no_grad():
            table_f = prior_table.to(real_x.device, _F32)
            match_m = assign.row_comp[None] == safe_cdist(real_x, coal.comp_c).argmin(1)[:, None]
            has_inv, inv_idx = match_m.any(1), safe_cdist(real_z, table_f).masked_fill(~match_m, _INF).argmin(1)
            nn_z = (
                safe_cdist(table_f, diag_inf=True).amin(1).clamp_min(1e-4)
                if table_f.shape[0] >= 2
                else table_f.new_ones(table_f.shape[0])
            )
            inv_tgt, ell_z = table_f[inv_idx].to(real_z.dtype), (0.25 * nn_z.median()).clamp_min(1e-4).to(real_z.dtype)
            w_mask, hi_f = has_inv.to(real_z.dtype), has_inv.float()
            inv_rel = ((real_z.float() - table_f[inv_idx]).norm(dim=-1) / nn_z[inv_idx]).clamp_max(10)
            n_hits = hi_f.sum()
            bad_inv = ((inv_rel > 0.12) & has_inv).any() | (~has_inv.all())
            inv_mean = (inv_rel * hi_f).sum() / n_hits.clamp_min(1)
            tail_hit = (1.6 * inv_mean).maximum(where(bad_inv, inv_mean.new_full((), 0.2), inv_mean.new_zeros(())))
            tail_inv = where(n_hits > 0.5, tail_hit, inv_mean.new_full((), 0.25))
            r_tail = assign.tail_residual.float().maximum(tail_inv)
        loss_inv = (w_mask * born_infeld_cost(real_z, inv_tgt, ell_z, aligned=True)[0]).sum() / w_mask.sum().clamp_min(
            1
        )
        return loss_fwd + loss_inv, r_tail

    def joint_relativistic_loss(
        self,
        x_fake: Tensor,
        y_real: Tensor,
        d_fake: Tensor,
        d_real: Tensor,
        *,
        record: Any = None,
        cond_fake: Tensor | None = None,
        cond_real: Tensor | None = None,
        allow_state_update: bool = True,
        thermally_scaled: bool = False,
    ) -> Tensor:
        """Joint BiGAN / ByteGAN relativistic transport loss."""
        if self.weight == 0:
            return d_fake[:0].sum()
        n_f, n_r, mutate = d_fake.shape[0], d_real.shape[0], _can_mutate_state(record, allow_state_update)
        x_proj = project_features(x_fake.detach(), self.max_spatial, self.max_flat_dim)
        y_proj = project_features(y_real.detach(), self.max_spatial, self.max_flat_dim)
        ell = self._update_ell(y_proj, record, mutate)
        with no_grad():
            df, dr = _compute_log_odds(d_fake, d_real)
            mask = _discrete_cond_mask(cond_fake, cond_real, n_f, n_r)
            plan = self._sinkhorn_slack(born_infeld_cost(x_proj, y_proj, ell)[1], df, dr, mask)
            p_align, p_row_slack, p_col_slack = plan[:n_f, :n_r], plan[:n_f, n_r], plan[n_f, :n_r]
            self.last_p_xy_slack = p_row_slack
        fake_flat, real_flat = _flatten_logits(d_fake), _flatten_logits(d_real)
        mid = 0.5 * (fake_flat.detach().mean() + real_flat.detach().mean())
        rel = (p_align * F.softplus(real_flat[None] - fake_flat[:, None])).sum()
        rel = (
            rel + (p_row_slack * F.softplus(mid - fake_flat)).sum() + (p_col_slack * F.softplus(real_flat - mid)).sum()
        )
        loss = self.weight * rel
        if record is not None and cond_fake is None and cond_real is None:
            prior_table, batch_rows = recover_active_prior_rows(x_fake, record)
            split = _split_joint_prior_batch(x_fake, y_real, prior_table, batch_rows)
            if split is not None:
                cb = self._bigan_codebook_loss(split[0], split[2], split[3], prior_table, batch_rows, record, mutate)
                if cb is not None:
                    cb_loss, r_tail = cb
                    if mutate and hasattr(record, "observe_thermal"):
                        record.observe_thermal(0.0, 0.0, 0.0, 0.0, 0.0, r_tail=r_tail.item())
                    t_s, w_cb = float(getattr(record, "thermal_scale", 1.0)), self.weight * max(self.chart_weight, 1)
                    loss = (loss if thermally_scaled else _scale_grad(loss, t_s)) + _scale_grad(w_cb * cb_loss, t_s)
        return loss

    def conditional_fiber_loss(
        self,
        x_fake: Tensor,
        y_real: Tensor,
        c_cont: Tensor,
        *,
        cond_disc: Tensor | None = None,
        record: Any = None,
        allow_state_update: bool = True,
    ) -> Tensor:
        """Dispatch Law 3B conditional fiber loss with configured operator weights."""
        kwargs = {k: getattr(self, k) for k in _FIBER_CONFIG_KEYS}
        return conditional_fiber_loss(
            x_fake, y_real, c_cont, cond_disc=cond_disc, record=record, allow_state_update=allow_state_update, **kwargs
        )


def prepare_axiom_transport_step(
    transport: AxiomTransportLoss | None,
    record: Any,
    real: Tensor,
    *,
    prior_table: Tensor | None = None,
    batch_rows: Tensor | None = None,
    is_discrete_prior: bool = False,
    use_simplex: bool = False,
    simplex_chart_weight: float = 0.0,
) -> bool:
    """Bind active prior rows and atomic/simplicial flags for the current training step."""
    if is_discrete_prior and prior_table is not None and batch_rows is not None:
        set_active_prior_rows(prior_table, batch_rows)
        if record is not None:
            record._active_prior_table, record._active_batch_rows = prior_table, batch_rows.detach()
    is_atomic = False
    if use_simplex and real.shape[0] >= 8:
        max_sp, max_fd = getattr(transport, "max_spatial", 8), getattr(transport, "max_flat_dim", 256)
        is_atomic = compute_discrete_atomic_gate(project_features(real.detach(), max_sp, max_fd), record=record) > 0.5
    simplex_on = bool(use_simplex and not is_atomic and float(simplex_chart_weight) > 0)
    if hasattr(transport, "bind_discrete_prior_step"):
        assert transport is not None
        transport.bind_discrete_prior_step(bool(is_discrete_prior), simplicial_active=simplex_on)
    if hasattr(record, "_simplicial_step"):
        record._simplicial_step = simplex_on
    return simplex_on


sym_eigensplit_zero_eigh = sym_eigensplit_halley

__all__ = [
    "AxiomCondFiberBank",
    "AxiomTransportLoss",
    "compute_contrast_confidence",
    "compute_discrete_atomic_gate",
    "conditional_fiber_loss",
    "prepare_axiom_transport_step",
    "project_features",
    "project_transport_features",
    "sym_eigensplit_zero_eigh",
]
