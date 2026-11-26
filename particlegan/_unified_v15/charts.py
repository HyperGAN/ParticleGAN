"""Persistent multi-batch streaming chart bank and Laguerre hard commitment for Track 1B.

Maintains a clock-free online sufficient-statistics chart atlas
``(S, N, V, M_fake)`` of capacity ``K_max`` (breaking the ``B_r < M`` minibatch
horizon without O(K^2) Sinkhorn cost or RNG draws) and computes mass-conserving
Laguerre dual potentials with hard discrete chart commitment and a C^1 intra-mode
halo dead-zone so generator updates never suffer cross-valley barycentric
averaging (0.5-ghosts / inter-mode filaments) or single-batch multinomial jitter.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .._stats import pairwise_sq_dist

_PHI1 = 0.6180339887498949
_PHI2 = 0.7548776662466927


def deterministic_spatial_pool2d(x: torch.Tensor, max_spatial: int = 8) -> Tuple[torch.Tensor, float]:
    """CUDA-deterministic spatial block-mean projection returning ``(flat_fp32, area_ratio)``.

    Exact identity (``area_ratio = 1.0``) on all vectors and ``<= max_spatial`` images.
    Uses pure reshape + reduction (never ``adaptive_avg_pool2d_backward_cuda``) so backward
    is 100% deterministic under ``torch.use_deterministic_algorithms(True)``.
    """
    b_n = x.shape[0]
    if x.ndim >= 4 and (x.shape[-2] > max_spatial or x.shape[-1] > max_spatial):
        h_in, w_in = int(x.shape[-2]), int(x.shape[-1])
        h_out = min(int(max_spatial), h_in)
        w_out = min(int(max_spatial), w_in)
        x4d = x.reshape(-1, x.shape[-3], h_in, w_in).float()
        pad_h = (-h_in) % h_out
        pad_w = (-w_in) % w_out
        if pad_h or pad_w:
            x4d = F.pad(x4d, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x4d.shape[-2], x4d.shape[-1]
        kh, kw = hp // h_out, wp // w_out
        pooled = x4d.reshape(x4d.shape[0], x4d.shape[1], h_out, kh, w_out, kw).mean(dim=(3, 5))
        area_ratio = math.sqrt(float(h_in * w_in) / float(max(1, h_out * w_out)))
        return pooled.reshape(b_n, -1), area_ratio
    return x.reshape(b_n, -1).float(), 1.0


def project_chart_features(x: torch.Tensor, max_spatial: int = 8) -> torch.Tensor:
    """Project samples to bounded chart coordinates (exact identity on vectors and <=8x8 images)."""
    flat, _ = deterministic_spatial_pool2d(x, max_spatial=max_spatial)
    return flat


def project_transport_features(x: torch.Tensor, max_spatial: int = 8) -> torch.Tensor:
    """Differentiable CUDA-deterministic bounded-spatial transport projection."""
    if x.ndim >= 4 and (x.shape[-2] > max_spatial or x.shape[-1] > max_spatial):
        orig_dtype = x.dtype
        flat, area_ratio = deterministic_spatial_pool2d(x, max_spatial=max_spatial)
        return (flat * area_ratio).to(dtype=orig_dtype)
    return x.reshape(x.shape[0], -1)


def chart_effective_alpha(
    batch_size: int,
    num_charts: int,
    *,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.025,
    alpha_max: float = 0.35,
) -> float:
    """Clock-free (K, B) hit-rate EMA rate."""
    if batch_size <= 0 or num_charts <= 0:
        raise ValueError("batch_size and num_charts must be positive")
    ratio = float(batch_size) / float(num_charts)
    scaled = float(alpha_base) * (ratio ** float(hit_exponent))
    return float(min(alpha_max, max(alpha_min, scaled)))


def _batch_nn_scale_u(y_u: torch.Tensor, quantile: float = 0.25) -> torch.Tensor:
    """Unit-normalized intra-batch nearest-neighbor scale for novelty deduplication."""
    b_r = y_u.shape[0]
    if b_r <= 1:
        return y_u.new_tensor(1e-2)
    d_yy = torch.cdist(y_u, y_u, p=2.0)
    d_yy.fill_diagonal_(0.0)
    d_max = d_yy.amax()
    if float(d_max.item()) <= 0.0:
        return y_u.new_tensor(1e-3)
    thresh = 1e-5 * d_max
    pos_mask = d_yy > thresh
    pos_min = torch.where(pos_mask, d_yy, float("inf")).amin()
    d_yy.fill_diagonal_(float("inf"))
    nn_d = d_yy.amin(dim=-1)
    q_raw = torch.quantile(nn_d, quantile)
    tie_base = torch.where(torch.isfinite(pos_min), 0.15 * pos_min, 0.05 * d_max)
    return torch.where(q_raw <= thresh, tie_base, q_raw).clamp_min(1e-5)


def _novelty_initial_charts(y_flat: torch.Tensor, k_cap: int) -> torch.Tensor:
    """Deterministic farthest-point initial chart seeding without intra-mode duplication."""
    b_r, dim = y_flat.shape
    if b_r <= 2 or k_cap <= 2:
        k0 = min(max(1, k_cap), b_r)
        return y_flat[:k0].clone()
    scale = y_flat.abs().amax().clamp_min(1e-12)
    y_u = y_flat / scale
    d_yy_init = torch.cdist(y_u, y_u, p=2.0)
    d_max_u = d_yy_init.amax()
    ell_u = _batch_nn_scale_u(y_u, quantile=0.25)
    # Cap novelty deduplication radius at 8% of cloud diameter so B_r < M singleton
    # batches never merge distinct grid/mixture modes into coarse macro-cells.
    r_dedup_u = torch.minimum(2.1 * ell_u, (0.08 * d_max_u).clamp_min(1e-5))
    novelty_thresh_sq = r_dedup_u.square()

    idx_dim = torch.arange(1, dim + 1, device=y_flat.device, dtype=torch.float32)
    proj_dir = torch.remainder(idx_dim * _PHI1, 1.0) - 0.5
    proj_dir = proj_dir / proj_dir.norm().clamp_min(1e-8)
    first_idx = int(torch.argmin(y_u @ proj_dir).item())

    selected = [first_idx]
    min_d2 = (y_u - y_u[first_idx].unsqueeze(0)).square().sum(dim=1)
    min_d2[first_idx] = -1.0

    max_k0 = min(int(k_cap), b_r)
    min_k0 = min(4, max_k0)
    while len(selected) < max_k0:
        best_d2, nxt = min_d2.max(dim=0)
        if len(selected) >= min_k0 and float(best_d2.item()) <= float(novelty_thresh_sq.item()):
            break
        nxt_i = int(nxt.item())
        if float(best_d2.item()) <= 0.0:
            break
        selected.append(nxt_i)
        d2_new = (y_u - y_u[nxt_i].unsqueeze(0)).square().sum(dim=1)
        min_d2 = torch.minimum(min_d2, d2_new)
        min_d2[nxt_i] = -1.0

    sel_t = torch.tensor(selected, device=y_flat.device, dtype=torch.long)
    return y_flat[sel_t].clone()


def _estimate_initial_sq_dev(points: torch.Tensor, y_u_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Estimate initial per-chart squared deviation per unit mass from local spacing."""
    k = points.shape[0]
    if k == 0:
        return torch.empty((0,), device=points.device, dtype=torch.float32)
    scale = points.abs().amax().clamp_min(1e-12)
    if y_u_ref is not None and y_u_ref.shape[0] >= 2:
        ell_ref = _batch_nn_scale_u(y_u_ref, quantile=0.25) * scale
        d2_yc = pairwise_sq_dist(y_u_ref * scale / scale, points / scale)
        assign = d2_yc.argmin(dim=1)
        oh = F.one_hot(assign, num_classes=k).to(dtype=torch.float32)
        counts = oh.sum(dim=0)
        diff_sq = (y_u_ref * scale - points[assign]).square().sum(dim=1)
        sq_sum = oh.mT @ diff_sq
        fallback = ell_ref.square().clamp_min((1e-8 * scale * scale).clamp_min(1e-24))
        if k >= 2:
            u_pts = points / scale
            d2_cc = pairwise_sq_dist(u_pts)
            eye_c = torch.eye(k, device=points.device, dtype=torch.bool)
            nn_cc_d2 = d2_cc.masked_fill(eye_c, float("inf")).amin(dim=1) * (scale * scale)
            singleton_var = (0.06 * nn_cc_d2).clamp_min((1e-8 * scale * scale).clamp_min(1e-24))
            fallback_vec = torch.minimum(fallback.expand(k), singleton_var)
        else:
            fallback_vec = fallback.expand(k)
        return torch.where(
            counts >= 2.0,
            (sq_sum / counts.clamp_min(1.0)).clamp_min(0.25 * fallback_vec),
            fallback_vec,
        )
    if k <= 1:
        return torch.full((k,), max(1e-24, 1e-3 * float((scale * scale).item())), device=points.device, dtype=torch.float32)
    u = points / scale
    d2 = pairwise_sq_dist(u)
    eye = torch.eye(k, device=points.device, dtype=torch.bool)
    nn_d2 = d2.masked_fill(eye, float("inf")).amin(dim=1)
    pos = nn_d2[nn_d2 > 0.0]
    fallback = pos.quantile(0.25) if pos.numel() > 0 else u.new_tensor(1e-4)
    est_u = torch.where(nn_d2 > 0.0, 0.25 * nn_d2, 0.25 * fallback).clamp_min(1e-8)
    return (est_u * (scale * scale)).clamp_min(1e-24)


@torch.no_grad()
def update_streaming_charts(
    y_real: torch.Tensor,
    chart_sum: Optional[torch.Tensor],
    chart_count: Optional[torch.Tensor],
    chart_sq_dev: Optional[torch.Tensor],
    chart_fake_mass: Optional[torch.Tensor],
    revival_phase: float,
    revivals: int,
    *,
    max_charts: int = 256,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.025,
    alpha_max: float = 0.35,
    revival_tau: float = 0.008,
    split_eps: float = 1e-3,
    max_spatial: int = 8,
    allow_dead_revival: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    int,
]:
    """Update persistent ``(S, N, V, M_fake)`` chart bank from ``y_real``.

    Returns ``(C, w_real, radii, batch_real_count, S, N, V, M_fake, phase, revivals)``.
    Uses novelty-gated farthest-point insertion up to ``max_charts`` so disconnected
    modes each get clean macro-charts while ``M > B_r`` grids (e.g. ``grid100`` at
    ``B_r = 64``) accumulate all ``100`` modes across a few minibatches.
    """
    y = project_chart_features(y_real.detach(), max_spatial=max_spatial)
    b_r, dim = y.shape
    if b_r == 0 or dim == 0:
        raise ValueError("y_real must have positive batch size and feature dimension")

    k_cap = max(2, int(max_charts))
    scale_init = y.abs().amax().clamp_min(1e-12)
    y_u_init = y / scale_init

    if (
        chart_sum is None
        or chart_count is None
        or chart_sq_dev is None
        or chart_fake_mass is None
        or chart_sum.ndim != 2
        or chart_count.ndim != 1
        or chart_sq_dev.ndim != 1
        or chart_fake_mass.ndim != 1
        or chart_sum.shape[1] != dim
        or chart_sum.shape[0] != chart_count.shape[0]
        or chart_sum.shape[0] != chart_sq_dev.shape[0]
        or chart_sum.shape[0] != chart_fake_mass.shape[0]
        or chart_sum.shape[0] == 0
    ):
        c_init = _novelty_initial_charts(y, k_cap)
        k0 = c_init.shape[0]
        d2_init = pairwise_sq_dist(y_u_init, c_init / scale_init)
        assign0 = d2_init.argmin(dim=1)
        oh0 = F.one_hot(assign0, num_classes=k0).to(dtype=torch.float32)
        b_cnt0 = oh0.sum(dim=0)
        new_count = b_cnt0.clamp_min(0.5)
        b_sum0 = oh0.mT @ y
        c_emp = torch.where((b_cnt0 >= 1.0).unsqueeze(1), b_sum0 / b_cnt0.clamp_min(1.0).unsqueeze(1), c_init)
        new_sum = c_emp * new_count.unsqueeze(1)
        var_init = _estimate_initial_sq_dev(c_emp, y_u_ref=y_u_init)
        new_sq_dev = (var_init * new_count).clamp_min(1e-24)
        weights = new_count / new_count.sum().clamp_min(1e-8)
        new_fake_mass = weights.clone()
        radii = (new_sq_dev / new_count).clamp_min(1e-24).sqrt()
        return (
            c_emp,
            weights,
            radii,
            b_cnt0,
            new_sum,
            new_count,
            new_sq_dev,
            new_fake_mass,
            float(revival_phase),
            int(revivals),
        )

    cur_sum = chart_sum.to(device=y.device, dtype=torch.float32)
    cur_count = chart_count.to(device=y.device, dtype=torch.float32)
    cur_sq_dev = chart_sq_dev.to(device=y.device, dtype=torch.float32)
    cur_fake_mass = chart_fake_mass.to(device=y.device, dtype=torch.float32)

    # Novelty-gated progressive streaming growth when unseen modes arrive (breaks B_r < M cleanly).
    k_cur = cur_sum.shape[0]
    if k_cur < k_cap:
        c_existing = cur_sum / cur_count.clamp_min(1e-8).unsqueeze(1)
        r_existing = (cur_sq_dev / cur_count.clamp_min(1e-8)).clamp_min(1e-24).sqrt()
        scale_g = torch.maximum(y.abs().amax(), c_existing.abs().amax()).clamp_min(1e-12)
        y_ug = y / scale_g
        c_ug = c_existing / scale_g
        ell_ug = _batch_nn_scale_u(y_ug, quantile=0.25)
        d2_grow = pairwise_sq_dist(y_ug, c_ug)
        d_max_ug = torch.sqrt(d2_grow.amax().clamp_min(1e-8))
        ell_ug_eff = torch.minimum(ell_ug, (0.04 * d_max_ug).clamp_min(1e-5))
        min_d2_u, near_idx = d2_grow.min(dim=1)
        r_near_u = torch.minimum(
            torch.maximum(r_existing[near_idx] / scale_g, ell_ug_eff),
            (0.06 * d_max_ug).clamp_min(1e-5),
        )
        novel_thresh_u = (2.2 * r_near_u).square()

        added_indices = []
        cand_mask = min_d2_u > novel_thresh_u
        if bool(cand_mask.any()):
            min_d2_work = min_d2_u.clone()
            thresh_work = novel_thresh_u.clone()
            ell_u_sq = (2.1 * ell_ug_eff).square()
            while k_cur + len(added_indices) < k_cap:
                masked_scores = torch.where(min_d2_work > thresh_work, min_d2_work, y_ug.new_tensor(-1.0))
                best_val, nxt = masked_scores.max(dim=0)
                if float(best_val.item()) <= 0.0:
                    break
                nxt_i = int(nxt.item())
                added_indices.append(nxt_i)
                d2_new = (y_ug - y_ug[nxt_i].unsqueeze(0)).square().sum(dim=1)
                is_closer = d2_new < min_d2_work
                min_d2_work = torch.minimum(min_d2_work, d2_new)
                thresh_work = torch.where(is_closer, ell_u_sq, thresh_work)
                min_d2_work[nxt_i] = -1.0

        if added_indices:
            add_t = torch.tensor(added_indices, device=y.device, dtype=torch.long)
            c_add = y[add_t].clone()
            delta_k = c_add.shape[0]
            # Initial count matches 1-hit accumulated scale so rare/unseen modes retain honest mass.
            total_n = float(cur_count.sum().item())
            add_n_val = max(1e-3, total_n / float(max(b_r, k_cur + delta_k)))
            add_n = torch.full((delta_k,), add_n_val, device=y.device, dtype=torch.float32)
            ell_phys = ell_ug_eff * scale_g
            var_add = torch.full(
                (delta_k,),
                float((ell_phys * ell_phys).clamp_min((1e-8 * scale_g * scale_g).clamp_min(1e-24)).item()),
                device=y.device,
                dtype=torch.float32,
            )
            cur_sum = torch.cat([cur_sum, c_add * add_n.unsqueeze(1)], dim=0)
            cur_count = torch.cat([cur_count, add_n], dim=0)
            cur_sq_dev = torch.cat([cur_sq_dev, (var_add * add_n).clamp_min(1e-24)], dim=0)
            # Newly discovered real chart initially has zero observed fake mass until G visits it.
            cur_fake_mass = torch.cat(
                [cur_fake_mass, torch.zeros(delta_k, device=y.device, dtype=torch.float32)],
                dim=0,
            )

    k = cur_sum.shape[0]
    centroids_prev = cur_sum / cur_count.clamp_min(1e-8).unsqueeze(1)
    scale_u = torch.maximum(y.abs().amax(), centroids_prev.abs().amax()).clamp_min(1e-12)
    y_u = y / scale_u
    c_u = centroids_prev / scale_u

    d2_u = pairwise_sq_dist(y_u, c_u)
    d2_min_u, assign = d2_u.min(dim=1)
    d2_min = d2_min_u * (scale_u * scale_u)

    oh_batch = F.one_hot(assign, num_classes=k).to(dtype=torch.float32)
    batch_count = oh_batch.sum(dim=0)
    batch_sum = oh_batch.mT @ y

    diff_sq = (y - centroids_prev[assign]).square().sum(dim=1)
    batch_sq_dev = oh_batch.mT @ diff_sq

    alpha = chart_effective_alpha(
        b_r,
        k,
        alpha_base=alpha_base,
        hit_exponent=hit_exponent,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    pair_spread_u = (2.0 * y_u.var(dim=0, unbiased=False).sum()).clamp_min(1e-12)
    r2c_rms_t = torch.sqrt(d2_min_u.mean() / pair_spread_u)
    prev_var = (cur_sq_dev / cur_count.clamp_min(1e-8)).clamp_min((1e-10 * scale_u * scale_u).clamp_min(1e-24))
    out_of_chart = (d2_min > 12.0 * prev_var[assign]).to(dtype=torch.float32).mean()
    r2c_rms, out_frac = torch.stack((r2c_rms_t, out_of_chart)).tolist()
    shock_r2c = max(0.0, min(1.0, (float(r2c_rms) - 0.50) / 0.35))
    shock_out = max(0.0, min(1.0, (float(out_frac) - 0.55) / 0.30))
    shock = max(shock_r2c, shock_out)
    if shock > 0.88 and b_r >= 8:
        c_new = _novelty_initial_charts(y, k_cap)
        k_new = c_new.shape[0]
        d2_n = pairwise_sq_dist(y / scale_u, c_new / scale_u)
        asg_n = d2_n.argmin(dim=1)
        b_cnt_n = F.one_hot(asg_n, num_classes=k_new).to(dtype=torch.float32).sum(dim=0)
        new_count = b_cnt_n.clamp_min(0.5)
        new_sum = c_new * new_count.unsqueeze(1)
        var_new = _estimate_initial_sq_dev(c_new, y_u_ref=y_u)
        new_sq_dev = (var_new * new_count).clamp_min(1e-24)
        weights = new_count / new_count.sum().clamp_min(1e-8)
        new_fake_mass = weights.clone()
        radii = (new_sq_dev / new_count).clamp_min(1e-24).sqrt()
        return (
            c_new,
            weights,
            radii,
            b_cnt_n,
            new_sum,
            new_count,
            new_sq_dev,
            new_fake_mass,
            float(revival_phase),
            int(revivals),
        )

    alpha_eff = float(min(0.65, alpha + (0.65 - alpha) * (shock * shock)))

    new_sum = (1.0 - alpha_eff) * cur_sum + alpha_eff * batch_sum
    new_count = (1.0 - alpha_eff) * cur_count + alpha_eff * batch_count
    # Keep variance anchored when batch_count == 0 on an unvisited mode:
    # both new_sq_dev and new_count decay by (1 - alpha_eff), so their ratio V_k / N_k
    # remains exact on unvisited modes without decaying toward zero!
    new_sq_dev = ((1.0 - alpha_eff) * cur_sq_dev + alpha_eff * batch_sq_dev).clamp_min(1e-24)

    mean_count = new_count.mean()
    dead_mask = new_count < (float(revival_tau) * mean_count)
    phase = float(revival_phase)
    total_revivals = int(revivals)

    if allow_dead_revival and bool(dead_mask.any()) and k >= 2:
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(1).tolist()
        coord_std = y.std(dim=0, unbiased=False) + (1e-6 * scale_u).clamp_min(1e-15)
        dim_idx = torch.arange(1, dim + 1, device=y.device, dtype=torch.float32)

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
            q_tensor = torch.tensor([q], device=y.device, dtype=torch.float32)
            p_idx = int(torch.searchsorted(donor_cdf, q_tensor).item())
            p_idx = max(0, min(k - 1, p_idx))
            if p_idx == d_idx:
                p_idx = (d_idx + 1) % k

            c_donor = new_sum[p_idx] / new_count[p_idx].clamp_min(1e-8)
            v_donor = (new_sq_dev[p_idx] / new_count[p_idx].clamp_min(1e-8)).clamp_min(1e-24)
            half_total = 0.5 * (new_count[p_idx] + new_count[d_idx])
            half_fake = 0.5 * (cur_fake_mass[p_idx] + cur_fake_mass[d_idx])
            angles = dim_idx * ((phase + (offset + 1) * _PHI1) * (2.0 * math.pi))
            direction = torch.cos(angles)
            direction = direction / direction.norm().clamp_min(1e-8)
            delta = float(split_eps) * coord_std * direction
            new_count[p_idx] = half_total
            new_count[d_idx] = half_total
            new_sum[p_idx] = half_total * (c_donor - delta)
            new_sum[d_idx] = half_total * (c_donor + delta)
            new_sq_dev[p_idx] = half_total * v_donor
            new_sq_dev[d_idx] = half_total * v_donor
            cur_fake_mass[p_idx] = half_fake
            cur_fake_mass[d_idx] = half_fake
            total_revivals += 1

        phase = (phase + _PHI1) % 1.0

    centroids = new_sum / new_count.clamp_min(1e-8).unsqueeze(1)
    weights = new_count / new_count.sum().clamp_min(1e-8)
    radii = (new_sq_dev / new_count.clamp_min(1e-8)).clamp_min(1e-24).sqrt()
    return (
        centroids,
        weights,
        radii,
        batch_count,
        new_sum,
        new_count,
        new_sq_dev,
        cur_fake_mass,
        phase,
        total_revivals,
    )


@torch.no_grad()
def compute_chart_valley_gate(
    y_feat: torch.Tensor,
    charts: torch.Tensor,
    ell_chart: torch.Tensor,
    rho: float,
) -> float:
    """Persistent-chart valley indicator ``w_valley in [0, 1]`` immune to ``B_r < M`` singletons."""
    b_r = y_feat.shape[0]
    if b_r < 6 or charts.shape[0] < 2:
        return 0.0
    scale = torch.maximum(y_feat.abs().amax(), charts.abs().amax()).clamp_min(1e-12)
    y_u = y_feat.float() / scale
    c_u = charts.float() / scale
    d_yc = torch.cdist(y_u, c_u, p=2.0)
    yc_q75 = torch.quantile(d_yc.amin(dim=1), 0.75) * scale
    y_mid_u = 0.5 * (y_u + torch.roll(y_u, shifts=1, dims=0))
    d_mc = torch.cdist(y_mid_u, c_u, p=2.0)
    mid_q75 = torch.quantile(d_mc.amin(dim=1), 0.75) * scale
    denom = torch.maximum(yc_q75, ell_chart.detach().float()).clamp_min((1e-6 * scale).clamp_min(1e-15))
    nu_val = mid_q75 / denom
    u_val = ((nu_val - 0.85 * float(rho)) / 1.50).clamp(0.0, 1.0)
    return float((u_val * u_val * (3.0 - 2.0 * u_val)).item())


@torch.no_grad()
def laguerre_hard_commitment(
    x_feat: torch.Tensor,
    charts: torch.Tensor,
    weights: torch.Tensor,
    chart_radii: torch.Tensor,
    ell_chart: torch.Tensor,
    *,
    eps_chart: float = 0.08,
    iters: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced semi-discrete Laguerre assignment with hard Monge chart selection.

    Solves ``iters`` log-domain Sinkhorn scaling steps under metric Born-Infeld
    distances between uniform fake marginals ``1 / B_f`` and chart masses ``w_k``
    so starved modes raise their Laguerre dual price ``v_k`` until exact mass
    share ``w_k`` is claimed, then commits each fake sample ``i`` to its single
    winning chart ``k*(i) = argmax_k log Pi_{ik}`` (zero cross-mode averaging).
    """
    xf = x_feat.detach().float()
    cf = charts.detach().float()
    b_f = xf.shape[0]
    w = weights.detach().float().clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    scale = torch.maximum(
        torch.maximum(xf.abs().amax(), cf.abs().amax()),
        ell_chart.detach().float(),
    ).clamp_min(1e-12)
    d2_u = pairwise_sq_dist(xf / scale, cf / scale)
    ell_u = (ell_chart.detach().float() / scale).clamp_min(1e-6)
    bi_u = torch.sqrt(ell_u.square() + d2_u) - ell_u
    med_bi = bi_u.median().clamp_min(1e-8)
    norm_cost = (bi_u / med_bi).clamp_max(200.0)

    eps_val = max(1e-3, float(eps_chart))
    log_a = xf.new_full((b_f,), -math.log(max(1, b_f)))
    log_b = torch.log(w)
    log_kernel = -norm_cost / eps_val

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(int(iters)):
        v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)
        u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)

    log_pi = log_kernel + u.unsqueeze(1) + v.unsqueeze(0)
    k = cf.shape[0]
    if b_f >= 2 and k >= 2:
        # Exact quota-respecting Monge assignment with proximity-aware, index-unbiased
        # tie-breaking (closes D-06 B_f < K deterministic index-bias eviction without
        # distorting unequal-mass Hamilton remainders).
        w_cpu = w.detach().cpu()
        raw_q = w_cpu * float(b_f)
        quotas = torch.floor(raw_q).to(dtype=torch.long)
        rem = int(b_f - int(quotas.sum().item()))
        if rem > 0:
            frac = raw_q - quotas.to(dtype=torch.float32)
            prox_k = torch.exp(-norm_cost.amin(dim=0)).detach().cpu()
            weyl_k = torch.remainder((torch.arange(k, dtype=torch.float32, device="cpu") + 0.5) * _PHI1, 1.0)
            rem_score = frac + 1e-4 * prox_k + 1e-7 * weyl_k
            top_rem = torch.topk(rem_score, k=min(rem, k), largest=True).indices
            quotas[top_rem] += 1
        s_cpu = log_pi.detach().cpu()
        top2 = torch.topk(s_cpu, k=2, dim=1, largest=True).values
        weyl_b = torch.remainder((torch.arange(b_f, dtype=torch.float32, device="cpu") + 0.5) * _PHI1, 1.0)
        margins = top2[:, 0] - top2[:, 1] + 1e-7 * weyl_b
        order = torch.argsort(margins, descending=True)
        win_cpu = torch.empty(b_f, dtype=torch.long, device="cpu")
        neg_inf = float("-inf")
        s_work = s_cpu.clone()
        s_work[:, quotas <= 0] = neg_inf
        q_list = quotas.tolist()
        for idx_i in order.tolist():
            c_win = int(s_work[idx_i].argmax().item())
            win_cpu[idx_i] = c_win
            q_list[c_win] -= 1
            if q_list[c_win] <= 0:
                s_work[:, c_win] = neg_inf
        win_idx = win_cpu.to(device=xf.device)
    else:
        win_idx = log_pi.argmax(dim=1)
    return cf[win_idx], chart_radii.detach().float()[win_idx], win_idx


@torch.no_grad()
def deconflicted_chart_routing(
    x_feat_det: torch.Tensor,
    charts: torch.Tensor,
    w_real: torch.Tensor,
    chart_radii: torch.Tensor,
    batch_real_count: torch.Tensor,
    cur_fake_mass: torch.Tensor,
    ell_chart: torch.Tensor,
    *,
    w_valley: float,
    halo_mult: float = 2.15,
    s_thermal: float = 1.0,
    allow_update: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute hard chart targets, halo radii, routing weights, absent-chart shield, and updated fake EMA.

    Returns ``(target_charts, target_halos, route_weight, absent_shield, next_fake_mass)``:
    - ``route_weight`` ``[B_f]`` is strictly ``0`` inside settled non-surplus mode halos
      and positive only on (1) inter-mode valley stragglers outside ``r_halo`` (pulled to
      ``cf[near_idx]`` without cross-valley slot swaps) or (2) persistent ``> 2.8 * SE``
      overpopulated-to-starved chart reassignments (pulled to ``cf[win_idx]``).
    - ``absent_shield`` ``[B_f]`` in ``[0, 1]`` shields in-halo non-surplus particles
      from false minibatch-absence critic-slack kicks when ``B_r < K`` (e.g. ``grid100``).
    """
    xf = x_feat_det.float()
    cf = charts.float()
    b_f, k = xf.shape[0], cf.shape[0]
    ell_f = ell_chart.detach().float().clamp_min(1e-12)
    scale = torch.maximum(torch.maximum(xf.abs().amax(), cf.abs().amax()), ell_f).clamp_min(1e-12)

    # Strictly intra-mode halo radius: never inflate chart_radii by global/AKT ell_f,
    # and cap at 0.35 * nearest inter-chart distance so distinct modes never overlap halos.
    dist_floor = (1e-6 * scale).clamp_min(1e-15)
    if k >= 2:
        d_cc = torch.cdist(cf / scale, cf / scale, p=2.0) * scale
        d_cc.fill_diagonal_(float("inf"))
        nn_cc = d_cc.amin(dim=1).clamp_min(dist_floor)
        radii_eff = chart_radii.float().clamp_min(0.02 * nn_cc)
        r_halo_all = torch.minimum(float(halo_mult) * radii_eff, 0.35 * nn_cc)
        basin_rad_all = torch.minimum(1.35 * r_halo_all, 0.45 * nn_cc)
    else:
        radii_eff = chart_radii.float().clamp_min((1e-5 * scale).clamp_min(1e-15))
        r_halo_all = float(halo_mult) * radii_eff
        basin_rad_all = 1.35 * r_halo_all

    d_xc = torch.cdist(xf / scale, cf / scale, p=2.0) * scale
    d_near, near_idx = d_xc.min(dim=1)
    r_halo_near = r_halo_all[near_idx]
    basin_near = basin_rad_all[near_idx]

    # Update persistent in-basin fake chart occupancy EMA (zero RNG, clock-free, CUDA-deterministic).
    in_basin = d_near <= basin_near
    if bool(in_basin.any()):
        batch_fake_cnt = F.one_hot(near_idx[in_basin], num_classes=k).to(dtype=torch.float32).sum(dim=0)
    else:
        batch_fake_cnt = torch.zeros(k, device=xf.device, dtype=torch.float32)
    batch_fake_frac = batch_fake_cnt / float(max(1, b_f))
    alpha_f = chart_effective_alpha(max(1, b_f), max(1, k))
    next_fake_mass = (
        (1.0 - alpha_f) * cur_fake_mass.float() + alpha_f * batch_fake_frac
        if allow_update
        else cur_fake_mass.float()
    )

    # Binomial EMA standard-error deadband eliminates single-batch multinomial jitter!
    var_factor = 2.0 * (alpha_f / max(1e-6, 2.0 - alpha_f)) / float(max(1, b_f))
    se_k = torch.sqrt((var_factor * w_real.float()).clamp_min(1e-8))
    deficit_k = F.relu(w_real.float() - next_fake_mass - 1.60 * se_k)
    surplus_k = F.relu(next_fake_mass - w_real.float() - 1.60 * se_k)

    # Shield settled in-halo non-surplus samples whose chart had 0 real draws this batch (B_r < K).
    if w_valley > 1e-4 and k > 4:
        u_in = ((r_halo_near - d_near) / (0.35 * r_halo_near).clamp_min(dist_floor)).clamp(0.0, 1.0)
        s_in = u_in * u_in * (3.0 - 2.0 * u_in)
        is_absent = (batch_real_count <= 0.0).to(dtype=torch.float32)[near_idx]
        no_surplus = (surplus_k <= 1e-6).to(dtype=torch.float32)[near_idx]
        absent_shield = float(w_valley) * s_in * is_absent * no_surplus
    else:
        absent_shield = torch.zeros(b_f, device=xf.device, dtype=torch.float32)

    # Scalar centered log-odds has 1 degree of freedom and already uniquely balances K <= 2 modes;
    # multi-mode cross-valley ambiguity begins strictly at K >= 3 disconnected charts!
    if w_valley <= 1e-4 or k < 3:
        zeros = torch.zeros(b_f, device=xf.device, dtype=torch.float32)
        return cf[near_idx], r_halo_near, zeros, absent_shield, next_fake_mass

    # 1. Valley / out-of-halo release gate for inter-mode bridges & 0.5-ghosts.
    u_out = ((d_near - r_halo_near) / (0.45 * r_halo_near).clamp_min(dist_floor)).clamp(0.0, 1.0)
    g_valley_row = float(w_valley) * (u_out * u_out * (3.0 - 2.0 * u_out))

    # 2. Quota-balanced Laguerre commitment across K >= 3 charts (avoids Voronoi init asymmetry
    #    on valley stragglers while rebalancing >1.6 SE surplus -> deficit charts under s_thermal > 0.02).
    w_target = w_real.float() + 2.0 * deficit_k
    w_target = w_target / w_target.sum().clamp_min(1e-8)
    _, _, win_idx = laguerre_hard_commitment(
        xf, cf, w_target, radii_eff, ell_f, eps_chart=0.08, iters=8
    )

    u_th = min(1.0, max(0.0, (float(s_thermal) - 0.02) / 0.18))
    s_th_gate = u_th * u_th * (3.0 - 2.0 * u_th)
    total_def = float(deficit_k.sum().item())
    if total_def > 1e-4 and s_th_gate > 1e-4:
        win_def = deficit_k[win_idx]
        src_sur = surplus_k[near_idx]
        cross_chart = (win_idx != near_idx) & (win_def > 0.0) & (src_sur > 0.0)
        rebal_strength = torch.clamp(
            (win_def / (w_real[win_idx].clamp_min(1e-6))).clamp(0.0, 1.0),
            0.0,
            1.0,
        )
        g_rebal_row = torch.where(
            cross_chart,
            float(w_valley * s_th_gate) * rebal_strength,
            torch.zeros_like(g_valley_row),
        )
    else:
        g_rebal_row = torch.zeros_like(g_valley_row)

    # Valley stragglers commit to quota-balanced win_idx while hot (s_th_gate > 0.5) so K>=3
    # modes populate evenly, and settle to local near_idx once thermally cool.
    target_idx = win_idx if s_th_gate > 0.5 else torch.where(g_rebal_row > 0.0, win_idx, near_idx)
    target_c = cf[target_idx]
    target_halo = r_halo_all[target_idx]
    route_weight = torch.maximum(g_valley_row, g_rebal_row)
    return target_c, target_halo, route_weight, absent_shield, next_fake_mass
