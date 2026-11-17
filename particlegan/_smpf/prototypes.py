"""Online clock-free s-means sufficient-statistics prototype bank for SMPF.

Adapts the four core algorithmic pillars of s-means (Kocabay, Yuksel & Yalcin,
2026; https://suayptalha.github.io/s-means-blog/) to continuous adversarial
transport without dataset-size clocks, epoch-horizon schedules, or training
RNG stream consumption:
1. Multi-round deterministic D^1.5 coreset seeding (4 rounds, power=1.5).
2. Cache/SRAM-friendly tiled L2 squared-distance assignment (addmm tiles).
3. Distortion x usage^0.35 anisotropic RMS dead-prototype revival across all
   dimensions d >= 1 using online per-prototype sse_ema and inter-cluster
   per-feature variance feat_scale.
4. State-settling power-cosine step scaling plus progressive fast-window
   accumulation with Empirical-Bayes count-adaptive centroid shrinkage (tau=0.20).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

# Golden-ratio and plastic-constant low-discrepancy increments.
_PHI1 = 0.6180339887498949
_PHI2 = 0.7548776662466927
_PHI3 = 0.5436890126920764


def smeans_effective_alpha(
    batch_size: int,
    num_prototypes: int,
    *,
    ema_residual: float = 1.0,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.015,
    alpha_max: float = 0.35,
    settle_horizon: float = 0.16,
) -> float:
    """Clock-free (K, B) hit-rate EMA rate modulated by s-means power-cosine settling.

    Replaces s-means's epoch progress ``step / T_steps`` with dimensionless
    transport unsettledness ``u = clip(ema_residual / settle_horizon, 0, 1)``,
    preserving full hit-rate responsiveness during acquisition/reacquisition
    (``u = 1``) while decaying smoothly via power-cosine ``0.92`` at equilibrium.
    """
    if batch_size <= 0 or num_prototypes <= 0:
        raise ValueError("batch_size and num_prototypes must be positive")
    ratio = float(batch_size) / float(num_prototypes)
    hit_alpha = float(alpha_base) * (ratio ** float(hit_exponent))
    hit_alpha = min(float(alpha_max), max(float(alpha_min), hit_alpha))
    u_live = max(0.0, min(1.0, float(ema_residual) / max(float(settle_horizon), 1e-6)))
    cosine_factor = 0.25 + 0.75 * ((0.5 * (1.0 - math.cos(math.pi * u_live))) ** 0.92)
    return float(min(alpha_max, max(alpha_min, hit_alpha * cosine_factor)))


def tiled_smeans_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    tile_rows: int = 512,
    tile_cols: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest centroid indices and squared Euclidean distances via addmm tiles."""
    x_f = x.detach().to(dtype=torch.float32)
    c_f = centroids.detach().to(device=x_f.device, dtype=torch.float32)
    n, k = int(x_f.shape[0]), int(c_f.shape[0])
    if n == 0 or k == 0:
        raise ValueError("x and centroids must be nonempty")
    x_norm = (x_f * x_f).sum(dim=1)
    c_norm = (c_f * c_f).sum(dim=1)
    c_t = c_f.t().contiguous()

    labels = torch.empty((n,), dtype=torch.long, device=x_f.device)
    sq_dists = torch.empty((n,), dtype=torch.float32, device=x_f.device)

    eff_r = max(1, min(int(tile_rows), n))
    eff_c = max(1, min(int(tile_cols), k))

    for i0 in range(0, n, eff_r):
        i1 = min(i0 + eff_r, n)
        xb = x_f[i0:i1]
        xbn = x_norm[i0:i1]
        best_val = None
        best_idx = None
        for j0 in range(0, k, eff_c):
            j1 = min(j0 + eff_c, k)
            tile = torch.addmm(c_norm[j0:j1], xb, c_t[:, j0:j1], alpha=-2.0, beta=1.0)
            vals, locs = tile.min(dim=1)
            if best_val is None:
                best_val = vals
                best_idx = locs + j0
            else:
                mask = vals < best_val
                best_val = torch.where(mask, vals, best_val)
                best_idx = torch.where(mask, locs + j0, best_idx)
        assert best_val is not None and best_idx is not None
        labels[i0:i1] = best_idx
        sq_dists[i0:i1] = (xbn + best_val).clamp_min_(0.0)

    return labels, sq_dists


def d15_initial_prototypes(
    x: torch.Tensor,
    k: int,
    *,
    rounds: int = 4,
    power: float = 1.5,
) -> torch.Tensor:
    """Deterministic multi-round D^1.5 geometric coreset seeding (zero RNG draws)."""
    x_f = x.detach().to(dtype=torch.float32)
    n, dim = int(x_f.shape[0]), int(x_f.shape[1])
    if k >= n:
        return x_f.clone()
    if k <= 0:
        raise ValueError("k must be positive")

    idx_dim = torch.arange(1, dim + 1, device=x_f.device, dtype=torch.float32)
    coord_scale = x_f.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_white = (x_f - x_f.mean(dim=0, keepdim=True)) / coord_scale

    nr = max(2, int(rounds))
    k0 = max(1, min(k, max(4, k // (nr * 2))))

    dir0 = torch.sin(idx_dim * (2.0 * math.pi * _PHI1))
    dir0 = dir0 / dir0.norm().clamp_min(1e-8)
    order0 = torch.argsort(x_white @ dir0, stable=True)
    slots0 = ((torch.arange(k0, device=x_f.device, dtype=torch.long) * n) + (n // 2)) // k0
    slots0 = slots0.clamp(0, n - 1)
    picked0 = order0[slots0]

    picked_mask = torch.zeros(n, dtype=torch.bool, device=x_f.device)
    picked_mask[picked0] = True
    picked_list = [picked0]
    centroids = x_f[picked0]

    _, dmin_sq = tiled_smeans_assign(x_f, centroids)
    step_k = max(1, (k - k0 + nr - 2) // max(1, nr - 1))

    for r_idx in range(1, nr):
        cur_k = int(centroids.shape[0])
        if cur_k >= k:
            break
        cand_n = min(step_k if r_idx < nr - 1 else (k - cur_k), k - cur_k)
        unpicked_idx = torch.nonzero(~picked_mask, as_tuple=False).squeeze(1)
        if unpicked_idx.numel() == 0:
            break
        if unpicked_idx.numel() <= cand_n:
            chosen = unpicked_idx
        else:
            dir_r = torch.cos(idx_dim * (2.0 * math.pi * ((_PHI2 * (r_idx + 1)) % 1.0)))
            dir_r = dir_r / dir_r.norm().clamp_min(1e-8)
            sub_order = torch.argsort(x_white[unpicked_idx] @ dir_r, stable=True)
            ordered_idx = unpicked_idx[sub_order]
            w_sub = (dmin_sq[ordered_idx].clamp_min(1e-12) ** (0.5 * float(power))) + 1e-12
            cdf = torch.cumsum(w_sub, dim=0)
            cdf = cdf / cdf[-1].clamp_min(1e-12)
            q = (torch.arange(cand_n, device=x_f.device, dtype=torch.float32) + 0.5) / float(cand_n)
            locs = torch.searchsorted(cdf, q).clamp(0, ordered_idx.numel() - 1)
            # Deduplicate any colliding quantile slots deterministically without CPU loops when unique suffices.
            unique_locs = torch.unique(locs, sorted=True)
            if unique_locs.numel() < cand_n:
                avail = torch.ones(ordered_idx.numel(), dtype=torch.bool, device=x_f.device)
                avail[unique_locs] = False
                rem_needed = cand_n - int(unique_locs.numel())
                rem_pool = torch.nonzero(avail, as_tuple=False).squeeze(1)
                extra_w = w_sub[rem_pool]
                extra_sel = rem_pool[torch.topk(extra_w, k=min(rem_needed, int(rem_pool.numel()))).indices]
                unique_locs = torch.cat([unique_locs, extra_sel], dim=0)
            chosen = ordered_idx[unique_locs[:cand_n]]

        picked_mask[chosen] = True
        picked_list.append(chosen)
        new_c = x_f[chosen]
        centroids = torch.cat([centroids, new_c], dim=0)
        _, new_d_sq = tiled_smeans_assign(x_f, new_c)
        dmin_sq = torch.minimum(dmin_sq, new_d_sq)

    if int(centroids.shape[0]) < k:
        rem = torch.nonzero(~picked_mask, as_tuple=False).squeeze(1)[: (k - int(centroids.shape[0]))]
        if rem.numel() > 0:
            centroids = torch.cat([centroids, x_f[rem]], dim=0)
    return centroids[:k].contiguous()


def _empirical_bayes_readout(
    centroids: torch.Tensor,
    usage: torch.Tensor,
    sse_ema: torch.Tensor,
    fast_sum: torch.Tensor,
    fast_count: torch.Tensor,
    *,
    eb_tau: float = 0.20,
    out_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(c_eb, weights, ell_proto)`` using Empirical-Bayes count shrinkage."""
    mean_fast = fast_count.mean().clamp_min(1e-8)
    tau_scaled = float(eb_tau) * mean_fast
    w_eb = fast_count / (fast_count + tau_scaled).clamp_min(1e-12)
    fast_mean = fast_sum / fast_count.clamp_min(1e-8).unsqueeze(1)
    c_eb = torch.where(
        (fast_count > 1e-8).unsqueeze(1),
        w_eb.unsqueeze(1) * fast_mean + (1.0 - w_eb.unsqueeze(1)) * centroids,
        centroids,
    )
    weights = usage / usage.sum().clamp_min(1e-8)
    ell_proto = (sse_ema / usage.clamp_min(1e-6)).clamp_min(1e-10).sqrt()
    return (
        c_eb.to(dtype=out_dtype),
        weights.to(dtype=out_dtype),
        ell_proto.to(dtype=out_dtype),
    )


@torch.no_grad()
def update_smeans_prototypes(
    x_real: torch.Tensor,
    proto_centroids: torch.Tensor | None,
    proto_usage: torch.Tensor | None,
    proto_sse: torch.Tensor | None,
    feat_scale: torch.Tensor | None,
    fast_sum: torch.Tensor | None,
    fast_count: torch.Tensor | None,
    revival_phase: float,
    revivals: int,
    *,
    ema_residual: float = 1.0,
    max_prototypes: int = 256,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.015,
    alpha_max: float = 0.35,
    revival_tau: float = 0.10,
    split_mult: float = 0.55,
    donor_usage_w: float = 0.35,
    eb_tau: float = 0.20,
    sse_decay: float = 0.85,
    var_beta: float = 0.85,
) -> Tuple[
    torch.Tensor,
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
    """Update online s-means prototype bank and return EB centroids, weights, radii, and bank state."""
    x = x_real.detach().flatten(1).to(dtype=torch.float32)
    b_r, dim = int(x.shape[0]), int(x.shape[1])
    if b_r == 0 or dim == 0:
        raise ValueError("x_real must have positive batch size and feature dimension")

    need_init = (
        proto_centroids is None
        or proto_usage is None
        or proto_sse is None
        or feat_scale is None
        or fast_sum is None
        or fast_count is None
        or proto_centroids.ndim != 2
        or proto_usage.ndim != 1
        or proto_sse.ndim != 1
        or feat_scale.ndim != 1
        or fast_sum.ndim != 2
        or fast_count.ndim != 1
        or proto_centroids.shape[0] == 0
        or proto_centroids.shape[1] != dim
        or proto_usage.shape[0] != proto_centroids.shape[0]
        or proto_sse.shape[0] != proto_centroids.shape[0]
        or feat_scale.shape[0] != dim
        or fast_sum.shape != proto_centroids.shape
        or fast_count.shape != proto_usage.shape
    )

    if need_init:
        k = min(int(max_prototypes), b_r)
        c_init = d15_initial_prototypes(x, k, rounds=4, power=1.5)
        assign0, sq0 = tiled_smeans_assign(x, c_init)
        b_cnt0 = torch.zeros(k, device=x.device, dtype=torch.float32)
        b_cnt0.scatter_add_(0, assign0, torch.ones(b_r, device=x.device, dtype=torch.float32))
        b_sum0 = torch.zeros((k, dim), device=x.device, dtype=torch.float32)
        b_sum0.index_add_(0, assign0, x)
        b_sse0 = torch.zeros(k, device=x.device, dtype=torch.float32)
        b_sse0.scatter_add_(0, assign0, sq0)

        floor_u = max(1e-4, 0.05 * float(b_r) / float(k))
        touched0 = b_cnt0 > 0
        c_refined = torch.where(
            touched0.unsqueeze(1),
            b_sum0 / b_cnt0.clamp_min(1.0).unsqueeze(1),
            c_init,
        )
        u_init = b_cnt0.clamp_min(floor_u)
        # Seed empty initial prototypes with median empirical cluster distortion.
        pos_sse = b_sse0[touched0] / b_cnt0[touched0].clamp_min(1.0)
        med_mse0 = pos_sse.median() if pos_sse.numel() > 0 else x.var(dim=0, unbiased=False).sum().clamp_min(1e-6)
        sse_init = torch.where(touched0, b_sse0.clamp_min(1e-8 * u_init), u_init * med_mse0.clamp_min(1e-8))
        fscale_init = x.var(dim=0, unbiased=False).clamp_min(1e-8)
        fsum_init = c_refined * u_init.unsqueeze(1)
        fcnt_init = u_init.clone()

        c_eb, weights, ell_proto = _empirical_bayes_readout(
            c_refined, u_init, sse_init, fsum_init, fcnt_init, eb_tau=eb_tau, out_dtype=x_real.dtype
        )
        return (
            c_eb,
            weights,
            ell_proto,
            c_refined,
            u_init,
            sse_init,
            fscale_init,
            fsum_init,
            fcnt_init,
            float(revival_phase),
            int(revivals),
        )

    assert proto_centroids is not None and proto_usage is not None and proto_sse is not None
    assert feat_scale is not None and fast_sum is not None and fast_count is not None

    cur_c = proto_centroids.to(device=x.device, dtype=torch.float32)
    cur_u = proto_usage.to(device=x.device, dtype=torch.float32)
    cur_sse = proto_sse.to(device=x.device, dtype=torch.float32)
    cur_fscale = feat_scale.to(device=x.device, dtype=torch.float32)
    cur_fsum = fast_sum.to(device=x.device, dtype=torch.float32)
    cur_fcnt = fast_count.to(device=x.device, dtype=torch.float32)
    k = int(cur_c.shape[0])

    assign, sq_dists = tiled_smeans_assign(x, cur_c)
    batch_count = torch.zeros(k, device=x.device, dtype=torch.float32)
    batch_count.scatter_add_(0, assign, torch.ones(b_r, device=x.device, dtype=torch.float32))
    batch_sum = torch.zeros((k, dim), device=x.device, dtype=torch.float32)
    batch_sum.index_add_(0, assign, x)
    batch_sse = torch.zeros(k, device=x.device, dtype=torch.float32)
    batch_sse.scatter_add_(0, assign, sq_dists)

    alpha = smeans_effective_alpha(
        b_r,
        k,
        ema_residual=ema_residual,
        alpha_base=alpha_base,
        hit_exponent=hit_exponent,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    # Clock-free distribution-shift & Poisson empty-cell shock detector.
    pair_spread = (2.0 * x.var(dim=0, unbiased=False).sum()).clamp_min(1e-12)
    r2p_rms_t = torch.sqrt(sq_dists.mean() / pair_spread)
    if k >= 16 and b_r >= 16:
        cur_w = cur_u / cur_u.sum().clamp_min(1e-8)
        expected_zero_t = torch.exp(-float(b_r) * cur_w).mean()
        actual_zero_t = (batch_count == 0).to(dtype=torch.float32).mean()
        r2p_rms, expected_zero, actual_zero = torch.stack(
            (r2p_rms_t, expected_zero_t, actual_zero_t)
        ).tolist()
        shock_zero = max(0.0, min(1.0, (float(actual_zero) - float(expected_zero) - 0.35) / 0.25))
    else:
        r2p_rms = float(r2p_rms_t.item())
        shock_zero = 0.0
    shock_r2p = max(0.0, min(1.0, (float(r2p_rms) - 0.42) / 0.35))
    shock = max(shock_r2p, shock_zero)
    alpha_eff = float(min(0.50, alpha + (0.50 - alpha) * (shock * shock)))

    # Online inter-cluster per-feature variance (s-means O(Kd) feat_scale update).
    batch_mu = batch_sum.sum(dim=0) / float(b_r)
    cur_var = (
        (batch_sum - batch_count.unsqueeze(1) * batch_mu.unsqueeze(0)).pow(2).sum(dim=0) / float(b_r)
    ).clamp_min(1e-10)
    new_fscale = float(var_beta) * cur_fscale + (1.0 - float(var_beta)) * cur_var

    # Count-mass EMA centroid, usage, and matched-retention distortion SSE update.
    # Matching SSE retention (1 - alpha_eff) to usage retention keeps ell_k^2 = sse_k / u_k
    # invariant on batches where rare-mode prototypes receive zero hits.
    touched = batch_count > 0
    safe_c = torch.where(touched, batch_count, torch.ones_like(batch_count))
    batch_mean = batch_sum / safe_c.unsqueeze(1)
    retained = cur_u * (1.0 - alpha_eff)
    mass = retained + alpha_eff * batch_count
    w_upd = (alpha_eff * batch_count) / torch.where(touched, mass, torch.ones_like(mass)).clamp_min(1e-12)
    upd_c = cur_c + w_upd.unsqueeze(1) * (batch_mean - cur_c)
    new_c = torch.where(touched.unsqueeze(1), upd_c, cur_c)
    new_u = torch.where(touched, mass, retained).clamp_min(1e-8)
    new_sse = ((1.0 - alpha_eff) * cur_sse + alpha_eff * batch_sse).clamp_min(1e-12)

    # Clock-free progressive settling-weighted fast accumulator for Empirical-Bayes shrinkage.
    u_live = max(0.0, min(1.0, float(ema_residual) / 0.16))
    w_pass = 0.65 + 0.70 * (1.0 - u_live)
    alpha_fast = max(min(0.60, 2.0 * alpha_eff), shock * shock)
    new_fsum = (1.0 - alpha_fast) * cur_fsum + (alpha_fast * w_pass) * batch_sum
    new_fcnt = ((1.0 - alpha_fast) * cur_fcnt + (alpha_fast * w_pass) * batch_count).clamp_min(1e-8)

    # Vectorized distortion-weighted anisotropic dead-prototype revival (s-means all d >= 1).
    thresh = new_u.mean() * float(revival_tau)
    unused = new_u < thresh
    donors = (~unused) & (new_u > thresh)
    score_vec = new_sse.clamp_min(1e-12) * (new_u.clamp_min(1e-12) ** float(donor_usage_w))
    if bool(donors.any()):
        donor_mean_score = score_vec[donors].mean()
        eligible_donors = donors & (score_vec >= 0.15 * donor_mean_score)
        if not bool(eligible_donors.any()):
            eligible_donors = donors
    else:
        eligible_donors = donors
    n_un, n_do = torch.stack((unused.sum(), eligible_donors.sum())).tolist()
    n_sp = min(int(n_un), int(n_do))
    phase = float(revival_phase)
    total_revivals = int(revivals)

    if n_sp > 0 and k >= 2:
        k_idx = torch.arange(k, device=x.device, dtype=torch.float32)
        u_det = torch.remainder((k_idx + 0.5) / float(k) + phase * _PHI2, 1.0).clamp(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(u_det))
        keys = torch.where(eligible_donors, torch.log(score_vec.clamp_min(1e-30)) + 0.35 * gumbel, float("-inf"))
        src = torch.topk(keys, k=n_sp).indices
        tgt = torch.nonzero(unused, as_tuple=False).squeeze(1)[:n_sp]

        fvec = new_fscale.clamp_min(1e-10).sqrt()
        fvec = fvec / fvec.norm().clamp_min(1e-8) * math.sqrt(float(dim))

        dc = new_c[src].clone()
        rad = float(split_mult) * (
            new_sse[src] / (new_u[src].clamp_min(1e-6) * float(dim))
        ).clamp_min(1e-10).sqrt().unsqueeze(1)

        s_idx = torch.arange(1, n_sp + 1, device=x.device, dtype=torch.float32).unsqueeze(1)
        d_idx = torch.arange(1, dim + 1, device=x.device, dtype=torch.float32).unsqueeze(0)
        rad_phase = torch.remainder(phase + s_idx * _PHI1 + d_idx * _PHI3, 1.0)
        radem = torch.where(rad_phase < 0.5, -1.0, 1.0)
        delta = rad * radem * fvec.unsqueeze(0)

        half_u = (0.5 * (new_u[src] + new_u[tgt])).clamp_min(1e-8)
        half_sse = (0.5 * (new_sse[src] + new_sse[tgt])).clamp_min(1e-12)
        half_fcnt = (0.5 * (new_fcnt[src] + new_fcnt[tgt])).clamp_min(1e-8)

        c_minus = dc - delta
        c_plus = dc + delta
        new_c[src] = c_minus
        new_c[tgt] = c_plus
        new_u[src] = half_u
        new_u[tgt] = half_u
        new_sse[src] = half_sse
        new_sse[tgt] = half_sse
        new_fcnt[src] = half_fcnt
        new_fcnt[tgt] = half_fcnt
        new_fsum[src] = half_fcnt.unsqueeze(1) * c_minus
        new_fsum[tgt] = half_fcnt.unsqueeze(1) * c_plus

        total_revivals += n_sp
        phase = (phase + _PHI1) % 1.0

    c_eb, weights, ell_proto = _empirical_bayes_readout(
        new_c, new_u, new_sse, new_fsum, new_fcnt, eb_tau=eb_tau, out_dtype=x_real.dtype
    )
    return (
        c_eb,
        weights,
        ell_proto,
        new_c,
        new_u,
        new_sse,
        new_fscale,
        new_fsum,
        new_fcnt,
        phase,
        total_revivals,
    )
