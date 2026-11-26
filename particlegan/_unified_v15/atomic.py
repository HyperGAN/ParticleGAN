"""Unified-v15 scale-invariant discrete-atomic support detection, Zipf-fair M!=K quotas,
1-stays / surplus-moves bijection ledger, and joint BiGAN inverse codebook contraction
for ``self_coupling='feydy_atomic_v15'``.

Upgrades ``_t1_coupling/atomic.py`` to close E10 and D-06:
- Removes ``y_scale.clamp_min(1.0)`` for exact scale & translation invariance across all decades.
- Replaces the brittle machine-precision-only duplicate gate with a bimodal pairwise log-gap
  and dimension-normalized ``sigma_coord / Delta`` discrete-cluster separation gate (handling
  both zero-noise codebooks like ByteGAN and noisy non-hypercube codebooks with ``sigma/Delta <= 0.025``
  without false-triggering on continuous densities, curved manifolds, or ``grid100``).
- Eliminates ``1e-7 * arange(k)`` / ``1e-6 * arange(k)`` index bias in ``compute_atomic_quotas``
  and ``update_atomic_ledger`` via proximity-aware and Weyl golden-ratio tie-breaking so unequal
  Zipf weights and ``M != K`` capacities never evict high-index charts.

All primitives are strictly recipe-owned, task-agnostic, clock-free, zero-RNG, CUDA-deterministic,
and free of ``autograd.grad`` / SVD / ``eigh`` / ``pinv``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

_PHI1 = 0.6180339887498949


def _weyl_tie(n: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Deterministic index-unbiased Weyl golden-ratio tie-breaker in ``[0, 1)``."""
    return torch.remainder((torch.arange(n, dtype=torch.float32, device=device) + 0.5) * _PHI1, 1.0)


@torch.no_grad()
def compute_discrete_atomic_gate(
    y_flat: torch.Tensor,
    *,
    charts: Optional[torch.Tensor] = None,
    radii: Optional[torch.Tensor] = None,
) -> float:
    """Return smooth indicator ``g_atom in [0, 1]`` for finite discrete/tight-cluster atomic support.

    Distinguishes discrete atomic targets — both exact zero-noise atoms (``sigma == 0``) and
    well-separated noisy discrete clusters (``sigma_coord / Delta <= 0.024`` with a clean empty
    annular gap) — from continuous manifolds, 1D/nD smooth densities, and broad Gaussian mixtures:
    - Single point mass (``d_max == 0``): returns ``0.0`` (handled by point-mass rail escape).
    - Continuous 1D/2D/nD manifolds, ``grid100`` (``sigma/Delta = 0.03``), and broad mixtures: ``0.0``.
    - Finite zero-noise or tight noisy discrete codebooks (any ``K >= 2``, ``D >= 1``, any scale): ``~1.0``.
    """
    b_r = int(y_flat.shape[0])
    if b_r < 4:
        return 0.0
    yf = y_flat.detach().reshape(b_r, -1).float()
    dim = int(yf.shape[1])
    yf_c = yf - yf.mean(dim=0, keepdim=True)
    y_scale = yf_c.abs().amax()
    if not bool(torch.isfinite(y_scale)) or float(y_scale.item()) <= 1e-30:
        return 0.0

    # Deterministic strided subsample caps pairwise O(B^2) work at B_eff <= 256.
    if b_r > 256:
        idx_sub = (torch.arange(256, device=yf.device, dtype=torch.long) * b_r) // 256
        y_u = yf_c[idx_sub] / y_scale
        b_eff = 256
    else:
        y_u = yf_c / y_scale
        b_eff = b_r

    d_yy = torch.cdist(y_u, y_u, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    d_yy.fill_diagonal_(0.0)
    d_max = float(d_yy.amax().item())
    if not math.isfinite(d_max) or d_max <= 1e-12:
        return 0.0
    d_yy_inf = d_yy.clone()
    d_yy_inf.fill_diagonal_(float("inf"))
    nn_y = d_yy_inf.amin(dim=-1)

    # Branch 1: exact / machine-precision discrete atoms (sigma == 0, e.g. ByteGAN / five_word).
    tau_0 = 1e-6 * d_max
    is_atom = nn_y <= tau_0
    n_atom = int(is_atom.sum().item())
    if n_atom >= 2:
        pos_nn = nn_y[~is_atom]
        d_pos = float(torch.median(pos_nn).item()) if pos_nn.numel() > 0 else d_max
        tau_mid = min(0.20 * d_pos, 0.05 * d_max)
        if tau_mid >= 10.0 * tau_0:
            n_near = int(((nn_y > tau_0) & (nn_y <= tau_mid)).sum().item())
            purity = float(n_atom) / float(n_atom + n_near)
            u = min(1.0, max(0.0, (purity - 0.70) / 0.20))
            g_exact = float(u * u * (3.0 - 2.0 * u))
            if g_exact > 0.0:
                return g_exact

    # Branch 2: noise-tolerant separated discrete clusters via empty pairwise log-gap & sigma/Delta.
    if b_eff >= 16:
        iu, ju = torch.triu_indices(b_eff, b_eff, offset=1, device=y_u.device)
        pd = d_yy[iu, ju].sort().values
        min_in = max(4, b_eff // 8)
        lo_idx = torch.nonzero((pd > tau_0) & (pd < 0.20 * d_max), as_tuple=False).squeeze(-1)
        if lo_idx.numel() > 0:
            valid_i = lo_idx[(lo_idx >= (min_in - 1)) & (lo_idx < (pd.numel() - 1))]
            if valid_i.numel() > 0:
                ratios = pd[valid_i + 1] / pd[valid_i].clamp_min(1e-12)
                best_r, best_pos = ratios.max(dim=0)
                gap_ratio = float(best_r.item())
                if gap_ratio >= 2.2:
                    split_i = int(valid_i[int(best_pos.item())].item())
                    # Require high effective codebook cardinality (Simpson K_eff = 1 / intra_frac >= 8)
                    # on noisy (sigma > 0) clusters so low-K continuous Gaussian mixtures (e.g. 4-mode
                    # mixtures with small sigma/Delta) never false-trigger discrete-atomic locking;
                    # low-K exact zero-noise tasks (e.g. five_word) are handled by Branch 1 above.
                    intra_frac = float(split_i + 1) / float(max(1, int(pd.numel())))
                    k_eff = 1.0 / max(intra_frac, 1e-6)
                    a_cut = float(pd[split_i].item())
                    b_cut = float(pd[split_i + 1].item())
                    if k_eff >= 7.0 and int((nn_y <= a_cut).sum().item()) >= max(4, b_eff // 4):
                        far_mat = torch.where(d_yy >= b_cut, d_yy, float("inf"))
                        inter_nn = far_mat.amin(dim=-1)
                        inter_fin = inter_nn[torch.isfinite(inter_nn)]
                        if inter_fin.numel() >= b_eff // 2:
                            delta_est = float(torch.quantile(inter_fin, 0.25).item())
                            if delta_est > 3.0 * a_cut:
                                in_pairs = pd[: split_i + 1]
                                sig_coord = math.sqrt(float(in_pairs.square().mean().item()) / (2.0 * float(dim)))
                                ratio_sd = sig_coord / max(delta_est, 1e-12)
                                u_sd = min(1.0, max(0.0, (0.0265 - ratio_sd) / 0.0035))
                                u_gap = min(1.0, max(0.0, (gap_ratio - 2.2) / 1.3))
                                u_k = min(1.0, max(0.0, (k_eff - 7.0) / 5.0))
                                s_sd = u_sd * u_sd * (3.0 - 2.0 * u_sd)
                                s_gap = u_gap * u_gap * (3.0 - 2.0 * u_gap)
                                s_k = u_k * u_k * (3.0 - 2.0 * u_k)
                                g_noisy = float(s_sd * s_gap * s_k)
                                if g_noisy > 0.0:
                                    return g_noisy

    # Branch 3: persistent sharp/tight-cluster fallback for rare zero-duplicate batches when K >> B.
    if (
        charts is not None
        and radii is not None
        and charts.ndim == 2
        and charts.shape[0] >= 3
        and charts.shape[1] == dim
        and radii.ndim == 1
        and radii.shape[0] == charts.shape[0]
    ):
        cf = charts.detach().float().to(device=yf.device)
        rf = radii.detach().float().to(device=yf.device)
        cf_c = cf - cf.mean(dim=0, keepdim=True)
        sc = torch.maximum(y_scale, cf_c.abs().amax()).clamp_min(1e-30)
        d_cc = torch.cdist(cf / sc, cf / sc, p=2.0) * sc
        d_cc.fill_diagonal_(float("inf"))
        nn_cc = d_cc.amin(dim=1).clamp_min(1e-12 * sc)
        rad_sd = float(torch.median(rf / (math.sqrt(float(dim)) * nn_cc)).item())
        if rad_sd < 0.025:
            d_yc = torch.cdist(yf / sc, cf / sc, p=2.0) * sc
            q75_yc = float(torch.quantile(d_yc.amin(dim=1), 0.75).item())
            med_nn = float(torch.median(nn_cc).item())
            if q75_yc < 0.08 * med_nn:
                u_r = min(1.0, max(0.0, (0.025 - rad_sd) / 0.004))
                return float(u_r * u_r * (3.0 - 2.0 * u_r))
    return 0.0


@torch.no_grad()
def compute_atomic_quotas(
    num_particles: int,
    w_real: torch.Tensor,
    *,
    chart_affinity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deterministic integer particle quotas summing to ``num_particles`` (CPU ``int64[K]``).

    When ``M >= K``, guarantees >= 1 particle per discovered chart (exact 1-to-1 unit quotas
    at ``M == K``) and apportions surplus ``M - K`` proportionally to ``relu(M * w_k - 1)`` via
    largest remainders. When ``M < K``, applies Hamilton apportionment with proximity-aware
    and Weyl golden-ratio tie-breaking (eliminating ``-1e-7 * arange(k)`` high-index eviction).
    """
    m = int(num_particles)
    k = int(w_real.shape[0])
    if m < 1 or k < 1:
        raise ValueError("num_particles and num_charts must be positive")
    w = w_real.detach().float().cpu().clamp_min(0.0)
    w_sum = float(w.sum().item())
    w = torch.full((k,), 1.0 / float(k), dtype=torch.float32) if w_sum <= 1e-12 else (w / w_sum)

    weyl = _weyl_tie(k, device="cpu")
    aff = None
    if chart_affinity is not None and chart_affinity.numel() == k:
        aff = chart_affinity.detach().float().cpu().reshape(k).clamp(0.0, 1.0)

    if m >= k:
        quotas = torch.ones(k, dtype=torch.long)
        rem = m - k
        if rem > 0:
            extra = (w * float(m) - 1.0).clamp_min(0.0)
            esum = float(extra.sum().item())
            extra_scaled = (w * float(rem)) if esum <= 1e-12 else (extra * (float(rem) / esum))
            add_floor = torch.floor(extra_scaled).to(dtype=torch.long)
            quotas = quotas + add_floor
            rem2 = int(m - int(quotas.sum().item()))
            if rem2 > 0:
                score = (extra_scaled - add_floor.float()) + 1e-7 * weyl
                if aff is not None:
                    score = score + 1e-4 * aff
                top_k = torch.topk(score, k=min(rem2, k), largest=True).indices
                quotas[top_k] += 1
        return quotas

    raw = w * float(m)
    quotas = torch.floor(raw).to(dtype=torch.long)
    rem = int(m - int(quotas.sum().item()))
    if rem > 0:
        score = (raw - quotas.float()) + 1e-7 * weyl
        if aff is not None:
            score = score + 0.08 * aff
        top_k = torch.topk(score, k=min(rem, k), largest=True).indices
        quotas[top_k] += 1
    return quotas


@torch.no_grad()
def update_atomic_ledger(
    x_feat_det: torch.Tensor,
    rows: torch.Tensor,
    num_particles: int,
    charts: torch.Tensor,
    w_real: torch.Tensor,
    prev_row_chart: Optional[torch.Tensor],
    prev_row_dist: Optional[torch.Tensor],
    prev_row_target: Optional[torch.Tensor],
    *,
    allow_update: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Maintain persistent 1-stays / surplus-moves particle-to-chart bijection across minibatches.

    Returns ``(batch_target_idx, nn_cc, next_row_chart, next_row_dist, next_row_target, r_atomic)``.
    """
    dev = x_feat_det.device
    xf = x_feat_det.detach().float()
    cf = charts.detach().float().to(device=dev)
    k = int(cf.shape[0])
    m = int(num_particles)

    scale = torch.maximum(xf.abs().amax(), cf.abs().amax()).clamp_min(1e-12)
    d_xc = torch.cdist(xf / scale, cf / scale, p=2.0) * scale
    d_near, near_k = d_xc.min(dim=1)

    if k >= 2:
        d_cc = torch.cdist(cf / scale, cf / scale, p=2.0) * scale
        d_cc.fill_diagonal_(float("inf"))
        nn_cc = d_cc.amin(dim=1).clamp_min(1e-12 * scale)
    else:
        nn_cc = torch.ones((k,), device=dev, dtype=torch.float32) * scale

    if (
        prev_row_chart is not None
        and prev_row_dist is not None
        and prev_row_target is not None
        and int(prev_row_chart.shape[0]) == m
        and int(prev_row_dist.shape[0]) == m
        and int(prev_row_target.shape[0]) == m
    ):
        row_chart_cpu = prev_row_chart.detach().to(device="cpu", dtype=torch.long).clone()
        row_dist_cpu = prev_row_dist.detach().to(device="cpu", dtype=torch.float32).clone()
        row_target_cpu = prev_row_target.detach().to(device="cpu", dtype=torch.long).clone()
    else:
        row_chart_cpu = torch.full((m,), -1, dtype=torch.long)
        row_dist_cpu = torch.full((m,), float("inf"), dtype=torch.float32)
        row_target_cpu = torch.full((m,), -1, dtype=torch.long)

    bad_c = (row_chart_cpu < 0) | (row_chart_cpu >= k)
    if bool(bad_c.any()):
        row_chart_cpu[bad_c] = -1
        row_dist_cpu[bad_c] = float("inf")
    bad_t = (row_target_cpu < 0) | (row_target_cpu >= k)
    if bool(bad_t.any()):
        row_target_cpu[bad_t] = -1

    rows_cpu = rows.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if not allow_update:
        tgt_list = row_target_cpu[rows_cpu]
        near_cpu = near_k.detach().cpu()
        eff_tgt = torch.where(tgt_list >= 0, tgt_list, near_cpu).to(device=dev)
        return (
            eff_tgt,
            nn_cc,
            row_chart_cpu.to(device=dev),
            row_dist_cpu.to(device=dev),
            row_target_cpu.to(device=dev),
            0.0,
        )

    d_xc_cpu = d_xc.detach().cpu()
    d_near_cpu = d_near.detach().cpu()
    near_k_cpu = near_k.detach().cpu()

    # 1. Record latest observed position for each particle in this minibatch (first occurrence).
    seen_first: Dict[int, int] = {}
    rows_list = rows_cpu.tolist()
    near_list = near_k_cpu.tolist()
    dnear_list = d_near_cpu.tolist()
    for i_b, r_id in enumerate(rows_list):
        if 0 <= r_id < m and r_id not in seen_first:
            seen_first[r_id] = i_b
            row_chart_cpu[r_id] = int(near_list[i_b])
            row_dist_cpu[r_id] = float(dnear_list[i_b])

    # 2. Compute exact proximity-aware, index-unbiased quotas summing to M.
    chart_aff = torch.exp(-(d_xc.amin(dim=0) / nn_cc.clamp_min(1e-12))).detach().cpu()
    quotas = compute_atomic_quotas(m, w_real, chart_affinity=chart_aff)
    q_list = quotas.tolist()

    # 3. Enforce quota ceilings (closest stay, surplus released deterministically).
    rc_list = row_chart_cpu.tolist()
    rd_list = row_dist_cpu.tolist()
    rt_list = row_target_cpu.tolist()

    chart_members: list[list[int]] = [[] for _ in range(k)]
    for p_id, t_id in enumerate(rt_list):
        if 0 <= t_id < k:
            chart_members[t_id].append(p_id)

    occ = [len(mem) for mem in chart_members]
    for c_id in range(k):
        cap = q_list[c_id]
        if occ[c_id] > cap:
            members = chart_members[c_id]
            members.sort(
                key=lambda pid: (
                    0 if rc_list[pid] == c_id else 1,
                    rd_list[pid] if math.isfinite(rd_list[pid]) else 1e30,
                    pid,
                )
            )
            for pid in members[cap:]:
                rt_list[pid] = -1
            occ[c_id] = cap

    # 4. Assign unassigned particles present in this batch to their nearest deficit chart.
    unassigned_batch = [(i_b, r_id) for r_id, i_b in seen_first.items() if rt_list[r_id] < 0]
    if unassigned_batch:
        unassigned_batch.sort(key=lambda pair: (dnear_list[pair[0]], pair[1]))
        avail_mask = torch.tensor([occ[c] < q_list[c] for c in range(k)], dtype=torch.bool)
        tie_k = (1e-6 * float(scale.item())) * _weyl_tie(k, device="cpu")
        for i_b, r_id in unassigned_batch:
            if not bool(avail_mask.any()):
                k_best = int(near_list[i_b])
                rt_list[r_id] = k_best
                occ[k_best] += 1
                continue
            scores = (d_xc_cpu[i_b] + tie_k).masked_fill(~avail_mask, float("inf"))
            k_best = int(scores.argmin().item())
            rt_list[r_id] = k_best
            occ[k_best] += 1
            if occ[k_best] >= q_list[k_best]:
                avail_mask[k_best] = False

    row_target_cpu = torch.tensor(rt_list, dtype=torch.long)
    batch_target = row_target_cpu[rows_cpu].to(device=dev)

    # 5. Scale-invariant discrete atomic thermal residual.
    nn_cc_cpu = nn_cc.detach().cpu()
    valid_t = row_target_cpu >= 0
    safe_t = row_target_cpu.clamp(0, k - 1)
    batch_tgt_d = d_xc.gather(1, batch_target.unsqueeze(1)).squeeze(1)
    batch_rel = (batch_tgt_d / nn_cc[batch_target].clamp_min(1e-12)).clamp_max(10.0)
    r_crisp = 1.6 * float(batch_rel.mean().item())

    landed = valid_t & (row_chart_cpu == safe_t) & (row_dist_cpu <= (0.09 * nn_cc_cpu[safe_t]))
    n_unsettled = int(m - int(landed.sum().item()))
    target_occ = (
        torch.bincount(row_target_cpu[valid_t], minlength=k)
        if bool(valid_t.any())
        else torch.zeros(k, dtype=torch.long)
    )
    any_deficit = bool((target_occ < quotas).any())
    r_cov = (0.18 + 0.12 * (float(max(n_unsettled, 1)) / float(max(1, m)))) if (n_unsettled > 0 or any_deficit) else 0.0
    r_atomic = max(r_crisp, r_cov)

    return (
        batch_target,
        nn_cc,
        row_chart_cpu.to(device=dev),
        row_dist_cpu.to(device=dev),
        row_target_cpu.to(device=dev),
        float(r_atomic),
    )


@torch.no_grad()
def match_inverse_atomic_targets(
    real_x_det: torch.Tensor,
    real_z_det: torch.Tensor,
    charts: torch.Tensor,
    row_target: torch.Tensor,
    table_det: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match each real atom ``real_x[j]`` to its assigned latent table code ``table[m*(k)]``.

    Returns ``(inv_z_target, has_match_mask, ell_z, inv_nn_margin)`` with zero extra forwards.
    """
    dev = real_x_det.device
    b_r = int(real_x_det.shape[0])
    k = int(charts.shape[0])
    m = int(table_det.shape[0])

    rx = real_x_det.float()
    cf = charts.detach().float().to(device=dev)
    sc = torch.maximum(rx.abs().amax(), cf.abs().amax()).clamp_min(1e-12)
    real_k = torch.cdist(rx / sc, cf / sc, p=2.0).argmin(dim=1)

    tab_f = table_det.detach().float().to(device=dev)
    sc_z = tab_f.abs().amax().clamp_min(1e-12)
    d_zz = torch.cdist(tab_f / sc_z, tab_f / sc_z, p=2.0) * sc_z
    if m >= 2:
        d_zz.fill_diagonal_(float("inf"))
        nn_z = d_zz.amin(dim=1).clamp_min(1e-6 * sc_z)
        ell_z = (0.25 * torch.median(nn_z)).clamp_min(1e-6 * sc_z)
    else:
        nn_z = torch.ones((m,), device=dev, dtype=torch.float32) * sc_z
        ell_z = tab_f.new_tensor(0.25, dtype=torch.float32) * sc_z

    rt_cpu = row_target.detach().to(device="cpu", dtype=torch.long).tolist()
    members_by_chart: list[list[int]] = [[] for _ in range(k)]
    for p_id, c_id in enumerate(rt_cpu):
        if 0 <= c_id < k:
            members_by_chart[c_id].append(p_id)

    rz_cpu = real_z_det.detach().float().cpu()
    tab_cpu = tab_f.cpu()
    real_k_list = real_k.detach().cpu().tolist()

    inv_idx_list = [0] * b_r
    has_list = [False] * b_r
    for j, c_id in enumerate(real_k_list):
        mem = members_by_chart[c_id]
        if not mem:
            continue
        has_list[j] = True
        if len(mem) == 1:
            inv_idx_list[j] = mem[0]
        else:
            mem_t = torch.tensor(mem, dtype=torch.long)
            dists_m = (tab_cpu[mem_t] - rz_cpu[j].unsqueeze(0)).square().sum(dim=-1)
            inv_idx_list[j] = mem[int(dists_m.argmin().item())]

    inv_idx = torch.tensor(inv_idx_list, device=dev, dtype=torch.long)
    has_mask = torch.tensor(has_list, device=dev, dtype=torch.bool)
    inv_z_target = table_det.detach()[inv_idx]
    inv_margin = nn_z[inv_idx]
    return inv_z_target, has_mask, ell_z.to(dtype=table_det.dtype), inv_margin


def split_joint_table_pair(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    table: Optional[torch.Tensor],
    rows: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Verify and split joint ``(x, z)`` BiGAN tensors using the exact sampled table codes."""
    if table is None or rows is None:
        return None
    if not isinstance(table, torch.Tensor) or not isinstance(rows, torch.Tensor):
        return None
    if table.ndim != 2 or rows.ndim != 1:
        return None
    b_f = int(x_fake.shape[0])
    b_r = int(y_real.shape[0])
    if int(rows.shape[0]) != b_f or b_f < 4 or b_r < 4:
        return None
    z_dim = int(table.shape[1])
    x_f_flat = x_fake.reshape(b_f, -1)
    x_r_flat = y_real.reshape(b_r, -1)
    d_joint = int(x_f_flat.shape[1])
    if d_joint <= z_dim or int(x_r_flat.shape[1]) != d_joint:
        return None
    if rows.device != table.device:
        rows = rows.to(device=table.device)
    if bool((rows < 0).any()) or bool((rows >= int(table.shape[0])).any()):
        return None
    z_rows_det = table.detach()[rows].float()
    if z_rows_det.device != x_f_flat.device:
        z_rows_det = z_rows_det.to(device=x_f_flat.device)

    tol = max(1e-5, 1e-5 * float(z_rows_det.abs().amax().item()))
    suffix_err = float((x_f_flat[:, -z_dim:].detach().float() - z_rows_det).abs().amax().item())
    if suffix_err <= tol:
        return (
            x_f_flat[:, :-z_dim],
            x_f_flat[:, -z_dim:],
            x_r_flat[:, :-z_dim],
            x_r_flat[:, -z_dim:],
        )
    prefix_err = float((x_f_flat[:, :z_dim].detach().float() - z_rows_det).abs().amax().item())
    if prefix_err <= tol:
        return (
            x_f_flat[:, z_dim:],
            x_f_flat[:, :z_dim],
            x_r_flat[:, z_dim:],
            x_r_flat[:, :z_dim],
        )
    return None


__all__ = [
    "compute_atomic_quotas",
    "compute_discrete_atomic_gate",
    "match_inverse_atomic_targets",
    "split_joint_table_pair",
    "update_atomic_ledger",
]
