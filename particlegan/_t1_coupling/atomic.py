"""Scale-invariant discrete-atomic support detection, 1-stays / surplus-moves bijection ledger,
and joint BiGAN inverse codebook contraction for ``self_coupling='feydy_atomic'``.

All primitives are strictly recipe-owned, task-agnostic (zero hardcoded cardinalities,
dimensions, or coordinate alphabets), clock-free, zero-RNG, CUDA-deterministic, and
free of ``autograd.grad`` / SVD / ``eigh`` / ``pinv``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


@torch.no_grad()
def compute_discrete_atomic_gate(
    y_flat: torch.Tensor,
    *,
    charts: Optional[torch.Tensor] = None,
    radii: Optional[torch.Tensor] = None,
) -> float:
    """Return smooth indicator ``g_atom in [0, 1]`` for finite discrete atomic support.

    Distinguishes discrete atomic targets (exact/machine-precision duplicate draws separated
    by an empty annular gap from positive inter-atom spacings) from continuous manifolds and
    1D continuous densities (whose nearest-neighbor distances are dense right above zero):
    - Single point mass (``d_max == 0``): returns ``0.0`` (handled by point-mass rail escape).
    - Continuous 1D/2D/nD manifolds & noisy mixtures: returns ``0.0``.
    - Finite discrete atomic distributions (any ``K >= 2`` in any ``R^D``): returns ``1.0``.
    """
    b_r = int(y_flat.shape[0])
    if b_r < 4:
        return 0.0
    yf = y_flat.detach().reshape(b_r, -1).float()
    y_scale = yf.abs().amax().clamp_min(1.0)
    y_u = yf / y_scale
    d_yy = torch.cdist(y_u, y_u, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    d_yy.fill_diagonal_(0.0)
    d_max = float(d_yy.amax().item())
    if not math.isfinite(d_max) or d_max <= 1e-12:
        return 0.0
    d_yy.fill_diagonal_(float("inf"))
    nn_y = d_yy.amin(dim=-1)
    tau_0 = 1e-6 * d_max
    is_atom = nn_y <= tau_0
    n_atom = int(is_atom.sum().item())
    if n_atom >= 2:
        pos_nn = nn_y[~is_atom]
        if pos_nn.numel() > 0:
            d_pos = float(torch.median(pos_nn).item())
            tau_mid = min(0.20 * d_pos, 0.05 * d_max)
        else:
            tau_mid = 0.05 * d_max
        if tau_mid < 10.0 * tau_0:
            return 0.0
        n_near = int(((nn_y > tau_0) & (nn_y <= tau_mid)).sum().item())
        purity = float(n_atom) / float(n_atom + n_near)
        u = min(1.0, max(0.0, (purity - 0.70) / 0.20))
        return float(u * u * (3.0 - 2.0 * u))

    # Persistent sharp-atom fallback for rare zero-duplicate batches when K >> B:
    if (
        charts is not None
        and radii is not None
        and charts.ndim == 2
        and charts.shape[0] >= 3
        and charts.shape[1] == yf.shape[1]
        and radii.ndim == 1
        and radii.shape[0] == charts.shape[0]
    ):
        cf = charts.detach().float().to(device=yf.device)
        rf = radii.detach().float().to(device=yf.device)
        sc = torch.maximum(y_scale, cf.abs().amax()).clamp_min(1.0)
        d_cc = torch.cdist(cf / sc, cf / sc, p=2.0) * sc
        d_cc.fill_diagonal_(float("inf"))
        nn_cc = d_cc.amin(dim=1).clamp_min(1e-6)
        rad_ratio = float(torch.median(rf / nn_cc).item())
        if rad_ratio < 1e-3:
            d_yc = torch.cdist(yf / sc, cf / sc, p=2.0) * sc
            q75_yc = float(torch.quantile(d_yc.amin(dim=1), 0.75).item())
            med_nn = float(torch.median(nn_cc).item())
            if q75_yc < 1e-3 * med_nn:
                return 1.0
    return 0.0


@torch.no_grad()
def compute_atomic_quotas(num_particles: int, w_real: torch.Tensor) -> torch.Tensor:
    """Deterministic integer particle quotas summing to ``num_particles`` (CPU ``int64[K]``).

    When ``M >= K``, allocates at least one particle per discovered chart first (guaranteeing
    exact 1-to-1 unit quotas whenever ``M == K``, immune to minibatch EMA count jitter) and
    apportions any surplus ``M - K`` proportionally to ``relu(M * w_k - 1)`` via largest remainders.
    When ``M < K``, applies deterministic largest-remainder Hamilton apportionment.
    """
    m = int(num_particles)
    k = int(w_real.shape[0])
    if m < 1 or k < 1:
        raise ValueError("num_particles and num_charts must be positive")
    w = w_real.detach().float().cpu().clamp_min(0.0)
    w_sum = float(w.sum().item())
    if w_sum <= 1e-12:
        w = torch.full((k,), 1.0 / float(k), dtype=torch.float32)
    else:
        w = w / w_sum

    tie = 1e-7 * torch.arange(k, dtype=torch.float32)
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
                frac = (extra_scaled - add_floor.float()) - tie
                top_k = torch.topk(frac, k=min(rem2, k), largest=True).indices
                quotas[top_k] += 1
        return quotas

    raw = w * float(m)
    quotas = torch.floor(raw).to(dtype=torch.long)
    rem = int(m - int(quotas.sum().item()))
    if rem > 0:
        frac = (raw - quotas.float()) - tie
        top_k = torch.topk(frac, k=min(rem, k), largest=True).indices
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

    scale = torch.maximum(xf.abs().amax(), cf.abs().amax()).clamp_min(1.0)
    d_xc = torch.cdist(xf / scale, cf / scale, p=2.0) * scale
    d_near, near_k = d_xc.min(dim=1)

    if k >= 2:
        d_cc = torch.cdist(cf / scale, cf / scale, p=2.0) * scale
        d_cc.fill_diagonal_(float("inf"))
        nn_cc = d_cc.amin(dim=1).clamp_min(1e-6)
    else:
        nn_cc = torch.ones((k,), device=dev, dtype=torch.float32)

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

    # Invalidate out-of-range chart ids (e.g. after a chart shock reset).
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

    # 2. Compute exact quotas summing to M.
    quotas = compute_atomic_quotas(m, w_real)
    q_list = quotas.tolist()

    # 3. Enforce quota ceilings (1 stays, surplus released deterministically).
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
    unassigned_batch = [
        (i_b, r_id)
        for r_id, i_b in seen_first.items()
        if rt_list[r_id] < 0
    ]
    if unassigned_batch:
        unassigned_batch.sort(key=lambda pair: (dnear_list[pair[0]], pair[1]))
        avail_mask = torch.tensor([occ[c] < q_list[c] for c in range(k)], dtype=torch.bool)
        tie_k = 1e-6 * torch.arange(k, dtype=torch.float32)
        for i_b, r_id in unassigned_batch:
            if not bool(avail_mask.any()):
                # Should never occur by pigeonhole (sum(q) == M), fail closed to nearest chart.
                k_best = int(near_list[i_b])
                rt_list[r_id] = k_best
                occ[k_best] += 1
                continue
            scores = d_xc_cpu[i_b] + tie_k
            scores = scores.masked_fill(~avail_mask, float("inf"))
            k_best = int(scores.argmin().item())
            rt_list[r_id] = k_best
            occ[k_best] += 1
            if occ[k_best] >= q_list[k_best]:
                avail_mask[k_best] = False

    row_target_cpu = torch.tensor(rt_list, dtype=torch.long)
    batch_target_cpu = row_target_cpu[rows_cpu]
    batch_target = batch_target_cpu.to(device=dev)

    # 5. Scale-invariant discrete atomic thermal residual:
    #    Keep thermal active (r_atomic >= 0.18 > thermal_tau=0.084) while ANY particle is
    #    unassigned or has not crisply landed within 0.09 * nn_cc of its assigned atom;
    #    once all M particles land cleanly 1-to-1, cool smoothly by crispness error.
    nn_cc_cpu = nn_cc.detach().cpu()
    valid_t = row_target_cpu >= 0
    safe_t = row_target_cpu.clamp(0, k - 1)
    # Also check batch-level crispness on currently assigned targets.
    batch_tgt_d = d_xc.gather(1, batch_target.unsqueeze(1)).squeeze(1)
    batch_rel = (batch_tgt_d / nn_cc[batch_target].clamp_min(1e-6)).clamp_max(10.0)
    r_crisp = 1.6 * float(batch_rel.mean().item())

    # Update row_dist for particles already sitting at their target chart vs away from target:
    landed = (
        valid_t
        & (row_chart_cpu == safe_t)
        & (row_dist_cpu <= (0.09 * nn_cc_cpu[safe_t]))
    )
    n_unsettled = int(m - int(landed.sum().item()))
    # Also verify no chart is currently under-occupied relative to quotas:
    target_occ = torch.bincount(row_target_cpu[valid_t], minlength=k) if bool(valid_t.any()) else torch.zeros(k, dtype=torch.long)
    any_deficit = bool((target_occ < quotas).any())
    if n_unsettled > 0 or any_deficit:
        r_cov = 0.18 + 0.12 * (float(max(n_unsettled, 1)) / float(max(1, m)))
    else:
        r_cov = 0.0
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
    sc = torch.maximum(rx.abs().amax(), cf.abs().amax()).clamp_min(1.0)
    real_k = torch.cdist(rx / sc, cf / sc, p=2.0).argmin(dim=1)

    tab_f = table_det.detach().float().to(device=dev)
    sc_z = tab_f.abs().amax().clamp_min(1.0)
    d_zz = torch.cdist(tab_f / sc_z, tab_f / sc_z, p=2.0) * sc_z
    if m >= 2:
        d_zz.fill_diagonal_(float("inf"))
        nn_z = d_zz.amin(dim=1).clamp_min(1e-4)
        ell_z = (0.25 * torch.median(nn_z)).clamp_min(1e-4)
    else:
        nn_z = torch.ones((m,), device=dev, dtype=torch.float32)
        ell_z = tab_f.new_tensor(0.25, dtype=torch.float32)

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
    """Verify and split joint ``(x, z)`` BiGAN tensors using the exact sampled table codes.

    Returns ``(fake_x, fake_z, real_x, real_z)`` preserving full autograd graphs, or ``None``
    when the inputs are not an exact zero-sigma joint ``(x, z)`` / ``(z, x)`` table pair.
    """
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

    suffix_err = float((x_f_flat[:, -z_dim:].detach().float() - z_rows_det).abs().amax().item())
    if suffix_err <= 1e-5:
        return (
            x_f_flat[:, :-z_dim],
            x_f_flat[:, -z_dim:],
            x_r_flat[:, :-z_dim],
            x_r_flat[:, -z_dim:],
        )
    prefix_err = float((x_f_flat[:, :z_dim].detach().float() - z_rows_det).abs().amax().item())
    if prefix_err <= 1e-5:
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
