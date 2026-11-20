"""First-principles Riemannian Gauge-Deconflicted Transport (RGDT) metric & transport operators.

Implements three clock-free, RNG-free operators addressing Scale-Mass Gauge
Non-Identifiability and Unbalanced Horizon Starvation:
1. Symmetric Target-Induced Conformal Riemannian Horizon ``ell_Y(u, v) = sqrt(ell_Y(u) ell_Y(v))``
   applied identically to cross-coupling ``(X, Y)`` and Feydy self-coupling ``(X, X)``.
2. Orphan-Gated Bipartite Monge Deficit-to-Surplus Macro-Transport coupling surplus
   fake row-slack to genuinely uncovered real orphan columns via sharp Sinkhorn + Weyl
   inverse-CDF 1-to-1 assignment (breaking equidistant symmetry saddles without Poisson shot-noise drift).
3. Exchangeability-Invariant Pooled-Tangent Residual Conditional Discrepancy removing
   base-manifold conditional drift ``m(c_i) - m(c_k)`` via ``(X_det + Y_det)/2`` kernel-linear
   regression with exact zero finite-sample leverage bias at ``q_theta = p``.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

_PHI1 = 0.6180339887498949
_PHI2 = 0.7548776662466927


def sq_euclidean_f64(x_flat: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
    """Differentiable float64 squared Euclidean distance matrix ``[B_x, B_y]``."""
    x64 = x_flat.double()
    y64 = y_flat.double()
    sq_raw = (
        x64.square().sum(dim=-1, keepdim=True)
        + y64.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (x64 @ y64.T)
    )
    return sq_raw + (sq_raw.detach().clamp_min(0.0) - sq_raw.detach())


def born_infeld_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    ell: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Overflow-safe, cancellation-free Born-Infeld pairwise costs ``(c_phys, c_hat)``.

    ``ell`` may be scalar ``[]``, vector ``[B_x, 1]``, or symmetric conformal matrix ``[B_x, B_y]``.
    """
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    dim = max(1, x_flat.shape[-1])
    sq64 = sq_euclidean_f64(x_flat, y_flat)
    ell64 = ell.detach().clamp_min(1e-6).double()
    rad = torch.sqrt(ell64.square() + sq64)
    val = sq64 / (rad + ell64)
    c_phys64 = (rad - ell64) + (val - (rad - ell64)).detach()

    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
    c_phys = c_phys64.clamp_max(fmax).to(x_flat.dtype)
    with torch.no_grad():
        ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
        c_hat = (val.detach() / ell64).clamp_max(ratio_max).to(x_flat.dtype)
    return c_phys, c_hat


def born_infeld_aligned(x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    """Row-wise aligned physical Born-Infeld chord cost ``[N]``."""
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    dim = max(1, x_flat.shape[-1])
    ell64 = ell.detach().clamp_min(1e-6).double()
    sq64 = (x_flat.double() - y_flat.double()).square().sum(dim=-1)
    rad = torch.sqrt(ell64.square() + sq64)
    val = sq64 / (rad + ell64)
    c64 = (rad - ell64) + (val - (rad - ell64)).detach()
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
    return c64.clamp_max(fmax).to(x_flat.dtype)


@torch.no_grad()
def born_infeld_c_hat_detached(
    x: torch.Tensor,
    y: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Fast pure-float32 overflow-safe detached dimensionless cost ``c_hat``."""
    x_flat = x.detach().reshape(x.shape[0], -1).float()
    y_flat = y.detach().reshape(y.shape[0], -1).float()
    dim = max(1, x_flat.shape[-1])
    ell_f = ell.detach().float().clamp_min(1e-6)
    scale = torch.maximum(
        torch.maximum(x_flat.abs().amax(), y_flat.abs().amax()),
        ell_f.amax(),
    ).clamp_min(1.0)
    dist_scaled = torch.cdist(x_flat / scale, y_flat / scale, p=2.0)
    inv_u = ell_f / scale
    rad = torch.sqrt(inv_u.square() + dist_scaled.square())
    val_hat = dist_scaled.square() / ((rad + inv_u) * inv_u).clamp_min(1e-38)
    ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
    return val_hat.clamp_max(ratio_max).to(x.dtype)


@torch.no_grad()
def batch_ell_quantile(
    y_real: torch.Tensor,
    nn_quantile: float = 0.25,
) -> torch.Tensor:
    """Zero-sync scale-equivariant nearest-neighbor horizon with 1D Weibull AKT scaling."""
    b_y = y_real.shape[0]
    if b_y <= 1:
        return y_real.new_tensor(1.0, dtype=torch.float32)
    y_flat = y_real.detach().reshape(b_y, -1).float()
    dim = max(1, y_flat.shape[-1])
    akt_exp = max(0.0, (1.0 / float(dim)) - 0.5)
    if akt_exp > 0.0:
        q_rate = max(1e-6, -math.log(max(1e-6, 1.0 - float(nn_quantile))))
        akt_factor = ((2.0 * math.e * float(b_y)) / q_rate) ** akt_exp
    else:
        akt_factor = 1.0

    scale = y_flat.abs().amax().clamp_min(1.0)
    y_unit = y_flat / scale
    dist_unit = torch.cdist(y_unit, y_unit, p=2.0)
    dist_unit.fill_diagonal_(0.0)
    d_max = dist_unit.amax()
    thresh = 1e-5 * d_max
    pos_min = torch.where(dist_unit > thresh, dist_unit, float("inf")).amin()

    dist_unit.fill_diagonal_(float("inf"))
    k_nn = min(b_y - 1, max(1, b_y // 200))
    if k_nn == 1:
        nn_dist = dist_unit.amin(dim=-1)
    else:
        nn_dist = torch.topk(dist_unit, k=k_nn, dim=-1, largest=False).values[:, -1]
    q_raw = torch.quantile(nn_dist, nn_quantile)
    rms_unit = y_unit.square().mean().sqrt()
    tie_base = torch.where(torch.isfinite(pos_min), 0.15 * pos_min, 0.05 * d_max)
    raw_unit = torch.where(q_raw <= thresh, tie_base, q_raw)
    nonzero_val = scale * akt_factor * raw_unit + 1e-6
    dirac_val = 0.05 * torch.clamp_min(scale * rms_unit, 1.0) + 1e-6
    return torch.where(d_max == 0.0, dirac_val, nonzero_val)


@torch.no_grad()
def update_scalar_ell(
    y_real: torch.Tensor,
    *,
    record: Any = None,
    fallback_ell: Optional[torch.Tensor] = None,
    beta_ell: float = 0.1,
    nn_quantile: float = 0.25,
    allow_update: bool = True,
) -> torch.Tensor:
    """Update instance-local scalar horizon EMA without host-device synchronization."""
    device = y_real.device
    if record is not None and hasattr(record, "rgdt_ell_initialized"):
        init_cur = bool(record.rgdt_ell_initialized)
        ell_t = getattr(record, "_rgdt_ell_t", None)
        if ell_t is None or ell_t.device != device:
            val = float(record.rgdt_ell_ema)
            ell_t = torch.tensor(val, dtype=torch.float32, device=device)
            record._rgdt_ell_t = ell_t
    else:
        init_cur = fallback_ell is not None and float(fallback_ell.item()) > 0.0
        ell_t = (
            fallback_ell.to(device=device, dtype=torch.float32)
            if fallback_ell is not None
            else torch.tensor(0.0, dtype=torch.float32, device=device)
        )

    if not allow_update:
        if init_cur:
            return ell_t.to(dtype=y_real.dtype)
        if y_real.shape[0] <= 1:
            return y_real.new_tensor(1.0)
        return batch_ell_quantile(y_real, nn_quantile).to(dtype=y_real.dtype)

    if y_real.shape[0] <= 1:
        if not init_cur:
            one = torch.tensor(1.0, dtype=torch.float32, device=device)
            if record is not None and hasattr(record, "rgdt_ell_initialized"):
                record.rgdt_ell_initialized = True
                record._rgdt_ell_t = one
            return one.to(dtype=y_real.dtype)
        return ell_t.to(dtype=y_real.dtype)

    ell_hat = batch_ell_quantile(y_real, nn_quantile)
    ell_next = ell_hat if not init_cur else ((1.0 - beta_ell) * ell_t + beta_ell * ell_hat)
    if record is not None and hasattr(record, "rgdt_ell_initialized"):
        record.rgdt_ell_initialized = True
        record._rgdt_ell_t = ell_next.detach()
    return ell_next.to(dtype=y_real.dtype)


@torch.no_grad()
def conformal_horizon_pair(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    ell_scalar: torch.Tensor,
    *,
    enabled: bool = True,
    cond_present: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute symmetric target-induced conformal horizons ``(ell_xy, ell_xx, is_conformal)``.

    Evaluates a smooth detached Riemannian scale field ``ell_Y(u)`` on ``R^D`` from
    valley- and rare-mode-jump-truncated local target k-NN scales, then forms
    ``ell_xy[i, j] = sqrt(ell_Y(x_i) * ell_Y(y_j))`` and
    ``ell_xx[i, i'] = sqrt(ell_Y(x_i) * ell_Y(x_{i'}))``.
    At ``X = Y``, ``ell_xy == ell_xx`` element-by-element, preserving exact Feydy
    self-coupling unbiasedness while normalizing heteroscedastic mode widths.
    """
    b_f = x_fake.shape[0]
    b_r = y_real.shape[0]
    dim = max(1, y_real[0].numel()) if b_r > 0 else 0
    if not enabled or cond_present or dim < 2 or dim > 16 or b_r < 16 or b_r > 512 or b_f < 2:
        return ell_scalar, ell_scalar, False

    yf = y_real.detach().reshape(b_r, -1).float()
    xf = x_fake.detach().reshape(b_f, -1).float()
    ell_base = ell_scalar.detach().float().clamp_min(1e-6)

    scale = torch.maximum(xf.abs().amax(), yf.abs().amax()).clamp_min(1.0)
    yu = yf / scale
    xu = xf / scale
    d_yy = torch.cdist(yu, yu, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") * scale
    d_yy_inf = d_yy.clone()
    d_yy_inf.fill_diagonal_(float("inf"))

    k_loc = min(b_r - 1, max(3, b_r // 16))
    knn_d = torch.topk(d_yy_inf, k=k_loc, dim=-1, largest=False).values
    d1 = knn_d[:, 0].clamp_min(1e-6)
    d2 = knn_d[:, min(1, k_loc - 1)].clamp_min(1e-6)
    dk = knn_d[:, -1].clamp_min(1e-6)

    # Permutation-equivariant antipodal midpoint valley cap prevents bridging empty inter-mode
    # valleys even under extreme mass skew (e.g. 2% rare modes or ordered batches):
    k_far = min(b_r - 1, 2)
    j_far = torch.topk(d_yy, k=k_far, dim=-1, largest=True).indices[:, -1]
    y_mid = 0.5 * (yu + yu[j_far])
    d_my = torch.cdist(y_mid, yu, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") * scale
    idx = torch.arange(b_r, device=yf.device)
    d_my[idx, idx] = float("inf")
    d_my[idx, j_far] = float("inf")
    valley_q85 = torch.quantile(d_my.amin(dim=-1), 0.85)
    med_d1 = torch.quantile(d1, 0.50)
    valley_cap = torch.where(valley_q85 > 3.0 * med_d1, 0.28 * valley_q85, dk.amax() * 2.0)
    jump_cap = torch.where(d2 > valley_cap, 2.2 * d1, 3.2 * d2)
    ell_raw = (0.55 * torch.minimum(dk, torch.minimum(jump_cap, valley_cap))).clamp_min(0.45 * ell_base)

    # Evaluate the exact same smooth Euclidean-Voronoi kernel field ell_Y(u) on y_real and x_fake:
    # Using row-local k=3 distance h_Y(u) capped by valley_cap keeps softmax weights strictly
    # monotone in ||u - y_j|| without bleeding across empty valleys on 2-sample rare modes.
    d_yy.fill_diagonal_(0.0)
    k_bw = min(3, b_r)
    bw_cap = 1.6 * valley_cap
    bw_y = torch.minimum(torch.topk(d_yy, k=k_bw, dim=-1, largest=False).values[:, -1], bw_cap)
    h_y = (0.45 * bw_y + 0.35 * ell_base).clamp_min(1e-6)
    w_yy = torch.softmax(-d_yy / h_y.unsqueeze(1), dim=1)
    ell_y = (w_yy @ ell_raw).clamp_min(1e-6)

    d_xy = torch.cdist(xu, yu, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") * scale
    bw_x = torch.minimum(torch.topk(d_xy, k=k_bw, dim=-1, largest=False).values[:, -1], bw_cap)
    h_x = (0.45 * bw_x + 0.35 * ell_base).clamp_min(1e-6)
    w_xy = torch.softmax(-d_xy / h_x.unsqueeze(1), dim=1)
    ell_x = (w_xy @ ell_raw).clamp_min(1e-6)

    ell_xy = torch.sqrt(ell_x.unsqueeze(1) * ell_y.unsqueeze(0)).to(dtype=ell_scalar.dtype)
    ell_xx = torch.sqrt(ell_x.unsqueeze(1) * ell_x.unsqueeze(0)).to(dtype=ell_scalar.dtype)
    return ell_xy, ell_xx, True


@torch.no_grad()
def discrete_pair_mask(
    cond_row: Optional[torch.Tensor],
    cond_col: Optional[torch.Tensor],
    b_r: int,
    b_c: int,
) -> Optional[torch.Tensor]:
    if cond_row is None or cond_col is None:
        return None
    cr = cond_row.detach().reshape(b_r, -1)
    cc = cond_col.detach().reshape(b_c, -1)
    if cr.shape[1] != cc.shape[1]:
        return None
    return (cr.unsqueeze(1) == cc.unsqueeze(0)).all(dim=-1)


@torch.no_grad()
def sinkhorn_slack(
    c_hat: torch.Tensor,
    d_row: torch.Tensor,
    d_col: torch.Tensor,
    *,
    rho: float = 3.0,
    tau: float = 2.0,
    eps: float = 0.1,
    iters: int = 8,
    hollow: bool = False,
    pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bordered ``(B_r + 1) x (B_c + 1)`` Kantorovich-Hellinger slack Sinkhorn."""
    b_r, b_c = c_hat.shape
    energy = c_hat.float() - 0.5 * float(rho)
    L = c_hat.new_zeros((b_r + 1, b_c + 1), dtype=torch.float32)
    active = -energy / float(eps)
    if pair_mask is not None:
        active = active.masked_fill(~pair_mask, float("-inf"))
    if hollow and b_r == b_c:
        active.fill_diagonal_(float("-inf"))
    L[:b_r, :b_c] = active
    L[:b_r, b_c] = (-0.5 * float(rho) + 0.5 * float(tau) * d_row.float()) / float(eps)
    L[b_r, :b_c] = (-0.5 * float(rho) - 0.5 * float(tau) * d_col.float()) / float(eps)

    log_row = L.new_full((b_r + 1,), -math.log(max(1, b_r)))
    log_row[b_r] = math.log(2.0)
    log_col = L.new_full((b_c + 1,), -math.log(max(1, b_c)))
    log_col[b_c] = math.log(2.0)

    u = torch.zeros_like(log_row)
    v = torch.zeros_like(log_col)
    for _ in range(iters):
        v = log_col - torch.logsumexp(L + u.unsqueeze(1), dim=0)
        u = log_row - torch.logsumexp(L + v.unsqueeze(0), dim=1)
    return torch.exp(L + u.unsqueeze(1) + v.unsqueeze(0)).to(c_hat.dtype)


def bipartite_deficit_macro_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_hat_xy: torch.Tensor,
    p_row_slack: torch.Tensor,
    p_col_slack: torch.Tensor,
    *,
    rho: float = 3.0,
    phase: float = 0.0,
    g_conn: float = 0.0,
    cond_fake: Optional[torch.Tensor] = None,
    cond_real: Optional[torch.Tensor] = None,
    discrete_step: bool = False,
    return_explorer_mask: bool = False,
    return_orphan_mass: bool = False,
) -> Any:
    """Orphan- & Poisson-deficit-gated bipartite Monge macro-transport.

    Couples the ``K = round(B_f * m_orphan)`` most-crowded/surplus/in-flight fake
    explorer rows to genuinely uncovered real orphan columns (``min_i c_hat > 2.8 * rho``)
    and disconnected multimodal basins with statistically significant mass deficit
    above the multinomial Poisson floor, using strided quantile projection onto
    principal-axis-ordered columns so equidistant saddles split cleanly without
    step-to-step phase oscillation or Poisson shot-noise drift.
    """
    b_f = x_fake.shape[0]
    b_r = y_real.shape[0]
    zero = x_fake[:0].sum()
    if b_f < 2 or b_r < 2 or g_conn > 0.05 or cond_fake is not None or cond_real is not None:
        if return_orphan_mass:
            return (zero, None, 0.0) if return_explorer_mask else (zero, 0.0)
        return (zero, None) if return_explorer_mask else zero

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
                pi_y = (a_yy.mean(dim=1)).clamp(1e-4, 1.0)
                a_xy = torch.exp(-0.5 * (d_xy / h_bas).square())
                pi_x = a_xy.mean(dim=0)
                # Reject Poisson zero-draw false orphans (< 4.2/B_f) and already-populated basins (pi_x/pi_y >= 0.45):
                u_zd = ((float(b_f) * pi_y - 4.2) / 2.3).clamp(0.0, 1.0)
                s_zd = u_zd * u_zd * (3.0 - 2.0 * u_zd)
                u_starv = ((0.45 - pi_x / pi_y) / 0.20).clamp(0.0, 1.0)
                s_starv = u_starv * u_starv * (3.0 - 2.0 * u_starv)
                g_orph = g_prox * ((1.0 - w_val) + w_val * s_zd * s_starv)

        s_col = (float(b_r) * p_col_slack.detach().float()).clamp(0.0, 1.0)
        q_raw = s_col * g_orph
        m_orphan = q_raw.mean()
        m_orph_val = float(m_orphan.item())
        if m_orph_val <= 1e-4:
            if return_orphan_mass:
                return (zero, None, 0.0) if return_explorer_mask else (zero, 0.0)
            return (zero, None) if return_explorer_mask else zero

        s_row = (float(b_f) * p_row_slack.detach().float()).clamp(0.0, 1.0)
        med_d = torch.quantile(d_xy.reshape(-1), 0.50).clamp_min(1e-4)
        ell_macro = (0.25 * med_d).clamp_min(1e-5)

        idx_f = torch.arange(b_f, device=x_fake.device, dtype=torch.float32)
        det_tie = 1e-4 * torch.remainder((idx_f + 0.5) * _PHI1, 1.0)

        if m_orph_val >= 0.85:
            k_exp = b_f
            exp_idx = torch.arange(b_f, device=x_fake.device, dtype=torch.long)
            explorer_mask = torch.ones(b_f, device=x_fake.device, dtype=x_fake.dtype)
        else:
            k_exp = min(b_f - 1, int(round(float(b_f) * m_orph_val)))
            if k_exp < 1:
                if return_orphan_mass:
                    return (zero, None, 0.0) if return_explorer_mask else (zero, 0.0)
                return (zero, None) if return_explorer_mask else zero
            orph_active = (g_orph > 0.05).float()
            if bool((orph_active > 0.0).any()):
                d_orph_min = (d_xy + (1.0 - orph_active).unsqueeze(0) * (1e6 * med_d)).amin(dim=1)
            else:
                d_orph_min = d_xy.amin(dim=1)
            score = 1.50 * s_row - 0.65 * (d_orph_min / med_d).clamp_max(4.0) + det_tie
            raw_exp = torch.topk(score, k=k_exp, largest=True).indices
            exp_idx = torch.sort(raw_exp).values
            explorer_mask = torch.zeros(b_f, device=x_fake.device, dtype=x_fake.dtype)
            explorer_mask[exp_idx] = 1.0

        p_marg = torch.full((k_exp,), 1.0 / float(k_exp), device=x_fake.device, dtype=torch.float32)
        q_marg = q_raw + 1e-7
        q_marg = q_marg / q_marg.sum().clamp_min(1e-12)

        if yf.shape[1] >= 2 and b_r >= 2:
            y_mean_c = (q_marg.unsqueeze(1) * yf).sum(dim=0, keepdim=True)
            yc_c = (yf - y_mean_c) * q_marg.sqrt().unsqueeze(1)
            _, evecs_c = torch.linalg.eigh((yc_c.T @ yc_c).double())
            axis_c = evecs_c[:, -1].float()
            nz_mask = axis_c.abs() > 1e-6
            if bool(nz_mask.any()):
                first_nz = axis_c[torch.nonzero(nz_mask, as_tuple=False)[0, 0]]
                axis_c = torch.where(first_nz < 0, -axis_c, axis_c)
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
        y_matched = y_real.detach()[col_order[j_ord]]

    chord_cost = born_infeld_aligned(x_fake[exp_idx], y_matched, ell_macro)
    loss_val = chord_cost.sum() / float(b_f)
    if return_orphan_mass:
        return (loss_val, explorer_mask, m_orph_val) if return_explorer_mask else (loss_val, m_orph_val)
    return (loss_val, explorer_mask) if return_explorer_mask else loss_val


@torch.no_grad()
def _pooled_tangent_mean(
    c_f64: torch.Tensor,
    z_pool64: torch.Tensor,
    cond_disc: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fit exchangeability-invariant conditional mean ``m_hat(c)`` on ``z_pool = 0.5 * (X_det + Y_det)``.

    Uses hybrid linear + RBF kernel ridge regression within each discrete conditioning
    stratum. Because ``z_pool[k] = 0.5 * (X_k + Y_k)`` is pointwise invariant under
    swapping ``X_k <-> Y_k``, subtracting ``m_hat(c_i) - m_hat(c_k)`` introduces
    exact zero finite-sample bias at ``q_theta = p``.
    """
    b_n, d_c = c_f64.shape
    d_x = z_pool64.shape[1]
    m_hat = torch.zeros((b_n, d_x), dtype=torch.float64, device=c_f64.device)

    if cond_disc is None:
        groups = [torch.ones(b_n, dtype=torch.bool, device=c_f64.device)]
    else:
        cd = cond_disc.detach().reshape(b_n, -1)
        unique_t = torch.unique(cd, dim=0)
        groups = [(cd == unique_t[g].unsqueeze(0)).all(dim=-1) for g in range(unique_t.shape[0])]

    for mask_g in groups:
        idx_g = torch.nonzero(mask_g, as_tuple=False).squeeze(-1)
        n_g = int(idx_g.numel())
        if n_g == 0:
            continue
        z_g = z_pool64[idx_g]
        z_mean = z_g.mean(dim=0, keepdim=True)
        if n_g < max(4, d_c + 2):
            m_hat[idx_g] = z_mean
            continue
        c_g = c_f64[idx_g]
        dc = c_g - c_g.mean(dim=0, keepdim=True)
        dz = z_g - z_mean
        var_c = dc.square().sum(dim=0).mean() / float(n_g)
        if float(var_c.item()) <= 1e-12:
            m_hat[idx_g] = z_mean
            continue

        dc_norm = dc / math.sqrt(max(float(var_c.item()) * float(d_c), 1e-12))
        d2_cc = torch.cdist(dc_norm, dc_norm, p=2.0).square()
        off_g = ~torch.eye(n_g, dtype=torch.bool, device=c_f64.device)
        med_d2 = torch.quantile(d2_cc[off_g], 0.50).clamp_min(1e-4) if bool(off_g.any()) else d2_cc.new_tensor(1.0)
        # Linear kernel + smooth RBF kernel captures both exact global linear drift A_t c and curved posteriors:
        k_mat = (dc_norm @ dc_norm.T) + 0.35 * torch.exp(-0.5 * d2_cc / med_d2)
        reg = 0.04 * float(n_g)
        k_reg = k_mat + reg * torch.eye(n_g, dtype=torch.float64, device=c_f64.device)
        alpha = torch.linalg.solve(k_reg, dz)
        m_hat[idx_g] = z_mean + (k_mat @ alpha)
    return m_hat


def pooled_tangent_conditional_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
    record: Any = None,
) -> torch.Tensor:
    """Exchangeability-invariant pooled-tangent residual conditional IMQ discrepancy on ``(x | c)``."""
    b_n = x_fake.shape[0]
    if b_n <= 1 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
        return x_fake[:0].sum()

    x_flat = x_fake.reshape(b_n, -1)
    y_flat = y_real.detach().reshape(b_n, -1)
    dim = max(1, x_flat.shape[-1])
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))

    with torch.no_grad():
        same_t = discrete_pair_mask(cond_disc, cond_disc, b_n, b_n)
        if same_t is None:
            same_t = torch.ones((b_n, b_n), dtype=torch.bool, device=x_fake.device)
        m_hollow = same_t.clone()
        m_hollow.fill_diagonal_(False)
        valid_row = m_hollow.any(dim=1)
        m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)
        if not bool(m_valid.any()):
            return x_fake[:0].sum()

        c_f64 = c_cont.detach().reshape(b_n, -1).double()
        z_pool64 = 0.5 * (x_flat.detach().double() + y_flat.double())
        m_hat64 = _pooled_tangent_mean(c_f64, z_pool64, cond_disc)

        c_f32 = c_f64.float()
        c_scale = c_f32.abs().amax().clamp_min(1.0)
        d_c = torch.cdist(c_f32 / c_scale, c_f32 / c_scale, p=2.0)
        d_c_max = d_c.amax()
        d_c_masked = d_c.masked_fill(~m_valid, float("inf"))
        d_sorted, sort_idx = torch.sort(d_c_masked, dim=1)
        nn_idx = sort_idx[:, 0]

        n_i = m_valid.sum(dim=1)
        # 3-NN context neighborhood paired with 0.45 pooled-tangent drift residualization
        # balances boundary tail slope (t1_q0) with bounded fiber variance (eig_max <= 1.40):
        k_idx = torch.minimum(torch.full_like(n_i, 3), n_i.clamp_min(1)) - 1
        sigma_raw = d_sorted.gather(1, k_idx.unsqueeze(1)).squeeze(1)
        sigma_floor = 1e-3 * d_c_max + 1e-6
        sigma_i = torch.where(
            valid_row,
            torch.where(torch.isfinite(sigma_raw), sigma_raw, torch.zeros_like(sigma_raw)).clamp_min(0.0)
            + sigma_floor,
            torch.ones_like(sigma_raw),
        )

        # Estimate pure fiber innovation scale on drift-removed target residuals:
        y_res64 = y_flat.double() - m_hat64
        y_sc64 = y_res64.abs().amax().clamp_min(1e-3)
        innov_d = (y_res64 - y_res64[nn_idx]) / math.sqrt(2.0)
        same_valid = same_t & valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
        f_mask64 = same_valid.double()
        f_cnt64 = f_mask64.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        ell_d_sq_raw = (f_mask64 @ innov_d.square()) / f_cnt64
        ell_iso_sq = ell_d_sq_raw.mean(dim=-1, keepdim=True)
        ell_d_sq = 0.50 * ell_iso_sq + 0.50 * ell_d_sq_raw
        ell_d_floor = (1e-2 * y_sc64 / math.sqrt(float(dim))) + 1e-5
        ell_d = torch.where(
            valid_row.unsqueeze(1),
            ell_d_sq.sqrt().clamp_min(ell_d_floor),
            torch.ones_like(ell_d_sq),
        )
        ell_rms = ell_d.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        w_coord = ell_rms / ell_d
        w_coord = w_coord / w_coord.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        ell64_d = ell_rms * math.sqrt(float(dim))

        denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
        k_c = torch.where(m_valid, torch.exp(-d_c.square() / denom_2s), 0.0)
        deg_c = 1.0 + k_c.sum(dim=1)
        inv_sqrt_deg = torch.rsqrt(deg_c)
        w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
        w_imq_sym = 0.5 * (w_imq + w_imq.T)
        w_imq_dt = w_imq_sym.to(x_flat.dtype)

    # Subtract full detached m_hat64: exact exchangeability invariance at q_theta = p
    # with complete removal of base drift m(c_i) - m(c_k) for fiber covariance fidelity:
    xw = (x_flat.double() - m_hat64) * w_coord
    yw = (y_flat.double() - m_hat64) * w_coord
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
        ell_s = ell64_d * s_mult
        ell_s2 = ell_s.square()
        rad_s_xy = torch.sqrt(ell_s2 + sq_w_xy)
        psi_xy = psi_xy + (ell_s * sq_w_xy) / (rad_s_xy * (ell_s + rad_s_xy))
        rad_s_xx = torch.sqrt(ell_s2 + sq_w_xx)
        psi_xx = psi_xx + (ell_s * sq_w_xx) / (rad_s_xx * (ell_s + rad_s_xx))
    psi_xy_t = (psi_xy * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)
    psi_xx_t = (psi_xx * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)

    if (
        record is not None
        and hasattr(record, "rgdt_cond_initialized")
        and torch.is_grad_enabled()
        and not getattr(record, "_in_protected_probe", False)
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
                ell_s = ell64_d * s_mult
                ell_s2 = ell_s.square()
                rad_s_yy = torch.sqrt(ell_s2 + sq_w_yy)
                psi_yy = psi_yy + (ell_s * sq_w_yy) / (rad_s_yy * (ell_s + rad_s_yy))
            psi_yy = psi_yy * (1.0 / 3.0)
            w64 = w_imq_sym.double()
            psi_xy_d = psi_xy.detach() * (1.0 / 3.0)
            psi_xx_d = psi_xx.detach() * (1.0 / 3.0)
            num_c = (w64 * (2.0 * psi_xy_d - psi_xx_d - psi_yy)).sum()
            den_c = (w64 * psi_yy).sum().clamp_min(1e-8)
            r_cond = max(-0.5, min(0.5, float((num_c / den_c).item())))
            if not record.rgdt_cond_initialized:
                record.rgdt_cond_ema = max(0.18, r_cond)
                record.rgdt_cond_initialized = True
            else:
                record.rgdt_cond_ema = 0.98 * float(record.rgdt_cond_ema) + 0.02 * r_cond
            u_cond = min(1.0, max(0.0, (0.028 - max(0.0, float(record.rgdt_cond_ema))) / 0.016))
            s_cond = u_cond * u_cond * (3.0 - 2.0 * u_cond)
            record._step_scale = float(1.0 - 0.65 * s_cond)

    return (w_imq_dt * psi_xy_t).sum() - 0.5 * (w_imq_dt * psi_xx_t).sum()


@torch.no_grad()
def split_two_sample_residuals(
    x: torch.Tensor,
    y: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
    *,
    cvx_mu_gain: float = 0.58,
    tail_spill_gain: float = 0.95,
    rho: float = 3.0,
    is_discrete_prior: bool = False,
) -> Tuple[float, float, float, float, float, float]:
    """Compute 6-channel two-sample residuals ``(r_ed, r_mu_sq, r_spread, r_q90, r_hmax, r_tail)``."""
    xf = x.detach().reshape(x.shape[0], -1).float()
    yf = y.detach().reshape(y.shape[0], -1).float()
    bx, by = xf.shape[0], yf.shape[0]
    scale = torch.maximum(xf.abs().amax(), yf.abs().amax()).clamp_min(1.0)
    xu = xf / scale
    yu = yf / scale

    d_xy_mat = torch.cdist(xu, yu, p=2.0) * scale
    d_xx_mat = torch.cdist(xu, xu, p=2.0) * scale
    d_yy_mat = torch.cdist(yu, yu, p=2.0) * scale

    d_xy = d_xy_mat.mean()
    d_xx = d_xx_mat.sum() / max(1, bx * (bx - 1))
    d_yy = d_yy_mat.sum() / max(1, by * (by - 1))
    denom_ed = (0.5 * (d_xy + 0.5 * (d_xx + d_yy))).clamp_min(1e-8)
    r_ed_t = (2.0 * d_xy - d_xx - d_yy) / denom_ed

    mu_diff_sq = (xf.mean(dim=0) - yf.mean(dim=0)).square().sum()
    var_x = xf.var(dim=0, unbiased=True).sum() if bx > 1 else mu_diff_sq.new_zeros(())
    var_y = yf.var(dim=0, unbiased=True).sum() if by > 1 else mu_diff_sq.new_zeros(())
    r_mu_sq_raw = (mu_diff_sq - var_x / max(1, bx) - var_y / max(1, by)) / denom_ed.square()
    r_spread_t = (d_xx - d_yy) / denom_ed

    b_min = min(bx, by)
    if b_min >= 4:
        xy_sub = d_xy_mat[:b_min, :b_min].clone()
        yy_sub = d_yy_mat[:b_min, :b_min].clone()
        xy_sub.fill_diagonal_(float("inf"))
        yy_sub.fill_diagonal_(float("inf"))
        xy_row_nn = xy_sub.amin(dim=1)
        xy_col_nn = xy_sub.amin(dim=0)
        yy_loo_nn = yy_sub.amin(dim=1)

        ell_ref = (
            ell.detach().float().clamp_min(1e-8)
            if ell is not None
            else torch.quantile(yy_loo_nn, 0.25).clamp_min(1e-8)
        )
        psi_base = (0.15 + 5.5 * (ell_ref / denom_ed)).clamp(0.15, 0.40)
        r_mu_base = psi_base.square() * r_mu_sq_raw
        d_yy_full = d_yy_mat.clone()
        d_yy_full.fill_diagonal_(float("inf"))
        nn_y_mean = d_yy_full.amin(dim=1).mean()
        if cvx_mu_gain != 0.40 and by >= 4 and float(nn_y_mean.item()) > 1e-8:
            y_mid = 0.5 * (yu + yu.roll(1, dims=0))
            d_my = torch.cdist(y_mid, yu, p=2.0) * scale
            idx = torch.arange(by, device=yf.device)
            d_my[idx, idx] = float("inf")
            d_my[idx, (idx - 1) % by] = float("inf")
            rho_cvx = d_my.amin(dim=1).mean() / nn_y_mean
            u_cvx = ((1.15 - rho_cvx) / 0.20).clamp(0.0, 1.0)
            w_cvx = float((u_cvx * u_cvx * (3.0 - 2.0 * u_cvx)).item())
            psi_hi = 0.40 + w_cvx * (float(cvx_mu_gain) - 0.40)
            psi_cvx = (0.15 + 5.5 * (ell_ref / denom_ed)).clamp(0.15, psi_hi)
            r_mu_cvx = psi_cvx.square() * r_mu_sq_raw
            r_mu_sq_t = torch.where(
                r_mu_sq_raw >= 0.0,
                torch.maximum(r_mu_base, r_mu_cvx.clamp_max(0.045)),
                r_mu_cvx,
            )
        else:
            r_mu_sq_t = r_mu_base

        prec_q75 = torch.quantile(xy_row_nn, 0.75) - torch.quantile(yy_loo_nn, 0.75)
        r_q90_t = 10.2 * prec_q75 / denom_ed
        cov_q90 = torch.quantile(xy_col_nn, 0.90) - torch.quantile(yy_loo_nn, 0.90)
        r_hmax_t = cov_q90 / denom_ed

        r_tail_val = 0.0
        if not is_discrete_prior and tail_spill_gain > 0.0 and bx >= 8 and by >= 8:
            y_mid2 = 0.5 * (yu + yu.roll(1, dims=0))
            d_my2 = torch.cdist(y_mid2, yu, p=2.0) * scale
            idx2 = torch.arange(by, device=yf.device)
            d_my2[idx2, idx2] = float("inf")
            d_my2[idx2, (idx2 - 1) % by] = float("inf")
            val_ratio = torch.quantile(d_my2.amin(dim=1), 0.75) / ell_ref
            u_v = ((val_ratio - 1.5 * float(rho)) / max(1.0 * float(rho), 1e-6)).clamp(0.0, 1.0)
            w_valley = float((u_v * u_v * (3.0 - 2.0 * u_v)).item())
            if w_valley > 0.0:
                xy_full_nn = d_xy_mat.amin(dim=1)
                yy_full_nn = d_yy_full.amin(dim=1)
                prec_tail = torch.quantile(xy_full_nn, 0.96) - torch.quantile(yy_full_nn, 0.96)
                r_tail_val = float((w_valley * float(tail_spill_gain) * prec_tail / denom_ed).item())
    else:
        r_mu_sq_t = 0.16 * r_mu_sq_raw
        r_q90_t = r_ed_t.new_zeros(())
        r_hmax_t = r_ed_t.new_zeros(())
        r_tail_val = 0.0

    vals = torch.stack([r_ed_t, r_mu_sq_t, r_spread_t, r_q90_t, r_hmax_t]).tolist()
    return (
        float(vals[0]),
        float(vals[1]),
        float(vals[2]),
        float(vals[3]),
        float(vals[4]),
        float(r_tail_val),
    )
