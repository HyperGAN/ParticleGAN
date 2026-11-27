"""Axiom L2 Law 3A Transport Identifiability: Dimension-Free OT, Conformal Horizons & Atlas Routing.

Composes L0/L1 primitives (``math_ops``, ``state``, ``atlas``, ``fiber``) into
a scale- and dimension-equivariant transport operator with energy-preserving
feature projection, continuous ``kappa_struct``, dimensionless Born-Infeld OT,
conformal ``k=3`` horizons, ``k=4`` orphan rescue, unified GPU ``PrototypeAtlas``
and ``HamiltonCommitmentLedger`` routing, and ``g_conn``-gated simplicial charts.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple
import weakref

import torch
from torch import nn
import torch.nn.functional as F

from .atlas import (
    MAX_LEDGER_PARTICLES,
    CoalescedAtlas,
    LedgerAssignment,
    PrototypeAtlas,
    batch_annular_gap_ratio,
    cov_trace_mean,
    manifold_connectivity_stats,
)
from .fiber import AxiomCondFiberBank, conditional_fiber_loss
from .math_ops import (
    PHI1,
    PHI2,
    born_infeld_cost,
    log_sinkhorn,
    power_principal_axis,
    safe_cdist,
    sinkhorn_with_slack,
    smoothstep01,
    sym_eigensplit_halley,
)
from .state import recover_active_prior_rows, set_active_prior_rows

_MAX_FLAT_DIM: int = 256
_ATLAS_TENSOR_KEYS = (
    "proto_c",
    "proto_cov",
    "proto_w",
    "w_fake_ema",
    "proto_n",
    "psi",
    "v_push",
)
_AtlasCache = weakref.WeakKeyDictionary


class _HetState(NamedTuple):
    exy: torch.Tensor
    exx: torch.Tensor
    ey_loc: torch.Tensor
    g_het: torch.Tensor
    nn1_y: Optional[torch.Tensor]
    s_near: Optional[torch.Tensor]
    wv_sup: torch.Tensor
    hv_t: Optional[torch.Tensor]
    dyy: Optional[torch.Tensor]


class _PlanOut(NamedTuple):
    pxy_a: torch.Tensor
    pxy_s: torch.Tensor
    pxy_cs: torch.Tensor
    pxx_a: torch.Tensor
    af: torch.Tensor
    sf: torch.Tensor
    p_sl: torch.Tensor


class _SimpSamples(NamedTuple):
    z_simp: torch.Tensor
    ne: int
    v0e: torch.Tensor
    v1e: torch.Tensor
    te: torch.Tensor
    v0t: Optional[torch.Tensor]
    v1t: Optional[torch.Tensor]
    v2t: Optional[torch.Tensor]
    z_tri: Optional[torch.Tensor]


def _scale_grad(t: torch.Tensor, scale: float) -> torch.Tensor:
    if scale < 1.0 and t.requires_grad:
        return t * scale + t.detach() * (1.0 - scale)
    return t


def _can_mutate(record: Any, allow: bool = True) -> bool:
    if not (allow and torch.is_grad_enabled()):
        return False
    return not (
        getattr(record, "_in_probe", False)
        or getattr(record, "_in_protected_probe", False)
    )


def _discrete_mask(
    cr: Optional[torch.Tensor], cc: Optional[torch.Tensor], br: int, bc: int
) -> Optional[torch.Tensor]:
    if cr is None or cc is None or int(cr.shape[0]) != br or int(cc.shape[0]) != bc:
        return None
    rf, cf = cr.detach().reshape(br, -1), cc.detach().reshape(bc, -1)
    if int(rf.shape[1]) != int(cf.shape[1]):
        return None
    return (rf.unsqueeze(1) == cf.unsqueeze(0)).all(dim=-1)


def _valid_particle_table(
    ptab: Optional[torch.Tensor], brows: Optional[torch.Tensor], bf: int
) -> bool:
    if ptab is None or brows is None or ptab.ndim != 2 or brows.ndim != 1:
        return False
    return int(brows.shape[0]) == bf and 1 <= int(ptab.shape[0]) <= MAX_LEDGER_PARTICLES


# ---------------------------------------------------------------------------
# Law 3.1: Dimension-Free Feature Projection & Continuous Contrast Confidence
# ---------------------------------------------------------------------------


def project_features(
    x: torch.Tensor, max_spatial: int = 8, max_flat_dim: int = _MAX_FLAT_DIM
) -> torch.Tensor:
    """Energy-preserving projection to ``<= max_spatial^2`` spatial and ``<= max_flat_dim`` flat."""
    b = int(x.shape[0])
    if x.ndim >= 4 and (x.shape[-2] > max_spatial or x.shape[-1] > max_spatial):
        hi, wi = int(x.shape[-2]), int(x.shape[-1])
        ho, wo = min(int(max_spatial), hi), min(int(max_spatial), wi)
        x4 = x.reshape(-1, int(x.shape[-3]), hi, wi).float()
        ph, pw = (-hi) % ho, (-wi) % wo
        if ph or pw:
            x4 = F.pad(x4, (0, pw, 0, ph), mode="replicate")
        blk = x4.reshape(
            x4.shape[0], x4.shape[1], ho, x4.shape[-2] // ho, wo, x4.shape[-1] // wo
        )
        z = blk.mean(dim=(3, 5)).reshape(b, -1)
        z = z * math.sqrt(float(hi * wi) / float(max(1, ho * wo)))
    else:
        z = x.reshape(b, -1).float()

    d, cap = int(z.shape[1]), max(1, int(max_flat_dim))
    if d > cap:
        groups = (d + cap - 1) // cap
        pad = groups * cap - d
        if pad:
            z = F.pad(z, (0, pad), mode="replicate")
        z = z.reshape(b, groups, cap).mean(dim=1) * math.sqrt(float(d) / float(cap))
    return z.to(x.dtype)


project_transport_features = project_features


@torch.no_grad()
def compute_contrast_confidence(
    y_feat: torch.Tensor, *, max_b: int = 64
) -> torch.Tensor:
    """Continuous distance-contrast confidence ``kappa_struct(Y) in [0, 1]``."""
    z = y_feat.detach().reshape(int(y_feat.shape[0]), -1).float()
    b = int(z.shape[0])
    if b < 4:
        return z.new_ones(())
    if b > max_b:
        idx = (torch.arange(max_b, device=z.device, dtype=torch.long) * b) // max_b
        z, b = z.index_select(0, idx), max_b
    dmat = safe_cdist(z)
    s_all = dmat.reshape(-1).sort().values
    dmax = s_all[-1]
    m = b * (b - 1)
    med_all = (0.5 * (s_all[b + (m - 1) // 2] + s_all[b + m // 2])).clamp_min(1e-30)
    eye = torch.eye(b, dtype=torch.bool, device=z.device)
    s_nn = dmat.masked_fill(eye, float("inf")).amin(dim=1).sort().values
    med_nn = 0.5 * (s_nn[(b - 1) // 2] + s_nn[b // 2])
    kappa = smoothstep01((0.65 - med_nn / med_all) / 0.35)
    return torch.where(dmax <= 1e-24, z.new_ones(()), kappa)


@torch.no_grad()
def _sharp_charts_atomic(
    yf: torch.Tensor,
    charts: Optional[torch.Tensor],
    radii: Optional[torch.Tensor],
) -> bool:
    if charts is None or int(charts.shape[0]) < 4:
        return False
    cf = charts.detach().to(device=yf.device, dtype=torch.float32)
    mnn = torch.quantile(safe_cdist(cf, diag_inf=True).amin(dim=1), 0.50)
    qd = torch.quantile(safe_cdist(yf, cf).amin(dim=1), 0.75)
    qr = torch.quantile(radii.float(), 0.75) if radii is not None else yf.new_zeros(())
    return bool(((mnn > 1e-12) & (qd < 1e-3 * mnn) & (qr < 1e-3 * mnn)).item())


@torch.no_grad()
def compute_discrete_atomic_gate(
    y_cf: torch.Tensor,
    *,
    charts: Optional[torch.Tensor] = None,
    radii: Optional[torch.Tensor] = None,
    record: Any = None,
) -> float:
    """Annular-gap duplicate purity gate (1.0 on exact discrete atoms, 0.0 on continuous/noisy data)."""
    yf = y_cf.detach().reshape(int(y_cf.shape[0]), -1).float()
    if int(yf.shape[0]) < 8:
        return 0.0
    if _sharp_charts_atomic(yf, charts, radii):
        return 1.0
    cs = getattr(record, "chart_state", None) if record is not None else None
    if isinstance(cs, dict) and cs.get("initialized") and "rho_atom_ema" in cs:
        rv = cs["rho_atom_ema"]
        if (float(rv.item()) if isinstance(rv, torch.Tensor) else float(rv)) < 1e-3:
            return 1.0
    d = safe_cdist(yf)
    dm = d.amax()
    tol = 1e-6 * dm
    nn_d = d.clone().fill_diagonal_(float("inf")).amin(dim=1)
    na = (nn_d <= tol).float().sum()
    sep_m = d > tol
    n_sep = sep_m.sum()
    ds = torch.where(sep_m, d, d.new_tensor(float("inf"))).reshape(-1).sort().values
    ntol = torch.maximum(10.0 * tol, 0.25 * ds[(n_sep // 20).clamp_max(ds.numel() - 1)])
    near = ((nn_d > tol) & (nn_d < ntol)).float().sum()
    gate = smoothstep01((na / (na + near).clamp_min(1.0) - 0.70) / 0.20)
    out = torch.where((dm > 1e-24) & (na >= 2.0) & (n_sep > 0), gate, dm.new_zeros(()))
    return float(out.item())


# ---------------------------------------------------------------------------
# Law 3.2: Scale-Invariant Horizons, Heteroscedasticity & k=4 Orphan Rescue
# ---------------------------------------------------------------------------


@torch.no_grad()
def _batch_ell(y_flat: torch.Tensor, q: float = 0.25) -> torch.Tensor:
    b = int(y_flat.shape[0])
    if b <= 1:
        return y_flat.new_tensor(1.0, dtype=torch.float32)
    dim = max(1, int(y_flat.shape[-1]))
    akt_e = max(0.0, 1.0 / float(dim) - 0.5)
    den_q = max(1e-6, -math.log(max(1e-6, 1.0 - float(q))))
    akt_f = ((2.0 * math.e * float(b)) / den_q) ** akt_e if akt_e > 0.0 else 1.0
    d = safe_cdist(y_flat)
    dm = d.amax()
    thr = 1e-5 * dm
    pmin = torch.where(d > thr, d, d.new_tensor(float("inf"))).amin()
    qr = torch.quantile(d.fill_diagonal_(float("inf")).amin(dim=-1), q)
    fb = torch.where(torch.isfinite(pmin), 0.15 * pmin, 0.05 * dm)
    val = torch.where(qr <= thr, fb, qr * akt_f).clamp_min(1e-5 * dm)
    return torch.where(dm <= 0.0, y_flat.new_tensor(1e-3, dtype=torch.float32), val)


@torch.no_grad()
def _coord_scale(xp: torch.Tensor, yp: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    yf, xf = yp.detach().float(), xp.detach().float()
    rms_y = (yf - yf.mean(dim=0, keepdim=True)).square().sum(dim=-1).mean().sqrt()
    rms_x = (xf - xf.mean(dim=0, keepdim=True)).square().sum(dim=-1).mean().sqrt()
    gap = (xf.mean(dim=0) - yf.mean(dim=0)).norm()
    s_raw = torch.maximum(
        torch.maximum(rms_y, 0.25 * rms_x), torch.maximum(gap, ell.detach().float())
    )
    return torch.where(s_raw > 0.0, s_raw.clamp_min(1e-30), s_raw.new_ones(()))


@torch.no_grad()
def _valley_width(yf: torch.Tensor, dyy: torch.Tensor) -> torch.Tensor:
    jf = torch.topk(dyy, k=min(int(yf.shape[0]) - 1, 2), dim=-1, largest=True).indices[
        :, -1
    ]
    dmy = safe_cdist(0.5 * (yf + yf[jf]), yf)
    ix = torch.arange(int(yf.shape[0]), device=yf.device)
    dmy[ix, ix] = float("inf")
    dmy[ix, jf] = float("inf")
    return torch.quantile(dmy.amin(dim=-1), 0.85)


@torch.no_grad()
def _conformal_horizon(
    xp: torch.Tensor, yp: torch.Tensor, ell: torch.Tensor, g_conn: float, cond_p: bool
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    bf, br = int(xp.shape[0]), int(yp.shape[0])
    if g_conn > 0.5 or cond_p or bf < 12 or br < 12:
        return ell, ell, None, None, None
    eg = ell.detach().float().clamp_min(1e-30)
    ef = 1e-6 * eg
    dyy = safe_cdist(yp)
    vals = (
        dyy.clone()
        .fill_diagonal_(float("inf"))
        .topk(min(3, br - 1), dim=1, largest=False)
        .values
    )
    d1, d2 = vals[:, 0].clamp_min(ef), vals[:, min(1, vals.shape[1] - 1)].clamp_min(ef)
    d3 = vals[:, min(2, vals.shape[1] - 1)].clamp_min(ef)
    rpt = torch.where((d2 <= 2.8 * d1) & (d3 <= 2.8 * d2), d2, d1)
    pds, pix = dyy.topk(min(8, br), dim=1, largest=False)
    inm = (pds <= (3.2 * d2).unsqueeze(1)).float()
    inm[:, 0] = 1.0
    rp = (rpt[pix] * inm).sum(dim=1) / inm.sum(dim=1).clamp_min(1.0)
    ratio = torch.quantile(rp, 0.85) / torch.quantile(rp, 0.15).clamp_min(ef)
    gw = smoothstep01((ratio - 2.85) / 0.85)
    if float(gw.item()) <= 1e-4:
        return ell, ell, None, None, dyy
    s_scaled = (eg * (rp / torch.quantile(rp, 0.25).clamp_min(ef))).clamp(
        0.70 * eg, 4.50 * eg
    )
    sraw = (1.0 - gw) * eg + gw * s_scaled
    tbw = (2.0 * (2.4 * sraw).square()).clamp_min(1e-30).unsqueeze(0)
    ey = (torch.softmax(-dyy.square() / tbw, dim=1) @ sraw).clamp_min(ef)
    ex = (torch.softmax(-safe_cdist(xp, yp).square() / tbw, dim=1) @ sraw).clamp_min(ef)
    exy = torch.sqrt(ex.unsqueeze(1) * ey.unsqueeze(0)).to(ell.dtype)
    exx = torch.sqrt(ex.unsqueeze(1) * ex.unsqueeze(0)).to(ell.dtype)
    return exy, exx, ex, ey, dyy


@torch.no_grad()
def _heteroscedastic_horizon(
    xp: torch.Tensor,
    yp: torch.Tensor,
    ell: torch.Tensor,
    ex_ms: Optional[torch.Tensor],
    ey_ms: Optional[torch.Tensor],
    dyy_c: Optional[torch.Tensor],
    wv_c: torch.Tensor,
    g_conn: float,
    record: Any,
) -> _HetState:
    bf, br = int(xp.shape[0]), int(yp.shape[0])
    dyy = dyy_c.clone() if dyy_c is not None else safe_cdist(yp)
    dyy_s = dyy.clone().fill_diagonal_(0.0)
    eb = ell.detach().float().clamp_min(1e-30)
    v_gate = smoothstep01((_valley_width(yp.float(), dyy_s) / (2.0 * eb) - 2.55) / 1.50)
    wv_sup = torch.maximum(wv_c, v_gate * (1.0 - g_conn))
    pos = dyy_s > 0.0
    ds_nz = (
        torch.where(pos, dyy_s, dyy_s.new_tensor(float("inf")))
        .reshape(-1)
        .sort()
        .values
    )
    n_nz = pos.sum()
    md_s = torch.where(
        n_nz > 0, ds_nz[(n_nz // 4).clamp_max(ds_nz.numel() - 1)].clamp_min(1e-30), eb
    )
    bw = (
        (1.15 * md_s * (ey_ms / eb)).clamp_min(1e-30).unsqueeze(1)
        if ey_ms is not None
        else (1.15 * md_s)
    )
    rho_k = torch.exp(-dyy_s.square() / (2.0 * bw.square()).clamp_min(1e-30)).sum(dim=1)
    ec_t = (
        wv_sup
        * (torch.quantile(rho_k, 0.85) / torch.quantile(rho_k, 0.03).clamp_min(1.0))
    ).clamp(0.0, 20.0)
    if record is not None and hasattr(record, "het_initialized"):
        hv_t = (
            (0.92 * float(record.het_ema) + 0.08 * ec_t)
            if record.het_initialized
            else ec_t.clamp_max(3.0)
        )
    else:
        hv_t = ec_t
    g_het = smoothstep01((hv_t - 3.4) / 1.0)
    dyy_i = dyy_s.clone().fill_diagonal_(float("inf"))
    nn1_y = dyy_i.amin(dim=1)
    nn2_y = dyy_i.topk(min(2, br - 1), dim=1, largest=False).values[:, -1]
    s_sp = g_het * smoothstep01((nn2_y / eb - 2.8) / 1.4)
    ey_base = ey_ms if ey_ms is not None else eb.expand(br)
    ey_f = torch.maximum(ey_base, eb + s_sp * (0.64 * nn2_y - eb).clamp(0.0, 3.4 * eb))
    nd_s, ny_s = safe_cdist(xp, yp).min(dim=1)
    ne_s = ey_f[ny_s]
    s_near = s_sp[ny_s] * smoothstep01(
        (2.8 * ne_s - nd_s) / (1.2 * ne_s).clamp_min(1e-30)
    )
    ex_base = ex_ms if ex_ms is not None else eb.expand(bf)
    ex_f = torch.maximum(ex_base, eb + s_near * (ne_s - eb)).to(ell.dtype)
    ey_loc = ey_f.to(ell.dtype)
    exy = torch.sqrt(ex_f.unsqueeze(1) * ey_loc.unsqueeze(0))
    exx = torch.sqrt(ex_f.unsqueeze(1) * ex_f.unsqueeze(0))
    return _HetState(exy, exx, ey_loc, g_het, nn1_y, s_near, wv_sup, hv_t, dyy)


def _hetero_outlier_attraction(
    cxy: torch.Tensor,
    chxy: torch.Tensor,
    ey_loc: torch.Tensor,
    nn1_y: torch.Tensor,
    dft: torch.Tensor,
    sf: torch.Tensor,
    g_het: torch.Tensor,
    pxy_a: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    with torch.no_grad():
        bf = int(cxy.shape[0])
        gb = smoothstep01((pxy_a.detach().float().sum() - 0.72) / 0.12)
        ef_y, ch_f = ey_loc.detach().float().clamp_min(1e-30), chxy.detach().float()
        rn = ch_f.amin(dim=1)
        q98 = torch.quantile(
            (torch.sqrt(ef_y.square() + nn1_y.square()) - ef_y) / ef_y, 0.98
        )
        tg = torch.maximum(ef_y.new_tensor(4.2 * rho), 2.10 * q98)
        gate = (g_het * gb) * smoothstep01((rn - tg) / (0.40 * tg))
        orow = (
            gate
            * torch.sigmoid((-dft.detach().float() - 0.35) / 0.20)
            * sf.detach().float().pow(4)
        )
        wo = torch.softmax(-(ch_f - rn.unsqueeze(1)) / 0.25, dim=1).to(cxy.dtype)
    return 2.2 * ((orow.to(cxy.dtype) / float(bf)).unsqueeze(1) * wo * cxy).sum()


def _orphan_macro_loss(
    xp: torch.Tensor,
    yp: torch.Tensor,
    ch_xy: torch.Tensor,
    prs: torch.Tensor,
    pcs: torch.Tensor,
    *,
    rho: float,
    g_conn: float,
) -> torch.Tensor:
    """Scale-invariant k=4 density-ratio orphan rescue with 1D principal-axis Sinkhorn handoff."""
    bf, br = int(xp.shape[0]), int(yp.shape[0])
    if bf < 2 or br < 2 or g_conn > 0.05:
        return xp[:0].sum()
    with torch.no_grad():
        g_o = smoothstep01(
            (ch_xy.detach().float().amin(dim=0) - 2.80 * rho) / max(1.20 * rho, 1e-6)
        )
        dxy, yf = safe_cdist(xp, yp), yp.detach().float()
        if br >= 8 and bf >= 8:
            dyy = safe_cdist(yf)
            dyy_i = dyy.clone().fill_diagonal_(float("inf"))
            kl = min(4, br - 1, bf)
            k_xy = dxy.topk(kl, dim=0, largest=False).values[-1]
            k_yy = dyy_i.topk(kl, dim=1, largest=False).values[:, -1].clamp_min(1e-30)
            g_o = g_o * smoothstep01((k_xy / k_yy - 3.52) / 1.30)
            vq = _valley_width(yf, dyy)
            med_nn = torch.quantile(dyy_i.amin(dim=1), 0.50).clamp_min(1e-30)
            wv = smoothstep01((vq / med_nn - 3.20) / 1.80)
            hb = (0.38 * vq).clamp_min(1e-30)
            py = torch.exp(-0.5 * (dyy / hb).square()).mean(dim=1).clamp(1e-4, 1.0)
            px = torch.exp(-0.5 * (dxy / hb).square()).mean(dim=0)
            g_o = g_o * (
                (1.0 - wv)
                + wv
                * smoothstep01((bf * py - 4.2) / 2.3)
                * smoothstep01((0.45 - px / py) / 0.20)
            )

        qr = (br * pcs.detach().float()).clamp(0.0, 1.0) * g_o
        mo = float(qr.mean().item())
        ke = bf if mo >= 0.85 else min(bf - 1, int(round(bf * mo)))
        if mo <= 1e-4 or ke < 1:
            return xp[:0].sum()
        md = torch.quantile(dxy.reshape(-1), 0.50).clamp_min(1e-30)
        oa = (g_o > 0.05).float()
        dm_or = (dxy + (1.0 - oa).unsqueeze(0) * (1e6 * md)).amin(dim=1)
        dm = torch.where((oa > 0.0).any(), dm_or, dxy.amin(dim=1))
        tie = 1e-4 * torch.remainder(
            (torch.arange(bf, device=xp.device, dtype=torch.float32) + 0.5) * PHI1, 1.0
        )
        if ke == bf:
            ei = torch.arange(bf, device=xp.device)
        else:
            sc = (
                1.50 * (bf * prs.detach().float()).clamp(0.0, 1.0)
                - 0.65 * (dm / md).clamp_max(4.0)
                + tie
            )
            ei = torch.sort(torch.topk(sc, k=ke).indices).values
        qm = (qr + 1e-7) / (qr + 1e-7).sum().clamp_min(1e-12)
        yc = (
            yf - (qm.unsqueeze(1) * yf).sum(dim=0, keepdim=True)
        ) * qm.sqrt().unsqueeze(1)
        if int(yf.shape[1]) >= 2:
            _, axis = power_principal_axis((yc.T @ yc).double(), iters=6, seed=0)
            co = torch.argsort(yf @ axis.float())
        else:
            co = torch.argsort(yf[:, 0])
        pi = log_sinkhorn(
            -(dxy[ei][:, co] / md) / 0.03,
            dxy.new_full((ke,), -math.log(ke)),
            qm[co].clamp_min(1e-12).log(),
            iters=10,
        ).exp_()
        cdf = torch.cumsum(pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12), dim=1)
        cdf[:, -1] = 1.0
        ie = torch.arange(ke, device=xp.device)
        rank = torch.where(ie % 2 == 0, ie // 2, (ke + 1) // 2 + ie // 2).float()
        u_q = ((rank + 0.5) / float(ke)).clamp(1e-6, 1.0 - 1e-6)
        ym = yp.detach()[
            co[torch.searchsorted(cdf, u_q.unsqueeze(1)).squeeze(1).clamp(0, br - 1)]
        ]
    return born_infeld_cost(xp[ei], ym, (0.25 * md).clamp_min(1e-30), aligned=True)[
        0
    ].sum() / float(bf)


@torch.no_grad()
def _two_sample_residuals(
    x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor, *, cvx_mu_gain: float = 0.58
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.detach().reshape(int(x.shape[0]), -1).float()
    yf = y.detach().reshape(int(y.shape[0]), -1).float()
    bx, by = int(xf.shape[0]), int(yf.shape[0])
    dxy, dxx, dyy = safe_cdist(xf, yf), safe_cdist(xf), safe_cdist(yf)
    mxy = dxy.mean()
    mxx, myy = (
        dxx.sum() / float(max(1, bx * (bx - 1))),
        dyy.sum() / float(max(1, by * (by - 1))),
    )
    den = (0.5 * (mxy + 0.5 * (mxx + myy))).clamp_min(1e-30)
    r_ed, r_sp, bm = (2.0 * mxy - mxx - myy) / den, (mxx - myy) / den, min(bx, by)
    vx = xf.var(dim=0).sum() / float(max(1, bx)) if bx > 1 else den.new_zeros(())
    vy = yf.var(dim=0).sum() / float(max(1, by)) if by > 1 else den.new_zeros(())
    mu_sq = ((xf.mean(dim=0) - yf.mean(dim=0)).square().sum() - vx - vy) / den.square()
    if bm < 4:
        return r_ed, 0.16 * mu_sq, r_sp, r_ed, r_ed
    xs = dxy[:bm, :bm].clone().fill_diagonal_(float("inf"))
    yn = dyy[:bm, :bm].clone().fill_diagonal_(float("inf")).amin(dim=1)
    el_r = ell.detach().float().clamp_min(1e-30) / den
    r_mb = (0.15 + 5.5 * el_r).clamp(0.15, 0.40).square() * mu_sq
    r_mu = r_mb
    if cvx_mu_gain != 0.40 and by >= 4:
        nn_m = dyy.clone().fill_diagonal_(float("inf")).amin(dim=1).mean()
        dmy = safe_cdist(0.5 * (yf + yf.roll(1, 0)), yf)
        iy = torch.arange(by, device=yf.device)
        dmy[iy, iy] = dmy[iy, (iy + 1) % by] = float("inf")
        wc = torch.where(
            nn_m > 1e-30,
            smoothstep01(
                (1.15 - dmy.amin(dim=1).mean() / nn_m.clamp_min(1e-30)) / 0.20
            ),
            nn_m.new_zeros(()),
        )
        rcx = (0.15 + 5.5 * el_r).clamp(
            0.15, 0.40 + wc * (cvx_mu_gain - 0.40)
        ).square() * mu_sq
        r_mu = torch.where(
            nn_m > 1e-30,
            torch.where(mu_sq >= 0.0, torch.maximum(r_mb, rcx.clamp_max(0.045)), rcx),
            r_mb,
        )
    r_q90 = (
        10.2 * (torch.quantile(xs.amin(dim=1), 0.75) - torch.quantile(yn, 0.75)) / den
    )
    r_hm = (torch.quantile(xs.amin(dim=0), 0.90) - torch.quantile(yn, 0.90)) / den
    return r_ed, r_mu, r_sp, r_q90, r_hm


# ---------------------------------------------------------------------------
# Law 3.3: Unified Atlas / Hamilton Ledger State Sync & BiGAN Pair Split
# ---------------------------------------------------------------------------


def _export_atlas_live(atlas: PrototypeAtlas) -> Dict[str, Any]:
    led = atlas.ledger
    led_dict = (
        {
            "schema": 2,
            "initialized": True,
            "num_particles": int(led.num_particles),
            "dim": int(led.dim),
            "row_comp": led.row_comp,
            "row_target": led.row_target,
            "row_dist": led.row_dist,
        }
        if led.initialized
        else None
    )
    rho_v = (
        atlas._rho_atom_t if atlas._rho_atom_t is not None else float(atlas._rho_atom_f)
    )
    gc_v = atlas._g_conn_t if atlas._g_conn_t is not None else float(atlas._g_conn_f)
    return {
        "schema": 2,
        "max_prototypes": int(atlas.max_prototypes),
        "beta_n": float(atlas.beta_n),
        "initialized": bool(atlas.initialized),
        "dim": int(atlas.dim),
        "rho_atom_ema": rho_v,
        "g_conn_ema": gc_v,
        **{k: getattr(atlas, k) for k in _ATLAS_TENSOR_KEYS},
        "ledger": led_dict,
    }


def _hydrate_atlas_from_dict(
    state: Dict[str, Any], max_p: int, device: torch.device
) -> PrototypeAtlas:
    atlas = PrototypeAtlas(
        max_prototypes=int(state.get("max_prototypes", max_p)), device=device
    )
    clean = dict(state)
    for sk in ("rho_atom_ema", "g_conn_ema"):
        val = clean.get(sk)
        if isinstance(val, torch.Tensor):
            clean[sk] = float(val.item())
    atlas.load_state_dict(clean, device=device)
    return atlas


def _split_joint_pair(
    xf: torch.Tensor,
    yr: torch.Tensor,
    tab: Optional[torch.Tensor],
    rows: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    bf, br = int(xf.shape[0]), int(yr.shape[0])
    if not _valid_particle_table(tab, rows, bf) or bf < 4 or br < 4:
        return None
    assert tab is not None and rows is not None
    zd = int(tab.shape[1])
    xff, yrf = xf.reshape(bf, -1), yr.reshape(br, -1)
    if int(xff.shape[1]) <= zd or int(yrf.shape[1]) != int(xff.shape[1]):
        return None
    rl = rows.to(device=tab.device, dtype=torch.long)
    zr = tab.detach()[rl.clamp(0, int(tab.shape[0]) - 1)].to(
        device=xff.device, dtype=torch.float32
    )
    oob = ((rl < 0).any() | (rl >= int(tab.shape[0])).any()).float().to(xff.device)
    e_tail = (xff[:, -zd:].detach().float() - zr).abs().amax()
    e_head = (xff[:, :zd].detach().float() - zr).abs().amax()
    oob_v, tail_v, head_v = torch.stack((oob, e_tail, e_head)).tolist()
    if oob_v > 0.5:
        return None
    if tail_v <= 1e-5:
        return xff[:, :-zd], xff[:, -zd:], yrf[:, :-zd], yrf[:, -zd:]
    if head_v <= 1e-5:
        return xff[:, zd:], xff[:, :zd], yrf[:, zd:], yrf[:, :zd]
    return None


def _halo_radii(
    coal: CoalescedAtlas, halo_mult: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_c, dim = int(coal.comp_c.shape[0]), int(coal.comp_c.shape[1])
    nn_cc = (
        safe_cdist(coal.comp_c, diag_inf=True).amin(dim=1).clamp_min(1e-30)
        if k_c >= 2
        else coal.comp_c.new_ones((k_c,))
    )
    r_comp = (float(dim) * cov_trace_mean(coal.comp_cov)).clamp_min(0.0).sqrt()
    return torch.minimum(float(halo_mult) * r_comp, 0.35 * nn_cc), nn_cc


def _halo_bi_loss(
    xp: torch.Tensor, target: torch.Tensor, r_halo: torch.Tensor, ell_ref: torch.Tensor
) -> torch.Tensor:
    diff = xp.float() - target.detach().float()
    ec = ell_ref.detach().float().clamp_min(1e-30)
    eps2 = (1e-12 * ec).square().clamp_min(1e-36)
    uh = F.relu(torch.sqrt(diff.square().sum(dim=-1) + eps2) - r_halo.detach().float())
    return (
        (uh.square() / (torch.sqrt(ec.square() + uh.square()) + ec).clamp_min(1e-30))
        .mean()
        .to(xp.dtype)
    )


# ---------------------------------------------------------------------------
# Law 3.4: Cleanly Factored Simplicial Manifold Interpolation (3 Cohesive Helpers)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _curve_gate(y_flat: torch.Tensor, g_conn: float) -> float:
    if g_conn <= 1e-4 or int(y_flat.shape[0]) < 12:
        return 0.0
    if int(y_flat.shape[1]) < 2:
        return float(g_conn)
    k_pat = min(9, int(y_flat.shape[0]))
    pat = y_flat.float()[
        safe_cdist(y_flat).topk(k=k_pat, dim=-1, largest=False).indices
    ].double()
    cen = pat - pat.mean(dim=1, keepdim=True)
    gram = (
        torch.bmm(cen.transpose(1, 2), cen)
        if cen.shape[2] < cen.shape[1]
        else torch.bmm(cen, cen.transpose(1, 2))
    )
    lam1, _ = power_principal_axis(gram, iters=6, seed=0)
    tr = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-30)
    r_perp = ((tr - lam1).clamp_min(0.0) / lam1.clamp_min(1e-30)).median()
    return float(g_conn * float(smoothstep01((0.11 - r_perp) / 0.06).item()))


@torch.no_grad()
def _simplicial_gates(
    yf: torch.Tensor, chart_weight: float
) -> Tuple[float, float, float, torch.Tensor, float]:
    if chart_weight <= 0.0 or int(yf.shape[0]) < 4:
        return 0.0, 0.0, 0.1, yf.float().abs().amax().clamp_min(1e-12), 0.0
    g_graph, g_mac_raw, q90_t, y_sc = manifold_connectivity_stats(yf)
    atom_open = smoothstep01((batch_annular_gap_ratio(yf) - 0.15) / 0.20)
    g_conn, g_mac, q90_u = torch.stack(
        (g_graph * atom_open, g_mac_raw * atom_open, q90_t)
    ).tolist()
    return (
        float(g_conn),
        float(g_mac),
        float(q90_u),
        y_sc,
        _curve_gate(yf, float(g_conn)),
    )


@torch.no_grad()
def _sample_simplicial_barycenters(
    zd: torch.Tensor, phi: float, nq: int, g_curv: float
) -> _SimpSamples:
    mp, dev = int(zd.shape[0]), zd.device
    dzz = (
        torch.cdist(zd.double(), zd.double(), p=2.0)
        + 1e-12 * torch.arange(mp, device=dev, dtype=torch.float64).unsqueeze(0)
    ).fill_diagonal_(float("inf"))
    knn = min(4, mp - 1)
    nbrs = dzz.topk(knn, dim=-1, largest=False).indices
    base_e = (nq // 2) if (mp >= 3 and knn >= 2 and nq >= 4) else nq
    nt = int(round((1.0 - g_curv) * float(nq - base_e)))
    ne, re, ke = nq - nt, torch.arange(nq - nt, device=dev), min(3, knn)
    if g_curv > 0.5:
        nd = max(1, mp * ke)
        pid = (re + int(phi * float(nd))) % nd
        v0e, v1e = pid // ke, nbrs[pid // ke, pid % ke]
        ue = (
            (((re // nd).float() + 0.5) / float(max(1, (ne + nd - 1) // nd))) + phi
        ) % 1.0
    else:
        v0e = (re + int(phi * mp)) % mp
        v1e = nbrs[v0e, re % ke]
        ue = (((re.float() + 0.5) / float(max(1, ne))) + phi) % 1.0
    te = (0.20 + 0.60 * ue).to(zd.dtype)
    z_simp = (1.0 - te.unsqueeze(1)) * zd[v0e] + te.unsqueeze(1) * zd[v1e]
    if nt <= 0:
        return _SimpSamples(z_simp, ne, v0e, v1e, te, None, None, None, None)
    rt, pt = torch.arange(nt, device=dev), (phi * PHI2) % 1.0
    v0t, c1 = (rt + int(pt * mp)) % mp, rt % knn
    v1t = nbrs[v0t, c1]
    v2t = nbrs[v0t, (c1 + 1 + ((rt // knn) % max(1, knn - 1))) % knn]
    ang = 2.0 * math.pi * ((((rt.float() + 0.5) / float(nt)) + pt) % 1.0)
    rw = 0.12 * (rt % 2 == 1).to(zd.dtype)
    w0 = (1.0 / 3.0 + rw * torch.cos(ang).to(zd.dtype)).unsqueeze(1)
    w1 = (1.0 / 3.0 + rw * torch.cos(ang + 2.0 * math.pi / 3.0).to(zd.dtype)).unsqueeze(
        1
    )
    z_tri = w0 * zd[v0t] + w1 * zd[v1t] + (1.0 - w0 - w1) * zd[v2t]
    return _SimpSamples(
        torch.cat((z_simp, z_tri), dim=0), ne, v0e, v1e, te, v0t, v1t, v2t, z_tri
    )


def _simplicial_isometry_loss(
    xs: torch.Tensor,
    xv: torch.Tensor,
    zd: torch.Tensor,
    ell: torch.Tensor,
    g_mac: float,
    smp: _SimpSamples,
    use_f64: bool,
) -> torch.Tensor:
    d0e = born_infeld_cost(xs[: smp.ne], xv[smp.v0e], ell, aligned=True, fp64=use_f64)[
        0
    ]
    d1e = born_infeld_cost(xs[: smp.ne], xv[smp.v1e], ell, aligned=True, fp64=use_f64)[
        0
    ]
    iso_l = (
        ((1.0 - smp.te.to(d0e.dtype)) * d0e - smp.te.to(d0e.dtype) * d1e).abs().mean()
    )
    if smp.v0t is None or smp.v1t is None or smp.v2t is None or smp.z_tri is None:
        return iso_l
    d0t = born_infeld_cost(xs[smp.ne :], xv[smp.v0t], ell, aligned=True, fp64=use_f64)[
        0
    ]
    d1t = born_infeld_cost(xs[smp.ne :], xv[smp.v1t], ell, aligned=True, fp64=use_f64)[
        0
    ]
    d2t = born_infeld_cost(xs[smp.ne :], xv[smp.v2t], ell, aligned=True, fp64=use_f64)[
        0
    ]
    with torch.no_grad():
        vx0, vx1, vx2 = xv[smp.v0t].double(), xv[smp.v1t].double(), xv[smp.v2t].double()
        e64 = ell.detach().clamp_min(1e-30).double()
        ea, eb, ec = (
            (vx0 - vx1).norm(dim=-1),
            (vx1 - vx2).norm(dim=-1),
            (vx2 - vx0).norm(dim=-1),
        )
        sp = 0.5 * (ea + eb + ec)
        heron = (
            (
                sp
                * (sp - ea).clamp_min(0.0)
                * (sp - eb).clamp_min(0.0)
                * (sp - ec).clamp_min(0.0)
            )
            .clamp_min(0.0)
            .sqrt()
        )
        mr = 0.70 * (
            torch.sqrt(e64.square() + (heron / sp.clamp_min(1e-8)).square()) - e64
        )
        if g_mac > 1e-4:
            z0, z1, z2, zt = (
                zd[smp.v0t].double(),
                zd[smp.v1t].double(),
                zd[smp.v2t].double(),
                smp.z_tri.detach().double(),
            )
            zdm = torch.maximum(
                torch.maximum((z0 - z1).norm(dim=-1), (z1 - z2).norm(dim=-1)),
                (z2 - z0).norm(dim=-1),
            ).clamp_min(1e-8)
            zn = torch.minimum(
                torch.minimum((zt - z0).norm(dim=-1), (zt - z1).norm(dim=-1)),
                (zt - z2).norm(dim=-1),
            )
            mr = (1.0 - g_mac + g_mac * smoothstep01((zn / zdm - 0.12) / 0.10)) * mr
        mg = mr.to(d0t.dtype)
    return 0.5 * (
        iso_l + (F.relu(mg - d0t) + F.relu(mg - d1t) + F.relu(mg - d2t)).mean() / 3.0
    )


def _simplicial_geodesic_chart_loss(
    z_table: torch.Tensor,
    xv_live: Optional[torch.Tensor],
    xv: torch.Tensor,
    yf: torch.Tensor,
    ell: torch.Tensor,
    g_conn: float,
    g_curv: float,
    q90_u: float,
    y_sc: torch.Tensor,
    phi: float,
    chart_weight: float,
    use_f64: bool,
) -> torch.Tensor:
    mp, br = int(z_table.shape[0]), int(yf.shape[0])
    yu = yf.float() / y_sc
    loss = ell.new_zeros((), dtype=xv.dtype)
    if xv_live is not None:
        cp_vy, _ = born_infeld_cost(xv_live, yf, ell, fp64=use_f64)
        with torch.no_grad():
            lv = (
                -(
                    cp_vy.detach().float()
                    / cp_vy.detach().float().mean().clamp_min(1e-12)
                )
                / 0.08
            )
            pvy = (
                log_sinkhorn(
                    lv,
                    lv.new_full((mp,), -math.log(mp)),
                    lv.new_full((br,), -math.log(br)),
                    iters=10,
                )
                .exp_()
                .to(cp_vy.dtype)
            )
        loss = loss + (1.10 * g_conn) * (pvy * cp_vy).sum()
        if g_curv > 1e-4:
            tgt = yf[(pvy.float() @ safe_cdist(yu)).argmin(dim=1)]
            loss = (
                loss
                + (0.65 * g_conn * g_curv)
                * born_infeld_cost(xv_live, tgt, 0.5 * ell, aligned=True, fp64=use_f64)[
                    0
                ].mean()
            )
    if not z_table.requires_grad:
        return loss
    with torch.no_grad():
        yc = (
            yu
            if (mp + br) <= 512
            else yu[
                (
                    (torch.arange(min(br, 128), device=yu.device) * br) // min(br, 128)
                    + int(phi * br)
                )
                % br
            ]
        )
        vc = torch.cat((xv.float() / y_sc, yc), dim=0)
        dv = safe_cdist(vc.double() if use_f64 else vc).double()
        gm = dv * (1.0 + (dv / max(1e-4, 1.35 * q90_u)).square())
        for _ in range(6):
            gm = torch.minimum(gm, (gm.unsqueeze(2) + gm.unsqueeze(0)).amin(dim=1))
        gvv = gm[:mp, :mp]
        off = ~torch.eye(mp, dtype=torch.bool, device=z_table.device)
        ghat = gvv / gvv[off].square().mean().clamp_min(1e-12).sqrt()
        wl = torch.where(off, (0.20 + ghat.square()).pow(-2.0), 0.0)
        wg = torch.where(off, 1.0 + 2.0 * ghat.square(), 0.0)
        wl_n, wg_n = wl / wl.sum().clamp_min(1e-12), wg / wg.sum().clamp_min(1e-12)
        wch = wl_n if g_curv <= 1e-4 else ((1.0 - g_curv) * wl_n + g_curv * wg_n)
    z64 = z_table.double()
    dz2 = (
        z64.square().sum(dim=-1, keepdim=True)
        + z64.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (z64 @ z64.T)
    )
    dz2 = dz2 + (dz2.detach().clamp_min(0.0) - dz2.detach())
    dzh = torch.sqrt(dz2 + 1e-12) / dz2[off].mean().clamp_min(1e-12).sqrt()
    mds = (wch * (dzh - ghat).square()).sum().to(loss.dtype)
    loss = (
        loss
        + (g_conn * chart_weight * (1.0 + 18.0 * g_curv))
        * ell.detach().to(loss.dtype)
        * mds
    )
    if g_curv > 1e-4 and int(z_table.shape[1]) >= 2:
        zc = z64 - z64.mean(dim=0, keepdim=True)
        fsq = zc.square().sum().clamp_min(1e-12)
        with torch.no_grad():
            _, ax = power_principal_axis((zc.T @ zc).detach(), iters=6, seed=0)
        col = ((fsq - (zc @ ax).square().sum()).clamp_min(0.0) / fsq).to(loss.dtype)
        loss = (
            loss
            + (3.2 * g_conn * g_curv * chart_weight) * ell.detach().to(loss.dtype) * col
        )
    return loss


# ---------------------------------------------------------------------------
# AxiomTransportLoss & Training Seam
# ---------------------------------------------------------------------------


class AxiomTransportLoss(nn.Module):
    """Unified Law 3A + Law 3B transport operator for the Axiom v6 family."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        rho: float = 3.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        chart_capacity: int = 256,
        chart_weight: float = 1.4,
        halo_mult: float = 2.1,
        max_spatial: int = 8,
        max_flat_dim: int = _MAX_FLAT_DIM,
        mean_op_weight: float = 0.36,
        cov_op_weight: float = 0.35,
        imq_active_scale: float = 0.40,
        bank_mean_blend: float = 0.75,
    ) -> None:
        super().__init__()
        self.weight, self.rho, self.tau, self.eps, self.iters = (
            float(weight),
            float(rho),
            float(tau),
            float(eps),
            int(iters),
        )
        self.chart_capacity = max(2, int(chart_capacity))
        self.chart_weight, self.halo_mult = float(chart_weight), float(halo_mult)
        self.max_spatial, self.max_flat_dim = int(max_spatial), int(max_flat_dim)
        self.mean_op_weight, self.cov_op_weight = (
            float(mean_op_weight),
            float(cov_op_weight),
        )
        self.imq_active_scale = float(imq_active_scale)
        self.bank_mean_blend = float(bank_mean_blend)
        self._discrete_step: bool = False
        self._simplicial_step: bool = False
        self._ell_local: Optional[torch.Tensor] = None
        self.last_p_xy_slack: Optional[torch.Tensor] = None
        self._atlas_local: Optional[PrototypeAtlas] = None
        self._record_atlases: Any = _AtlasCache()

    def bind_discrete_prior_step(
        self, active: bool, *, simplicial_active: bool = False
    ) -> None:
        self._discrete_step = bool(active)
        self._simplicial_step = bool(simplicial_active)

    @staticmethod
    def _log_odds(
        df: torch.Tensor, dr: torch.Tensor, c_max: float = 20.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ff = (
            df.reshape(int(df.shape[0]), -1).mean(dim=1)
            if df.ndim > 1
            else df.reshape(-1)
        )
        rf = (
            dr.detach().reshape(int(dr.shape[0]), -1).mean(dim=1)
            if dr.ndim > 1
            else dr.detach().reshape(-1)
        )
        db = 0.5 * (ff.detach().double().mean() + rf.double().mean())
        return (
            c_max * torch.tanh((ff.double() - db).to(ff.dtype) / c_max),
            c_max * torch.tanh((rf.double() - db).to(rf.dtype) / c_max),
        )

    @staticmethod
    def _update_record_ell(
        eh_t: torch.Tensor, record: Any, can_mut: bool, dtype: torch.dtype
    ) -> torch.Tensor:
        if hasattr(record, "dev_ema") and hasattr(record, "_ema_t"):
            prev = record.dev_ema("ell_ema", eh_t.device).to(torch.float32)
            ratio = torch.maximum(
                eh_t / prev.clamp_min(1e-30), prev / eh_t.clamp_min(1e-30)
            )
            alpha = torch.where(
                ratio > 8.0, eh_t.new_tensor(0.80), eh_t.new_tensor(0.10)
            )
            cur = (
                ((1.0 - alpha) * prev + alpha * eh_t)
                if record.ell_initialized
                else eh_t
            )
            if can_mut:
                record._ema_t["ell_ema"] = cur.double()
                record.ell_initialized = True
                return cur.to(dtype)
            return (prev if record.ell_initialized else eh_t).to(dtype)
        eh = float(eh_t.item())
        if can_mut:
            record.ell_ema = (
                (0.90 * float(record.ell_ema) + 0.10 * eh)
                if record.ell_initialized
                else eh
            )
            record.ell_initialized = True
        return eh_t.new_tensor(
            float(record.ell_ema) if record.ell_initialized else eh, dtype=dtype
        )

    def _resolve_ell(
        self, yp: torch.Tensor, record: Any, can_mut: bool
    ) -> torch.Tensor:
        with torch.no_grad():
            eh_t = _batch_ell(yp, 0.25)
            if record is not None and hasattr(record, "ell_initialized"):
                return self._update_record_ell(eh_t, record, can_mut, yp.dtype)
            if can_mut:
                if self._ell_local is None:
                    self._ell_local = eh_t.detach().clone()
                else:
                    prev = self._ell_local.to(yp.device)
                    ratio = torch.maximum(
                        eh_t / prev.clamp_min(1e-30), prev / eh_t.clamp_min(1e-30)
                    )
                    alpha = torch.where(
                        ratio > 8.0, eh_t.new_tensor(0.80), eh_t.new_tensor(0.10)
                    )
                    self._ell_local = (1.0 - alpha) * prev + alpha * eh_t
            src = self._ell_local.to(yp.device) if self._ell_local is not None else eh_t
            return src.to(yp.dtype)

    def _resolve_atlas(self, record: Any, device: torch.device) -> PrototypeAtlas:
        if record is None:
            if self._atlas_local is None:
                self._atlas_local = PrototypeAtlas(
                    max_prototypes=self.chart_capacity, device=device
                )
            return self._atlas_local
        cached = self._record_atlases.get(record)
        cs = getattr(record, "chart_state", None)
        if cached is not None and cached[1] is cs:
            return cached[0]
        if isinstance(cs, dict) and cs.get("schema") == 2 and "proto_c" in cs:
            atlas = _hydrate_atlas_from_dict(cs, self.chart_capacity, device)
        else:
            atlas = PrototypeAtlas(max_prototypes=self.chart_capacity, device=device)
        self._record_atlases[record] = (atlas, cs)
        return atlas

    def _sync_atlas_to_record(self, atlas: PrototypeAtlas, record: Any) -> None:
        if record is None or not hasattr(record, "chart_state"):
            return
        live = _export_atlas_live(atlas)
        record.chart_state = live
        if hasattr(record, "atomic_ledger"):
            record.atomic_ledger = live["ledger"]
        self._record_atlases[record] = (atlas, live)

    def _atlas_routing_term(
        self,
        xp: torch.Tensor,
        yp: torch.Tensor,
        x_fake: torch.Tensor,
        ell: torch.Tensor,
        g_conn: float,
        record: Any,
        can_mut: bool,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[LedgerAssignment],
    ]:
        z0 = ell.new_zeros((), dtype=torch.float32)
        atlas = self._resolve_atlas(record, yp.device)
        if can_mut or not atlas.initialized:
            atlas.update(yp)
        coal = atlas.coalesce(yp if can_mut else None)
        if int(coal.comp_c.shape[0]) < 2:
            if can_mut:
                self._sync_atlas_to_record(atlas, record)
            return None, None, z0, z0, None

        r_halo, nn_cc = _halo_radii(coal, self.halo_mult)
        ptab, brows = recover_active_prior_rows(x_fake, record)
        asgn = (
            atlas.commit_particles(
                xp, brows, int(ptab.shape[0]), coalesced=coal, allow_update=can_mut
            )
            if _valid_particle_table(ptab, brows, int(xp.shape[0]))
            else None
        )
        if can_mut:
            self._sync_atlas_to_record(atlas, record)

        with torch.no_grad():
            d_mid = safe_cdist(0.5 * (yp + yp.roll(1, 0)), coal.comp_c).amin(dim=1)
            d_yp = safe_cdist(yp, coal.comp_c).amin(dim=1)
            ec = ell.detach().float().clamp_min(1e-30)
            wv_c = smoothstep01(
                (
                    torch.quantile(d_mid, 0.75)
                    / torch.maximum(torch.quantile(d_yp, 0.75), ec)
                    - 0.85 * self.rho
                )
                / 1.50
            ) * (1.0 - g_conn)
            bhits = torch.bincount(
                safe_cdist(yp, coal.comp_c).argmin(dim=1),
                minlength=int(coal.comp_c.shape[0]),
            )

        if asgn is not None:
            tk = asgn.batch_comp
            rho_t = (
                atlas._rho_atom_t
                if atlas._rho_atom_t is not None
                else z0.new_full((), atlas._rho_atom_f)
            )
            is_atom = smoothstep01((0.005 - rho_t) / 0.004)
            ell_c = torch.where(
                is_atom > 0.5,
                (0.15 * torch.median(nn_cc)).clamp_min(1e-30),
                ell.detach().float(),
            )
            route_l = _halo_bi_loss(xp, asgn.batch_target, r_halo[tk], ell_c)
            with torch.no_grad():
                d_to = (xp.detach().float() - asgn.batch_target).norm(dim=-1)
                rh_f = r_halo[tk].clamp_min(0.05 * nn_cc[tk])
                ab_sh = (bhits[tk] <= 0).float() * smoothstep01(
                    (rh_f - d_to) / (0.35 * rh_f)
                )
            return route_l, ab_sh, is_atom * asgn.tail_residual, wv_c, asgn

        if not bool(coal.all_iso.item()) or (1.0 - g_conn) <= 1e-4:
            return None, None, z0, wv_c, None
        near_d, near_i = safe_cdist(xp, coal.comp_c).min(dim=1)
        route_l = wv_c.to(xp.dtype) * _halo_bi_loss(
            xp, coal.comp_c[near_i], r_halo[near_i], ell
        )
        ab_sh = (
            wv_c
            * (bhits[near_i] <= 0).float()
            * smoothstep01(
                (r_halo[near_i] - near_d) / (0.35 * r_halo[near_i]).clamp_min(1e-30)
            )
        )
        return route_l, ab_sh, z0, wv_c, None

    @torch.no_grad()
    def _compute_transport_plans(
        self,
        chxy: torch.Tensor,
        chxx: torch.Tensor,
        dft: torch.Tensor,
        drt: torch.Tensor,
        cond_fake: Optional[torch.Tensor],
        cond_real: Optional[torch.Tensor],
        disc_step: bool,
        s_near: Optional[torch.Tensor],
        ab_sh: Optional[torch.Tensor],
    ) -> _PlanOut:
        bf, br = int(chxy.shape[0]), int(chxy.shape[1])
        pxy = sinkhorn_with_slack(
            chxy,
            dft.detach(),
            drt.detach(),
            rho=self.rho,
            tau=self.tau,
            eps=self.eps,
            iters=self.iters,
            mask=_discrete_mask(cond_fake, cond_real, bf, br),
        )
        mxx = _discrete_mask(cond_fake, cond_fake, bf, bf)
        if disc_step and bf > 1:
            ux = (chxx > 1e-12) | torch.eye(bf, dtype=torch.bool, device=chxx.device)
            mxx = ux if mxx is None else (mxx & ux)
        zf = dft.new_zeros(bf)
        pxx_a = sinkhorn_with_slack(
            chxx,
            zf,
            zf,
            rho=self.rho,
            tau=self.tau,
            eps=self.eps,
            iters=self.iters,
            mask=mxx,
        )[:bf, :bf].fill_diagonal_(0.0)
        pxy_a, pxy_s, pxy_cs = pxy[:bf, :br], pxy[:bf, br], pxy[bf, :br]
        self.last_p_xy_slack = pxy_s.detach()
        af = (float(bf) * pxy_a.sum(dim=1)).clamp(0.0, 1.0)
        if s_near is not None:
            af = (af + 0.50 * s_near.to(af.dtype)).clamp(0.0, 1.0)
        sf = (float(bf) * pxy_s).clamp(0.0, 1.0)
        p_sl = (
            pxy_s
            * (1.0 + 0.35 * sf)
            * ((1.0 - ab_sh.to(pxy_s.dtype)) if ab_sh is not None else 1.0)
        )
        return _PlanOut(pxy_a, pxy_s, pxy_cs, pxx_a, af, sf, p_sl)

    def _observe_step_thermal(
        self,
        xp: torch.Tensor,
        yp: torch.Tensor,
        ell: torch.Tensor,
        het: _HetState,
        r_tail_t: torch.Tensor,
        disc_step: bool,
        has_ledger: bool,
        record: Any,
    ) -> None:
        bf, br = int(xp.shape[0]), int(yp.shape[0])
        r5 = _two_sample_residuals(xp, yp, ell, cvx_mu_gain=0.58)
        if not disc_step and not has_ledger and bf >= 8 and br >= 8:
            dyy_t = het.dyy if het.dyy is not None else safe_cdist(yp)
            dxy_t = safe_cdist(xp, yp)
            off_t = ~torch.eye(br, dtype=torch.bool, device=yp.device)
            q_gap = torch.quantile(dxy_t.amin(dim=1), 0.96) - torch.quantile(
                dyy_t.masked_fill(~off_t, float("inf")).amin(dim=1), 0.96
            )
            r_tail_t = torch.maximum(
                r_tail_t,
                het.wv_sup * 1.20 * q_gap / dyy_t[off_t].mean().clamp_min(1e-30),
            )
        pack = [*r5, r_tail_t]
        has_het = het.hv_t is not None and hasattr(record, "het_initialized")
        if has_het and het.hv_t is not None:
            pack.append(het.hv_t)
        vals = torch.stack(pack).tolist()
        if has_het:
            record.het_ema, record.het_initialized = float(vals[6]), True
        record.observe_thermal(
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
            float(vals[4]),
            r_tail=float(vals[5]),
            shock_mu_ratio=0.35,
        )

    def _finalize_thermal(
        self,
        total: torch.Tensor,
        coord_norm: torch.Tensor,
        critic_ot: torch.Tensor,
        critic_rel: torch.Tensor,
        kappa: float,
        xp: torch.Tensor,
        yp: torch.Tensor,
        ell: torch.Tensor,
        het: _HetState,
        r_tail_t: torch.Tensor,
        disc_step: bool,
        simp_step: bool,
        cond_p: bool,
        has_ledger: bool,
        can_mut: bool,
        thermally_scaled: bool,
        record: Any,
    ) -> torch.Tensor:
        sim_on = simp_step or bool(getattr(record, "_simplicial_step", False))
        do_obs = (
            can_mut
            and not sim_on
            and not cond_p
            and int(xp.shape[0]) >= 4
            and int(yp.shape[0]) >= 4
        )
        if do_obs and hasattr(record, "observe_thermal"):
            with torch.no_grad():
                self._observe_step_thermal(
                    xp, yp, ell, het, r_tail_t, disc_step, has_ledger, record
                )
        elif can_mut and het.hv_t is not None and hasattr(record, "het_initialized"):
            record.het_ema, record.het_initialized = float(het.hv_t.item()), True
        sth = float(getattr(record, "thermal_scale", 1.0))
        if thermally_scaled:
            return (
                kappa * (_scale_grad(coord_norm, sth) + critic_ot)
                + (1.0 - kappa) * critic_rel
            )
        return _scale_grad(total, sth)

    def _assemble_transport_terms(
        self,
        xp: torch.Tensor,
        yp: torch.Tensor,
        x_fake: torch.Tensor,
        dft: torch.Tensor,
        drt: torch.Tensor,
        ell: torch.Tensor,
        g_conn: float,
        cond_fake: Optional[torch.Tensor],
        cond_real: Optional[torch.Tensor],
        disc_step: bool,
        can_mut: bool,
        record: Any,
    ) -> Tuple[torch.Tensor, _PlanOut, _HetState, torch.Tensor, bool]:
        bf, br = int(xp.shape[0]), int(yp.shape[0])
        cond_p = cond_fake is not None or cond_real is not None
        exy, exx, ex_ms, ey_ms, dyy_c = _conformal_horizon(xp, yp, ell, g_conn, cond_p)
        z0 = ell.new_zeros((), dtype=torch.float32)
        route_l, ab_sh, r_tail_t, wv_c, asgn = None, None, z0, z0, None
        if not cond_p and self.chart_weight > 0.0 and br >= 4 and bf >= 2:
            route_l, ab_sh, r_tail_t, wv_c, asgn = self._atlas_routing_term(
                xp, yp, x_fake, ell, g_conn, record, can_mut
            )
        if not cond_p and asgn is None and bf >= 8 and br >= 8:
            het = _heteroscedastic_horizon(
                xp, yp, ell, ex_ms, ey_ms, dyy_c, wv_c, g_conn, record
            )
        else:
            ey_loc = ey_ms.to(ell.dtype) if ey_ms is not None else ell
            het = _HetState(exy, exx, ey_loc, z0, None, None, wv_c, None, dyy_c)
        (cxy, chxy), (cxx, chxx) = (
            born_infeld_cost(xp, yp, het.exy),
            born_infeld_cost(xp, xp, het.exx),
        )
        plan = self._compute_transport_plans(
            chxy, chxx, dft, drt, cond_fake, cond_real, disc_step, het.s_near, ab_sh
        )
        coord = (plan.pxy_a * cxy).sum() - 0.5 * (
            plan.pxx_a * torch.sqrt(plan.af.unsqueeze(1) * plan.af.unsqueeze(0)) * cxx
        ).sum()
        if route_l is not None:
            coord = coord + self.chart_weight * route_l.to(coord.dtype)
        if het.nn1_y is not None:
            coord = coord + _hetero_outlier_attraction(
                cxy,
                chxy,
                het.ey_loc,
                het.nn1_y,
                dft,
                plan.sf,
                het.g_het,
                plan.pxy_a,
                self.rho,
            )
        if not cond_p:
            coord = coord + _orphan_macro_loss(
                xp, yp, chxy, plan.pxy_s, plan.pxy_cs, rho=self.rho, g_conn=g_conn
            )
        return coord, plan, het, r_tail_t, (asgn is not None)

    def forward(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        record: Any = None,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
        allow_state_update: bool = True,
        thermally_scaled: bool = False,
    ) -> torch.Tensor:
        """Dimension-free, scale-invariant Law 3A Born-Infeld OT + relativistic critic transport."""
        if self.weight == 0.0:
            return x_fake[:0].sum()
        disc_step, simp_step = bool(self._discrete_step), bool(self._simplicial_step)
        self._discrete_step = self._simplicial_step = False
        bf, br = int(x_fake.shape[0]), int(y_real.shape[0])
        can_mut = _can_mutate(record, allow_state_update)
        cond_p = cond_fake is not None or cond_real is not None

        yp = project_features(y_real.detach(), self.max_spatial, self.max_flat_dim)
        dft, drt = self._log_odds(d_fake, d_real)
        kappa = float(compute_contrast_confidence(yp).item())
        rel_diff = (drt - dft) if bf == br else (drt.unsqueeze(0) - dft.unsqueeze(1))
        critic_rel = 0.50 * self.tau * (F.softplus(rel_diff).mean() - math.log(2.0))
        if kappa <= 1e-4:
            if record is not None and hasattr(record, "_disc_gate"):
                record._disc_gate = 0.0
            return self.weight * critic_rel

        xp = project_features(x_fake, self.max_spatial, self.max_flat_dim)
        ell = self._resolve_ell(yp, record, can_mut)
        s_scale = _coord_scale(xp, yp, ell).to(xp.dtype)
        g_conn = float(manifold_connectivity_stats(yp)[0].item())
        if record is not None and hasattr(record, "_disc_gate"):
            record._disc_gate = 1.0 - g_conn

        coord, plan, het, r_tail_t, has_ledger = self._assemble_transport_terms(
            xp,
            yp,
            x_fake,
            dft,
            drt,
            ell,
            g_conn,
            cond_fake,
            cond_real,
            disc_step,
            can_mut,
            record,
        )
        coord_norm = coord / s_scale
        w_rel = 0.03 * kappa + 0.50 * (1.0 - kappa)
        rel_mat = F.softplus(drt.unsqueeze(0) - dft.unsqueeze(1)) - math.log(2.0)
        critic_ot = (
            -(plan.p_sl * (0.5 * self.tau * dft)).sum()
            + (w_rel * self.tau) * (plan.pxy_a * rel_mat).sum()
        )
        total = kappa * (coord_norm + critic_ot) + (1.0 - kappa) * critic_rel
        if record is not None:
            total = self._finalize_thermal(
                total,
                coord_norm,
                critic_ot,
                critic_rel,
                kappa,
                xp,
                yp,
                ell,
                het,
                r_tail_t,
                disc_step,
                simp_step,
                cond_p,
                has_ledger,
                can_mut,
                thermally_scaled,
                record,
            )
        return self.weight * total

    @staticmethod
    def _observe_simplicial_thermal(
        xv: torch.Tensor,
        xs: torch.Tensor,
        yf: torch.Tensor,
        ell: torch.Tensor,
        g_mac: float,
        g_curv: float,
        record: Any,
    ) -> None:
        if g_mac <= 1e-4 or int(yf.shape[0]) < 4 or not _can_mutate(record):
            return
        if not hasattr(record, "observe_thermal"):
            return
        with torch.no_grad():
            r5 = torch.stack(
                _two_sample_residuals(torch.cat((xv, xs.detach()), dim=0), yf, ell)
            ).tolist()
            record.observe_thermal(
                float(r5[0]),
                float(r5[1]),
                float(r5[2]),
                (1.0 - 0.50 * g_curv) * float(r5[3]),
                float(r5[4]),
                tau=0.084 * (1.0 + 0.60 * g_curv),
                shock_mu_ratio=0.35 + 0.55 * g_curv,
                shock_ed_floor=0.30 + 0.50 * g_curv,
                lock_hysteresis=0.22 * g_curv,
            )

    def simplicial_manifold_loss(
        self,
        z_table: torch.Tensor,
        gen_fn: Callable[[torch.Tensor], torch.Tensor],
        y_real: torch.Tensor,
        *,
        record: Any = None,
        iso_weight: float = 0.45,
        chart_weight: float = 4.0,
    ) -> torch.Tensor:
        """Simplicial manifold interpolation gated smoothly by connectivity ``g_conn in [0, 1]``."""
        mp, br = int(z_table.shape[0]), int(y_real.shape[0])
        if self.weight == 0.0 or mp < 2 or br < 1:
            return y_real[:0].sum()
        phi = float(getattr(record, "chord_phase", 0.0)) if record is not None else 0.0
        if mp > 512:
            idx = (
                (torch.arange(128, device=z_table.device) * mp) // 128 + int(phi * mp)
            ) % mp
            z_table, mp = z_table[idx], 128
        yf, use_f64 = y_real.detach().reshape(br, -1), (mp <= 256)
        with torch.no_grad():
            g_conn, g_mac, q90_u, y_sc, g_curv = _simplicial_gates(yf, chart_weight)
            zd = z_table.detach()
            ell = self._resolve_ell(yf, record, False)
            nq = max(4, br if br <= 512 else 128)
            smp = _sample_simplicial_barycenters(zd, phi, nq, g_curv)

        chart_on = g_conn > 1e-4 and chart_weight > 0.0
        xv_live = gen_fn(z_table).reshape(mp, -1) if chart_on else None
        xv = (
            xv_live.detach()
            if xv_live is not None
            else gen_fn(zd).detach().reshape(mp, -1)
        )
        xs = gen_fn(smp.z_simp).reshape(nq, -1)
        self._observe_simplicial_thermal(xv, xs, yf, ell, g_mac, g_curv, record)

        cp_sr, ch_sr = born_infeld_cost(xs, yf, ell, fp64=use_f64)
        total = (
            (torch.softmax(-ch_sr / self.eps, dim=1).to(cp_sr.dtype) * cp_sr)
            .sum(dim=1)
            .mean()
        )
        eff_iso = (
            ((1.0 - g_mac) * iso_weight + g_mac * 0.25) if g_mac > 1e-4 else iso_weight
        )
        if eff_iso > 0.0:
            total = total + eff_iso * _simplicial_isometry_loss(
                xs, xv, zd, ell, g_mac, smp, use_f64
            )
        if chart_on:
            total = total + _simplicial_geodesic_chart_loss(
                z_table,
                xv_live,
                xv,
                yf,
                ell,
                g_conn,
                g_curv,
                q90_u,
                y_sc,
                phi,
                chart_weight,
                use_f64,
            )
        sth = (
            float(getattr(record, "thermal_scale", 1.0)) if record is not None else 1.0
        )
        return self.weight * _scale_grad(total, sth)

    def _bigan_codebook_loss(
        self,
        fx: torch.Tensor,
        rx: torch.Tensor,
        rz: torch.Tensor,
        ptab: torch.Tensor,
        brows: torch.Tensor,
        record: Any,
        can_mut: bool,
    ) -> Optional[Tuple[torch.Tensor, float]]:
        atlas = self._resolve_atlas(record, rx.device)
        if can_mut or not atlas.initialized:
            atlas.update(rx.detach())
        coal = atlas.coalesce(rx.detach() if can_mut else None)
        if int(coal.comp_c.shape[0]) < 3:
            return None
        asgn = atlas.commit_particles(
            fx, brows, int(ptab.shape[0]), coalesced=coal, allow_update=can_mut
        )
        if can_mut:
            self._sync_atlas_to_record(atlas, record)
        if asgn is None:
            return None

        r_halo, nn_cc = _halo_radii(coal, self.halo_mult)
        ell_x = (0.15 * torch.median(nn_cc)).clamp_min(1e-30)
        l_fwd = _halo_bi_loss(fx, asgn.batch_target, r_halo[asgn.batch_comp], ell_x)

        with torch.no_grad():
            tab_f = ptab.detach().to(device=rx.device, dtype=torch.float32)
            rx_comp = safe_cdist(rx.detach(), coal.comp_c).argmin(dim=1)
            match_m = asgn.row_comp.unsqueeze(0) == rx_comp.unsqueeze(1)
            has_inv = match_m.any(dim=1)
            inv_i = (
                safe_cdist(rz.detach(), tab_f)
                .masked_fill(~match_m, float("inf"))
                .argmin(dim=1)
            )
            nn_z = (
                safe_cdist(tab_f, diag_inf=True).amin(dim=1).clamp_min(1e-4)
                if int(tab_f.shape[0]) >= 2
                else tab_f.new_ones((int(tab_f.shape[0]),))
            )
            inv_t, ell_z, wm = (
                tab_f[inv_i].to(rz.dtype),
                (0.25 * torch.median(nn_z)).clamp_min(1e-4).to(rz.dtype),
                has_inv.to(rz.dtype),
            )
            ir = (
                (rz.detach().float() - inv_t.float()).norm(dim=-1) / nn_z[inv_i]
            ).clamp_max(10.0)
            n_hi = has_inv.float().sum()
            bad = (((ir > 0.12) & has_inv).any() | (~has_inv).any()).float()
            hi_any, bad_inv, ir_mf, r_fwd_f = torch.stack(
                (
                    n_hi,
                    bad,
                    (ir * has_inv.float()).sum() / n_hi.clamp_min(1.0),
                    asgn.tail_residual,
                )
            ).tolist()
        if hi_any > 0.5:
            l_inv = (
                wm * born_infeld_cost(rz, inv_t, ell_z, aligned=True)[0]
            ).sum() / wm.sum().clamp_min(1.0)
            r_inv = max(1.6 * float(ir_mf), 0.20 if bad_inv > 0.5 else 0.0)
        else:
            l_inv, r_inv = rz.new_zeros(()), 0.25
        return l_fwd + l_inv, max(float(r_fwd_f), r_inv)

    def joint_relativistic_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        record: Any = None,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
        allow_state_update: bool = True,
        thermally_scaled: bool = False,
    ) -> torch.Tensor:
        """Joint BiGAN / ByteGAN relativistic transport + unified forward/inverse codebook bijection."""
        if self.weight == 0.0:
            return d_fake[:0].sum()
        bf, br = int(d_fake.shape[0]), int(d_real.shape[0])
        can_mut = _can_mutate(record, allow_state_update)
        xp = project_features(x_fake.detach(), self.max_spatial, self.max_flat_dim)
        yp = project_features(y_real.detach(), self.max_spatial, self.max_flat_dim)
        ell = self._resolve_ell(yp, record, can_mut)
        with torch.no_grad():
            dft, drt = self._log_odds(d_fake.detach(), d_real.detach())
            pxy = sinkhorn_with_slack(
                born_infeld_cost(xp, yp, ell)[1],
                dft,
                drt,
                rho=self.rho,
                tau=self.tau,
                eps=self.eps,
                iters=self.iters,
                mask=_discrete_mask(cond_fake, cond_real, bf, br),
            )
            pact, prs, pcs = pxy[:bf, :br], pxy[:bf, br], pxy[bf, :br]
            self.last_p_xy_slack = prs
        ff = (
            d_fake.reshape(bf, -1).mean(dim=1)
            if d_fake.ndim > 1
            else d_fake.reshape(bf)
        )
        rf = (
            d_real.reshape(br, -1).mean(dim=1)
            if d_real.ndim > 1
            else d_real.reshape(br)
        )
        db = 0.5 * (ff.detach().mean() + rf.detach().mean())
        loss = self.weight * (
            (pact * F.softplus(rf.unsqueeze(0) - ff.unsqueeze(1))).sum()
            + (prs * F.softplus(db - ff)).sum()
            + (pcs * F.softplus(rf - db)).sum()
        )
        if record is not None and cond_fake is None and cond_real is None:
            ptab, brows = recover_active_prior_rows(x_fake, record)
            sp = _split_joint_pair(x_fake, y_real, ptab, brows)
            if sp is not None and ptab is not None and brows is not None:
                cb = self._bigan_codebook_loss(
                    sp[0], sp[2], sp[3], ptab, brows, record, can_mut
                )
                if cb is not None:
                    l_cb, r_tail = cb
                    if can_mut and hasattr(record, "observe_thermal"):
                        record.observe_thermal(0.0, 0.0, 0.0, 0.0, 0.0, r_tail=r_tail)
                    sth = float(getattr(record, "thermal_scale", 1.0))
                    base_l = loss if thermally_scaled else _scale_grad(loss, sth)
                    loss = base_l + _scale_grad(
                        self.weight * max(self.chart_weight, 1.0) * l_cb, sth
                    )
        return loss

    def conditional_fiber_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        c_cont: torch.Tensor,
        *,
        cond_disc: Optional[torch.Tensor] = None,
        record: Any = None,
        allow_state_update: bool = True,
    ) -> torch.Tensor:
        """Dispatch Law 3B conditional fiber loss via ``fiber.conditional_fiber_loss``."""
        return conditional_fiber_loss(
            x_fake,
            y_real,
            c_cont,
            cond_disc=cond_disc,
            record=record,
            mean_op_weight=self.mean_op_weight,
            cov_op_weight=self.cov_op_weight,
            imq_active_scale=self.imq_active_scale,
            bank_mean_blend=self.bank_mean_blend,
            allow_state_update=allow_state_update,
        )


def prepare_axiom_transport_step(
    transport: Optional[AxiomTransportLoss],
    record: Any,
    real: torch.Tensor,
    *,
    prior_table: Optional[torch.Tensor] = None,
    batch_rows: Optional[torch.Tensor] = None,
    is_discrete_prior: bool = False,
    use_simplex: bool = False,
    simplex_chart_weight: float = 0.0,
) -> bool:
    """Seam helper binding active prior rows and atomic/simplicial flags without peeking into atlas internals."""
    if is_discrete_prior and prior_table is not None and batch_rows is not None:
        set_active_prior_rows(prior_table, batch_rows)
        if record is not None:
            record._active_prior_table = prior_table
            record._active_batch_rows = batch_rows.detach()
    is_atomic = False
    if use_simplex and int(real.shape[0]) >= 8:
        max_sp = getattr(transport, "max_spatial", 8)
        max_fl = getattr(transport, "max_flat_dim", _MAX_FLAT_DIM)
        y_proj = project_features(
            real.detach(), max_spatial=max_sp, max_flat_dim=max_fl
        )
        is_atomic = compute_discrete_atomic_gate(y_proj, record=record) > 0.5
    simplicial_active = bool(
        use_simplex and not is_atomic and float(simplex_chart_weight) > 0.0
    )
    if transport is not None and hasattr(transport, "bind_discrete_prior_step"):
        transport.bind_discrete_prior_step(
            bool(is_discrete_prior), simplicial_active=simplicial_active
        )
    if record is not None and hasattr(record, "_simplicial_step"):
        record._simplicial_step = simplicial_active
    return simplicial_active


sym_eigensplit_zero_eigh = sym_eigensplit_halley


def zero_eigh_symmetric(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removed in Axiom v6, use ``math_ops.sym_eigensplit_halley`` or ``power_principal_axis``."""
    raise NotImplementedError(
        "zero_eigh_symmetric was removed in Axiom v6, use math_ops.sym_eigensplit_halley"
    )


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
    "zero_eigh_symmetric",
]
