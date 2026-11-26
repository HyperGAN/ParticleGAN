"""Axiom Module C: Law 3A (Spatial Metric & Coupling) + Law 3B (Conditional Fiber Bank)."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

_PHI1, _PHI2, _BANK_SCHEMA, _MAX_STRATA, _MAX_RING, _MAX_FULL_DC, _MAX_BANK_DC = 0.6180339887498949, 0.7548776662466927, 1, 64, 384, 6, 32
_PKT_KEYS, _SCALAR_KEYS = frozenset({"schema", "d_c", "d_x", "strata"}), ("n_eff", "n_fast", "w_eff", "w_fast")
_F64_KEYS = ("mu_c", "m2_c", "g_pp", "g_px", "g_py", "g_pp_fast", "g_px_fast", "g_py_fast", "s_xx", "s_yy", "ring_c", "ring_x", "ring_y")
_STRATUM_KEYS = frozenset({"key", *_SCALAR_KEYS, *_F64_KEYS})
def _ss01(u: torch.Tensor) -> torch.Tensor: c = u.clamp(0.0, 1.0); return c * c * (3.0 - 2.0 * c)
def _ss01_f(u: float) -> float: c = min(1.0, max(0.0, float(u))); return c * c * (3.0 - 2.0 * c)
def _can_mut(rec: Any, allow: bool = True) -> bool: return bool(allow and torch.is_grad_enabled() and not getattr(rec, "_in_probe", False) and not getattr(rec, "_in_protected_probe", False))
def _scale_grad(t: torch.Tensor, s: float) -> torch.Tensor: return (t * s + t.detach() * (1.0 - s)) if (s < 1.0 and t.requires_grad) else t

@torch.no_grad()
def zero_eigh_symmetric(mat: torch.Tensor, *, sweeps: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact zero-eigh symmetric eigensolver for ``[..., d, d]`` with canonical column signs."""
    dt, d = mat.dtype, int(mat.shape[-1])
    if mat.ndim < 2 or mat.shape[-2] != d or d < 1: raise ValueError(f"expected [..., d, d] with d>=1, got {tuple(mat.shape)}")
    sym = 0.5 * (mat.to(torch.float64) + mat.to(torch.float64).transpose(-1, -2)); bsh, am = sym.shape[:-2], sym.reshape(-1, d, d)
    if d == 2:
        a, b, c = am[:, 0, 0], am[:, 0, 1], am[:, 1, 1]; tr, df = 0.5 * (a + c), 0.5 * (a - c); rad = torch.hypot(df, b); deg = rad <= 1e-30 * (a.abs() + c.abs() + 1e-30)
        vx, vy = torch.where(deg, torch.zeros_like(a), torch.where(df >= 0.0, df + rad, b)), torch.where(deg, torch.ones_like(a), torch.where(df >= 0.0, b, rad - df))
        nrm = torch.hypot(vx, vy).clamp_min(1e-30); c1, s1 = vx / nrm, vy / nrm
        vecs = torch.stack((torch.stack((-s1, c1), -1), torch.stack((c1, s1), -1)), -1).reshape(*bsh, 2, 2); piv = torch.gather(vecs, -2, vecs.abs().argmax(dim=-2, keepdim=True))
        return torch.stack((tr - rad, tr + rad), -1).reshape(*bsh, 2).to(dt), (vecs * torch.where(piv < 0.0, -1.0, 1.0)).to(dt)
    am, nb = am.clone(), int(am.shape[0]); vm = torch.eye(d, dtype=torch.float64, device=am.device).unsqueeze(0).expand(nb, d, d).clone()
    for _ in range(0 if d == 1 else int(sweeps)):
        for p in range(d - 1):
            for q in range(p + 1, d):
                apq, app, aqq = am[:, p, q].clone(), am[:, p, p].clone(), am[:, q, q].clone(); act = apq.abs() > 1e-15 * (app.abs() + aqq.abs() + 1e-30)
                if not bool(act.any()): continue
                tau = (aqq - app) / (2.0 * torch.where(act, apq, torch.ones_like(apq)))
                tv = torch.where(act, torch.where(tau >= 0.0, 1.0, -1.0) / (tau.abs() + torch.hypot(tau, torch.ones_like(tau))), torch.zeros_like(tau))
                cv = torch.rsqrt(1.0 + tv * tv); cc, sc = cv.unsqueeze(1), (tv * cv).unsqueeze(1); rp, rq, vp, vq = am[:, p, :].clone(), am[:, q, :].clone(), vm[:, :, p].clone(), vm[:, :, q].clone()
                am[:, p, :] = am[:, :, p] = cc * rp - sc * rq; am[:, q, :] = am[:, :, q] = sc * rp + cc * rq
                am[:, p, p], am[:, q, q] = app - tv * apq, aqq + tv * apq; am[:, p, q] = am[:, q, p] = torch.where(act, torch.zeros_like(apq), apq); vm[:, :, p], vm[:, :, q] = cc * vp - sc * vq, sc * vp + cc * vq
    vu = torch.diagonal(am, dim1=-2, dim2=-1); oi = torch.argsort(vu, dim=-1)
    vals, vecs = torch.gather(vu, -1, oi).reshape(*bsh, d), torch.gather(vm, -1, oi.unsqueeze(1).expand(nb, d, d)).reshape(*bsh, d, d)
    piv = torch.gather(vecs, -2, vecs.abs().argmax(dim=-2, keepdim=True)); return vals.to(dt), (vecs * torch.where(piv < 0.0, -1.0, 1.0)).to(dt)
@torch.no_grad()
def sym_eigensplit_zero_eigh(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split symmetric ``[..., d, d]`` into PSD ``E_+`` and NSD ``E_-`` with zero eigh."""
    d, dev, dt = int(mat.shape[-1]), mat.device, mat.dtype
    if d <= 4:
        vals, vecs = zero_eigh_symmetric(mat.detach().cpu().to(torch.float64)); ut = vecs.transpose(-1, -2)
        return ((vecs * vals.clamp_min(0.0).unsqueeze(-2)) @ ut).to(device=dev, dtype=dt), ((vecs * vals.clamp_max(0.0).unsqueeze(-2)) @ ut).to(device=dev, dtype=dt)
    sym = 0.5 * (mat.to(torch.float64) + mat.to(torch.float64).transpose(-1, -2)); eye = torch.eye(d, dtype=torch.float64, device=dev)
    s_rms = (sym.square().sum((-2, -1), keepdim=True) / float(d)).sqrt(); x = torch.where(s_rms > 1e-30, sym / s_rms.clamp_min(1e-30), torch.zeros_like(sym))
    for _ in range(7): x2 = x @ x; L, _ = torch.linalg.cholesky_ex(eye + 3.0 * x2, check_errors=False); x = torch.cholesky_solve(x @ (3.0 * eye + x2), L)
    abs_a = 0.5 * (x @ sym + sym @ x); return (0.5 * (sym + abs_a)).to(dt), (0.5 * (sym - abs_a)).to(dt)
def _sym_inv_sqrt(cov_f64: torch.Tensor) -> torch.Tensor:
    s, d = 0.5 * (cov_f64 + cov_f64.T), int(cov_f64.shape[0])
    if d <= 2:
        vals, vecs = zero_eigh_symmetric(s); return (vecs * vals.clamp_min(max(1e-4 * max(float(vals.clamp_min(0.0).mean().item()), 1e-4), 1e-6)).rsqrt().unsqueeze(0)) @ vecs.T
    eye = torch.eye(d, dtype=torch.float64, device=s.device); a = s + max(1e-4 * max(float(torch.diagonal(s).clamp_min(0.0).mean().item()), 1e-4), 1e-6) * eye
    sc = float((a @ a).square().sum().pow(0.25).clamp_min(1e-30).item()) * 0.60; y, z = a / sc, eye.clone()
    for _ in range(16): t = 0.5 * (3.0 * eye - z @ y); y, z = y @ t, t @ z
    return 0.5 * (z + z.T) / math.sqrt(sc)
def _power_axis(gram: torch.Tensor, iters: int = 6) -> torch.Tensor:
    v = torch.remainder(torch.arange(1, gram.shape[0] + 1, device=gram.device, dtype=gram.dtype) * _PHI1 + 0.25, 1.0) - 0.5
    v = v / v.norm().clamp_min(1e-12); vc = gram @ v; n0, vc = vc.norm(), vc / vc.norm().clamp_min(1e-20)
    for _ in range(max(0, iters - 1)): gv = gram @ vc; vc = gv / gv.norm().clamp_min(1e-20)
    v = torch.where(n0 > 1e-20, vc, v); return torch.where(v[v.abs().argmax()] < 0.0, -v, v)
def project_transport_features(x: torch.Tensor, max_spatial: int = 8) -> torch.Tensor:
    """Differentiable CUDA-deterministic energy-preserving spatial projection."""
    b = x.shape[0]
    if x.ndim >= 4 and (x.shape[-2] > max_spatial or x.shape[-1] > max_spatial):
        hi, wi, ho, wo = int(x.shape[-2]), int(x.shape[-1]), min(int(max_spatial), int(x.shape[-2])), min(int(max_spatial), int(x.shape[-1]))
        x4, ph, pw = x.reshape(-1, x.shape[-3], hi, wi).float(), (-hi) % ho, (-wi) % wo
        x4 = F.pad(x4, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x4
        return (x4.reshape(x4.shape[0], x4.shape[1], ho, x4.shape[-2] // ho, wo, x4.shape[-1] // wo).mean((3, 5)).reshape(b, -1) * math.sqrt(float(hi * wi) / float(max(1, ho * wo)))).to(x.dtype)
    return x.reshape(b, -1)
def _sc_safe(v: torch.Tensor) -> torch.Tensor: return torch.where(v < 1e-2, v.clamp_min(1e-12), v.clamp_min(1.0))
@torch.no_grad()
def _cdist_s(a: torch.Tensor, b: Optional[torch.Tensor] = None, *, diag_inf: bool = False) -> torch.Tensor:
    af = a.detach().reshape(a.shape[0], -1).float(); bf = af if b is None else b.detach().reshape(b.shape[0], -1).float()
    sc = _sc_safe(torch.maximum(af.abs().amax(), bf.abs().amax())); d = torch.cdist(af / sc, bf / sc, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") * sc
    return d.fill_diagonal_(float("inf")) if (diag_inf and b is None) else d
def _bi_cost(x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor, *, aligned: bool = False, f64: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = torch.float64 if f64 else torch.float32; xf, yf = x.reshape(x.shape[0], -1).to(dt), y.reshape(y.shape[0], -1).to(dt)
    sc = _sc_safe(torch.maximum(torch.maximum(xf.detach().abs().amax(), yf.detach().abs().amax()), ell.detach().to(dt).amax())); el = ell.detach().to(dt).clamp_min(1e-6 * sc.clamp_max(1.0)); xu, yu, eu = xf / sc, yf / sc, el / sc
    sq = (xu - yu).square().sum(-1) if aligned else (xu.square().sum(-1, keepdim=True) + yu.square().sum(-1).unsqueeze(0) - 2.0 * (xu @ yu.T))
    sq = sq if aligned else (sq + (sq.detach().clamp_min(0.0) - sq.detach())); rad = torch.sqrt(eu.square() + sq); cu = (rad - eu) + (sq / (rad + eu).clamp_min(1e-30) - (rad - eu)).detach()
    return (cu * sc).to(x.dtype), (cu.detach() / eu.clamp_min(1e-30)).to(torch.float32)
def _discrete_mask(cr: Optional[torch.Tensor], cc: Optional[torch.Tensor], br: int, bc: int) -> Optional[torch.Tensor]:
    if cr is None or cc is None or cr.shape[0] != br or cc.shape[0] != bc: return None
    rf, cf = cr.detach().reshape(br, -1), cc.detach().reshape(bc, -1)
    return (rf.unsqueeze(1) == cf.unsqueeze(0)).all(-1) if rf.shape[1] == cf.shape[1] else None
def _log_sinkhorn(L: torch.Tensor, lr: torch.Tensor, lc: torch.Tensor, iters: int) -> torch.Tensor:
    buf, u, v = torch.empty_like(L), torch.zeros_like(lr), torch.zeros_like(lc)
    for _ in range(max(1, iters)):
        torch.add(L, u.unsqueeze(1), out=buf); m0 = buf.amax(0, keepdim=True); v = lc - (m0.squeeze(0) + buf.sub_(m0).exp_().sum(0).clamp_min(1e-30).log_())
        torch.add(L, v.unsqueeze(0), out=buf); m1 = buf.amax(1, keepdim=True); u = lr - (m1.squeeze(1) + buf.sub_(m1).exp_().sum(1).clamp_min(1e-30).log_())
    return torch.add(L, u.unsqueeze(1), out=buf).add_(v.unsqueeze(0))
def _sinkhorn_slack(c_hat: torch.Tensor, dr: torch.Tensor, dc: torch.Tensor, *, rho: float, tau: float, eps: float, iters: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    br, bc = c_hat.shape; L = c_hat.new_zeros((br + 1, bc + 1), dtype=torch.float32)
    act = -c_hat.float().clamp(0.0, 1e6) / eps; L[:br, :bc] = act.masked_fill(~mask, float("-inf")) if mask is not None else act
    L[:br, bc], L[br, :bc] = (-0.5 * rho + 0.5 * tau * dr.float()) / eps, (-0.5 * rho - 0.5 * tau * dc.float()) / eps
    lr, lc = L.new_full((br + 1,), -math.log(max(1, br))), L.new_full((bc + 1,), -math.log(max(1, bc))); lr[br] = lc[bc] = math.log(2.0)
    return _log_sinkhorn(L, lr, lc, iters).exp_().to(c_hat.dtype)
def _batch_ell(y_flat: torch.Tensor, q: float = 0.25) -> torch.Tensor:
    if y_flat.shape[0] <= 1: return y_flat.new_tensor(1.0, dtype=torch.float32)
    dim = max(1, int(y_flat.shape[-1])); akt_e = max(0.0, 1.0 / float(dim) - 0.5)
    akt_f = (((2.0 * math.e * float(y_flat.shape[0])) / max(1e-6, -math.log(max(1e-6, 1.0 - float(q))))) ** akt_e) if akt_e > 0.0 else 1.0
    sc = _sc_safe(y_flat.detach().float().abs().amax()); d = _cdist_s(y_flat) / sc; dm = d.amax(); thr = 1e-5 * dm
    pmin, qr = torch.where(d > thr, d, float("inf")).amin(), torch.quantile(d.fill_diagonal_(float("inf")).amin(-1), q)
    val = torch.where(qr <= thr, torch.where(torch.isfinite(pmin), 0.15 * pmin, 0.05 * dm), qr * akt_f).clamp_min(1e-5)
    return torch.where(dm <= 0.0, y_flat.new_tensor(1e-3, dtype=torch.float32), val) * sc
def _valley(yf: torch.Tensor, dyy: torch.Tensor) -> torch.Tensor:
    jf = torch.topk(dyy, k=min(yf.shape[0] - 1, 2), dim=-1, largest=True).indices[:, -1]
    dmy, ix = _cdist_s(0.5 * (yf + yf[jf]), yf), torch.arange(yf.shape[0], device=yf.device); dmy[ix, ix] = dmy[ix, jf] = float("inf"); return torch.quantile(dmy.amin(-1), 0.85)
@torch.no_grad()
def _manifold_gates(y_flat: torch.Tensor) -> Tuple[float, float, float, torch.Tensor]:
    br, ys = y_flat.shape[0], _sc_safe(y_flat.float().abs().amax())
    if br < 4: return 0.0, 0.0, 0.1, ys
    yu = (y_flat.float() / ys)[:: max(1, br // 256)] if br > 512 else (y_flat.float() / ys); dyy, off = _cdist_s(yu), ~torch.eye(yu.shape[0], dtype=torch.bool, device=yu.device)
    q90 = torch.quantile(dyy.masked_fill(~off, float("inf")).amin(-1), 0.90).clamp_min(1e-8); span, m = torch.quantile(dyy[off], 0.90) / q90, dyy.clone()
    for _ in range(8): m = torch.minimum(m, torch.maximum(m.unsqueeze(2), m.unsqueeze(0)).amin(1))
    sc_sp = _ss01((10.0 - m.amax() / q90) / 3.0) * _ss01((span - 1.15) / 0.25)
    v = torch.stack((sc_sp, sc_sp * _ss01((span - 3.80) / 1.00), q90.clamp_min(1e-4))).tolist(); return float(v[0]), float(v[1]), float(v[2]), ys
def _curve_gate(y_flat: torch.Tensor, g_conn: float) -> float:
    if g_conn <= 1e-4 or y_flat.shape[0] < 12: return 0.0
    if y_flat.shape[1] < 2: return float(g_conn)
    yu = y_flat.float() / _sc_safe(y_flat.float().abs().amax()); pat = yu[_cdist_s(yu).topk(k=min(9, y_flat.shape[0]), dim=-1, largest=False).indices].double(); cen = pat - pat.mean(1, keepdim=True)
    ev = zero_eigh_symmetric(torch.bmm(cen.transpose(1, 2), cen) if cen.shape[2] < cen.shape[1] else torch.bmm(cen, cen.transpose(1, 2)), sweeps=6)[0].clamp_min(0.0)
    return float(g_conn * float(_ss01((0.11 - torch.median(ev[:, -2] / ev[:, -1].clamp_min(1e-12))) / 0.06).item()))
@torch.no_grad()
def _conformal_horizon(xp: torch.Tensor, yp: torch.Tensor, ell: torch.Tensor, g_conn: float, cond_p: bool, *, return_aux: bool = False) -> Tuple[Any, ...]:
    bf, br = xp.shape[0], yp.shape[0]
    if g_conn > 0.5 or cond_p or bf < 12 or br < 12: return (ell, ell, None, None, None) if return_aux else (ell, ell)
    xf, yf = xp.detach().float(), yp.detach().float(); sc = _sc_safe(torch.maximum(torch.maximum(xf.abs().amax(), yf.abs().amax()), ell.detach().float())); ef, eg = 1e-6 * sc.clamp_max(1.0), ell.detach().float().clamp_min(1e-6 * sc.clamp_max(1.0))
    yu, xu = yf / sc, xf / sc; dyy = torch.cdist(yu, yu, p=2.0) * sc; vals = dyy.clone().fill_diagonal_(float("inf")).topk(min(3, br - 1), dim=1, largest=False).values
    d1, d2, d3 = vals[:, 0].clamp_min(ef), vals[:, min(1, vals.shape[1] - 1)].clamp_min(ef), vals[:, min(2, vals.shape[1] - 1)].clamp_min(ef)
    rpt = torch.where((d2 <= 2.8 * d1) & (d3 <= 2.8 * d2), d2, d1); pds, pix = dyy.topk(min(8, br), dim=1, largest=False)
    inm = (pds <= (3.2 * d2).unsqueeze(1)).float(); inm[:, 0] = 1.0; rp = (rpt[pix] * inm).sum(1) / inm.sum(1).clamp_min(1.0)
    gw_t = _ss01((torch.quantile(rp, 0.85) / torch.quantile(rp, 0.15).clamp_min(ef) - 2.85) / 0.85)
    if float(gw_t.item()) <= 1e-4: return (ell, ell, None, None, dyy) if return_aux else (ell, ell)
    sraw = (1.0 - gw_t) * eg + gw_t * (eg * (rp / torch.quantile(rp, 0.25).clamp_min(ef))).clamp(0.70 * eg, 4.50 * eg); tbw = 2.0 * (2.4 * sraw / sc).clamp_min(1e-5).unsqueeze(0).square()
    ey, ex = (torch.softmax(-(dyy / sc).square() / tbw, dim=1) @ sraw).clamp_min(ef), (torch.softmax(-torch.cdist(xu, yu, p=2.0).square() / tbw, dim=1) @ sraw).clamp_min(ef)
    exy, exx = torch.sqrt(ex.unsqueeze(1) * ey.unsqueeze(0)).to(ell.dtype), torch.sqrt(ex.unsqueeze(1) * ex.unsqueeze(0)).to(ell.dtype)
    return (exy, exx, ex, ey, dyy) if return_aux else (exy, exx)
def _orphan_macro_loss(xp: torch.Tensor, yp: torch.Tensor, ch_xy: torch.Tensor, prs: torch.Tensor, pcs: torch.Tensor, *, rho: float, g_conn: float) -> torch.Tensor:
    bf, br = xp.shape[0], yp.shape[0]
    if bf < 2 or br < 2 or g_conn > 0.05: return xp[:0].sum()
    with torch.no_grad():
        g_o, dxy, yf = _ss01((ch_xy.detach().float().amin(0) - 2.80 * rho) / max(1.20 * rho, 1e-6)), _cdist_s(xp, yp), yp.detach().float()
        if br >= 8 and bf >= 8:
            dyy = _cdist_s(yf); dyy_i, kl = dyy.clone().fill_diagonal_(float("inf")), min(4, br - 1, bf)
            g_o = g_o * _ss01((dxy.topk(kl, dim=0, largest=False).values[-1] / dyy_i.topk(kl, dim=1, largest=False).values[:, -1].clamp_min(1e-24) - 3.52) / 1.30)
            vq = _valley(yf, dyy); wv, hb = _ss01((vq / torch.quantile(dyy_i.amin(1), 0.50).clamp_min(1e-24) - 3.20) / 1.80), (0.38 * vq).clamp_min(1e-12)
            py, px = torch.exp(-0.5 * (dyy / hb).square()).mean(1).clamp(1e-4, 1.0), torch.exp(-0.5 * (dxy / hb).square()).mean(0)
            g_o = g_o * ((1.0 - wv) + wv * _ss01((bf * py - 4.2) / 2.3) * _ss01((0.45 - px / py) / 0.20))
        qr = (br * pcs.detach().float()).clamp(0.0, 1.0) * g_o; mo = float(qr.mean().item()); ke = bf if mo >= 0.85 else min(bf - 1, int(round(bf * mo)))
        if mo <= 1e-4 or ke < 1: return xp[:0].sum()
        md, oa = torch.quantile(dxy.reshape(-1), 0.50).clamp_min(1e-12), (g_o > 0.05).float(); dm = torch.where((oa > 0.0).any(), (dxy + (1.0 - oa).unsqueeze(0) * (1e6 * md)).amin(1), dxy.amin(1))
        tie = 1e-4 * torch.remainder((torch.arange(bf, device=xp.device, dtype=torch.float32) + 0.5) * _PHI1, 1.0)
        ei = torch.arange(bf, device=xp.device) if ke == bf else torch.sort(torch.topk(1.50 * (bf * prs.detach().float()).clamp(0.0, 1.0) - 0.65 * (dm / md).clamp_max(4.0) + tie, k=ke).indices).values
        qm = (qr + 1e-7) / (qr + 1e-7).sum().clamp_min(1e-12); yc = (yf - (qm.unsqueeze(1) * yf).sum(0, keepdim=True)) * qm.sqrt().unsqueeze(1)
        co = torch.argsort(yf @ _power_axis((yc.T @ yc).double(), 6).float()) if yf.shape[1] >= 2 else torch.argsort(yf[:, 0])
        pi = _log_sinkhorn(-(dxy[ei][:, co] / md) / 0.03, dxy.new_full((ke,), -math.log(ke)), qm[co].clamp_min(1e-12).log(), 10).exp_()
        cdf = torch.cumsum(pi / pi.sum(1, keepdim=True).clamp_min(1e-12), 1); cdf[:, -1] = 1.0; ie = torch.arange(ke, device=xp.device)
        ym = yp.detach()[co[torch.searchsorted(cdf, ((torch.where(ie % 2 == 0, ie // 2, (ke + 1) // 2 + ie // 2).float() + 0.5) / float(ke)).clamp(1e-6, 1.0 - 1e-6).unsqueeze(1)).squeeze(1).clamp(0, br - 1)]]
    return _bi_cost(xp[ei], ym, (0.25 * md).clamp_min(1e-24), aligned=True)[0].sum() / float(bf)
@torch.no_grad()
def _atomic_gate_t(y_cf: torch.Tensor, *, charts: Optional[torch.Tensor] = None, radii: Optional[torch.Tensor] = None) -> torch.Tensor:
    z0 = y_cf.new_zeros((), dtype=torch.float32)
    if y_cf.shape[0] < 8: return z0
    hit_c = y_cf.new_zeros((), dtype=torch.bool)
    if charts is not None and charts.shape[0] >= 4:
        cf = charts.detach().to(device=y_cf.device, dtype=torch.float32); mnn = torch.quantile(_cdist_s(cf, diag_inf=True).amin(1), 0.50)
        hit_c = (mnn > 1e-12) & (torch.quantile(_cdist_s(y_cf, cf).amin(1), 0.75) < 1e-3 * mnn) & ((torch.quantile(radii.float(), 0.75) if radii is not None else z0) < 1e-3 * mnn)
    d = _cdist_s(y_cf); dm = d.amax(); tol, nn_d = 1e-6 * dm, d.clone().fill_diagonal_(float("inf")).amin(1)
    na, sep_m = (nn_d <= tol).float().sum(), d > tol; n_sep = sep_m.sum(); ds = torch.where(sep_m, d, d.new_tensor(float("inf"))).reshape(-1).sort().values
    ntol = torch.maximum(10.0 * tol, 0.25 * ds[(n_sep // 20).clamp_max(ds.numel() - 1)]); near = ((nn_d > tol) & (nn_d < ntol)).float().sum()
    u = ((na / (na + near).clamp_min(1.0) - 0.70) / 0.20).clamp(0.0, 1.0)
    return torch.where(hit_c, z0.new_ones(()), torch.where((dm > 1e-24) & (na >= 2.0) & (n_sep > 0), u * u * (3.0 - 2.0 * u), z0))
@torch.no_grad()
def compute_discrete_atomic_gate(y_cf: torch.Tensor, *, charts: Optional[torch.Tensor] = None, radii: Optional[torch.Tensor] = None) -> float:
    """Annular-gap duplicate purity gate: 1.0 on exact discrete atoms, 0.0 on interp_* & noisy MoGs."""
    return float(_atomic_gate_t(y_cf, charts=charts, radii=radii).item()) if y_cf.shape[0] >= 8 else 0.0
@torch.no_grad()
def _hamilton_quotas(m_tot: int, weights: torch.Tensor, active_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
    k, dev = int(weights.numel()), weights.device; w = weights.detach().float().clamp_min(0.0); ws = float(w.sum().item())
    w = (w / ws) if ws > 1e-12 else torch.full((k,), 1.0 / max(1, k), device=dev); weyl = torch.remainder((torch.arange(k, device=dev, dtype=torch.float32) + 0.5) * _PHI1, 1.0)
    tie = active_scores.detach().float().to(dev) if active_scores is not None else weyl
    if m_tot >= k:
        rem = m_tot - k
        if rem == 0: return torch.ones(k, dtype=torch.long, device=dev)
        ex = (w * float(m_tot) - 1.0).clamp_min(0.0); es = float(ex.sum().item()); sc = (w * float(rem)) if es <= 1e-12 else (ex * (float(rem) / es))
        q, frac = torch.ones(k, dtype=torch.long, device=dev) + torch.floor(sc).long(), sc - torch.floor(sc)
    else: raw = float(m_tot) * w; q, frac = torch.floor(raw).long(), raw - torch.floor(raw)
    rem2 = m_tot - int(q.sum().item())
    if rem2 > 0:
        oi = torch.argsort(weyl, descending=True, stable=True); oi = oi[torch.argsort(tie[oi], descending=True, stable=True)]; q[oi[torch.argsort(frac[oi], descending=True, stable=True)][:rem2]] += 1
    return q
@torch.no_grad()
def _update_atomic_ledger(xf: torch.Tensor, rows: torch.Tensor, charts: torch.Tensor, weights: torch.Tensor, m_tot: int, prev: Optional[Dict[str, Any]], *, allow_update: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], float]:
    b, k, orig_dev = xf.shape[0], charts.shape[0], xf.device; dev = torch.device("cpu") if (xf.is_cuda and m_tot <= 512 and k <= 256) else orig_dev
    cf, xf32 = charts.detach().to(device=dev, dtype=torch.float32).reshape(k, -1), xf.detach().to(device=dev, dtype=torch.float32).reshape(b, -1)
    nn_cc = _cdist_s(cf, diag_inf=True).amin(1).clamp_min(1e-12) if k >= 2 else torch.ones((k,), device=dev)
    ok = prev is not None and isinstance(prev.get("row_chart"), torch.Tensor) and prev["row_chart"].numel() == m_tot and prev["row_target"].shape == (m_tot, cf.shape[1])
    rc, rt = (prev["row_chart"].to(device=dev, dtype=torch.long).clone(), prev["row_target"].to(device=dev, dtype=torch.float32).clone()) if ok else (torch.full((m_tot,), -1, dtype=torch.long, device=dev), torch.zeros((m_tot, cf.shape[1]), dtype=torch.float32, device=dev))
    rd = prev["row_dist"].to(device=dev, dtype=torch.float32).clone() if (ok and isinstance(prev.get("row_dist"), torch.Tensor) and prev["row_dist"].numel() == m_tot) else torch.where(rc >= 0, torch.zeros(m_tot, device=dev), torch.full((m_tot,), 1e30, device=dev))
    if bool((rc >= 0).any()):
        ia = torch.nonzero(rc >= 0, as_tuple=False).squeeze(-1); rc[ia] = _cdist_s(rt[ia], cf).argmin(1); rd[ia] = torch.maximum(rd[ia], (rt[ia] - cf[rc[ia]]).norm(dim=1)); rt[ia] = cf[rc[ia]]
    rl = rows.detach().reshape(-1).long().to(dev); ur, inv = torch.unique(rl, sorted=True, return_inverse=True)
    fpos = torch.full((ur.numel(),), b, dtype=torch.long, device=dev).scatter_reduce_(0, inv, torch.arange(b, device=dev), reduce="amin", include_self=True)
    d_xc = _cdist_s(xf32[fpos], cf); quotas = _hamilton_quotas(m_tot, weights.to(dev), active_scores=-d_xc.amin(0))
    if allow_update:
        hc = rc[ur] >= 0
        if bool(hc.any()): rd[ur[hc]] = d_xc[hc, rc[ur[hc]]]
        aid = torch.nonzero(rc >= 0, as_tuple=False).squeeze(-1)
        if aid.numel() > 0:
            o = torch.argsort(rd[aid].clamp_max(1e6) + 1e-6 * torch.remainder((aid.float() + 0.5) * _PHI1, 1.0), stable=True); o = o[torch.argsort(rc[aid][o], stable=True)]
            sid, sc_o = aid[o], rc[aid][o]; cnt = torch.bincount(sc_o, minlength=k); ev = sid[torch.arange(sid.numel(), device=dev) - (torch.cumsum(cnt, 0) - cnt)[sc_o] >= quotas[sc_o]]; rc[ev] = -1; rd[ev] = 1e30
        deficit, un_u = (quotas - torch.bincount(rc[rc >= 0], minlength=k)).clamp_min(0), torch.nonzero(rc[ur] < 0, as_tuple=False).squeeze(-1)
        if un_u.numel() > 0 and bool((deficit > 0).any()):
            for u_pos in torch.argsort(d_xc[un_u].masked_fill((deficit <= 0).unsqueeze(0), float("inf")).amin(1), stable=True):
                if not bool((deficit > 0).any()): break
                ui = un_u[u_pos]; bci = d_xc[ui].masked_fill(deficit <= 0, float("inf")).argmin(); rc[ur[ui]], rt[ur[ui]], rd[ur[ui]], deficit[bci] = bci, cf[bci], d_xc[ui, bci], deficit[bci] - 1
    tk = torch.where(rc[rl] >= 0, rc[rl].clamp_min(0), _cdist_s(xf32, cf).argmin(1)); occ = torch.bincount(rc[rc >= 0], minlength=k)
    n_uns = int(m_tot - int(((rc >= 0) & (rd <= 0.09 * nn_cc[rc.clamp_min(0)])).sum().item())); r_cov = (0.18 + 0.12 * (float(max(n_uns, 1)) / float(max(1, m_tot)))) if (n_uns > 0 or bool((occ < quotas).any())) else 0.0
    return tk.to(orig_dev), nn_cc.to(orig_dev), {"row_chart": rc, "row_target": rt, "row_dist": rd}, max(1.6 * float(((xf32 - cf[tk]).norm(dim=1) / nn_cc[tk]).clamp_max(10.0).mean().item()), r_cov)
@torch.no_grad()
def _update_charts(y_cf: torch.Tensor, prev: Optional[Dict[str, Any]], *, max_charts: int, allow_update: bool, loc_ell: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    b, dim, dev, kcap, sc = y_cf.shape[0], y_cf.shape[1], y_cf.device, max(2, int(max_charts)), _sc_safe(y_cf.abs().amax())
    is_init = isinstance(prev, dict) and isinstance(prev.get("sum"), torch.Tensor) and prev["sum"].ndim == 2 and prev["sum"].shape[1] == dim and prev["sum"].shape[0] > 0
    cp = dmin = asg = None
    if is_init:
        cs, cc, cq = (prev[k].to(device=dev, dtype=torch.float32) for k in ("sum", "count", "sq_dev"))
        fm = (prev["fake_mass"] if isinstance(prev.get("fake_mass"), torch.Tensor) and prev["fake_mass"].shape == cc.shape else (cc / cc.sum().clamp_min(1e-8))).to(device=dev, dtype=torch.float32)
        cp = cs / cc.clamp_min(1e-8).unsqueeze(1); dmin, asg = _cdist_s(y_cf, cp).min(1)
        if not allow_update: return cp, cc / cc.sum().clamp_min(1e-8), (cq / cc.clamp_min(1e-8)).clamp_min(1e-24).sqrt(), F.one_hot(asg, num_classes=cs.shape[0]).float().sum(0), prev
    else:
        pd = torch.remainder(torch.arange(1, dim + 1, device=dev, dtype=torch.float32) * _PHI1, 1.0) - 0.5; c0 = y_cf[torch.argmin((y_cf / sc) @ (pd / pd.norm().clamp_min(1e-8)))].unsqueeze(0).clone()
        cs, cc, cq, fm = c0 * 0.5, torch.full((1,), 0.5, device=dev), (_batch_ell(y_cf, 0.25).square().clamp_min(1e-24) * 0.5).reshape(1).clamp_min(1e-24), torch.ones((1,), device=dev)
    if cs.shape[0] < kcap:
        ce, re = (cp if cp is not None else cs / cc.clamp_min(1e-8).unsqueeze(1)), (cq / cc.clamp_min(1e-8)).clamp_min(1e-24).sqrt(); sg = _sc_safe(torch.maximum(sc, ce.abs().amax()))
        yug = y_cf / sg; dg = _cdist_s(yug, ce / sg); dmg = dg.amax().clamp_min(1e-5); eug = torch.minimum(_batch_ell(yug, 0.25), (0.04 * dmg).clamp_min(1e-5)); mw, nr = dg.square().min(1)
        eu_i = torch.maximum((loc_ell.detach().float() / sg).clamp_max(0.18 * dmg), eug) if (loc_ell is not None and loc_ell.shape[0] == b) else eug.expand(b)
        tw, e2_i, adds = (2.2 * torch.maximum(torch.minimum(re[nr] / sg, (0.06 * dmg).clamp_min(1e-5)), eu_i)).square(), (2.1 * eu_i).square(), []
        if bool((mw > tw).any()):
            yc_c, mw_c, tw_c, e2_c = yug.cpu(), mw.cpu(), tw.cpu(), e2_i.cpu()
            while cs.shape[0] + len(adds) < kcap:
                bv, nx = torch.where(mw_c > tw_c, mw_c, mw_c.new_tensor(-1.0)).max(0)
                if float(bv.item()) <= 0.0: break
                ni = int(nx.item()); adds.append(ni); dn = (yc_c - yc_c[ni]).square().sum(1); tw_c, mw_c = torch.where(dn < mw_c, torch.maximum(e2_c, e2_c[ni]), tw_c), torch.minimum(mw_c, dn); mw_c[ni] = -1.0
        if adds:
            ai = torch.tensor(adds, device=dev, dtype=torch.long); ca, an = y_cf[ai], (cc.sum() / float(max(b, cs.shape[0] + len(adds)))).clamp_min(1e-3).expand(len(adds))
            cs, cc, cq, fm = torch.cat([cs, ca * an.unsqueeze(1)], 0), torch.cat([cc, an], 0), torch.cat([cq, ((eu_i[ai] * sg).square().clamp_min(1e-24) * an).clamp_min(1e-24)], 0), torch.cat([fm, torch.zeros((len(adds),), device=dev)], 0); cp = dmin = asg = None
    if cp is None or dmin is None or asg is None: cp = cs / cc.clamp_min(1e-8).unsqueeze(1); dmin, asg = _cdist_s(y_cf, cp).min(1)
    oh = F.one_hot(asg, num_classes=cs.shape[0]).float(); bcnt = oh.sum(0); al0 = min(0.35, max(0.025, 0.08 * ((float(b) / float(cs.shape[0])) ** 0.45)))
    shock_t = ((torch.sqrt(dmin.square().mean() / (2.0 * y_cf.var(0, unbiased=False).sum()).clamp_min(1e-24)) - 0.40) / 0.30).clamp(0.0, 1.0) if is_init else dmin.new_zeros(())
    al = (al0 + (0.65 - al0) * shock_t * shock_t).clamp_max(0.65)
    ns = ((1.0 - al) * cs + al * (oh.T @ y_cf)) if is_init else torch.where((bcnt >= 1.0).unsqueeze(1), oh.T @ y_cf, cp * 0.5); nc = ((1.0 - al) * cc + al * bcnt) if is_init else bcnt.clamp_min(0.5)
    nq = (((1.0 - al) * cq + al * (oh.T @ (y_cf - cp[asg]).square().sum(1))) if is_init else (oh.T @ (y_cf - (ns / nc.unsqueeze(1))[asg]).square().sum(1))).clamp_min(1e-24)
    return ns / nc.clamp_min(1e-8).unsqueeze(1), nc / nc.sum().clamp_min(1e-8), (nq / nc.clamp_min(1e-8)).clamp_min(1e-24).sqrt(), bcnt, {"sum": ns, "count": nc, "sq_dev": nq, "fake_mass": fm.clone(), "revival_phase": 0.0, "revivals": 0, "data_scale": sc.detach()}
@torch.no_grad()
def _deconflicted_routing(xf: torch.Tensor, cf: torch.Tensor, wr: torch.Tensor, rads: torch.Tensor, bcnt: torch.Tensor, cur_fm: torch.Tensor, ec: torch.Tensor, *, wv_c: float | torch.Tensor, halo_mult: float, s_th: float, allow_update: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bf, k, dev = xf.shape[0], cf.shape[0], xf.device
    if k >= 2: nn_cc = _cdist_s(cf, diag_inf=True).amin(1).clamp_min(1e-12); r_eff = rads.float().clamp_min(0.02 * nn_cc); r_halo, r_bas = torch.minimum(float(halo_mult) * r_eff, 0.35 * nn_cc), torch.minimum(1.35 * torch.minimum(float(halo_mult) * r_eff, 0.35 * nn_cc), 0.45 * nn_cc)
    else: r_eff = rads.float().clamp_min(1e-12); r_halo, r_bas = float(halo_mult) * r_eff, 1.35 * float(halo_mult) * r_eff
    d_near, near_i = _cdist_s(xf, cf).min(1); bf_frac = (F.one_hot(near_i, num_classes=k).float() * (d_near <= r_bas[near_i]).float().unsqueeze(1)).sum(0) / float(max(1, bf))
    al_f = min(0.35, max(0.025, 0.08 * ((float(bf) / float(max(1, k))) ** 0.45))); next_fm = ((1.0 - al_f) * cur_fm.float() + al_f * bf_frac) if allow_update else cur_fm.float()
    se_k = torch.sqrt((2.0 * (al_f / max(1e-6, 2.0 - al_f)) / float(max(1, bf)) * wr.float()).clamp_min(1e-8)); def_k, sur_k = F.relu(wr.float() - next_fm - 1.60 * se_k), F.relu(next_fm - wr.float() - 1.60 * se_k)
    wv_t = wv_c.to(device=dev, dtype=torch.float32) if isinstance(wv_c, torch.Tensor) else xf.new_full((), float(wv_c), dtype=torch.float32)
    ab_sh = (wv_t * _ss01((r_halo[near_i] - d_near) / (0.35 * r_halo[near_i]).clamp_min(1e-12)) * (bcnt <= 0.0).float()[near_i] * (sur_k <= 1e-6).float()[near_i]) if k > 4 else torch.zeros(bf, device=dev)
    if k < 3 or (not isinstance(wv_c, torch.Tensor) and float(wv_c) <= 1e-4): return cf[near_i], r_halo[near_i], torch.zeros(bf, device=dev), ab_sh, next_fm
    g_val = wv_t * _ss01((d_near - r_halo[near_i]) / (0.45 * r_halo[near_i]).clamp_min(1e-12)); wt = wr.float() + 2.0 * def_k; wt = wt / wt.sum().clamp_min(1e-8)
    bi = torch.sqrt(ec.square() + _cdist_s(xf, cf).square()) - ec; lpi = _log_sinkhorn(-(bi / bi.median().clamp_min(1e-8)).clamp_max(200.0) / 0.08, xf.new_full((bf,), -math.log(max(1, bf)), dtype=torch.float32), wt.clamp_min(1e-12).log(), 8)
    pk_c = torch.cat((lpi.detach(), (wt * float(bf)).unsqueeze(0)), dim=0).cpu(); scpu, rq = pk_c[:bf], pk_c[bf]
    q = torch.floor(rq).long(); rem = bf - int(q.sum().item())
    if rem > 0: q[torch.topk((rq - q.float()) + 1e-7 * torch.remainder((torch.arange(k, dtype=torch.float32) + 0.5) * _PHI1, 1.0), k=min(rem, k), largest=True).indices] += 1
    top2 = torch.topk(scpu, k=2, dim=1, largest=True).values
    ord_i, ql, wcpu = torch.argsort(top2[:, 0] - top2[:, 1] + 1e-7 * torch.remainder((torch.arange(bf, dtype=torch.float32) + 0.5) * _PHI1, 1.0), descending=True).tolist(), q.tolist(), torch.empty(bf, dtype=torch.long)
    scpu[:, q <= 0] = float("-inf")
    for ii in ord_i:
        cw = int(scpu[ii].argmax().item()); wcpu[ii] = cw; ql[cw] -= 1
        if ql[cw] <= 0: scpu[:, cw] = float("-inf")
    win, sth_g = wcpu.to(dev), _ss01_f((float(s_th) - 0.02) / 0.18)
    g_reb = torch.where((win != near_i) & (def_k[win] > 0.0) & (sur_k[near_i] > 0.0) & (def_k.sum() > 1e-4) & (wv_t > 1e-4), (wv_t * sth_g) * (def_k[win] / wr[win].clamp_min(1e-6)).clamp(0.0, 1.0), torch.zeros_like(g_val)) if sth_g > 1e-4 else torch.zeros_like(g_val)
    ti = win if sth_g > 0.5 else torch.where(g_reb > 0.0, win, near_i)
    return cf[ti], r_halo[ti], torch.where(wv_t > 1e-4, torch.maximum(g_val, g_reb), torch.zeros_like(g_val)), ab_sh, next_fm

def _recover_prior_rows(x_fake: torch.Tensor, record: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    fn = getattr(x_fake, "grad_fn", None)
    if fn is None: return None, None
    q, seen = [(fn, 0)], {id(fn)}
    while q and len(seen) <= 256:
        node, dep = q.pop(0); v = getattr(node, "variable", None)
        if isinstance(v, torch.Tensor) and (ar := getattr(v, "_axiom_active_rows", None)) is not None:
            if record is not None: v._axiom_record = record
            return v, ar
        if dep < 8:
            for nxt, _ in getattr(node, "next_functions", ()):
                if nxt is not None and id(nxt) not in seen: seen.add(id(nxt)); q.append((nxt, dep + 1))
    return None, None
def _split_joint_pair(xf: torch.Tensor, yr: torch.Tensor, tab: Optional[torch.Tensor], rows: Optional[torch.Tensor]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if tab is None or rows is None or tab.ndim != 2 or rows.ndim != 1 or rows.shape[0] != xf.shape[0] or xf.shape[0] < 4 or yr.shape[0] < 4: return None
    zd, xff, yrf, rl = int(tab.shape[1]), xf.reshape(xf.shape[0], -1), yr.reshape(yr.shape[0], -1), rows.to(device=tab.device, dtype=torch.long)
    if xff.shape[1] <= zd or yrf.shape[1] != xff.shape[1]: return None
    zr = tab.detach()[rl.clamp(0, tab.shape[0] - 1)].to(device=xff.device, dtype=torch.float32)
    oob, e_tail, e_head = torch.stack((((rl < 0).any() | (rl >= tab.shape[0]).any()).float(), (xff[:, -zd:].detach().float() - zr).abs().amax(), (xff[:, :zd].detach().float() - zr).abs().amax())).tolist()
    if oob > 0.5: return None
    return (xff[:, :-zd], xff[:, -zd:], yrf[:, :-zd], yrf[:, -zd:]) if e_tail <= 1e-5 else ((xff[:, zd:], xff[:, :zd], yrf[:, zd:], yrf[:, :zd]) if e_head <= 1e-5 else None)
def _basis_omega(c: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, gain: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt, dc = c.dtype, int(c.shape[1]); u = (c - mu.to(dt).unsqueeze(0)) / std.to(dt).unsqueeze(0); ones, r2 = torch.ones((c.shape[0], 1), dtype=dt, device=c.device), u.square().mean(1); om = 1.0 + gain * r2.clamp_max(4.5)
    if dc <= _MAX_FULL_DC:
        parts = [ones, u, 2.0 * (F.softplus(u - 0.8) - 0.425), 2.0 * (F.softplus(-u - 0.8) - 0.425)]
        if dc >= 2: ia, ib = torch.triu_indices(dc, dc, offset=1, device=c.device); parts.append((u[:, ia] * u[:, ib]) * torch.rsqrt(1.0 + 0.15 * (u[:, ia].square() + u[:, ib].square())))
        return torch.cat(parts, 1), om, u
    return torch.cat((ones, u, 2.0 * (F.softplus(r2.clamp_min(1e-6).sqrt() - 1.0).unsqueeze(1) - 0.3133)), 1), om, u
def _solve_spd(gram: torch.Tensor, ridge: torch.Tensor, rhs: torch.Tensor, *, need_inv: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    am = 0.5 * (gram + gram.T) + torch.diag(ridge); lm, info = torch.linalg.cholesky_ex(am, check_errors=False)
    if int(info.item()) != 0: ev, ec = zero_eigh_symmetric(am); ai = (ec * ev.clamp_min(1e-6).reciprocal().unsqueeze(0)) @ ec.T; return ai @ rhs, (0.5 * (ai + ai.T) if need_inv else None)
    ai = torch.cholesky_solve(torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device), lm) if need_inv else None
    return torch.cholesky_solve(rhs, lm), (0.5 * (ai + ai.T) if ai is not None else None)
class AxiomCondFiberBank:
    """Law 3B: Zero-eigh FP64 cross-batch conditional operator bank + strictly FP64 FIFO ring."""
    def __init__(self, *, gamma_base: float = 0.96, gamma_fast: float = 0.85, gamma_min: float = 0.50, innov_threshold: float = 2.5, lin_ridge: float = 1e-3, nl_ridge: float = 0.025, ring_capacity: int = _MAX_RING, knn_k: int = 20, tail_leverage_gain: float = 0.90) -> None:
        self.gamma_base, self.gamma_fast, self.gamma_min, self.innov_threshold = float(gamma_base), float(gamma_fast), float(gamma_min), float(innov_threshold)
        self.lin_ridge, self.nl_ridge, self.ring_capacity, self.knn_k, self.tail_leverage_gain = float(lin_ridge), float(nl_ridge), min(int(ring_capacity), _MAX_RING), int(knn_k), float(tail_leverage_gain)
        self.d_c, self.d_x, self.strata = None, None, {}
    def reset(self) -> None: self.d_c, self.d_x, self.strata = None, None, {}
    def copy(self) -> "AxiomCondFiberBank":
        dup = AxiomCondFiberBank(gamma_base=self.gamma_base, gamma_fast=self.gamma_fast, gamma_min=self.gamma_min, innov_threshold=self.innov_threshold, lin_ridge=self.lin_ridge, nl_ridge=self.nl_ridge, ring_capacity=self.ring_capacity, knn_k=self.knn_k, tail_leverage_gain=self.tail_leverage_gain); dup.load_packet(self.to_packet()); return dup
    @staticmethod
    def _bdim(dc: int) -> int: return (1 + 3 * dc + (dc * (dc - 1)) // 2) if dc <= _MAX_FULL_DC else (2 + dc)
    def _ridge(self, dc: int) -> torch.Tensor: r = torch.full((self._bdim(dc),), self.nl_ridge, dtype=torch.float64, device="cpu"); r[: 1 + dc] = self.lin_ridge; return r
    @staticmethod
    def validate_packet(packet: Any) -> Dict[str, Any]:
        if not isinstance(packet, dict) or set(packet) != _PKT_KEYS or type(packet["schema"]) is not int or packet["schema"] != _BANK_SCHEMA: raise ValueError("invalid header")
        dc, dx, strata = packet["d_c"], packet["d_x"], packet["strata"]
        if not isinstance(strata, list) or len(strata) > _MAX_STRATA or (dc is None and dx is None and strata): raise ValueError("invalid strata")
        if dc is None and dx is None: return packet
        if type(dc) is not int or type(dx) is not int or dc < 1 or dc > _MAX_BANK_DC or dx < 1: raise ValueError("invalid bank dims")
        p, seen = AxiomCondFiberBank._bdim(dc), set(); exp = {"mu_c": (dc,), "m2_c": (dc,), "g_pp": (p, p), "g_px": (p, dx), "g_py": (p, dx), "g_pp_fast": (p, p), "g_px_fast": (p, dx), "g_py_fast": (p, dx), "s_xx": (dx, dx), "s_yy": (dx, dx)}
        for e in strata:
            if not isinstance(e, dict) or set(e) != _STRATUM_KEYS or not isinstance(e["key"], tuple) or not e["key"] or not all(type(v) is int for v in e["key"]) or e["key"] in seen: raise ValueError("invalid stratum")
            seen.add(e["key"])
            if any(isinstance(e[k], bool) or not isinstance(e[k], (int, float)) or not math.isfinite(e[k]) or float(e[k]) <= 0.0 for k in _SCALAR_KEYS): raise ValueError("invalid scalar")
            if any(not isinstance(e[k], torch.Tensor) or e[k].dtype != torch.float64 or not bool(torch.isfinite(e[k]).all()) or (k in exp and tuple(e[k].shape) != exp[k]) for k in _F64_KEYS): raise ValueError("invalid float64 tensor")
            if bool((e["m2_c"] < 0.0).any()) or any(bool((torch.diagonal(e[k]) < -1e-9).any()) for k in ("g_pp", "g_pp_fast", "s_xx", "s_yy")): raise ValueError("negative variance/diag")
            rc, rx, ry = e["ring_c"], e["ring_x"], e["ring_y"]
            if rc.ndim != 2 or rx.ndim != 2 or ry.ndim != 2 or rc.shape[1] != dc or rx.shape[1] != dx or ry.shape[1] != dx or rc.shape[0] != rx.shape[0] or rc.shape[0] != ry.shape[0] or rc.shape[0] > _MAX_RING: raise ValueError("invalid ring")
        return packet
    def to_packet(self) -> Dict[str, Any]: return {"schema": _BANK_SCHEMA, "d_c": self.d_c, "d_x": self.d_x, "strata": [{"key": tuple(int(v) for v in k), **{sk: float(st[sk]) for sk in _SCALAR_KEYS}, **{tk: st[tk].detach().cpu().to(torch.float64).clone() for tk in _F64_KEYS}} for k, st in sorted(self.strata.items())]}
    def load_packet(self, packet: Dict[str, Any]) -> None:
        self.validate_packet(packet); self.d_c, self.d_x = packet["d_c"], packet["d_x"]
        self.strata = {tuple(int(v) for v in e["key"]): {**{sk: float(e[sk]) for sk in _SCALAR_KEYS}, **{tk: e[tk].detach().cpu().to(torch.float64).clone() for tk in _F64_KEYS}} for e in packet["strata"]}
    @torch.no_grad()
    def query_stratum(self, key: Tuple[int, ...], c_g: torch.Tensor, x_g: torch.Tensor, y_g: torch.Tensor) -> Optional[Dict[str, Any]]:
        dc, dx = int(c_g.shape[1]), int(x_g.shape[1]); st, p = (self.strata.get(key) if (dc <= _MAX_BANK_DC and self.d_c == dc and self.d_x == dx) else None), self._bdim(dc)
        if st is None or float(st["n_eff"]) < float(max(p + 2, 12)) or int(st["ring_c"].shape[0]) < 8: return None
        with torch.device("cpu"):
            n_eff, rlen, ng = float(st["n_eff"]), int(st["ring_c"].shape[0]), int(c_g.shape[0]); c64, x64, y64 = c_g.detach().cpu().double(), x_g.detach().cpu().double(), y_g.detach().cpu().double()
            mu_c, std_c = st["mu_c"], (st["m2_c"] / max(n_eff, 1.0)).clamp_min(1e-6).sqrt(); phi_q, om_q, u_q = _basis_omega(c64, mu_c, std_c, self.tail_leverage_gain)
            iw, iwf, ridge = 1.0 / max(float(st["w_eff"]), 1.0), 1.0 / max(float(st["w_fast"]), 1.0), self._ridge(dc)
            sol_s, mis = _solve_spd(st["g_pp"] * iw, ridge, torch.cat((st["g_px"], st["g_py"]), 1) * iw); bxs, bys = sol_s[:, :dx], sol_s[:, dx:]; bp = 0.5 * (bxs + bys)
            mxs, mys, mpq = phi_q @ bxs, phi_q @ bys, phi_q @ bp; dmb = phi_q @ _solve_spd(st["g_pp_fast"] * iwf, ridge, st["g_px_fast"] * iwf, need_inv=False)[0] - mys
            dml = ((phi_q @ mis) @ ((phi_q * om_q.unsqueeze(1)).T @ (x64 - mys)) - ((phi_q @ mis) * (phi_q * om_q.unsqueeze(1))).sum(1, keepdim=True) * (x64 - mys)) / (float(om_q.sum().item()) - om_q).clamp_min(1.0).unsqueeze(1) if (ng >= 2 and mis is not None) else dmb
            w_mat, rc, rxr, ryr = _sym_inv_sqrt(0.5 * (st["s_xx"] + st["s_yy"]) / n_eff), st["ring_c"], st["ring_x"], st["ring_y"]
            phi_r, _, u_r = _basis_omega(rc, mu_c, std_c, self.tail_leverage_gain); mpr, dqr = phi_r @ bp, torch.cdist(u_q, u_r, p=2.0)
            hq = dqr.topk(min(self.knn_k, max(6, rlen // 5), rlen), dim=1, largest=False).values[:, -1].clamp_min(0.25 * math.sqrt(float(dc)))
            lk = -0.5 * (dqr / hq.unsqueeze(1)).square() - (torch.arange(rlen - 1, -1, -1, dtype=torch.float64) / 128.0).unsqueeze(0)
            al, rrx, rry = torch.softmax(lk - lk.amax(1, keepdim=True), dim=1), rxr - mpr, ryr - mpr; dcov = torch.einsum("nm,mij->nij", al, rrx.unsqueeze(2) * rrx.unsqueeze(1) - rry.unsqueeze(2) * rry.unsqueeze(1))
            merr = max(0.0, float(((bxs - bys) @ w_mat).square().sum().item()) / float(max(1, dx)) - 2.0 * float(p) / max(n_eff, 1.0))
            zi = max(math.sqrt(ng / max(1, dx)) * max(float(((x64 - mxs).mean(0) @ w_mat).norm().item()), float(((y64 - mys).mean(0) @ w_mat).norm().item())), float(((c64.mean(0) - mu_c) / std_c).norm().item()) * math.sqrt(ng / max(1, dc)))
            return {"n_eff": n_eff, "ring_len": rlen, "omega": om_q, "m_x": mpq + 0.5 * (al @ (rxr - ryr)), "m_pool": mpq, "w_mat": w_mat, "delta_m_bank": dmb, "delta_m_loo": dml, "delta_cov_loc": dcov, "mean_err_sq": merr, "z_innov": zi}
    @torch.no_grad()
    def update_stratum(self, key: Tuple[int, ...], c_g: torch.Tensor, x_g: torch.Tensor, y_g: torch.Tensor, *, z_innov: float = 0.0) -> None:
        with torch.device("cpu"):
            c64, x64, y64 = c_g.detach().cpu().double(), x_g.detach().cpu().double(), y_g.detach().cpu().double(); ng, dc, dx = int(c64.shape[0]), int(c64.shape[1]), int(x64.shape[1])
            if ng == 0 or dc > _MAX_BANK_DC or (key not in self.strata and len(self.strata) >= _MAX_STRATA): return
            if self.d_c != dc or self.d_x != dx: self.reset(); self.d_c, self.d_x = dc, dx
            cbar, cm2, st, cap = c64.mean(0), (c64 - c64.mean(0, keepdim=True)).square().sum(0), self.strata.get(key), self.ring_capacity
            dm = 1.0 if st is None else math.exp(-0.5 * max(0.0, float(z_innov) - self.innov_threshold) ** 2)
            gs, gf = (0.0, 0.0) if st is None else (self.gamma_min + (self.gamma_base - self.gamma_min) * dm, self.gamma_min + (self.gamma_fast - self.gamma_min) * dm)
            no = 0.0 if st is None else gs * float(st["n_eff"]); nn_ = no + float(ng); mu_n = cbar.clone() if st is None else (st["mu_c"] + (float(ng) / nn_) * (cbar - st["mu_c"]))
            m2_n = cm2.clone() if st is None else (gs * st["m2_c"] + cm2 + (no * float(ng) / nn_) * (cbar - st["mu_c"]).square()).clamp_min(0.0)
            phi, om, _ = _basis_omega(c64, mu_n, (m2_n / max(nn_, 1.0)).clamp_min(1e-6).sqrt(), self.tail_leverage_gain); wb = float(om.sum().item())
            we_n, wf_n, wp = (0.0 if st is None else gs * float(st["w_eff"])) + wb, (0.0 if st is None else gf * float(st["w_fast"])) + wb, phi * om.unsqueeze(1)
            gn = {"g_pp": wp.T @ phi, "g_px": wp.T @ x64, "g_py": wp.T @ y64}; upd, updf = {k: (v if st is None else gs * st[k] + v) for k, v in gn.items()}, {f"{k}_fast": (v.clone() if st is None else gf * st[f"{k}_fast"] + v) for k, v in gn.items()}
            sol, _ = _solve_spd(upd["g_pp"] / max(we_n, 1.0), self._ridge(dc), torch.cat((upd["g_px"], upd["g_py"]), 1) / max(we_n, 1.0), need_inv=False)
            rx, ry = x64 - phi @ sol[:, :dx], y64 - phi @ sol[:, dx:]; keep = 0 if st is None else (int(round(st["ring_c"].shape[0] * max(0.25, dm))) if dm < 0.9 else int(st["ring_c"].shape[0]))
            sl = slice(-keep, None) if keep > 0 else slice(0, 0); rings = {rk: (t[-cap:].clone() if st is None else torch.cat((st[rk][sl], t), 0)[-cap:]) for rk, t in (("ring_c", c64), ("ring_x", x64), ("ring_y", y64))}
            self.strata[key] = {"n_eff": nn_, "n_fast": nn_ if st is None else gf * float(st["n_fast"]) + float(ng), "w_eff": we_n, "w_fast": wf_n, "mu_c": mu_n, "m2_c": m2_n, **upd, **updf, "s_xx": (rx.T @ rx) if st is None else (gs * st["s_xx"] + rx.T @ rx), "s_yy": (ry.T @ ry) if st is None else (gs * st["s_yy"] + ry.T @ ry), **rings}
def _pooled_mean_fb(c32: torch.Tensor, z32: torch.Tensor, groups: List[Tuple[Tuple[int, ...], torch.Tensor]]) -> torch.Tensor:
    dc, mh, eyec = c32.shape[1], torch.zeros_like(z32), torch.eye(c32.shape[1], dtype=torch.float32, device=c32.device)
    for _, ic in groups:
        idx = ic.to(c32.device); ng, zg, zm = int(idx.numel()), z32[idx], z32[idx].mean(0, keepdim=True)
        if ng < max(4, dc + 2): mh[idx] = zm; continue
        dc_v = c32[idx] - c32[idx].mean(0, keepdim=True); mh[idx] = zm + dc_v @ torch.cholesky_solve(dc_v.T @ (zg - zm), torch.linalg.cholesky_ex((dc_v.T @ dc_v) + (1e-4 * dc_v.square().sum().clamp_min(1e-8) / max(1, dc)) * eyec, check_errors=False)[0])
    return mh
def _two_sample_residuals(x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor, *, cvx_mu_gain: float = 0.58) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xf, yf = x.detach().reshape(x.shape[0], -1).float(), y.detach().reshape(y.shape[0], -1).float(); bx, by, dxy, dxx, dyy = xf.shape[0], yf.shape[0], _cdist_s(xf, yf), _cdist_s(xf), _cdist_s(yf)
    mxy, mxx, myy = dxy.mean(), dxx.sum() / max(1, bx * (bx - 1)), dyy.sum() / max(1, by * (by - 1)); den = (0.5 * (mxy + 0.5 * (mxx + myy))).clamp_min(1e-12)
    r_ed, r_sp, bm = (2.0 * mxy - mxx - myy) / den, (mxx - myy) / den, min(bx, by)
    mu_sq = ((xf.mean(0) - yf.mean(0)).square().sum() - (xf.var(0).sum() / max(1, bx) if bx > 1 else 0.0) - (yf.var(0).sum() / max(1, by) if by > 1 else 0.0)) / den.square()
    if bm >= 4:
        xs, yn, el_r = dxy[:bm, :bm].clone().fill_diagonal_(float("inf")), dyy[:bm, :bm].clone().fill_diagonal_(float("inf")).amin(1), ell.detach().float().clamp_min(1e-24) / den
        r_mb = (0.15 + 5.5 * el_r).clamp(0.15, 0.40).square() * mu_sq; r_mu = r_mb
        if cvx_mu_gain != 0.40 and by >= 4:
            nn_m = dyy.clone().fill_diagonal_(float("inf")).amin(1).mean(); dmy, iy = _cdist_s(0.5 * (yf + yf.roll(1, 0)), yf), torch.arange(by, device=yf.device); dmy[iy, iy] = dmy[iy, (iy + 1) % by] = float("inf")
            wc = torch.where(nn_m > 1e-12, _ss01((1.15 - dmy.amin(1).mean() / nn_m.clamp_min(1e-12)) / 0.20), nn_m.new_zeros(()))
            rcx = torch.minimum(torch.maximum((0.15 + 5.5 * el_r), el_r.new_full((), 0.15)), 0.40 + wc * (cvx_mu_gain - 0.40)).square() * mu_sq
            r_mu = torch.where(nn_m > 1e-12, torch.where(mu_sq >= 0.0, torch.maximum(r_mb, rcx.clamp_max(0.045)), rcx), r_mb)
        r_q90, r_hm = 10.2 * (torch.quantile(xs.amin(1), 0.75) - torch.quantile(yn, 0.75)) / den, (torch.quantile(xs.amin(0), 0.90) - torch.quantile(yn, 0.90)) / den
    else: r_mu, r_q90, r_hm = 0.16 * mu_sq, r_ed, r_ed
    return r_ed, r_mu, r_sp, r_q90, r_hm
def _imq_3scale(a: torch.Tensor, b: torch.Tensor, ef: torch.Tensor) -> torch.Tensor:
    sq = a.square().sum(-1, keepdim=True) + b.square().sum(-1).unsqueeze(0) - 2.0 * (a @ b.T); sq = sq + (sq.detach().clamp_min(0.0) - sq.detach()); out = sq.new_zeros(sq.shape)
    for s in (0.35, 1.0, 2.8): es = ef * s; r = torch.sqrt(es.square() + sq); out = out + (es * sq) / (r * (es + r))
    return out * (1.0 / 3.0)

class AxiomTransportLoss(nn.Module):
    """Unified Law 3A + Law 3B transport operator for the Axiom family."""
    def __init__(self, *, weight: float = 1.0, rho: float = 3.0, tau: float = 2.0, eps: float = 0.1, iters: int = 8, chart_capacity: int = 256, chart_weight: float = 1.4, halo_mult: float = 2.1, max_spatial: int = 8, mean_op_weight: float = 0.36, cov_op_weight: float = 0.35, imq_active_scale: float = 0.40, bank_mean_blend: float = 0.85) -> None:
        super().__init__(); self.weight, self.rho, self.tau, self.eps, self.iters = float(weight), float(rho), float(tau), float(eps), int(iters)
        self.chart_capacity, self.chart_weight, self.halo_mult, self.max_spatial = int(chart_capacity), float(chart_weight), float(halo_mult), int(max_spatial)
        self.mean_op_weight, self.cov_op_weight, self.imq_active_scale, self.bank_mean_blend = float(mean_op_weight), float(cov_op_weight), float(imq_active_scale), float(bank_mean_blend)
        self._discrete_step = self._simplicial_step = False; self._ell_local = self.last_p_xy_slack = self._mg_cache = None
    def bind_discrete_prior_step(self, active: bool, *, simplicial_active: bool = False) -> None: self._discrete_step, self._simplicial_step = bool(active), bool(simplicial_active)
    def _cached_manifold_gates(self, yp: torch.Tensor) -> Tuple[float, float, float, torch.Tensor]:
        key = (int(yp.data_ptr()), int(yp._version), tuple(yp.shape), str(yp.device))
        if self._mg_cache is not None and self._mg_cache[0] == key: return self._mg_cache[1]
        res = _manifold_gates(yp); self._mg_cache = (key, res, yp.untyped_storage()); return res
    def _resolve_ell(self, yp: torch.Tensor, record: Any, can_mut: bool) -> torch.Tensor:
        with torch.no_grad():
            eh_t = _batch_ell(yp, 0.25)
            if record is not None and hasattr(record, "ell_initialized"):
                if hasattr(record, "_ema_t") and hasattr(record, "_dev_ema"):
                    prev = record._dev_ema("ell_ema", yp.device).to(torch.float32); cur = eh_t if not record.ell_initialized else (0.90 * prev + 0.10 * eh_t)
                    if can_mut: record._ema_t["ell_ema"], record.ell_initialized = cur.double(), True
                    return (cur if can_mut else (prev if record.ell_initialized else eh_t)).to(yp.dtype)
                eh = float(eh_t.item())
                if can_mut: record.ell_ema, record.ell_initialized = (eh if not record.ell_initialized else 0.90 * float(record.ell_ema) + 0.10 * eh), True
                return yp.new_tensor(float(record.ell_ema) if record.ell_initialized else eh, dtype=yp.dtype)
            if can_mut: self._ell_local = eh_t.detach().clone() if self._ell_local is None else (0.90 * self._ell_local.to(yp.device) + 0.10 * eh_t)
            return (self._ell_local.to(yp.device) if self._ell_local is not None else eh_t).to(yp.dtype)
    @staticmethod
    def _log_odds(df: torch.Tensor, dr: torch.Tensor, c_max: float = 20.0) -> Tuple[torch.Tensor, torch.Tensor]:
        ff, rf = (df.reshape(df.shape[0], -1).mean(1) if df.ndim > 1 else df.reshape(-1)), (dr.detach().reshape(dr.shape[0], -1).mean(1) if dr.ndim > 1 else dr.detach().reshape(-1))
        db = 0.5 * (ff.detach().double().mean() + rf.double().mean()); return c_max * torch.tanh((ff.double() - db).to(ff.dtype) / c_max), c_max * torch.tanh((rf.double() - db).to(rf.dtype) / c_max)
    def _atomic_match(self, fx: torch.Tensor, rx: torch.Tensor, record: Any, ptab: torch.Tensor, brows: torch.Tensor, can_mut: bool) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], float]]:
        if ptab.ndim != 2 or brows.ndim != 1 or brows.shape[0] != fx.shape[0]: return None
        charts, wr, rads, _, ncs = _update_charts(rx.detach().float(), getattr(record, "chart_state", None), max_charts=self.chart_capacity, allow_update=can_mut)
        oob_f, ag_f = torch.stack((((brows < 0).any() | (brows >= ptab.shape[0]).any()).float().to(rx.device), _atomic_gate_t(rx.detach(), charts=charts, radii=rads))).tolist()
        if oob_f > 0.5 or ag_f <= 0.5 or charts.shape[0] < 2: return None
        tk, nn_cc, nal, r_fwd = _update_atomic_ledger(fx.detach(), brows, charts, wr, int(ptab.shape[0]), getattr(record, "atomic_ledger", None), allow_update=can_mut)
        if can_mut and record is not None: record.chart_state, record.atomic_ledger = ncs, nal
        return _bi_cost(fx, charts[tk].detach().to(fx.dtype), (0.15 * torch.median(nn_cc)).clamp_min(1e-12).to(fx.dtype), aligned=True)[0].mean(), charts, nal, r_fwd
    def forward(self, x_fake: torch.Tensor, y_real: torch.Tensor, d_fake: torch.Tensor, d_real: torch.Tensor, *, record: Any = None, cond_fake: Optional[torch.Tensor] = None, cond_real: Optional[torch.Tensor] = None, allow_state_update: bool = True) -> torch.Tensor:
        if self.weight == 0.0: return x_fake[:0].sum()
        disc_step, simp_step = bool(self._discrete_step), bool(self._simplicial_step); self._discrete_step = self._simplicial_step = False
        bf, br, can_mut, cond_p = x_fake.shape[0], y_real.shape[0], _can_mut(record, allow_state_update), (cond_fake is not None or cond_real is not None)
        xp, yp = project_transport_features(x_fake, self.max_spatial), project_transport_features(y_real.detach(), self.max_spatial)
        ell, (g_conn, _, _, _) = self._resolve_ell(yp, record, can_mut), self._cached_manifold_gates(yp)
        if record is not None and hasattr(record, "_disc_gate"): record._disc_gate = 1.0 - g_conn
        exy, exx, ex_ms, ey_ms, dyy_c = _conformal_horizon(xp, yp, ell, g_conn, cond_p, return_aux=True)
        ey_loc, g_het_t, nn1_y, s_near, wv_c_t, am, c_tgt, c_halo, c_rw, ab_sh = (ey_ms.to(ell.dtype) if ey_ms is not None else ell), ell.new_zeros((), dtype=torch.float32), None, None, ell.new_zeros((), dtype=torch.float32), None, None, None, None, None
        if not cond_p and self.chart_weight > 0.0 and br >= 4 and bf >= 2:
            ptab, brows = (getattr(record, "_active_prior_table", None), getattr(record, "_active_batch_rows", None)) if record is not None else (None, None)
            if ptab is None or brows is None: ptab, brows = _recover_prior_rows(x_fake, record)
            has_pt = ptab is not None and brows is not None and ptab.ndim == 2 and brows.ndim == 1 and brows.shape[0] == bf
            if has_pt or (1.0 - g_conn) > 1e-4:
                charts, wr, rads, bcnt, ncs = _update_charts(yp.float(), getattr(record, "chart_state", None) if record is not None else None, max_charts=self.chart_capacity, allow_update=can_mut, loc_ell=ey_ms)
                if charts.shape[0] >= 2:
                    with torch.no_grad():
                        ec = ell.detach().float().clamp_min(1e-24)
                        wv_c_t = (_ss01((torch.quantile(_cdist_s(0.5 * (yp + yp.roll(1, 0)), charts).amin(1), 0.75) / torch.maximum(torch.quantile(_cdist_s(yp, charts).amin(1), 0.75), ec).clamp_min(1e-24) - 0.85 * self.rho) / 1.50) * (1.0 - g_conn)) if (1.0 - g_conn) > 1e-4 else ec.new_zeros(())
                        if has_pt: oob_f, ag_f, wv_f = torch.stack((((brows < 0).any() | (brows >= ptab.shape[0]).any()).float().to(yp.device), _atomic_gate_t(yp.detach(), charts=charts, radii=rads), wv_c_t)).tolist()
                        else: oob_f, ag_f, wv_f = 1.0, 0.0, float(wv_c_t.item())
                    if has_pt and oob_f <= 0.5 and ag_f > 0.5:
                        tk, nn_cc, nal, r_fwd = _update_atomic_ledger(xp.detach(), brows, charts, wr, int(ptab.shape[0]), getattr(record, "atomic_ledger", None), allow_update=can_mut)
                        if can_mut and record is not None: record.chart_state, record.atomic_ledger = ncs, nal
                        am = (_bi_cost(xp, charts[tk].detach().to(xp.dtype), (0.15 * torch.median(nn_cc)).clamp_min(1e-12).to(xp.dtype), aligned=True)[0].mean(), charts, nal, r_fwd)
                        with torch.no_grad():
                            rl, rc = brows.detach().reshape(-1).long().to(xp.device), nal["row_chart"].to(xp.device)
                            ab_sh = (torch.bincount(_cdist_s(yp.detach().float(), charts).argmin(1), minlength=charts.shape[0])[torch.where(rc[rl] >= 0, rc[rl].clamp_min(0), _cdist_s(xp.detach().float(), charts).argmin(1))] <= 0).float()
                    elif (1.0 - g_conn) > 1e-4:
                        c_tgt, c_halo, c_rw, ab_sh, ncs["fake_mass"] = _deconflicted_routing(xp.detach().float(), charts, wr, rads, bcnt, ncs["fake_mass"], ec, wv_c=wv_f, halo_mult=self.halo_mult, s_th=float(getattr(record, "thermal_scale", 1.0)) if record is not None else 1.0, allow_update=can_mut)
                        if can_mut and record is not None: record.chart_state = ncs
        wv_sup_t, dyy_cached, hv_t = wv_c_t, dyy_c, None
        if (not cond_p) and am is None and bf >= 8 and br >= 8:
            with torch.no_grad():
                dyy_s = dyy_c.clone() if dyy_c is not None else _cdist_s(yp); dyy_cached = dyy_s; dyy_s = dyy_s.clone(); dyy_s.fill_diagonal_(0.0); eb = ell.detach().float().clamp_min(1e-24)
                wv_sup_t = torch.maximum(wv_c_t, _ss01((_valley(yp.float(), dyy_s) / (2.0 * eb) - 0.85 * self.rho) / 1.50) * (1.0 - g_conn))
                ds_nz, n_nz = torch.where(dyy_s > 0.0, dyy_s, dyy_s.new_tensor(float("inf"))).reshape(-1).sort().values, (dyy_s > 0.0).sum()
                md_s = torch.where(n_nz > 0, ds_nz[(n_nz // 4).clamp_max(ds_nz.numel() - 1)].clamp_min(1e-12), eb)
                bw_k = (1.15 * md_s * (ey_ms / eb)).clamp_min(1e-12).unsqueeze(1) if ey_ms is not None else (1.15 * md_s)
                rho_k = torch.exp(-dyy_s.square() / (2.0 * bw_k.square()).clamp_min(1e-24)).sum(1); ec_t = (wv_sup_t * (torch.quantile(rho_k, 0.85) / torch.quantile(rho_k, 0.03).clamp_min(1.0))).clamp(0.0, 20.0)
                hv_t = (ec_t.clamp_max(3.0) if not record.het_initialized else (0.92 * float(record.het_ema) + 0.08 * ec_t)) if (record is not None and hasattr(record, "het_initialized")) else ec_t
                g_het_t = _ss01((hv_t - 3.4) / 1.0); dyy_i = dyy_s.clone().fill_diagonal_(float("inf")); nn1_y, nn2_y = dyy_i.amin(1), dyy_i.topk(min(2, br - 1), dim=1, largest=False).values[:, -1]
                s_sp = g_het_t * _ss01((nn2_y / eb - 2.8) / 1.4); ey_f = torch.maximum(ey_ms if ey_ms is not None else eb.expand(br), eb + s_sp * (0.64 * nn2_y - eb).clamp(0.0, 3.4 * eb)); ey_loc = ey_f.to(ell.dtype)
                nd_s, ny_s = _cdist_s(xp, yp).min(1); ne_s = ey_f[ny_s]; s_near = s_sp[ny_s] * _ss01((2.8 * ne_s - nd_s) / (1.2 * ne_s).clamp_min(1e-12))
                ex_f = torch.maximum(ex_ms if ex_ms is not None else eb.expand(bf), eb + s_near * (ne_s - eb)).to(ell.dtype); exy, exx = torch.sqrt(ex_f.unsqueeze(1) * ey_loc.unsqueeze(0)), torch.sqrt(ex_f.unsqueeze(1) * ex_f.unsqueeze(0))
        (cxy, chxy), (cxx, chxx), (dft, drt) = _bi_cost(xp, yp, exy), _bi_cost(xp, xp, exx), self._log_odds(d_fake, d_real)
        with torch.no_grad():
            Pxy = _sinkhorn_slack(chxy, dft.detach(), drt.detach(), rho=self.rho, tau=self.tau, eps=self.eps, iters=self.iters, mask=_discrete_mask(cond_fake, cond_real, bf, br))
            mxx = _discrete_mask(cond_fake, cond_fake, bf, bf)
            if disc_step and bf > 1: ux = (chxx > 1e-12) | torch.eye(bf, dtype=torch.bool, device=x_fake.device); mxx = ux if mxx is None else (mxx & ux)
            zf = dft.new_zeros(bf); Pxx_a = _sinkhorn_slack(chxx, zf, zf, rho=self.rho, tau=self.tau, eps=self.eps, iters=self.iters, mask=mxx)[:bf, :bf].fill_diagonal_(0.0)
            Pxy_a, Pxy_s, Pxy_cs = Pxy[:bf, :br], Pxy[:bf, br], Pxy[bf, :br]; self.last_p_xy_slack = Pxy_s.detach(); af = (bf * Pxy_a.sum(1)).clamp(0.0, 1.0)
            if s_near is not None: af = (af + 0.50 * s_near.to(af.dtype)).clamp(0.0, 1.0)
            sf = (float(bf) * Pxy_s).clamp(0.0, 1.0); p_sl = Pxy_s * (1.0 + 0.35 * sf) * ((1.0 - ab_sh.to(Pxy_s.dtype)) if ab_sh is not None else 1.0)
        coord = (Pxy_a * cxy).sum() - 0.5 * (Pxx_a * torch.sqrt(af.unsqueeze(1) * af.unsqueeze(0)) * cxx).sum()
        is_dn = x_fake.ndim >= 4 and (int(x_fake.shape[-2]) > self.max_spatial or int(x_fake.shape[-1]) > self.max_spatial)
        c_rel_w = (2.5 * torch.maximum(ell.detach(), 0.20 * (yp.float() - yp.float().mean(0, keepdim=True)).square().sum(-1).mean().sqrt().to(ell.dtype))) if is_dn else 0.03
        critic_t, r_tail_t = -(p_sl * (0.5 * self.tau * dft)).sum() + (c_rel_w * self.tau) * (Pxy_a * (F.softplus(drt.unsqueeze(0) - dft.unsqueeze(1)) - math.log(2.0))).sum(), ell.new_zeros((), dtype=torch.float32)
        if am is not None: coord, r_tail_t = coord + self.chart_weight * am[0].to(coord.dtype), ell.new_full((), float(am[3]), dtype=torch.float32)
        elif c_tgt is not None and c_rw is not None:
            ec = ell.detach().float().clamp_min(1e-24); diff = xp.float() - c_tgt.detach(); sd = _sc_safe(torch.maximum(diff.detach().abs().amax(), ec))
            uh = F.relu(torch.sqrt((diff / sd).square().sum(-1) + 1e-12) * sd - c_halo.detach())
            coord = coord + self.chart_weight * ((c_rw.to(coord.dtype) / float(bf)) * (uh.square() / (torch.sqrt(ec.square() + uh.square()) + ec).clamp_min(1e-30)).to(coord.dtype)).sum()
        if nn1_y is not None:
            with torch.no_grad():
                gb_t, ef_y, ch_f = _ss01((Pxy_a.detach().float().sum() - 0.72) / 0.12), ey_loc.detach().float().clamp_min(1e-24), chxy.detach().float(); rn = ch_f.amin(1)
                tg = torch.maximum(ef_y.new_tensor(4.2 * self.rho), 2.10 * torch.quantile((torch.sqrt(ef_y.square() + nn1_y.square()) - ef_y) / ef_y, 0.98))
                orow = (g_het_t * gb_t) * _ss01((rn - tg) / (0.40 * tg)) * torch.sigmoid((-dft.detach().float() - 0.35) / 0.20) * sf.detach().float().pow(4); wo = torch.softmax(-(ch_f - rn.unsqueeze(1)) / 0.25, dim=1).to(cxy.dtype)
            coord = coord + 2.2 * ((orow.to(cxy.dtype) / float(bf)).unsqueeze(1) * wo * cxy).sum()
        if not cond_p: coord = coord + _orphan_macro_loss(xp, yp, chxy, Pxy_s, Pxy_cs, rho=self.rho, g_conn=g_conn)
        total = coord + critic_t
        if record is not None:
            if can_mut and not is_dn and not (simp_step or getattr(record, "_simplicial_step", False)) and not cond_p and bf >= 4 and br >= 4 and hasattr(record, "observe_thermal"):
                with torch.no_grad():
                    r5 = _two_sample_residuals(xp, yp, ell, cvx_mu_gain=0.58)
                    if (not disc_step) and am is None and bf >= 8 and br >= 8:
                        dyy_t, dxy_t, off_t = (dyy_cached if dyy_cached is not None else _cdist_s(yp)), _cdist_s(xp, yp), ~torch.eye(br, dtype=torch.bool, device=yp.device)
                        r_tail_t = torch.maximum(r_tail_t, wv_sup_t * 1.20 * (torch.quantile(dxy_t.amin(1), 0.96) - torch.quantile(dyy_t.masked_fill(~off_t, float("inf")).amin(1), 0.96)) / dyy_t[off_t].mean().clamp_min(1e-12))
                    pv = torch.stack([*r5, r_tail_t] + ([hv_t] if (hv_t is not None and hasattr(record, "het_initialized")) else [])).tolist()
                    if hv_t is not None and hasattr(record, "het_initialized"): record.het_ema, record.het_initialized = float(pv[6]), True
                    record.observe_thermal(float(pv[0]), float(pv[1]), float(pv[2]), float(pv[3]), float(pv[4]), r_tail=float(pv[5]), shock_mu_ratio=0.35)
            elif can_mut and hv_t is not None and hasattr(record, "het_initialized"): record.het_ema, record.het_initialized = float(hv_t.item()), True
            sth = float(getattr(record, "thermal_scale", 1.0)); total = (_scale_grad(coord, sth) + critic_t) if getattr(d_fake, "_axiom_thermally_scaled", False) else _scale_grad(total, sth)
        return self.weight * total
    def simplicial_manifold_loss(self, z_table: torch.Tensor, gen_fn: Any, y_real: torch.Tensor, *, record: Any = None, iso_weight: float = 0.45, chart_weight: float = 4.0) -> torch.Tensor:
        mp, br = int(z_table.shape[0]), int(y_real.shape[0])
        if self.weight == 0.0 or mp < 2 or br < 1: return y_real[:0].sum()
        phi = float(getattr(record, "chord_phase", 0.0)) if record is not None else 0.0
        if mp > 512: z_table, mp = z_table[((torch.arange(128, device=z_table.device) * mp) // 128 + int(phi * mp)) % mp], 128
        yf, use_f64 = y_real.detach().reshape(br, -1), (mp <= 256)
        with torch.no_grad():
            g_conn, g_mac, q90_u, y_sc = self._cached_manifold_gates(yf) if (chart_weight > 0.0 and br >= 4) else (0.0, 0.0, 0.1, _sc_safe(yf.float().abs().amax()))
            g_curv, zd, ell = (_curve_gate(yf, g_conn) if chart_weight > 0.0 else 0.0), z_table.detach(), self._resolve_ell(yf, record, False)
            dzz = (torch.cdist(zd.double(), zd.double(), p=2.0) + 1e-12 * torch.arange(mp, device=zd.device, dtype=torch.float64).unsqueeze(0)).fill_diagonal_(float("inf"))
            knn, nq = min(4, mp - 1), max(4, br if br <= 512 else 128); nbrs = dzz.topk(knn, dim=-1, largest=False).indices
            nt = int(round((1.0 - g_curv) * float(nq - ((nq // 2) if (mp >= 3 and knn >= 2 and nq >= 4) else nq)))); ne, re, ke = nq - nt, torch.arange(nq - nt, device=zd.device), min(3, knn)
            if g_curv > 0.5:
                nd = max(1, mp * ke); pid = (re + int(phi * float(nd))) % nd; v0e, v1e, ue = pid // ke, nbrs[pid // ke, pid % ke], ((((re // nd).float() + 0.5) / float(max(1, (ne + nd - 1) // nd))) + phi) % 1.0
            else: v0e, v1e, ue = (re + int(phi * mp)) % mp, nbrs[(re + int(phi * mp)) % mp, re % ke], (((re.float() + 0.5) / float(max(1, ne))) + phi) % 1.0
            te = (0.20 + 0.60 * ue).to(zd.dtype); z_simp = (1.0 - te.unsqueeze(1)) * zd[v0e] + te.unsqueeze(1) * zd[v1e]
            if nt > 0:
                rt, pt = torch.arange(nt, device=zd.device), (phi * _PHI2) % 1.0; v0t, c1 = (rt + int(pt * mp)) % mp, rt % knn; v1t, v2t = nbrs[v0t, c1], nbrs[v0t, (c1 + 1 + ((rt // knn) % max(1, knn - 1))) % knn]
                ang, rw = 2.0 * math.pi * ((((rt.float() + 0.5) / float(nt)) + pt) % 1.0), 0.12 * (rt % 2 == 1).to(zd.dtype)
                w0, w1 = (1.0 / 3.0 + rw * torch.cos(ang).to(zd.dtype)).unsqueeze(1), (1.0 / 3.0 + rw * torch.cos(ang + 2.0 * math.pi / 3.0).to(zd.dtype)).unsqueeze(1)
                z_tri = w0 * zd[v0t] + w1 * zd[v1t] + (1.0 - w0 - w1) * zd[v2t]; z_simp = torch.cat([z_simp, z_tri], 0)
        xv_live = gen_fn(z_table).reshape(mp, -1) if (g_conn > 1e-4 and chart_weight > 0.0) else None
        xv, xs = (xv_live.detach() if xv_live is not None else gen_fn(z_table.detach()).detach().reshape(mp, -1)), gen_fn(z_simp).reshape(nq, -1)
        if g_mac > 1e-4 and _can_mut(record) and record is not None and br >= 4 and hasattr(record, "observe_thermal"):
            with torch.no_grad():
                r5 = torch.stack(_two_sample_residuals(torch.cat([xv, xs.detach()], 0), yf, ell)).tolist()
                record.observe_thermal(float(r5[0]), float(r5[1]), float(r5[2]), (1.0 - 0.50 * g_curv) * float(r5[3]), float(r5[4]), tau=0.084 * (1.0 + 0.60 * g_curv), shock_mu_ratio=0.35 + 0.55 * g_curv, shock_ed_floor=0.30 + 0.50 * g_curv, lock_hysteresis=0.22 * g_curv)
        cp_sr, ch_sr = _bi_cost(xs, yf, ell, f64=use_f64); total, eff_iso = (torch.softmax(-ch_sr / self.eps, dim=1).to(cp_sr.dtype) * cp_sr).sum(1).mean(), (((1.0 - g_mac) * iso_weight + g_mac * 0.25) if g_mac > 1e-4 else iso_weight)
        if eff_iso > 0.0:
            d0e, d1e = _bi_cost(xs[:ne], xv[v0e], ell, aligned=True, f64=use_f64)[0], _bi_cost(xs[:ne], xv[v1e], ell, aligned=True, f64=use_f64)[0]; iso_l = ((1.0 - te.to(d0e.dtype)) * d0e - te.to(d0e.dtype) * d1e).abs().mean()
            if nt > 0:
                d0t, d1t, d2t = (_bi_cost(xs[ne:], xv[vt], ell, aligned=True, f64=use_f64)[0] for vt in (v0t, v1t, v2t))
                with torch.no_grad():
                    vx0, vx1, vx2, e64 = xv[v0t].double(), xv[v1t].double(), xv[v2t].double(), ell.detach().clamp_min(1e-24).double(); ea, eb, ec = (vx0 - vx1).norm(dim=-1), (vx1 - vx2).norm(dim=-1), (vx2 - vx0).norm(dim=-1); sp = 0.5 * (ea + eb + ec)
                    mr = 0.70 * (torch.sqrt(e64.square() + (torch.sqrt((sp * (sp - ea).clamp_min(0.0) * (sp - eb).clamp_min(0.0) * (sp - ec).clamp_min(0.0)).clamp_min(0.0)) / sp.clamp_min(1e-8)).square()) - e64)
                    if g_mac > 1e-4:
                        z0, z1, z2, zt = zd[v0t].double(), zd[v1t].double(), zd[v2t].double(), z_tri.detach().double(); zdm = torch.maximum(torch.maximum((z0 - z1).norm(dim=-1), (z1 - z2).norm(dim=-1)), (z2 - z0).norm(dim=-1)).clamp_min(1e-8)
                        mr = (1.0 - g_mac + g_mac * _ss01((torch.minimum(torch.minimum((zt - z0).norm(dim=-1), (zt - z1).norm(dim=-1)), (zt - z2).norm(dim=-1)) / zdm - 0.12) / 0.10)) * mr
                    mg = mr.to(d0t.dtype)
                iso_l = 0.5 * (iso_l + (F.relu(mg - d0t) + F.relu(mg - d1t) + F.relu(mg - d2t)).mean() / 3.0)
            total = total + eff_iso * iso_l
        if g_conn > 1e-4 and chart_weight > 0.0:
            yu = yf.float() / y_sc
            if xv_live is not None:
                cp_vy, _ = _bi_cost(xv_live, yf, ell, f64=use_f64)
                with torch.no_grad(): Lv = -(cp_vy.detach().float() / cp_vy.detach().float().mean().clamp_min(1e-12)) / 0.08; Pvy = _log_sinkhorn(Lv, Lv.new_full((mp,), -math.log(mp)), Lv.new_full((br,), -math.log(br)), 10).exp_().to(cp_vy.dtype)
                total = total + (1.10 * g_conn) * (Pvy * cp_vy).sum()
                if g_curv > 1e-4: total = total + (0.65 * g_conn * g_curv) * _bi_cost(xv_live, yf[(Pvy.float() @ _cdist_s(yu)).argmin(1)], 0.5 * ell, aligned=True, f64=use_f64)[0].mean()
            if z_table.requires_grad:
                with torch.no_grad():
                    yc = yu if (mp + br) <= 512 else yu[((torch.arange(min(br, 128), device=yu.device) * br) // min(br, 128) + int(phi * br)) % br]
                    vc = torch.cat([xv.float() / y_sc, yc], 0).double() if use_f64 else torch.cat([xv.float() / y_sc, yc], 0); dv = torch.cdist(vc, vc, p=2.0); gm = dv * (1.0 + (dv / max(1e-4, 1.35 * q90_u)).square())
                    for _ in range(6): gm = torch.minimum(gm, (gm.unsqueeze(2) + gm.unsqueeze(0)).amin(1))
                    gvv, off = gm[:mp, :mp].double(), ~torch.eye(mp, dtype=torch.bool, device=z_table.device); ghat = gvv / torch.sqrt(gvv[off].square().mean().clamp_min(1e-12))
                    wl, wg = torch.where(off, (0.20 + ghat.square()).pow(-2.0), 0.0), torch.where(off, 1.0 + 2.0 * ghat.square(), 0.0)
                    wch = (wl / wl.sum().clamp_min(1e-12)) if g_curv <= 1e-4 else ((1.0 - g_curv) * (wl / wl.sum().clamp_min(1e-12)) + g_curv * (wg / wg.sum().clamp_min(1e-12)))
                z64 = z_table.double(); dz2 = z64.square().sum(-1, keepdim=True) + z64.square().sum(-1).unsqueeze(0) - 2.0 * (z64 @ z64.T); dz2 = dz2 + (dz2.detach().clamp_min(0.0) - dz2.detach())
                dzh = torch.sqrt(dz2 + 1e-12) / torch.sqrt(dz2[off].mean().clamp_min(1e-12))
                total = total + (g_conn * chart_weight * (1.0 + 18.0 * g_curv)) * ell.detach().to(total.dtype) * (wch * (dzh - ghat).square()).sum().to(total.dtype)
                if g_curv > 1e-4 and z_table.shape[1] >= 2:
                    zc = z64 - z64.mean(0, keepdim=True); fsq = zc.square().sum().clamp_min(1e-12)
                    with torch.no_grad(): ax = _power_axis((zc.T @ zc).detach(), 6)
                    total = total + (3.2 * g_conn * g_curv * chart_weight) * ell.detach().to(total.dtype) * ((fsq - (zc @ ax).square().sum()).clamp_min(0.0) / fsq).to(total.dtype)
        return self.weight * (_scale_grad(total, float(getattr(record, "thermal_scale", 1.0))) if record is not None else total)
    def joint_relativistic_loss(self, x_fake: torch.Tensor, y_real: torch.Tensor, d_fake: torch.Tensor, d_real: torch.Tensor, *, record: Any = None, cond_fake: Optional[torch.Tensor] = None, cond_real: Optional[torch.Tensor] = None, allow_state_update: bool = True) -> torch.Tensor:
        if self.weight == 0.0: return d_fake[:0].sum()
        bf, br, can_mut = d_fake.shape[0], d_real.shape[0], _can_mut(record, allow_state_update)
        xp, yp = project_transport_features(x_fake.detach(), self.max_spatial), project_transport_features(y_real.detach(), self.max_spatial); ell = self._resolve_ell(yp, record, can_mut)
        with torch.no_grad():
            Pxy = _sinkhorn_slack(_bi_cost(xp, yp, ell)[1], *self._log_odds(d_fake.detach(), d_real.detach()), rho=self.rho, tau=self.tau, eps=self.eps, iters=self.iters, mask=_discrete_mask(cond_fake, cond_real, bf, br))
            Pact, Prs, Pcs = Pxy[:bf, :br], Pxy[:bf, br], Pxy[bf, :br]; self.last_p_xy_slack = Prs
        ff, rf = (d_fake.reshape(bf, -1).mean(1) if d_fake.ndim > 1 else d_fake.reshape(bf)), (d_real.reshape(br, -1).mean(1) if d_real.ndim > 1 else d_real.reshape(br))
        db = 0.5 * (ff.detach().mean() + rf.detach().mean()); loss = self.weight * ((Pact * F.softplus(rf.unsqueeze(0) - ff.unsqueeze(1))).sum() + (Prs * F.softplus(db - ff)).sum() + (Pcs * F.softplus(rf - db)).sum())
        if record is not None and cond_fake is None and cond_real is None:
            ptab, brows = getattr(record, "_active_prior_table", None), getattr(record, "_active_batch_rows", None)
            if ptab is None or brows is None: ptab, brows = _recover_prior_rows(x_fake, record)
            sp = _split_joint_pair(x_fake, y_real, ptab, brows)
            if sp is not None and ptab is not None and brows is not None:
                fx, _, rx, rz = sp; am = self._atomic_match(fx, rx, record, ptab, brows, can_mut)
                if am is not None and am[1].shape[0] >= 3:
                    l_fwd, charts, nal, r_fwd = am
                    with torch.no_grad():
                        tab_f = ptab.detach().to(device=rx.device, dtype=torch.float32); match_m = nal["row_chart"].to(rx.device).unsqueeze(0) == _cdist_s(rx, charts).argmin(1).unsqueeze(1)
                        has_inv, inv_i = match_m.any(1), _cdist_s(rz, tab_f).masked_fill(~match_m, float("inf")).argmin(1); nn_z = _cdist_s(tab_f, diag_inf=True).amin(1).clamp_min(1e-4) if tab_f.shape[0] >= 2 else torch.ones((tab_f.shape[0],), device=rx.device)
                        inv_t, ell_z, wm = tab_f[inv_i].to(rz.dtype), (0.25 * torch.median(nn_z)).clamp_min(1e-4).to(rz.dtype), has_inv.to(rz.dtype)
                        ir = ((rz.detach().float() - inv_t.float()).norm(dim=-1) / nn_z[inv_i]).clamp_max(10.0); n_hi = has_inv.float().sum()
                        hi_any, bad_inv, ir_mf = torch.stack((n_hi, (((ir > 0.12) & has_inv).any() | (~has_inv).any()).float(), (ir * has_inv.float()).sum() / n_hi.clamp_min(1.0))).tolist()
                    l_inv, r_inv = (((wm * _bi_cost(rz, inv_t, ell_z, aligned=True)[0]).sum() / wm.sum().clamp_min(1.0), max(1.6 * float(ir_mf), 0.20 if bad_inv > 0.5 else 0.0)) if hi_any > 0.5 else (rz.new_zeros(()), 0.25))
                    if can_mut and hasattr(record, "observe_thermal"): record.observe_thermal(0.0, 0.0, 0.0, 0.0, 0.0, r_tail=max(r_fwd, r_inv))
                    sth = float(getattr(record, "thermal_scale", 1.0)); loss = (loss if getattr(d_fake, "_axiom_thermally_scaled", False) else _scale_grad(loss, sth)) + _scale_grad(self.weight * max(self.chart_weight, 1.0) * (l_fwd + l_inv), sth)
        return loss
    def conditional_fiber_loss(self, x_fake: torch.Tensor, y_real: torch.Tensor, c_cont: torch.Tensor, *, cond_disc: Optional[torch.Tensor] = None, record: Any = None) -> torch.Tensor:
        bn = x_fake.shape[0]
        if bn <= 1 or y_real.shape[0] != bn or c_cont.shape[0] != bn: return x_fake[:0].sum()
        xf, yf = x_fake.reshape(bn, -1), y_real.detach().reshape(bn, -1); dev, dim, can_mut = xf.device, max(1, xf.shape[-1]), (record is not None and _can_mut(record))
        fmax, bank = torch.finfo(xf.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0)), (getattr(record, "_cond_bank_obj", None) if record is not None else None)
        if record is not None and bank is None:
            bank = AxiomCondFiberBank()
            if getattr(record, "cond_fiber_bank", None) is not None: bank.load_packet(record.cond_fiber_bank)
            record._cond_bank_obj = bank
        with torch.no_grad():
            st = _discrete_mask(cond_disc, cond_disc, bn, bn); same_t = torch.ones((bn, bn), dtype=torch.bool, device=dev) if st is None else st
            mh = same_t.clone().fill_diagonal_(False); vrow = mh.any(1); mval = mh & vrow.unsqueeze(1) & vrow.unsqueeze(0)
            if not bool(mval.any()): return x_fake[:0].sum()
            c32, x32, y32, cpu = c_cont.detach().reshape(bn, -1).float(), xf.detach().float(), yf.float(), torch.device("cpu")
            dc, cxy64 = int(c32.shape[1]), torch.cat((c32, x32, y32), 1).to(device=cpu, dtype=torch.float64); c64, x64, y64 = cxy64[:, :dc], cxy64[:, dc : dc + dim], cxy64[:, dc + dim :]
            if cond_disc is None: groups = [((0,), torch.arange(bn, dtype=torch.long, device=cpu))]
            else:
                bkt: Dict[Tuple[int, ...], List[int]] = {}
                for i, row in enumerate(cond_disc.detach().reshape(bn, -1).to(cpu, torch.int64).tolist()): bkt.setdefault(tuple(int(v) for v in row), []).append(i)
                groups = [(k, torch.tensor(bkt[k], dtype=torch.long, device=cpu)) for k in sorted(bkt)]
            has_op, op_w, op_dm, op_mx, op_ec, op_rw = torch.zeros(bn, dtype=torch.bool, device=cpu), torch.eye(dim, device=cpu).unsqueeze(0).expand(bn, dim, dim).clone(), torch.zeros((bn, dim), device=cpu), torch.zeros((bn, dim), device=cpu), torch.zeros((bn, dim, dim), device=cpu), torch.zeros(bn, device=cpu)
            mp_cpu, mp_g, pend, r_worst = torch.zeros((bn, dim), device=cpu), torch.zeros((bn, 1), device=cpu), [], 0.0
            if bank is not None:
                for key, ic in groups:
                    q, zi = bank.query_stratum(key, c64[ic], x64[ic], y64[ic]), 0.0
                    if q is not None:
                        wg, wm = float(q["ring_len"]) / (float(q["ring_len"]) + 32.0), q["w_mat"]; mp_cpu[ic], mp_g[ic], wm32 = q["m_pool"].float(), wg, wm.float()
                        dmv = ((1.0 - self.bank_mean_blend) * q["delta_m_loo"] + self.bank_mean_blend * q["delta_m_bank"]) @ wm; ndm = dmv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        ec = torch.matmul(wm32.unsqueeze(0), torch.matmul(q["delta_cov_loc"].float(), wm32.unsqueeze(0))); ec = 0.5 * (ec + ec.transpose(-1, -2)); nec, om = ec.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-8), q["omega"].float()
                        has_op[ic], op_w[ic], op_dm[ic], op_mx[ic], op_ec[ic], op_rw[ic] = True, wm32.unsqueeze(0), (dmv * (3.0 * torch.tanh(ndm / 3.0) / ndm)).float(), q["m_x"].float(), ec * (1.5 * torch.tanh(nec / 1.5) / nec), wg * (om / om.mean().clamp_min(1e-6))
                        zi, r_worst = float(q["z_innov"]), max(r_worst, float(q["mean_err_sq"]))
                    pend.append((key, ic, zi))
            mhat = mp_cpu.to(dev) if bool(has_op.all()) else ((mp_g.to(dev) * mp_cpu.to(dev) + (1.0 - mp_g.to(dev)) * _pooled_mean_fb(c32, 0.5 * (x32 + y32), groups)) if bool(has_op.any()) else _pooled_mean_fb(c32, 0.5 * (x32 + y32), groups))
            if bank is not None and can_mut:
                for key, ic, zi in pend: bank.update_stratum(key, c64[ic], x64[ic], y64[ic], z_innov=zi)
            dcm = _cdist_s(c32 / c32.abs().amax().clamp_min(1e-12)); ds, si = dcm.masked_fill(~mval, float("inf")).topk(k=min(4, bn), dim=1, largest=False)
            sr = ds.gather(1, (torch.minimum(torch.full_like(mval.sum(1), min(3, ds.shape[1])), mval.sum(1).clamp_min(1)) - 1).unsqueeze(1)).squeeze(1)
            sig = torch.where(vrow, torch.where(torch.isfinite(sr), sr, torch.zeros_like(sr)).clamp_min(0.0) + 1e-3 * dcm.amax() + 1e-6, torch.ones_like(sr))
            yr, fm = y32 - mhat, (same_t & vrow.unsqueeze(0) & vrow.unsqueeze(1)).float(); eds = (fm @ ((yr - yr[si[:, 0]]) / math.sqrt(2.0)).square()) / fm.sum(1).clamp_min(1.0).unsqueeze(1)
            ed = torch.where(vrow.unsqueeze(1), (0.5 * eds.mean(-1, keepdim=True) + 0.5 * eds).sqrt().clamp_min(1e-2 * yr.abs().amax().clamp_min(1e-3) / math.sqrt(dim) + 1e-5), torch.ones_like(eds))
            erms = ed.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-5); wc, ef = (erms / ed) / (erms / ed).square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8), erms * math.sqrt(dim)
            kc = torch.where(mval, torch.exp(-dcm.square() / (2.0 * sig.unsqueeze(1) * sig.unsqueeze(0)).clamp_min(1e-12)), 0.0); isd = torch.rsqrt(1.0 + kc.sum(1)); w0 = (kc * isd.unsqueeze(1) * isd.unsqueeze(0)) / bn; wi, any_op = 0.5 * (w0 + w0.T), bool(has_op.any())
            if any_op: ep_d, en_d = sym_eigensplit_zero_eigh(op_ec.to(dev)); ow_d, odm_d, omx_d, orw_d = op_w.to(dev), op_dm.to(dev), op_mx.to(dev), op_rw.to(dev)
        xw, yw = (xf.float() - mhat) * wc, (y32 - mhat) * wc; pxy, pxx, wi_d = _imq_3scale(xw, yw, ef), _imq_3scale(xw, xw, ef), wi.to(xf.dtype)
        l_imq, l_op = (wi_d * pxy.clamp_max(fmax).to(xf.dtype)).sum() - 0.5 * (wi_d * pxx.clamp_max(fmax).to(xf.dtype)).sum(), xf.new_zeros(())
        if any_op:
            inv_d = 1.0 / float(dim)
            if self.mean_op_weight > 0.0: l_op = l_op + self.mean_op_weight * (orw_d * (torch.einsum("nij,nj->ni", ow_d, xf.float()) * odm_d).sum(-1) * inv_d).mean().to(xf.dtype)
            if self.cov_op_weight > 0.0: rxw = torch.einsum("nij,nj->ni", ow_d, xf.float() - omx_d); rxs = rxw * torch.rsqrt(1.0 + 0.25 * rxw.square().sum(-1, keepdim=True) * inv_d); l_op = l_op + self.cov_op_weight * (orw_d * (0.5 * torch.einsum("ni,nij,nj->n", rxw, ep_d, rxw) + 0.5 * torch.einsum("ni,nij,nj->n", rxs, en_d, rxs)) * inv_d).mean().to(xf.dtype)
        if record is not None and hasattr(record, "cond_initialized") and can_mut:
            with torch.no_grad():
                pyy = _imq_3scale(yw.detach(), yw.detach(), ef); r_c = max(-0.5, min(0.5, float(((wi * (2.0 * pxy.detach() - pxx.detach() - pyy)).sum() / (wi * pyy).sum().clamp_min(1e-8)).item()))) + min(0.5, 0.11 * max(0.0, r_worst - 0.10))
                record.cond_ema, record.cond_initialized = (max(0.18, r_c) if not record.cond_initialized else (0.99 * float(record.cond_ema) + 0.01 * r_c)), True
                if hasattr(record, "step_scale"): uc = min(1.0, max(0.0, (0.050 - max(0.0, float(record.cond_ema))) / 0.024)); record.step_scale = float(1.0 - 0.85 * (uc * uc * (3.0 - 2.0 * uc)))
        return ((self.imq_active_scale if any_op else 1.0) * l_imq + l_op).clamp(-fmax, fmax).to(xf.dtype)

__all__ = ["AxiomCondFiberBank", "AxiomTransportLoss", "compute_discrete_atomic_gate", "project_transport_features", "sym_eigensplit_zero_eigh", "zero_eigh_symmetric"]
