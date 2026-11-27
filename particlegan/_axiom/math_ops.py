"""Axiom L0 stateless math primitives (zero eigh/svd/pinv/autograd.grad/caches)."""

from __future__ import annotations

import math

import torch

PHI1: float = 0.6180339887498949
PHI2: float = 0.7548776662466927
_F32, _F64, _BOOL, _LONG = torch.float32, torch.float64, torch.bool, torch.long
_QUINTIC = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def _sq_sym(m, dt):
    if m.ndim < 2 or m.shape[-1] != m.shape[-2] or m.shape[-1] < 1:
        raise ValueError("expected square")
    x = m.to(dt)
    return 0.5 * (x + x.transpose(-1, -2))


def _amax_sc(*ts):
    am = ts[0].detach().abs().amax()
    for t in ts[1:]:
        am = torch.maximum(am, t.detach().abs().amax())
    return torch.where(am > 0.0, am.clamp_min(1e-30), 1.0)


def _norm_step(v, fb, dim, eps):
    vn = v.norm(dim=dim, keepdim=True)
    return torch.where(vn > eps, v / vn.clamp_min(eps), fb)


def _bnorm(x):
    return x.flatten(1).norm(dim=1).view(-1, 1, 1)


def smoothstep01(x):
    u = x.clamp(0.0, 1.0) if isinstance(x, torch.Tensor) else min(1.0, max(0.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


@torch.no_grad()
def me_unit_rows(x: torch.Tensor):
    if x.ndim != 2:
        raise ValueError("expected 2D")
    r, c = x.shape
    if r == 0 or c == 0:
        return x.new_zeros((r, c), dtype=_F32), x.new_zeros((r, 1), dtype=_F32), x.new_zeros((r, 1), dtype=_BOOL)
    xf = x.float()
    fin = torch.isfinite(xf).all(dim=1, keepdim=True)
    xs = torch.where(fin, xf, 0.0)
    gm = xs.abs().amax(dim=1, keepdim=True)
    act = fin & (gm > 0.0)
    one = torch.ones_like(gm)
    _, exp = torch.frexp(torch.where(act, gm, one))
    e1 = (-exp) // 2
    xm = (xs * torch.ldexp(one, e1)) * torch.ldexp(one, (-exp) - e1)
    nrm = xm.norm(dim=1, keepdim=True)
    ok = act & (nrm > 0.0) & torch.isfinite(nrm)
    nc = nrm.clamp_min(1e-30)
    return torch.where(ok, xm / nc, 0.0), torch.where(ok, exp.float() + torch.log2(nc), 0.0), ok


@torch.no_grad()
def mantissa_exponent_normalize(g: torch.Tensor):
    if g.numel() == 0:
        return g.float().clone(), g.new_zeros((), dtype=_F32), g.new_zeros((), dtype=_BOOL)
    u, l2, ok = me_unit_rows(g.float().reshape(1, -1))
    return u.reshape_as(g), l2.reshape(()), ok.reshape(())


@torch.no_grad()
def weyl_rademacher(shape, *, phase=0.0, salt=0, device=None, dtype=_F32):
    dims = tuple(int(s) for s in shape)
    n = math.prod(dims) if dims else 1
    if n == 0:
        return torch.empty(dims, device=device, dtype=dtype)
    sw = (int(math.floor((float(phase) % 1.0) * 1073741824.0)) + int(salt) * 7919) & 0xFFFFFFFF
    h = (torch.arange(n, device=device, dtype=torch.int64) * 2654435761 + sw * 3266489917 + 374761393) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    return torch.where(((h ^ (h >> 16)) & 1) == 0, 1.0, -1.0).to(dtype).reshape(dims)


def _weyl_vec(dim, seed, dev, dt):
    raw = torch.remainder(torch.arange(1, dim + 1, device=dev, dtype=dt) * PHI1 + (seed + 0.5) * PHI2, 1.0) - 0.5
    nrm = raw.norm()
    return torch.where(nrm > 1e-8, raw / nrm.clamp_min(1e-12), torch.full((dim,), dim**-0.5, device=dev, dtype=dt))


@torch.no_grad()
def power_principal_axis(mat: torch.Tensor, *, iters=6, seed=0):
    dt = _F64 if mat.dtype == _F64 else _F32
    sym = _sq_sym(mat, dt)
    sc = sym.abs().amax(dim=(-2, -1), keepdim=True)
    act = sc > 0.0
    asc = torch.where(act, sym / sc.clamp_min(1e-300 if dt == _F64 else 1e-38), 0.0)
    u = _weyl_vec(mat.shape[-1], seed, mat.device, dt).expand(*sym.shape[:-1]).clone()
    for _ in range(max(1, int(iters))):
        u = _norm_step(torch.matmul(asc, u.unsqueeze(-1)).squeeze(-1), u, -1, 1e-12)
    u = torch.where(torch.gather(u, -1, u.abs().argmax(dim=-1, keepdim=True)) < 0.0, -u, u)
    lam = (u * torch.matmul(asc, u.unsqueeze(-1)).squeeze(-1)).sum(dim=-1) * sc.squeeze(-1).squeeze(-1)
    return torch.where(act.squeeze(-1).squeeze(-1), lam, 0.0).to(mat.dtype), u.to(mat.dtype)


def _init_pow_v(iv, bs, nd, ub, dev):
    fb = _weyl_vec(nd, 0, dev, _F32)
    if iv is None:
        return fb.unsqueeze(0).expand(bs, nd).clone()
    v = iv.detach().to(dev, _F32)
    return _norm_step((v.unsqueeze(0) if ub and v.ndim == 1 else v).reshape(bs, nd), fb, 1, 1e-12)


@torch.no_grad()
def stable_rank_power(mat: torch.Tensor, *, iters=2, init_v=None, return_v=False):
    if mat.ndim not in (2, 3):
        raise ValueError("expected 2D/3D")
    ub = mat.ndim == 2
    b = mat.unsqueeze(0) if ub else mat
    bs, rd, cd = b.shape
    if b.numel() == 0:
        er = mat.new_zeros(() if ub else (bs,), dtype=_F32)
        return (er, mat.new_zeros((max(rd, cd),) if ub else (bs, max(rd, cd)), dtype=_F32)) if return_v else er
    guf, _, okf = me_unit_rows(b.float().reshape(bs, rd * cd))
    gu = guf.view(bs, rd, cd)
    mc = gu.transpose(1, 2) if rd > cd else gu
    md, nd = mc.shape[1], mc.shape[2]
    v = _init_pow_v(init_v, bs, nd, ub, mc.device)
    for _ in range(max(1, int(iters))):
        u = torch.bmm(mc, v.unsqueeze(-1)).squeeze(-1)
        u = _norm_step(u, u, 1, 1e-20)
        v = _norm_step(torch.bmm(mc.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1), v, 1, 1e-20)
    s1 = torch.bmm(mc, v.unsqueeze(-1)).squeeze(-1).square().sum(dim=1)
    rs = torch.where(okf.squeeze(1), (1.0 / s1.clamp_min(1e-30)).clamp(1.0, float(md)), float(md))
    if ub:
        rs, v = rs.squeeze(0), v.squeeze(0)
    return (rs, v) if return_v else rs


@torch.no_grad()
def sym_eigensplit_halley(mat: torch.Tensor, *, steps=8):
    sym = _sq_sym(mat, _F64)
    d = sym.shape[-1]
    eye = torch.eye(d, dtype=_F64, device=sym.device)
    rms = (sym.square().sum(dim=(-2, -1), keepdim=True) / d).sqrt()
    x = torch.where(rms > 0.0, sym / rms.clamp_min(1e-300), 0.0)
    for _ in range(max(1, int(steps))):
        x2 = torch.matmul(x, x)
        ch, _ = torch.linalg.cholesky_ex(eye + 3.0 * x2, check_errors=False)
        x = torch.cholesky_solve(torch.matmul(x, 3.0 * eye + x2), ch)
        x = 0.5 * (x + x.transpose(-1, -2))
    ab = 0.5 * (torch.matmul(x, sym) + torch.matmul(sym, x))
    return (0.5 * (sym + ab)).to(mat.dtype), (0.5 * (sym - ab)).to(mat.dtype)


@torch.no_grad()
def sym_inv_sqrt_ns(cov: torch.Tensor, *, eps=0.0, steps=20):
    a = _sq_sym(cov, _F64)
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("invalid eps")
    eye = torch.eye(a.shape[-1], dtype=_F64, device=a.device)
    if eps > 0.0:
        a = a + eps * eye
    s0 = a.abs().amax(dim=(-2, -1), keepdim=True)
    act = s0 > 0.0
    an = torch.where(act, a / s0.clamp_min(1e-300), eye)
    sr = torch.matmul(an, an).square().sum(dim=(-2, -1), keepdim=True).pow(0.25) * 0.60
    y, z = an / sr.clamp_min(1e-300), eye.expand_as(a).clone()
    for _ in range(max(1, int(steps))):
        t = 0.5 * (3.0 * eye - torch.matmul(z, y))
        y, z = torch.matmul(y, t), torch.matmul(t, z)
    out = 0.5 * (z + z.transpose(-1, -2)) * torch.rsqrt((sr * s0).clamp_min(1e-300))
    return torch.where(act, out, 0.0).to(cov.dtype)


def _add_ridge(m, ridge):
    eye = torch.eye(m.shape[-1], dtype=m.dtype, device=m.device)
    if isinstance(ridge, torch.Tensor):
        r = ridge.to(m.device, m.dtype)
        return m + (torch.diag_embed(r) if r.ndim >= 1 else r * eye)
    return m + float(ridge) * eye if ridge != 0.0 else m


@torch.no_grad()
def solve_spd_cholesky(gram: torch.Tensor, rhs: torch.Tensor, *, ridge=0.0, return_inv=False):
    am = _add_ridge(_sq_sym(gram, _F64), ridge)
    d = am.shape[-1]
    ch, info = torch.linalg.cholesky_ex(am, check_errors=False)
    if bool((info != 0).any()):
        ep, _ = sym_eigensplit_halley(am)
        dsc = ep.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
        cf, _ = torch.linalg.cholesky_ex(
            ep + (1e-6 * dsc) * torch.eye(d, dtype=_F64, device=am.device), check_errors=False
        )
        ch = torch.where((info != 0).unsqueeze(-1).unsqueeze(-1), cf, ch)
    r = rhs.to(am.device, _F64)
    vec = r.ndim == am.ndim - 1
    sol = torch.cholesky_solve(r.unsqueeze(-1) if vec else r, ch)
    out = (sol.squeeze(-1) if vec else sol).to(rhs.dtype)
    if not return_inv:
        return out, None
    inv = torch.cholesky_solve(torch.eye(d, dtype=_F64, device=am.device).expand_as(am), ch)
    return out, (0.5 * (inv + inv.transpose(-1, -2))).to(gram.dtype)


def _ns_smooth(rs, dev, bs):
    if isinstance(rs, torch.Tensor):
        return rs.to(dev, _F32).reshape(bs, 1, 1).clamp_min(0.0), True
    sv = max(0.0, float(rs))
    return torch.full((bs, 1, 1), sv, device=dev, dtype=_F32), sv > 0.0


def _ns_prep(xi, rs):
    bs, md = xi.shape[0], xi.shape[1]
    sr, hs = _ns_smooth(rs, xi.device, bs)
    dt = torch.bfloat16 if (xi.is_cuda and md > 256) else _F32
    if not hs:
        return (xi / 1.02).to(dt), None
    if md > 512:
        return (xi / 1.02).to(dt), torch.rsqrt(1.0 + sr.square())
    ia = 1.0 / (torch.sqrt(1.0 + sr.square()).clamp_min(1e-30) * 1.02)
    rhs = torch.eye(md, device=xi.device, dtype=dt).unsqueeze(0).expand(bs, md, md) * ((sr / math.sqrt(md)) * ia).to(dt)
    return torch.cat(((xi * ia).to(dt), rhs), dim=2), None


@torch.no_grad()
def fp32_tikhonov_ns_batched(mats: torch.Tensor, steps=5, *, rel_smooth=0.0, pre_normalized=False):
    if mats.ndim != 3:
        raise ValueError("expected 3D")
    bs, rd, cd = mats.shape
    dt0 = mats.dtype
    if mats.numel() == 0:
        return torch.zeros_like(mats)
    if pre_normalized:
        xu, ok = mats.float(), None
    else:
        xuf, _, okf = me_unit_rows(mats.float().reshape(bs, rd * cd))
        xu, ok = xuf.view(bs, rd, cd), okf.view(bs, 1, 1)
    if min(rd, cd) == 1:
        sr, _ = _ns_smooth(rel_smooth, mats.device, bs)
        res = xu * torch.rsqrt(1.0 + sr.square())
        return (res if ok is None else torch.where(ok, res, 0.0)).to(dt0)
    tr = rd > cd
    xi = xu.transpose(1, 2) if tr else xu
    md, nd = xi.shape[1], xi.shape[2]
    z, psc = _ns_prep(xi, rel_smooth)
    for i in range(max(0, int(steps))):
        a, b, c = _QUINTIC[min(i, len(_QUINTIC) - 1)]
        g = torch.bmm(z, z.transpose(1, 2))
        z = a * z + torch.bmm(b * g + c * torch.bmm(g, g), z)
    zf = z.float()
    sc = (math.sqrt(md) / _bnorm(zf).clamp_min(1e-8)).clamp_max(1.15)
    out = zf[:, :, :nd] * (sc if psc is None else sc * psc)
    out = out.transpose(1, 2) if tr else out
    val = torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    return torch.where(val if ok is None else ok & val, out, 0.0).to(dt0)


@torch.no_grad()
def fp32_tikhonov_ns_2d(mat: torch.Tensor, steps=5, *, rel_smooth=0.0):
    if mat.ndim != 2:
        raise ValueError("expected 2D")
    sm = rel_smooth.reshape(1) if isinstance(rel_smooth, torch.Tensor) else rel_smooth
    return fp32_tikhonov_ns_batched(mat.unsqueeze(0), steps=steps, rel_smooth=sm).squeeze(0)


@torch.no_grad()
def schatten6_decay_batched(weights: torch.Tensor, rates: torch.Tensor, *, lam=0.25, tau=2.0):
    if weights.ndim != 3:
        raise ValueError("expected 3D")
    bs, rd, cd = weights.shape
    if weights.numel() == 0:
        return torch.zeros_like(weights, dtype=_F32)
    tr = rd > cd
    ww = weights.transpose(1, 2) if tr else weights
    wh = ww.float() / tau
    dt = torch.bfloat16 if (weights.is_cuda and ww.shape[1] >= 64) else _F32
    wb = wh.to(dt)
    g1 = torch.bmm(wb, wb.transpose(1, 2))
    g2 = torch.bmm(g1, g1)
    g2f = g2.float()
    sg2 = g2f.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    g2s = g2f / sg2
    g4u = torch.bmm(g2s.to(dt), g2s.to(dt)).float()
    rg = _bnorm(g4u) / _bnorm(g2s).clamp_min(1e-12)
    gate = smoothstep01(((rg * sg2).clamp_min(0.0).pow(0.25) - 1.0) / 0.25)
    w5 = torch.bmm(g2, wb).float()
    nr = (_bnorm(w5) / _bnorm(wh).clamp_min(1e-12)).clamp_min(1e-6)
    rv = rates.to(weights.device, _F32).reshape(bs, 1, 1)
    eff = torch.minimum((lam / tau) * rv * gate, 0.25 / nr)
    out = -w5 * (eff * tau)
    ok = torch.isfinite(out).all(dim=(1, 2), keepdim=True) & torch.isfinite(eff) & (gate > 0.0)
    out = torch.where(ok, out, 0.0)
    return out.transpose(1, 2) if tr else out


def _solve_2d(c0, c1, rho, d2):
    tol = 1e-10 * max(1.0, math.sqrt(max(0.0, d2)))
    bd, best = d2, None
    for ck, oth, w in ((c0, c1 - c0 * rho, (c0, 0.0)), (c1, c0 - c1 * rho, (0.0, c1))):
        if ck >= -tol and oth <= tol and ck * ck < bd:
            best, bd = (max(0.0, w[0]), max(0.0, w[1])), ck * ck
    det = 1.0 - rho * rho
    if det > 1e-14:
        m0, m1 = (c0 - rho * c1) / det, (c1 - rho * c0) / det
        db = m0 * m0 + m1 * m1 + 2.0 * m0 * m1 * rho
        if m0 >= -tol and m1 >= -tol and (c0 - m0 - m1 * rho) <= tol and (c1 - m0 * rho - m1) <= tol and db < bd:
            best = (max(0.0, m0), max(0.0, m1))
    return best


def solve_nonascent_multipliers(ad, a_norm_sq, d_norm_sq, a01=0.0):
    nc = len(ad)
    if nc != len(a_norm_sq) or nc > 2:
        raise ValueError("at most two constraints")
    if nc == 0:
        return (), False, False
    d2 = max(0.0, float(d_norm_sq))
    act = [(i, math.sqrt(float(n2)), float(v)) for i, (n2, v) in enumerate(zip(a_norm_sq, ad)) if float(n2) > 0.0]
    zs = (0.0,) * nc
    if not act:
        return zs, False, False
    if len(act) == 1:
        idx, ell, dot = act[0]
        cv = dot / ell
        if cv <= 0.0:
            return zs, False, False
        if cv * cv >= d2:
            return zs, True, True
        al = [0.0] * nc
        al[idx] = cv / ell
        return tuple(al), False, True
    (_, l0, ad0), (_, l1, ad1) = act
    c0, c1 = ad0 / l0, ad1 / l1
    if c0 <= 0.0 and c1 <= 0.0:
        return (0.0, 0.0), False, False
    sol = _solve_2d(c0, c1, max(-1.0, min(1.0, float(a01) / (l0 * l1))), d2)
    return ((0.0, 0.0), True, True) if sol is None else ((sol[0] / l0, sol[1] / l1), False, True)


def _pdot(xs, ys, z0):
    acc = z0
    for xp, yp in zip(xs, ys):
        if xp is not None and yp is not None:
            acc = acc + (xp.double() * yp.double()).sum()
    return acc


def _solve_kkt(d2, n0, a0, n1=None, a1=None, a01=None):
    if n1 is None:
        v = torch.stack((n0, a0, d2)).tolist()
        return solve_nonascent_multipliers((v[1],), (v[0],), v[2])
    v = torch.stack((n0, n1, a0, a1, a01, d2)).tolist()
    return solve_nonascent_multipliers((v[2], v[3]), (v[0], v[1]), v[5], v[4])


def _proj_dense(disp, normals):
    nc = normals.shape[0]
    if nc > 2:
        raise ValueError("at most two constraints")
    if nc == 0:
        return disp.clone()
    d64, a64 = disp.double(), normals.double()
    an2, ad, d2 = (a64 * a64).sum(dim=1), a64 @ d64, (d64 * d64).sum()
    al, iz, pr = (
        _solve_kkt(d2, an2[0], ad[0])
        if nc == 1
        else _solve_kkt(d2, an2[0], ad[0], an2[1], ad[1], (a64[0] * a64[1]).sum())
    )
    if not pr:
        return disp.clone()
    if iz:
        return torch.zeros_like(disp)
    out = d64 - al[0] * a64[0]
    if nc == 2 and al[1] != 0.0:
        out = out - al[1] * a64[1]
    return out.to(disp)


def _apply_stream(disps, normals, al):
    act = [(a, g) for a, g in zip(al, normals) if a != 0.0]
    out = []
    for i, dp in enumerate(disps):
        acc = dp.double()
        for a, g in act:
            ap = g[i]
            if ap is not None:
                acc = acc - a * ap.double()
        out.append(acc.to(dp))
    return out


def _proj_stream(disps, normals):
    nc = len(normals)
    if nc > 2:
        raise ValueError("at most two constraints")
    if nc == 0 or not disps:
        return [d.clone() for d in disps]
    z0 = torch.zeros((), dtype=_F64, device=disps[0].device)
    d2, n0, a0 = _pdot(disps, disps, z0), _pdot(normals[0], normals[0], z0), _pdot(normals[0], disps, z0)
    if nc == 1:
        al, iz, pr = _solve_kkt(d2, n0, a0)
    else:
        al, iz, pr = _solve_kkt(
            d2,
            n0,
            a0,
            _pdot(normals[1], normals[1], z0),
            _pdot(normals[1], disps, z0),
            _pdot(normals[0], normals[1], z0),
        )
    if not pr:
        return [d.clone() for d in disps]
    return [torch.zeros_like(d) for d in disps] if iz else _apply_stream(disps, normals, al)


@torch.no_grad()
def zero_pinv_project_nonascent(displacement, normals):
    if isinstance(displacement, torch.Tensor):
        if not isinstance(normals, torch.Tensor):
            raise TypeError("tensor normals required")
        return _proj_dense(displacement, normals)
    return _proj_stream(displacement, normals)


def safe_cdist(a: torch.Tensor, b=None, *, diag_inf=False):
    na = a.shape[0]
    af = a.reshape(na, -1).float()
    bf = af if b is None else b.reshape(b.shape[0], -1).float()
    if na == 0 or bf.shape[0] == 0:
        return af.new_empty((na, bf.shape[0]), dtype=_F32)
    sc = _amax_sc(af, bf)
    dist = torch.cdist(af / sc, bf / sc, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") * sc
    if diag_inf and b is None:
        return (
            dist.masked_fill(torch.eye(na, device=dist.device, dtype=_BOOL), float("inf"))
            if dist.requires_grad
            else dist.fill_diagonal_(float("inf"))
        )
    return dist


def born_infeld_cost(x: torch.Tensor, y: torch.Tensor, ell, *, aligned=False, fp64=False):
    dt = _F64 if fp64 else _F32
    xf, yf = x.reshape(x.shape[0], -1).to(dt), y.reshape(y.shape[0], -1).to(dt)
    et = ell.detach().to(xf.device, dt) if isinstance(ell, torch.Tensor) else xf.new_tensor(float(ell))
    sc = _amax_sc(xf, yf, et)
    xu, yu, eu = xf / sc, yf / sc, (et / sc).clamp_min(1e-12 if fp64 else 1e-7)
    if aligned:
        sq = (xu - yu).square().sum(dim=-1)
    else:
        s0 = xu.square().sum(dim=-1, keepdim=True) + yu.square().sum(dim=-1).unsqueeze(0) - 2.0 * (xu @ yu.T)
        sq = s0 + (s0.detach().clamp_min(0.0) - s0.detach())
    rad = torch.sqrt(eu.square() + sq)
    cu = (rad - eu) + (sq / (rad + eu).clamp_min(1e-30) - (rad - eu)).detach()
    return (cu * sc).to(x.dtype), (cu.detach() / eu.clamp_min(1e-30)).to(_F32)


def _lse_dual(lk, dual, dim, target, buf):
    torch.add(lk, dual.unsqueeze(1 - dim), out=buf)
    m = buf.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, 0.0)
    return target - (m.squeeze(dim) + buf.sub_(m).exp_().sum(dim=dim).clamp_min(1e-30).log_())


@torch.no_grad()
def log_sinkhorn(log_kernel: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor, *, iters=10):
    buf, u, v = torch.empty_like(log_kernel), torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(max(1, int(iters))):
        v = _lse_dual(log_kernel, u, 0, log_b, buf)
        u = _lse_dual(log_kernel, v, 1, log_a, buf)
    return torch.add(log_kernel, u.unsqueeze(1), out=buf).add_(v.unsqueeze(0))


@torch.no_grad()
def sinkhorn_with_slack(c_hat, d_row, d_col, *, rho, tau, eps, iters=10, mask=None):
    br, bc = c_hat.shape[0], c_hat.shape[1]
    lk = c_hat.new_zeros((br + 1, bc + 1), dtype=_F32)
    act = -c_hat.float().clamp(0.0, 1e6) / eps
    lk[:br, :bc] = act.masked_fill(~mask, float("-inf")) if mask is not None else act
    lk[:br, bc] = (-0.5 * rho + 0.5 * tau * d_row.float()) / eps
    lk[br, :bc] = (-0.5 * rho - 0.5 * tau * d_col.float()) / eps
    lr, lc = lk.new_full((br + 1,), -math.log(max(1, br))), lk.new_full((bc + 1,), -math.log(max(1, bc)))
    lr[br] = lc[bc] = math.log(2.0)
    return log_sinkhorn(lk, lr, lc, iters=iters).exp_().to(c_hat.dtype)


def _norm_weyl(weights):
    k, dev = weights.numel(), weights.device
    w = weights.detach().double().clamp_min(0.0)
    ws = w.sum()
    w = torch.where(ws > 1e-12, w / ws.clamp_min(1e-12), 1.0 / k)
    weyl = torch.remainder((torch.arange(k, device=dev, dtype=_F64) + 0.5) * PHI1, 1.0)
    return w, torch.argsort(weyl, descending=True, stable=True)


def _rank(oi, key):
    return oi[torch.argsort(key[oi], descending=True, stable=True)]


def _add_rem(q, oi, total):
    rem = torch.tensor(int(total), dtype=_LONG, device=q.device) - q.sum()
    return q.scatter_add_(0, oi, (torch.arange(q.numel(), device=q.device, dtype=_LONG) < rem.clamp_min(0)).to(_LONG))


@torch.no_grad()
def hamilton_quotas(total: int, weights: torch.Tensor, *, active_scores=None, min_one_when_feasible=True):
    k, dev = weights.numel(), weights.device
    if k == 0 or total <= 0:
        return torch.zeros(k, dtype=_LONG, device=dev)
    w, oi = _norm_weyl(weights)
    if min_one_when_feasible and total >= k:
        rem = int(total) - k
        if rem == 0:
            return torch.ones(k, dtype=_LONG, device=dev)
        ex = (w * total - 1.0).clamp_min(0.0)
        es = ex.sum()
        sc = torch.where(es > 1e-12, ex * (rem / es.clamp_min(1e-12)), w * rem)
        q, frac = torch.ones(k, dtype=_LONG, device=dev) + torch.floor(sc).long(), sc - torch.floor(sc)
    else:
        raw = w * total
        q, frac = torch.floor(raw).long(), raw - torch.floor(raw)
    if active_scores is not None:
        oi = _rank(oi, active_scores.detach().to(dev, _F64).reshape(k))
    return _add_rem(q, _rank(oi, frac), total)


@torch.no_grad()
def allocate_deficit_quotas(proto_w: torch.Tensor, w_fake_ema: torch.Tensor, n_slots: int):
    k, dev = proto_w.numel(), proto_w.device
    if k == 0 or n_slots <= 0:
        return torch.zeros(k, device=dev, dtype=_LONG)
    w, oi = _norm_weyl(proto_w)
    df = (proto_w.detach() - w_fake_ema.detach()).double()
    if n_slots < k:
        oi = _rank(_rank(oi, w), df)
        return torch.zeros(k, device=dev, dtype=_LONG).scatter_(
            0, oi, (torch.arange(k, device=dev, dtype=_LONG) < n_slots).to(_LONG)
        )
    ex = w * n_slots
    q = torch.floor(ex).to(_LONG)
    return _add_rem(q, _rank(_rank(oi, df), ex - q.double()), n_slots)


__all__ = [
    "PHI1",
    "PHI2",
    "allocate_deficit_quotas",
    "born_infeld_cost",
    "fp32_tikhonov_ns_2d",
    "fp32_tikhonov_ns_batched",
    "hamilton_quotas",
    "log_sinkhorn",
    "mantissa_exponent_normalize",
    "me_unit_rows",
    "power_principal_axis",
    "safe_cdist",
    "schatten6_decay_batched",
    "sinkhorn_with_slack",
    "smoothstep01",
    "solve_nonascent_multipliers",
    "solve_spd_cholesky",
    "stable_rank_power",
    "sym_eigensplit_halley",
    "sym_inv_sqrt_ns",
    "weyl_rademacher",
    "zero_pinv_project_nonascent",
]
