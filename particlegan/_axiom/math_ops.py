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


def _sym_square(mat, dtype):
    if mat.ndim < 2 or mat.shape[-1] != mat.shape[-2] or mat.shape[-1] < 1:
        raise ValueError("expected square")
    out = mat.to(dtype)
    return 0.5 * (out + out.transpose(-1, -2))


def _max_scale(*tensors):
    peak = tensors[0].detach().abs().amax()
    for tensor in tensors[1:]:
        peak = torch.maximum(peak, tensor.detach().abs().amax())
    return torch.where(peak > 0.0, peak.clamp_min(1e-30), 1.0)


def _unit_vec(vec, fallback, dim, eps):
    norm = vec.norm(dim=dim, keepdim=True)
    return torch.where(norm > eps, vec / norm.clamp_min(eps), fallback)


def _batch_norm(batch):
    return batch.flatten(1).norm(dim=1).view(-1, 1, 1)


def smoothstep01(x):
    """C1 cubic smoothstep on [0, 1]."""
    u = x.clamp(0.0, 1.0) if isinstance(x, torch.Tensor) else min(1.0, max(0.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


@torch.no_grad()
def me_unit_rows(x: torch.Tensor):
    """Scale-free IEEE-754 mantissa-exponent row normalization."""
    if x.ndim != 2:
        raise ValueError("expected 2D")
    rows, cols = x.shape
    if rows == 0 or cols == 0:
        return (
            x.new_zeros((rows, cols), dtype=_F32),
            x.new_zeros((rows, 1), dtype=_F32),
            x.new_zeros((rows, 1), dtype=_BOOL),
        )
    xf = x.float()
    finite = torch.isfinite(xf).all(dim=1, keepdim=True)
    safe = torch.where(finite, xf, 0.0)
    peak = safe.abs().amax(dim=1, keepdim=True)
    active = finite & (peak > 0.0)
    ones = torch.ones_like(peak)
    _, exp = torch.frexp(torch.where(active, peak, ones))
    half = (-exp) // 2
    mant = (safe * torch.ldexp(ones, half)) * torch.ldexp(ones, (-exp) - half)
    norm = mant.norm(dim=1, keepdim=True)
    valid = active & (norm > 0.0) & torch.isfinite(norm)
    clamped = norm.clamp_min(1e-30)
    return torch.where(valid, mant / clamped, 0.0), torch.where(valid, exp.float() + torch.log2(clamped), 0.0), valid


@torch.no_grad()
def mantissa_exponent_normalize(grad: torch.Tensor):
    """Return ``(unit, log2_norm, valid)`` without underflow or overflow."""
    if grad.numel() == 0:
        return grad.float().clone(), grad.new_zeros((), dtype=_F32), grad.new_zeros((), dtype=_BOOL)
    unit, log2_norm, valid = me_unit_rows(grad.float().reshape(1, -1))
    return unit.reshape_as(grad), log2_norm.reshape(()), valid.reshape(())


@torch.no_grad()
def weyl_rademacher(shape, *, phase=0.0, salt=0, device=None, dtype=_F32):
    """Deterministic +/-1 Rademacher probe tensor with zero RNG mutation."""
    dims = tuple(int(s) for s in shape)
    total = math.prod(dims) if dims else 1
    if total == 0:
        return torch.empty(dims, device=device, dtype=dtype)
    seed = (int(math.floor((float(phase) % 1.0) * 1073741824.0)) + int(salt) * 7919) & 0xFFFFFFFF
    val = (
        torch.arange(total, device=device, dtype=torch.int64) * 2654435761 + seed * 3266489917 + 374761393
    ) & 0xFFFFFFFF
    val = ((val ^ (val >> 15)) * 2246822519) & 0xFFFFFFFF
    val = ((val ^ (val >> 13)) * 3266489917) & 0xFFFFFFFF
    return torch.where(((val ^ (val >> 16)) & 1) == 0, 1.0, -1.0).to(dtype).reshape(dims)


def _weyl_unit(dim, seed, dev, dtype):
    raw = torch.remainder(torch.arange(1, dim + 1, device=dev, dtype=dtype) * PHI1 + (seed + 0.5) * PHI2, 1.0) - 0.5
    norm = raw.norm()
    return torch.where(norm > 1e-8, raw / norm.clamp_min(1e-12), torch.full((dim,), dim**-0.5, device=dev, dtype=dtype))


@torch.no_grad()
def power_principal_axis(mat: torch.Tensor, *, iters=6, seed=0):
    """Deterministic power iteration returning ``(lambda_1, u_1)``."""
    dtype = _F64 if mat.dtype == _F64 else _F32
    sym = _sym_square(mat, dtype)
    scale = sym.abs().amax(dim=(-2, -1), keepdim=True)
    active = scale > 0.0
    scaled = torch.where(active, sym / scale.clamp_min(1e-300 if dtype == _F64 else 1e-38), 0.0)
    vec = _weyl_unit(mat.shape[-1], seed, mat.device, dtype).expand(*sym.shape[:-1]).clone()
    for _ in range(max(1, int(iters))):
        vec = _unit_vec(torch.matmul(scaled, vec.unsqueeze(-1)).squeeze(-1), vec, -1, 1e-12)
    vec = torch.where(torch.gather(vec, -1, vec.abs().argmax(dim=-1, keepdim=True)) < 0.0, -vec, vec)
    lam = (vec * torch.matmul(scaled, vec.unsqueeze(-1)).squeeze(-1)).sum(dim=-1) * scale.squeeze(-1).squeeze(-1)
    return torch.where(active.squeeze(-1).squeeze(-1), lam, 0.0).to(mat.dtype), vec.to(mat.dtype)


def _init_pow_vec(init_v, batch_n, cols, unbatched, dev):
    fallback = _weyl_unit(cols, 0, dev, _F32)
    if init_v is None:
        return fallback.unsqueeze(0).expand(batch_n, cols).clone()
    vec = init_v.detach().to(dev, _F32)
    return _unit_vec(
        (vec.unsqueeze(0) if unbatched and vec.ndim == 1 else vec).reshape(batch_n, cols), fallback, 1, 1e-12
    )


@torch.no_grad()
def stable_rank_power(mat: torch.Tensor, *, iters=2, init_v=None, return_v=False):
    """Estimate matrix stable rank via power iteration."""
    if mat.ndim not in (2, 3):
        raise ValueError("expected 2D/3D")
    unbatched = mat.ndim == 2
    batch = mat.unsqueeze(0) if unbatched else mat
    batch_n, rows, cols = batch.shape
    if batch.numel() == 0:
        empty = mat.new_zeros(() if unbatched else (batch_n,), dtype=_F32)
        if not return_v:
            return empty
        return empty, mat.new_zeros((max(rows, cols),) if unbatched else (batch_n, max(rows, cols)), dtype=_F32)
    unit_flat, _, valid = me_unit_rows(batch.float().reshape(batch_n, rows * cols))
    unit = unit_flat.view(batch_n, rows, cols)
    wide = unit.transpose(1, 2) if rows > cols else unit
    short_d, long_d = wide.shape[1], wide.shape[2]
    vec = _init_pow_vec(init_v, batch_n, long_d, unbatched, wide.device)
    for _ in range(max(1, int(iters))):
        left = torch.bmm(wide, vec.unsqueeze(-1)).squeeze(-1)
        left = _unit_vec(left, left, 1, 1e-20)
        vec = _unit_vec(torch.bmm(wide.transpose(1, 2), left.unsqueeze(-1)).squeeze(-1), vec, 1, 1e-20)
    s1_sq = torch.bmm(wide, vec.unsqueeze(-1)).squeeze(-1).square().sum(dim=1)
    rank = torch.where(valid.squeeze(1), (1.0 / s1_sq.clamp_min(1e-30)).clamp(1.0, float(short_d)), float(short_d))
    if unbatched:
        rank, vec = rank.squeeze(0), vec.squeeze(0)
    return (rank, vec) if return_v else rank


@torch.no_grad()
def sym_eigensplit_halley(mat: torch.Tensor, *, steps=8):
    """Split symmetric matrix into PSD/NSD parts via FP64 Pade-(1,1) Halley."""
    sym = _sym_square(mat, _F64)
    dim = sym.shape[-1]
    eye = torch.eye(dim, dtype=_F64, device=sym.device)
    rms = (sym.square().sum(dim=(-2, -1), keepdim=True) / dim).sqrt()
    sign = torch.where(rms > 0.0, sym / rms.clamp_min(1e-300), 0.0)
    for _ in range(max(1, int(steps))):
        sq = torch.matmul(sign, sign)
        chol, _ = torch.linalg.cholesky_ex(eye + 3.0 * sq, check_errors=False)
        sign = torch.cholesky_solve(torch.matmul(sign, 3.0 * eye + sq), chol)
        sign = 0.5 * (sign + sign.transpose(-1, -2))
    abs_sym = 0.5 * (torch.matmul(sign, sym) + torch.matmul(sym, sign))
    return (0.5 * (sym + abs_sym)).to(mat.dtype), (0.5 * (sym - abs_sym)).to(mat.dtype)


@torch.no_grad()
def sym_inv_sqrt_ns(cov: torch.Tensor, *, eps=0.0, steps=20):
    """Compute symmetric ``(cov + eps I)^{-1/2}`` via FP64 coupled Newton-Schulz."""
    sym = _sym_square(cov, _F64)
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("invalid eps")
    eye = torch.eye(sym.shape[-1], dtype=_F64, device=sym.device)
    if eps > 0.0:
        sym = sym + eps * eye
    scale = sym.abs().amax(dim=(-2, -1), keepdim=True)
    active = scale > 0.0
    normed = torch.where(active, sym / scale.clamp_min(1e-300), eye)
    spec = torch.matmul(normed, normed).square().sum(dim=(-2, -1), keepdim=True).pow(0.25) * 0.60
    y_mat, z_mat = normed / spec.clamp_min(1e-300), eye.expand_as(sym).clone()
    for _ in range(max(1, int(steps))):
        step_m = 0.5 * (3.0 * eye - torch.matmul(z_mat, y_mat))
        y_mat, z_mat = torch.matmul(y_mat, step_m), torch.matmul(step_m, z_mat)
    out = 0.5 * (z_mat + z_mat.transpose(-1, -2)) * torch.rsqrt((spec * scale).clamp_min(1e-300))
    return torch.where(active, out, 0.0).to(cov.dtype)


@torch.no_grad()
def solve_spd_cholesky(gram: torch.Tensor, rhs: torch.Tensor, *, ridge=0.0, return_inv=False):
    """Solve ``(gram + ridge) X = rhs`` via Cholesky with Halley PSD fallback."""
    mat = _sym_square(gram, _F64)
    dim = mat.shape[-1]
    eye = torch.eye(dim, dtype=_F64, device=mat.device)
    if isinstance(ridge, torch.Tensor):
        ridge_t = ridge.to(mat.device, _F64)
        mat = mat + (torch.diag_embed(ridge_t) if ridge_t.ndim >= 1 else ridge_t * eye)
    elif ridge != 0.0:
        mat = mat + float(ridge) * eye
    chol, info = torch.linalg.cholesky_ex(mat, check_errors=False)
    if bool((info != 0).any()):
        pos, _ = sym_eigensplit_halley(mat)
        diag_sc = pos.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
        fb_chol, _ = torch.linalg.cholesky_ex(pos + (1e-6 * diag_sc) * eye, check_errors=False)
        chol = torch.where((info != 0).unsqueeze(-1).unsqueeze(-1), fb_chol, chol)
    rhs64 = rhs.to(mat.device, _F64)
    is_vec = rhs64.ndim == mat.ndim - 1
    sol = torch.cholesky_solve(rhs64.unsqueeze(-1) if is_vec else rhs64, chol)
    out = (sol.squeeze(-1) if is_vec else sol).to(rhs.dtype)
    if not return_inv:
        return out, None
    inv = torch.cholesky_solve(eye.expand_as(mat), chol)
    return out, (0.5 * (inv + inv.transpose(-1, -2))).to(gram.dtype)


def _ns_smooth(rel_smooth, dev, batch_n):
    if isinstance(rel_smooth, torch.Tensor):
        return rel_smooth.to(dev, _F32).reshape(batch_n, 1, 1).clamp_min(0.0), True
    val = max(0.0, float(rel_smooth))
    return torch.full((batch_n, 1, 1), val, device=dev, dtype=_F32), val > 0.0


def _ns_augment(wide, rel_smooth):
    batch_n, short_d = wide.shape[0], wide.shape[1]
    smooth, has_sm = _ns_smooth(rel_smooth, wide.device, batch_n)
    dtype = torch.bfloat16 if (wide.is_cuda and short_d > 256) else _F32
    if not has_sm:
        return (wide / 1.02).to(dtype), None
    if short_d > 512:
        return (wide / 1.02).to(dtype), torch.rsqrt(1.0 + smooth.square())
    inv_a = 1.0 / (torch.sqrt(1.0 + smooth.square()).clamp_min(1e-30) * 1.02)
    eye = torch.eye(short_d, device=wide.device, dtype=dtype).unsqueeze(0).expand(batch_n, short_d, short_d)
    return torch.cat(((wide * inv_a).to(dtype), eye * ((smooth / math.sqrt(short_d)) * inv_a).to(dtype)), dim=2), None


@torch.no_grad()
def fp32_tikhonov_ns_batched(mats: torch.Tensor, steps=5, *, rel_smooth=0.0, pre_normalized=False):
    """Batched quintic Tikhonov Newton-Schulz polar orthogonalization."""
    if mats.ndim != 3:
        raise ValueError("expected 3D")
    batch_n, rows, cols = mats.shape
    orig_dt = mats.dtype
    if mats.numel() == 0:
        return torch.zeros_like(mats)
    if pre_normalized:
        unit, valid = mats.float(), None
    else:
        unit_flat, _, valid_f = me_unit_rows(mats.float().reshape(batch_n, rows * cols))
        unit, valid = unit_flat.view(batch_n, rows, cols), valid_f.view(batch_n, 1, 1)
    if min(rows, cols) == 1:
        smooth, _ = _ns_smooth(rel_smooth, mats.device, batch_n)
        scaled = unit * torch.rsqrt(1.0 + smooth.square())
        return (scaled if valid is None else torch.where(valid, scaled, 0.0)).to(orig_dt)
    trans = rows > cols
    wide = unit.transpose(1, 2) if trans else unit
    short_d, long_d = wide.shape[1], wide.shape[2]
    z_mat, post_sc = _ns_augment(wide, rel_smooth)
    for idx in range(max(0, int(steps))):
        ca, cb, cc = _QUINTIC[min(idx, len(_QUINTIC) - 1)]
        gram = torch.bmm(z_mat, z_mat.transpose(1, 2))
        z_mat = ca * z_mat + torch.bmm(cb * gram + cc * torch.bmm(gram, gram), z_mat)
    z_f32 = z_mat.float()
    gain = (math.sqrt(short_d) / _batch_norm(z_f32).clamp_min(1e-8)).clamp_max(1.15)
    out = z_f32[:, :, :long_d] * (gain if post_sc is None else gain * post_sc)
    out = out.transpose(1, 2) if trans else out
    fin = torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    return torch.where(fin if valid is None else valid & fin, out, 0.0).to(orig_dt)


@torch.no_grad()
def fp32_tikhonov_ns_2d(mat: torch.Tensor, steps=5, *, rel_smooth=0.0):
    """Single-matrix quintic Tikhonov Newton-Schulz polar orthogonalization."""
    if mat.ndim != 2:
        raise ValueError("expected 2D")
    sm = rel_smooth.reshape(1) if isinstance(rel_smooth, torch.Tensor) else rel_smooth
    return fp32_tikhonov_ns_batched(mat.unsqueeze(0), steps=steps, rel_smooth=sm).squeeze(0)


@torch.no_grad()
def schatten6_decay_batched(weights: torch.Tensor, rates: torch.Tensor, *, lam=0.25, tau=2.0):
    """Exact-deadband Schatten-6 spectral ceiling decay above ``tau``."""
    if weights.ndim != 3:
        raise ValueError("expected 3D")
    batch_n, rows, cols = weights.shape
    if weights.numel() == 0:
        return torch.zeros_like(weights, dtype=_F32)
    trans = rows > cols
    wide = weights.transpose(1, 2) if trans else weights
    scaled_w = wide.float() / tau
    dtype = torch.bfloat16 if (weights.is_cuda and wide.shape[1] >= 64) else _F32
    work_w = scaled_w.to(dtype)
    gram1 = torch.bmm(work_w, work_w.transpose(1, 2))
    gram2 = torch.bmm(gram1, gram1)
    gram2_f = gram2.float()
    peak = gram2_f.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    gram2_u = gram2_f / peak
    gram4 = torch.bmm(gram2_u.to(dtype), gram2_u.to(dtype)).float()
    ratio = _batch_norm(gram4) / _batch_norm(gram2_u).clamp_min(1e-12)
    gate = smoothstep01(((ratio * peak).clamp_min(0.0).pow(0.25) - 1.0) / 0.25)
    pow5 = torch.bmm(gram2, work_w).float()
    norm_r = (_batch_norm(pow5) / _batch_norm(scaled_w).clamp_min(1e-12)).clamp_min(1e-6)
    rate_v = rates.to(weights.device, _F32).reshape(batch_n, 1, 1)
    eff = torch.minimum((lam / tau) * rate_v * gate, 0.25 / norm_r)
    out = -pow5 * (eff * tau)
    valid = torch.isfinite(out).all(dim=(1, 2), keepdim=True) & torch.isfinite(eff) & (gate > 0.0)
    out = torch.where(valid, out, 0.0)
    return out.transpose(1, 2) if trans else out


def _solve_2d_kkt(c0, c1, rho, d2):
    tol = 1e-10 * max(1.0, math.sqrt(max(0.0, d2)))
    best_d, best = d2, None
    for ck, oth, pair in ((c0, c1 - c0 * rho, (c0, 0.0)), (c1, c0 - c1 * rho, (0.0, c1))):
        if ck >= -tol and oth <= tol and ck * ck < best_d:
            best, best_d = (max(0.0, pair[0]), max(0.0, pair[1])), ck * ck
    det = 1.0 - rho * rho
    if det > 1e-14:
        m0, m1 = (c0 - rho * c1) / det, (c1 - rho * c0) / det
        dist = m0 * m0 + m1 * m1 + 2.0 * m0 * m1 * rho
        ok = m0 >= -tol and m1 >= -tol and (c0 - m0 - m1 * rho) <= tol and (c1 - m0 * rho - m1) <= tol
        if ok and dist < best_d:
            best = (max(0.0, m0), max(0.0, m1))
    return best


def solve_nonascent_multipliers(ad, a_norm_sq, d_norm_sq, a01=0.0):
    """Exact zero-pinv KKT multipliers for at most two non-ascent constraints."""
    n_cons = len(ad)
    if n_cons != len(a_norm_sq) or n_cons > 2:
        raise ValueError("at most two constraints")
    if n_cons == 0:
        return (), False, False
    d2 = max(0.0, float(d_norm_sq))
    active = [(i, math.sqrt(float(n2)), float(v)) for i, (n2, v) in enumerate(zip(a_norm_sq, ad)) if float(n2) > 0.0]
    zeros = (0.0,) * n_cons
    if not active:
        return zeros, False, False
    if len(active) == 1:
        idx, length, dot = active[0]
        proj = dot / length
        if proj <= 0.0:
            return zeros, False, False
        if proj * proj >= d2:
            return zeros, True, True
        alphas = [0.0] * n_cons
        alphas[idx] = proj / length
        return tuple(alphas), False, True
    (_, len0, ad0), (_, len1, ad1) = active
    c0, c1 = ad0 / len0, ad1 / len1
    if c0 <= 0.0 and c1 <= 0.0:
        return (0.0, 0.0), False, False
    sol = _solve_2d_kkt(c0, c1, max(-1.0, min(1.0, float(a01) / (len0 * len1))), d2)
    return ((0.0, 0.0), True, True) if sol is None else ((sol[0] / len0, sol[1] / len1), False, True)


def tensor_dot(x, y):
    dtype = _F64 if (x.dtype == _F64 or y.dtype == _F64) else _F32
    xf = x if x.dtype == dtype else x.to(dtype)
    yf = y if y.dtype == dtype else y.to(dtype)
    return torch.dot(xf.reshape(-1), yf.reshape(-1)).double()


def stream_dot(xs, ys, zero):
    acc = zero
    for xp, yp in zip(xs, ys):
        if xp is not None and yp is not None and xp.numel() > 0:
            acc = acc + tensor_dot(xp, yp)
    return acc


def _solve_kkt(d2, n0, a0, n1=None, a1=None, a01=None):
    if n1 is None:
        v = torch.stack((n0, a0, d2)).tolist()
        return solve_nonascent_multipliers((v[1],), (v[0],), v[2])
    v = torch.stack((n0, n1, a0, a1, a01, d2)).tolist()
    return solve_nonascent_multipliers((v[2], v[3]), (v[0], v[1]), v[5], v[4])


def _proj_dense(disp, normals):
    n_cons = normals.shape[0]
    if n_cons > 2:
        raise ValueError("at most two constraints")
    if n_cons == 0:
        return disp.clone()
    d64, a64 = disp.double(), normals.double()
    an2, ad, d2 = (a64 * a64).sum(dim=1), a64 @ d64, (d64 * d64).sum()
    alphas, is_zero, proj = (
        _solve_kkt(d2, an2[0], ad[0])
        if n_cons == 1
        else _solve_kkt(d2, an2[0], ad[0], an2[1], ad[1], (a64[0] * a64[1]).sum())
    )
    if not proj:
        return disp.clone()
    if is_zero:
        return torch.zeros_like(disp)
    out = d64 - alphas[0] * a64[0]
    if n_cons == 2 and alphas[1] != 0.0:
        out = out - alphas[1] * a64[1]
    return out.to(disp)


def apply_stream_projection(disps, normals, alphas):
    active = [(alpha, grp) for alpha, grp in zip(alphas, normals) if alpha != 0.0]
    out = []
    for idx, disp in enumerate(disps):
        dtype = _F64 if disp.dtype == _F64 else _F32
        acc = disp.to(dtype).clone()
        for alpha, grp in active:
            if (normal := grp[idx]) is not None:
                acc.add_(normal if normal.dtype == dtype else normal.to(dtype), alpha=-alpha)
        out.append(acc.to(disp.dtype))
    return out


def _proj_stream(disps, normals):
    n_cons = len(normals)
    if n_cons > 2:
        raise ValueError("at most two constraints")
    if n_cons == 0 or not disps:
        return [d.clone() for d in disps]
    zero = torch.zeros((), dtype=_F64, device=disps[0].device)
    d2, n0, a0 = (
        stream_dot(disps, disps, zero),
        stream_dot(normals[0], normals[0], zero),
        stream_dot(normals[0], disps, zero),
    )
    if n_cons == 1:
        alphas, is_zero, proj = _solve_kkt(d2, n0, a0)
    else:
        alphas, is_zero, proj = _solve_kkt(
            d2,
            n0,
            a0,
            stream_dot(normals[1], normals[1], zero),
            stream_dot(normals[1], disps, zero),
            stream_dot(normals[0], normals[1], zero),
        )
    if not proj:
        return [d.clone() for d in disps]
    return [torch.zeros_like(d) for d in disps] if is_zero else apply_stream_projection(disps, normals, alphas)


@torch.no_grad()
def zero_pinv_project_nonascent(displacement, normals):
    """Project dense or parameter-list displacement onto the non-ascent cone."""
    if isinstance(displacement, torch.Tensor):
        if not isinstance(normals, torch.Tensor):
            raise TypeError("tensor normals required")
        return _proj_dense(displacement, normals)
    return _proj_stream(displacement, normals)


def safe_cdist(a: torch.Tensor, b=None, *, diag_inf=False):
    """Scale-invariant pairwise Euclidean distance matrix."""
    rows_a = a.shape[0]
    af = a.reshape(rows_a, -1).float()
    bf = af if b is None else b.reshape(b.shape[0], -1).float()
    if rows_a == 0 or bf.shape[0] == 0:
        return af.new_empty((rows_a, bf.shape[0]), dtype=_F32)
    scale = _max_scale(af, bf)
    dist = torch.cdist(af / scale, bf / scale, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") * scale
    if diag_inf and b is None:
        return (
            dist.masked_fill(torch.eye(rows_a, device=dist.device, dtype=_BOOL), float("inf"))
            if dist.requires_grad
            else dist.fill_diagonal_(float("inf"))
        )
    return dist


def born_infeld_cost(x: torch.Tensor, y: torch.Tensor, ell, *, aligned=False, fp64=False):
    """Cancellation-free Born-Infeld cost ``sqrt(ell^2 + ||x-y||^2) - ell``."""
    dtype = _F64 if fp64 else _F32
    xf, yf = x.reshape(x.shape[0], -1).to(dtype), y.reshape(y.shape[0], -1).to(dtype)
    ell_t = ell.detach().to(xf.device, dtype) if isinstance(ell, torch.Tensor) else xf.new_tensor(float(ell))
    scale = _max_scale(xf, yf, ell_t)
    xu, yu, eu = xf / scale, yf / scale, (ell_t / scale).clamp_min(1e-12 if fp64 else 1e-7)
    if aligned:
        sq = (xu - yu).square().sum(dim=-1)
    else:
        raw = xu.square().sum(dim=-1, keepdim=True) + yu.square().sum(dim=-1).unsqueeze(0) - 2.0 * (xu @ yu.T)
        sq = raw + (raw.detach().clamp_min(0.0) - raw.detach())
    rad = torch.sqrt(eu.square() + sq)
    cost = (rad - eu) + (sq / (rad + eu).clamp_min(1e-30) - (rad - eu)).detach()
    return (cost * scale).to(x.dtype), (cost.detach() / eu.clamp_min(1e-30)).to(_F32)


def _lse_dual(log_k, dual, dim, target, buf):
    torch.add(log_k, dual.unsqueeze(1 - dim), out=buf)
    peak = buf.amax(dim=dim, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, 0.0)
    return target - (peak.squeeze(dim) + buf.sub_(peak).exp_().sum(dim=dim).clamp_min(1e-30).log_())


@torch.no_grad()
def log_sinkhorn(log_kernel: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor, *, iters=10):
    """Stabilized log-domain Sinkhorn scaling."""
    buf, u_dual, v_dual = torch.empty_like(log_kernel), torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(max(1, int(iters))):
        v_dual = _lse_dual(log_kernel, u_dual, 0, log_b, buf)
        u_dual = _lse_dual(log_kernel, v_dual, 1, log_a, buf)
    return torch.add(log_kernel, u_dual.unsqueeze(1), out=buf).add_(v_dual.unsqueeze(0))


@torch.no_grad()
def sinkhorn_with_slack(c_hat, d_row, d_col, *, rho, tau, eps, iters=10, mask=None):
    """Unbalanced Sinkhorn plan with row/column slack channels."""
    rows, cols = c_hat.shape[0], c_hat.shape[1]
    log_k = c_hat.new_zeros((rows + 1, cols + 1), dtype=_F32)
    act = -c_hat.float().clamp(0.0, 1e6) / eps
    log_k[:rows, :cols] = act.masked_fill(~mask, float("-inf")) if mask is not None else act
    log_k[:rows, cols] = (-0.5 * rho + 0.5 * tau * d_row.float()) / eps
    log_k[rows, :cols] = (-0.5 * rho - 0.5 * tau * d_col.float()) / eps
    log_r = log_k.new_full((rows + 1,), -math.log(max(1, rows)))
    log_c = log_k.new_full((cols + 1,), -math.log(max(1, cols)))
    log_r[rows] = log_c[cols] = math.log(2.0)
    return log_sinkhorn(log_k, log_r, log_c, iters=iters).exp_().to(c_hat.dtype)


def _norm_weyl(weights):
    count, dev = weights.numel(), weights.device
    prob = weights.detach().double().clamp_min(0.0)
    total = prob.sum()
    prob = torch.where(total > 1e-12, prob / total.clamp_min(1e-12), 1.0 / count)
    weyl = torch.remainder((torch.arange(count, device=dev, dtype=_F64) + 0.5) * PHI1, 1.0)
    return prob, torch.argsort(weyl, descending=True, stable=True)


def _rank_by(order, key):
    return order[torch.argsort(key[order], descending=True, stable=True)]


def _add_rem(quotas, order, total):
    rem = torch.tensor(int(total), dtype=_LONG, device=quotas.device) - quotas.sum()
    return quotas.scatter_add_(
        0, order, (torch.arange(quotas.numel(), device=quotas.device, dtype=_LONG) < rem.clamp_min(0)).to(_LONG)
    )


@torch.no_grad()
def hamilton_quotas(total: int, weights: torch.Tensor, *, active_scores=None, min_one_when_feasible=True):
    """GPU-vectorized largest-remainder Hamilton integer quotas."""
    count, dev = weights.numel(), weights.device
    if count == 0 or total <= 0:
        return torch.zeros(count, dtype=_LONG, device=dev)
    prob, order = _norm_weyl(weights)
    if min_one_when_feasible and total >= count:
        rem = int(total) - count
        if rem == 0:
            return torch.ones(count, dtype=_LONG, device=dev)
        excess = (prob * total - 1.0).clamp_min(0.0)
        ex_sum = excess.sum()
        scaled = torch.where(ex_sum > 1e-12, excess * (rem / ex_sum.clamp_min(1e-12)), prob * rem)
        quotas, frac = (
            torch.ones(count, dtype=_LONG, device=dev) + torch.floor(scaled).long(),
            scaled - torch.floor(scaled),
        )
    else:
        raw = prob * total
        quotas, frac = torch.floor(raw).long(), raw - torch.floor(raw)
    if active_scores is not None:
        order = _rank_by(order, active_scores.detach().to(dev, _F64).reshape(count))
    return _add_rem(quotas, _rank_by(order, frac), total)


@torch.no_grad()
def allocate_deficit_quotas(proto_w: torch.Tensor, w_fake_ema: torch.Tensor, n_slots: int):
    """Allocate slots prioritizing starved prototypes by mass deficit."""
    count, dev = proto_w.numel(), proto_w.device
    if count == 0 or n_slots <= 0:
        return torch.zeros(count, device=dev, dtype=_LONG)
    prob, order = _norm_weyl(proto_w)
    deficit = (proto_w.detach() - w_fake_ema.detach()).double()
    if n_slots < count:
        order = _rank_by(_rank_by(order, prob), deficit)
        return torch.zeros(count, device=dev, dtype=_LONG).scatter_(
            0, order, (torch.arange(count, device=dev, dtype=_LONG) < n_slots).to(_LONG)
        )
    expected = prob * n_slots
    quotas = torch.floor(expected).to(_LONG)
    return _add_rem(quotas, _rank_by(_rank_by(order, deficit), expected - quotas.double()), n_slots)


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
