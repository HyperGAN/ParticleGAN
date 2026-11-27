"""Axiom L0 stateless math primitives (zero eigh/svd/pinv/autograd.grad/caches)."""

from __future__ import annotations

import math

import torch

PHI1: float = 0.6180339887498949
PHI2: float = 0.7548776662466927
_F32, _F64, _BOOL, _LONG = torch.float32, torch.float64, torch.bool, torch.long
_QUINTIC_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def _symmetrize_square(mat: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mat.ndim < 2 or mat.shape[-1] != mat.shape[-2] or mat.shape[-1] < 1:
        raise ValueError("expected non-empty square matrix")
    casted = mat.to(dtype)
    return 0.5 * (casted + casted.transpose(-1, -2))


def _abs_max_scale(*tensors: torch.Tensor) -> torch.Tensor:
    peak = tensors[0].detach().abs().amax()
    for tensor in tensors[1:]:
        peak = torch.maximum(peak, tensor.detach().abs().amax())
    return torch.where(peak > 0.0, peak.clamp_min(1e-30), 1.0)


def _unit_or_fallback(vec: torch.Tensor, fallback: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    norm = vec.norm(dim=dim, keepdim=True)
    return torch.where(norm > eps, vec / norm.clamp_min(eps), fallback)


def _batch_frob_norm(batch: torch.Tensor) -> torch.Tensor:
    return batch.flatten(1).norm(dim=1).view(-1, 1, 1)


def smoothstep01(x):
    """Cubic C1 smoothstep mapping (-inf, 0] -> 0 and [1, +inf) -> 1."""
    u = x.clamp(0.0, 1.0) if isinstance(x, torch.Tensor) else min(1.0, max(0.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


@torch.no_grad()
def me_unit_rows(x: torch.Tensor):
    """IEEE-754 mantissa-exponent row normalization safe down to subnormal scales."""
    if x.ndim != 2:
        raise ValueError("expected 2D tensor")
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
    half_exp = (-exp) // 2
    mantissa = (safe * torch.ldexp(ones, half_exp)) * torch.ldexp(ones, (-exp) - half_exp)
    norm = mantissa.norm(dim=1, keepdim=True)
    valid = active & (norm > 0.0) & torch.isfinite(norm)
    clamped = norm.clamp_min(1e-30)
    return (
        torch.where(valid, mantissa / clamped, 0.0),
        torch.where(valid, exp.float() + torch.log2(clamped), 0.0),
        valid,
    )


@torch.no_grad()
def mantissa_exponent_normalize(grad: torch.Tensor):
    """Return ``(unit_tensor, log2_norm, is_valid)`` without underflow or overflow."""
    if grad.numel() == 0:
        return grad.float().clone(), grad.new_zeros((), dtype=_F32), grad.new_zeros((), dtype=_BOOL)
    unit, log2_norm, valid = me_unit_rows(grad.float().reshape(1, -1))
    return unit.reshape_as(grad), log2_norm.reshape(()), valid.reshape(())


@torch.no_grad()
def weyl_rademacher(shape, *, phase=0.0, salt=0, device=None, dtype=_F32):
    """Deterministic +/-1 Rademacher probe tensor with zero global RNG mutation."""
    dims = tuple(int(size) for size in shape)
    total = math.prod(dims) if dims else 1
    if total == 0:
        return torch.empty(dims, device=device, dtype=dtype)
    seed_word = (int(math.floor((float(phase) % 1.0) * 1073741824.0)) + int(salt) * 7919) & 0xFFFFFFFF
    hashed = (
        torch.arange(total, device=device, dtype=torch.int64) * 2654435761 + seed_word * 3266489917 + 374761393
    ) & 0xFFFFFFFF
    hashed = ((hashed ^ (hashed >> 15)) * 2246822519) & 0xFFFFFFFF
    hashed = ((hashed ^ (hashed >> 13)) * 3266489917) & 0xFFFFFFFF
    return torch.where(((hashed ^ (hashed >> 16)) & 1) == 0, 1.0, -1.0).to(dtype).reshape(dims)


def _weyl_unit_vector(dim: int, seed: int, device, dtype: torch.dtype) -> torch.Tensor:
    raw = torch.remainder(torch.arange(1, dim + 1, device=device, dtype=dtype) * PHI1 + (seed + 0.5) * PHI2, 1.0) - 0.5
    norm = raw.norm()
    fallback = torch.full((dim,), dim**-0.5, device=device, dtype=dtype)
    return torch.where(norm > 1e-8, raw / norm.clamp_min(1e-12), fallback)


@torch.no_grad()
def power_principal_axis(mat: torch.Tensor, *, iters=6, seed=0):
    """Deterministic Weyl-seeded power iteration returning ``(lambda_1, u_1)``."""
    work_dtype = _F64 if mat.dtype == _F64 else _F32
    sym = _symmetrize_square(mat, work_dtype)
    scale = sym.abs().amax(dim=(-2, -1), keepdim=True)
    active = scale > 0.0
    floor = 1e-300 if work_dtype == _F64 else 1e-38
    scaled = torch.where(active, sym / scale.clamp_min(floor), 0.0)
    vec = _weyl_unit_vector(mat.shape[-1], seed, mat.device, work_dtype).expand(*sym.shape[:-1]).clone()
    for _ in range(max(1, int(iters))):
        vec = _unit_or_fallback(torch.matmul(scaled, vec.unsqueeze(-1)).squeeze(-1), vec, -1, 1e-12)
    vec = torch.where(torch.gather(vec, -1, vec.abs().argmax(dim=-1, keepdim=True)) < 0.0, -vec, vec)
    rayleigh = (vec * torch.matmul(scaled, vec.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
    lam = rayleigh * scale.squeeze(-1).squeeze(-1)
    return torch.where(active.squeeze(-1).squeeze(-1), lam, 0.0).to(mat.dtype), vec.to(mat.dtype)


def _init_power_vector(init_v, batch_size: int, cols: int, unbatched: bool, device) -> torch.Tensor:
    fallback = _weyl_unit_vector(cols, 0, device, _F32)
    if init_v is None:
        return fallback.unsqueeze(0).expand(batch_size, cols).clone()
    vec = init_v.detach().to(device, _F32)
    reshaped = (vec.unsqueeze(0) if unbatched and vec.ndim == 1 else vec).reshape(batch_size, cols)
    return _unit_or_fallback(reshaped, fallback, 1, 1e-12)


@torch.no_grad()
def stable_rank_power(mat: torch.Tensor, *, iters=2, init_v=None, return_v=False):
    """Estimate matrix stable rank ``||A||_F^2 / ||A||_2^2`` via power iteration."""
    if mat.ndim not in (2, 3):
        raise ValueError("expected 2D or 3D tensor")
    unbatched = mat.ndim == 2
    batch = mat.unsqueeze(0) if unbatched else mat
    batch_size, rows, cols = batch.shape
    if batch.numel() == 0:
        empty_sr = mat.new_zeros(() if unbatched else (batch_size,), dtype=_F32)
        if not return_v:
            return empty_sr
        return empty_sr, mat.new_zeros((max(rows, cols),) if unbatched else (batch_size, max(rows, cols)), dtype=_F32)
    unit_flat, _, valid = me_unit_rows(batch.float().reshape(batch_size, rows * cols))
    unit_mat = unit_flat.view(batch_size, rows, cols)
    wide = unit_mat.transpose(1, 2) if rows > cols else unit_mat
    short_dim, long_dim = wide.shape[1], wide.shape[2]
    vec = _init_power_vector(init_v, batch_size, long_dim, unbatched, wide.device)
    for _ in range(max(1, int(iters))):
        u = torch.bmm(wide, vec.unsqueeze(-1)).squeeze(-1)
        u = _unit_or_fallback(u, u, 1, 1e-20)
        vec = _unit_or_fallback(torch.bmm(wide.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1), vec, 1, 1e-20)
    sigma1_sq = torch.bmm(wide, vec.unsqueeze(-1)).squeeze(-1).square().sum(dim=1)
    stable_rank = torch.where(
        valid.squeeze(1), (1.0 / sigma1_sq.clamp_min(1e-30)).clamp(1.0, float(short_dim)), float(short_dim)
    )
    if unbatched:
        stable_rank, vec = stable_rank.squeeze(0), vec.squeeze(0)
    return (stable_rank, vec) if return_v else stable_rank


@torch.no_grad()
def sym_eigensplit_halley(mat: torch.Tensor, *, steps=8):
    """Split a symmetric matrix into PSD and NSD parts via FP64 Pade-(1,1) Halley."""
    sym = _symmetrize_square(mat, _F64)
    dim = sym.shape[-1]
    eye = torch.eye(dim, dtype=_F64, device=sym.device)
    rms = (sym.square().sum(dim=(-2, -1), keepdim=True) / dim).sqrt()
    sign_mat = torch.where(rms > 0.0, sym / rms.clamp_min(1e-300), 0.0)
    for _ in range(max(1, int(steps))):
        sq = torch.matmul(sign_mat, sign_mat)
        chol, _ = torch.linalg.cholesky_ex(eye + 3.0 * sq, check_errors=False)
        sign_mat = torch.cholesky_solve(torch.matmul(sign_mat, 3.0 * eye + sq), chol)
        sign_mat = 0.5 * (sign_mat + sign_mat.transpose(-1, -2))
    abs_sym = 0.5 * (torch.matmul(sign_mat, sym) + torch.matmul(sym, sign_mat))
    return (0.5 * (sym + abs_sym)).to(mat.dtype), (0.5 * (sym - abs_sym)).to(mat.dtype)


@torch.no_grad()
def sym_inv_sqrt_ns(cov: torch.Tensor, *, eps=0.0, steps=20):
    """Compute symmetric ``(cov + eps I)^{-1/2}`` via FP64 coupled Newton-Schulz."""
    sym = _symmetrize_square(cov, _F64)
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("invalid eps")
    eye = torch.eye(sym.shape[-1], dtype=_F64, device=sym.device)
    if eps > 0.0:
        sym = sym + eps * eye
    scale = sym.abs().amax(dim=(-2, -1), keepdim=True)
    active = scale > 0.0
    normed = torch.where(active, sym / scale.clamp_min(1e-300), eye)
    spec_est = torch.matmul(normed, normed).square().sum(dim=(-2, -1), keepdim=True).pow(0.25) * 0.60
    y_mat, z_mat = normed / spec_est.clamp_min(1e-300), eye.expand_as(sym).clone()
    for _ in range(max(1, int(steps))):
        step_mat = 0.5 * (3.0 * eye - torch.matmul(z_mat, y_mat))
        y_mat, z_mat = torch.matmul(y_mat, step_mat), torch.matmul(step_mat, z_mat)
    inv_sqrt = 0.5 * (z_mat + z_mat.transpose(-1, -2)) * torch.rsqrt((spec_est * scale).clamp_min(1e-300))
    return torch.where(active, inv_sqrt, 0.0).to(cov.dtype)


def _add_ridge(mat: torch.Tensor, ridge) -> torch.Tensor:
    eye = torch.eye(mat.shape[-1], dtype=mat.dtype, device=mat.device)
    if isinstance(ridge, torch.Tensor):
        ridge_t = ridge.to(mat.device, mat.dtype)
        return mat + (torch.diag_embed(ridge_t) if ridge_t.ndim >= 1 else ridge_t * eye)
    return mat + float(ridge) * eye if ridge != 0.0 else mat


@torch.no_grad()
def solve_spd_cholesky(gram: torch.Tensor, rhs: torch.Tensor, *, ridge=0.0, return_inv=False):
    """Solve ``(gram + ridge) X = rhs`` via Cholesky with Halley PSD fallback."""
    mat = _add_ridge(_symmetrize_square(gram, _F64), ridge)
    dim = mat.shape[-1]
    chol, info = torch.linalg.cholesky_ex(mat, check_errors=False)
    if bool((info != 0).any()):
        pos_part, _ = sym_eigensplit_halley(mat)
        diag_scale = pos_part.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
        fallback_chol, _ = torch.linalg.cholesky_ex(
            pos_part + (1e-6 * diag_scale) * torch.eye(dim, dtype=_F64, device=mat.device), check_errors=False
        )
        chol = torch.where((info != 0).unsqueeze(-1).unsqueeze(-1), fallback_chol, chol)
    rhs_f64 = rhs.to(mat.device, _F64)
    is_vec = rhs_f64.ndim == mat.ndim - 1
    sol = torch.cholesky_solve(rhs_f64.unsqueeze(-1) if is_vec else rhs_f64, chol)
    out = (sol.squeeze(-1) if is_vec else sol).to(rhs.dtype)
    if not return_inv:
        return out, None
    inv = torch.cholesky_solve(torch.eye(dim, dtype=_F64, device=mat.device).expand_as(mat), chol)
    return out, (0.5 * (inv + inv.transpose(-1, -2))).to(gram.dtype)


def _ns_smooth_tensor(rel_smooth, device, batch_size: int):
    if isinstance(rel_smooth, torch.Tensor):
        return rel_smooth.to(device, _F32).reshape(batch_size, 1, 1).clamp_min(0.0), True
    scalar = max(0.0, float(rel_smooth))
    return torch.full((batch_size, 1, 1), scalar, device=device, dtype=_F32), scalar > 0.0


def _ns_prepare_augmented(wide: torch.Tensor, rel_smooth):
    batch_size, short_dim = wide.shape[0], wide.shape[1]
    smooth, has_smooth = _ns_smooth_tensor(rel_smooth, wide.device, batch_size)
    work_dtype = torch.bfloat16 if (wide.is_cuda and short_dim > 256) else _F32
    if not has_smooth:
        return (wide / 1.02).to(work_dtype), None
    if short_dim > 512:
        return (wide / 1.02).to(work_dtype), torch.rsqrt(1.0 + smooth.square())
    inv_aug = 1.0 / (torch.sqrt(1.0 + smooth.square()).clamp_min(1e-30) * 1.02)
    eye = (
        torch.eye(short_dim, device=wide.device, dtype=work_dtype).unsqueeze(0).expand(batch_size, short_dim, short_dim)
    )
    ridge_block = eye * ((smooth / math.sqrt(short_dim)) * inv_aug).to(work_dtype)
    return torch.cat(((wide * inv_aug).to(work_dtype), ridge_block), dim=2), None


@torch.no_grad()
def fp32_tikhonov_ns_batched(mats: torch.Tensor, steps=5, *, rel_smooth=0.0, pre_normalized=False):
    """Batched quintic Tikhonov Newton-Schulz polar orthogonalization (FP32 for m <= 256)."""
    if mats.ndim != 3:
        raise ValueError("expected 3D tensor")
    batch_size, rows, cols = mats.shape
    orig_dtype = mats.dtype
    if mats.numel() == 0:
        return torch.zeros_like(mats)
    if pre_normalized:
        unit, valid = mats.float(), None
    else:
        unit_flat, _, valid_flat = me_unit_rows(mats.float().reshape(batch_size, rows * cols))
        unit, valid = unit_flat.view(batch_size, rows, cols), valid_flat.view(batch_size, 1, 1)
    if min(rows, cols) == 1:
        smooth, _ = _ns_smooth_tensor(rel_smooth, mats.device, batch_size)
        scaled = unit * torch.rsqrt(1.0 + smooth.square())
        return (scaled if valid is None else torch.where(valid, scaled, 0.0)).to(orig_dtype)
    transposed = rows > cols
    wide = unit.transpose(1, 2) if transposed else unit
    short_dim, long_dim = wide.shape[1], wide.shape[2]
    z_mat, post_scale = _ns_prepare_augmented(wide, rel_smooth)
    for step_idx in range(max(0, int(steps))):
        a_coef, b_coef, c_coef = _QUINTIC_COEFFS[min(step_idx, len(_QUINTIC_COEFFS) - 1)]
        gram = torch.bmm(z_mat, z_mat.transpose(1, 2))
        z_mat = a_coef * z_mat + torch.bmm(b_coef * gram + c_coef * torch.bmm(gram, gram), z_mat)
    z_fp32 = z_mat.float()
    gain = (math.sqrt(short_dim) / _batch_frob_norm(z_fp32).clamp_min(1e-8)).clamp_max(1.15)
    out = z_fp32[:, :, :long_dim] * (gain if post_scale is None else gain * post_scale)
    out = out.transpose(1, 2) if transposed else out
    finite = torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    return torch.where(finite if valid is None else valid & finite, out, 0.0).to(orig_dtype)


@torch.no_grad()
def fp32_tikhonov_ns_2d(mat: torch.Tensor, steps=5, *, rel_smooth=0.0):
    """Single-matrix quintic Tikhonov Newton-Schulz polar orthogonalization."""
    if mat.ndim != 2:
        raise ValueError("expected 2D tensor")
    smooth = rel_smooth.reshape(1) if isinstance(rel_smooth, torch.Tensor) else rel_smooth
    return fp32_tikhonov_ns_batched(mat.unsqueeze(0), steps=steps, rel_smooth=smooth).squeeze(0)


@torch.no_grad()
def schatten6_decay_batched(weights: torch.Tensor, rates: torch.Tensor, *, lam=0.25, tau=2.0):
    """Exact-deadband Schatten-6 spectral ceiling decay above singular threshold ``tau``."""
    if weights.ndim != 3:
        raise ValueError("expected 3D tensor")
    batch_size, rows, cols = weights.shape
    if weights.numel() == 0:
        return torch.zeros_like(weights, dtype=_F32)
    transposed = rows > cols
    wide = weights.transpose(1, 2) if transposed else weights
    scaled_w = wide.float() / tau
    work_dtype = torch.bfloat16 if (weights.is_cuda and wide.shape[1] >= 64) else _F32
    work_w = scaled_w.to(work_dtype)
    gram1 = torch.bmm(work_w, work_w.transpose(1, 2))
    gram2 = torch.bmm(gram1, gram1)
    gram2_f32 = gram2.float()
    gram2_peak = gram2_f32.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    gram2_unit = gram2_f32 / gram2_peak
    gram4 = torch.bmm(gram2_unit.to(work_dtype), gram2_unit.to(work_dtype)).float()
    ratio = _batch_frob_norm(gram4) / _batch_frob_norm(gram2_unit).clamp_min(1e-12)
    gate = smoothstep01(((ratio * gram2_peak).clamp_min(0.0).pow(0.25) - 1.0) / 0.25)
    pow5 = torch.bmm(gram2, work_w).float()
    norm_ratio = (_batch_frob_norm(pow5) / _batch_frob_norm(scaled_w).clamp_min(1e-12)).clamp_min(1e-6)
    rate_vec = rates.to(weights.device, _F32).reshape(batch_size, 1, 1)
    eff_rate = torch.minimum((lam / tau) * rate_vec * gate, 0.25 / norm_ratio)
    decay = -pow5 * (eff_rate * tau)
    valid = torch.isfinite(decay).all(dim=(1, 2), keepdim=True) & torch.isfinite(eff_rate) & (gate > 0.0)
    decay = torch.where(valid, decay, 0.0)
    return decay.transpose(1, 2) if transposed else decay


def _solve_two_constraint_kkt(c0: float, c1: float, rho: float, d_norm_sq: float):
    tol = 1e-10 * max(1.0, math.sqrt(max(0.0, d_norm_sq)))
    best_dist, best_sol = d_norm_sq, None
    for ck, other, weights in ((c0, c1 - c0 * rho, (c0, 0.0)), (c1, c0 - c1 * rho, (0.0, c1))):
        if ck >= -tol and other <= tol and ck * ck < best_dist:
            best_sol, best_dist = (max(0.0, weights[0]), max(0.0, weights[1])), ck * ck
    det = 1.0 - rho * rho
    if det > 1e-14:
        m0, m1 = (c0 - rho * c1) / det, (c1 - rho * c0) / det
        dist_sq = m0 * m0 + m1 * m1 + 2.0 * m0 * m1 * rho
        feasible = m0 >= -tol and m1 >= -tol and (c0 - m0 - m1 * rho) <= tol and (c1 - m0 * rho - m1) <= tol
        if feasible and dist_sq < best_dist:
            best_sol = (max(0.0, m0), max(0.0, m1))
    return best_sol


def solve_nonascent_multipliers(ad, a_norm_sq, d_norm_sq, a01=0.0):
    """Solve exact zero-pinv KKT multipliers for at most two non-ascent constraints."""
    num_constraints = len(ad)
    if num_constraints != len(a_norm_sq) or num_constraints > 2:
        raise ValueError("at most two constraints supported")
    if num_constraints == 0:
        return (), False, False
    d2 = max(0.0, float(d_norm_sq))
    active = [(i, math.sqrt(float(n2)), float(v)) for i, (n2, v) in enumerate(zip(a_norm_sq, ad)) if float(n2) > 0.0]
    zeros = (0.0,) * num_constraints
    if not active:
        return zeros, False, False
    if len(active) == 1:
        idx, length, dot_val = active[0]
        proj_len = dot_val / length
        if proj_len <= 0.0:
            return zeros, False, False
        if proj_len * proj_len >= d2:
            return zeros, True, True
        alphas = [0.0] * num_constraints
        alphas[idx] = proj_len / length
        return tuple(alphas), False, True
    (_, len0, ad0), (_, len1, ad1) = active
    c0, c1 = ad0 / len0, ad1 / len1
    if c0 <= 0.0 and c1 <= 0.0:
        return (0.0, 0.0), False, False
    sol = _solve_two_constraint_kkt(c0, c1, max(-1.0, min(1.0, float(a01) / (len0 * len1))), d2)
    return ((0.0, 0.0), True, True) if sol is None else ((sol[0] / len0, sol[1] / len1), False, True)


def tensor_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute scalar inner product in FP32 (or FP64 if either input is FP64) with FP64 scalar output."""
    work_dtype = _F64 if (x.dtype == _F64 or y.dtype == _F64) else _F32
    xf = x if x.dtype == work_dtype else x.to(work_dtype)
    yf = y if y.dtype == work_dtype else y.to(work_dtype)
    return torch.dot(xf.reshape(-1), yf.reshape(-1)).double()


def stream_dot(xs, ys, zero: torch.Tensor) -> torch.Tensor:
    """Accumulate per-parameter inner products into an FP64 scalar without full-tensor FP64 casts."""
    acc = zero
    for xp, yp in zip(xs, ys):
        if xp is not None and yp is not None and xp.numel() > 0:
            acc = acc + tensor_dot(xp, yp)
    return acc


def _solve_kkt_tensors(d2, n0, a0, n1=None, a1=None, a01=None):
    if n1 is None:
        vals = torch.stack((n0, a0, d2)).tolist()
        return solve_nonascent_multipliers((vals[1],), (vals[0],), vals[2])
    vals = torch.stack((n0, n1, a0, a1, a01, d2)).tolist()
    return solve_nonascent_multipliers((vals[2], vals[3]), (vals[0], vals[1]), vals[5], vals[4])


def _project_dense(disp: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    num_constraints = normals.shape[0]
    if num_constraints > 2:
        raise ValueError("at most two constraints supported")
    if num_constraints == 0:
        return disp.clone()
    d64, a64 = disp.double(), normals.double()
    an2, ad, d2 = (a64 * a64).sum(dim=1), a64 @ d64, (d64 * d64).sum()
    alphas, is_zero, projected = (
        _solve_kkt_tensors(d2, an2[0], ad[0])
        if num_constraints == 1
        else _solve_kkt_tensors(d2, an2[0], ad[0], an2[1], ad[1], (a64[0] * a64[1]).sum())
    )
    if not projected:
        return disp.clone()
    if is_zero:
        return torch.zeros_like(disp)
    out = d64 - alphas[0] * a64[0]
    if num_constraints == 2 and alphas[1] != 0.0:
        out = out - alphas[1] * a64[1]
    return out.to(disp)


def apply_stream_projection(disps, normals, multipliers):
    """Subtract active constraint normal multiples per parameter without FP64 tensor promotion."""
    active = [(alpha, grp) for alpha, grp in zip(multipliers, normals) if alpha != 0.0]
    out = []
    for idx, disp in enumerate(disps):
        work_dtype = _F64 if disp.dtype == _F64 else _F32
        acc = disp.to(work_dtype).clone()
        for alpha, grp in active:
            normal = grp[idx]
            if normal is not None:
                acc.add_(normal if normal.dtype == work_dtype else normal.to(work_dtype), alpha=-alpha)
        out.append(acc.to(disp.dtype))
    return out


def _project_stream(disps, normals):
    num_constraints = len(normals)
    if num_constraints > 2:
        raise ValueError("at most two constraints supported")
    if num_constraints == 0 or not disps:
        return [disp.clone() for disp in disps]
    zero = torch.zeros((), dtype=_F64, device=disps[0].device)
    d2 = stream_dot(disps, disps, zero)
    n0 = stream_dot(normals[0], normals[0], zero)
    a0 = stream_dot(normals[0], disps, zero)
    if num_constraints == 1:
        alphas, is_zero, projected = _solve_kkt_tensors(d2, n0, a0)
    else:
        alphas, is_zero, projected = _solve_kkt_tensors(
            d2,
            n0,
            a0,
            stream_dot(normals[1], normals[1], zero),
            stream_dot(normals[1], disps, zero),
            stream_dot(normals[0], normals[1], zero),
        )
    if not projected:
        return [disp.clone() for disp in disps]
    return [torch.zeros_like(disp) for disp in disps] if is_zero else apply_stream_projection(disps, normals, alphas)


@torch.no_grad()
def zero_pinv_project_nonascent(displacement, normals):
    """Project a dense vector or parameter-list displacement onto the non-ascent cone."""
    if isinstance(displacement, torch.Tensor):
        if not isinstance(normals, torch.Tensor):
            raise TypeError("tensor normals required")
        return _project_dense(displacement, normals)
    return _project_stream(displacement, normals)


def safe_cdist(a: torch.Tensor, b=None, *, diag_inf=False):
    """Scale-invariant pairwise Euclidean distance matrix safe across extreme scales."""
    rows_a = a.shape[0]
    af = a.reshape(rows_a, -1).float()
    bf = af if b is None else b.reshape(b.shape[0], -1).float()
    if rows_a == 0 or bf.shape[0] == 0:
        return af.new_empty((rows_a, bf.shape[0]), dtype=_F32)
    scale = _abs_max_scale(af, bf)
    dist = torch.cdist(af / scale, bf / scale, p=2.0, compute_mode="donot_use_mm_for_euclid_dist") * scale
    if diag_inf and b is None:
        if dist.requires_grad:
            return dist.masked_fill(torch.eye(rows_a, device=dist.device, dtype=_BOOL), float("inf"))
        return dist.fill_diagonal_(float("inf"))
    return dist


def born_infeld_cost(x: torch.Tensor, y: torch.Tensor, ell, *, aligned=False, fp64=False):
    """Cancellation-free Born-Infeld transport cost ``sqrt(ell^2 + ||x-y||^2) - ell``."""
    work_dtype = _F64 if fp64 else _F32
    xf, yf = x.reshape(x.shape[0], -1).to(work_dtype), y.reshape(y.shape[0], -1).to(work_dtype)
    ell_t = ell.detach().to(xf.device, work_dtype) if isinstance(ell, torch.Tensor) else xf.new_tensor(float(ell))
    scale = _abs_max_scale(xf, yf, ell_t)
    xu, yu = xf / scale, yf / scale
    eu = (ell_t / scale).clamp_min(1e-12 if fp64 else 1e-7)
    if aligned:
        sq = (xu - yu).square().sum(dim=-1)
    else:
        raw_sq = xu.square().sum(dim=-1, keepdim=True) + yu.square().sum(dim=-1).unsqueeze(0) - 2.0 * (xu @ yu.T)
        sq = raw_sq + (raw_sq.detach().clamp_min(0.0) - raw_sq.detach())
    rad = torch.sqrt(eu.square() + sq)
    cost_unit = (rad - eu) + (sq / (rad + eu).clamp_min(1e-30) - (rad - eu)).detach()
    return (cost_unit * scale).to(x.dtype), (cost_unit.detach() / eu.clamp_min(1e-30)).to(_F32)


def _logsumexp_dual(log_kernel, dual, dim: int, target, buf):
    torch.add(log_kernel, dual.unsqueeze(1 - dim), out=buf)
    peak = buf.amax(dim=dim, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, 0.0)
    return target - (peak.squeeze(dim) + buf.sub_(peak).exp_().sum(dim=dim).clamp_min(1e-30).log_())


@torch.no_grad()
def log_sinkhorn(log_kernel: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor, *, iters=10):
    """Stabilized log-domain Sinkhorn scaling returning log transport plan."""
    buf, u_dual, v_dual = torch.empty_like(log_kernel), torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(max(1, int(iters))):
        v_dual = _logsumexp_dual(log_kernel, u_dual, 0, log_b, buf)
        u_dual = _logsumexp_dual(log_kernel, v_dual, 1, log_a, buf)
    return torch.add(log_kernel, u_dual.unsqueeze(1), out=buf).add_(v_dual.unsqueeze(0))


@torch.no_grad()
def sinkhorn_with_slack(c_hat, d_row, d_col, *, rho, tau, eps, iters=10, mask=None):
    """Unbalanced Sinkhorn plan augmented with row/column slack channels."""
    rows, cols = c_hat.shape[0], c_hat.shape[1]
    log_k = c_hat.new_zeros((rows + 1, cols + 1), dtype=_F32)
    active = -c_hat.float().clamp(0.0, 1e6) / eps
    log_k[:rows, :cols] = active.masked_fill(~mask, float("-inf")) if mask is not None else active
    log_k[:rows, cols] = (-0.5 * rho + 0.5 * tau * d_row.float()) / eps
    log_k[rows, :cols] = (-0.5 * rho - 0.5 * tau * d_col.float()) / eps
    log_r = log_k.new_full((rows + 1,), -math.log(max(1, rows)))
    log_c = log_k.new_full((cols + 1,), -math.log(max(1, cols)))
    log_r[rows] = log_c[cols] = math.log(2.0)
    return log_sinkhorn(log_k, log_r, log_c, iters=iters).exp_().to(c_hat.dtype)


def _normalize_with_weyl_order(weights: torch.Tensor):
    count, dev = weights.numel(), weights.device
    prob = weights.detach().double().clamp_min(0.0)
    total = prob.sum()
    prob = torch.where(total > 1e-12, prob / total.clamp_min(1e-12), 1.0 / count)
    weyl = torch.remainder((torch.arange(count, device=dev, dtype=_F64) + 0.5) * PHI1, 1.0)
    return prob, torch.argsort(weyl, descending=True, stable=True)


def _stable_rank(order: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return order[torch.argsort(key[order], descending=True, stable=True)]


def _distribute_remainder(quotas: torch.Tensor, order: torch.Tensor, total: int) -> torch.Tensor:
    rem = torch.tensor(int(total), dtype=_LONG, device=quotas.device) - quotas.sum()
    bonus = (torch.arange(quotas.numel(), device=quotas.device, dtype=_LONG) < rem.clamp_min(0)).to(_LONG)
    return quotas.scatter_add_(0, order, bonus)


@torch.no_grad()
def hamilton_quotas(total: int, weights: torch.Tensor, *, active_scores=None, min_one_when_feasible=True):
    """GPU-vectorized largest-remainder Hamilton integer quotas with Weyl tie-breaking."""
    count, dev = weights.numel(), weights.device
    if count == 0 or total <= 0:
        return torch.zeros(count, dtype=_LONG, device=dev)
    prob, order = _normalize_with_weyl_order(weights)
    if min_one_when_feasible and total >= count:
        rem = int(total) - count
        if rem == 0:
            return torch.ones(count, dtype=_LONG, device=dev)
        excess = (prob * total - 1.0).clamp_min(0.0)
        ex_sum = excess.sum()
        scaled = torch.where(ex_sum > 1e-12, excess * (rem / ex_sum.clamp_min(1e-12)), prob * rem)
        quotas = torch.ones(count, dtype=_LONG, device=dev) + torch.floor(scaled).long()
        frac = scaled - torch.floor(scaled)
    else:
        raw = prob * total
        quotas, frac = torch.floor(raw).long(), raw - torch.floor(raw)
    if active_scores is not None:
        order = _stable_rank(order, active_scores.detach().to(dev, _F64).reshape(count))
    return _distribute_remainder(quotas, _stable_rank(order, frac), total)


@torch.no_grad()
def allocate_deficit_quotas(proto_w: torch.Tensor, w_fake_ema: torch.Tensor, n_slots: int):
    """Allocate integer particle slots prioritizing starved prototypes by mass deficit."""
    count, dev = proto_w.numel(), proto_w.device
    if count == 0 or n_slots <= 0:
        return torch.zeros(count, device=dev, dtype=_LONG)
    prob, order = _normalize_with_weyl_order(proto_w)
    deficit = (proto_w.detach() - w_fake_ema.detach()).double()
    if n_slots < count:
        order = _stable_rank(_stable_rank(order, prob), deficit)
        mask = (torch.arange(count, device=dev, dtype=_LONG) < n_slots).to(_LONG)
        return torch.zeros(count, device=dev, dtype=_LONG).scatter_(0, order, mask)
    expected = prob * n_slots
    quotas = torch.floor(expected).to(_LONG)
    return _distribute_remainder(
        quotas, _stable_rank(_stable_rank(order, deficit), expected - quotas.double()), n_slots
    )


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
