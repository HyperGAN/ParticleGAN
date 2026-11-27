"""Axiom L0 Math Primitives: stateless zero-eigh/svd/pinv/autograd.grad tensor ops.

Pure L0 numerical layer shared by all Axiom Laws 1-4 modules. Enforces:
- zero internal ``_axiom`` or legacy sibling imports,
- zero ``torch.autograd.grad``, ``svd``, ``eigh``, ``pinv``, or ``torch.compile``,
- zero module-global tensor caches and zero global RNG mutation,
- strict clock-freedom and scale-equivariance across IEEE-754 regimes.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, overload

import torch

PHI1: float = 0.6180339887498949
PHI2: float = 0.7548776662466927

_QUINTIC_COEFFS: Tuple[Tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


# ---------------------------------------------------------------------------
# Constants & Smooth Gates
# ---------------------------------------------------------------------------


@overload
def smoothstep01(x: float) -> float: ...


@overload
def smoothstep01(x: torch.Tensor) -> torch.Tensor: ...


def smoothstep01(x: float | torch.Tensor) -> float | torch.Tensor:
    """C1 cubic Hermite gate mapping ``(-inf, 0] -> 0`` and ``[1, +inf) -> 1``."""
    if isinstance(x, torch.Tensor):
        u = x.clamp(0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)
    u = min(1.0, max(0.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


# ---------------------------------------------------------------------------
# IEEE-754 Scale-Free Normalization
# ---------------------------------------------------------------------------


@torch.no_grad()
def me_unit_rows(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """IEEE-754 mantissa-exponent row normalization without underflow down to subnormals.

    Returns ``(unit_rows [B, D], log2_norm [B, 1], valid [B, 1])`` in FP32/bool.
    """
    if x.ndim != 2:
        raise ValueError(
            f"me_unit_rows expects a 2D tensor, got shape {tuple(x.shape)}"
        )
    rows, cols = int(x.shape[0]), int(x.shape[1])
    if rows == 0 or cols == 0:
        return (
            x.new_zeros((rows, cols), dtype=torch.float32),
            x.new_zeros((rows, 1), dtype=torch.float32),
            x.new_zeros((rows, 1), dtype=torch.bool),
        )
    xf = x.float()
    fin = torch.isfinite(xf).all(dim=1, keepdim=True)
    xs = torch.where(fin, xf, torch.zeros_like(xf))
    gm = xs.abs().amax(dim=1, keepdim=True)
    active = fin & (gm > 0.0)
    one = torch.ones_like(gm)
    _, exp = torch.frexp(torch.where(active, gm, one))
    # Split -exp into two halves so ldexp never overflows/underflows even for subnormals.
    e1 = (-exp) // 2
    e2 = (-exp) - e1
    xm = (xs * torch.ldexp(one, e1)) * torch.ldexp(one, e2)
    nrm = xm.norm(dim=1, keepdim=True)
    ok = active & (nrm > 0.0) & torch.isfinite(nrm)
    unit = torch.where(ok, xm / nrm.clamp_min(1e-30), torch.zeros_like(xm))
    log2_norm = torch.where(
        ok,
        exp.float() + torch.log2(nrm.clamp_min(1e-30)),
        torch.zeros_like(gm),
    )
    return unit, log2_norm, ok


@torch.no_grad()
def mantissa_exponent_normalize(
    g: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact IEEE-754 power-of-2 mantissa-exponent normalization across ``[1e-44, 3e+38]``."""
    if g.numel() == 0:
        return (
            g.float().clone(),
            g.new_zeros((), dtype=torch.float32),
            g.new_zeros((), dtype=torch.bool),
        )
    unit, log2_norm, ok = me_unit_rows(g.float().reshape(1, -1))
    return unit.reshape_as(g), log2_norm.reshape(()), ok.reshape(())


# ---------------------------------------------------------------------------
# Deterministic Low-Discrepancy Probes
# ---------------------------------------------------------------------------


@torch.no_grad()
def weyl_rademacher(
    shape: Sequence[int],
    *,
    phase: float = 0.0,
    salt: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic ``{-1, +1}`` Rademacher probe tensor with zero global RNG mutation."""
    dims = tuple(int(s) for s in shape)
    numel = math.prod(dims) if dims else 1
    if numel == 0:
        return torch.empty(dims, device=device, dtype=dtype)
    seed_word = (
        int(math.floor((float(phase) % 1.0) * 1073741824.0)) + int(salt) * 7919
    ) & 0xFFFFFFFF
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    h = (idx * 2654435761 + seed_word * 3266489917 + 374761393) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    signs = torch.where(((h ^ (h >> 16)) & 1) == 0, 1.0, -1.0)
    return signs.to(dtype=dtype).reshape(dims)


# ---------------------------------------------------------------------------
# Zero-eigh / Zero-svd Linear Algebra
# ---------------------------------------------------------------------------


def _weyl_unit_vector(
    dim: int, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    idx = torch.arange(1, dim + 1, device=device, dtype=dtype)
    raw = torch.remainder(idx * PHI1 + (float(seed) + 0.5) * PHI2, 1.0) - 0.5
    nrm = raw.norm()
    fallback = torch.full((dim,), float(dim) ** -0.5, device=device, dtype=dtype)
    return torch.where(nrm > 1e-8, raw / nrm.clamp_min(1e-12), fallback)


@torch.no_grad()
def power_principal_axis(
    mat: torch.Tensor,
    *,
    iters: int = 6,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic Weyl-seeded power iteration returning ``(lambda_1, u_1)`` for symmetric ``[..., d, d]``."""
    if mat.ndim < 2 or mat.shape[-1] != mat.shape[-2] or mat.shape[-1] < 1:
        raise ValueError(
            f"power_principal_axis expects [..., d, d] with d>=1, got {tuple(mat.shape)}"
        )
    dim = int(mat.shape[-1])
    work_dt = torch.float64 if mat.dtype == torch.float64 else torch.float32
    m_work = mat.to(work_dt)
    sym = 0.5 * (m_work + m_work.transpose(-1, -2))
    scale = sym.abs().amax(dim=(-2, -1), keepdim=True)
    floor = 1e-300 if work_dt == torch.float64 else 1e-38
    active = scale > 0.0
    a_scaled = torch.where(active, sym / scale.clamp_min(floor), torch.zeros_like(sym))

    v0 = _weyl_unit_vector(dim, seed, mat.device, work_dt)
    u = v0.expand(*sym.shape[:-1]).clone()
    for _ in range(max(1, int(iters))):
        w = torch.matmul(a_scaled, u.unsqueeze(-1)).squeeze(-1)
        wn = w.norm(dim=-1, keepdim=True)
        u = torch.where(wn > 1e-12, w / wn.clamp_min(1e-12), u)

    piv_idx = u.abs().argmax(dim=-1, keepdim=True)
    piv = torch.gather(u, -1, piv_idx)
    u = torch.where(piv < 0.0, -u, u)
    au = torch.matmul(a_scaled, u.unsqueeze(-1)).squeeze(-1)
    lam1 = (u * au).sum(dim=-1) * scale.squeeze(-1).squeeze(-1)
    lam1 = torch.where(active.squeeze(-1).squeeze(-1), lam1, torch.zeros_like(lam1))
    return lam1.to(mat.dtype), u.to(mat.dtype)


def _init_power_vec(
    init_v: Optional[torch.Tensor],
    bs: int,
    nd: int,
    unbatched: bool,
    device: torch.device,
) -> torch.Tensor:
    fallback = _weyl_unit_vector(nd, 0, device, torch.float32)
    if init_v is None:
        return fallback.unsqueeze(0).expand(bs, nd).clone()
    v = init_v.detach().to(device=device, dtype=torch.float32)
    if unbatched and v.ndim == 1:
        v = v.unsqueeze(0)
    v = v.reshape(bs, nd)
    vn = v.norm(dim=1, keepdim=True)
    return torch.where(vn > 1e-12, v / vn.clamp_min(1e-12), fallback)


@torch.no_grad()
def stable_rank_power(
    mat: torch.Tensor,
    *,
    iters: int = 2,
    init_v: Optional[torch.Tensor] = None,
    return_v: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """2-step power-iteration stable rank ``||A||_F^2 / sigma_1(A)^2`` for 2D ``[r, c]`` or 3D ``[B, r, c]``."""
    if mat.ndim not in (2, 3):
        raise ValueError(
            f"stable_rank_power expects a 2D or 3D tensor, got shape {tuple(mat.shape)}"
        )
    unbatched = mat.ndim == 2
    batch = mat.unsqueeze(0) if unbatched else mat
    bs, rd, cd = int(batch.shape[0]), int(batch.shape[1]), int(batch.shape[2])
    if batch.numel() == 0:
        empty_r = mat.new_zeros(() if unbatched else (bs,), dtype=torch.float32)
        if not return_v:
            return empty_r
        v_shape = (max(rd, cd),) if unbatched else (bs, max(rd, cd))
        return empty_r, mat.new_zeros(v_shape, dtype=torch.float32)

    gu_flat, _, ok_flat = me_unit_rows(batch.float().reshape(bs, rd * cd))
    gu = gu_flat.view(bs, rd, cd)
    mc = gu.transpose(1, 2) if rd > cd else gu
    md, nd = int(mc.shape[1]), int(mc.shape[2])
    v = _init_power_vec(init_v, bs, nd, unbatched, mc.device)

    for _ in range(max(1, int(iters))):
        u = torch.bmm(mc, v.unsqueeze(-1)).squeeze(-1)
        un = u.norm(dim=1, keepdim=True)
        u = torch.where(un > 1e-20, u / un.clamp_min(1e-20), u)
        v_next = torch.bmm(mc.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
        vnn = v_next.norm(dim=1, keepdim=True)
        v = torch.where(vnn > 1e-20, v_next / vnn.clamp_min(1e-20), v)

    sig1_sq = torch.bmm(mc, v.unsqueeze(-1)).squeeze(-1).square().sum(dim=1)
    r_raw = (1.0 / sig1_sq.clamp_min(1e-30)).clamp(1.0, float(md))
    r_stab = torch.where(ok_flat.squeeze(1), r_raw, r_raw.new_full((bs,), float(md)))
    if unbatched:
        r_stab, v = r_stab.squeeze(0), v.squeeze(0)
    return (r_stab, v) if return_v else r_stab


@torch.no_grad()
def sym_eigensplit_halley(
    mat: torch.Tensor,
    *,
    steps: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split symmetric ``[..., d, d]`` into PSD ``E_+`` and NSD ``E_-`` via FP64 Pade-(1,1) Halley sign.

    Valid uniformly for all ``d >= 1`` (zero ``eigh``, zero ``d<=4`` split). Because ``I + 3 X^2 >= I``
    is strictly SPD for every finite symmetric ``X``, Cholesky factorization never fails.
    """
    if mat.ndim < 2 or mat.shape[-1] != mat.shape[-2] or mat.shape[-1] < 1:
        raise ValueError(
            f"sym_eigensplit_halley expects [..., d, d] with d>=1, got {tuple(mat.shape)}"
        )
    dim = int(mat.shape[-1])
    m64 = mat.to(torch.float64)
    sym = 0.5 * (m64 + m64.transpose(-1, -2))
    eye = torch.eye(dim, dtype=torch.float64, device=mat.device)
    s_rms = (sym.square().sum(dim=(-2, -1), keepdim=True) / float(dim)).sqrt()
    x = torch.where(s_rms > 0.0, sym / s_rms.clamp_min(1e-300), torch.zeros_like(sym))
    for _ in range(max(1, int(steps))):
        x2 = torch.matmul(x, x)
        chol, _ = torch.linalg.cholesky_ex(eye + 3.0 * x2, check_errors=False)
        x = torch.cholesky_solve(torch.matmul(x, 3.0 * eye + x2), chol)
        x = 0.5 * (x + x.transpose(-1, -2))
    abs_a = 0.5 * (torch.matmul(x, sym) + torch.matmul(sym, x))
    return (0.5 * (sym + abs_a)).to(mat.dtype), (0.5 * (sym - abs_a)).to(mat.dtype)


@torch.no_grad()
def sym_inv_sqrt_ns(
    cov: torch.Tensor,
    *,
    eps: float = 0.0,
    steps: int = 20,
) -> torch.Tensor:
    """FP64 coupled Newton-Schulz symmetric inverse square root ``A^{-1/2}`` for all ``d >= 1``."""
    if cov.ndim < 2 or cov.shape[-1] != cov.shape[-2] or cov.shape[-1] < 1:
        raise ValueError(
            f"sym_inv_sqrt_ns expects [..., d, d] with d>=1, got {tuple(cov.shape)}"
        )
    if eps < 0.0 or not math.isfinite(float(eps)):
        raise ValueError(f"sym_inv_sqrt_ns requires finite eps >= 0, got {eps!r}")
    dim = int(cov.shape[-1])
    c64 = cov.to(torch.float64)
    eye = torch.eye(dim, dtype=torch.float64, device=cov.device)
    a = 0.5 * (c64 + c64.transpose(-1, -2))
    if eps > 0.0:
        a = a + float(eps) * eye
    s0 = a.abs().amax(dim=(-2, -1), keepdim=True)
    active = s0 > 0.0
    a_norm = torch.where(active, a / s0.clamp_min(1e-300), eye)
    a2 = torch.matmul(a_norm, a_norm)
    sc_rel = a2.square().sum(dim=(-2, -1), keepdim=True).pow(0.25) * 0.60
    y = a_norm / sc_rel.clamp_min(1e-300)
    z = eye.expand_as(a).clone()
    for _ in range(max(1, int(steps))):
        t = 0.5 * (3.0 * eye - torch.matmul(z, y))
        y = torch.matmul(y, t)
        z = torch.matmul(t, z)
    inv_sqrt = (
        0.5 * (z + z.transpose(-1, -2)) * torch.rsqrt((sc_rel * s0).clamp_min(1e-300))
    )
    return torch.where(active, inv_sqrt, torch.zeros_like(inv_sqrt)).to(cov.dtype)


def _add_diagonal_ridge(mat: torch.Tensor, ridge: float | torch.Tensor) -> torch.Tensor:
    dim = int(mat.shape[-1])
    eye = torch.eye(dim, dtype=mat.dtype, device=mat.device)
    if isinstance(ridge, torch.Tensor):
        r64 = ridge.to(device=mat.device, dtype=mat.dtype)
        return mat + (torch.diag_embed(r64) if r64.ndim >= 1 else r64 * eye)
    return mat + float(ridge) * eye if float(ridge) != 0.0 else mat


@torch.no_grad()
def solve_spd_cholesky(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    *,
    ridge: float | torch.Tensor = 0.0,
    return_inv: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Solve SPD system ``(gram + diag(ridge)) X = rhs`` in FP64 via Cholesky with Halley-PSD fallback."""
    if gram.ndim < 2 or gram.shape[-1] != gram.shape[-2] or gram.shape[-1] < 1:
        raise ValueError(
            f"solve_spd_cholesky expects [..., d, d] with d>=1, got {tuple(gram.shape)}"
        )
    dim = int(gram.shape[-1])
    g64 = gram.to(torch.float64)
    am = _add_diagonal_ridge(0.5 * (g64 + g64.transpose(-1, -2)), ridge)
    chol, info = torch.linalg.cholesky_ex(am, check_errors=False)
    if bool((info != 0).any()):
        e_pos, _ = sym_eigensplit_halley(am)
        diag_sc = (
            e_pos.diagonal(dim1=-2, dim2=-1)
            .mean(dim=-1, keepdim=True)
            .unsqueeze(-1)
            .clamp_min(1.0)
        )
        am_safe = e_pos + (1e-6 * diag_sc) * torch.eye(
            dim, dtype=torch.float64, device=am.device
        )
        chol_fb, _ = torch.linalg.cholesky_ex(am_safe, check_errors=False)
        chol = torch.where((info != 0).unsqueeze(-1).unsqueeze(-1), chol_fb, chol)
    rhs64 = rhs.to(device=am.device, dtype=torch.float64)
    is_vec = rhs64.ndim == am.ndim - 1
    sol = torch.cholesky_solve(rhs64.unsqueeze(-1) if is_vec else rhs64, chol)
    sol_out = (sol.squeeze(-1) if is_vec else sol).to(rhs.dtype)
    if not return_inv:
        return sol_out, None
    eye = torch.eye(dim, dtype=torch.float64, device=am.device).expand_as(am)
    inv = torch.cholesky_solve(eye, chol)
    return sol_out, (0.5 * (inv + inv.transpose(-1, -2))).to(gram.dtype)


# ---------------------------------------------------------------------------
# Spectral Conditioning & Exact Zero-pinv KKT Projection
# ---------------------------------------------------------------------------


def _quintic_ns_iterate(z: torch.Tensor, steps: int) -> torch.Tensor:
    n_coeff = len(_QUINTIC_COEFFS)
    for idx in range(max(0, int(steps))):
        a, b, c = _QUINTIC_COEFFS[min(idx, n_coeff - 1)]
        gram = torch.bmm(z, z.transpose(1, 2))
        z = a * z + torch.bmm(b * gram + c * torch.bmm(gram, gram), z)
    return z


def _resolve_tikhonov_smooth(
    rel_smooth: float | torch.Tensor,
    bs: int,
    device: torch.device,
) -> Tuple[torch.Tensor, bool]:
    if isinstance(rel_smooth, torch.Tensor):
        sr = rel_smooth.to(device=device, dtype=torch.float32).reshape(bs, 1, 1)
        return sr.clamp_min(0.0), True
    sr_val = max(0.0, float(rel_smooth))
    return torch.full(
        (bs, 1, 1), sr_val, device=device, dtype=torch.float32
    ), sr_val > 0.0


def _build_tikhonov_ns_input(
    xi: torch.Tensor,
    sr: torch.Tensor,
    has_smooth: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bs, md = int(xi.shape[0]), int(xi.shape[1])
    work_dt = torch.bfloat16 if (xi.is_cuda and md > 256) else torch.float32
    if not has_smooth:
        return (xi / 1.02).to(work_dt), None
    if md > 512:
        return (xi / 1.02).to(work_dt), torch.rsqrt(1.0 + sr.square())
    inv_aug = 1.0 / (torch.sqrt(1.0 + sr.square()).clamp_min(1e-30) * 1.02)
    eye_b = (
        torch.eye(md, device=xi.device, dtype=work_dt).unsqueeze(0).expand(bs, md, md)
    )
    rhs_block = eye_b * ((sr / math.sqrt(float(md))) * inv_aug).to(work_dt)
    return torch.cat(((xi * inv_aug).to(work_dt), rhs_block), dim=2), None


@torch.no_grad()
def fp32_tikhonov_ns_batched(
    mats: torch.Tensor,
    steps: int = 5,
    *,
    rel_smooth: float | torch.Tensor = 0.0,
    pre_normalized: bool = False,
) -> torch.Tensor:
    """Batched zero-SVD Tikhonov Quintic Newton-Schulz (FP32 for ``m<=256``, BF16 only ``m>256`` CUDA)."""
    if mats.ndim != 3:
        raise ValueError(
            f"fp32_tikhonov_ns_batched expects a 3D batch, got shape {tuple(mats.shape)}"
        )
    bs, rd, cd = int(mats.shape[0]), int(mats.shape[1]), int(mats.shape[2])
    orig_dtype = mats.dtype
    if mats.numel() == 0:
        return torch.zeros_like(mats)

    if pre_normalized:
        xu, ok = mats.float(), None
    else:
        xu_flat, _, ok_flat = me_unit_rows(mats.float().reshape(bs, rd * cd))
        xu, ok = xu_flat.view(bs, rd, cd), ok_flat.view(bs, 1, 1)

    sr, has_smooth = _resolve_tikhonov_smooth(rel_smooth, bs, mats.device)
    if min(rd, cd) == 1:
        res = xu * torch.rsqrt(1.0 + sr.square()) if has_smooth else xu
        return (res if ok is None else torch.where(ok, res, torch.zeros_like(res))).to(
            orig_dtype
        )

    transposed = rd > cd
    xi = xu.transpose(1, 2) if transposed else xu
    md, nd = int(xi.shape[1]), int(xi.shape[2])
    z, post_scale = _build_tikhonov_ns_input(xi, sr, has_smooth)

    zf = _quintic_ns_iterate(z, steps).float()
    zf_norm = zf.flatten(1).norm(dim=1).view(bs, 1, 1).clamp_min(1e-8)
    sc = (math.sqrt(float(md)) / zf_norm).clamp_max(1.15)
    if post_scale is not None:
        sc = sc * post_scale
    out = zf[:, :, :nd] * sc
    if transposed:
        out = out.transpose(1, 2)
    valid = torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    if ok is not None:
        valid = ok & valid
    return torch.where(valid, out, torch.zeros_like(out)).to(orig_dtype)


@torch.no_grad()
def fp32_tikhonov_ns_2d(
    mat: torch.Tensor,
    steps: int = 5,
    *,
    rel_smooth: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Single-matrix wrapper around :func:`fp32_tikhonov_ns_batched`."""
    if mat.ndim != 2:
        raise ValueError(
            f"fp32_tikhonov_ns_2d expects a 2D matrix, got shape {tuple(mat.shape)}"
        )
    smooth_arg = (
        rel_smooth.reshape(1) if isinstance(rel_smooth, torch.Tensor) else rel_smooth
    )
    return fp32_tikhonov_ns_batched(
        mat.unsqueeze(0), steps=steps, rel_smooth=smooth_arg
    ).squeeze(0)


@torch.no_grad()
def schatten6_decay_batched(
    weights: torch.Tensor,
    rates: torch.Tensor,
    *,
    lam: float = 0.25,
    tau: float = 2.0,
) -> torch.Tensor:
    """Zero-SVD, zero-sync, exact-deadband Schatten-6 spectral ceiling (``tau=2.0``) on ``[B, r, c]``."""
    if weights.ndim != 3:
        raise ValueError(
            f"schatten6_decay_batched expects a 3D batch, got shape {tuple(weights.shape)}"
        )
    bs, rd, cd = int(weights.shape[0]), int(weights.shape[1]), int(weights.shape[2])
    if bs == 0 or rd == 0 or cd == 0:
        return torch.zeros_like(weights, dtype=torch.float32)

    transposed = rd > cd
    w_work = weights.transpose(1, 2) if transposed else weights
    md = int(w_work.shape[1])
    wh = w_work.float() / float(tau)
    w_dt = torch.bfloat16 if (weights.is_cuda and md >= 64) else torch.float32
    wb = wh.to(w_dt)

    g1 = torch.bmm(wb, wb.transpose(1, 2))
    g2 = torch.bmm(g1, g1)
    g2f = g2.float()
    sg2 = g2f.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    g2s = g2f / sg2
    g4u = torch.bmm(g2s.to(w_dt), g2s.to(w_dt)).float()

    ratio_g = g4u.flatten(1).norm(dim=1).view(bs, 1, 1) / g2s.flatten(1).norm(
        dim=1
    ).view(bs, 1, 1).clamp_min(1e-12)
    rho = (ratio_g * sg2).clamp_min(0.0).pow(0.25)
    gate = smoothstep01((rho - 1.0) / 0.25)

    w5 = torch.bmm(g2, wb).float()
    norm_ratio = (
        w5.flatten(1).norm(dim=1).view(bs, 1, 1)
        / wh.flatten(1).norm(dim=1).view(bs, 1, 1).clamp_min(1e-12)
    ).clamp_min(1e-6)
    rates_v = rates.to(device=weights.device, dtype=torch.float32).reshape(bs, 1, 1)
    eff = torch.minimum((float(lam) / float(tau)) * rates_v * gate, 0.25 / norm_ratio)
    out = -w5 * (eff * float(tau))
    valid = (
        torch.isfinite(out).all(dim=(1, 2), keepdim=True)
        & torch.isfinite(eff)
        & (gate > 0.0)
    )
    out = torch.where(valid, out, torch.zeros_like(out))
    return out.transpose(1, 2) if transposed else out


def _solve_nonascent_2d(
    c0: float,
    c1: float,
    rho: float,
    d2: float,
) -> Optional[Tuple[float, float]]:
    """Exact 2-constraint KKT active-set solver in unit-normal coordinates."""
    tol = 1e-10 * max(1.0, math.sqrt(max(0.0, d2)))
    best_dist = d2
    best: Optional[Tuple[float, float]] = None

    for ck, other, weights in (
        (c0, c1 - c0 * rho, (c0, 0.0)),
        (c1, c0 - c1 * rho, (0.0, c1)),
    ):
        dist = ck * ck
        if ck >= -tol and other <= tol and dist < best_dist:
            best = (max(0.0, weights[0]), max(0.0, weights[1]))
            best_dist = dist

    det = 1.0 - rho * rho
    if det > 1e-14:
        m0 = (c0 - rho * c1) / det
        m1 = (c1 - rho * c0) / det
        r0 = c0 - m0 - m1 * rho
        r1 = c1 - m0 * rho - m1
        dist_both = m0 * m0 + m1 * m1 + 2.0 * m0 * m1 * rho
        if (
            m0 >= -tol
            and m1 >= -tol
            and r0 <= tol
            and r1 <= tol
            and dist_both < best_dist
        ):
            best = (max(0.0, m0), max(0.0, m1))
    return best


def _solve_nonascent_1d(
    idx: int,
    ell: float,
    dot: float,
    d2: float,
    nc: int,
) -> Tuple[Tuple[float, ...], bool, bool]:
    zeros = (0.0,) * nc
    c_val = dot / ell
    if c_val <= 0.0:
        return zeros, False, False
    if c_val * c_val >= d2:
        return zeros, True, True
    alphas = [0.0] * nc
    alphas[idx] = c_val / ell
    return tuple(alphas), False, True


def solve_nonascent_multipliers(
    ad: Sequence[float],
    a_norm_sq: Sequence[float],
    d_norm_sq: float,
    a01: float = 0.0,
) -> Tuple[Tuple[float, ...], bool, bool]:
    """Exact zero-pinv KKT multipliers for ``min ||x - d||^2`` s.t. ``a_i . x <= 0`` (``K <= 2``).

    Returns ``(alphas, is_zero, projected)`` where ``x* = 0`` if ``is_zero`` else
    ``x* = d - sum_k alphas[k] * a_k``.
    """
    nc = len(ad)
    if nc != len(a_norm_sq) or nc > 2:
        raise ValueError(
            "Axiom direction_blend requires equal-length inputs with at most two constraints"
        )
    if nc == 0:
        return (), False, False

    d2 = max(0.0, float(d_norm_sq))
    active = []
    for idx, (n2, dot) in enumerate(zip(a_norm_sq, ad)):
        if float(n2) > 0.0:
            active.append((idx, math.sqrt(float(n2)), float(dot)))
    if not active:
        return (0.0,) * nc, False, False
    if len(active) == 1:
        idx, ell, dot = active[0]
        return _solve_nonascent_1d(idx, ell, dot, d2, nc)

    (_, l0, ad0), (_, l1, ad1) = active[0], active[1]
    c0, c1 = ad0 / l0, ad1 / l1
    if c0 <= 0.0 and c1 <= 0.0:
        return (0.0, 0.0), False, False
    rho = max(-1.0, min(1.0, float(a01) / (l0 * l1)))
    sol = _solve_nonascent_2d(c0, c1, rho, d2)
    if sol is None:
        return (0.0, 0.0), True, True
    return (sol[0] / l0, sol[1] / l1), False, True


def _project_dense_nonascent(
    displacement: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    nc = int(normals.shape[0])
    if nc > 2:
        raise ValueError("Axiom direction_blend supports at most two protected losses")
    if nc == 0:
        return displacement.clone()
    d64 = displacement.double()
    a64 = normals.double()
    a_n2 = (a64 * a64).sum(dim=1)
    ad_t = a64 @ d64
    d2_t = (d64 * d64).sum()
    if nc == 1:
        n2_0, ad_0, d2_val = torch.stack((a_n2[0], ad_t[0], d2_t)).tolist()
        alphas, is_zero, projected = solve_nonascent_multipliers(
            (ad_0,), (n2_0,), d2_val
        )
    else:
        a01_t = (a64[0] * a64[1]).sum()
        packed = torch.stack((a_n2[0], a_n2[1], ad_t[0], ad_t[1], a01_t, d2_t))
        n2_0, n2_1, ad_0, ad_1, a01_val, d2_val = packed.tolist()
        alphas, is_zero, projected = solve_nonascent_multipliers(
            (ad_0, ad_1), (n2_0, n2_1), d2_val, a01_val
        )
    if not projected:
        return displacement.clone()
    if is_zero:
        return torch.zeros_like(displacement)
    out = d64 - alphas[0] * a64[0]
    if nc == 2 and alphas[1] != 0.0:
        out = out - alphas[1] * a64[1]
    return out.to(displacement)


def _param_dot(
    xs: Sequence[Optional[torch.Tensor]],
    ys: Sequence[Optional[torch.Tensor]],
    z0: torch.Tensor,
) -> torch.Tensor:
    acc = z0
    for x_p, y_p in zip(xs, ys):
        if x_p is not None and y_p is not None:
            acc = acc + (x_p.double() * y_p.double()).sum()
    return acc


def _streaming_dot_scalars(
    disps: Sequence[torch.Tensor],
    normals: Sequence[Sequence[Optional[torch.Tensor]]],
) -> Tuple[List[float], List[float], float, float]:
    z0 = torch.zeros((), dtype=torch.float64, device=disps[0].device)
    d2_t = _param_dot(disps, disps, z0)
    n2_0 = _param_dot(normals[0], normals[0], z0)
    ad_0 = _param_dot(normals[0], disps, z0)
    if len(normals) == 1:
        n2_v, ad_v, d2_v = torch.stack((n2_0, ad_0, d2_t)).tolist()
        return [ad_v], [n2_v], d2_v, 0.0
    n2_1 = _param_dot(normals[1], normals[1], z0)
    ad_1 = _param_dot(normals[1], disps, z0)
    a01_t = _param_dot(normals[0], normals[1], z0)
    vals = torch.stack((n2_0, n2_1, ad_0, ad_1, a01_t, d2_t)).tolist()
    return [vals[2], vals[3]], [vals[0], vals[1]], vals[5], vals[4]


def _apply_streaming_multipliers(
    disps: Sequence[torch.Tensor],
    normals: Sequence[Sequence[Optional[torch.Tensor]]],
    alphas: Tuple[float, ...],
) -> List[torch.Tensor]:
    active = [(alpha, grp) for alpha, grp in zip(alphas, normals) if alpha != 0.0]
    out: List[torch.Tensor] = []
    for idx, d_p in enumerate(disps):
        acc = d_p.double()
        for alpha, grp in active:
            a_p = grp[idx]
            if a_p is not None:
                acc = acc - alpha * a_p.double()
        out.append(acc.to(d_p))
    return out


def _project_streaming_nonascent(
    disps: Sequence[torch.Tensor],
    normals: Sequence[Sequence[Optional[torch.Tensor]]],
) -> List[torch.Tensor]:
    nc = len(normals)
    if nc > 2:
        raise ValueError("Axiom direction_blend supports at most two protected losses")
    if nc == 0 or not disps:
        return [d.clone() for d in disps]
    ad, a_n2, d2_val, a01_val = _streaming_dot_scalars(disps, normals)
    alphas, is_zero, projected = solve_nonascent_multipliers(ad, a_n2, d2_val, a01_val)
    if not projected:
        return [d.clone() for d in disps]
    if is_zero:
        return [torch.zeros_like(d) for d in disps]
    return _apply_streaming_multipliers(disps, normals, alphas)


@overload
def zero_pinv_project_nonascent(
    displacement: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor: ...


@overload
def zero_pinv_project_nonascent(
    displacement: Sequence[torch.Tensor],
    normals: Sequence[Sequence[Optional[torch.Tensor]]],
) -> List[torch.Tensor]: ...


@torch.no_grad()
def zero_pinv_project_nonascent(
    displacement: torch.Tensor | Sequence[torch.Tensor],
    normals: torch.Tensor | Sequence[Sequence[Optional[torch.Tensor]]],
) -> torch.Tensor | List[torch.Tensor]:
    """Exact nearest displacement satisfying ``a_i . d <= 0`` for 1 or 2 constraints (zero ``pinv``).

    Accepts either dense tensors ``(d [N], normals [K, N])`` or streaming per-parameter sequences
    ``(displacements, normals_per_constraint)`` without full-model concatenation.
    """
    if isinstance(displacement, torch.Tensor):
        if not isinstance(normals, torch.Tensor):
            raise TypeError("dense displacement requires a tensor normals argument")
        return _project_dense_nonascent(displacement, normals)
    return _project_streaming_nonascent(displacement, normals)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Transport & Quota Primitives
# ---------------------------------------------------------------------------


def safe_cdist(
    a: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    *,
    diag_inf: bool = False,
) -> torch.Tensor:
    """Overflow/underflow-safe Euclidean pairwise distances with optional ``+inf`` self-diagonal."""
    n_a = int(a.shape[0])
    af = a.reshape(n_a, -1).float()
    bf = af if b is None else b.reshape(int(b.shape[0]), -1).float()
    if n_a == 0 or bf.shape[0] == 0:
        return af.new_empty((n_a, int(bf.shape[0])), dtype=torch.float32)
    amax = torch.maximum(af.detach().abs().amax(), bf.detach().abs().amax())
    sc = torch.where(amax > 0.0, amax.clamp_min(1e-30), amax.new_ones(()))
    dist = (
        torch.cdist(
            af / sc, bf / sc, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
        )
        * sc
    )
    if diag_inf and b is None:
        if dist.requires_grad:
            return dist.masked_fill(
                torch.eye(n_a, device=dist.device, dtype=torch.bool), float("inf")
            )
        return dist.fill_diagonal_(float("inf"))
    return dist


def born_infeld_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    ell: float | torch.Tensor,
    *,
    aligned: bool = False,
    fp64: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cancellation-free Born-Infeld transport cost ``sqrt(ell^2 + ||x-y||^2) - ell`` and ratio ``cost / ell``."""
    work_dt = torch.float64 if fp64 else torch.float32
    xf = x.reshape(int(x.shape[0]), -1).to(work_dt)
    yf = y.reshape(int(y.shape[0]), -1).to(work_dt)
    ell_t = (
        ell.detach().to(device=xf.device, dtype=work_dt)
        if isinstance(ell, torch.Tensor)
        else xf.new_tensor(float(ell))
    )

    amax = torch.maximum(
        torch.maximum(xf.detach().abs().amax(), yf.detach().abs().amax()),
        ell_t.abs().amax(),
    )
    sc = torch.where(amax > 0.0, amax.clamp_min(1e-30), amax.new_ones(()))
    min_eu = 1e-12 if fp64 else 1e-7
    xu, yu = xf / sc, yf / sc
    eu = (ell_t / sc).clamp_min(min_eu)

    if aligned:
        sq = (xu - yu).square().sum(dim=-1)
    else:
        sq_raw = (
            xu.square().sum(dim=-1, keepdim=True)
            + yu.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * (xu @ yu.T)
        )
        sq = sq_raw + (sq_raw.detach().clamp_min(0.0) - sq_raw.detach())

    rad = torch.sqrt(eu.square() + sq)
    cu_exact = sq / (rad + eu).clamp_min(1e-30)
    cu = (rad - eu) + (cu_exact - (rad - eu)).detach()
    return (cu * sc).to(x.dtype), (cu.detach() / eu.clamp_min(1e-30)).to(torch.float32)


@torch.no_grad()
def log_sinkhorn(
    log_kernel: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    *,
    iters: int = 10,
) -> torch.Tensor:
    """Stabilized log-domain Sinkhorn-Knopp scaling returning the log transport plan."""
    buf = torch.empty_like(log_kernel)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(max(1, int(iters))):
        torch.add(log_kernel, u.unsqueeze(1), out=buf)
        m0 = buf.amax(dim=0, keepdim=True)
        m0 = torch.where(torch.isfinite(m0), m0, torch.zeros_like(m0))
        v = log_b - (
            m0.squeeze(0) + buf.sub_(m0).exp_().sum(dim=0).clamp_min(1e-30).log_()
        )
        torch.add(log_kernel, v.unsqueeze(0), out=buf)
        m1 = buf.amax(dim=1, keepdim=True)
        m1 = torch.where(torch.isfinite(m1), m1, torch.zeros_like(m1))
        u = log_a - (
            m1.squeeze(1) + buf.sub_(m1).exp_().sum(dim=1).clamp_min(1e-30).log_()
        )
    return torch.add(log_kernel, u.unsqueeze(1), out=buf).add_(v.unsqueeze(0))


@torch.no_grad()
def sinkhorn_with_slack(
    c_hat: torch.Tensor,
    d_row: torch.Tensor,
    d_col: torch.Tensor,
    *,
    rho: float,
    tau: float,
    eps: float,
    iters: int = 10,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Semi-relaxed Sinkhorn plan with an augmented slack row and column ``(B_r + 1, B_c + 1)``."""
    br, bc = int(c_hat.shape[0]), int(c_hat.shape[1])
    log_k = c_hat.new_zeros((br + 1, bc + 1), dtype=torch.float32)
    active = -c_hat.float().clamp(0.0, 1e6) / float(eps)
    log_k[:br, :bc] = (
        active.masked_fill(~mask, float("-inf")) if mask is not None else active
    )
    log_k[:br, bc] = (-0.5 * float(rho) + 0.5 * float(tau) * d_row.float()) / float(eps)
    log_k[br, :bc] = (-0.5 * float(rho) - 0.5 * float(tau) * d_col.float()) / float(eps)

    log_r = log_k.new_full((br + 1,), -math.log(max(1, br)))
    log_c = log_k.new_full((bc + 1,), -math.log(max(1, bc)))
    log_r[br] = math.log(2.0)
    log_c[bc] = math.log(2.0)
    return log_sinkhorn(log_k, log_r, log_c, iters=iters).exp_().to(c_hat.dtype)


@torch.no_grad()
def hamilton_quotas(
    total: int,
    weights: torch.Tensor,
    *,
    active_scores: Optional[torch.Tensor] = None,
    min_one_when_feasible: bool = True,
) -> torch.Tensor:
    """Zero-sync GPU-vectorized Hamilton largest-remainder quotas with Weyl tie-breaking.

    Guarantees ``quotas.sum() == total`` and ``quotas >= 1`` whenever ``min_one_when_feasible``
    is True and ``total >= len(weights)``.
    """
    k = int(weights.numel())
    dev = weights.device
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=dev)
    if total <= 0:
        return torch.zeros(k, dtype=torch.long, device=dev)

    w = weights.detach().double().clamp_min(0.0)
    ws = w.sum()
    w = torch.where(
        ws > 1e-12, w / ws.clamp_min(1e-12), w.new_full((k,), 1.0 / float(k))
    )
    weyl = torch.remainder(
        (torch.arange(k, device=dev, dtype=torch.float64) + 0.5) * PHI1, 1.0
    )

    if min_one_when_feasible and total >= k:
        rem = int(total) - k
        if rem == 0:
            return torch.ones(k, dtype=torch.long, device=dev)
        excess = (w * float(total) - 1.0).clamp_min(0.0)
        es = excess.sum()
        scaled = torch.where(
            es > 1e-12, excess * (float(rem) / es.clamp_min(1e-12)), w * float(rem)
        )
        q = torch.ones(k, dtype=torch.long, device=dev) + torch.floor(scaled).long()
        frac = scaled - torch.floor(scaled)
    else:
        raw = w * float(total)
        q = torch.floor(raw).long()
        frac = raw - torch.floor(raw)

    order = torch.argsort(weyl, descending=True, stable=True)
    if active_scores is not None:
        scores = active_scores.detach().to(device=dev, dtype=torch.float64).reshape(k)
        order = order[torch.argsort(scores[order], descending=True, stable=True)]
    order = order[torch.argsort(frac[order], descending=True, stable=True)]

    rem2 = torch.tensor(int(total), dtype=torch.long, device=dev) - q.sum()
    add = (torch.arange(k, device=dev, dtype=torch.long) < rem2.clamp_min(0)).to(
        torch.long
    )
    q.scatter_add_(0, order, add)
    return q


@torch.no_grad()
def allocate_deficit_quotas(
    proto_w: torch.Tensor,
    w_fake_ema: torch.Tensor,
    n_slots: int,
) -> torch.Tensor:
    """Zero-sync unbiased slot allocation via descending ``(proto_w - w_fake_ema, proto_w)`` + Weyl tie-break."""
    k = int(proto_w.numel())
    dev = proto_w.device
    if k == 0 or n_slots <= 0:
        return torch.zeros(k, device=dev, dtype=torch.long)

    w = proto_w.detach().double().clamp_min(0.0)
    ws = w.sum()
    w = torch.where(
        ws > 1e-12, w / ws.clamp_min(1e-12), w.new_full((k,), 1.0 / float(k))
    )
    deficit = (proto_w.detach() - w_fake_ema.detach()).double()
    weyl = torch.remainder(
        (torch.arange(k, device=dev, dtype=torch.float64) + 0.5) * PHI1, 1.0
    )
    order = torch.argsort(weyl, descending=True, stable=True)
    pos = torch.arange(k, device=dev, dtype=torch.long)

    if n_slots < k:
        order = order[torch.argsort(w[order], descending=True, stable=True)]
        order = order[torch.argsort(deficit[order], descending=True, stable=True)]
        q = torch.zeros(k, device=dev, dtype=torch.long)
        q.scatter_(0, order, (pos < int(n_slots)).to(torch.long))
        return q

    exact = w * float(n_slots)
    q = torch.floor(exact).to(torch.long)
    rem = torch.tensor(int(n_slots), device=dev, dtype=torch.long) - q.sum()
    order = order[torch.argsort(deficit[order], descending=True, stable=True)]
    order = order[
        torch.argsort((exact - q.double())[order], descending=True, stable=True)
    ]
    q.scatter_add_(0, order, (pos < rem.clamp_min(0)).to(torch.long))
    return q


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
