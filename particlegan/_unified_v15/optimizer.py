"""Unified-v15 Energy-Optimal 4-Law Equilibrated Optimizer (`unified_v15_equil`).

Closes the Law-4 mathematical defects identified in the Four-Laws audit while preserving
zero-SVD, zero-`pinv`, zero-`autograd.grad`, zero-extra-sync GPU efficiency:
1. **D-02 Exact FP32 Quintic Polar (`m <= 256`) & Rank-1 Norm Cap:**
   - Runs both Tikhonov-smoothed and unsmoothed (`s=0`) 5-step Quintic Newton-Schulz polar
     kernels in exact FP32 for all ``m <= 256`` (BF16 strictly for ``m > 256`` on CUDA),
     eliminating the ``m = 64..256`` BF16 off-diagonal leakage when ``smoothing == 0`` or
     ``fg_damp`` fades Tikhonov smoothing to zero.
   - Caps ``(target_frob / out_frob).clamp_max(1.15)`` uniformly across all branches so
     rank-1 / rank-deficient updates keep ``sigma_1 <= 1.25`` (never ``sqrt(m)`` blowup).
2. **D-03 Exact Scale-Safe Norms & 1D Invariance (`1e-37` Safe):**
   - Uses ``amax``-factored norms across neural-G, slow, prior, and generator-free critic
     paths (no FP32 ``1e-25`` norm underflow).
   - Removes unscaled additive ``eps`` from all 1D normalization denominators so 1D vector
     parameters maintain exact degree-0 scale invariance down to ``1e-37``.
3. **Strict `_conv_layouts` Dispatch & Ungrouped `1x1 Conv2d == 2D Linear` Equivalence:**
   - Classifies spatial convolutions strictly via registered ``_conv_layouts`` (never raw
     ``p.ndim == 4``), routing ungrouped ``1x1 Conv2d`` through the exact ``2D Linear``
     geometry (G 5-step Quintic NS, D ``aspect * sqrt(m)`` + exact-deadband Schatten-6
     with zero-sync Frobenius short-circuit) while keeping spatial ``k x k`` (``k > 1``),
     grouped Conv2d, and ConvTranspose2d at ``0`` NS / ``0`` Schatten-6 for full DCGAN speed.
4. **D-04 Smooth ``C^1`` Trust Band, Verified Iterative Backtracking & Mode-Matched Probes:**
   - Exact ``C^1``-smooth ``_band_scale`` with exact ``1.0`` deadband (``0`` extra forwards
     inside the deadband) and up to 4-step verified iterative backtracking whenever
     ``alpha != 1.0`` (``0`` post-rescale overshoots on saturating or superlinear maps).
   - Scale-homogeneous ``coord_scale`` (no fixed ``0.05`` floor) and ``origin_damping``
     relative to tracked ``ema_data_scale``, plus mode-matched ``eval()`` ``y_pre`` capture
     before generator updates for custom training-masked generators.
5. **D-05 Ceiling-Free Degree-0 & Spike-Immune Two-Player Scale Reference (Defects A, B, F):**
   - Replaces the fixed ``s_base`` ceiling and unguarded peak ratchet with a Huber-capped
     scale reference ``peak_new = max(peak_prev * p_dec, min(cur, 4 * peak_prev))``
     (``p_dec = 0.9995`` on cooled two-objective conditional steps, ``1.0`` elsewhere)
     and a smooth, width- and depth-invariant two-player mean-Frobenius scale reference
     ``S_mean(G) = mean(peak_frob(G))``, ``S_mean(D) = max(mean(peak_frob(D)), S_mean(G))``:
     ``s_eff(p) = min((k_role * s_base) * S_mean, 0.50 * peak_p)`` with ``k_G = 1.0``,
     ``k_D = 2.5``, ``G_ref = 1.0 * S_mean(G)`` for coupled prior tables, and
     ``k_gf = 14.0`` per-parameter settling in the generator-free critic path. Preserves
     exact trajectory degree-0 across 24+ decades, zero Frobenius width drift, continuous
     equilibrium settling at every scale, and >=95% recovery from ``1e4x`` spikes.
"""
from __future__ import annotations

from copy import copy, deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer

from .._unified_v4.optimizer import _QUINTIC_COEFFS, _smoothstep01
from .critic import _eval_v15_probe_forward as _eval_probe_forward
from .._unified_v11.optimizer import (
    UnifiedV11EquilibratedOptimizer,
    collect_conv_layouts,
    zero_pinv_project_nonascent,
)

_FLIP_SKETCH_DIM = 67  # Prime -> coprime to all power-of-2 layer widths/strides (268 B total, 0 B/param)
_PEAK_GROWTH_CAP = 4.0  # Huber-capped per-step upward growth -> immune to transient 1e4x spikes, passes natural <=4x jumps
_REL_SMOOTH_MAX = 0.50
_REL_SMOOTH_GEN = 1.0  # Two-player mean-Frobenius settling multiplier for generators/encoders
_REL_SMOOTH_CRITIC = 2.5  # Two-player mean-Frobenius settling multiplier for neural-G critics
_REL_SMOOTH_GF_CRITIC = 14.0  # Per-param relative settling multiplier for generator-free critics
_REL_SMOOTH_PRIOR_REF = 1.0  # Coupled prior scale reference multiplier on mean-Frobenius scale S_mean
_CRITIC_SCHATTEN6_LAM = 0.25
_CRITIC_SCHATTEN6_TAU = 2.0
_CRITIC_SCHATTEN6_BAND = 0.25
_BAND_C1_WIDTH = 0.05

_EYE_CACHE: Dict[Tuple[torch.device, int, torch.dtype], torch.Tensor] = {}


def _cached_eye(m_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, m_dim, dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(m_dim, device=device, dtype=dtype)
        if len(_EYE_CACHE) < 128:
            _EYE_CACHE[key] = eye
    return eye


def _scale_safe_mean_tensors(vals: List[torch.Tensor]) -> torch.Tensor:
    """Exact ``1e-37``-safe, overflow-safe, zero-sync deterministic mean across 0D device tensors."""
    if len(vals) == 1:
        return vals[0]
    stacked = torch.stack(vals)
    v_max = stacked.amax()
    active = v_max > 1e-38
    v_safe = torch.where(active, v_max, torch.ones_like(v_max))
    return torch.where(active, (stacked / v_safe).mean() * v_safe, v_max)


def _huber_peak_f(peak_prev: float, cur_sv: float, p_dec: float) -> float:
    """Exact degree-1, spike-immune Huber-capped scale reference update (Python float)."""
    if peak_prev <= 0.0:
        return cur_sv
    return max(peak_prev * p_dec, min(cur_sv, peak_prev * _PEAK_GROWTH_CAP))


def _huber_peak_t(
    peak_prev: torch.Tensor,
    cur_sv: torch.Tensor,
    valid: torch.Tensor,
    p_dec: float,
) -> torch.Tensor:
    """Exact degree-1, spike-immune Huber-capped scale reference update (zero-sync device tensor)."""
    cur_safe = torch.where(valid, cur_sv, peak_prev.new_zeros(()))
    cur_eff = torch.where(peak_prev > 0.0, torch.minimum(cur_safe, peak_prev * _PEAK_GROWTH_CAP), cur_safe)
    cand = torch.maximum(peak_prev * p_dec, cur_eff)
    return torch.where(valid & torch.isfinite(cand), cand, peak_prev)


def _band_scale(
    delta: float,
    delta_min: float,
    delta_max: float,
    *,
    alpha_max: float = 48.0,
    alpha_min: float = 1e-4,
) -> float:
    """Strictly ``C^2``-smooth trust-band gain with exact ``1.0`` deadband on ``[delta_min, delta_max]``."""
    if not math.isfinite(delta) or delta <= 1e-12:
        return 1.0
    w2 = _BAND_C1_WIDTH * _BAND_C1_WIDTH
    if delta_max > 0.0 and delta > delta_max:
        u = (delta / delta_max) - 1.0
        r_tilde = 1.0 + (u * u * u) / (u * u + w2)
        return float(max(alpha_min, 1.0 / r_tilde))
    if delta_min > 0.0 and delta < delta_min:
        denom_lo = delta + delta_min / max(1.0, alpha_max - 1.0)
        u_lo = (delta_min - delta) / denom_lo
        gain = 1.0 + (u_lo * u_lo * u_lo) / (u_lo * u_lo + w2)
        return float(min(alpha_max, max(1.0, gain)))
    return 1.0


@torch.no_grad()
def quintic_orthogonalize_2d(
    mat: torch.Tensor,
    steps: int = 5,
    *,
    smoothing: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zero-SVD scale-normalized 5-step Quintic polar (exact FP32 for ``m <= 256``, BF16 for ``m > 256`` on CUDA)."""
    if mat.ndim != 2:
        raise ValueError(f"quintic_orthogonalize_2d expects a 2D matrix, got shape {tuple(mat.shape)}")
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return torch.zeros_like(mat)
    s_in: float | torch.Tensor = smoothing.reshape(1) if isinstance(smoothing, torch.Tensor) else smoothing
    return quintic_orthogonalize_batched(mat.unsqueeze(0), steps=steps, smoothing=s_in).squeeze(0)


@torch.no_grad()
def quintic_orthogonalize_batched(
    mats: torch.Tensor,
    steps: int = 5,
    *,
    smoothing: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Batched zero-SVD 5-step Quintic polar (exact FP32 for ``m <= 256``, BF16 for ``m > 256`` on CUDA)."""
    if mats.ndim != 3:
        raise ValueError(f"quintic_orthogonalize_batched expects a 3D batch, got shape {tuple(mats.shape)}")
    b_sz, r_dim, c_dim = int(mats.shape[0]), int(mats.shape[1]), int(mats.shape[2])
    if b_sz == 0 or r_dim == 0 or c_dim == 0:
        return torch.zeros_like(mats)
    orig_dtype = mats.dtype
    x_f32 = mats.float()
    if isinstance(smoothing, torch.Tensor):
        s_ten = smoothing.to(device=mats.device, dtype=torch.float32).reshape(b_sz, 1, 1).clamp_min(0.0)
        has_smooth = True
    else:
        s_val = float(smoothing)
        has_smooth = s_val > 0.0
        s_ten = x_f32.new_full((b_sz, 1, 1), s_val) if has_smooth else None
    max_abs = x_f32.abs().amax(dim=(1, 2), keepdim=True)
    finite_mat = torch.isfinite(x_f32).all(dim=(1, 2), keepdim=True)
    m_small = min(r_dim, c_dim)
    if has_smooth and s_ten is not None and m_small <= 512:
        scale_ref = torch.maximum(torch.where(finite_mat, max_abs, torch.zeros_like(max_abs)), s_ten)
        safe_max = torch.where(scale_ref > 0.0, scale_ref, torch.ones_like(scale_ref))
        x_scaled = torch.where(finite_mat, x_f32 / safe_max, torch.zeros_like(x_f32))
        s_scaled = s_ten / safe_max
        frob = x_scaled.flatten(1).norm(dim=1).view(b_sz, 1, 1)
        active = finite_mat & (frob > 1e-30)
        if r_dim == 1 or c_dim == 1:
            unit = x_scaled / torch.hypot(frob, s_scaled).clamp_min(1e-30)
            return torch.where(active, unit, torch.zeros_like(unit)).to(orig_dtype)
        transposed = r_dim > c_dim
        x_in = x_scaled.transpose(1, 2) if transposed else x_scaled
        m_dim, n_dim = int(x_in.shape[1]), int(x_in.shape[2])
        aug_frob = torch.sqrt(frob.square() + float(m_dim) * s_scaled.square()).clamp_min(1e-30)
        inv_aug = 1.0 / (aug_frob * 1.02)
        work_dtype = torch.bfloat16 if (mats.is_cuda and m_dim > 256) else torch.float32
        x_part = (x_in * inv_aug).to(dtype=work_dtype)
        eye_b = _cached_eye(m_dim, mats.device, work_dtype).unsqueeze(0).expand(b_sz, m_dim, m_dim)
        y_part = eye_b * (s_scaled * inv_aug).to(dtype=work_dtype)
        z = torch.cat((x_part, y_part), dim=2)
        for a, b, c in _QUINTIC_COEFFS[:steps]:
            gram = torch.bmm(z, z.transpose(1, 2))
            gram2 = torch.bmm(gram, gram)
            z = a * z + torch.bmm(b * gram + c * gram2, z)
        z_f32 = z.float()
        joint_frob = z_f32.flatten(1).norm(dim=1).view(b_sz, 1, 1).clamp_min(1e-8)
        x_out = z_f32[:, :, :n_dim]
        if transposed:
            x_out = x_out.transpose(1, 2)
        out = x_out * (math.sqrt(float(m_dim)) / joint_frob).clamp_max(1.15)
        out_ok = active & torch.isfinite(out).all(dim=(1, 2), keepdim=True)
        out = torch.where(out_ok, out, torch.zeros_like(out))
        return out.to(orig_dtype)

    safe_max = torch.where(finite_mat & (max_abs > 0.0), max_abs, torch.ones_like(max_abs))
    x_scaled = torch.where(finite_mat, x_f32 / safe_max, torch.zeros_like(x_f32))
    frob = x_scaled.flatten(1).norm(dim=1).view(b_sz, 1, 1)
    active = finite_mat & (frob > 1e-30)
    if r_dim == 1 or c_dim == 1:
        unit = x_scaled / frob.clamp_min(1e-30)
        return torch.where(active, unit, torch.zeros_like(unit)).to(orig_dtype)

    x_norm = x_scaled / (frob * 1.02 + 1e-30)
    transposed = r_dim > c_dim
    if transposed:
        x_norm = x_norm.transpose(1, 2)

    work_dtype = torch.bfloat16 if (mats.is_cuda and m_small > 256) else torch.float32
    x = x_norm.to(dtype=work_dtype)
    for a, b, c in _QUINTIC_COEFFS[:steps]:
        gram = torch.bmm(x, x.transpose(1, 2))
        gram2 = torch.bmm(gram, gram)
        x = a * x + torch.bmm(b * gram + c * gram2, x)

    if transposed:
        x = x.transpose(1, 2)
    out = x.float()
    target_frob = math.sqrt(float(m_small))
    out_frob = out.flatten(1).norm(dim=1).view(b_sz, 1, 1).clamp_min(1e-8)
    frob_gain = (target_frob / out_frob).clamp_max(1.15)
    if has_smooth and s_ten is not None:
        g_rms = (frob * safe_max) / max(target_frob, 1.0)
        s_scale = g_rms / torch.hypot(g_rms, s_ten).clamp_min(1e-30)
        out = out * (frob_gain * s_scale)
    else:
        out = out * frob_gain
    out_ok = active & torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    out = torch.where(out_ok, out, torch.zeros_like(out))
    return out.to(orig_dtype)


@torch.no_grad()
def _schatten6_spectral_decay_2d(
    w_canon: torch.Tensor,
    rate_scale: torch.Tensor,
    *,
    lam: float = _CRITIC_SCHATTEN6_LAM,
    tau: float = _CRITIC_SCHATTEN6_TAU,
) -> torch.Tensor:
    """Zero-SVD, exact-deadband Schatten-6 spectral decay for canonical 2D matrix (``m <= n``)."""
    return _schatten6_spectral_decay_batched(w_canon.unsqueeze(0), rate_scale.reshape(1), lam=lam, tau=tau).squeeze(0)


@torch.no_grad()
def _schatten6_spectral_decay_batched(
    w_canons: torch.Tensor,
    rate_scales: torch.Tensor,
    *,
    lam: float = _CRITIC_SCHATTEN6_LAM,
    tau: float = _CRITIC_SCHATTEN6_TAU,
) -> torch.Tensor:
    """Batched zero-SVD, exact-deadband Schatten-6 spectral decay for ``[B, m, n]`` (``m <= n``)."""
    b_sz, m_dim = int(w_canons.shape[0]), int(w_canons.shape[1])
    w_hat = w_canons.float() / tau
    w_norms = w_hat.flatten(1).norm(dim=1).view(b_sz, 1, 1)
    if (not w_canons.is_cuda) and float(w_norms.amax().item()) <= 1.0:
        return torch.zeros_like(w_hat)
    work_dtype = torch.bfloat16 if (w_canons.is_cuda and m_dim > 256) else torch.float32
    wb = w_hat.to(dtype=work_dtype)
    gram = torch.bmm(wb, wb.transpose(1, 2))
    gram2 = torch.bmm(gram, gram)
    g2_f = gram2.float()
    s_g2 = g2_f.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    g2_scaled = g2_f / s_g2
    g2_u = g2_scaled.to(dtype=work_dtype)
    g4_u = torch.bmm(g2_u, g2_u).float()
    g2_n = g2_scaled.flatten(1).norm(dim=1).view(b_sz, 1, 1).clamp_min(1e-12)
    g4_n = g4_u.flatten(1).norm(dim=1).view(b_sz, 1, 1)
    rho_est = ((g4_n / g2_n) * s_g2).clamp_min(0.0).pow(0.25)
    t_gate = ((rho_est - 1.0) / _CRITIC_SCHATTEN6_BAND).clamp(0.0, 1.0)
    gate = t_gate * t_gate * (3.0 - 2.0 * t_gate)

    w5 = torch.bmm(gram2, wb).float()
    w_norm = w_norms.clamp_min(1e-12)
    c_norm = w5.flatten(1).norm(dim=1).view(b_sz, 1, 1)
    ratio = (c_norm / w_norm).clamp_min(1e-6)
    raw_step = (lam / tau) * rate_scales.view(b_sz, 1, 1) * gate
    eff_step = torch.minimum(raw_step, 0.25 / ratio)
    out = -w5 * (eff_step * tau)
    finite_ok = torch.isfinite(out).all(dim=(1, 2), keepdim=True) & torch.isfinite(eff_step) & (gate > 0.0)
    return torch.where(finite_ok, out, torch.zeros_like(out))


def _update_scale_safe_moments(
    st: Dict[str, Any],
    p: torch.Tensor,
    gf_safe: torch.Tensor,
    g_max: torch.Tensor,
    finite: torch.Tensor,
    g_active: torch.Tensor,
    step_t: int,
    b1_use: float,
    b2_eff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-safe ``1e-37``-invariant first/second moment update returning ``(m_hat, m_norm, raw_coh, sc)``."""
    if "m_unit" not in st:
        st["m_unit"] = torch.zeros_like(p, dtype=torch.float32)
        st["m_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
        st["v_rms"] = torch.zeros_like(p, dtype=torch.float32)
        st["v_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
    m_prev_sc = st["m_scale"]
    v_prev_sc = st["v_scale"]
    sc = torch.maximum(torch.maximum(m_prev_sc, v_prev_sc), g_max).clamp_min(1e-38)
    m_rescaled = st["m_unit"] * (m_prev_sc / sc)
    v_rescaled = st["v_rms"] * (v_prev_sc / sc)
    g_scaled = gf_safe / sc
    m_next = torch.where(finite, m_rescaled.mul(b1_use).add(g_scaled, alpha=1.0 - b1_use), m_rescaled)
    v_next = torch.where(
        finite,
        (b2_eff * v_rescaled.square() + (1.0 - b2_eff) * g_scaled.square()).sqrt(),
        v_rescaled,
    )
    new_m_sc = m_next.abs().amax()
    safe_m_sc = torch.where(new_m_sc > 1e-38, new_m_sc, torch.ones_like(new_m_sc))
    st["m_unit"].copy_(torch.where(new_m_sc > 1e-38, m_next / safe_m_sc, torch.zeros_like(m_next)))
    st["m_scale"].copy_(torch.where(new_m_sc > 1e-38, new_m_sc, torch.zeros_like(new_m_sc)) * sc)

    new_v_sc = v_next.abs().amax()
    safe_v_sc = torch.where(new_v_sc > 1e-38, new_v_sc, torch.ones_like(new_v_sc))
    st["v_rms"].copy_(torch.where(new_v_sc > 1e-38, v_next / safe_v_sc, torch.zeros_like(v_next)))
    st["v_scale"].copy_(torch.where(new_v_sc > 1e-38, new_v_sc, torch.zeros_like(new_v_sc)) * sc)

    m_hat = m_next / max(1.0 - (b1_use ** step_t), 1e-8)
    v_hat = v_next / max(math.sqrt(1.0 - (b2_eff ** step_t)), 1e-8)
    m_norm = m_hat.norm()
    v_norm = v_hat.norm()
    raw_coh = torch.where(
        g_active & (v_norm > 1e-30) & (m_norm > 1e-30),
        (m_norm / v_norm.clamp_min(1e-30)).clamp(0.0, 1.0),
        torch.zeros_like(m_norm),
    )
    return m_hat, m_norm, raw_coh, sc


def _enqueue_s6(
    pending_s6: Dict[Tuple[int, int], List[Tuple[torch.Tensor, int, int, float, bool, torch.Tensor]]],
    p: torch.Tensor,
    r_d: int,
    c_d: int,
    aspect_factor: float,
    rate_t: torch.Tensor,
) -> None:
    needs_t_s6 = r_d > c_d
    key_s6 = (c_d, r_d) if needs_t_s6 else (r_d, c_d)
    pending_s6.setdefault(key_s6, []).append((p, r_d, c_d, aspect_factor, needs_t_s6, rate_t))


def _sketch_grad_67(grad: torch.Tensor, roll: int) -> torch.Tensor:
    """Deterministic, degree-0 scale-invariant prime-67 signed CountSketch of a gradient tensor."""
    gf = grad.detach().float().reshape(-1)
    finite = torch.isfinite(gf).all()
    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
    g_max = gf_safe.abs().amax()
    safe_sc = torch.where(g_max > 1e-38, g_max, torch.ones_like(g_max))
    gf_u = torch.where(g_max > 1e-38, gf_safe / safe_sc, torch.zeros_like(gf_safe))
    n = int(gf_u.numel())
    rem = n % _FLIP_SKETCH_DIM
    if rem:
        gf_u = torch.cat((gf_u, gf_u.new_zeros(_FLIP_SKETCH_DIM - rem)), dim=0)
    rows = gf_u.view(-1, _FLIP_SKETCH_DIM)
    sk = (rows[0::2].sum(dim=0) - rows[1::2].sum(dim=0)) if int(rows.shape[0]) > 1 else rows[0]
    sk = sk / sk.norm().clamp_min(1e-30)
    if roll:
        sk = torch.roll(sk, shifts=roll, dims=0)
    return sk


class UnifiedV15Optimizer(UnifiedV11EquilibratedOptimizer):
    """Energy-optimal, zero-sync, Law-4-hardened equilibrated optimizer (`unified_v15_equil`)."""

    family = "unified_v15_equil"
    recipe_optimizer_family = "unified_v15_equil"
    _SUPPORTS_ZEROSYNC_V4_TENSORS: bool = True

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 0.014,
        betas: Tuple[float, float] = (0.0, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        smoothing: float = 0.0,
        direction_blend: bool = False,
        generator_free: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
            smoothing=smoothing,
            direction_blend=direction_blend,
            **options,
        )
        self.family = "unified_v15_equil"
        self.recipe_optimizer_family = "unified_v15_equil"
        self.ns_2d_activations: int = 0
        self.ns_5step_calls: int = 0
        self.fast_path_updates: int = 0
        self.critic_ns_calls: int = 0
        self.s6_skipped_deadband: int = 0
        self._generator_free: bool = bool(generator_free)
        self._ema_flip_cos: float = 0.0
        self._prev_flip_sketch: Optional[torch.Tensor] = None
        self._g_peak_ref: float = 0.0
        self._g_peak_ref_t: Optional[torch.Tensor] = None
        self._dense_metrics_cache: Dict[int, Tuple[int, int, int, float, float]] = {}
        self._conv_metrics_cache: Dict[Tuple[int, Tuple[bool, int]], Tuple[float, float]] = {}
        for grp in self.param_groups:
            if grp["algorithm"] == "unified_v11_equil":
                grp["algorithm"] = self.family

    def _is_spatial_conv(self, p: torch.Tensor) -> bool:
        """True strictly for registered Conv2d/ConvTranspose2d weights excluding ungrouped 1x1 Conv2d."""
        if p.ndim != 4:
            return False
        layout = self._conv_layouts.get(id(p))
        if layout is None:
            return False
        is_trans, groups_c = layout
        if (not is_trans) and groups_c == 1 and int(p.shape[2]) == 1 and int(p.shape[3]) == 1:
            return False
        return True

    def current_stiff_gate(self) -> float:
        """Floor-overshoot or sustained period-2 flip stiffness gate (0.0 on healthy & ResBN trunks)."""
        ema_fg = float(self._ema_func_gain)
        g_abs = _smoothstep01((0.056 - ema_fg) / 0.006)
        g_flip = _smoothstep01((0.60 - ema_fg) / 0.30) * _smoothstep01((-float(self._ema_flip_cos) - 0.50) / 0.15)
        return max(g_abs, g_flip)

    def _update_period2_flip_feedback(self, *, s_therm: float) -> None:
        """Zero-B/param (268 B total) prime-67 CountSketch closed-loop Period-2 Flip-Bifurcation controller."""
        ema_fg = float(self._ema_func_gain)
        if ema_fg < 0.60 and s_therm >= 0.5:
            sk_sum: Optional[torch.Tensor] = None
            idx_l = 0
            for grp in self.param_groups:
                if self._player(grp["role"]) != "generator":
                    continue
                for p in grp["params"]:
                    if (
                        p.ndim >= 2
                        and min(int(p.shape[0]), int(p.numel() // max(1, int(p.shape[0])))) >= 2
                        and p.grad is not None
                    ):
                        sk_l = _sketch_grad_67(p.grad, (idx_l * 13) % _FLIP_SKETCH_DIM)
                        sk_sum = sk_l if sk_sum is None else (sk_sum + sk_l)
                        idx_l += 1
            if sk_sum is not None:
                sk_nrm = sk_sum.norm()
                sk_u = sk_sum / sk_nrm.clamp_min(1e-30)
                prev = self._prev_flip_sketch
                if prev is not None:
                    if prev.device != sk_u.device:
                        prev = prev.to(device=sk_u.device, dtype=sk_u.dtype)
                    rho_val, nrm_val = torch.stack(((sk_u * prev).sum(), sk_nrm)).tolist()
                    if math.isfinite(nrm_val) and nrm_val > 1e-30 and math.isfinite(rho_val):
                        rho = max(-1.0, min(1.0, float(rho_val)))
                        self._ema_flip_cos = float(0.90 * self._ema_flip_cos + 0.10 * rho)
                        self._prev_flip_sketch = sk_u
                else:
                    nrm_val = float(sk_nrm.item())
                    if math.isfinite(nrm_val) and nrm_val > 1e-30:
                        self._prev_flip_sketch = sk_u
            flip_excess = _smoothstep01((-self._ema_flip_cos - 0.45) / 0.20)
            if flip_excess > 0.0 and ema_fg < 0.55:
                self._ema_func_gain = max(0.05, float(self._ema_func_gain) * (1.0 - 0.04 * flip_excess))
        else:
            self._prev_flip_sketch = None
            self._ema_flip_cos = float(self._ema_flip_cos * 0.95)

    def _conv_metrics(self, p: torch.Tensor) -> Tuple[float, float]:
        """Polar-matched, stride-aware 4D spatial Conv2d/ConvTranspose2d Frobenius-MP scale (`0` NS)."""
        c0, c1, kh, kw = (int(s) for s in p.shape)
        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
        out_ch = c1 * groups_c if is_trans else c0
        in_ch = c0 if is_trans else c1 * groups_c
        sv_denom_sqrt = math.sqrt(float(max(1, min(out_ch, in_ch) * kh * kw)))
        if kh == 1 and kw == 1:
            factor = math.sqrt(float(out_ch) / float(max(1, in_ch)))
            return factor, sv_denom_sqrt
        m_g = float(max(1, min(out_ch // max(1, groups_c), in_ch // max(1, groups_c))))
        k_eff = float(max(1, (kh * kw) // 4)) if is_trans else float(max(1, kh * kw))
        factor = math.sqrt(float(out_ch) / float(max(1, in_ch))) * math.sqrt(float(groups_c) / k_eff) * (m_g ** 0.25)
        return factor, sv_denom_sqrt

    def _dense_metrics(self, p: torch.Tensor) -> Tuple[float, float]:
        """RMS-to-RMS aspect factor and singular-value RMS denominator for a matrix (``ndim >= 2``)."""
        r_d = int(p.shape[0])
        c_d = int(p.numel() // max(1, r_d))
        factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
        sv_denom_sqrt = math.sqrt(float(max(1, min(r_d, c_d))))
        return factor, sv_denom_sqrt

    def _cached_conv_metrics(self, p: torch.Tensor) -> Tuple[float, float]:
        layout = self._conv_layouts.get(id(p), (False, 1))
        key = (id(p), layout)
        cached = self._conv_metrics_cache.get(key)
        if cached is None:
            cached = self._conv_metrics(p)
            self._conv_metrics_cache[key] = cached
        return cached

    def _cached_dense_metrics(self, p: torch.Tensor) -> Tuple[int, int, int, float, float]:
        key = id(p)
        cached = self._dense_metrics_cache.get(key)
        if cached is None:
            r_d = int(p.shape[0])
            c_d = int(p.numel() // max(1, r_d))
            m_dim = min(r_d, c_d)
            factor, sv_denom_sqrt = self._dense_metrics(p)
            cached = (r_d, c_d, m_dim, factor, sv_denom_sqrt)
            self._dense_metrics_cache[key] = cached
        return cached

    def _compute_two_obj_pyth_scales(self, *, is_cond_fiber: bool) -> Dict[int, float]:
        """Role-level K=2 Triangle-Inequality ball-radius compensation on cooled conditional-fiber steps."""
        normals = self._protected
        if (not is_cond_fiber) or normals is None or int(normals.shape[0]) != 2:
            return {}

        n0_all = normals[0]
        n1_all = normals[1]
        stat_roles: List[str] = []
        stat_tensors: List[torch.Tensor] = []
        role_pids: Dict[str, List[int]] = {}

        offset = 0
        for grp in self.param_groups:
            role = grp["role"]
            pids_r = role_pids.setdefault(role, [])
            for p in grp["params"]:
                sz = p.numel()
                g0_flat = n0_all[offset : offset + sz]
                g1_flat = n1_all[offset : offset + sz]
                offset += sz
                pids_r.append(id(p))
                if p.grad is not None and sz > 0:
                    g0_d = g0_flat.double()
                    g1_d = g1_flat.double()
                    stat_roles.append(role)
                    stat_tensors.append((g0_d * g0_d).sum())
                    stat_tensors.append((g1_d * g1_d).sum())
                    stat_tensors.append((g0_d * g1_d).sum())

        role_acc: Dict[str, List[float]] = {r: [0.0, 0.0, 0.0] for r in role_pids}
        if stat_tensors:
            vals = torch.stack(stat_tensors).tolist()
            for idx_p, role in enumerate(stat_roles):
                a2 = float(vals[3 * idx_p])
                b2 = float(vals[3 * idx_p + 1])
                dot = float(vals[3 * idx_p + 2])
                if a2 > 1e-24 and b2 > 1e-24 and math.isfinite(a2) and math.isfinite(b2) and math.isfinite(dot):
                    acc = role_acc[role]
                    acc[0] += a2
                    acc[1] += b2
                    acc[2] += dot

        param_mults: Dict[int, float] = {}
        for role, pids_r in role_pids.items():
            a2, b2, dot = role_acc[role]
            if a2 > 1e-24 and b2 > 1e-24:
                a = math.sqrt(a2)
                b = math.sqrt(b2)
                sum_sq = max(a2 + b2 + 2.0 * dot, 1e-36)
                mult = min(2.0, max(1.0, (a + b) / math.sqrt(sum_sq)))
            else:
                mult = 1.0
            for pid in pids_r:
                param_mults[pid] = mult
        return param_mults

    def _apply_prior_param(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        st: Dict[str, Any],
        base_lr_prior: float,
        b1_eff: float,
        s_base_prior: float,
        g_ref_f: float,
        g_ref_t: Optional[torch.Tensor],
        p_dec: float,
        sampled_req: bool,
    ) -> None:
        """Zero-sync scale-homogeneous, spike-immune row-normalized update for particle prior tables."""
        r_scales = self._row_scales.get(p)
        use_zerosync_sort = sampled_req and r_scales is None
        if use_zerosync_sort:
            raw_rows = self._sampled_rows[p]
            if raw_rows.numel() == 0:
                return
            rows_idx, _ = torch.sort(raw_rows)
            n_raw = int(rows_idx.shape[0])
            is_first = torch.ones(n_raw, device=p.device, dtype=torch.bool)
            if n_raw > 1:
                is_first[1:] = rows_idx[1:] != rows_idx[:-1]
            n_unique_t = is_first.sum().clamp_min(1).float()
            w_u_col = is_first.float().unsqueeze(1)
        else:
            rows_opt = (
                self._resolve_unique_sampled_rows(p)
                if sampled_req
                else torch.arange(len(p), device=p.device, dtype=torch.long)
            )
            if rows_opt is None or rows_opt.numel() == 0:
                return
            rows_idx = rows_opt
            is_first = None
            n_unique_t = None
            w_u_col = None

        gr = grad[rows_idx].float()
        fin_row = torch.isfinite(gr).all(dim=-1, keepdim=True)
        gr_safe = torch.where(fin_row, gr, torch.zeros_like(gr))
        if b1_eff > 0.0:
            if "m_prior" not in st:
                st["m_prior"] = torch.zeros_like(p, dtype=torch.float32)
            m_prev = st["m_prior"][rows_idx]
            m_new = torch.where(fin_row, m_prev.mul(b1_eff).add(gr_safe, alpha=1.0 - b1_eff), m_prev)
            if use_zerosync_sort and w_u_col is not None:
                st["m_prior"].index_add_(0, rows_idx, (m_new - m_prev) * w_u_col)
            else:
                st["m_prior"][rows_idx] = m_new
            dir_r = m_new
        else:
            dir_r = gr_safe

        g_max_r = gr_safe.abs().amax(dim=-1, keepdim=True)
        g_ext_r = (g_max_r > 1e-38) & ((g_max_r < 1e-18) | (g_max_r > 1e18))
        g_sc_r = torch.where(g_ext_r, g_max_r, torch.ones_like(g_max_r))
        g_norm = torch.where(g_ext_r, (gr_safe / g_sc_r).norm(dim=-1, keepdim=True) * g_sc_r, gr_safe.norm(dim=-1, keepdim=True))
        if b1_eff > 0.0:
            d_max_r = dir_r.abs().amax(dim=-1, keepdim=True)
            d_ext_r = (d_max_r > 1e-38) & ((d_max_r < 1e-18) | (d_max_r > 1e18))
            d_sc_r = torch.where(d_ext_r, d_max_r, torch.ones_like(d_max_r))
            d_norm = torch.where(d_ext_r, (dir_r / d_sc_r).norm(dim=-1, keepdim=True) * d_sc_r, dir_r.norm(dim=-1, keepdim=True))
        else:
            d_norm = g_norm

        if use_zerosync_sort and is_first is not None and w_u_col is not None and n_unique_t is not None:
            active_r = is_first.unsqueeze(1) & fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)
            d_u = torch.where(is_first.unsqueeze(1), d_norm, torch.zeros_like(d_norm))
            cur_raw = ((d_norm.square() * w_u_col).sum() / n_unique_t).sqrt()
            d_u_max = d_u.amax()
            d_u_ext = torch.isfinite(d_u_max) & (d_u_max > 1e-38) & ((d_u_max < 1e-18) | (d_u_max > 1e18))
            safe_d_u = torch.where(d_u_ext, d_u_max, d_u_max.new_ones(()))
            cur_row_rms = torch.where(
                d_u_ext,
                ((d_u / safe_d_u).square().sum() / n_unique_t).sqrt() * safe_d_u,
                cur_raw,
            )
        else:
            active_r = fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)
            d_u_max = d_norm.amax()
            d_u_ext = torch.isfinite(d_u_max) & (d_u_max > 1e-38) & ((d_u_max < 1e-18) | (d_u_max > 1e18))
            safe_d_u = torch.where(d_u_ext, d_u_max, d_u_max.new_ones(()))
            cur_row_rms = torch.where(
                d_u_ext,
                (d_norm / safe_d_u).square().mean().sqrt() * safe_d_u,
                d_norm.square().mean().sqrt(),
            )

        if "peak_rms" not in st:
            st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
        peak_r = _huber_peak_t(st["peak_rms"], cur_row_rms, torch.isfinite(cur_row_rms), p_dec)
        st["peak_rms"].copy_(peak_r)

        if s_base_prior > 0.0:
            if g_ref_f > 0.0 or g_ref_t is not None:
                if use_zerosync_sort and n_unique_t is not None:
                    s_prior_eff = torch.where(
                        n_unique_t > 128.0,
                        (s_base_prior * 128.0) / n_unique_t.clamp_min(1.0),
                        peak_r.new_full((), s_base_prior),
                    )
                else:
                    n_u_f = float(len(rows_idx))
                    s_prior_eff = peak_r.new_full((), s_base_prior * (128.0 / n_u_f) if n_u_f > 128.0 else s_base_prior)
                if g_ref_f > 0.0:
                    s_abs_p = s_prior_eff * g_ref_f
                else:
                    assert g_ref_t is not None
                    s_abs_p = s_prior_eff * g_ref_t.to(device=peak_r.device, dtype=peak_r.dtype)
            else:
                # Generator-free prior-only optimizer: peak_r already scales as 1/n_unique -> no double 128/n_unique!
                s_abs_p = peak_r * (s_base_prior * math.sqrt(float(max(1, int(p.shape[0])))))
            s_p_row = torch.minimum(s_abs_p, 2.0 * peak_r)
            denom_r = torch.hypot(d_norm, s_p_row).clamp_min(1e-36)
        else:
            denom_r = d_norm.clamp_min(1e-36)
        unit_r = torch.where(active_r, dir_r / denom_r, torch.zeros_like(dir_r))
        if (not use_zerosync_sort) and r_scales is not None and len(r_scales) == len(rows_idx):
            unit_r = unit_r * r_scales.unsqueeze(1)
        p.index_add_(0, rows_idx, unit_r.to(p.dtype), alpha=-base_lr_prior)

    def _apply_slow_param_zerosync(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        step_t: int,
        rate_p: float | torch.Tensor,
        rate_positive: bool,
        b1_eff: float,
        b2_eff: float,
        s_base: float,
        peak_precomputed: Optional[torch.Tensor],
        scale_ref_f: float = 0.0,
        scale_ref_t: Optional[torch.Tensor] = None,
        s_therm: float,
        p_dec: float,
        stiff_gate: float,
        is_critic_role: bool,
        track_secant: bool,
        coh_vals: List[torch.Tensor],
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]],
        pending_dense: Dict[
            Tuple[int, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
        ],
        pending_s6: Optional[
            Dict[Tuple[int, int], List[Tuple[torch.Tensor, int, int, float, bool, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Zero-sync device-tensor 1e-37-safe / stiff 2D NS path (0 spatial Conv2d NS & 0 neural-G Critic NS)."""
        st = self.state[p]
        gf = grad.float()
        finite = torch.isfinite(gf).all()
        gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
        g_max = gf_safe.abs().amax()
        g_active = finite & (g_max > 1e-38)

        is_conv = self._is_spatial_conv(p)
        aspect_factor = 1.0
        r_d = c_d = m_dim = 1
        if is_conv:
            factor, sv_denom_sqrt = self._cached_conv_metrics(p)
        elif p.ndim >= 2:
            r_d, c_d, m_dim, aspect_factor, sv_denom_sqrt = self._cached_dense_metrics(p)
            factor = aspect_factor * math.sqrt(float(m_dim)) if (is_critic_role and m_dim >= 2) else aspect_factor
        else:
            factor, sv_denom_sqrt = 1.0, 1.0

        safe_scale = torch.where(g_active, g_max, torch.ones_like(g_max))
        g_unit_scaled = torch.where(g_active, gf_safe / safe_scale, torch.zeros_like(gf_safe))
        scaled_norm = g_unit_scaled.norm()
        if peak_precomputed is not None:
            peak_new_t = peak_precomputed
        else:
            cur_sv_rms_t = (scaled_norm * safe_scale) / sv_denom_sqrt
            if "peak_rms" not in st:
                init_peak = float(st.get("peak_rms_val", 0.0))
                st["peak_rms"] = gf_safe.new_full((), init_peak)
            peak_new_t = _huber_peak_t(st["peak_rms"], cur_sv_rms_t, finite & torch.isfinite(cur_sv_rms_t), p_dec)
            st["peak_rms"].copy_(peak_new_t)

        has_smooth = s_base > 0.0
        if has_smooth:
            rel_mult = _REL_SMOOTH_CRITIC if is_critic_role else _REL_SMOOTH_GEN
            if scale_ref_f > 0.0:
                ref_t = peak_new_t.new_full((), scale_ref_f)
            elif scale_ref_t is not None:
                ref_t = scale_ref_t.to(device=peak_new_t.device, dtype=peak_new_t.dtype)
            else:
                ref_t = peak_new_t * sv_denom_sqrt
            s_eff_t = torch.minimum((rel_mult * s_base) * ref_t, _REL_SMOOTH_MAX * peak_new_t)
        else:
            s_eff_t = gf_safe.new_zeros(())

        is_scalar_1d = p.ndim == 1 and p.numel() == 1
        use_moments = (b1_eff > 0.0) or (stiff_gate > 0.0) or is_scalar_1d

        if use_moments:
            b1_use = max(b1_eff, 0.80) if is_scalar_1d else b1_eff
            m_hat, m_norm, raw_coh, sc = _update_scale_safe_moments(
                st, p, gf_safe, g_max, finite, g_active, step_t, b1_use, b2_eff
            )
            if is_scalar_1d:
                coh = raw_coh.clamp(0.35, 1.0)
            elif stiff_gate > 0.0:
                coh = (1.0 - stiff_gate) + stiff_gate * raw_coh
            else:
                coh = torch.ones_like(m_norm)

            if not rate_positive:
                return
            if track_secant:
                coh_vals.append(coh)

            # Stiff 2D dense / ungrouped 1x1 Conv2d matrices on GENERATOR ONLY:
            if (not is_conv) and (not is_critic_role) and p.ndim >= 2 and m_dim >= 2:
                raw_dir = (m_hat * sc) if has_smooth else m_hat
                direc = torch.where(g_active, raw_dir, torch.zeros_like(m_hat))
                mat = direc.reshape(r_d, c_d)
                step_mult = (-rate_p * factor) * coh
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
                )
                return

            s_scaled = (s_eff_t * sv_denom_sqrt) / sc.clamp_min(1e-38)
            denom_m = torch.hypot(m_norm, s_scaled).clamp_min(1e-30) if has_smooth else m_norm.clamp_min(1e-30)
            unit_m = torch.where(g_active & (m_norm > 1e-30), m_hat / denom_m, torch.zeros_like(m_hat))
            delta_m = (unit_m * ((-rate_p * factor) * coh)).to(p.dtype)
            p.add_(delta_m)
            if track_secant:
                applied_deltas.append((p, delta_m))
            if pending_s6 is not None and is_critic_role and (not is_conv) and p.ndim >= 2 and m_dim >= 2 and s_therm > 0.0:
                s6_rate_val = rate_p * s_therm
                s6_rate_t = (
                    s6_rate_val.to(device=gf_safe.device, dtype=gf_safe.dtype)
                    if isinstance(s6_rate_val, torch.Tensor)
                    else gf_safe.new_full((), float(s6_rate_val))
                )
                _enqueue_s6(pending_s6, p, r_d, c_d, aspect_factor, torch.where(g_active, s6_rate_t, gf_safe.new_zeros(())))
            return

        # Zero-state 1e-37-safe slow fallback (extreme scales <1e-18 / >1e18 / NaN / non-FP32):
        if not rate_positive:
            return
        if track_secant:
            coh_vals.append(gf_safe.new_ones(()))
        if (not is_conv) and (not is_critic_role) and p.ndim >= 2 and m_dim >= 2:
            raw_dir = gf_safe if has_smooth else g_unit_scaled
            direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
            mat = direc.reshape(r_d, c_d)
            step_mult = (
                (-rate_p * factor).to(device=gf_safe.device, dtype=gf_safe.dtype)
                if isinstance(rate_p, torch.Tensor)
                else gf_safe.new_full((), -float(rate_p) * factor)
            )
            needs_t = r_d > c_d
            mat_canon = mat.T if needs_t else mat
            m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
            pending_dense.setdefault(m_key, []).append(
                (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
            )
            return
        if has_smooth:
            s_scaled = (s_eff_t * sv_denom_sqrt) / safe_scale.clamp_min(1e-38)
            denom = torch.hypot(scaled_norm, s_scaled).clamp_min(1e-30)
        else:
            denom = scaled_norm.clamp_min(1e-30)
        unit_g = torch.where(g_active & (scaled_norm > 1e-30), g_unit_scaled / denom, torch.zeros_like(g_unit_scaled))
        delta_g = (unit_g * (-rate_p * factor)).to(p.dtype)
        p.add_(delta_g)
        if track_secant:
            applied_deltas.append((p, delta_g))
        if pending_s6 is not None and is_critic_role and (not is_conv) and p.ndim >= 2 and m_dim >= 2 and s_therm > 0.0:
            s6_rate_val = rate_p * s_therm
            s6_rate_t = (
                s6_rate_val.to(device=gf_safe.device, dtype=gf_safe.dtype)
                if isinstance(s6_rate_val, torch.Tensor)
                else gf_safe.new_full((), float(s6_rate_val))
            )
            _enqueue_s6(pending_s6, p, r_d, c_d, aspect_factor, torch.where(g_active, s6_rate_t, gf_safe.new_zeros(())))

    def _flush_pending_dense(
        self,
        pending_dense: Dict[
            Tuple[int, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
        ],
        applied_deltas: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        *,
        count_critic_ns: bool = False,
    ) -> None:
        """Apply batched 5-step Quintic NS updates for queued 2D / ungrouped 1x1 Conv2d matrices."""
        for (m_r, m_c, has_s_b), bucket in pending_dense.items():
            if (not count_critic_ns) or min(m_r, m_c) >= 2:
                self.ns_2d_activations += len(bucket)
                self.ns_5step_calls += len(bucket)
                if count_critic_ns:
                    self.critic_ns_calls += len(bucket)
            else:
                self.fast_path_updates += len(bucket)
            mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
            s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
            orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
            for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                p_i.add_(delta_i)
                if trk_i and applied_deltas is not None:
                    applied_deltas.append((p_i, delta_i))

    @torch.no_grad()
    def _step_generator_free_critic(self, closure: Any = None) -> Any:
        """Zero-sync, 1e-37-safe FP32 5-step 2D Critic NS strictly for generator-free tasks."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is not None and (p.grad.is_sparse or p.grad.is_complex()):
                    raise ValueError(f"{self.family} requires dense real gradients")

        try:
            rec = self._active_record()
            step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
            s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            p_dec = 0.9995 if s_therm >= 0.5 else 1.0
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}

            for grp in self.param_groups:
                b1, b2 = grp["betas"]
                s_base_gf = float(self.smoothing) if self.smoothing > 0.0 else 0.0
                b1_eff = float(b1)
                b2_eff = min(max(float(b2), 0.95), 0.999)
                rate_p = float(grp["lr"]) * step_scale

                for p in grp["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    st = self.state[p]
                    step_t = int(st.get("step", 0)) + 1
                    st["step"] = step_t

                    gf = grad.float()
                    finite = torch.isfinite(gf).all()
                    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                    g_max = gf_safe.abs().amax()
                    g_active = finite & (g_max > 1e-38)

                    is_conv = self._is_spatial_conv(p)
                    r_d = c_d = m_dim = 1
                    if is_conv:
                        factor, sv_denom_sqrt = self._cached_conv_metrics(p)
                    elif p.ndim >= 2:
                        r_d, c_d, m_dim, factor, sv_denom_sqrt = self._cached_dense_metrics(p)
                    else:
                        factor, sv_denom_sqrt = 1.0, 1.0

                    safe_scale = torch.where(g_active, g_max, torch.ones_like(g_max))
                    g_unit_scaled = torch.where(g_active, gf_safe / safe_scale, torch.zeros_like(gf_safe))
                    scaled_norm = g_unit_scaled.norm()
                    cur_sv_rms = (scaled_norm * safe_scale) / sv_denom_sqrt

                    if "peak_rms" not in st:
                        st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    peak_new = _huber_peak_t(st["peak_rms"], cur_sv_rms, finite & torch.isfinite(cur_sv_rms), p_dec)
                    st["peak_rms"].copy_(peak_new)

                    has_smooth = s_base_gf > 0.0
                    if has_smooth:
                        s_eff_t = torch.minimum((_REL_SMOOTH_GF_CRITIC * s_base_gf) * peak_new, _REL_SMOOTH_MAX * peak_new)
                    else:
                        s_eff_t = gf_safe.new_zeros(())

                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    if (b1_eff > 0.0) or is_scalar_1d:
                        b1_use = max(b1_eff, 0.80) if is_scalar_1d else b1_eff
                        m_hat, m_norm, raw_coh, sc = _update_scale_safe_moments(
                            st, p, gf_safe, g_max, finite, g_active, step_t, b1_use, b2_eff
                        )
                        coh = raw_coh.clamp(0.35, 1.0) if is_scalar_1d else torch.ones_like(m_norm)
                        direc_u = torch.where(g_active, m_hat, torch.zeros_like(m_hat))
                        d_norm_u = m_norm
                        d_scale = sc
                        direc_2d = torch.where(g_active, (m_hat * sc) if has_smooth else m_hat, torch.zeros_like(m_hat))
                    else:
                        direc_u = g_unit_scaled
                        d_norm_u = scaled_norm
                        d_scale = safe_scale
                        direc_2d = torch.where(g_active, gf_safe if has_smooth else g_unit_scaled, torch.zeros_like(gf_safe))
                        coh = gf_safe.new_ones(())

                    if rate_p <= 0.0:
                        continue

                    if is_conv or p.ndim < 2:
                        s_scaled = (s_eff_t * sv_denom_sqrt) / d_scale.clamp_min(1e-38)
                        denom_u = torch.hypot(d_norm_u, s_scaled).clamp_min(1e-30) if has_smooth else d_norm_u.clamp_min(1e-30)
                        unit_u = torch.where(g_active & (d_norm_u > 1e-30), direc_u / denom_u, torch.zeros_like(direc_u))
                        p.add_((unit_u * ((-rate_p * factor) * coh)).to(p.dtype))
                        self.fast_path_updates += 1
                    else:
                        mat = direc_2d.reshape(r_d, c_d)
                        step_mult = (-rate_p * factor) * coh
                        needs_t = r_d > c_d
                        mat_canon = mat.T if needs_t else mat
                        m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                        pending_dense.setdefault(m_key, []).append((p, mat_canon, step_mult, s_eff_t, False, needs_t))

            self._flush_pending_dense(pending_dense, count_critic_ns=True)
            self.last_step_gain = 1.0
            self.last_probe_consumed = False
        finally:
            self.abort_pending()

        if hasattr(self, "record") and self.record is not None:
            self.record.record_step(self)
        return loss

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if getattr(self, "critic", None) is not None and self._generator_free:
            return self._step_generator_free_critic(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.direction_blend and (closure is not None or (self._protected is None and not self._protected_single)):
            self.abort_pending()
            raise ValueError("direction_blend step requires explicit protected-loss backward")

        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.grad.is_complex():
                    raise ValueError(f"{self.family} requires dense real gradients")
                if (
                    grp["algorithm"] == "unified_prior"
                    and grp.get("sampled_rows_required", True)
                    and p not in self._sampled_rows
                ):
                    raise ValueError("row-normalized prior updates require actual sampled rows")

        if self.direction_blend and self._protected is not None:
            all_params = self._parameters()
            originals = [p.detach().clone() for p in all_params]
        else:
            all_params = None
            originals = None

        try:
            rec = self._active_record()
            step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
            s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            is_simp = bool(getattr(rec, "_simplicial_step", False)) if rec is not None else False
            is_cond_fiber = bool(getattr(rec, "t1c_cond_initialized", False)) if rec is not None else False
            is_critic_opt = getattr(self, "critic", None) is not None
            is_two_obj_cond = (
                is_cond_fiber
                and self.direction_blend
                and self._protected is not None
                and int(self._protected.shape[0]) == 2
            )
            p_dec = 0.9995 if (is_two_obj_cond and s_therm >= 0.5) else 1.0

            # Scale-homogeneous origin collapse guard relative to tracked data scale (never crushes 1e-5x targets):
            data_scale_ref = max(1e-12, float(getattr(rec, "ema_data_scale", 1.0))) if rec is not None else 1.0
            od_thresh = 0.05 * data_scale_ref
            origin_damping = 1.0
            origin_damping_t: Optional[torch.Tensor] = None
            if self._origin_norm_t is not None:
                origin_damping_t = (self._origin_norm_t / od_thresh).clamp_max(1.0)
            elif self._origin_norm is not None and self._origin_norm < od_thresh:
                origin_damping = min(1.0, self._origin_norm / od_thresh)

            if not is_critic_opt:
                self._update_period2_flip_feedback(s_therm=s_therm)

            ema_fg = float(self._ema_func_gain)
            stiff_gate_raw = self.current_stiff_gate()
            pre_gain_base = 1.0 if ema_fg >= 0.75 else ema_fg

            # Bandpass on-manifold two-player critic coupling (zero extra syncs):
            d_couple_f: float = 1.0
            d_couple_g_pre: float = 1.0
            d_couple_pending: bool = False
            if is_critic_opt and rec is not None:
                g_fg = float(getattr(rec, "_g_func_gain", 1.0))
                if g_fg < 0.75:
                    d_couple_g_pre = max(0.12, g_fg)
                    if (
                        getattr(rec, "_ema_margin_t", None) is not None
                        or getattr(rec, "_margin_init_t", None) is not None
                    ):
                        d_couple_pending = True
                    elif getattr(rec, "margin_initialized", False):
                        ema_m = float(getattr(rec, "ema_margin", 0.0))
                        m_gate = _smoothstep01((ema_m - 0.28) / 0.22) * _smoothstep01((1.40 - ema_m) / 0.30)
                        d_couple_f = (1.0 - m_gate) + m_gate * d_couple_g_pre

            # Strictly 0 critic secant probes; generator secant trust band active on non-simplicial/non-2-loss:
            gen_probe_active = (
                (not is_critic_opt)
                and rec is not None
                and getattr(rec, "_g_probe_cache", None) is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )

            # Mode-matched eval() y_pre baseline before updating generator weights (lazy 1-step check -> 0 steady-state overhead on standard MLPs & BN DCGAN):
            y_pre_eval_override: Optional[torch.Tensor] = None
            if gen_probe_active:
                _pc = getattr(rec, "_g_probe_cache", None)
                if _pc is not None:
                    _g_mod_p, _p_args_p, _p_kwargs_p, _y_pre_p = _pc
                    if not getattr(_g_mod_p, "_unified_v15_gen_has_bn", False):
                        _needs_ev = getattr(_g_mod_p, "_unified_v15_needs_eval_pre", None)
                        if _needs_ev is None:
                            _y_ev = _eval_probe_forward(_g_mod_p, _p_args_p, _p_kwargs_p, rec)
                            if (
                                _y_ev is not None
                                and _y_ev.shape == _y_pre_p.shape
                                and bool(((_y_ev - _y_pre_p).abs().amax() > 1e-5).item())
                            ):
                                _g_mod_p._unified_v15_needs_eval_pre = True
                                y_pre_eval_override = _y_ev
                            else:
                                _g_mod_p._unified_v15_needs_eval_pre = False
                        elif _needs_ev:
                            y_pre_eval_override = _eval_probe_forward(_g_mod_p, _p_args_p, _p_kwargs_p, rec)

            pyth_param_mults: Dict[int, float] = (
                self._compute_two_obj_pyth_scales(is_cond_fiber=is_cond_fiber)
                if is_two_obj_cond
                else {}
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            fast_secant_deltas: List[Tuple[torch.Tensor, float]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}
            pending_s6: Dict[
                Tuple[int, int],
                List[Tuple[torch.Tensor, int, int, float, bool, torch.Tensor]],
            ] = {}

            fast_grads: List[torch.Tensor] = []
            fast_meta: List[Tuple[Any, ...]] = []
            dense2d_grads: List[torch.Tensor] = []
            dense2d_meta: List[Tuple[Any, ...]] = []
            s6_norm_params: List[torch.Tensor] = []
            slow_queue: List[Tuple[Any, ...]] = []
            prior_queue: List[Tuple[Any, ...]] = []

            pre_gain_applied: float | torch.Tensor = 1.0
            has_gen_v4_tensor = isinstance(self._v4_g_scale, torch.Tensor) or isinstance(
                self._v4_wakeup_boost, torch.Tensor
            )
            if isinstance(self._v4_wakeup_boost, torch.Tensor):
                pre_g_tensor: Optional[torch.Tensor] = (pre_gain_base * self._v4_wakeup_boost).clamp_max(1.0)
                pre_g_float = 1.0
            else:
                pre_g_tensor = None
                pre_g_float = min(1.0, pre_gain_base * float(self._v4_wakeup_boost))

            coh_vals: List[torch.Tensor] = []
            fast_coh_count = 0

            for grp in self.param_groups:
                role = grp["role"]
                player = self._player(role)
                is_critic_role = player == "critic"
                stiff_gate = stiff_gate_raw if player == "generator" else 0.0
                b1, b2 = grp["betas"]
                fg_damp = _smoothstep01((0.30 - ema_fg) / 0.12) if player == "generator" else 0.0
                s_base = self.smoothing * (1.0 - max(stiff_gate, fg_damp))
                c_rel = float(s_base) if s_base > 0.0 else 0.0
                s_base_prior = float(self.smoothing) if (self.smoothing > 0.0 and self._v4_prior_scale >= 1.0) else 0.0

                if player == "prior":
                    b1_eff = max(float(b1), 0.75) if self._v4_prior_scale < 1.0 else float(b1)
                elif stiff_gate > 0.0:
                    b1_eff = max(float(b1), 0.75 * stiff_gate)
                else:
                    b1_eff = float(b1)
                b2_eff = min(max(float(b2), 0.95), 0.999)

                if player == "generator":
                    pre_gain_applied = pre_g_tensor if pre_g_tensor is not None else pre_g_float
                    lr_no_v4 = float(grp["lr"]) * step_scale
                    is_gen_v4 = True
                elif is_critic_role:
                    pre_gain_applied = 1.0
                    lr_no_v4 = float(grp["lr"]) * step_scale
                    is_gen_v4 = False
                else:
                    lr_no_v4 = float(grp["lr"]) * step_scale * self._v4_prior_scale
                    is_gen_v4 = False

                for p in grp["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    st = self.state[p]
                    step_t = int(st.get("step", 0)) + 1
                    st["step"] = step_t

                    if player == "prior":
                        base_lr_prior = lr_no_v4 * pyth_param_mults.get(id(p), 1.0)
                        if base_lr_prior > 0.0:
                            prior_queue.append(
                                (p, grad, st, base_lr_prior, b1_eff, s_base_prior, bool(grp.get("sampled_rows_required", True)))
                            )
                        continue

                    pyth_m = pyth_param_mults.get(id(p), 1.0)
                    lr_p_no_v4 = lr_no_v4 * pyth_m
                    apply_od = role == "generator" and step_t > 80
                    track_secant = gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids)
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    is_conv = self._is_spatial_conv(p)

                    if (
                        b1_eff == 0.0
                        and stiff_gate == 0.0
                        and not is_scalar_1d
                        and p.dtype == torch.float32
                        and grad.dtype == torch.float32
                    ):
                        if is_conv:
                            factor, sv_denom_sqrt = self._cached_conv_metrics(p)
                            fast_grads.append(grad)
                            fast_meta.append(
                                (
                                    p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                                    c_rel, factor, sv_denom_sqrt, stiff_gate, track_secant,
                                    is_critic_role, factor, 1, 1, 1, -1,
                                )
                            )
                        elif p.ndim >= 2:
                            r_d, c_d, m_dim, aspect_factor, sv_denom_sqrt = self._cached_dense_metrics(p)
                            if m_dim == 1:
                                fast_grads.append(grad)
                                fast_meta.append(
                                    (
                                        p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                                        c_rel, aspect_factor, sv_denom_sqrt, stiff_gate, track_secant,
                                        is_critic_role, aspect_factor, r_d, c_d, m_dim, -1,
                                    )
                                )
                            elif is_critic_role:
                                factor_d = aspect_factor * math.sqrt(float(m_dim))
                                s6_norm_idx = -1
                                if s_therm > 0.0:
                                    s6_norm_idx = len(s6_norm_params)
                                    s6_norm_params.append(p)
                                fast_grads.append(grad)
                                fast_meta.append(
                                    (
                                        p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                                        c_rel, factor_d, sv_denom_sqrt, stiff_gate, track_secant,
                                        True, aspect_factor, r_d, c_d, m_dim, s6_norm_idx,
                                    )
                                )
                            else:
                                dense2d_grads.append(grad)
                                dense2d_meta.append(
                                    (
                                        p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                                        c_rel, aspect_factor, sv_denom_sqrt, track_secant, r_d, c_d,
                                    )
                                )
                        else:
                            fast_grads.append(grad)
                            fast_meta.append(
                                (
                                    p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                                    c_rel, 1.0, 1.0, stiff_gate, track_secant,
                                    is_critic_role, 1.0, 1, 1, 1, -1,
                                )
                            )
                    else:
                        slow_queue.append(
                            (
                                p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b1_eff, b2_eff,
                                c_rel, stiff_gate, is_critic_role, track_secant,
                            )
                        )

            # 1. Single Batched Foreach-Norm + Zero-Extra-Sync Scalar & S6-Weight-Norm Packing:
            all_norm_tensors = fast_grads + dense2d_grads + s6_norm_params
            n_fast = len(fast_grads)
            n_dense2d = len(dense2d_grads)
            all_norm_vals: List[float] = []
            v4_g_scale_f: Optional[float] = None if isinstance(self._v4_g_scale, torch.Tensor) else float(self._v4_g_scale)
            v4_pre_g_f: Optional[float] = None if pre_g_tensor is not None else pre_g_float
            od_f: Optional[float] = None if origin_damping_t is not None else origin_damping
            g_pub_t_rec: Optional[torch.Tensor] = (
                getattr(rec, "_v15_g_peak_ref_t", None) if (is_critic_opt and rec is not None) else None
            )
            g_pub_f: float = (
                float(getattr(rec, "_v15_g_peak_ref", 0.0)) if (is_critic_opt and rec is not None) else 0.0
            )
            g_pub_t_packed = False

            if all_norm_tensors:
                norms = torch._foreach_norm(all_norm_tensors)
                ref_dev = all_norm_tensors[0].device
                extra_t: List[torch.Tensor] = []
                idx_gs = idx_pg = idx_od = idx_gpub = -1
                dc_ema_keys: List[str] = []
                idx_dc_ema = -1
                if isinstance(self._v4_g_scale, torch.Tensor):
                    idx_gs = len(norms) + len(extra_t)
                    extra_t.append(self._v4_g_scale.to(device=ref_dev, dtype=torch.float32))
                if pre_g_tensor is not None:
                    idx_pg = len(norms) + len(extra_t)
                    extra_t.append(pre_g_tensor.to(device=ref_dev, dtype=torch.float32))
                if origin_damping_t is not None:
                    idx_od = len(norms) + len(extra_t)
                    extra_t.append(origin_damping_t.to(device=ref_dev, dtype=torch.float32))
                if g_pub_t_rec is not None and g_pub_t_rec.device == ref_dev:
                    idx_gpub = len(norms) + len(extra_t)
                    extra_t.append(g_pub_t_rec.to(device=ref_dev, dtype=torch.float32))
                    g_pub_t_packed = True
                if d_couple_pending and rec is not None and hasattr(rec, "_take_pending_ema_tensors"):
                    dc_keys_cand, dc_tens_cand = rec._take_pending_ema_tensors()
                    if dc_tens_cand and all(t.device == ref_dev for t in dc_tens_cand):
                        idx_dc_ema = len(norms) + len(extra_t)
                        dc_ema_keys = dc_keys_cand
                        for t_e in dc_tens_cand:
                            extra_t.append(t_e.to(device=ref_dev, dtype=torch.float32))
                packed = torch.stack((*norms, *extra_t)).tolist()
                all_norm_vals = packed[: len(norms)]
                if idx_gs >= 0:
                    v4_g_scale_f = float(packed[idx_gs])
                if idx_pg >= 0:
                    v4_pre_g_f = float(packed[idx_pg])
                    pre_gain_applied = v4_pre_g_f
                if idx_od >= 0:
                    od_f = float(packed[idx_od])
                if idx_gpub >= 0:
                    gp_val = float(packed[idx_gpub])
                    if math.isfinite(gp_val) and gp_val > 0.0:
                        g_pub_f = gp_val
                if idx_dc_ema >= 0 and rec is not None:
                    rec._apply_flushed_ema_values(
                        dc_ema_keys,
                        [float(v) for v in packed[idx_dc_ema : idx_dc_ema + len(dc_ema_keys)]],
                    )
                    if getattr(rec, "margin_initialized", False):
                        ema_m = float(getattr(rec, "ema_margin", 0.0))
                        m_gate = _smoothstep01((ema_m - 0.28) / 0.22) * _smoothstep01((1.40 - ema_m) / 0.30)
                        d_couple_f = (1.0 - m_gate) + m_gate * d_couple_g_pre
                    d_couple_pending = False

            if d_couple_pending and rec is not None:
                if getattr(rec, "margin_initialized", False):
                    ema_m = float(getattr(rec, "ema_margin", 0.0))
                    m_gate = _smoothstep01((ema_m - 0.28) / 0.22) * _smoothstep01((1.40 - ema_m) / 0.30)
                    d_couple_f = (1.0 - m_gate) + m_gate * d_couple_g_pre
                d_couple_pending = False

            gen_v4_mult_f = (
                (v4_g_scale_f * v4_pre_g_f)
                if (v4_g_scale_f is not None and v4_pre_g_f is not None)
                else None
            )
            gen_v4_mult_t: Optional[torch.Tensor] = None
            if gen_v4_mult_f is None and has_gen_v4_tensor:
                g_s_val: float | torch.Tensor = self._v4_g_scale
                p_g_val: float | torch.Tensor = pre_g_tensor if pre_g_tensor is not None else pre_g_float
                gen_v4_mult_t = g_s_val * p_g_val  # type: ignore[assignment]

            # 1a. Scale-homogeneous Huber peak update & two-player mean-Frobenius reference S_mean (0 extra syncs!):
            fast_ready: List[Tuple[Any, ...]] = []
            dense2d_ready: List[Tuple[Any, ...]] = []
            peak_frob_vals_f: List[float] = []

            for idx_f, item_f in enumerate(fast_meta):
                (
                    p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                    s_base_p, factor, sv_denom_sqrt, stiff_gate, track_sec,
                    is_critic_role, aspect_factor, r_d, c_d, m_dim, s6_norm_idx,
                ) = item_f
                g_norm = all_norm_vals[idx_f]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append(
                        (
                            p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, 0.0, b2_eff,
                            s_base_p, stiff_gate, is_critic_role, track_sec,
                        )
                    )
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                if "peak_rms" in st:
                    peak_prev = max(peak_prev, float(st["peak_rms"].item()))
                    del st["peak_rms"]
                peak_new = _huber_peak_f(peak_prev, cur_sv, p_dec)
                st["peak_rms_val"] = peak_new
                peak_frob_vals_f.append(peak_new * sv_denom_sqrt)
                fast_ready.append((item_f, g_norm, peak_new))

            for idx_d, item_d in enumerate(dense2d_meta):
                (
                    p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                    s_base_p, factor, sv_denom_sqrt, track_sec, r_d, c_d,
                ) = item_d
                g_norm = all_norm_vals[n_fast + idx_d]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append(
                        (
                            p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, 0.0, b2_eff,
                            s_base_p, 0.0, False, track_sec,
                        )
                    )
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                if "peak_rms" in st:
                    peak_prev = max(peak_prev, float(st["peak_rms"].item()))
                    del st["peak_rms"]
                peak_new = _huber_peak_f(peak_prev, cur_sv, p_dec)
                st["peak_rms_val"] = peak_new
                peak_frob_vals_f.append(peak_new * sv_denom_sqrt)
                dense2d_ready.append((item_d, g_norm, peak_new))

            opt_mean_f = (sum(peak_frob_vals_f) / len(peak_frob_vals_f)) if peak_frob_vals_f else 0.0
            if (not is_critic_opt) and opt_mean_f > 0.0:
                self._g_peak_ref = opt_mean_f
                self._g_peak_ref_t = None
                if rec is not None:
                    rec._v15_g_peak_ref = opt_mean_f
                    rec._v15_g_peak_ref_t = None
            if is_critic_opt:
                if opt_mean_f > 0.0 and (not g_pub_t_packed) and g_pub_t_rec is not None:
                    gp_val = float(g_pub_t_rec.item())
                    if math.isfinite(gp_val) and gp_val > 0.0:
                        g_pub_f = gp_val
                if g_pub_f > 0.0:
                    self._g_peak_ref = g_pub_f
                    self._g_peak_ref_t = None
                s_ref_f = max(opt_mean_f, g_pub_f) if opt_mean_f > 0.0 else 0.0
            else:
                s_ref_f = opt_mean_f
            g_ref_f = _REL_SMOOTH_PRIOR_REF * s_ref_f

            # Zero-State, Zero-Copy Foreach Frobenius-MP Fast Path (Spatial Conv2d, Neural-G Critic D, 1D, m=1):
            for (
                (
                    p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                    s_base_p, factor, sv_denom_sqrt, stiff_gate, track_sec,
                    is_critic_role, aspect_factor, r_d, c_d, m_dim, s6_norm_idx,
                ),
                g_norm,
                peak_new,
            ) in fast_ready:
                rate_p = (
                    lr_p_no_v4
                    * (gen_v4_mult_f if is_gen_v4 else (d_couple_f if is_critic_role else 1.0))
                    * (od_f if apply_od else 1.0)
                )
                if rate_p <= 0.0:
                    continue
                if s_base_p > 0.0:
                    rel_mult = _REL_SMOOTH_CRITIC if is_critic_role else _REL_SMOOTH_GEN
                    ref_p = s_ref_f if s_ref_f > 0.0 else (peak_new * sv_denom_sqrt)
                    s_eff = min((rel_mult * s_base_p) * ref_p, _REL_SMOOTH_MAX * peak_new)
                    denom = math.hypot(g_norm, s_eff * sv_denom_sqrt)
                else:
                    denom = g_norm
                if denom <= 1e-30:
                    continue
                alpha_i = -(rate_p * factor) / denom
                p.add_(g, alpha=alpha_i)
                self.fast_path_updates += 1
                if s6_norm_idx >= 0:
                    w_norm_before = all_norm_vals[n_fast + n_dense2d + s6_norm_idx]
                    step_bound = abs(alpha_i) * g_norm
                    if math.isfinite(w_norm_before) and (w_norm_before + step_bound) <= _CRITIC_SCHATTEN6_TAU * aspect_factor:
                        self.s6_skipped_deadband += 1
                    else:
                        _enqueue_s6(pending_s6, p, r_d, c_d, aspect_factor, g.new_full((), rate_p * s_therm))
                if track_sec:
                    fast_secant_deltas.append((p, alpha_i))
                    fast_coh_count += 1

            # 1b. Non-stiff Generator 2D Dense & Ungrouped 1x1 Conv2d Matrices (min(r, c) >= 2 -> batched Quintic NS):
            for (
                (
                    p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff,
                    s_base_p, factor, sv_denom_sqrt, track_sec, r_d, c_d,
                ),
                g_norm,
                peak_new,
            ) in dense2d_ready:
                rate_p = lr_p_no_v4 * (gen_v4_mult_f if is_gen_v4 else 1.0) * (od_f if apply_od else 1.0)
                if rate_p <= 0.0:
                    continue
                has_s = s_base_p > 0.0
                if has_s:
                    ref_d = s_ref_f if s_ref_f > 0.0 else (peak_new * sv_denom_sqrt)
                    s_eff = min((_REL_SMOOTH_GEN * s_base_p) * ref_d, _REL_SMOOTH_MAX * peak_new)
                else:
                    s_eff = 0.0
                mat = g.reshape(r_d, c_d)
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                if track_sec:
                    fast_coh_count += 1
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_s)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, g.new_full((), -rate_p * factor), g.new_full((), s_eff), track_sec, needs_t)
                )

            # 2. Zero-Sync Slow Path (stiff 2D NS on G, scalar 1D, extreme scales <1e-18 / >1e18 / NaN / non-FP32):
            slow_s_ref_t: Optional[torch.Tensor] = None
            slow_g_ref_t: Optional[torch.Tensor] = None
            slow_peaks_pre: List[Optional[torch.Tensor]] = [None] * len(slow_queue)
            if slow_queue and opt_mean_f <= 0.0:
                pf_slow: List[torch.Tensor] = []
                for idx_s, s_item in enumerate(slow_queue):
                    p_s, g_s = s_item[0], s_item[1]
                    st_s = self.state[p_s]
                    gf_s = g_s.float()
                    fin_s = torch.isfinite(gf_s).all()
                    gfs_safe = torch.where(fin_s, gf_s, torch.zeros_like(gf_s))
                    gm_s = gfs_safe.abs().amax()
                    ga_s = fin_s & (gm_s > 1e-38)
                    sc_s = torch.where(ga_s, gm_s, torch.ones_like(gm_s))
                    gu_s = torch.where(ga_s, gfs_safe / sc_s, torch.zeros_like(gfs_safe))
                    if self._is_spatial_conv(p_s):
                        _, sv_s = self._cached_conv_metrics(p_s)
                    elif p_s.ndim >= 2:
                        _, _, _, _, sv_s = self._cached_dense_metrics(p_s)
                    else:
                        sv_s = 1.0
                    csv_s = (gu_s.norm() * sc_s) / sv_s
                    if "peak_rms" not in st_s:
                        st_s["peak_rms"] = gfs_safe.new_full((), float(st_s.get("peak_rms_val", 0.0)))
                    pk_s = _huber_peak_t(st_s["peak_rms"], csv_s, fin_s & torch.isfinite(csv_s), p_dec)
                    st_s["peak_rms"].copy_(pk_s)
                    slow_peaks_pre[idx_s] = pk_s
                    pf_slow.append(pk_s * sv_s)
                if pf_slow:
                    slow_opt_mean_t = _scale_safe_mean_tensors(pf_slow)
                    if not is_critic_opt:
                        self._g_peak_ref = 0.0
                        self._g_peak_ref_t = slow_opt_mean_t.detach().clone()
                        if rec is not None:
                            rec._v15_g_peak_ref = 0.0
                            rec._v15_g_peak_ref_t = self._g_peak_ref_t
                        slow_s_ref_t = slow_opt_mean_t
                    else:
                        if g_pub_t_rec is not None:
                            g_pub_d = g_pub_t_rec.to(device=slow_opt_mean_t.device, dtype=slow_opt_mean_t.dtype)
                            slow_s_ref_t = torch.maximum(slow_opt_mean_t, g_pub_d)
                            self._g_peak_ref = 0.0
                            self._g_peak_ref_t = g_pub_d.detach().clone()
                        elif g_pub_f > 0.0:
                            slow_s_ref_t = torch.maximum(slow_opt_mean_t, slow_opt_mean_t.new_full((), g_pub_f))
                        else:
                            slow_s_ref_t = slow_opt_mean_t
                    slow_g_ref_t = _REL_SMOOTH_PRIOR_REF * slow_s_ref_t

            for idx_s, (
                p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b1_eff, b2_eff,
                s_base_p, stiff_gate, is_critic_role, track_sec,
            ) in enumerate(slow_queue):
                if is_gen_v4 and gen_v4_mult_f is None and gen_v4_mult_t is not None:
                    rate_val: float | torch.Tensor = lr_p_no_v4 * gen_v4_mult_t
                    if apply_od:
                        rate_val = rate_val * (origin_damping_t if origin_damping_t is not None else origin_damping)
                    rate_pos = lr_p_no_v4 > 0.0
                else:
                    base_r = lr_p_no_v4 * (gen_v4_mult_f if is_gen_v4 else (d_couple_f if is_critic_role else 1.0))
                    if apply_od and origin_damping_t is not None and od_f is None:
                        rate_val = base_r * origin_damping_t
                        rate_pos = base_r > 0.0
                    else:
                        rate_val = base_r * ((od_f if od_f is not None else origin_damping) if apply_od else 1.0)
                        rate_pos = float(rate_val) > 0.0
                self._apply_slow_param_zerosync(
                    p,
                    g,
                    step_t=step_t,
                    rate_p=rate_val,
                    rate_positive=rate_pos,
                    b1_eff=b1_eff,
                    b2_eff=b2_eff,
                    s_base=s_base_p,
                    peak_precomputed=slow_peaks_pre[idx_s],
                    scale_ref_f=s_ref_f,
                    scale_ref_t=slow_s_ref_t,
                    s_therm=s_therm,
                    p_dec=p_dec,
                    stiff_gate=stiff_gate,
                    is_critic_role=is_critic_role,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_dense,
                    pending_s6=pending_s6,
                )

            # 2b. Zero-sync scale-homogeneous Prior updates sharing the optimizer G_ref:
            for p_pr, g_pr, st_pr, base_lr_pr, b1_pr, s_base_pr, sampled_req_pr in prior_queue:
                self._apply_prior_param(
                    p_pr,
                    g_pr,
                    st=st_pr,
                    base_lr_prior=base_lr_pr,
                    b1_eff=b1_pr,
                    s_base_prior=s_base_pr,
                    g_ref_f=g_ref_f,
                    g_ref_t=slow_g_ref_t,
                    p_dec=p_dec,
                    sampled_req=sampled_req_pr,
                )

            # 3. Batched 5-step fused Z=[X,Y] Quintic NS strictly for Generator 2D / 1x1 Conv2d matrices:
            self._flush_pending_dense(pending_dense, applied_deltas=applied_deltas, count_critic_ns=False)

            # 4. Law 4 Generator-Only C^1 Secant Trust Band with Verified Iterative Backtracking (0 critic probes!):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            pre_gain_f = 1.0
            if gen_probe_active and (fast_secant_deltas or applied_deltas):
                probe_cache = getattr(rec, "_g_probe_cache", None)
                rec._g_probe_cache = None
                if probe_cache is not None:
                    g_mod, p_args, p_kwargs, y_pre_cached = probe_cache
                    y_pre = (
                        y_pre_eval_override
                        if (y_pre_eval_override is not None and y_pre_eval_override.shape == y_pre_cached.shape)
                        else y_pre_cached
                    )
                    y_post = _eval_probe_forward(g_mod, p_args, p_kwargs, rec)
                    if y_post is not None and y_post.shape == y_pre.shape:
                        probe_ran = True
                        out_dim = max(1, int(y_pre[0].numel()))
                        delta_rms_t = (y_post - y_pre).square().mean().sqrt().double()
                        total_coh_n = fast_coh_count + len(coh_vals)
                        if coh_vals:
                            slow_coh_sum_t = torch.stack(coh_vals).sum().double()
                            mean_coh_t = (float(fast_coh_count) + slow_coh_sum_t) / float(max(1, total_coh_n))
                        else:
                            mean_coh_t = delta_rms_t.new_full((), 1.0)
                        has_pg_t = isinstance(pre_gain_applied, torch.Tensor)
                        ema_keys, ema_tens = (
                            rec._take_pending_ema_tensors()
                            if hasattr(rec, "_take_pending_ema_tensors")
                            else ([], [])
                        )
                        batch_g: List[torch.Tensor] = [delta_rms_t, mean_coh_t]
                        if has_pg_t:
                            batch_g.append(pre_gain_applied.to(device=delta_rms_t.device, dtype=torch.float64))
                        can_pack_ema = bool(ema_tens) and all(t.device == delta_rms_t.device for t in ema_tens)
                        if can_pack_ema:
                            batch_g.extend(ema_tens)
                        vals_g = torch.stack(batch_g).tolist()
                        delta_rms = float(vals_g[0])
                        mean_coh = float(vals_g[1])
                        off_g = 2
                        if has_pg_t:
                            pre_gain_f = float(vals_g[off_g])
                            off_g += 1
                        else:
                            pre_gain_f = float(pre_gain_applied)
                        if can_pack_ema:
                            rec._apply_flushed_ema_values(ema_keys, vals_g[off_g:])
                        data_scale = float(getattr(rec, "ema_data_scale", 1.0))
                        coord_scale = max(1e-12, data_scale / math.sqrt(float(out_dim)))
                        delta_rel = delta_rms / coord_scale
                        push_res = float(getattr(rec, "ema_push_res", 0.0))
                        delta_g_max = (0.045 + 0.16 * min(1.5, push_res)) * step_scale
                        if (not is_simp) and push_res > 0.18 and s_therm > 0.10:
                            off_gate = _smoothstep01((push_res - 0.18) / 0.25)
                            delta_g_min = 0.012 * off_gate * step_scale
                        else:
                            delta_g_min = 0.0
                        alpha_target = _band_scale(delta_rel, delta_g_min, delta_g_max, alpha_max=48.0)
                        coh_eff = max(0.05, mean_coh)
                        if math.isfinite(delta_rel) and delta_rel > 1e-12:
                            free_ratio = (delta_g_max / delta_rel) * pre_gain_f * coh_eff
                        step_active_probe = math.isfinite(delta_rel) and delta_rel >= 0.012

                        # Verified iterative backtracking when contracting strongly (<0.80) or amplifying (>1.0):
                        alpha_cur = 1.0
                        if alpha_target != 1.0:
                            adj = alpha_target - alpha_cur
                            for p_i, alpha_i in fast_secant_deltas:
                                if p_i.grad is not None:
                                    p_i.add_(p_i.grad, alpha=adj * alpha_i)
                            for p_i, d_i in applied_deltas:
                                p_i.add_(d_i, alpha=adj)
                            alpha_cur = alpha_target
                            if alpha_target < 0.80 or alpha_target > 1.0:
                                for _ in range(4):
                                    y_chk = _eval_probe_forward(g_mod, p_args, p_kwargs, rec)
                                    if y_chk is None or y_chk.shape != y_pre.shape:
                                        break
                                    delta_chk = float((y_chk - y_pre).square().mean().sqrt().double().item()) / coord_scale
                                    if not math.isfinite(delta_chk):
                                        alpha_next = max(1e-4, alpha_cur * 0.25)
                                    elif delta_g_max > 0.0 and delta_chk > 1.05 * delta_g_max:
                                        shrink = min(0.85, (0.95 * delta_g_max) / delta_chk)
                                        alpha_next = max(1e-4, alpha_cur * shrink)
                                    else:
                                        break
                                    if abs(alpha_next - alpha_cur) <= 1e-9:
                                        break
                                    adj_bt = alpha_next - alpha_cur
                                    for p_i, alpha_i in fast_secant_deltas:
                                        if p_i.grad is not None:
                                            p_i.add_(p_i.grad, alpha=adj_bt * alpha_i)
                                    for p_i, d_i in applied_deltas:
                                        p_i.add_(d_i, alpha=adj_bt)
                                    alpha_cur = alpha_next
                        alpha_gain = alpha_cur
                if probe_ran:
                    if alpha_gain < 1.0:
                        raw_fg = pre_gain_f * alpha_gain
                        if stiff_gate_raw > 0.0 and s_therm >= 0.5:
                            raw_fg = min(raw_fg, free_ratio)
                        target_fg = max(0.05, min(1.0, raw_fg))
                        self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * target_fg)
                    elif stiff_gate_raw > 0.0 and s_therm >= 0.5 and free_ratio < self._ema_func_gain:
                        target_fg = max(0.05, min(1.0, free_ratio))
                        self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * target_fg)
                    elif s_therm >= 0.5 and step_active_probe and free_ratio > 1.5:
                        target_fg = max(0.05, min(1.0, free_ratio))
                        if target_fg > self._ema_func_gain:
                            self._ema_func_gain = float(0.999 * self._ema_func_gain + 0.001 * target_fg)
                if rec is not None:
                    g_stiff_post = self.current_stiff_gate()
                    rec._g_func_gain = float((1.0 - g_stiff_post) * self._ema_func_gain + g_stiff_post)
            elif rec is not None and not is_critic_opt:
                rec._g_probe_cache = None

            # 4b. Zero-state, zero-SVD exact-deadband batched Schatten-6 spectral ceiling on 2D / 1x1 Conv2d critic weights:
            for bucket_s6 in pending_s6.values():
                w_list = []
                r_list = []
                for p_i, r_i, c_i, fac_i, needs_t, s6r_i in bucket_s6:
                    w_norm_m = p_i.float().reshape(r_i, c_i) / fac_i
                    w_list.append(w_norm_m.T if needs_t else w_norm_m)
                    r_list.append(s6r_i)
                s6_batch = _schatten6_spectral_decay_batched(
                    torch.stack(w_list, dim=0),
                    torch.stack(r_list, dim=0),
                )
                for idx_b, (p_i, _, _, fac_i, needs_t, _) in enumerate(bucket_s6):
                    disp_s6 = s6_batch[idx_b] * fac_i
                    if needs_t:
                        disp_s6 = disp_s6.T
                    p_i.add_(disp_s6.reshape_as(p_i).to(p_i.dtype))

            # 5. Strictly Zero-pinv Analytic Non-Ascent Projection on multi-objective conflict steps:
            if self.direction_blend and originals is not None and self._protected is not None and all_params is not None:
                normals = self._protected.double()
                displacement = torch.cat([(p - old).reshape(-1) for p, old in zip(all_params, originals)])
                disp64 = displacement.double()
                before = normals @ disp64
                stats = self.constraint_geometry_stats
                blend = self.direction_blend_stats
                projected = False
                if bool((before > 0.0).any()):
                    blend["conflict_steps"] += 1
                    a_nrm = normals.norm(dim=1)
                    a_unit = normals[a_nrm > 0] / a_nrm[a_nrm > 0, None]
                    possible = int(a_unit.shape[0]) == 0 or float(a_unit.mean(dim=0).norm().item()) > 1e-12
                    direction = (
                        zero_pinv_project_nonascent(displacement, normals)
                        if possible
                        else torch.zeros_like(displacement)
                    )
                    blend["blended_steps" if possible else "pareto_stalls"] += 1
                    offset = 0
                    for p, old in zip(all_params, originals):
                        size = p.numel()
                        p.copy_(old + direction[offset : offset + size].view_as(p) if possible else old)
                        offset += size
                    applied64 = (
                        torch.cat([(p - old).reshape(-1) for p, old in zip(all_params, originals)]).double()
                        if possible
                        else torch.zeros_like(disp64)
                    )
                    projected = not torch.equal(applied64, disp64)
                    disp_nrm = float(disp64.norm().item())
                    ratio = float(applied64.norm().item()) / disp_nrm if disp_nrm > 0.0 else 0.0
                    blend["retained_norm_ratio_sum"] += ratio
                    blend["max_retained_norm_ratio"] = max(blend["max_retained_norm_ratio"], ratio)
                    after_max = float((normals @ applied64).max().item())
                else:
                    after_max = float(before.max().item())
                stats["steps"] += 1
                stats["projected_steps"] += int(projected)
                stats["max_derivative_before"] = max(stats["max_derivative_before"], float(before.max().item()))
                stats["max_derivative_after"] = max(stats["max_derivative_after"], after_max)

            self.last_step_gain = float(alpha_gain)
            self.last_probe_consumed = bool(probe_ran)
            if probe_ran:
                self.probe_consumptions += 1
        finally:
            self.abort_pending()

        if hasattr(self, "record") and self.record is not None:
            self.record.record_step(self)
        return loss

    def state_dict(self) -> Dict[str, Any]:
        result = Optimizer.state_dict(self)
        g_ref_out = float(self._g_peak_ref_t.item()) if self._g_peak_ref_t is not None else float(self._g_peak_ref)
        meta: Dict[str, Any] = {
            "schema": 1,
            "family": self.family,
            "generator_free": bool(self._generator_free),
            "last_step_gain": float(self.last_step_gain),
            "ema_func_gain": float(self._ema_func_gain),
            "ema_flip_cos": float(self._ema_flip_cos),
            "g_peak_ref": g_ref_out,
            "prev_flip_sketch": (
                None
                if self._prev_flip_sketch is None
                else [float(x) for x in self._prev_flip_sketch.detach().cpu().tolist()]
            ),
            "direction_blend": bool(self.direction_blend),
            "sampled_rows": [
                self.sampled_rows_for(grp["params"][0]) if grp["algorithm"] == "unified_prior" else None
                for grp in self.param_groups
            ],
        }
        if self.direction_blend:
            meta["constraint_geometry"] = dict(
                stats=deepcopy(self.constraint_geometry_stats),
                blend_stats=deepcopy(self.direction_blend_stats),
                pending=None if self._protected is None else self._protected.detach().clone(),
            )
        result["unified_v15_equil"] = meta
        if hasattr(self, "record") and self.record is not None:
            result["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return result

    def validate_state_dict(self, saved: Any) -> None:
        expected_keys = {"state", "param_groups", "unified_v15_equil"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError("invalid UnifiedV15Optimizer checkpoint schema")
        meta = saved["unified_v15_equil"]
        expected_meta_keys = {
            "schema",
            "family",
            "generator_free",
            "last_step_gain",
            "ema_func_gain",
            "ema_flip_cos",
            "g_peak_ref",
            "prev_flip_sketch",
            "direction_blend",
            "sampled_rows",
        }
        if self.direction_blend:
            expected_meta_keys.add("constraint_geometry")
        if (
            not isinstance(meta, dict)
            or set(meta) != expected_meta_keys
            or type(meta["schema"]) is not int
            or meta["schema"] != 1
            or meta["family"] != self.family
            or type(meta["generator_free"]) is not bool
            or meta["generator_free"] != self._generator_free
            or type(meta["direction_blend"]) is not bool
            or meta["direction_blend"] != self.direction_blend
            or not isinstance(meta["sampled_rows"], list)
            or len(meta["sampled_rows"]) != len(self.param_groups)
        ):
            raise ValueError("incompatible UnifiedV15Optimizer metadata")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) <= 0.0:
            raise ValueError("invalid UnifiedV15Optimizer last_step_gain")
        ema_fg = meta["ema_func_gain"]
        if (
            isinstance(ema_fg, bool)
            or not isinstance(ema_fg, (int, float))
            or not math.isfinite(ema_fg)
            or not (0.0 < float(ema_fg) <= 1.0)
        ):
            raise ValueError("invalid UnifiedV15Optimizer ema_func_gain")
        ema_flip = meta["ema_flip_cos"]
        if (
            isinstance(ema_flip, bool)
            or not isinstance(ema_flip, (int, float))
            or not math.isfinite(ema_flip)
            or not (-1.0 <= float(ema_flip) <= 1.0)
        ):
            raise ValueError("invalid UnifiedV15Optimizer ema_flip_cos")
        g_ref_v = meta["g_peak_ref"]
        if (
            isinstance(g_ref_v, bool)
            or not isinstance(g_ref_v, (int, float))
            or not math.isfinite(g_ref_v)
            or float(g_ref_v) < 0.0
        ):
            raise ValueError("invalid UnifiedV15Optimizer g_peak_ref")
        prev_sk = meta["prev_flip_sketch"]
        if prev_sk is not None:
            if (
                not isinstance(prev_sk, list)
                or len(prev_sk) != _FLIP_SKETCH_DIM
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in prev_sk
                )
            ):
                raise ValueError("invalid UnifiedV15Optimizer prev_flip_sketch")
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            if (
                not isinstance(cg, dict)
                or set(cg) != {"stats", "blend_stats", "pending"}
                or not isinstance(cg["stats"], dict)
                or set(cg["stats"]) != set(self.constraint_geometry_stats)
                or not isinstance(cg["blend_stats"], dict)
                or set(cg["blend_stats"]) != set(self.direction_blend_stats)
            ):
                raise ValueError("invalid UnifiedV15Optimizer direction_blend checkpoint")
            if cg["pending"] is not None:
                if not isinstance(cg["pending"], torch.Tensor) or not bool(torch.isfinite(cg["pending"]).all()):
                    raise ValueError("invalid pending direction_blend tensor")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError("UnifiedV15Optimizer group count mismatch")
        for cur_grp, sav_grp in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav_grp, dict) or len(sav_grp.get("params", ())) != len(cur_grp["params"]):
                raise ValueError("UnifiedV15Optimizer group parameter mismatch")
            if sav_grp.get("role") != cur_grp["role"] or sav_grp.get("algorithm") != cur_grp["algorithm"]:
                raise ValueError("UnifiedV15Optimizer group role/algorithm mismatch")
        if not isinstance(saved["state"], dict):
            raise ValueError("invalid UnifiedV15Optimizer state map")
        for st_val in saved["state"].values():
            if not isinstance(st_val, dict):
                raise ValueError("invalid UnifiedV15Optimizer parameter state")
            step_v = st_val.get("step", 0)
            if type(step_v) is not int or step_v < 0:
                raise ValueError("invalid UnifiedV15Optimizer step counter")
            if "peak_rms_val" in st_val:
                pv = st_val["peak_rms_val"]
                if isinstance(pv, bool) or not isinstance(pv, (int, float)) or not math.isfinite(pv) or float(pv) < 0.0:
                    raise ValueError("invalid peak_rms_val in UnifiedV15Optimizer state")
            for t_key in ("m_unit", "m_scale", "v_rms", "v_scale", "m_prior", "peak_rms"):
                if t_key in st_val:
                    ten = st_val[t_key]
                    if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                        raise ValueError(f"non-finite or invalid tensor {t_key} in UnifiedV15Optimizer state")
                    if t_key == "peak_rms" and (ten.ndim != 0 or float(ten.item()) < 0.0):
                        raise ValueError("invalid peak_rms tensor in UnifiedV15Optimizer state")
        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid UnifiedV15Optimizer critic regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        meta = state_dict["unified_v15_equil"]
        clean = {k: v for k, v in state_dict.items() if k not in ("unified_v15_equil", "regularizer")}
        Optimizer.load_state_dict(self, clean)
        self.clear_sampled_rows()
        self.last_step_gain = float(meta["last_step_gain"])
        self._ema_func_gain = float(meta["ema_func_gain"])
        self._ema_flip_cos = float(meta["ema_flip_cos"])
        g_ref_saved = float(meta["g_peak_ref"])
        self._g_peak_ref = g_ref_saved
        self._g_peak_ref_t = None
        prev_sk = meta["prev_flip_sketch"]
        if prev_sk is None:
            self._prev_flip_sketch = None
        else:
            ref_dev = (
                self.param_groups[0]["params"][0].device
                if (self.param_groups and self.param_groups[0]["params"])
                else torch.device("cpu")
            )
            self._prev_flip_sketch = torch.tensor(prev_sk, dtype=torch.float32, device=ref_dev)
        self._generator_free = bool(meta["generator_free"])
        self.last_probe_consumed = False
        self.probe_consumptions = 0
        self.ns_2d_activations = 0
        self.ns_5step_calls = 0
        self.fast_path_updates = 0
        self.critic_ns_calls = 0
        self.s6_skipped_deadband = 0
        self._dense_metrics_cache.clear()
        self._conv_metrics_cache.clear()
        for grp, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None and grp["algorithm"] == "unified_prior":
                self.set_sampled_rows(grp["params"][0], rows.to(grp["params"][0].device))
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            self.constraint_geometry_stats = deepcopy(cg["stats"])
            self.direction_blend_stats = deepcopy(cg["blend_stats"])
            self._protected = None if cg["pending"] is None else cg["pending"].detach().clone()
        else:
            self._protected = None
        if hasattr(self, "record") and self.record is not None:
            self.record.load_state_dict(state_dict["regularizer"]["record"])
            if self.param_groups and self.param_groups[0]["params"]:
                self.record.restore_device_state(self.param_groups[0]["params"][0].device)
        rec_l = self._active_record()
        if rec_l is not None:
            is_critic_l = getattr(self, "critic", None) is not None or all(
                self._player(grp["role"]) == "critic" for grp in self.param_groups
            )
            if not is_critic_l:
                rec_l._v15_g_peak_ref = g_ref_saved
                rec_l._v15_g_peak_ref_t = None
            elif (
                getattr(rec_l, "_v15_g_peak_ref", 0.0) == 0.0
                and getattr(rec_l, "_v15_g_peak_ref_t", None) is None
            ):
                rec_l._v15_g_peak_ref = g_ref_saved
                rec_l._v15_g_peak_ref_t = None


__all__ = [
    "UnifiedV15Optimizer",
    "collect_conv_layouts",
    "_band_scale",
    "quintic_orthogonalize_2d",
    "quintic_orthogonalize_batched",
    "_schatten6_spectral_decay_2d",
    "_schatten6_spectral_decay_batched",
]
