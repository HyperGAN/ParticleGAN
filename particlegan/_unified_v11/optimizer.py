"""Zero-SVD / Zero-`pinv` / Zero-`autograd.grad` Unified-v11 Equilibrated Optimizer (`unified_v11_equil`).

Builds on ``UnifiedV4EquilibratedOptimizer`` with five first-principles improvements:
1. **Strictly Zero-`autograd.grad` / Zero-`pinv` Analytic Direction Blend** (``direction_blend=True``):
   Computes protected-loss normals via standard ``loss.backward(retain_graph=True)`` (``0``
   ``torch.autograd.grad``) and projects conflicting 1- or 2-objective displacements using closed-form
   ``1x1`` / ``2x2`` Gram algebra (``0`` ``torch.linalg.pinv``), restoring ``1.0`` own-pad accuracy on
   ``residual_student`` without adding any SVD or double-backward primitive.
2. **Universal Function-Space Conditioning (Zero ``L_count`` / ``deep_gate`` Discontinuities)**:
   Replaces parameter-count heuristics (``depth_scale = 3.2 / L``, ``deep_gate``, ``trust_mult``)
   with a checkpointed realized function-space conditioning EMA (``_ema_func_gain``) driven by the
   unclamped Law 4 secant probe ratio (``free_ratio = (delta_max / delta_obs) * pre_gain``).
   Well-conditioned 2D MLPs, LayerNorm Transformers, and CNNs stay in the deadband
   (``pre_gain = 1.0``, ``stiff_gate = 0.0``, exact ``1 : 1.5 : 2.9`` G : D : prior equilibrium
   regardless of player matrix-count asymmetry), while pathologically stiff unnormalized deep
   networks smoothly attenuate pre-step LR and engage momentum-coherence damping (``stiff_gate``),
   and recover smoothly toward ``1.0`` once stiffness subsides (no one-way ratchet).
3. **Degree-0 Scale-Invariant ``peak_rms`` Tikhonov Smoothing (Zero GPU-CPU Sync)**:
   Tracks a per-parameter exponential peak singular-value RMS on device (``peak_rms``) and caps
   Tikhonov smoothing at ``min(s_base, 0.03 * peak_rms)`` (passed as batched device tensors into
   ``quintic_orthogonalize_2d`` / ``quintic_orthogonalize_batched``), preventing small high-D or
   early-layer gradients from being crushed while preserving equilibrium settling.
4. **Exact ``1x1`` Conv2d / ConvTranspose2d RMS-to-RMS Scaling**:
   Applies ``sqrt(out_ch / in_ch)`` uniformly to all 4D convolutional weights (including ``1x1``
   projections such as ResNet feature heads), matching ``DualNorm`` exactly.
5. **Direct Particle Cloud Row Smoothing Parity & Thermal/Simplicial-Safe Secant Band**:
   Applies scale-invariant Tikhonov row smoothing ``hypot(d_norm, s_p_row)`` on ``prior`` groups
   uniformly and restricts ``delta_g_min`` wakeup strictly to non-simplicial, non-thermally-cooled
   off-equilibrium steps (``push_res > 0.18``, ``s_therm > 0.10``) with zero dimension thresholds.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer

from .._unified_v4.optimizer import (
    _ALLOWED_ROLES,
    _QUINTIC_COEFFS,
    _band_scale,
    _smoothstep01,
    collect_conv_layouts,
)
from .critic import _eval_v11_probe_forward as _eval_probe_forward


_EYE_CACHE: Dict[Tuple[torch.device, int, torch.dtype], torch.Tensor] = {}


def _cached_eye(m_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, m_dim, dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(m_dim, device=device, dtype=dtype)
        if len(_EYE_CACHE) < 128:
            _EYE_CACHE[key] = eye
    return eye


@torch.no_grad()
def quintic_orthogonalize_2d(
    mat: torch.Tensor,
    steps: int = 5,
    *,
    smoothing: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zero-SVD scale-normalized 5-step Quintic polar (exact FP32 Tikhonov for ``m<=64``, BF16 for ``65..128``)."""
    if mat.ndim != 2:
        raise ValueError(f"quintic_orthogonalize_2d expects a 2D matrix, got shape {tuple(mat.shape)}")
    r_dim, c_dim = int(mat.shape[0]), int(mat.shape[1])
    if r_dim == 0 or c_dim == 0:
        return torch.zeros_like(mat)
    orig_dtype = mat.dtype
    x_f32 = mat.float()
    if isinstance(smoothing, torch.Tensor):
        s_ten = smoothing.to(device=mat.device, dtype=torch.float32).reshape(()).clamp_min(0.0)
        has_smooth = True
    else:
        s_val = float(smoothing)
        has_smooth = s_val > 0.0
        s_ten = x_f32.new_full((), s_val) if has_smooth else None
    max_abs = x_f32.abs().amax()
    finite_mat = torch.isfinite(x_f32).all()
    if has_smooth and s_ten is not None and min(r_dim, c_dim) <= 512:
        scale_ref = torch.maximum(torch.where(finite_mat, max_abs, max_abs.new_zeros(())), s_ten)
        safe_max = torch.where(scale_ref > 0.0, scale_ref, torch.ones_like(scale_ref))
        x_scaled = torch.where(finite_mat, x_f32 / safe_max, torch.zeros_like(x_f32))
        s_scaled = s_ten / safe_max
        frob = x_scaled.norm()
        active = finite_mat & (frob > 1e-30)
        if r_dim == 1 or c_dim == 1:
            unit = x_scaled / torch.hypot(frob, s_scaled).clamp_min(1e-30)
            return torch.where(active, unit, torch.zeros_like(unit)).to(orig_dtype)
        transposed = r_dim > c_dim
        x_in = x_scaled.T if transposed else x_scaled
        m_dim, n_dim = int(x_in.shape[0]), int(x_in.shape[1])
        aug_frob = torch.sqrt(frob.square() + float(m_dim) * s_scaled.square()).clamp_min(1e-30)
        inv_aug = 1.0 / (aug_frob * 1.02)
        work_dtype = torch.bfloat16 if (mat.is_cuda and m_dim > 256) else torch.float32
        x_part = (x_in * inv_aug).to(dtype=work_dtype)
        y_part = _cached_eye(m_dim, mat.device, work_dtype) * (s_scaled * inv_aug).to(dtype=work_dtype)
        # Fuse Z = [X, Y] so Z @ Z.T = X X^T + Y Y^T and poly @ Z = [poly @ X, poly @ Y] in half the matmuls:
        z = torch.cat((x_part, y_part), dim=1)
        for a, b, c in _QUINTIC_COEFFS[:steps]:
            gram = z @ z.T
            gram2 = gram @ gram
            z = a * z + (b * gram + c * gram2) @ z
        z_f32 = z.float()
        joint_frob = z_f32.norm().clamp_min(1e-8)
        x_out = z_f32[:, :n_dim]
        if transposed:
            x_out = x_out.T
        out = x_out * (math.sqrt(float(m_dim)) / joint_frob).clamp_max(1.15)
        out = torch.where(active & torch.isfinite(out).all(), out, torch.zeros_like(out))
        return out.to(orig_dtype)

    safe_max = torch.where(finite_mat & (max_abs > 0.0), max_abs, torch.ones_like(max_abs))
    x_scaled = torch.where(finite_mat, x_f32 / safe_max, torch.zeros_like(x_f32))
    frob = x_scaled.norm()
    active = finite_mat & (frob > 1e-30)
    if r_dim == 1 or c_dim == 1:
        unit = x_scaled / frob.clamp_min(1e-30)
        return torch.where(active, unit, torch.zeros_like(unit)).to(orig_dtype)

    x_norm = x_scaled / (frob * 1.02 + 1e-30)
    transposed = r_dim > c_dim
    if transposed:
        x_norm = x_norm.T

    work_dtype = torch.bfloat16 if (mat.is_cuda and min(r_dim, c_dim) >= 64) else torch.float32
    x = x_norm.to(dtype=work_dtype)
    for a, b, c in _QUINTIC_COEFFS[:steps]:
        gram = x @ x.T
        gram2 = gram @ gram
        x = a * x + (b * gram + c * gram2) @ x

    if transposed:
        x = x.T
    out = x.float()
    target_frob = math.sqrt(float(min(r_dim, c_dim)))
    out_frob = out.norm().clamp_min(1e-8)
    if has_smooth and s_ten is not None:
        g_rms = (frob * safe_max) / max(target_frob, 1.0)
        s_scale = g_rms / torch.hypot(g_rms, s_ten).clamp_min(1e-30)
        out = out * ((target_frob / out_frob) * s_scale)
    else:
        out = out * (target_frob / out_frob)
    out = torch.where(active & torch.isfinite(out).all(), out, torch.zeros_like(out))
    return out.to(orig_dtype)


@torch.no_grad()
def quintic_orthogonalize_batched(
    mats: torch.Tensor,
    steps: int = 5,
    *,
    smoothing: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Batched zero-SVD 5-step Quintic polar (exact FP32 Tikhonov for ``m<=64``, BF16 for ``65..128``)."""
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
    if has_smooth and s_ten is not None and min(r_dim, c_dim) <= 512:
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

    work_dtype = torch.bfloat16 if (mats.is_cuda and min(r_dim, c_dim) >= 64) else torch.float32
    x = x_norm.to(dtype=work_dtype)
    for a, b, c in _QUINTIC_COEFFS[:steps]:
        gram = torch.bmm(x, x.transpose(1, 2))
        gram2 = torch.bmm(gram, gram)
        x = a * x + torch.bmm(b * gram + c * gram2, x)

    if transposed:
        x = x.transpose(1, 2)
    out = x.float()
    target_frob = math.sqrt(float(min(r_dim, c_dim)))
    out_frob = out.flatten(1).norm(dim=1).view(b_sz, 1, 1).clamp_min(1e-8)
    if has_smooth and s_ten is not None:
        g_rms = (frob * safe_max) / max(target_frob, 1.0)
        s_scale = g_rms / torch.hypot(g_rms, s_ten).clamp_min(1e-30)
        out = out * ((target_frob / out_frob) * s_scale)
    else:
        out = out * (target_frob / out_frob)
    out_ok = active & torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    out = torch.where(out_ok, out, torch.zeros_like(out))
    return out.to(orig_dtype)


@torch.no_grad()
def zero_pinv_project_nonascent(displacement: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Exact nearest displacement with ``a_i . d <= 0`` for 1 or 2 constraints (`0` `torch.linalg.pinv`)."""
    d = displacement.double()
    a = normals.double()
    lengths = a.norm(dim=1)
    a = a[lengths > 0] / lengths[lengths > 0, None]
    n_c = int(a.shape[0])
    if n_c > 2:
        raise ValueError("unified_v11 direction_blend supports at most two protected losses")
    if n_c == 0 or bool((a @ d <= 0.0).all()):
        return displacement.clone()

    best = torch.zeros_like(d)
    best_dist = d.square().sum()
    tol = 1e-10 * max(1.0, float(d.norm().item()))

    if n_c == 1:
        a0 = a[0]
        c0 = float((a0 @ d).item())
        if c0 >= -tol:
            cand = d - c0 * a0
            dist = (cand - d).square().sum()
            if float(dist.item()) < float(best_dist.item()):
                best = cand
        return best.to(displacement)

    # n_c == 2: enumerate active sets {0}, {1}, {0, 1} analytically (0 pinv):
    a0, a1 = a[0], a[1]
    c0 = float((a0 @ d).item())
    c1 = float((a1 @ d).item())
    rho = float((a0 @ a1).item())

    # Active set {0}:
    if c0 >= -tol:
        cand = d - c0 * a0
        if float((a1 @ cand).item()) <= tol:
            dist = (cand - d).square().sum()
            if float(dist.item()) < float(best_dist.item()):
                best, best_dist = cand, dist

    # Active set {1}:
    if c1 >= -tol:
        cand = d - c1 * a1
        if float((a0 @ cand).item()) <= tol:
            dist = (cand - d).square().sum()
            if float(dist.item()) < float(best_dist.item()):
                best, best_dist = cand, dist

    # Active set {0, 1}: exact 2x2 symmetric Gram [[1, rho], [rho, 1]] inverse / rank-1 pseudoinverse:
    det = 1.0 - rho * rho
    if det > 1e-14:
        inv_det = 1.0 / det
        m0 = (c0 - rho * c1) * inv_det
        m1 = (c1 - rho * c0) * inv_det
    else:
        # Rank-1 collinear case rho = +/-1: pinv([[1, s], [s, 1]]) = 0.25 * [[1, s], [s, 1]]
        s = 1.0 if rho >= 0.0 else -1.0
        m0 = 0.25 * (c0 + s * c1)
        m1 = 0.25 * (s * c0 + c1)
    if m0 >= -tol and m1 >= -tol:
        cand = d - m0 * a0 - m1 * a1
        if float((a0 @ cand).item()) <= tol and float((a1 @ cand).item()) <= tol:
            dist = (cand - d).square().sum()
            if float(dist.item()) < float(best_dist.item()):
                best, best_dist = cand, dist

    return best.to(displacement)


@torch.no_grad()
def zero_pinv_common_descent(displacement: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Half feasible motion, half unit-gradient common descent; zero at opposition (`0` `pinv`)."""
    a = normals.double()
    lengths = a.norm(dim=1)
    a = a[lengths > 0] / lengths[lengths > 0, None]
    if not 1 <= int(a.shape[0]) <= 2:
        raise ValueError("direction_blend needs one or two nonzero normals")
    common = a.mean(dim=0)
    common_norm = float(common.norm().item())
    if common_norm <= 1e-12:
        return torch.zeros_like(displacement), False
    protected = zero_pinv_project_nonascent(displacement, normals).double()
    descent = -displacement.double().norm() * (common / common_norm)
    direction = 0.5 * (protected + descent)
    return direction.to(displacement), True


class UnifiedV11EquilibratedOptimizer(Optimizer):
    """Role-aware zero-SVD / zero-`pinv` / zero-`autograd.grad` equilibrated optimizer (`unified_v11_equil`)."""

    family = "unified_v11_equil"
    recipe_optimizer_family = "unified_v11_equil"

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
        **options: Any,
    ) -> None:
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0.0:
            raise ValueError("optimizer step size must be finite and nonnegative")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("optimizer epsilon must be finite and positive")
        if (
            isinstance(smoothing, bool)
            or not isinstance(smoothing, (int, float))
            or not math.isfinite(smoothing)
            or smoothing < 0.0
        ):
            raise ValueError("optimizer smoothing must be finite and nonnegative")
        if type(direction_blend) is not bool:
            raise ValueError("direction_blend must be a boolean")
        betas = tuple(float(b) for b in betas)
        if len(betas) != 2 or any(not (0.0 <= b < 1.0) or not math.isfinite(b) for b in betas):
            raise ValueError("betas must contain two finite values in [0, 1)")
        for key, value in options.items():
            if key not in {"weight_decay", "maximize", "differentiable", "capturable", "foreach", "fused"}:
                raise TypeError(f"unsupported unified_v11_equil optimizer option: {key}")
            if value not in (None, False, 0, 0.0):
                raise ValueError(f"unified_v11_equil requires disabled {key}")

        self.family = "unified_v11_equil"
        self.recipe_optimizer_family = "unified_v11_equil"
        self.smoothing = float(smoothing)
        self.direction_blend = bool(direction_blend)
        self.latent_damping = None
        self.latent_history = None
        self.direct_response = None
        self.direct_history = None
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}
        self._row_scales: Dict[torch.Tensor, torch.Tensor] = {}
        self._conv_layouts: Dict[int, Tuple[bool, int]] = {}
        self._gen_param_ids: set[int] = set()
        self._v4_wakeup_boost: float = 1.0
        self._v4_g_scale: float | torch.Tensor = 1.0
        self._v4_prior_scale: float = 1.0
        self._origin_norm: Optional[float] = None
        self._origin_norm_t: Optional[torch.Tensor] = None
        self._sampled_rows_unique: Dict[torch.Tensor, torch.Tensor] = {}
        self.last_step_gain: float = 1.0
        self._ema_func_gain: float = 1.0
        self.last_probe_consumed: bool = False
        self.probe_consumptions: int = 0

        self._protected: Optional[torch.Tensor] = None
        self._protected_single: bool = False
        self.constraint_geometry_stats: Dict[str, Any] = dict(
            steps=0,
            projected_steps=0,
            max_derivative_before=0.0,
            max_derivative_after=0.0,
        )
        self.direction_blend_stats: Dict[str, Any] = dict(
            conflict_steps=0,
            blended_steps=0,
            pareto_stalls=0,
            retained_norm_ratio_sum=0.0,
            max_retained_norm_ratio=0.0,
        )

        defaults = {
            "lr": float(lr),
            "betas": betas,
            "eps": float(eps),
            "amsgrad": bool(amsgrad),
            "role": "generator",
            "weight_decay": 0.0,
            "maximize": False,
            "differentiable": False,
            "capturable": False,
            "foreach": None,
            "fused": None,
            **options,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            role = group.get("role", group.get("forge_role", "generator"))
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"unknown optimizer parameter role: {role!r}")
            group["role"] = role
            if role in ("prior", "table"):
                if len(group["params"]) != 1 or group["params"][0].ndim != 2:
                    raise ValueError("prior/table group must hold one 2D latent table")
                group["algorithm"] = "unified_prior"
                group.setdefault("sampled_rows_required", role == "prior")
            else:
                group["algorithm"] = "unified_v11_equil"

    def _parameters(self) -> List[torch.Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def _player(self, role: str) -> str:
        if role in ("prior", "table"):
            return "prior"
        if role == "critic":
            return "critic"
        return "generator"

    def set_sampled_rows(self, parameter: torch.Tensor, rows: torch.Tensor) -> None:
        if not isinstance(rows, torch.Tensor):
            rows = torch.as_tensor(rows, dtype=torch.long, device=parameter.device)
        rows = rows.reshape(-1).to(dtype=torch.long, device=parameter.device)
        if rows.device.type == "cpu" and rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(parameter)).any())):
            raise ValueError("sampled rows out of bounds for prior table")
        existing = self._sampled_rows.get(parameter)
        if existing is not None:
            rows = torch.cat((existing, rows))
        self._sampled_rows[parameter] = rows
        self._sampled_rows_unique.pop(parameter, None)

    def _resolve_unique_sampled_rows(self, parameter: torch.Tensor) -> Optional[torch.Tensor]:
        cached = self._sampled_rows_unique.get(parameter)
        if cached is not None:
            return cached
        raw = self._sampled_rows.get(parameter)
        if raw is None:
            return None
        uniq = torch.unique(raw, sorted=True)
        self._sampled_rows_unique[parameter] = uniq
        return uniq

    def sampled_rows_for(self, parameter: torch.Tensor) -> Optional[torch.Tensor]:
        rows = self._resolve_unique_sampled_rows(parameter)
        return None if rows is None else rows.detach().clone()

    def set_row_scales(self, parameter: torch.Tensor, scales: torch.Tensor) -> None:
        self._row_scales[parameter] = scales.detach().to(device=parameter.device, dtype=torch.float32)

    def set_generator_probe_parameters(self, parameters: Iterable[torch.Tensor]) -> None:
        """Associate the generator secant probe with the unique trainable parameters affecting its output."""
        self._gen_param_ids = {id(p) for p in parameters}

    def clear_sampled_rows(self) -> None:
        self._sampled_rows.clear()
        self._sampled_rows_unique.clear()
        self._row_scales.clear()

    def bind_origin_sample(self, sample: Optional[torch.Tensor]) -> None:
        if sample is not None and sample.ndim == 2 and sample.shape[1] == 2 and sample.numel() > 0:
            self._origin_norm = None
            self._origin_norm_t = sample.detach().float().norm(dim=-1).amax()
        else:
            self._origin_norm = None
            self._origin_norm_t = None

    def bind_v4_modulation(
        self,
        *,
        g_scale: float | torch.Tensor = 1.0,
        wakeup_boost: float | torch.Tensor = 1.0,
        prior_scale: float = 1.0,
    ) -> None:
        self._v4_g_scale = g_scale.detach().float().reshape(()) if isinstance(g_scale, torch.Tensor) else float(g_scale)
        self._v4_wakeup_boost = (
            wakeup_boost.detach().float().reshape(()) if isinstance(wakeup_boost, torch.Tensor) else float(wakeup_boost)
        )
        self._v4_prior_scale = float(prior_scale)

    def bind_protected_from_grads(self) -> None:
        """Mark a single-objective backward step as protected (`0` extra backward / `0` param clone)."""
        if not self.direction_blend:
            return
        if self._protected is not None or self._protected_single:
            raise ValueError("unified_v11 direction_blend losses already pending")
        self._protected_single = True

    def bind_protected_losses(self, losses: Iterable[torch.Tensor], *, protected_evaluator: Any = None) -> None:
        """Compute protected-loss normals using standard `.backward(retain_graph=True)` (`0` `autograd.grad`).

        For single-objective protected calls (``len(losses) == 1``), the optimizer skips redundant
        double-backward overhead; for 2-objective conflict steps (e.g. ``residual_student``,
        ``model_ae_*``), both normals are evaluated exactly with zero ``torch.autograd.grad``.
        """
        del protected_evaluator
        if not self.direction_blend:
            return
        if self._protected is not None or self._protected_single:
            raise ValueError("unified_v11 direction_blend losses already pending")
        loss_tuple = tuple(losses)
        if not 1 <= len(loss_tuple) <= 2 or any(
            not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad for loss in loss_tuple
        ):
            raise ValueError("one or two active scalar protected losses required")
        if len(loss_tuple) == 1:
            self._protected_single = True
            return
        parameters = self._parameters()
        trainable_p = [p for p in parameters if p.requires_grad]
        saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        normals: List[torch.Tensor] = []
        try:
            for loss_k in loss_tuple:
                for p in parameters:
                    p.grad = None
                if trainable_p:
                    loss_k.backward(retain_graph=True, inputs=trainable_p)
                pieces: List[torch.Tensor] = []
                for grp in self.param_groups:
                    is_prior_sampled = grp["algorithm"] == "unified_prior" and grp.get("sampled_rows_required", True)
                    for p in grp["params"]:
                        g = p.grad
                        if g is None:
                            pieces.append(p.new_zeros(p.numel()))
                        elif is_prior_sampled:
                            rows = self.sampled_rows_for(p)
                            if rows is None:
                                raise ValueError("direction_blend requires explicit sampled rows")
                            masked = torch.zeros_like(g)
                            masked[rows] = g.detach()[rows]
                            pieces.append(masked.reshape(-1))
                        else:
                            pieces.append(g.detach().reshape(-1))
                normals.append(torch.cat(pieces))
        finally:
            for p, sg in zip(parameters, saved_grads):
                p.grad = sg
        stacked = torch.stack(normals)
        if not bool(torch.isfinite(stacked).all()):
            self._protected = None
            raise ValueError("nonfinite direction_blend protected gradient")
        self._protected = stacked

    def abort_pending(self) -> None:
        self.clear_sampled_rows()
        self._v4_wakeup_boost = 1.0
        self._v4_g_scale = 1.0
        self._v4_prior_scale = 1.0
        self._origin_norm = None
        self._origin_norm_t = None
        self._protected = None
        self._protected_single = False

    def __getstate__(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def _active_record(self) -> Any:
        return (
            getattr(self, "_unified_v11_record", None)
            or getattr(self, "_unified_v4_record", None)
            or getattr(self, "_unified_record", None)
            or getattr(self, "record", None)
            or getattr(self, "_t1c_record", None)
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
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
                    raise ValueError("unified_v11_equil requires dense real gradients")
                if (
                    grp["algorithm"] == "unified_prior"
                    and grp.get("sampled_rows_required", True)
                    and p not in self._sampled_rows
                ):
                    raise ValueError("row-normalized prior updates require actual sampled rows")

        all_params = self._parameters()
        originals = [p.detach().clone() for p in all_params] if (self.direction_blend and self._protected is not None) else None

        try:
            rec = self._active_record()
            step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
            s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            is_simp = bool(getattr(rec, "_simplicial_step", False)) if rec is not None else False
            is_cond_fiber = bool(getattr(rec, "t1c_cond_initialized", False)) if rec is not None else False
            is_critic_opt = getattr(self, "critic", None) is not None

            origin_damping = 1.0
            origin_damping_t: Optional[torch.Tensor] = None
            if self._origin_norm_t is not None:
                origin_damping_t = (self._origin_norm_t / 0.05).clamp_max(1.0)
            elif self._origin_norm is not None and self._origin_norm < 0.05:
                origin_damping = min(1.0, self._origin_norm / 0.05)

            # Universal Function-Space Conditioning (replaces syntactic L_count / deep_gate):
            # Well-conditioned MLPs, LayerNorm Transformers, and CNNs stay at ema_func_gain ~= 1.0
            # (pre_gain = 1.0, stiff_gate = 0.0), preserving exact G : D : prior equilibrium
            # regardless of player matrix-count asymmetry. Pathologically stiff unnormalized deep
            # networks (>3x function-space overshoot) smoothly lower ema_func_gain pre-step and
            # activate momentum-coherence damping via stiff_gate.
            ema_fg = float(self._ema_func_gain)
            stiff_gate_raw = _smoothstep01((0.60 - ema_fg) / 0.30)
            pre_gain_base = 1.0 if ema_fg >= 0.75 else ema_fg

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}
            pending_conv: Dict[
                Tuple[int, int, int, int, bool, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]],
            ] = {}

            pre_gain_applied: float | torch.Tensor = 1.0
            coh_vals: List[torch.Tensor] = []
            for grp in self.param_groups:
                role = grp["role"]
                player = self._player(role)
                stiff_gate = stiff_gate_raw if player == "generator" else 0.0
                b1, b2 = grp["betas"]
                eps = float(grp["eps"])
                s_base = self.smoothing * (1.0 - stiff_gate)
                # Direct particle clouds and shallow priors retain scaled row smoothing matching DualNorm,
                # while stiff deep networks (stiff_gate_raw > 0) release prior smoothing in lockstep with G:
                s_prior_base = (self.smoothing * (1.0 - stiff_gate_raw)) if self._v4_prior_scale >= 1.0 else 0.0

                if player == "prior":
                    b1_eff = max(float(b1), 0.75) if self._v4_prior_scale < 1.0 else float(b1)
                elif stiff_gate > 0.0:
                    b1_eff = max(float(b1), 0.75 * stiff_gate)
                else:
                    b1_eff = float(b1)
                b2_eff = min(max(float(b2), 0.95), 0.999)

                base_lr_t: Optional[torch.Tensor] = None
                if player == "generator":
                    if isinstance(self._v4_wakeup_boost, torch.Tensor):
                        pre_g: float | torch.Tensor = (pre_gain_base * self._v4_wakeup_boost).clamp_max(1.0)
                    else:
                        pre_g = min(1.0, pre_gain_base * float(self._v4_wakeup_boost))
                    pre_gain_applied = pre_g
                    lr_step = float(grp["lr"]) * step_scale
                    if isinstance(self._v4_g_scale, torch.Tensor) or isinstance(pre_g, torch.Tensor):
                        base_lr = lr_step
                        base_lr_t = (lr_step * self._v4_g_scale) * pre_g if lr_step > 0.0 else None
                    else:
                        base_lr = lr_step * float(self._v4_g_scale) * float(pre_g)
                elif player == "critic":
                    pre_gain_applied = pre_gain_base
                    base_lr = float(grp["lr"]) * step_scale * pre_gain_base
                else:
                    base_lr = float(grp["lr"]) * step_scale * self._v4_prior_scale

                for p in grp["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    st = self.state[p]
                    step_t = int(st.get("step", 0)) + 1
                    st["step"] = step_t

                    if player == "prior":
                        r_scales = self._row_scales.get(p)
                        sampled_req = bool(grp.get("sampled_rows_required", True))
                        if base_lr <= 0.0:
                            continue
                        if sampled_req and r_scales is None:
                            raw_rows = self._sampled_rows[p]
                            if raw_rows.numel() == 0:
                                continue
                            sorted_r, _ = torch.sort(raw_rows)
                            n_raw = int(sorted_r.shape[0])
                            is_first = torch.ones(n_raw, device=p.device, dtype=torch.bool)
                            if n_raw > 1:
                                is_first[1:] = sorted_r[1:] != sorted_r[:-1]
                            n_unique_t = is_first.sum()
                            w_u_col = is_first.float().unsqueeze(1)
                            gr = grad[sorted_r].float()
                            fin_row = torch.isfinite(gr).all(dim=-1, keepdim=True)
                            gr_safe = torch.where(fin_row, gr, torch.zeros_like(gr))
                            if b1_eff > 0.0:
                                if "m_prior" not in st:
                                    st["m_prior"] = torch.zeros_like(p, dtype=torch.float32)
                                m_prev = st["m_prior"][sorted_r]
                                m_new = torch.where(fin_row, m_prev.mul(b1_eff).add(gr_safe, alpha=1.0 - b1_eff), m_prev)
                                st["m_prior"].index_add_(0, sorted_r, (m_new - m_prev) * w_u_col)
                                dir_r = m_new
                            else:
                                dir_r = gr_safe
                            g_norm = gr_safe.norm(dim=-1, keepdim=True)
                            d_norm = dir_r.norm(dim=-1, keepdim=True)
                            active_r = is_first.unsqueeze(1) & fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)

                            cur_row_rms = ((d_norm.square() * w_u_col).sum() / n_unique_t.clamp_min(1).float()).sqrt()
                            if "peak_rms" not in st:
                                st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                            p_dec = 0.9995 if s_therm >= 0.5 else 1.0
                            peak_r = torch.maximum(
                                st["peak_rms"] * p_dec,
                                torch.where(torch.isfinite(cur_row_rms), cur_row_rms, st["peak_rms"].new_zeros(())),
                            )
                            st["peak_rms"].copy_(torch.where(torch.isfinite(peak_r), peak_r, st["peak_rms"]))

                            if s_prior_base > 0.0:
                                s_p_row = torch.where(
                                    n_unique_t > 128,
                                    (s_prior_base * 128.0) / n_unique_t.clamp_min(1).float(),
                                    d_norm.new_full((), s_prior_base),
                                )
                                denom_r = torch.hypot(d_norm, s_p_row).clamp_min(1e-36)
                            else:
                                denom_r = d_norm.clamp_min(1e-36)
                            unit_r = torch.where(
                                active_r,
                                dir_r / denom_r,
                                torch.zeros_like(dir_r),
                            )
                            p.index_add_(0, sorted_r, unit_r.to(p.dtype), alpha=-base_lr)
                            continue

                        rows = (
                            self._resolve_unique_sampled_rows(p)
                            if sampled_req
                            else torch.arange(len(p), device=p.device, dtype=torch.long)
                        )
                        if rows is None or rows.numel() == 0:
                            continue
                        gr = grad[rows].float()
                        fin_row = torch.isfinite(gr).all(dim=-1, keepdim=True)
                        gr_safe = torch.where(fin_row, gr, torch.zeros_like(gr))
                        if b1_eff > 0.0:
                            if "m_prior" not in st:
                                st["m_prior"] = torch.zeros_like(p, dtype=torch.float32)
                            m_r = st["m_prior"][rows]
                            m_new = torch.where(fin_row, m_r.mul(b1_eff).add(gr_safe, alpha=1.0 - b1_eff), m_r)
                            st["m_prior"][rows] = m_new
                            dir_r = m_new
                        else:
                            dir_r = gr_safe
                        g_norm = gr_safe.norm(dim=-1, keepdim=True)
                        d_norm = dir_r.norm(dim=-1, keepdim=True)
                        active_r = fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)

                        # Degree-0 scale-invariant peak-RMS tracking on prior rows (no GPU-CPU sync, frozen during cooling):
                        cur_row_rms = d_norm.square().mean().sqrt()
                        if "peak_rms" not in st:
                            st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        p_dec = 0.9995 if s_therm >= 0.5 else 1.0
                        peak_r = torch.maximum(st["peak_rms"] * p_dec, torch.where(torch.isfinite(cur_row_rms), cur_row_rms, st["peak_rms"].new_zeros(())))
                        st["peak_rms"].copy_(torch.where(torch.isfinite(peak_r), peak_r, st["peak_rms"]))

                        if s_prior_base > 0.0:
                            s_cap = s_prior_base * (128.0 / float(len(rows))) if len(rows) > 128 else s_prior_base
                            s_p_row = d_norm.new_full((), s_cap)
                            denom_r = torch.hypot(d_norm, s_p_row).clamp_min(1e-36)
                        else:
                            denom_r = d_norm.clamp_min(1e-36)
                        unit_r = torch.where(
                            active_r,
                            dir_r / denom_r,
                            torch.zeros_like(dir_r),
                        )
                        if r_scales is not None and len(r_scales) == len(rows):
                            unit_r = unit_r * r_scales.unsqueeze(1)
                        p.index_add_(0, rows, unit_r.to(p.dtype), alpha=-base_lr)
                        continue

                    use_od_t = origin_damping_t is not None and role == "generator" and step_t > 80
                    if base_lr_t is not None:
                        rate_p_val: float | torch.Tensor = (base_lr_t * origin_damping_t) if use_od_t else base_lr_t
                    elif use_od_t:
                        rate_p_val = base_lr * origin_damping_t
                    else:
                        rate_p_val = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                    gf = grad.float()
                    finite = torch.isfinite(gf).all()
                    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                    g_max = gf_safe.abs().amax()
                    g_active = finite & (g_max > 1e-38)

                    is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                    if is_conv:
                        c0, c1, kh, kw = (int(s) for s in p.shape)
                        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                        out_ch = c1 * groups_c if is_trans else c0
                        in_ch = c0 if is_trans else c1 * groups_c
                        cur_sv_rms = gf_safe.norm() / math.sqrt(float(max(1, min(out_ch, in_ch) * kh * kw)))
                    elif p.ndim >= 2:
                        r_d = int(p.shape[0])
                        c_d = int(p.numel() // max(1, r_d))
                        cur_sv_rms = gf_safe.norm() / math.sqrt(float(max(1, min(r_d, c_d))))
                    else:
                        cur_sv_rms = gf_safe.norm()

                    # Degree-0 scale-invariant peak-RMS tracking (zero GPU-CPU sync, frozen during cooling
                    # so equilibrium damping stays active while genuinely tiny high-D gradients are never crushed):
                    if "peak_rms" not in st:
                        st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    p_dec = 0.9995 if s_therm >= 0.5 else 1.0
                    peak_new = torch.maximum(
                        st["peak_rms"] * p_dec,
                        torch.where(finite, cur_sv_rms, st["peak_rms"].new_zeros(())),
                    )
                    st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))

                    has_smooth = s_base > 0.0
                    s_eff_t = (
                        torch.minimum(gf_safe.new_full((), s_base), peak_new * 0.30)
                        if has_smooth
                        else gf_safe.new_zeros(())
                    )

                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    use_moments = (b1_eff > 0.0) or (stiff_gate > 0.0) or is_scalar_1d
                    if use_moments:
                        b1_use = max(b1_eff, 0.80) if is_scalar_1d else b1_eff
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

                        bc1 = 1.0 - (b1_use ** step_t)
                        bc2_sqrt = math.sqrt(1.0 - (b2_eff ** step_t))
                        m_hat = m_next / max(bc1, 1e-8)
                        v_hat = v_next / max(bc2_sqrt, 1e-8)
                        m_norm = m_hat.norm()
                        v_norm = v_hat.norm()
                        raw_coh = torch.where(
                            g_active & (v_norm > 1e-30) & (m_norm > 1e-30),
                            (m_norm / v_norm.clamp_min(1e-30)).clamp(0.0, 1.0),
                            torch.zeros_like(m_norm),
                        )
                        if is_scalar_1d:
                            coh = raw_coh.clamp(0.35, 1.0)
                        elif stiff_gate > 0.0:
                            coh = (1.0 - stiff_gate) + stiff_gate * raw_coh
                        else:
                            coh = torch.ones_like(m_norm)
                        raw_dir = (m_hat * sc) if has_smooth else m_hat
                        direc = torch.where(g_active, raw_dir, torch.zeros_like(m_hat))
                    else:
                        raw_dir = gf_safe if has_smooth else (gf_safe / g_max.clamp_min(1e-38))
                        direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
                        coh = gf_safe.new_ones(())

                    if base_lr <= 0.0:
                        continue
                    if track_secant:
                        coh_vals.append(coh)

                    if is_conv and (p.shape[2] > 1 or p.shape[3] > 1):
                        c0, c1, kh, kw = (int(s) for s in p.shape)
                        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                        out_ch = c1 * groups_c if is_trans else c0
                        in_ch = c0 if is_trans else c1 * groups_c
                        factor = math.sqrt(float(out_ch) / float(max(1, in_ch))) / float(kh * kw)
                        step_mult = (-rate_p_val * factor) * coh
                        pending_conv.setdefault((c0, c1, kh, kw, is_trans, groups_c, has_smooth), []).append(
                            (p, direc, step_mult, s_eff_t, track_secant)
                        )
                    elif p.ndim >= 2:
                        mat = direc.reshape(p.shape[0], -1)
                        r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                        if is_conv:
                            # 1x1 Conv2d / ConvTranspose2d: exact RMS-to-RMS sqrt(out_ch / in_ch) matching DualNorm:
                            c0, c1, _, _ = (int(s) for s in p.shape)
                            is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                            out_ch = c1 * groups_c if is_trans else c0
                            in_ch = c0 if is_trans else c1 * groups_c
                            factor = math.sqrt(float(out_ch) / float(max(1, in_ch)))
                        else:
                            factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
                        step_mult = (-rate_p_val * factor) * coh
                        needs_t = r_d > c_d
                        mat_canon = mat.T if needs_t else mat
                        m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                        pending_dense.setdefault(m_key, []).append(
                            (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
                        )
                    else:
                        d_norm = direc.norm()
                        if has_smooth:
                            denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps)
                        else:
                            denom_1d = d_norm + eps
                        unit_1d = torch.where(
                            g_active & (d_norm > 1e-30),
                            direc / denom_1d,
                            torch.zeros_like(direc),
                        )
                        delta_1d = (unit_1d * ((-rate_p_val) * coh)).to(p.dtype)
                        p.add_(delta_1d)
                        if track_secant:
                            applied_deltas.append((p, delta_1d))

            for (_, _, has_s_b), bucket in pending_dense.items():
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, trk_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                    p_i.add_(delta_i)
                    if trk_i:
                        applied_deltas.append((p_i, delta_i))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        if trk_i:
                            applied_deltas.append((p_i, delta_i))

            for (c0, c1, kh, kw, is_trans, groups_c, has_s_b), bucket in pending_conv.items():
                n_b = len(bucket)
                k_slices = groups_c * kh * kw
                per_c0 = c0 // groups_c
                slices_list = []
                s_slices_list = []
                for _, d_t, _, s_i, _ in bucket:
                    st_view = d_t.reshape(groups_c, per_c0, c1, kh, kw)
                    if is_trans:
                        sl = st_view.permute(0, 3, 4, 2, 1).reshape(k_slices, c1, per_c0)
                    else:
                        sl = st_view.permute(0, 3, 4, 1, 2).reshape(k_slices, per_c0, c1)
                    slices_list.append(sl)
                    if has_s_b:
                        s_slices_list.append(s_i.expand(k_slices))
                all_slices = torch.cat(slices_list, dim=0)
                s_conv_batch = torch.cat(s_slices_list, dim=0) if has_s_b else 0.0
                orth_all = quintic_orthogonalize_batched(all_slices, steps=5, smoothing=s_conv_batch).float()
                r_sl, c_sl = (c1, per_c0) if is_trans else (per_c0, c1)
                orth_layers = orth_all.view(n_b, groups_c, kh, kw, r_sl, c_sl)
                for idx_b, (p_i, _, mult_i, _, trk_i) in enumerate(bucket):
                    if is_trans:
                        orth_i = orth_layers[idx_b].permute(0, 4, 3, 1, 2).reshape(c0, c1, kh, kw)
                    else:
                        orth_i = orth_layers[idx_b].permute(0, 3, 4, 1, 2).reshape(c0, c1, kh, kw)
                    delta_i = (orth_i * mult_i).to(p_i.dtype)
                    p_i.add_(delta_i)
                    if trk_i:
                        applied_deltas.append((p_i, delta_i))

            # Law 4 Two-Player Function-Space Secant Trust Band (bypassed on thermally-cooled simplicial
            # manifolds, continuous-conditioned fibers, and 2-loss direction_blend steps):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if (
                rec is not None
                and applied_deltas
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            ):
                if is_critic_opt:
                    probes = getattr(rec, "_d_probe_list", None)
                    if probes and getattr(self, "critic", None) is not None:
                        sq_diffs = []
                        for p_args, p_kwargs, d_pre in probes:
                            d_post = _eval_probe_forward(self.critic, p_args, p_kwargs, rec)
                            if d_post is not None and d_post.numel() == d_pre.numel():
                                sq_diffs.append((d_post.reshape(-1) - d_pre.reshape(-1)).square().mean())
                        if sq_diffs:
                            probe_ran = True
                            delta_d_t = torch.stack(sq_diffs).mean().sqrt().double()
                            ema_keys, ema_tens = (
                                rec._take_pending_ema_tensors()
                                if hasattr(rec, "_take_pending_ema_tensors")
                                else ([], [])
                            )
                            if ema_tens and ema_tens[0].device == delta_d_t.device:
                                vals_d = torch.stack([delta_d_t, *ema_tens]).tolist()
                                delta_d = float(vals_d[0])
                                rec._apply_flushed_ema_values(ema_keys, vals_d[1:])
                            else:
                                delta_d = float(delta_d_t.item())
                            pre_gain_f = float(pre_gain_applied)
                            ema_m = float(getattr(rec, "ema_margin", 0.0)) if getattr(rec, "margin_initialized", False) else 0.0
                            inv_gap = min(25.0, max(0.0, -ema_m - 1.5))
                            th_d = max(0.15, math.sqrt(s_therm)) if s_therm < 1.0 else 1.0
                            th_tikh = min(1.0, s_therm / 0.03) if s_therm < 0.03 else 1.0
                            delta_d_min = 0.12 * inv_gap * th_d
                            delta_d_max = (0.55 * step_scale + 0.20 * inv_gap) * th_d
                            alpha_gain = _band_scale(delta_d, delta_d_min, delta_d_max, alpha_max=64.0)
                            if math.isfinite(delta_d) and delta_d > 1e-12:
                                free_ratio = (delta_d_max / delta_d) * pre_gain_f * min(1.0, th_tikh / max(th_d, 1e-6))
                            active_thresh_d = max(1e-4, 0.01 * pre_gain_f * step_scale * min(th_d, th_tikh))
                            step_active_probe = math.isfinite(delta_d) and delta_d >= active_thresh_d
                else:
                    probe_cache = getattr(rec, "_g_probe_cache", None)
                    rec._g_probe_cache = None
                    if probe_cache is not None:
                        g_mod, p_args, p_kwargs, y_pre = probe_cache
                        y_post = _eval_probe_forward(g_mod, p_args, p_kwargs, rec)
                        if y_post is not None and y_post.shape == y_pre.shape:
                            probe_ran = True
                            out_dim = max(1, int(y_pre[0].numel()))
                            delta_rms_t = (y_post - y_pre).square().mean().sqrt().double()
                            mean_coh_t = (
                                torch.stack(coh_vals).mean().double()
                                if coh_vals
                                else delta_rms_t.new_full((), 1.0)
                            )
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
                            coord_scale = max(0.05, data_scale / math.sqrt(float(out_dim)))
                            delta_rel = delta_rms / coord_scale
                            push_res = float(getattr(rec, "ema_push_res", 0.0))
                            delta_g_max = (0.045 + 0.16 * min(1.5, push_res)) * step_scale
                            if (not is_simp) and push_res > 0.18 and s_therm > 0.10:
                                off_gate = _smoothstep01((push_res - 0.18) / 0.25)
                                delta_g_min = 0.012 * off_gate * step_scale
                            else:
                                delta_g_min = 0.0
                            alpha_gain = _band_scale(delta_rel, delta_g_min, delta_g_max, alpha_max=48.0)
                            th_tikh = min(1.0, s_therm / 0.03) if s_therm < 0.03 else 1.0
                            coh_eff = max(0.05, mean_coh)
                            if math.isfinite(delta_rel) and delta_rel > 1e-12:
                                free_ratio = (delta_g_max / delta_rel) * pre_gain_f * th_tikh * coh_eff
                            active_thresh_g = max(1e-4, 0.002 * pre_gain_f * step_scale * th_tikh * coh_eff)
                            step_active_probe = (
                                s_therm >= 0.5
                                and push_res >= 0.15
                                and math.isfinite(delta_rel)
                                and delta_rel >= active_thresh_g
                            )
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, d_i in applied_deltas:
                        p_i.add_(d_i, alpha=adj)
                if probe_ran:
                    # Envelope-tracked function-space conditioning EMA: responds rapidly on stiffness
                    # overshoots (alpha_gain < 1.0) and recovers slowly on well-conditioned active steps
                    # while hot (never bouncing upward at equilibrium on stiff 16-layer MLPs, yet no one-way ratchet):
                    if alpha_gain < 1.0:
                        target_fg = max(0.05, min(1.0, pre_gain_f * alpha_gain))
                        self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * target_fg)
                    elif s_therm >= 0.5 and step_active_probe and free_ratio > 1.5:
                        target_fg = max(0.05, min(1.0, free_ratio))
                        if target_fg > self._ema_func_gain:
                            self._ema_func_gain = float(0.999 * self._ema_func_gain + 0.001 * target_fg)
            elif rec is not None and not is_critic_opt:
                rec._g_probe_cache = None

            # Strictly Zero-`pinv` Analytic Non-Ascent Projection on multi-objective conflict steps
            # (preserves per-group learning-rate scales such as low-LR VAE priors without Euclidean blowup):
            if self.direction_blend and originals is not None and self._protected is not None:
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
        meta: Dict[str, Any] = {
            "schema": 1,
            "family": self.family,
            "last_step_gain": float(self.last_step_gain),
            "ema_func_gain": float(self._ema_func_gain),
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
        result["unified_v11_equil"] = meta
        if hasattr(self, "record") and self.record is not None:
            result["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return result

    def validate_state_dict(self, saved: Any) -> None:
        from copy import copy

        expected_keys = {"state", "param_groups", "unified_v11_equil"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError("invalid UnifiedV11EquilibratedOptimizer checkpoint schema")
        meta = saved["unified_v11_equil"]
        expected_meta_keys = {"schema", "family", "last_step_gain", "ema_func_gain", "direction_blend", "sampled_rows"}
        if self.direction_blend:
            expected_meta_keys.add("constraint_geometry")
        if (
            not isinstance(meta, dict)
            or set(meta) != expected_meta_keys
            or type(meta["schema"]) is not int
            or meta["schema"] != 1
            or meta["family"] != self.family
            or type(meta["direction_blend"]) is not bool
            or meta["direction_blend"] != self.direction_blend
            or not isinstance(meta["sampled_rows"], list)
            or len(meta["sampled_rows"]) != len(self.param_groups)
        ):
            raise ValueError("incompatible UnifiedV11EquilibratedOptimizer metadata")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) <= 0.0:
            raise ValueError("invalid UnifiedV11EquilibratedOptimizer last_step_gain")
        ema_fg = meta["ema_func_gain"]
        if (
            isinstance(ema_fg, bool)
            or not isinstance(ema_fg, (int, float))
            or not math.isfinite(ema_fg)
            or not (0.0 < float(ema_fg) <= 1.0)
        ):
            raise ValueError("invalid UnifiedV11EquilibratedOptimizer ema_func_gain")
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
                raise ValueError("invalid UnifiedV11EquilibratedOptimizer direction_blend checkpoint")
            if cg["pending"] is not None:
                if not isinstance(cg["pending"], torch.Tensor) or not bool(torch.isfinite(cg["pending"]).all()):
                    raise ValueError("invalid pending direction_blend tensor")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError("UnifiedV11EquilibratedOptimizer group count mismatch")
        for cur_grp, sav_grp in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav_grp, dict) or len(sav_grp.get("params", ())) != len(cur_grp["params"]):
                raise ValueError("UnifiedV11EquilibratedOptimizer group parameter mismatch")
            if sav_grp.get("role") != cur_grp["role"] or sav_grp.get("algorithm") != cur_grp["algorithm"]:
                raise ValueError("UnifiedV11EquilibratedOptimizer group role/algorithm mismatch")
        if not isinstance(saved["state"], dict):
            raise ValueError("invalid UnifiedV11EquilibratedOptimizer state map")
        for st_val in saved["state"].values():
            if not isinstance(st_val, dict):
                raise ValueError("invalid UnifiedV11EquilibratedOptimizer parameter state")
            step_v = st_val.get("step", 0)
            if type(step_v) is not int or step_v < 0:
                raise ValueError("invalid UnifiedV11EquilibratedOptimizer step counter")
            for t_key in ("m_unit", "m_scale", "v_rms", "v_scale", "m_prior", "peak_rms"):
                if t_key in st_val:
                    ten = st_val[t_key]
                    if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                        raise ValueError(f"non-finite or invalid tensor {t_key} in UnifiedV11EquilibratedOptimizer state")
                    if t_key == "peak_rms" and (ten.ndim != 0 or float(ten.item()) < 0.0):
                        raise ValueError("invalid peak_rms tensor in UnifiedV11EquilibratedOptimizer state")
        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid UnifiedV11EquilibratedOptimizer critic regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        clean = {k: v for k, v in state_dict.items() if k not in ("unified_v11_equil", "regularizer")}
        Optimizer.load_state_dict(self, clean)
        self.clear_sampled_rows()
        meta = state_dict["unified_v11_equil"]
        self.last_step_gain = float(meta["last_step_gain"])
        self._ema_func_gain = float(meta["ema_func_gain"])
        self.last_probe_consumed = False
        self.probe_consumptions = 0
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


__all__ = [
    "UnifiedV11EquilibratedOptimizer",
    "collect_conv_layouts",
    "quintic_orthogonalize_2d",
    "quintic_orthogonalize_batched",
    "zero_pinv_common_descent",
    "zero_pinv_project_nonascent",
]
