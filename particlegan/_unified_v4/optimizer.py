"""Zero-SVD Scale-Normalized Quintic Polar + Function-Space Secant Trust-Band Optimizer (`unified_equil`).

Unifies Law 4 (`t4_arch` + `t4_nash`) with `DualNorm` spectral geometry in strictly `0` `torch.linalg.svd`,
`0` `eigh`/`eigvalsh`, `0` `pinv`, and `0` `torch.autograd.grad` calls:
1. Underflow-safe ($10^{-38}$) dynamic scale-normalized moments + batched BF16 5-step Quintic Newton-Schulz
   polar orthogonalization across dense 2D layers and 4D Conv2d/ConvTranspose2d spatial offsets, plus exact
   closed-form rank-1 polar updates on `1 x C` / `R x 1` layers and L2-unit updates on 1D vectors.
2. Architectural depth scaling (`L > 5`) + momentum coherence gate + composite weight-norm trust clamping,
   preserving exact zero-momentum spectral DualNorm dynamics on shallow (`L <= 5`) networks while stabilizing
   16-layer skip-free (`grid100_mlp16`) and 16-block unscaled `tanh` (`grid100_res16_*`) architectures.
3. Detached `B_probe <= 16` function-space pushforward/logit secant trust-band regulation (`alpha_g`, `alpha_d`),
   clamping ballistic overshoot on `>548x` overpowered networks and amplifying compressed/inverted hostile
   initializations while remaining a strict no-op (`alpha == 1.0`) inside the balanced operating deadband.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

_QUINTIC_COEFFS: Tuple[Tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

_ALLOWED_ROLES = frozenset({"generator", "encoder", "router", "noise", "critic", "prior", "table"})


def _smoothstep01(x: float) -> float:
    u = min(1.0, max(0.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


def _band_scale(
    delta: float,
    delta_min: float,
    delta_max: float,
    *,
    alpha_max: float = 48.0,
    alpha_min: float = 1e-4,
) -> float:
    """Smooth impedance-matching gain keeping realized function-space step inside ``[delta_min, delta_max]``."""
    if not math.isfinite(delta) or delta <= 1e-12:
        return 1.0
    if delta_max > 0.0 and delta > delta_max:
        return float(max(alpha_min, delta_max / delta))
    if delta_min > 0.0 and delta < delta_min:
        gain = delta_min / (delta + delta_min / alpha_max)
        return float(min(alpha_max, max(1.0, gain)))
    return 1.0


def collect_conv_layouts(modules: Iterable[Optional[nn.Module]]) -> Dict[int, Tuple[bool, int]]:
    """Map ``id(weight) -> (is_conv_transpose, groups)`` for bound Conv2d / ConvTranspose2d modules."""
    layouts: Dict[int, Tuple[bool, int]] = {}
    for root in modules:
        if not isinstance(root, nn.Module):
            continue
        for child in root.modules():
            if isinstance(child, nn.ConvTranspose2d):
                layouts[id(child.weight)] = (True, int(child.groups))
            elif isinstance(child, nn.Conv2d):
                layouts[id(child.weight)] = (False, int(child.groups))
    return layouts


@torch.no_grad()
def _eval_probe_forward(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    record: Any,
) -> Optional[torch.Tensor]:
    """Evaluate ``module(*args, **kwargs)`` in ``eval()`` mode without mutating RNGs, hooks, or BN buffers."""
    modes = [(m, m.training) for m in module.modules()]
    prev_probe = bool(getattr(record, "_in_probe", False)) if record is not None else False
    if record is not None:
        record._in_probe = True
    try:
        module.eval()
        first_p = next(module.parameters(), None)
        use_bf16 = (
            first_p is not None
            and first_p.device.type == "cuda"
            and bool(getattr(module, "_unified_v4_bf16_autocast", False))
        )
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = module(*args, **kwargs)
        else:
            out = module(*args, **kwargs)
        target = out[0] if isinstance(out, (tuple, list)) and out else out
        if not isinstance(target, torch.Tensor):
            return None
        return target.detach().to(dtype=torch.float32)
    finally:
        if record is not None:
            record._in_probe = prev_probe
        for m, flag in modes:
            m.training = flag


@torch.no_grad()
def quintic_orthogonalize_2d(
    mat: torch.Tensor,
    steps: int = 5,
    *,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Zero-SVD scale-normalized 5-step Quintic Newton-Schulz polar (and exact Tikhonov-smoothed polar) on ``[R, C]``."""
    if mat.ndim != 2:
        raise ValueError(f"quintic_orthogonalize_2d expects a 2D matrix, got shape {tuple(mat.shape)}")
    r_dim, c_dim = int(mat.shape[0]), int(mat.shape[1])
    if r_dim == 0 or c_dim == 0:
        return torch.zeros_like(mat)
    orig_dtype = mat.dtype
    x_f32 = mat.float()
    s_val = float(smoothing)
    max_abs = x_f32.abs().amax()
    finite_mat = torch.isfinite(x_f32).all()
    if s_val > 0.0 and min(r_dim, c_dim) <= 128:
        scale_ref = torch.maximum(torch.where(finite_mat, max_abs, max_abs.new_zeros(())), max_abs.new_tensor(s_val))
        safe_max = torch.where(scale_ref > 0.0, scale_ref, torch.ones_like(scale_ref))
        x_scaled = torch.where(finite_mat, x_f32 / safe_max, torch.zeros_like(x_f32))
        s_scaled = s_val / safe_max
        frob = x_scaled.norm()
        active = finite_mat & (frob > 1e-30)
        if r_dim == 1 or c_dim == 1:
            unit = x_scaled / torch.hypot(frob, s_scaled).clamp_min(1e-30)
            return torch.where(active, unit, torch.zeros_like(unit)).to(orig_dtype)
        transposed = r_dim > c_dim
        x_in = x_scaled.T if transposed else x_scaled
        m_dim = int(x_in.shape[0])
        aug_frob = torch.sqrt(frob.square() + float(m_dim) * s_scaled.square()).clamp_min(1e-30)
        inv_aug = 1.0 / (aug_frob * 1.02)
        work_dtype = torch.bfloat16 if mat.is_cuda else torch.float32
        x = (x_in * inv_aug).to(dtype=work_dtype)
        y = (torch.eye(m_dim, device=mat.device, dtype=torch.float32) * (s_scaled * inv_aug)).to(dtype=work_dtype)
        for a, b, c in _QUINTIC_COEFFS[:steps]:
            gram = x @ x.T + y @ y.T
            gram2 = gram @ gram
            poly = b * gram + c * gram2
            x = a * x + poly @ x
            y = a * y + poly @ y
        if transposed:
            x = x.T
        out = x.float()
        joint_frob = torch.sqrt(out.square().sum() + y.float().square().sum()).clamp_min(1e-8)
        out = out * (math.sqrt(float(m_dim)) / joint_frob)
        out = torch.where(active & torch.isfinite(out).all(), out, torch.zeros_like(out))
        return out.to(orig_dtype)

    safe_max = torch.where(finite_mat & (max_abs > 0.0), max_abs, torch.ones_like(max_abs))
    x_scaled = torch.where(finite_mat, x_f32 / safe_max, torch.zeros_like(x_f32))
    frob = x_scaled.norm()
    active = finite_mat & (frob > 1e-30)
    # Closed-form exact polar factor for rank-1 shapes (R == 1 or C == 1):
    if r_dim == 1 or c_dim == 1:
        unit = x_scaled / frob.clamp_min(1e-30)
        return torch.where(active, unit, torch.zeros_like(unit)).to(orig_dtype)

    x_norm = x_scaled / (frob * 1.02 + 1e-30)
    transposed = r_dim > c_dim
    if transposed:
        x_norm = x_norm.T

    work_dtype = torch.bfloat16 if mat.is_cuda else torch.float32
    x = x_norm.to(dtype=work_dtype)
    for a, b, c in _QUINTIC_COEFFS[:steps]:
        gram = x @ x.T
        gram2 = gram @ gram
        x = a * x + (b * gram + c * gram2) @ x

    if transposed:
        x = x.T
    out = x.float()
    # Match exact SVD polar Frobenius norm sqrt(min(R, C)) on active matrices (with fast scalar RMS smoothing when s_val>0):
    target_frob = math.sqrt(float(min(r_dim, c_dim)))
    out_frob = out.norm().clamp_min(1e-8)
    if s_val > 0.0:
        g_rms = (frob * safe_max) / max(target_frob, 1.0)
        s_scale = g_rms / torch.hypot(g_rms, g_rms.new_tensor(s_val)).clamp_min(1e-30)
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
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Batched zero-SVD 5-step Quintic Newton-Schulz polar (and exact Tikhonov-smoothed polar) on ``[B, R, C]``."""
    if mats.ndim != 3:
        raise ValueError(f"quintic_orthogonalize_batched expects a 3D batch, got shape {tuple(mats.shape)}")
    b_sz, r_dim, c_dim = int(mats.shape[0]), int(mats.shape[1]), int(mats.shape[2])
    if b_sz == 0 or r_dim == 0 or c_dim == 0:
        return torch.zeros_like(mats)
    orig_dtype = mats.dtype
    x_f32 = mats.float()
    s_val = float(smoothing)
    max_abs = x_f32.abs().amax(dim=(1, 2), keepdim=True)
    finite_mat = torch.isfinite(x_f32).all(dim=(1, 2), keepdim=True)
    if s_val > 0.0 and min(r_dim, c_dim) <= 128:
        scale_ref = torch.maximum(torch.where(finite_mat, max_abs, torch.zeros_like(max_abs)), max_abs.new_tensor(s_val))
        safe_max = torch.where(scale_ref > 0.0, scale_ref, torch.ones_like(scale_ref))
        x_scaled = torch.where(finite_mat, x_f32 / safe_max, torch.zeros_like(x_f32))
        s_scaled = s_val / safe_max
        frob = x_scaled.flatten(1).norm(dim=1).view(b_sz, 1, 1)
        active = finite_mat & (frob > 1e-30)
        if r_dim == 1 or c_dim == 1:
            unit = x_scaled / torch.hypot(frob, s_scaled).clamp_min(1e-30)
            return torch.where(active, unit, torch.zeros_like(unit)).to(orig_dtype)
        transposed = r_dim > c_dim
        x_in = x_scaled.transpose(1, 2) if transposed else x_scaled
        m_dim = int(x_in.shape[1])
        aug_frob = torch.sqrt(frob.square() + float(m_dim) * s_scaled.square()).clamp_min(1e-30)
        inv_aug = 1.0 / (aug_frob * 1.02)
        work_dtype = torch.bfloat16 if mats.is_cuda else torch.float32
        x = (x_in * inv_aug).to(dtype=work_dtype)
        eye_b = torch.eye(m_dim, device=mats.device, dtype=torch.float32).unsqueeze(0).expand(b_sz, m_dim, m_dim)
        y = (eye_b * (s_scaled * inv_aug)).to(dtype=work_dtype)
        for a, b, c in _QUINTIC_COEFFS[:steps]:
            gram = torch.bmm(x, x.transpose(1, 2)) + torch.bmm(y, y.transpose(1, 2))
            gram2 = torch.bmm(gram, gram)
            poly = b * gram + c * gram2
            x = a * x + torch.bmm(poly, x)
            y = a * y + torch.bmm(poly, y)
        if transposed:
            x = x.transpose(1, 2)
        out = x.float()
        joint_frob = torch.sqrt(
            out.flatten(1).square().sum(dim=1) + y.float().flatten(1).square().sum(dim=1)
        ).view(b_sz, 1, 1).clamp_min(1e-8)
        out = out * (math.sqrt(float(m_dim)) / joint_frob)
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

    work_dtype = torch.bfloat16 if mats.is_cuda else torch.float32
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
    if s_val > 0.0:
        g_rms = (frob * safe_max) / max(target_frob, 1.0)
        s_scale = g_rms / torch.hypot(g_rms, g_rms.new_tensor(s_val)).clamp_min(1e-30)
        out = out * ((target_frob / out_frob) * s_scale)
    else:
        out = out * (target_frob / out_frob)
    out_ok = active & torch.isfinite(out).all(dim=(1, 2), keepdim=True)
    out = torch.where(out_ok, out, torch.zeros_like(out))
    return out.to(orig_dtype)


class UnifiedV4EquilibratedOptimizer(Optimizer):
    """Role-aware zero-SVD equilibrated optimizer (`family='unified_equil'`)."""

    family = "unified_equil"
    recipe_optimizer_family = "unified_equil"

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 0.014,
        betas: Tuple[float, float] = (0.0, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        smoothing: float = 0.0,
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
        betas = tuple(float(b) for b in betas)
        if len(betas) != 2 or any(not (0.0 <= b < 1.0) or not math.isfinite(b) for b in betas):
            raise ValueError("betas must contain two finite values in [0, 1)")
        for key, value in options.items():
            if key not in {"weight_decay", "maximize", "differentiable", "capturable", "foreach", "fused"}:
                raise TypeError(f"unsupported unified_equil optimizer option: {key}")
            if value not in (None, False, 0, 0.0):
                raise ValueError(f"unified_equil requires disabled {key}")

        self.family = "unified_equil"
        self.recipe_optimizer_family = "unified_equil"
        self.smoothing = float(smoothing)
        self.latent_damping = None
        self.latent_history = None
        self.direct_response = None
        self.direct_history = None
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}
        self._row_scales: Dict[torch.Tensor, torch.Tensor] = {}
        self._conv_layouts: Dict[int, Tuple[bool, int]] = {}
        self._gen_param_ids: set[int] = set()
        self._v4_wakeup_boost: float = 1.0
        self._v4_g_scale: float = 1.0
        self._v4_prior_scale: float = 1.0
        self._origin_norm: Optional[float] = None
        self.last_step_gain: float = 1.0

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
                group["algorithm"] = "unified_equil"

    def _player(self, role: str) -> str:
        if role in ("prior", "table"):
            return "prior"
        if role == "critic":
            return "critic"
        return "generator"

    def set_sampled_rows(self, parameter: torch.Tensor, rows: torch.Tensor) -> None:
        if not isinstance(rows, torch.Tensor):
            rows = torch.as_tensor(rows, dtype=torch.long, device=parameter.device)
        rows = torch.unique(rows.reshape(-1).to(dtype=torch.long, device=parameter.device), sorted=True)
        if rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(parameter)).any())):
            raise ValueError("sampled rows out of bounds for prior table")
        existing = self._sampled_rows.get(parameter)
        if existing is not None:
            rows = torch.unique(torch.cat((existing, rows)), sorted=True)
        self._sampled_rows[parameter] = rows

    def sampled_rows_for(self, parameter: torch.Tensor) -> Optional[torch.Tensor]:
        rows = self._sampled_rows.get(parameter)
        return None if rows is None else rows.detach().clone()

    def set_row_scales(self, parameter: torch.Tensor, scales: torch.Tensor) -> None:
        self._row_scales[parameter] = scales.detach().to(device=parameter.device, dtype=torch.float32)

    def clear_sampled_rows(self) -> None:
        self._sampled_rows.clear()
        self._row_scales.clear()

    def bind_origin_sample(self, sample: Optional[torch.Tensor]) -> None:
        if sample is not None and sample.ndim == 2 and sample.shape[1] == 2 and sample.numel() > 0:
            self._origin_norm = float(sample.detach().float().norm(dim=-1).amax().item())
        else:
            self._origin_norm = None

    def bind_v4_modulation(
        self,
        *,
        g_scale: float = 1.0,
        wakeup_boost: float = 1.0,
        prior_scale: float = 1.0,
    ) -> None:
        self._v4_g_scale = float(g_scale)
        self._v4_wakeup_boost = float(wakeup_boost)
        self._v4_prior_scale = float(prior_scale)

    def abort_pending(self) -> None:
        self.clear_sampled_rows()
        self._v4_wakeup_boost = 1.0
        self._v4_g_scale = 1.0
        self._v4_prior_scale = 1.0
        self._origin_norm = None

    def __getstate__(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def _active_record(self) -> Any:
        return (
            getattr(self, "_unified_v4_record", None)
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

        # Validate gradient types & sampled-row ownership before mutating any state:
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.grad.is_complex():
                    raise ValueError("unified_equil requires dense real gradients")
                if (
                    grp["algorithm"] == "unified_prior"
                    and grp.get("sampled_rows_required", True)
                    and p not in self._sampled_rows
                ):
                    raise ValueError("row-normalized prior updates require actual sampled rows")

        try:
            rec = self._active_record()
            step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
            s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            cond_active = bool(getattr(rec, "t1c_cond_initialized", False)) if rec is not None else False
            # Thermal cooling scales critic logits in _unified_v4_critic_forward_hook; match unified-v3
            # rhbi_step_damp=0.0 on neural networks while damping generator-free direct particle clouds.
            therm_g = 1.0
            therm_d = 1.0
            if s_therm < 1.0:
                u_log = min(1.0, max(0.0, -math.log10(max(1e-6, s_therm)) * (1.0 / 6.0)))
                u_log4 = (u_log * u_log) * (u_log * u_log)
                therm_direct_prior = 1.0 - 0.80 * u_log4
            else:
                therm_direct_prior = 1.0
            is_critic_opt = getattr(self, "critic", None) is not None

            origin_damping = 1.0
            if self._origin_norm is not None and self._origin_norm < 0.05:
                origin_damping = min(1.0, self._origin_norm / 0.05)

            n_mat_role: Dict[str, int] = {}
            total_gen_mats = 0
            for grp in self.param_groups:
                rl = grp["role"]
                pl = self._player(rl)
                for p in grp["params"]:
                    if pl != "prior" and p.ndim >= 2:
                        n_mat_role[rl] = n_mat_role.get(rl, 0) + 1
                        if pl == "generator":
                            total_gen_mats += 1

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for grp in self.param_groups:
                role = grp["role"]
                player = self._player(role)
                b1, b2 = grp["betas"]
                eps = float(grp["eps"])
                L_raw = max(1, n_mat_role.get(role, 1))
                L_count = 4 if (player == "generator" and L_raw == 2 and total_gen_mats >= 4) else L_raw
                deep_gate = _smoothstep01((float(L_count) - 5.0) / 4.0)
                s_eff = self.smoothing * (1.0 - deep_gate)
                s_prior = self.smoothing if (self._v4_prior_scale >= 1.0 and total_gen_mats > 0) else 0.0

                if player == "prior":
                    b1_eff = max(float(b1), 0.75) if self._v4_prior_scale < 1.0 else float(b1)
                elif cond_active:
                    b1_eff = max(float(b1), 0.82)
                elif deep_gate > 0.0:
                    b1_eff = max(float(b1), 0.75 * deep_gate)
                else:
                    b1_eff = float(b1)
                b2_eff = min(max(float(b2), 0.95), 0.999)
                c_zero = 1.0 / math.sqrt(float(max(1, L_count)))

                if player == "generator":
                    depth_scale = (
                        min(1.0, (3.2 / float(max(3, L_count))) * self._v4_wakeup_boost)
                        if deep_gate > 0.0
                        else 1.0
                    )
                    eff_depth = (1.0 - deep_gate) + deep_gate * depth_scale
                    base_lr = float(grp["lr"]) * step_scale * therm_g * self._v4_g_scale * eff_depth
                elif player == "critic":
                    depth_scale = (3.2 / float(max(3, L_count))) if deep_gate > 0.0 else 1.0
                    eff_depth = (1.0 - deep_gate) + deep_gate * depth_scale
                    base_lr = float(grp["lr"]) * step_scale * therm_d * eff_depth
                else:
                    therm_p = therm_direct_prior if total_gen_mats == 0 else 1.0
                    base_lr = float(grp["lr"]) * step_scale * therm_p * self._v4_prior_scale

                pending_dense: Dict[
                    Tuple[int, int],
                    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]],
                ] = {}
                pending_conv: Dict[
                    Tuple[int, int, int, int, bool, int],
                    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]],
                ] = {}

                for p in grp["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    st = self.state[p]
                    step_t = int(st.get("step", 0)) + 1
                    st["step"] = step_t

                    if player == "prior":
                        rows = (
                            self._sampled_rows[p]
                            if grp.get("sampled_rows_required", True)
                            else torch.arange(len(p), device=p.device, dtype=torch.long)
                        )
                        if rows.numel() == 0 or base_lr <= 0.0:
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
                        if s_prior > 0.0:
                            s_p_row = s_prior * (128.0 / float(len(rows))) if len(rows) > 128 else s_prior
                            denom_r = torch.hypot(d_norm, d_norm.new_tensor(s_p_row)).clamp_min(1e-36)
                        else:
                            denom_r = d_norm.clamp_min(1e-36)
                        unit_r = torch.where(
                            active_r,
                            dir_r / denom_r,
                            torch.zeros_like(dir_r),
                        )
                        r_scales = self._row_scales.get(p)
                        if r_scales is not None and len(r_scales) == len(rows):
                            unit_r = unit_r * r_scales.unsqueeze(1)
                        p.index_add_(0, rows, unit_r.to(p.dtype), alpha=-base_lr)
                        continue

                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                    gf = grad.float()
                    finite = torch.isfinite(gf).all()
                    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                    g_max = gf_safe.abs().amax()
                    g_active = finite & (g_max > 1e-38)

                    # Maintain underflow-safe moments when deep_gate > 0 or 1D scalar parameter:
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    use_moments = (b1_eff > 0.0) or (deep_gate > 0.0) or is_scalar_1d
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
                        elif deep_gate > 0.0:
                            coh = (1.0 - deep_gate) + deep_gate * raw_coh
                        else:
                            coh = torch.ones_like(m_norm)
                        raw_dir = (m_hat * sc) if s_eff > 0.0 else m_hat
                        direc = torch.where(g_active, raw_dir, torch.zeros_like(m_hat))
                    else:
                        raw_dir = gf_safe if s_eff > 0.0 else (gf_safe / g_max.clamp_min(1e-38))
                        direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
                        coh = gf_safe.new_ones(())

                    if rate_p <= 0.0:
                        continue

                    if p.ndim == 4 and (p.shape[2] > 1 or p.shape[3] > 1):
                        c0, c1, kh, kw = (int(s) for s in p.shape)
                        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                        out_ch = c1 * groups_c if is_trans else c0
                        in_ch = c0 if is_trans else c1 * groups_c
                        factor = math.sqrt(float(out_ch) / float(max(1, in_ch))) / float(kh * kw)
                        if deep_gate > 0.0:
                            w_op = 1.0 / math.sqrt(float(math.sqrt(float(max(1, out_ch * in_ch))) * float(kh * kw)))
                            w_rms = p.float().square().mean().sqrt()
                            r_ratio = (w_rms / max(w_op, 1e-8)).clamp(c_zero, 2.5)
                            trust_mult = (1.0 - deep_gate) + deep_gate * r_ratio
                        else:
                            trust_mult = direc.new_ones(())
                        if s_eff > 0.0:
                            g_slice_rms = (direc.norm() / math.sqrt(float(max(1, min(out_ch, in_ch) * kh * kw)))).clamp_min(0.0)
                            conv_smooth = g_slice_rms / torch.hypot(g_slice_rms, g_slice_rms.new_tensor(s_eff)).clamp_min(1e-30)
                            trust_mult = trust_mult * conv_smooth
                        step_mult = (-rate_p * factor) * coh * trust_mult
                        pending_conv.setdefault((c0, c1, kh, kw, is_trans, groups_c), []).append(
                            (p, direc, step_mult, track_secant)
                        )
                    elif p.ndim >= 2:
                        mat = direc.reshape(p.shape[0], -1)
                        r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                        factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
                        if deep_gate > 0.0:
                            w_op = 1.0 / math.sqrt(float(max(1, c_d)))
                            w_rms = p.float().square().mean().sqrt()
                            r_ratio = (w_rms / max(w_op, 1e-8)).clamp(c_zero, 2.5)
                            trust_mult = (1.0 - deep_gate) + deep_gate * r_ratio
                        else:
                            trust_mult = direc.new_ones(())
                        step_mult = (-rate_p * factor) * coh * trust_mult
                        pending_dense.setdefault((r_d, c_d), []).append((p, mat, step_mult, track_secant))
                    else:
                        d_norm = direc.norm()
                        if s_eff > 0.0:
                            denom_1d = torch.hypot(d_norm, d_norm.new_tensor(s_eff)).clamp_min(eps)
                        else:
                            denom_1d = d_norm + eps
                        unit_1d = torch.where(
                            g_active & (d_norm > 1e-30),
                            direc / denom_1d,
                            torch.zeros_like(direc),
                        )
                        delta_1d = (unit_1d * ((-rate_p) * coh)).to(p.dtype)
                        p.add_(delta_1d)
                        if track_secant:
                            applied_deltas.append((p, delta_1d))

                for _, bucket in pending_dense.items():
                    if len(bucket) == 1:
                        p_i, mat_i, mult_i, trk_i = bucket[0]
                        orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=s_eff).float()
                        delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        if trk_i:
                            applied_deltas.append((p_i, delta_i))
                    else:
                        mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                        orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_eff).float()
                        for idx_b, (p_i, _, mult_i, trk_i) in enumerate(bucket):
                            delta_i = (orths[idx_b] * mult_i).reshape_as(p_i).to(p_i.dtype)
                            p_i.add_(delta_i)
                            if trk_i:
                                applied_deltas.append((p_i, delta_i))

                for (c0, c1, kh, kw, is_trans, groups_c), bucket in pending_conv.items():
                    n_b = len(bucket)
                    k_slices = groups_c * kh * kw
                    per_c0 = c0 // groups_c
                    slices_list = []
                    for _, d_t, _, _ in bucket:
                        st_view = d_t.reshape(groups_c, per_c0, c1, kh, kw)
                        if is_trans:
                            # [groups, kh, kw, out/g=c1, in/g=per_c0]
                            sl = st_view.permute(0, 3, 4, 2, 1).reshape(k_slices, c1, per_c0)
                        else:
                            # [groups, kh, kw, out/g=per_c0, in/g=c1]
                            sl = st_view.permute(0, 3, 4, 1, 2).reshape(k_slices, per_c0, c1)
                        slices_list.append(sl)
                    all_slices = torch.cat(slices_list, dim=0)
                    orth_all = quintic_orthogonalize_batched(all_slices, steps=5).float()
                    r_sl, c_sl = (c1, per_c0) if is_trans else (per_c0, c1)
                    orth_layers = orth_all.view(n_b, groups_c, kh, kw, r_sl, c_sl)
                    for idx_b, (p_i, _, mult_i, trk_i) in enumerate(bucket):
                        if is_trans:
                            orth_i = orth_layers[idx_b].permute(0, 4, 3, 1, 2).reshape(c0, c1, kh, kw)
                        else:
                            orth_i = orth_layers[idx_b].permute(0, 3, 4, 1, 2).reshape(c0, c1, kh, kw)
                        delta_i = (orth_i * mult_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        if trk_i:
                            applied_deltas.append((p_i, delta_i))

            # Law 4 Two-Player Function-Space Secant Trust Band (no-op inside balanced operating deadband):
            alpha_gain = 1.0
            if rec is not None and applied_deltas and getattr(rec, "_enable_secant_probe", True):
                if is_critic_opt:
                    probes = getattr(rec, "_d_probe_list", None)
                    if probes and getattr(self, "critic", None) is not None:
                        sq_diffs = []
                        for p_args, p_kwargs, d_pre in probes:
                            d_post = _eval_probe_forward(self.critic, p_args, p_kwargs, rec)
                            if d_post is not None and d_post.numel() == d_pre.numel():
                                sq_diffs.append((d_post.reshape(-1) - d_pre.reshape(-1)).square().mean())
                        if sq_diffs:
                            delta_d = float(torch.stack(sq_diffs).mean().sqrt().item())
                            ema_m = float(getattr(rec, "ema_margin", 0.0)) if getattr(rec, "margin_initialized", False) else 0.0
                            inv_gap = min(25.0, max(0.0, -ema_m - 1.5))
                            s_th = float(getattr(rec, "_thermal_scale", 1.0))
                            th_d = max(0.15, math.sqrt(s_th)) if s_th < 1.0 else 1.0
                            delta_d_min = 0.12 * inv_gap * th_d
                            delta_d_max = (0.55 * step_scale + 0.20 * inv_gap) * th_d
                            alpha_gain = _band_scale(delta_d, delta_d_min, delta_d_max, alpha_max=64.0)
                else:
                    probe_cache = getattr(rec, "_g_probe_cache", None)
                    rec._g_probe_cache = None
                    if probe_cache is not None:
                        g_mod, p_args, p_kwargs, y_pre = probe_cache
                        y_post = _eval_probe_forward(g_mod, p_args, p_kwargs, rec)
                        if y_post is not None and y_post.shape == y_pre.shape:
                            out_dim = max(1, int(y_pre[0].numel()))
                            delta_rms = float((y_post - y_pre).square().mean().sqrt().item())
                            data_scale = float(getattr(rec, "ema_data_scale", 1.0))
                            coord_scale = max(0.05, data_scale / math.sqrt(float(out_dim)))
                            delta_rel = delta_rms / coord_scale
                            push_res = float(getattr(rec, "ema_push_res", 0.0))
                            s_th = float(getattr(rec, "_thermal_scale", 1.0))
                            delta_g_max = (0.045 + 0.105 * min(1.5, max(0.25, push_res))) * step_scale
                            if push_res > 0.18 and s_th > 0.10:
                                off_gate = _smoothstep01((push_res - 0.18) / 0.25)
                                delta_g_min = 0.012 * off_gate * step_scale
                            else:
                                delta_g_min = 0.0
                            alpha_gain = _band_scale(delta_rel, delta_g_min, delta_g_max, alpha_max=48.0)
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, d_i in applied_deltas:
                        p_i.add_(d_i, alpha=adj)

            self.last_step_gain = float(alpha_gain)
        finally:
            self.abort_pending()

        if hasattr(self, "record") and self.record is not None:
            self.record.record_step(self)
        return loss

    def state_dict(self) -> Dict[str, Any]:
        result = Optimizer.state_dict(self)
        result["unified_equil"] = {
            "schema": 1,
            "family": self.family,
            "last_step_gain": float(self.last_step_gain),
            "sampled_rows": [
                self._sampled_rows.get(grp["params"][0]) if grp["algorithm"] == "unified_prior" else None
                for grp in self.param_groups
            ],
        }
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

        expected_keys = {"state", "param_groups", "unified_equil"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError("invalid UnifiedV4EquilibratedOptimizer checkpoint schema")
        meta = saved["unified_equil"]
        if (
            not isinstance(meta, dict)
            or set(meta) != {"schema", "family", "last_step_gain", "sampled_rows"}
            or type(meta["schema"]) is not int
            or meta["schema"] != 1
            or meta["family"] != self.family
            or not isinstance(meta["sampled_rows"], list)
            or len(meta["sampled_rows"]) != len(self.param_groups)
        ):
            raise ValueError("incompatible UnifiedV4EquilibratedOptimizer metadata")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) <= 0.0:
            raise ValueError("invalid UnifiedV4EquilibratedOptimizer last_step_gain")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError("UnifiedV4EquilibratedOptimizer group count mismatch")
        for cur_grp, sav_grp in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav_grp, dict) or len(sav_grp.get("params", ())) != len(cur_grp["params"]):
                raise ValueError("UnifiedV4EquilibratedOptimizer group parameter mismatch")
            if sav_grp.get("role") != cur_grp["role"] or sav_grp.get("algorithm") != cur_grp["algorithm"]:
                raise ValueError("UnifiedV4EquilibratedOptimizer group role/algorithm mismatch")
        if not isinstance(saved["state"], dict):
            raise ValueError("invalid UnifiedV4EquilibratedOptimizer state map")
        for st_val in saved["state"].values():
            if not isinstance(st_val, dict):
                raise ValueError("invalid UnifiedV4EquilibratedOptimizer parameter state")
            step_v = st_val.get("step", 0)
            if type(step_v) is not int or step_v < 0:
                raise ValueError("invalid UnifiedV4EquilibratedOptimizer step counter")
            for t_key in ("m_unit", "m_scale", "v_rms", "v_scale", "m_prior"):
                if t_key in st_val:
                    ten = st_val[t_key]
                    if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                        raise ValueError(f"non-finite or invalid tensor {t_key} in UnifiedV4EquilibratedOptimizer state")
        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid UnifiedV4EquilibratedOptimizer critic regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        clean = {k: v for k, v in state_dict.items() if k not in ("unified_equil", "regularizer")}
        Optimizer.load_state_dict(self, clean)
        self.clear_sampled_rows()
        self.last_step_gain = float(state_dict["unified_equil"]["last_step_gain"])
        for grp, rows in zip(self.param_groups, state_dict["unified_equil"]["sampled_rows"]):
            if rows is not None and grp["algorithm"] == "unified_prior":
                self.set_sampled_rows(grp["params"][0], rows.to(grp["params"][0].device))
        if hasattr(self, "record") and self.record is not None:
            self.record.load_state_dict(state_dict["regularizer"]["record"])
            if self.param_groups and self.param_groups[0]["params"]:
                self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "UnifiedV4EquilibratedOptimizer",
    "collect_conv_layouts",
    "quintic_orthogonalize_2d",
    "quintic_orthogonalize_batched",
]
