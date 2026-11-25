"""Pure-elementwise O(params) Function-Space Impedance-Matched Adam optimizers for T4NashV3.

Strictly 0 SVD, 0 eigh, 0 pinv, 0 Newton-Schulz, and 0 torch.autograd.grad calls:
1. Degree-0 scale-invariant ``peak_rms`` equilibrium settling + closed-form
   Marchenko-Pastur RMS-to-RMS operator scaling across arbitrary layer widths.
2. Realized low-D function-space secant trust-band regulation (generator pushforward
   band + critic relativistic inversion-deficit band) on tiny detached probe slices
   (``B_probe <= 16``, ``eval()`` mode, zero RNG/BN mutation; skipped on high-D images).
3. Closed-form 1x1 / 2x2 Cramer's-rule non-ascent projection on
   ``NashV3ConstraintGeneratorAdam`` using standard 1st-order ``.backward()``
   with saved/restored ``.grad`` buffers (zero ``torch.autograd.grad`` calls).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn

from ..k3p import K3PCriticAdam, K3PGeneratorAdam, _closure_loss
from .record import NashV3StepRecord, _attach_nash_v3_critic_hook, _smoothstep01


def _adam_unit_direction(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: Dict[str, Any],
    betas: Tuple[float, float],
    eps: float,
    amsgrad: bool,
    *,
    settle_rel: float = 0.015,
) -> torch.Tensor:
    """Update Adam moments in ``state`` with scale-invariant peak-RMS equilibrium settling."""
    if len(state) == 0:
        state["step"] = torch.tensor(0.0, dtype=torch.float32)
        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["peak_rms"] = torch.zeros((), dtype=param.dtype, device=param.device)
        if amsgrad:
            state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
    elif "peak_rms" not in state:
        state["peak_rms"] = torch.zeros((), dtype=param.dtype, device=param.device)

    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    beta1, beta2 = float(betas[0]), float(betas[1])

    step_t = state["step"]
    if isinstance(step_t, torch.Tensor):
        step_t.add_(1.0)
        step_int = int(step_t.item())
    else:
        step_int = int(step_t) + 1
        state["step"] = torch.tensor(float(step_int), dtype=torch.float32)

    g = grad.detach()
    exp_avg.lerp_(g, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

    bias_correction1 = 1.0 - (beta1 ** step_int)
    bias_correction2 = 1.0 - (beta2 ** step_int)

    if amsgrad:
        max_exp_avg_sq = state["max_exp_avg_sq"]
        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        v_hat_sqrt = max_exp_avg_sq.to(dtype=torch.float32).sqrt() / math.sqrt(bias_correction2)
    else:
        v_hat_sqrt = exp_avg_sq.to(dtype=torch.float32).sqrt() / math.sqrt(bias_correction2)

    cur_rms = (v_hat_sqrt.square().mean() + 1e-36).sqrt()
    peak_prev = state["peak_rms"].to(dtype=torch.float32)
    peak_new = torch.maximum(peak_prev * 0.9995, cur_rms)
    state["peak_rms"].copy_(peak_new.to(dtype=state["peak_rms"].dtype))

    denom = v_hat_sqrt + (float(settle_rel) * peak_new + float(eps))
    m_hat = exp_avg.to(dtype=torch.float32) / bias_correction1
    return m_hat / denom


def _impedance_normalized_direction(
    param: torch.Tensor,
    u_adam: torch.Tensor,
    *,
    is_prior_table: bool = False,
    sampled_rows: torch.Tensor | None = None,
    is_direct_particles: bool = False,
    conv_transpose: bool = False,
    conv_groups: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Closed-form O(params) Marchenko-Pastur RMS-to-RMS operator normalization (0 SVD, 0 NS)."""
    if is_prior_table and param.ndim == 2:
        rows = sampled_rows if sampled_rows is not None else torch.arange(param.shape[0], device=param.device)
        if rows.numel() == 0:
            return u_adam.new_zeros((0, param.shape[1])), rows
        sel = u_adam[rows]
        row_norm = (sel.square().sum(dim=1, keepdim=True) + 1e-12).sqrt()
        denom = torch.hypot(row_norm, torch.full_like(row_norm, 0.04))
        return (sel / denom).to(dtype=param.dtype), rows

    if is_direct_particles and param.ndim >= 2:
        flat = u_adam.flatten(1)
        row_rms = (flat.square().mean(dim=1) + 1e-12).sqrt()
        denom = torch.hypot(row_rms, torch.full_like(row_rms, 0.025))
        view_shape = (param.shape[0],) + (1,) * (param.ndim - 1)
        return (u_adam / denom.reshape(view_shape)).to(dtype=param.dtype), None

    if param.ndim == 2:
        d_out, d_in = int(param.shape[0]), int(param.shape[1])
        fan = math.sqrt(max(1.0, float(d_out) / max(1.0, float(d_in))))
        if d_out == 1 and d_in >= 16:
            fan *= 0.65
        mp_denom = math.sqrt(max(1.0, float(d_out))) + math.sqrt(max(1.0, float(d_in)))
        return (u_adam * (fan / mp_denom)).to(dtype=param.dtype), None

    if param.ndim >= 3:
        r_dim, c_dim = int(param.shape[0]), int(param.shape[1])
        k_prod = math.prod(int(s) for s in param.shape[2:])
        g = max(1, int(conv_groups))
        if conv_transpose:
            d_in, d_out = float(r_dim), float(c_dim * g)
            k_eff = float(max(1, k_prod // 4))
        else:
            d_out, d_in = float(r_dim), float(c_dim * g)
            k_eff = float(max(1, k_prod))
        fan = math.sqrt(max(0.5, d_out / max(1.0, d_in)))
        mp_denom = (math.sqrt(max(1.0, float(r_dim))) + math.sqrt(max(1.0, float(c_dim)))) * math.sqrt(k_eff)
        return (u_adam * (fan / mp_denom)).to(dtype=param.dtype), None

    vec_norm = (u_adam.square().sum() + 1e-12).sqrt()
    scale_1d = math.sqrt(min(2.0, float(max(1, param.numel()))))
    return (u_adam * (scale_1d / torch.hypot(vec_norm, torch.full_like(vec_norm, 0.05)))).to(dtype=param.dtype), None


def _collect_conv_layouts(modules: Iterable[nn.Module | None]) -> Dict[int, Tuple[bool, int]]:
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
) -> torch.Tensor | None:
    """Evaluate ``module(*args, **kwargs)`` in ``eval()`` mode without mutating RNGs or BN buffers."""
    modes = [(m, m.training) for m in module.modules()]
    prev_probe = bool(getattr(record, "_in_probe", False))
    record._in_probe = True
    try:
        module.eval()
        first_p = next(module.parameters(), None)
        use_bf16 = (
            first_p is not None
            and first_p.device.type == "cuda"
            and bool(getattr(module, "_particlegan_bf16_autocast", getattr(module, "_nash_v3_bf16_autocast", False)))
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
        record._in_probe = prev_probe
        for m, flag in modes:
            m.training = flag


def _band_scale(
    delta: float,
    delta_min: float,
    delta_max: float,
    *,
    alpha_max: float = 12.0,
    alpha_min: float = 1e-4,
) -> float:
    if not math.isfinite(delta) or delta <= 1e-12:
        return 1.0
    if delta > delta_max and delta_max > 0.0:
        return float(max(alpha_min, delta_max / delta))
    if delta_min > 0.0 and delta < delta_min:
        gain = delta_min / (delta + delta_min / alpha_max)
        return float(min(alpha_max, max(1.0, gain)))
    return 1.0


def _project_nonascent_closed_form(displacement: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Exact closed-form 1x1 / 2x2 Cramer's rule non-ascent cone projection (0 SVD, 0 eigh, 0 pinv)."""
    d = displacement.to(dtype=torch.float64)
    a = normals.to(dtype=torch.float64)
    lengths = a.norm(dim=1)
    keep = lengths > 0.0
    a = a[keep] / lengths[keep, None]
    m = int(a.shape[0])
    if m > 2:
        raise ValueError("T4NashV3 constraint projection supports at most two protected losses")
    if m == 0:
        return displacement.clone()
    dots = a @ d
    if bool((dots <= 0.0).all()):
        return displacement.clone()
    tol = 1e-10 * max(1.0, float(d.norm().item()))
    best = torch.zeros_like(d)
    best_dist = float(d.square().sum().item())

    for idx in range(m):
        lam = float(dots[idx].item())
        if lam >= -tol:
            cand = d - lam * a[idx]
            if bool(((a @ cand) <= tol).all()):
                dist = float((cand - d).square().sum().item())
                if dist < best_dist:
                    best, best_dist = cand, dist

    if m == 2:
        c = float((a[0] @ a[1]).item())
        det = 1.0 - c * c
        if det > 1e-12:
            b0, b1 = float(dots[0].item()), float(dots[1].item())
            lam0 = (b0 - c * b1) / det
            lam1 = (b1 - c * b0) / det
            if lam0 >= -tol and lam1 >= -tol:
                cand = d - lam0 * a[0] - lam1 * a[1]
                if bool(((a @ cand) <= tol).all()):
                    dist = float((cand - d).square().sum().item())
                    if dist < best_dist:
                        best, best_dist = cand, dist

    return best.to(dtype=displacement.dtype)


def _validate_peak_rms_states(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st:
                continue
            if "peak_rms" not in st or not isinstance(st["peak_rms"], torch.Tensor):
                raise ValueError("T4NashV3 optimizer state is missing tensor 'peak_rms'")
            pr = st["peak_rms"]
            if pr.ndim != 0 or not bool(torch.isfinite(pr).all()) or float(pr.item()) < 0.0:
                raise ValueError("invalid T4NashV3 'peak_rms' scalar tensor")


class NashV3CriticAdam(K3PCriticAdam):
    """Pure-elementwise Function-Space Impedance-Matched Critic Adam for T4NashV3."""

    _OWN_ATTRS = (*K3PCriticAdam._OWN_ATTRS, "_conv_layouts", "last_step_gain")

    def __init__(
        self,
        params,
        *,
        critic: nn.Module,
        ema_critic: nn.Module | None = None,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs,
    ) -> None:
        del ema_critic
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = NashV3StepRecord()
        self._conv_layouts = _collect_conv_layouts((critic,))
        self.last_step_gain: float = 1.0
        _attach_nash_v3_critic_hook(critic, self.record)

    @torch.no_grad()
    def step(self, closure=None):
        loss = _closure_loss(closure)
        if self.guard is not None:
            self.guard.apply_(self)

        rec = self.record
        step_scale = float(getattr(rec, "_step_scale", 1.0))
        gamma_d = float(getattr(rec, "d_trust_scale", 1.0))
        s_therm = float(getattr(rec, "_thermal_scale", 1.0))
        therm_d = max(0.10, math.sqrt(s_therm)) if s_therm < 1.0 else 1.0

        probes = getattr(rec, "_d_probe_list", None)
        track_deltas = bool(probes and getattr(rec, "_enable_secant_probe", False))
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for group in self.param_groups:
            lr = float(group["lr"])
            betas = tuple(group["betas"])
            eps = float(group["eps"])
            amsgrad = bool(group.get("amsgrad", False))
            eff_lr = lr * step_scale * gamma_d * therm_d
            for p in group["params"]:
                if p.grad is None:
                    continue
                u_adam = _adam_unit_direction(p, p.grad, self.state[p], betas, eps, amsgrad)
                if eff_lr <= 0.0:
                    continue
                is_deconv, c_groups = self._conv_layouts.get(id(p), (False, 1))
                d_norm, _ = _impedance_normalized_direction(
                    p,
                    u_adam,
                    conv_transpose=is_deconv,
                    conv_groups=c_groups,
                )
                if track_deltas:
                    delta_p = d_norm * (-eff_lr)
                    p.add_(delta_p)
                    applied_deltas.append((p, delta_p))
                else:
                    p.add_(d_norm, alpha=-eff_lr)

        alpha_d = 1.0
        if track_deltas and applied_deltas:
            sq_diffs = []
            for p_args, p_kwargs, d_pre in probes:
                d_post = _eval_probe_forward(self.critic, p_args, p_kwargs, rec)
                if d_post is not None and d_post.numel() == d_pre.numel():
                    sq_diffs.append((d_post.reshape(-1) - d_pre.reshape(-1)).square().mean())
            if sq_diffs:
                delta_d = float(torch.stack(sq_diffs).mean().sqrt().item())
                ema_res = float(getattr(rec, "ema_residual", 1.0))
                ema_m = float(getattr(rec, "ema_margin", 0.0))
                inv_gap = min(25.0, max(0.0, -ema_m))
                hot_gate = _smoothstep01((s_therm - 0.05) / 0.35)
                res_active = _smoothstep01((ema_res - 0.024) / 0.040) * hot_gate
                need_cal = res_active * _smoothstep01((0.40 - ema_m) / 0.40)
                delta_d_min = (0.024 * res_active + 0.056 * need_cal + 0.10 * inv_gap) * therm_d
                delta_d_max = (0.36 * gamma_d * step_scale + 0.18 * inv_gap) * therm_d
                alpha_d = _band_scale(delta_d, delta_d_min, delta_d_max, alpha_max=80.0)
                if alpha_d != 1.0:
                    scale_adj = alpha_d - 1.0
                    for p, delta_p in applied_deltas:
                        p.add_(delta_p, alpha=scale_adj)

        self.last_step_gain = float(alpha_d)
        self.record.record_step(self)
        return loss

    def state_dict(self) -> Dict[str, Any]:
        out = super().state_dict()
        out["t4_nash_v3_opt"] = {"schema": 1, "last_step_gain": float(self.last_step_gain)}
        return out

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_copy = dict(state_dict)
        meta = state_copy.pop("t4_nash_v3_opt", None)
        if not isinstance(meta, dict) or set(meta) != {"schema", "last_step_gain"} or meta["schema"] != 1:
            raise ValueError("invalid or missing NashV3CriticAdam 't4_nash_v3_opt' state")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("invalid NashV3CriticAdam last_step_gain")
        super().load_state_dict(state_copy)
        _validate_peak_rms_states(self)
        self.last_step_gain = float(gain)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)


class NashV3GeneratorAdam(K3PGeneratorAdam):
    """Pure-elementwise Function-Space Impedance-Matched Generator/Prior Adam for T4NashV3."""

    _OWN_ATTRS = (
        *K3PGeneratorAdam._OWN_ATTRS,
        "_nash_v3_record",
        "_t4_record",
        "_sampled_rows",
        "_direct_ids",
        "_prior_param_ids",
        "_conv_layouts",
        "_gen_param_ids",
        "last_step_gain",
        "_protected",
        "constraint_geometry_stats",
    )

    def __init__(
        self,
        params,
        *,
        latent_table: torch.Tensor | None = None,
        direct_particles: Iterable[torch.Tensor] | None = None,
        module_owners: Iterable[nn.Module | None] = (),
        generator_module: nn.Module | None = None,
        latent_max_rate: float = 0.0,
        direct_betas: Tuple[float, float] = (0.0, 0.9),
        direct_gain: bool = False,
        saddle_escape: bool = False,
        **adam_kwargs,
    ) -> None:
        super().__init__(
            params,
            latent_table=latent_table,
            direct_particles=direct_particles,
            latent_max_rate=latent_max_rate,
            direct_betas=direct_betas,
            direct_gain=direct_gain,
            saddle_escape=saddle_escape,
            **adam_kwargs,
        )
        self._nash_v3_record: Any = None
        self._t4_record: Any = None
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}
        self._direct_ids = set() if direct_particles is None else {id(p) for p in direct_particles}
        self._prior_param_ids = {id(latent_table)} if latent_table is not None else set()
        owners = list(module_owners)
        if generator_module is not None and generator_module not in owners:
            owners.insert(0, generator_module)
        self._conv_layouts = _collect_conv_layouts(owners)
        self._gen_param_ids = (
            {id(p) for p in generator_module.parameters()}
            if isinstance(generator_module, nn.Module)
            else set()
        )
        self.last_step_gain: float = 1.0
        self._protected: torch.Tensor | None = None
        self.constraint_geometry_stats: Dict[str, Any] = {
            "steps": 0,
            "projected_steps": 0,
            "max_derivative_before": 0.0,
            "max_derivative_after": 0.0,
        }

    def _all_parameters(self) -> List[torch.Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def _bind_protected_losses(self, losses, *, protected_evaluator=None) -> None:
        """Compute protected gradients via standard 1st-order ``.backward()`` (strictly 0 ``torch.autograd.grad``)."""
        del protected_evaluator
        if self._protected is not None:
            raise ValueError("T4NashV3 protected losses already pending")
        losses = tuple(losses)
        if not 1 <= len(losses) <= 2 or any(loss.ndim != 0 or not loss.requires_grad for loss in losses):
            raise ValueError("one or two active scalar protected losses required")
        parameters = self._all_parameters()
        saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        normals = []
        try:
            for loss in losses:
                for p in parameters:
                    p.grad = None
                loss.backward(retain_graph=True)
                pieces = []
                for p in parameters:
                    g = p.grad
                    piece = torch.zeros_like(p) if g is None else g.detach().clone()
                    if id(p) in self._prior_param_ids or p in self._sampled_rows:
                        rows = self._sampled_rows.get(p)
                        if rows is not None:
                            masked = torch.zeros_like(piece)
                            masked[rows] = piece[rows]
                            piece = masked
                    pieces.append(piece.flatten())
                normals.append(torch.cat(pieces))
        finally:
            for p, prev in zip(parameters, saved_grads):
                p.grad = prev
        stacked = torch.stack(normals)
        if not bool(torch.isfinite(stacked).all()):
            raise ValueError("nonfinite T4NashV3 protected gradient")
        self._protected = stacked

    def set_sampled_rows(self, parameter: torch.Tensor, rows: torch.Tensor) -> None:
        if rows.ndim != 1 or rows.dtype != torch.long or rows.device != parameter.device:
            rows = rows.reshape(-1).to(device=parameter.device, dtype=torch.long)
        self._sampled_rows[parameter] = torch.unique(rows, sorted=True)

    def clear_sampled_rows(self) -> None:
        self._sampled_rows.clear()

    @torch.no_grad()
    def step(self, closure=None):
        loss = _closure_loss(closure)
        rec = getattr(self, "_nash_v3_record", None) or getattr(self, "_t4_record", None)
        step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
        gamma_g = float(getattr(rec, "g_trust_scale", 1.0)) if rec is not None else 1.0
        s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
        if s_therm <= 1e-4:
            therm_g = 1e-4
        elif s_therm < 1.0:
            therm_g = max(0.08, math.sqrt(s_therm))
        else:
            therm_g = 1.0

        all_params = self._all_parameters() if self._protected is not None else []
        originals = [p.detach().clone() for p in all_params] if self._protected is not None else []

        probe_cache = getattr(rec, "_g_probe_cache", None) if rec is not None else None
        track_gen_deltas = bool(probe_cache is not None and getattr(rec, "_enable_secant_probe", False))
        gen_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        try:
            for group in self.param_groups:
                lr = float(group["lr"])
                betas = tuple(group["betas"])
                eps = float(group["eps"])
                amsgrad = bool(group.get("amsgrad", False))
                role = group.get("role", group.get("forge_role", "generator"))
                is_prior_group = role in ("prior", "table")

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    is_direct = id(p) in self._direct_ids
                    is_prior_param = (
                        (is_prior_group and not is_direct)
                        or (id(p) in self._prior_param_ids)
                        or (p in self._sampled_rows)
                    )
                    rows = self._sampled_rows.get(p)

                    u_adam = _adam_unit_direction(p, p.grad, self.state[p], betas, eps, amsgrad)
                    if lr <= 0.0:
                        continue

                    if is_prior_param:
                        prior_therm = 1e-4 if s_therm <= 1e-4 else max(0.25, math.sqrt(s_therm))
                        eff_lr = min(lr, 0.065) * step_scale * prior_therm
                        d_rows, active_rows = _impedance_normalized_direction(
                            p, u_adam, is_prior_table=True, sampled_rows=rows
                        )
                        if active_rows is not None and active_rows.numel() > 0:
                            p.index_add_(0, active_rows, d_rows, alpha=-eff_lr)
                    elif is_direct:
                        eff_lr = max(0.010, min(lr, 0.030)) * step_scale * therm_g
                        d_norm, _ = _impedance_normalized_direction(
                            p, u_adam, is_direct_particles=True
                        )
                        p.add_(d_norm, alpha=-eff_lr)
                    else:
                        eff_lr = lr * step_scale * gamma_g * therm_g
                        is_deconv, c_groups = self._conv_layouts.get(id(p), (False, 1))
                        d_norm, _ = _impedance_normalized_direction(
                            p,
                            u_adam,
                            conv_transpose=is_deconv,
                            conv_groups=c_groups,
                        )
                        if track_gen_deltas:
                            delta_p = d_norm * (-eff_lr)
                            p.add_(delta_p)
                            gen_deltas.append((p, delta_p))
                        else:
                            p.add_(d_norm, alpha=-eff_lr)

            alpha_g = 1.0
            if track_gen_deltas and gen_deltas and probe_cache is not None and rec is not None:
                g_mod, p_args, p_kwargs, y_pre = probe_cache
                rec._g_probe_cache = None
                y_post = _eval_probe_forward(g_mod, p_args, p_kwargs, rec)
                if y_post is not None and y_post.shape == y_pre.shape:
                    out_dim = max(1, y_pre[0].numel())
                    delta_rms = float((y_post - y_pre).square().mean().sqrt().item())
                    data_scale = float(getattr(rec, "ema_data_scale", 1.0))
                    coord_scale = max(0.05, data_scale / math.sqrt(float(out_dim)))
                    delta_rel = delta_rms / coord_scale

                    ema_res = float(getattr(rec, "ema_residual", 1.0))
                    delta_g_max = (0.020 + 0.23 * min(1.5, ema_res)) * gamma_g * step_scale * therm_g

                    if ema_res > 0.024 and s_therm > 0.10:
                        active_gate = _smoothstep01((ema_res - 0.024) / 0.036) * _smoothstep01((s_therm - 0.10) / 0.40)
                        delta_g_min = (0.0045 + 0.042 * min(1.0, ema_res)) * active_gate * step_scale
                    else:
                        delta_g_min = 0.0

                    alpha_g = _band_scale(delta_rel, delta_g_min, delta_g_max, alpha_max=96.0)
                    if alpha_g != 1.0:
                        scale_adj = alpha_g - 1.0
                        for p, delta_p in gen_deltas:
                            p.add_(delta_p, alpha=scale_adj)
            elif rec is not None:
                rec._g_probe_cache = None

            if self._protected is not None and all_params:
                normals = self._protected
                displacement = torch.cat([(p - old).flatten() for p, old in zip(all_params, originals)])
                projected = _project_nonascent_closed_form(displacement, normals)
                before = normals.to(dtype=torch.float64) @ displacement.to(dtype=torch.float64)
                stats = self.constraint_geometry_stats
                stats["steps"] += 1
                changed = not torch.equal(projected, displacement)
                stats["projected_steps"] += int(changed)
                stats["max_derivative_before"] = max(stats["max_derivative_before"], float(before.max().item()))
                if changed:
                    offset = 0
                    for p, old in zip(all_params, originals):
                        sz = p.numel()
                        p.copy_(old + projected[offset : offset + sz].view_as(p))
                        offset += sz
                applied = torch.cat([(p - old).flatten() for p, old in zip(all_params, originals)])
                after = normals.to(dtype=torch.float64) @ applied.to(dtype=torch.float64)
                stats["max_derivative_after"] = max(stats["max_derivative_after"], float(after.max().item()))

            self.last_step_gain = float(alpha_g)
        finally:
            self._protected = None
            self.clear_sampled_rows()
        return loss

    def state_dict(self) -> Dict[str, Any]:
        out = super().state_dict()
        out["t4_nash_v3_opt"] = {
            "schema": 1,
            "last_step_gain": float(self.last_step_gain),
            "stats": dict(self.constraint_geometry_stats),
            "pending": None if self._protected is None else self._protected.detach().clone(),
        }
        return out

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_copy = dict(state_dict)
        meta = state_copy.pop("t4_nash_v3_opt", None)
        if (
            not isinstance(meta, dict)
            or set(meta) != {"schema", "last_step_gain", "stats", "pending"}
            or meta["schema"] != 1
            or not isinstance(meta["stats"], dict)
            or set(meta["stats"]) != set(self.constraint_geometry_stats)
        ):
            raise ValueError("invalid or missing NashV3GeneratorAdam 't4_nash_v3_opt' state")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("invalid NashV3GeneratorAdam last_step_gain")
        for key, val in meta["stats"].items():
            if key.endswith("steps"):
                if type(val) is not int or val < 0:
                    raise ValueError("invalid NashV3GeneratorAdam constraint step counter")
            elif isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError("invalid NashV3GeneratorAdam constraint derivative stat")
        pending = meta["pending"]
        total_numel = sum(p.numel() for p in self._all_parameters())
        if pending is not None and (
            not isinstance(pending, torch.Tensor)
            or pending.ndim != 2
            or not (1 <= int(pending.shape[0]) <= 2)
            or int(pending.shape[1]) != total_numel
            or not bool(torch.isfinite(pending).all())
        ):
            raise ValueError("invalid NashV3GeneratorAdam pending protected gradients")
        super().load_state_dict(state_copy)
        _validate_peak_rms_states(self)
        self.last_step_gain = float(gain)
        self.constraint_geometry_stats = dict(meta["stats"])
        self._protected = None if pending is None else pending.detach().clone()


class NashV3ConstraintGeneratorAdam(NashV3GeneratorAdam):
    """NashV3GeneratorAdam exposing ``bind_protected_losses`` only when ``constraint_geometry_mode != 'none'``."""

    bind_protected_losses = NashV3GeneratorAdam._bind_protected_losses


__all__ = [
    "NashV3ConstraintGeneratorAdam",
    "NashV3CriticAdam",
    "NashV3GeneratorAdam",
    "_adam_unit_direction",
    "_band_scale",
    "_collect_conv_layouts",
    "_impedance_normalized_direction",
    "_project_nonascent_closed_form",
]
