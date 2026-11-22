"""Causal Law-4 SGD & Asymmetric G/D Optimizer Ladder on `unified_v11`.

All six optimizer families subclass ``UnifiedV11EquilibratedOptimizer`` without
modifying ``particlegan/_unified_v11/``, inheriting its zero-``autograd.grad`` /
zero-``pinv`` ``direction_blend`` projection, strict checkpoint validation, and
``unified_v11`` hook compatibility while implementing exact causal update laws.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Tuple

import torch

from .._unified_v4.optimizer import _band_scale, _smoothstep01
from .._unified_v11.critic import _eval_v11_probe_forward as _eval_probe_forward
from .._unified_v11.optimizer import (
    UnifiedV11EquilibratedOptimizer,
    zero_pinv_project_nonascent,
)

LAW4_SGD_FAMILIES: Tuple[str, ...] = (
    "law4_pure_sgd",
    "law4_sgd_secant",
    "law4_sgd_g_equil_d",
    "law4_sgd_g_secant_equil_d",
    "law4_equil_g_sgd_d",
    "law4_frob_sgd_mp",
)


class _Law4SGDBase(UnifiedV11EquilibratedOptimizer):
    """Shared base for the Law-4 causal SGD ladder (subclasses ``UnifiedV11EquilibratedOptimizer``)."""

    family = "law4_pure_sgd"
    recipe_optimizer_family = "law4_pure_sgd"
    _use_secant_band: bool = False
    _equil_prior: bool = False
    _frob_mp_mode: bool = False

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
        cls_family = type(self).family
        self.family = cls_family
        self.recipe_optimizer_family = cls_family
        for grp in self.param_groups:
            if grp["algorithm"] == "unified_v11_equil":
                grp["algorithm"] = cls_family

    def _is_critic_optimizer(self) -> bool:
        if getattr(self, "critic", None) is not None:
            return True
        non_empty = [g for g in self.param_groups if g["params"]]
        return bool(non_empty) and all(g["role"] == "critic" for g in non_empty)

    def _apply_equil_prior_row_update(
        self,
        grp: dict[str, Any],
        p: torch.Tensor,
        grad: torch.Tensor,
        st: dict[str, Any],
        base_lr: float,
        *,
        stateless: bool = False,
    ) -> None:
        """Exact `unified_v11` row-normalized prior update (optionally strictly stateless)."""
        rows = (
            self._sampled_rows[p]
            if grp.get("sampled_rows_required", True)
            else torch.arange(len(p), device=p.device, dtype=torch.long)
        )
        if rows.numel() == 0 or base_lr <= 0.0:
            return
        gr = grad[rows].float()
        fin_row = torch.isfinite(gr).all(dim=-1, keepdim=True)
        gr_safe = torch.where(fin_row, gr, torch.zeros_like(gr))
        b1 = float(grp["betas"][0])
        b1_eff = 0.0 if stateless else (max(b1, 0.75) if self._v4_prior_scale < 1.0 else b1)
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

        cur_row_rms = d_norm.square().mean().sqrt()
        if "peak_rms" not in st:
            st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
        peak_r = torch.maximum(
            st["peak_rms"] * 0.9995,
            torch.where(torch.isfinite(cur_row_rms), cur_row_rms, st["peak_rms"].new_zeros(())),
        )
        st["peak_rms"].copy_(torch.where(torch.isfinite(peak_r), peak_r, st["peak_rms"]))

        s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0
        if s_prior_base > 0.0:
            s_cap = s_prior_base * (128.0 / float(len(rows))) if len(rows) > 128 else s_prior_base
            s_p_row = torch.minimum(d_norm.new_tensor(s_cap), peak_r * 0.03)
            denom_r = torch.hypot(d_norm, s_p_row).clamp_min(1e-36)
        else:
            denom_r = d_norm.clamp_min(1e-36)
        unit_r = torch.where(active_r, dir_r / denom_r, torch.zeros_like(dir_r))
        r_scales = self._row_scales.get(p)
        if r_scales is not None and len(r_scales) == len(rows):
            unit_r = unit_r * r_scales.unsqueeze(1)
        p.index_add_(0, rows, unit_r.to(p.dtype), alpha=-base_lr)

    def _apply_direction_blend_if_needed(
        self,
        all_params: List[torch.Tensor],
        originals: List[torch.Tensor] | None,
    ) -> None:
        if not (self.direction_blend and originals is not None and self._protected is not None):
            return
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

    def _run_secant_trust_band(
        self,
        rec: Any,
        is_critic_opt: bool,
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]],
        pre_gain_applied: float,
        step_scale: float,
        s_therm: float,
        is_simp: bool,
        is_cond_fiber: bool,
    ) -> Tuple[float, bool]:
        """Exact Law 4 player-level secant trust band (`delta_g_min/max`, `delta_d_min/max`)."""
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
            deltas_finite = all(bool(torch.isfinite(d_i).all()) for _, d_i in applied_deltas)
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
                        delta_d = float(torch.stack(sq_diffs).mean().sqrt().item())
                        ema_m = float(getattr(rec, "ema_margin", 0.0)) if getattr(rec, "margin_initialized", False) else 0.0
                        inv_gap = min(25.0, max(0.0, -ema_m - 1.5))
                        th_d = max(0.15, math.sqrt(s_therm)) if s_therm < 1.0 else 1.0
                        th_tikh = min(1.0, s_therm / 0.03) if s_therm < 0.03 else 1.0
                        delta_d_min = 0.12 * inv_gap * th_d
                        delta_d_max = (0.55 * step_scale + 0.20 * inv_gap) * th_d
                        if not math.isfinite(delta_d):
                            alpha_gain = 1e-4 if deltas_finite else 1.0
                        else:
                            alpha_gain = _band_scale(delta_d, delta_d_min, delta_d_max, alpha_max=64.0)
                        if math.isfinite(delta_d) and delta_d > 1e-12:
                            free_ratio = (delta_d_max / delta_d) * pre_gain_applied * min(1.0, th_tikh / max(th_d, 1e-6))
                        active_thresh_d = max(1e-4, 0.01 * pre_gain_applied * step_scale * min(th_d, th_tikh))
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
                        delta_rms = float((y_post - y_pre).square().mean().sqrt().item())
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
                        if not math.isfinite(delta_rel):
                            alpha_gain = 1e-4 if deltas_finite else 1.0
                        else:
                            alpha_gain = _band_scale(delta_rel, delta_g_min, delta_g_max, alpha_max=48.0)
                        th_tikh = min(1.0, s_therm / 0.03) if s_therm < 0.03 else 1.0
                        if math.isfinite(delta_rel) and delta_rel > 1e-12:
                            free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh
                        active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh)
                        step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
            if alpha_gain != 1.0:
                adj = alpha_gain - 1.0
                for p_i, d_i in applied_deltas:
                    p_i.add_(d_i, alpha=adj)
            if probe_ran:
                if alpha_gain < 1.0:
                    target_fg = max(0.01, min(1.0, pre_gain_applied * alpha_gain))
                    self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * target_fg)
                elif step_active_probe and free_ratio > max(1.15 * self._ema_func_gain, 0.01):
                    target_fg = max(0.01, min(1.0, 0.90 * free_ratio))
                    if target_fg > self._ema_func_gain:
                        self._ema_func_gain = float(0.995 * self._ema_func_gain + 0.005 * target_fg)
        elif rec is not None and not is_critic_opt:
            rec._g_probe_cache = None
        return float(alpha_gain), bool(probe_ran)

    @torch.no_grad()
    def _step_sgd_or_frob(self, closure: Any = None) -> Any:
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

        all_params = self._parameters()
        originals = (
            [p.detach().clone() for p in all_params]
            if (self.direction_blend and self._protected is not None)
            else None
        )

        try:
            rec = self._active_record()
            step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
            s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            is_simp = bool(getattr(rec, "_simplicial_step", False)) if rec is not None else False
            is_cond_fiber = bool(getattr(rec, "t1c_cond_initialized", False)) if rec is not None else False
            is_critic_opt = self._is_critic_optimizer()

            origin_damping = 1.0
            if self._frob_mp_mode and self._origin_norm is not None and self._origin_norm < 0.05:
                origin_damping = min(1.0, self._origin_norm / 0.05)

            ema_fg = float(self._ema_func_gain)
            pre_gain_base = (1.0 if ema_fg >= 0.75 else ema_fg) if self._use_secant_band else 1.0
            pre_gain_applied = pre_gain_base

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for grp in self.param_groups:
                role = grp["role"]
                player = self._player(role)

                if player == "prior":
                    if self._equil_prior:
                        prior_lr = float(grp["lr"]) * step_scale * self._v4_prior_scale
                    else:
                        prior_lr = float(grp["lr"])
                    for p in grp["params"]:
                        grad = p.grad
                        if grad is None:
                            continue
                        st = self.state[p]
                        st["step"] = int(st.get("step", 0)) + 1
                        if self._equil_prior:
                            self._apply_equil_prior_row_update(
                                grp,
                                p,
                                grad,
                                st,
                                prior_lr,
                                stateless=self._frob_mp_mode,
                            )
                        else:
                            # Pure unnormalized SGD on sampled prior rows only (unsampled rows untouched):
                            rows = (
                                self._sampled_rows[p]
                                if grp.get("sampled_rows_required", True)
                                else torch.arange(len(p), device=p.device, dtype=torch.long)
                            )
                            if rows.numel() > 0 and prior_lr > 0.0:
                                p.index_add_(0, rows, grad[rows].to(p.dtype), alpha=-prior_lr)
                    continue

                if self._frob_mp_mode:
                    if player == "generator":
                        pre_g = min(1.0, pre_gain_base * self._v4_wakeup_boost)
                        pre_gain_applied = pre_g
                        base_lr = float(grp["lr"]) * step_scale * self._v4_g_scale * pre_g
                    else:
                        pre_gain_applied = pre_gain_base
                        base_lr = float(grp["lr"]) * step_scale * pre_gain_base
                elif self._use_secant_band:
                    pre_gain_applied = pre_gain_base
                    base_lr = float(grp["lr"]) * pre_gain_base
                else:
                    pre_gain_applied = 1.0
                    base_lr = float(grp["lr"])

                for p in grp["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    st = self.state[p]
                    step_t = int(st.get("step", 0)) + 1
                    st["step"] = step_t
                    track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                    if not self._frob_mp_mode:
                        # Strictly pure unnormalized SGD: delta = -base_lr * grad (0 tensor states/param)
                        if base_lr <= 0.0:
                            continue
                        if self._use_secant_band and track_secant:
                            delta = grad.to(p.dtype).mul(-base_lr)
                            p.add_(delta)
                            applied_deltas.append((p, delta))
                        else:
                            p.add_(grad.to(p.dtype), alpha=-base_lr)
                        continue

                    # Rung 5: Zero-state layerwise Frobenius-normalized SGD + MP + peak_rms:
                    # delta_W = -lr_MP * G / hypot(||G||_F, s_eff), strictly O(params), 0 tensor states/param.
                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    gf = grad.float()
                    finite = torch.isfinite(gf).all()
                    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                    g_max = gf_safe.abs().amax()
                    g_active = finite & (g_max > 1e-38)
                    safe_max = torch.where(g_active, g_max, torch.ones_like(g_max))
                    gf_scaled = torch.where(g_active, gf_safe / safe_max, torch.zeros_like(gf_safe))
                    frob_scaled = gf_scaled.norm()
                    cur_frob = frob_scaled * safe_max

                    if "peak_rms" not in st:
                        st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    peak_new = torch.maximum(
                        st["peak_rms"] * 0.9995,
                        torch.where(finite, cur_frob, st["peak_rms"].new_zeros(())),
                    )
                    st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))

                    if rate_p <= 0.0:
                        continue

                    is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                    if is_conv:
                        c0, c1, kh, kw = (int(s) for s in p.shape)
                        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                        g_grp = max(1, int(groups_c))
                        if is_trans:
                            out_ch = float(c1 * g_grp)
                            in_ch = float(c0)
                            k_eff = float(max(1, (kh * kw) // 4))
                        else:
                            out_ch = float(c0)
                            in_ch = float(c1 * g_grp)
                            k_eff = float(max(1, kh * kw))
                        factor = math.sqrt(out_ch / max(1.0, in_ch * k_eff))
                        sv_scale = math.sqrt(float(max(1, min(int(out_ch), int(in_ch)) * kh * kw)))
                    elif p.ndim >= 2:
                        r_d = int(p.shape[0])
                        c_d = int(p.numel() // max(1, r_d))
                        factor = math.sqrt(float(r_d) / float(max(1, c_d)))
                        sv_scale = math.sqrt(float(max(1, min(r_d, c_d))))
                    else:
                        factor = 1.0
                        sv_scale = 1.0

                    has_smooth = self.smoothing > 0.0
                    if has_smooth:
                        s_cap = self.smoothing * sv_scale
                        s_eff_t = torch.minimum(gf_safe.new_tensor(s_cap), peak_new * 0.03)
                        s_scaled = s_eff_t / safe_max
                        denom_scaled = torch.hypot(frob_scaled, s_scaled).clamp_min(1e-30)
                    else:
                        denom_scaled = frob_scaled.clamp_min(1e-30)

                    unit_dir = torch.where(
                        g_active & (frob_scaled > 1e-30),
                        gf_scaled / denom_scaled,
                        torch.zeros_like(gf_scaled),
                    )
                    delta_f = (unit_dir * (-rate_p * factor)).to(p.dtype)
                    p.add_(delta_f)
                    if track_secant:
                        applied_deltas.append((p, delta_f))

            if self._use_secant_band:
                alpha_gain, probe_ran = self._run_secant_trust_band(
                    rec,
                    is_critic_opt,
                    applied_deltas,
                    pre_gain_applied,
                    step_scale,
                    s_therm,
                    is_simp,
                    is_cond_fiber,
                )
            else:
                alpha_gain, probe_ran = 1.0, False
                if rec is not None and not is_critic_opt:
                    rec._g_probe_cache = None

            self._apply_direction_blend_if_needed(all_params, originals)

            self.last_step_gain = float(alpha_gain)
            self.last_probe_consumed = bool(probe_ran)
            if probe_ran:
                self.probe_consumptions += 1
        finally:
            self.abort_pending()

        if hasattr(self, "record") and self.record is not None:
            self.record.record_step(self)
        return loss


class Law4PureSGDOptimizer(_Law4SGDBase):
    """Rung 1 (`law4_pure_sgd`): pure unnormalized SGD on G, D, and sampled prior rows."""

    family = "law4_pure_sgd"
    recipe_optimizer_family = "law4_pure_sgd"
    _use_secant_band = False
    _equil_prior = False
    _frob_mp_mode = False

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        return self._step_sgd_or_frob(closure=closure)


class Law4SGDSecantOptimizer(_Law4SGDBase):
    """Rung 2 (`law4_sgd_secant`): pure SGD wrapped ONLY by Law 4's global player-level secant trust band."""

    family = "law4_sgd_secant"
    recipe_optimizer_family = "law4_sgd_secant"
    _use_secant_band = True
    _equil_prior = False
    _frob_mp_mode = False

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        return self._step_sgd_or_frob(closure=closure)


class Law4SGDGEquilDOptimizer(_Law4SGDBase):
    """Rung 3a (`law4_sgd_g_equil_d`): pure unnormalized SGD on G; exact `unified_v11_equil` on D & prior."""

    family = "law4_sgd_g_equil_d"
    recipe_optimizer_family = "law4_sgd_g_equil_d"
    _use_secant_band = False
    _equil_prior = True
    _frob_mp_mode = False

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if self._is_critic_optimizer():
            return super().step(closure=closure)
        return self._step_sgd_or_frob(closure=closure)


class Law4SGDGSecantEquilDOptimizer(_Law4SGDBase):
    """Rung 3b (`law4_sgd_g_secant_equil_d`): pure SGD + Law 4 G secant band on G; exact `unified_v11_equil` on D & prior."""

    family = "law4_sgd_g_secant_equil_d"
    recipe_optimizer_family = "law4_sgd_g_secant_equil_d"
    _use_secant_band = True
    _equil_prior = True
    _frob_mp_mode = False

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if self._is_critic_optimizer():
            return super().step(closure=closure)
        return self._step_sgd_or_frob(closure=closure)


class Law4EquilGSGDDOptimizer(_Law4SGDBase):
    """Rung 4 (`law4_equil_g_sgd_d`): causal reverse control — exact `unified_v11_equil` on G & prior; pure SGD on D."""

    family = "law4_equil_g_sgd_d"
    recipe_optimizer_family = "law4_equil_g_sgd_d"
    _use_secant_band = False
    _equil_prior = False
    _frob_mp_mode = False

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if self._is_critic_optimizer():
            return self._step_sgd_or_frob(closure=closure)
        return super().step(closure=closure)


class Law4FrobSGDMPOptimizer(_Law4SGDBase):
    """Rung 5 (`law4_frob_sgd_mp`): zero-state layerwise Frobenius-SGD + MP + `peak_rms` + secant band."""

    family = "law4_frob_sgd_mp"
    recipe_optimizer_family = "law4_frob_sgd_mp"
    _use_secant_band = True
    _equil_prior = True
    _frob_mp_mode = True

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        return self._step_sgd_or_frob(closure=closure)


__all__ = [
    "LAW4_SGD_FAMILIES",
    "Law4EquilGSGDDOptimizer",
    "Law4FrobSGDMPOptimizer",
    "Law4PureSGDOptimizer",
    "Law4SGDGEquilDOptimizer",
    "Law4SGDGSecantEquilDOptimizer",
    "Law4SGDSecantOptimizer",
]
