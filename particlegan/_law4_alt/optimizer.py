"""Law-4 Alternative Optimizer Exploration on `unified_v11`.

Holds Laws 1-3 (`critic_formulation="unified_v11"`, `prior_regularizer="unified_v11"`,
`UnifiedV11StepRecord`, `UnifiedV11CondFiberBank`, prior row-normalized updates,
zero-`pinv`/`autograd.grad` `direction_blend`, and Law 4 realized function-space
secant trust band) 100% identical to `unified_v11_equil`, varying ONLY the
network optimizer geometry across six distinct families:

1. ``law4_adam_equil``      -- 2 states/param, 0 NS: $10^{-38}$-safe coordinate-wise
   RMS (Adam ``beta1=0.0, beta2=0.999``) + MP operator scaling + ``peak_rms`` + secant band.
2. ``law4_sign_mp``         -- 0 tensor states/param, 0 NS: $L_\\infty$ smooth SignSGD
   ``u = g / hypot(|g|, s_eff)`` + MP operator scaling + secant band.
3. ``law4_lion_mp``         -- 1 state/param, 0 NS: $L_\\infty$ Lion-style smooth sign
   momentum ``smooth_sign(beta1 * m + (1 - beta1) * g)`` + MP + ``peak_rms`` + secant band.
4. ``law4_muon_m``          -- 1 state/param, 5-step Quintic NS: LLM Muon-style positive
   Nesterov momentum (``beta1=0.85``) + 5-step Quintic NS + ``peak_rms`` + secant band.
5. ``law4_optimistic_equil``-- 1 state/param, 5-step Quintic NS: Optimistic / Extragradient
   minimax polar ``g_eff = g_t + lambda * (g_t - g_{t-1})`` (``lambda=0.5``) + 5-step Quintic NS.
6. ``law4_adafactor_mp``    -- Sublinear $O(m+n)$ state, 0 NS: Adafactor-style row/col
   factored RMS + MP operator scaling + ``peak_rms`` + secant band.
"""
from __future__ import annotations

from copy import copy, deepcopy
import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.optim import Optimizer

from .._unified_v4.optimizer import (
    _band_scale,
    _smoothstep01,
)
from .._unified_v11.critic import _eval_v11_probe_forward as _eval_probe_forward
from .._unified_v11.optimizer import (
    UnifiedV11EquilibratedOptimizer,
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
    zero_pinv_project_nonascent,
)


LAW4_ALT_FAMILIES: Tuple[str, ...] = (
    "law4_adam_equil",
    "law4_sign_mp",
    "law4_lion_mp",
    "law4_muon_m",
    "law4_optimistic_equil",
    "law4_adafactor_mp",
)


def _safe_sv_rms(gf_safe: torch.Tensor, g_max: torch.Tensor, sv_denom: float) -> torch.Tensor:
    """Underflow-safe (10^-38) singular-value RMS in FP32."""
    sc = g_max.clamp_min(1e-38)
    return ((gf_safe / sc).norm() / sv_denom) * g_max


def _safe_elem_rms(gf_safe: torch.Tensor, g_max: torch.Tensor) -> torch.Tensor:
    """Underflow-safe (10^-38) elementwise RMS in FP32."""
    sc = g_max.clamp_min(1e-38)
    return (gf_safe / sc).square().mean().sqrt() * g_max


def _safe_tikhonov_ratio(val: torch.Tensor, smooth: torch.Tensor) -> torch.Tensor:
    """Underflow-safe (10^-38) val / hypot(val, smooth)."""
    sc = torch.maximum(val.abs(), smooth.abs()).clamp_min(1e-38)
    v_s = val / sc
    s_s = smooth / sc
    return v_s / torch.hypot(v_s, s_s).clamp_min(1e-30)


def _mp_scale_for_tensor(
    p: torch.Tensor,
    is_conv: bool,
    conv_layouts: Dict[int, Tuple[bool, int]],
) -> Tuple[float, float, int]:
    """Return ``(mp_step_scale, sv_rms_denom, max_dim)`` for Marchenko-Pastur scaling.

    ``mp_step_scale`` equals ``factor / mp_denom`` for 2D/4D tensors and
    ``1.0 / sqrt(numel)`` for 1D vectors (for coordinate-wise unit-RMS directions).
    """
    if is_conv and (p.shape[2] > 1 or p.shape[3] > 1):
        c0, c1, kh, kw = (int(s) for s in p.shape)
        is_trans, groups_c = conv_layouts.get(id(p), (False, 1))
        g_grp = max(1, int(groups_c))
        if is_trans:
            out_ch, in_ch = float(c1 * g_grp), float(c0)
            k_eff = float(max(1, (kh * kw) // 4))
        else:
            out_ch, in_ch = float(c0), float(c1 * g_grp)
            k_eff = float(max(1, kh * kw))
        factor = math.sqrt(max(0.5, out_ch / max(1.0, in_ch)))
        mp_denom = (math.sqrt(max(1.0, float(c0))) + math.sqrt(max(1.0, float(c1)))) * math.sqrt(k_eff)
        sv_denom = math.sqrt(float(max(1, min(int(out_ch), int(in_ch)) * kh * kw)))
        return factor / mp_denom, sv_denom, max(1, int(max(out_ch, in_ch)))
    if p.ndim >= 2:
        r_d = int(p.shape[0])
        c_d = int(p.numel() // max(1, r_d))
        if is_conv:
            c0, c1, _, _ = (int(s) for s in p.shape)
            is_trans, groups_c = conv_layouts.get(id(p), (False, 1))
            out_ch = float(c1 * groups_c if is_trans else c0)
            in_ch = float(c0 if is_trans else c1 * groups_c)
            factor = math.sqrt(out_ch / max(1.0, in_ch))
            mp_denom = math.sqrt(max(1.0, out_ch)) + math.sqrt(max(1.0, in_ch))
            sv_denom = math.sqrt(float(max(1, min(int(out_ch), int(in_ch)))))
            return factor / mp_denom, sv_denom, max(1, int(max(out_ch, in_ch)))
        factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
        mp_denom = math.sqrt(max(1.0, float(r_d))) + math.sqrt(max(1.0, float(c_d)))
        sv_denom = math.sqrt(float(max(1, min(r_d, c_d))))
        return factor / mp_denom, sv_denom, max(1, max(r_d, c_d))
    return 1.0 / math.sqrt(float(max(1, p.numel()))), 1.0, 1


class _Law4AltBaseOptimizer(UnifiedV11EquilibratedOptimizer):
    """Shared Law-4 infrastructure holding Laws 1-3 + secant band + direction_blend identical to unified_v11."""

    family: str = "law4_alt_base"
    recipe_optimizer_family: str = "law4_alt_base"
    _allowed_state_keys: Tuple[str, ...] = ("step", "peak_rms", "m_prior")

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
        for group in self.param_groups:
            if group["algorithm"] == "unified_v11_equil":
                group["algorithm"] = cls_family

    def _apply_prior_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        st: Dict[str, Any],
        grp: Dict[str, Any],
        *,
        base_lr: float,
        b1_eff: float,
        s_prior_base: float,
    ) -> None:
        """Exact `unified_v11` row-normalized prior update (Law 1 held 100% constant)."""
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

    def _apply_network_groups(
        self,
        *,
        step_scale: float,
        origin_damping: float,
        stiff_gate_raw: float,
        pre_gain_base: float,
        is_critic_opt: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], float]:
        raise NotImplementedError

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
                    raise ValueError(f"{self.family} requires dense real gradients")
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
            if self._origin_norm is not None and self._origin_norm < 0.05:
                origin_damping = min(1.0, self._origin_norm / 0.05)

            ema_fg = float(self._ema_func_gain)
            stiff_gate_raw = _smoothstep01((0.60 - ema_fg) / 0.30)
            pre_gain_base = 1.0 if ema_fg >= 0.75 else ema_fg

            applied_deltas, coh_vals, pre_gain_applied = self._apply_network_groups(
                step_scale=step_scale,
                origin_damping=origin_damping,
                stiff_gate_raw=stiff_gate_raw,
                pre_gain_base=pre_gain_base,
                is_critic_opt=is_critic_opt,
            )

            # Exact Law 4 Two-Player Function-Space Secant Trust Band (identical to unified_v11):
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
                            delta_d = float(torch.stack(sq_diffs).mean().sqrt().item())
                            ema_m = float(getattr(rec, "ema_margin", 0.0)) if getattr(rec, "margin_initialized", False) else 0.0
                            inv_gap = min(25.0, max(0.0, -ema_m - 1.5))
                            th_d = max(0.15, math.sqrt(s_therm)) if s_therm < 1.0 else 1.0
                            th_tikh = min(1.0, s_therm / 0.03) if s_therm < 0.03 else 1.0
                            delta_d_min = 0.12 * inv_gap * th_d
                            delta_d_max = (0.55 * step_scale + 0.20 * inv_gap) * th_d
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
                            alpha_gain = _band_scale(delta_rel, delta_g_min, delta_g_max, alpha_max=48.0)
                            th_tikh = min(1.0, s_therm / 0.03) if s_therm < 0.03 else 1.0
                            mean_coh = float(torch.stack(coh_vals).mean().item()) if coh_vals else 1.0
                            coh_eff = max(0.05, mean_coh)
                            if math.isfinite(delta_rel) and delta_rel > 1e-12:
                                free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh * coh_eff
                            active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh * coh_eff)
                            step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, d_i in applied_deltas:
                        p_i.add_(d_i, alpha=adj)
                if probe_ran:
                    if alpha_gain < 1.0:
                        target_fg = max(0.05, min(1.0, pre_gain_applied * alpha_gain))
                        self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * target_fg)
                    elif step_active_probe and free_ratio > max(1.15 * self._ema_func_gain, 0.05):
                        target_fg = max(0.05, min(1.0, 0.90 * free_ratio))
                        if target_fg > self._ema_func_gain:
                            self._ema_func_gain = float(0.995 * self._ema_func_gain + 0.005 * target_fg)
            elif rec is not None and not is_critic_opt:
                rec._g_probe_cache = None

            # Exact Zero-`pinv` Analytic Non-Ascent Projection on multi-objective conflict steps:
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

    def _extra_meta(self) -> Dict[str, Any]:
        return {}

    def _validate_extra_meta(self, meta: Dict[str, Any]) -> None:
        del meta

    def state_dict(self) -> Dict[str, Any]:
        result = Optimizer.state_dict(self)
        meta: Dict[str, Any] = {
            "schema": 1,
            "family": self.family,
            "last_step_gain": float(self.last_step_gain),
            "ema_func_gain": float(self._ema_func_gain),
            "direction_blend": bool(self.direction_blend),
            "sampled_rows": [
                self._sampled_rows.get(grp["params"][0]) if grp["algorithm"] == "unified_prior" else None
                for grp in self.param_groups
            ],
            **self._extra_meta(),
        }
        if self.direction_blend:
            meta["constraint_geometry"] = dict(
                stats=deepcopy(self.constraint_geometry_stats),
                blend_stats=deepcopy(self.direction_blend_stats),
                pending=None if self._protected is None else self._protected.detach().clone(),
            )
        result["law4_alt"] = meta
        if hasattr(self, "record") and self.record is not None:
            result["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return result

    def validate_state_dict(self, saved: Any) -> None:
        expected_keys = {"state", "param_groups", "law4_alt"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError(f"invalid {type(self).__name__} checkpoint schema")
        meta = saved["law4_alt"]
        expected_meta_keys = {
            "schema",
            "family",
            "last_step_gain",
            "ema_func_gain",
            "direction_blend",
            "sampled_rows",
            *self._extra_meta().keys(),
        }
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
            raise ValueError(f"incompatible {type(self).__name__} metadata")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) <= 0.0:
            raise ValueError(f"invalid {type(self).__name__} last_step_gain")
        ema_fg = meta["ema_func_gain"]
        if (
            isinstance(ema_fg, bool)
            or not isinstance(ema_fg, (int, float))
            or not math.isfinite(ema_fg)
            or not (0.0 < float(ema_fg) <= 1.0)
        ):
            raise ValueError(f"invalid {type(self).__name__} ema_func_gain")
        self._validate_extra_meta(meta)
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
                raise ValueError(f"invalid {type(self).__name__} direction_blend checkpoint")
            if cg["pending"] is not None:
                if not isinstance(cg["pending"], torch.Tensor) or not bool(torch.isfinite(cg["pending"]).all()):
                    raise ValueError("invalid pending direction_blend tensor")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError(f"{type(self).__name__} group count mismatch")
        for cur_grp, sav_grp in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav_grp, dict) or len(sav_grp.get("params", ())) != len(cur_grp["params"]):
                raise ValueError(f"{type(self).__name__} group parameter mismatch")
            if sav_grp.get("role") != cur_grp["role"] or sav_grp.get("algorithm") != cur_grp["algorithm"]:
                raise ValueError(f"{type(self).__name__} group role/algorithm mismatch")
        if not isinstance(saved["state"], dict):
            raise ValueError(f"invalid {type(self).__name__} state map")
        allowed_keys = set(self._allowed_state_keys)
        for st_val in saved["state"].values():
            if not isinstance(st_val, dict):
                raise ValueError(f"invalid {type(self).__name__} parameter state")
            unknown_st = set(st_val) - allowed_keys
            if unknown_st:
                raise ValueError(f"unexpected state key(s) {sorted(unknown_st)} in {type(self).__name__}")
            step_v = st_val.get("step", 0)
            if type(step_v) is not int or step_v < 0:
                raise ValueError(f"invalid {type(self).__name__} step counter")
            for t_key, ten in st_val.items():
                if t_key == "step":
                    continue
                if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                    raise ValueError(f"non-finite or invalid tensor {t_key} in {type(self).__name__} state")
                if t_key in ("peak_rms", "peak_sv_rms", "m_scale", "v_scale", "g_prev_scale"):
                    if ten.ndim != 0 or float(ten.item()) < 0.0:
                        raise ValueError(f"invalid scalar {t_key} tensor in {type(self).__name__} state")
        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError(f"invalid {type(self).__name__} critic regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        clean = {k: v for k, v in state_dict.items() if k not in ("law4_alt", "regularizer")}
        Optimizer.load_state_dict(self, clean)
        self.clear_sampled_rows()
        meta = state_dict["law4_alt"]
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


# ---------------------------------------------------------------------------
# 1. law4_adam_equil: 2 states/param, 0 NS, homogeneous peak_rms Adam + MP
# ---------------------------------------------------------------------------

class Law4AdamEquilOptimizer(_Law4AltBaseOptimizer):
    """Homogeneous `peak_rms` coordinate-wise Adam (beta1=0.0, beta2=0.999) + MP + secant band (2 states, 0 NS)."""

    family = "law4_adam_equil"
    recipe_optimizer_family = "law4_adam_equil"
    _allowed_state_keys = ("step", "m_unit", "m_scale", "v_rms", "v_scale", "peak_rms", "peak_sv_rms", "m_prior")

    def __init__(self, params: Iterable[Any], *, settle_rel: float = 0.015, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.settle_rel = float(settle_rel)

    def _apply_network_groups(
        self,
        *,
        step_scale: float,
        origin_damping: float,
        stiff_gate_raw: float,
        pre_gain_base: float,
        is_critic_opt: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], float]:
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        coh_vals: List[torch.Tensor] = []
        pre_gain_applied = 1.0

        for grp in self.param_groups:
            role = grp["role"]
            player = self._player(role)
            stiff_gate = stiff_gate_raw if player == "generator" else 0.0
            b1, b2 = grp["betas"]
            eps = float(grp["eps"])
            s_base = self.smoothing * (1.0 - stiff_gate)
            s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0

            if player == "prior":
                b1_eff = max(float(b1), 0.75) if self._v4_prior_scale < 1.0 else float(b1)
            elif stiff_gate > 0.0:
                b1_eff = max(float(b1), 0.75 * stiff_gate)
            else:
                b1_eff = float(b1)
            b2_eff = min(max(float(b2), 0.95), 0.999)

            if player == "generator":
                pre_g = min(1.0, pre_gain_base * self._v4_wakeup_boost)
                pre_gain_applied = pre_g
                base_lr = float(grp["lr"]) * step_scale * self._v4_g_scale * pre_g
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
                    self._apply_prior_update(
                        p, grad, st, grp, base_lr=base_lr, b1_eff=b1_eff, s_prior_base=s_prior_base
                    )
                    continue

                rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                gf = grad.float()
                finite = torch.isfinite(gf).all()
                gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                g_max = gf_safe.abs().amax()
                g_active = finite & (g_max > 1e-38)

                is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                mp_scale, sv_denom, _ = _mp_scale_for_tensor(p, is_conv, self._conv_layouts)

                is_scalar_1d = p.ndim == 1 and p.numel() == 1
                b1_use = max(b1_eff, 0.80) if is_scalar_1d else b1_eff

                if "m_unit" not in st:
                    st["m_unit"] = torch.zeros_like(p, dtype=torch.float32)
                    st["m_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    st["v_rms"] = torch.zeros_like(p, dtype=torch.float32)
                    st["v_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    st["peak_sv_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)

                m_prev_sc = st["m_scale"]
                v_prev_sc = st["v_scale"]
                sc = torch.maximum(torch.maximum(m_prev_sc, v_prev_sc), g_max).clamp_min(1e-38)
                m_rescaled = st["m_unit"] * (m_prev_sc / sc)
                v_rescaled = st["v_rms"] * (v_prev_sc / sc)
                g_scaled = gf_safe / sc
                if b1_use > 0.0:
                    m_next = torch.where(finite, m_rescaled.mul(b1_use).add(g_scaled, alpha=1.0 - b1_use), m_rescaled)
                else:
                    m_next = torch.where(finite, g_scaled, m_rescaled)
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

                bc1 = 1.0 - (b1_use ** step_t) if b1_use > 0.0 else 1.0
                bc2_sqrt = math.sqrt(1.0 - (b2_eff ** step_t))
                m_hat = m_next / max(bc1, 1e-8)
                v_hat = v_next / max(bc2_sqrt, 1e-8)

                cur_rms = (v_hat.square().mean() + 1e-36).sqrt() * sc
                peak_new = torch.maximum(st["peak_rms"] * 0.9995, torch.where(finite, cur_rms, st["peak_rms"].new_zeros(())))
                st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))
                peak_scaled = (peak_new / sc.clamp_min(1e-38)).clamp_max(1e6)

                cur_sv_rms = _safe_sv_rms(gf_safe, g_max, sv_denom)
                peak_sv_new = torch.maximum(
                    st["peak_sv_rms"] * 0.9995,
                    torch.where(finite, cur_sv_rms, st["peak_sv_rms"].new_zeros(())),
                )
                st["peak_sv_rms"].copy_(torch.where(finite, peak_sv_new, st["peak_sv_rms"]))

                has_smooth = s_base > 0.0
                s_eff_t = (
                    torch.minimum(gf_safe.new_tensor(s_base), peak_sv_new * 0.03)
                    if has_smooth
                    else gf_safe.new_zeros(())
                )

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

                if rate_p <= 0.0:
                    continue
                if track_secant:
                    coh_vals.append(coh)

                if p.ndim >= 2:
                    denom_elem = v_hat + (self.settle_rel * peak_scaled + 1e-8)
                    u_elem = torch.where(g_active, m_hat / denom_elem, torch.zeros_like(m_hat))
                    if has_smooth:
                        g_sv = ((m_norm / sv_denom) * sc).clamp_min(0.0)
                        s_scale = _safe_tikhonov_ratio(g_sv, s_eff_t)
                    else:
                        s_scale = u_elem.new_ones(())
                    step_mult = (-rate_p * mp_scale) * coh * s_scale
                    delta = (u_elem * step_mult).to(p.dtype)
                else:
                    direc = torch.where(g_active, (m_hat * sc) if has_smooth else m_hat, torch.zeros_like(m_hat))
                    d_norm = direc.norm()
                    denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps) if has_smooth else (d_norm + eps)
                    unit_1d = torch.where(g_active & (d_norm > 1e-30), direc / denom_1d, torch.zeros_like(direc))
                    delta = (unit_1d * ((-rate_p) * coh)).to(p.dtype)

                p.add_(delta)
                if track_secant:
                    applied_deltas.append((p, delta))

        return applied_deltas, coh_vals, pre_gain_applied


# ---------------------------------------------------------------------------
# 2A. law4_sign_mp: 0 tensor states/param, 0 NS, L_inf smooth SignSGD-MP
# ---------------------------------------------------------------------------

class Law4SignMPOptimizer(_Law4AltBaseOptimizer):
    """Zero-state smooth SignSGD-MP (`u = g / hypot(|g|, s_eff)`) + MP scaling + secant band (0 tensor states, 0 NS)."""

    family = "law4_sign_mp"
    recipe_optimizer_family = "law4_sign_mp"
    _allowed_state_keys = ("step", "peak_rms", "m_prior")

    def _apply_network_groups(
        self,
        *,
        step_scale: float,
        origin_damping: float,
        stiff_gate_raw: float,
        pre_gain_base: float,
        is_critic_opt: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], float]:
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        coh_vals: List[torch.Tensor] = []
        pre_gain_applied = 1.0

        for grp in self.param_groups:
            role = grp["role"]
            player = self._player(role)
            stiff_gate = stiff_gate_raw if player == "generator" else 0.0
            b1 = float(grp["betas"][0])
            s_base = self.smoothing * (1.0 - stiff_gate)
            s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0
            b1_prior = max(b1, 0.75) if self._v4_prior_scale < 1.0 else b1

            if player == "generator":
                pre_g = min(1.0, pre_gain_base * self._v4_wakeup_boost)
                pre_gain_applied = pre_g
                base_lr = float(grp["lr"]) * step_scale * self._v4_g_scale * pre_g
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
                    self._apply_prior_update(
                        p, grad, st, grp, base_lr=base_lr, b1_eff=b1_prior, s_prior_base=s_prior_base
                    )
                    continue

                rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                gf = grad.float()
                finite = torch.isfinite(gf).all()
                gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                g_max = gf_safe.abs().amax()
                g_active = finite & (g_max > 1e-38)

                is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                mp_scale, _, max_dim = _mp_scale_for_tensor(p, is_conv, self._conv_layouts)

                cur_elem_rms = _safe_elem_rms(gf_safe, g_max)
                if "peak_rms" not in st:
                    st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                peak_new = torch.maximum(
                    st["peak_rms"] * 0.9995,
                    torch.where(finite, cur_elem_rms, st["peak_rms"].new_zeros(())),
                )
                st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))

                if rate_p <= 0.0:
                    continue

                has_smooth = s_base > 0.0
                if has_smooth:
                    s_elem_cap = s_base / math.sqrt(float(max_dim))
                    s_eff = torch.minimum(gf_safe.new_tensor(s_elem_cap), peak_new * 0.03)
                else:
                    s_eff = gf_safe.new_zeros(())

                sc = torch.maximum(g_max, s_eff).clamp_min(1e-38)
                g_s = gf_safe / sc
                s_s = s_eff / sc
                denom = torch.hypot(g_s.abs(), s_s)
                u_sign = torch.where(
                    g_active & (denom > 1e-30),
                    g_s / denom.clamp_min(1e-30),
                    torch.zeros_like(g_s),
                )
                coh = gf_safe.new_ones(())
                if track_secant:
                    coh_vals.append(coh)

                delta = (u_sign * (-rate_p * mp_scale)).to(p.dtype)
                p.add_(delta)
                if track_secant:
                    applied_deltas.append((p, delta))

        return applied_deltas, coh_vals, pre_gain_applied


# ---------------------------------------------------------------------------
# 2B. law4_lion_mp: 1 state/param, 0 NS, Lion-style smooth sign momentum + MP
# ---------------------------------------------------------------------------

class Law4LionMPOptimizer(_Law4AltBaseOptimizer):
    """1-state Lion-style smooth sign momentum (`smooth_sign(b1*m + (1-b1)*g)`) + MP + `peak_rms` + secant band."""

    family = "law4_lion_mp"
    recipe_optimizer_family = "law4_lion_mp"
    _allowed_state_keys = ("step", "m_unit", "m_scale", "peak_rms", "m_prior")

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lion_beta1: float = 0.5,
        lion_beta2: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        if not (0.0 <= lion_beta1 < 1.0) or not (0.0 <= lion_beta2 < 1.0):
            raise ValueError("lion_beta1 and lion_beta2 must lie in [0, 1)")
        self.lion_beta1 = float(lion_beta1)
        self.lion_beta2 = float(lion_beta2)

    def _extra_meta(self) -> Dict[str, Any]:
        return {"lion_betas": [float(self.lion_beta1), float(self.lion_beta2)]}

    def _validate_extra_meta(self, meta: Dict[str, Any]) -> None:
        lb = meta.get("lion_betas")
        if not isinstance(lb, list) or len(lb) != 2 or lb != [float(self.lion_beta1), float(self.lion_beta2)]:
            raise ValueError("incompatible Law4LionMPOptimizer lion_betas")

    def _apply_network_groups(
        self,
        *,
        step_scale: float,
        origin_damping: float,
        stiff_gate_raw: float,
        pre_gain_base: float,
        is_critic_opt: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], float]:
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        coh_vals: List[torch.Tensor] = []
        pre_gain_applied = 1.0
        lb1, lb2 = self.lion_beta1, self.lion_beta2

        for grp in self.param_groups:
            role = grp["role"]
            player = self._player(role)
            stiff_gate = stiff_gate_raw if player == "generator" else 0.0
            b1 = float(grp["betas"][0])
            s_base = self.smoothing * (1.0 - stiff_gate)
            s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0
            b1_prior = max(b1, 0.75) if self._v4_prior_scale < 1.0 else b1

            if player == "generator":
                pre_g = min(1.0, pre_gain_base * self._v4_wakeup_boost)
                pre_gain_applied = pre_g
                base_lr = float(grp["lr"]) * step_scale * self._v4_g_scale * pre_g
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
                    self._apply_prior_update(
                        p, grad, st, grp, base_lr=base_lr, b1_eff=b1_prior, s_prior_base=s_prior_base
                    )
                    continue

                rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                gf = grad.float()
                finite = torch.isfinite(gf).all()
                gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                g_max = gf_safe.abs().amax()
                g_active = finite & (g_max > 1e-38)

                is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                mp_scale, _, max_dim = _mp_scale_for_tensor(p, is_conv, self._conv_layouts)

                if "m_unit" not in st:
                    st["m_unit"] = torch.zeros_like(p, dtype=torch.float32)
                    st["m_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)

                m_prev_sc = st["m_scale"]
                sc = torch.maximum(m_prev_sc, g_max).clamp_min(1e-38)
                m_prev = st["m_unit"] * (m_prev_sc / sc)
                g_s = gf_safe / sc

                if step_t == 1:
                    c_t = g_s
                else:
                    bc_prev = max(1.0 - (lb2 ** (step_t - 1)), 1e-8)
                    m_hat_prev = m_prev / bc_prev
                    c_t = lb1 * m_hat_prev + (1.0 - lb1) * g_s

                m_next = torch.where(finite, m_prev.mul(lb2).add(g_s, alpha=1.0 - lb2), m_prev)
                new_m_sc = m_next.abs().amax()
                safe_m_sc = torch.where(new_m_sc > 1e-38, new_m_sc, torch.ones_like(new_m_sc))
                st["m_unit"].copy_(torch.where(new_m_sc > 1e-38, m_next / safe_m_sc, torch.zeros_like(m_next)))
                st["m_scale"].copy_(torch.where(new_m_sc > 1e-38, new_m_sc, torch.zeros_like(new_m_sc)) * sc)

                cur_elem_rms = (c_t.square().mean() + 1e-36).sqrt() * sc
                peak_new = torch.maximum(
                    st["peak_rms"] * 0.9995,
                    torch.where(finite, cur_elem_rms, st["peak_rms"].new_zeros(())),
                )
                st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))

                if rate_p <= 0.0:
                    continue

                has_smooth = s_base > 0.0
                if has_smooth:
                    s_elem_cap = s_base / math.sqrt(float(max_dim))
                    s_eff = torch.minimum(gf_safe.new_tensor(s_elem_cap), peak_new * 0.03)
                else:
                    s_eff = gf_safe.new_zeros(())

                s_c = s_eff / sc
                c_max = c_t.abs().amax()
                c_sc = torch.maximum(c_max, s_c).clamp_min(1e-30)
                c_norm = c_t / c_sc
                s_norm = s_c / c_sc
                denom = torch.hypot(c_norm.abs(), s_norm)
                u_lion = torch.where(
                    g_active & (c_max > 1e-38) & (denom > 1e-30),
                    c_norm / denom.clamp_min(1e-30),
                    torch.zeros_like(c_norm),
                )
                coh = gf_safe.new_ones(())
                if track_secant:
                    coh_vals.append(coh)

                delta = (u_lion * (-rate_p * mp_scale)).to(p.dtype)
                p.add_(delta)
                if track_secant:
                    applied_deltas.append((p, delta))

        return applied_deltas, coh_vals, pre_gain_applied


# ---------------------------------------------------------------------------
# 3. law4_muon_m: 1 state/param, Muon positive Nesterov (beta1=0.85) + 5-step NS
# ---------------------------------------------------------------------------

class Law4MuonMOptimizer(_Law4AltBaseOptimizer):
    """Muon-style positive Nesterov momentum (beta1=0.85) + 5-step Quintic NS + secant band (1 state/param)."""

    family = "law4_muon_m"
    recipe_optimizer_family = "law4_muon_m"
    _allowed_state_keys = ("step", "m_unit", "m_scale", "peak_rms", "m_prior")

    def __init__(self, params: Iterable[Any], *, muon_beta: float = 0.85, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        if not (0.0 <= muon_beta < 1.0):
            raise ValueError("muon_beta must lie in [0, 1)")
        self.muon_beta = float(muon_beta)

    def _extra_meta(self) -> Dict[str, Any]:
        return {"muon_beta": float(self.muon_beta)}

    def _validate_extra_meta(self, meta: Dict[str, Any]) -> None:
        if meta.get("muon_beta") != float(self.muon_beta):
            raise ValueError("incompatible Law4MuonMOptimizer muon_beta")

    def _apply_network_groups(
        self,
        *,
        step_scale: float,
        origin_damping: float,
        stiff_gate_raw: float,
        pre_gain_base: float,
        is_critic_opt: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], float]:
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pending_dense: Dict[
            Tuple[int, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
        ] = {}
        pending_conv: Dict[
            Tuple[int, int, int, int, bool, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]],
        ] = {}
        coh_vals: List[torch.Tensor] = []
        pre_gain_applied = 1.0
        mu = self.muon_beta

        for grp in self.param_groups:
            role = grp["role"]
            player = self._player(role)
            stiff_gate = stiff_gate_raw if player == "generator" else 0.0
            b1 = float(grp["betas"][0])
            eps = float(grp["eps"])
            s_base = self.smoothing * (1.0 - stiff_gate)
            s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0
            b1_prior = max(b1, 0.75) if self._v4_prior_scale < 1.0 else b1

            if player == "generator":
                pre_g = min(1.0, pre_gain_base * self._v4_wakeup_boost)
                pre_gain_applied = pre_g
                base_lr = float(grp["lr"]) * step_scale * self._v4_g_scale * pre_g
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
                    self._apply_prior_update(
                        p, grad, st, grp, base_lr=base_lr, b1_eff=b1_prior, s_prior_base=s_prior_base
                    )
                    continue

                rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                gf = grad.float()
                finite = torch.isfinite(gf).all()
                gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                g_max = gf_safe.abs().amax()
                g_active = finite & (g_max > 1e-38)

                is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                _, sv_denom, _ = _mp_scale_for_tensor(p, is_conv, self._conv_layouts)
                cur_sv_rms = _safe_sv_rms(gf_safe, g_max, sv_denom)

                if "m_unit" not in st:
                    st["m_unit"] = torch.zeros_like(p, dtype=torch.float32)
                    st["m_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)

                peak_new = torch.maximum(
                    st["peak_rms"] * 0.9995,
                    torch.where(finite, cur_sv_rms, st["peak_rms"].new_zeros(())),
                )
                st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))

                has_smooth = s_base > 0.0
                s_eff_t = (
                    torch.minimum(gf_safe.new_tensor(s_base), peak_new * 0.03)
                    if has_smooth
                    else gf_safe.new_zeros(())
                )

                m_prev_sc = st["m_scale"]
                sc = torch.maximum(m_prev_sc, g_max).clamp_min(1e-38)
                m_prev = st["m_unit"] * (m_prev_sc / sc)
                g_s = gf_safe / sc
                m_next = torch.where(finite, m_prev.mul(mu).add(g_s, alpha=1.0 - mu), m_prev)

                new_m_sc = m_next.abs().amax()
                safe_m_sc = torch.where(new_m_sc > 1e-38, new_m_sc, torch.ones_like(new_m_sc))
                st["m_unit"].copy_(torch.where(new_m_sc > 1e-38, m_next / safe_m_sc, torch.zeros_like(m_next)))
                st["m_scale"].copy_(torch.where(new_m_sc > 1e-38, new_m_sc, torch.zeros_like(new_m_sc)) * sc)

                bc1 = max(1.0 - (mu ** step_t), 1e-8)
                m_hat = m_next / bc1
                # Nesterov lookahead direction (equals g_s at step 1, mu*m_t + (1-mu)*g_t asymptotically):
                d_nest = mu * m_hat + (1.0 - mu) * g_s
                raw_dir = (d_nest * sc) if has_smooth else d_nest
                direc = torch.where(g_active, raw_dir, torch.zeros_like(d_nest))

                if stiff_gate > 0.0 and step_t > 1:
                    g_n = g_s.norm()
                    m_n = m_prev.norm()
                    cos_ag = torch.where(
                        g_active & (g_n > 1e-30) & (m_n > 1e-30),
                        ((g_s * m_prev).sum() / (g_n * m_n).clamp_min(1e-30)).clamp(0.0, 1.0),
                        torch.zeros_like(g_n),
                    )
                    coh = (1.0 - stiff_gate) + stiff_gate * cos_ag
                else:
                    coh = gf_safe.new_ones(())

                if rate_p <= 0.0:
                    continue
                if track_secant:
                    coh_vals.append(coh)

                if is_conv and (p.shape[2] > 1 or p.shape[3] > 1):
                    c0, c1, kh, kw = (int(s) for s in p.shape)
                    is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                    out_ch = c1 * groups_c if is_trans else c0
                    in_ch = c0 if is_trans else c1 * groups_c
                    factor = math.sqrt(float(out_ch) / float(max(1, in_ch))) / float(kh * kw)
                    step_mult = (-rate_p * factor) * coh
                    pending_conv.setdefault((c0, c1, kh, kw, is_trans, groups_c, has_smooth), []).append(
                        (p, direc, step_mult, s_eff_t, track_secant)
                    )
                elif p.ndim >= 2:
                    mat = direc.reshape(p.shape[0], -1)
                    r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                    if is_conv:
                        c0, c1, _, _ = (int(s) for s in p.shape)
                        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                        out_ch = c1 * groups_c if is_trans else c0
                        in_ch = c0 if is_trans else c1 * groups_c
                        factor = math.sqrt(float(out_ch) / float(max(1, in_ch)))
                    else:
                        factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
                    step_mult = (-rate_p * factor) * coh
                    needs_t = r_d > c_d
                    mat_canon = mat.T if needs_t else mat
                    m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                    pending_dense.setdefault(m_key, []).append(
                        (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
                    )
                else:
                    d_norm = direc.norm()
                    denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps) if has_smooth else (d_norm + eps)
                    unit_1d = torch.where(g_active & (d_norm > 1e-30), direc / denom_1d, torch.zeros_like(direc))
                    delta_1d = (unit_1d * ((-rate_p) * coh)).to(p.dtype)
                    p.add_(delta_1d)
                    if track_secant:
                        applied_deltas.append((p, delta_1d))

        _dispatch_quintic_buckets(pending_dense, pending_conv, applied_deltas)
        return applied_deltas, coh_vals, pre_gain_applied


# ---------------------------------------------------------------------------
# 4. law4_optimistic_equil: 1 state/param, Optimistic/Extragradient Polar NS
# ---------------------------------------------------------------------------

class Law4OptimisticEquilOptimizer(_Law4AltBaseOptimizer):
    """Optimistic / Extragradient Minimax Polar (`g_eff = g_t + lam*(g_t - g_{t-1})`) + 5-step Quintic NS."""

    family = "law4_optimistic_equil"
    recipe_optimizer_family = "law4_optimistic_equil"
    _allowed_state_keys = ("step", "g_prev_unit", "g_prev_scale", "peak_rms", "m_prior")

    def __init__(self, params: Iterable[Any], *, opt_lambda: float = 0.5, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        if not (0.0 <= opt_lambda <= 2.0):
            raise ValueError("opt_lambda must lie in [0, 2]")
        self.opt_lambda = float(opt_lambda)

    def _extra_meta(self) -> Dict[str, Any]:
        return {"opt_lambda": float(self.opt_lambda)}

    def _validate_extra_meta(self, meta: Dict[str, Any]) -> None:
        if meta.get("opt_lambda") != float(self.opt_lambda):
            raise ValueError("incompatible Law4OptimisticEquilOptimizer opt_lambda")

    def _apply_network_groups(
        self,
        *,
        step_scale: float,
        origin_damping: float,
        stiff_gate_raw: float,
        pre_gain_base: float,
        is_critic_opt: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], float]:
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pending_dense: Dict[
            Tuple[int, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
        ] = {}
        pending_conv: Dict[
            Tuple[int, int, int, int, bool, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]],
        ] = {}
        coh_vals: List[torch.Tensor] = []
        pre_gain_applied = 1.0
        lam = self.opt_lambda

        for grp in self.param_groups:
            role = grp["role"]
            player = self._player(role)
            stiff_gate = stiff_gate_raw if player == "generator" else 0.0
            b1 = float(grp["betas"][0])
            eps = float(grp["eps"])
            s_base = self.smoothing * (1.0 - stiff_gate)
            s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0
            b1_prior = max(b1, 0.75) if self._v4_prior_scale < 1.0 else b1

            if player == "generator":
                pre_g = min(1.0, pre_gain_base * self._v4_wakeup_boost)
                pre_gain_applied = pre_g
                base_lr = float(grp["lr"]) * step_scale * self._v4_g_scale * pre_g
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
                    self._apply_prior_update(
                        p, grad, st, grp, base_lr=base_lr, b1_eff=b1_prior, s_prior_base=s_prior_base
                    )
                    continue

                rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                gf = grad.float()
                finite = torch.isfinite(gf).all()
                gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                g_max = gf_safe.abs().amax()
                g_active = finite & (g_max > 1e-38)

                is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                _, sv_denom, _ = _mp_scale_for_tensor(p, is_conv, self._conv_layouts)
                cur_sv_rms = _safe_sv_rms(gf_safe, g_max, sv_denom)

                if "g_prev_unit" not in st:
                    st["g_prev_unit"] = torch.zeros_like(p, dtype=torch.float32)
                    st["g_prev_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)

                peak_new = torch.maximum(
                    st["peak_rms"] * 0.9995,
                    torch.where(finite, cur_sv_rms, st["peak_rms"].new_zeros(())),
                )
                st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))

                has_smooth = s_base > 0.0
                s_eff_t = (
                    torch.minimum(gf_safe.new_tensor(s_base), peak_new * 0.03)
                    if has_smooth
                    else gf_safe.new_zeros(())
                )

                prev_sc = st["g_prev_scale"]
                has_prev = (step_t > 1) & (prev_sc > 0.0)
                sc = torch.maximum(prev_sc, g_max).clamp_min(1e-38)
                g_prev_s = st["g_prev_unit"] * (prev_sc / sc)
                g_s = gf_safe / sc
                g_eff_s = torch.where(has_prev, g_s + lam * (g_s - g_prev_s), g_s)

                # Save current finite gradient as g_{t-1} for next step (10^-38 safe):
                safe_g_max = torch.where(g_active, g_max, torch.ones_like(g_max))
                new_unit = torch.where(g_active, gf_safe / safe_g_max, torch.zeros_like(gf_safe))
                new_sc = torch.where(g_active, g_max, torch.zeros_like(g_max))
                st["g_prev_unit"].copy_(torch.where(finite, new_unit, st["g_prev_unit"]))
                st["g_prev_scale"].copy_(torch.where(finite, new_sc, st["g_prev_scale"]))

                raw_dir = (g_eff_s * sc) if has_smooth else g_eff_s
                direc = torch.where(g_active, raw_dir, torch.zeros_like(g_eff_s))

                if stiff_gate > 0.0 and step_t > 1:
                    g_n = g_s.norm()
                    p_n = g_prev_s.norm()
                    cos_ag = torch.where(
                        g_active & (g_n > 1e-30) & (p_n > 1e-30),
                        ((g_s * g_prev_s).sum() / (g_n * p_n).clamp_min(1e-30)).clamp(0.0, 1.0),
                        torch.zeros_like(g_n),
                    )
                    coh = (1.0 - stiff_gate) + stiff_gate * cos_ag
                else:
                    coh = gf_safe.new_ones(())

                if rate_p <= 0.0:
                    continue
                if track_secant:
                    coh_vals.append(coh)

                if is_conv and (p.shape[2] > 1 or p.shape[3] > 1):
                    c0, c1, kh, kw = (int(s) for s in p.shape)
                    is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                    out_ch = c1 * groups_c if is_trans else c0
                    in_ch = c0 if is_trans else c1 * groups_c
                    factor = math.sqrt(float(out_ch) / float(max(1, in_ch))) / float(kh * kw)
                    step_mult = (-rate_p * factor) * coh
                    pending_conv.setdefault((c0, c1, kh, kw, is_trans, groups_c, has_smooth), []).append(
                        (p, direc, step_mult, s_eff_t, track_secant)
                    )
                elif p.ndim >= 2:
                    mat = direc.reshape(p.shape[0], -1)
                    r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                    if is_conv:
                        c0, c1, _, _ = (int(s) for s in p.shape)
                        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                        out_ch = c1 * groups_c if is_trans else c0
                        in_ch = c0 if is_trans else c1 * groups_c
                        factor = math.sqrt(float(out_ch) / float(max(1, in_ch)))
                    else:
                        factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
                    step_mult = (-rate_p * factor) * coh
                    needs_t = r_d > c_d
                    mat_canon = mat.T if needs_t else mat
                    m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                    pending_dense.setdefault(m_key, []).append(
                        (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
                    )
                else:
                    d_norm = direc.norm()
                    denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps) if has_smooth else (d_norm + eps)
                    unit_1d = torch.where(g_active & (d_norm > 1e-30), direc / denom_1d, torch.zeros_like(direc))
                    delta_1d = (unit_1d * ((-rate_p) * coh)).to(p.dtype)
                    p.add_(delta_1d)
                    if track_secant:
                        applied_deltas.append((p, delta_1d))

        _dispatch_quintic_buckets(pending_dense, pending_conv, applied_deltas)
        return applied_deltas, coh_vals, pre_gain_applied


# ---------------------------------------------------------------------------
# 5. law4_adafactor_mp: Sublinear O(m+n) factored RMS + MP + peak_rms (0 NS)
# ---------------------------------------------------------------------------

class Law4AdafactorMPOptimizer(_Law4AltBaseOptimizer):
    """Sublinear-memory O(m+n) row/col factored RMS (Adafactor) + MP + `peak_rms` + secant band (0 NS)."""

    family = "law4_adafactor_mp"
    recipe_optimizer_family = "law4_adafactor_mp"
    _allowed_state_keys = ("step", "v_row_unit", "v_col_unit", "v_scale", "peak_rms", "peak_sv_rms", "m_prior")

    def __init__(self, params: Iterable[Any], *, settle_rel: float = 0.015, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.settle_rel = float(settle_rel)

    def _apply_network_groups(
        self,
        *,
        step_scale: float,
        origin_damping: float,
        stiff_gate_raw: float,
        pre_gain_base: float,
        is_critic_opt: bool,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], float]:
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        coh_vals: List[torch.Tensor] = []
        pre_gain_applied = 1.0

        for grp in self.param_groups:
            role = grp["role"]
            player = self._player(role)
            stiff_gate = stiff_gate_raw if player == "generator" else 0.0
            b1, b2 = grp["betas"]
            eps = float(grp["eps"])
            s_base = self.smoothing * (1.0 - stiff_gate)
            s_prior_base = self.smoothing if self._v4_prior_scale >= 1.0 else 0.0
            b1_prior = max(float(b1), 0.75) if self._v4_prior_scale < 1.0 else float(b1)
            b2_eff = min(max(float(b2), 0.95), 0.999)

            if player == "generator":
                pre_g = min(1.0, pre_gain_base * self._v4_wakeup_boost)
                pre_gain_applied = pre_g
                base_lr = float(grp["lr"]) * step_scale * self._v4_g_scale * pre_g
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
                    self._apply_prior_update(
                        p, grad, st, grp, base_lr=base_lr, b1_eff=b1_prior, s_prior_base=s_prior_base
                    )
                    continue

                rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                track_secant = is_critic_opt or (not self._gen_param_ids or id(p) in self._gen_param_ids)

                gf = grad.float()
                finite = torch.isfinite(gf).all()
                gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                g_max = gf_safe.abs().amax()
                g_active = finite & (g_max > 1e-38)

                is_conv = (p.ndim == 4) and (id(p) in self._conv_layouts or p.shape[2] > 1 or p.shape[3] > 1)
                mp_scale, sv_denom, _ = _mp_scale_for_tensor(p, is_conv, self._conv_layouts)
                cur_sv_rms = _safe_sv_rms(gf_safe, g_max, sv_denom)

                has_smooth = s_base > 0.0

                if p.ndim >= 2:
                    r_d = int(p.shape[0])
                    c_d = int(p.numel() // max(1, r_d))
                    if "v_row_unit" not in st:
                        st["v_row_unit"] = torch.zeros(r_d, device=p.device, dtype=torch.float32)
                        st["v_col_unit"] = torch.zeros(c_d, device=p.device, dtype=torch.float32)
                        st["v_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        st["peak_sv_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)

                    v_prev_sc = st["v_scale"]
                    sc = torch.maximum(v_prev_sc, g_max).clamp_min(1e-38)
                    r_prev = st["v_row_unit"] * (v_prev_sc / sc)
                    c_prev = st["v_col_unit"] * (v_prev_sc / sc)
                    g_mat = (gf_safe / sc).reshape(r_d, c_d)
                    g_sq = g_mat.square()
                    g_row_sq = g_sq.mean(dim=1)
                    g_col_sq = g_sq.mean(dim=0)

                    r_next = torch.where(
                        finite,
                        (b2_eff * r_prev.square() + (1.0 - b2_eff) * g_row_sq).sqrt(),
                        r_prev,
                    )
                    c_next = torch.where(
                        finite,
                        (b2_eff * c_prev.square() + (1.0 - b2_eff) * g_col_sq).sqrt(),
                        c_prev,
                    )
                    new_v_sc = torch.maximum(r_next.amax(), c_next.amax())
                    safe_v_sc = torch.where(new_v_sc > 1e-38, new_v_sc, torch.ones_like(new_v_sc))
                    st["v_row_unit"].copy_(torch.where(new_v_sc > 1e-38, r_next / safe_v_sc, torch.zeros_like(r_next)))
                    st["v_col_unit"].copy_(torch.where(new_v_sc > 1e-38, c_next / safe_v_sc, torch.zeros_like(c_next)))
                    st["v_scale"].copy_(torch.where(new_v_sc > 1e-38, new_v_sc, torch.zeros_like(new_v_sc)) * sc)

                    bc2_sqrt = math.sqrt(1.0 - (b2_eff ** step_t))
                    r_hat = r_next / max(bc2_sqrt, 1e-8)
                    c_hat = c_next / max(bc2_sqrt, 1e-8)
                    r_rms = (r_hat.square().mean() + 1e-36).sqrt()
                    # Exact rank-1 factored RMS: sqrt((r_i^2 * c_j^2) / mean(r^2)) = (r_i * c_j) / rms(r):
                    v_hat_mat = (r_hat.unsqueeze(1) * c_hat.unsqueeze(0)) / r_rms.clamp_min(1e-30)

                    cur_rms = r_rms * sc
                    peak_new = torch.maximum(
                        st["peak_rms"] * 0.9995,
                        torch.where(finite, cur_rms, st["peak_rms"].new_zeros(())),
                    )
                    st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))
                    peak_scaled = (peak_new / sc.clamp_min(1e-38)).clamp_max(1e6)

                    peak_sv_new = torch.maximum(
                        st["peak_sv_rms"] * 0.9995,
                        torch.where(finite, cur_sv_rms, st["peak_sv_rms"].new_zeros(())),
                    )
                    st["peak_sv_rms"].copy_(torch.where(finite, peak_sv_new, st["peak_sv_rms"]))

                    if rate_p <= 0.0:
                        continue

                    s_eff_t = (
                        torch.minimum(gf_safe.new_tensor(s_base), peak_sv_new * 0.03)
                        if has_smooth
                        else gf_safe.new_zeros(())
                    )
                    denom_mat = v_hat_mat + (self.settle_rel * peak_scaled + 1e-8)
                    u_elem = torch.where(g_active, g_mat / denom_mat, torch.zeros_like(g_mat)).reshape_as(p)
                    if has_smooth:
                        s_scale = _safe_tikhonov_ratio(cur_sv_rms, s_eff_t)
                    else:
                        s_scale = u_elem.new_ones(())
                    coh = gf_safe.new_ones(())
                    if track_secant:
                        coh_vals.append(coh)
                    delta = (u_elem * ((-rate_p * mp_scale) * s_scale)).to(p.dtype)
                else:
                    if "peak_rms" not in st:
                        st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    peak_new = torch.maximum(
                        st["peak_rms"] * 0.9995,
                        torch.where(finite, cur_sv_rms, st["peak_rms"].new_zeros(())),
                    )
                    st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))
                    if rate_p <= 0.0:
                        continue
                    s_eff_t = (
                        torch.minimum(gf_safe.new_tensor(s_base), peak_new * 0.03)
                        if has_smooth
                        else gf_safe.new_zeros(())
                    )
                    raw_dir = gf_safe if has_smooth else (gf_safe / g_max.clamp_min(1e-38))
                    direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
                    d_norm = direc.norm()
                    denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps) if has_smooth else (d_norm + eps)
                    unit_1d = torch.where(g_active & (d_norm > 1e-30), direc / denom_1d, torch.zeros_like(direc))
                    coh = gf_safe.new_ones(())
                    if track_secant:
                        coh_vals.append(coh)
                    delta = (unit_1d * (-rate_p)).to(p.dtype)

                p.add_(delta)
                if track_secant:
                    applied_deltas.append((p, delta))

        return applied_deltas, coh_vals, pre_gain_applied


def _dispatch_quintic_buckets(
    pending_dense: Dict[
        Tuple[int, int, bool],
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
    ],
    pending_conv: Dict[
        Tuple[int, int, int, int, bool, int, bool],
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]],
    ],
    applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Shared batched 5-step Quintic NS dispatch (identical to unified_v11)."""
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


LAW4_ALT_OPTIMIZER_CLASSES: Dict[str, type] = {
    "law4_adam_equil": Law4AdamEquilOptimizer,
    "law4_sign_mp": Law4SignMPOptimizer,
    "law4_lion_mp": Law4LionMPOptimizer,
    "law4_muon_m": Law4MuonMOptimizer,
    "law4_optimistic_equil": Law4OptimisticEquilOptimizer,
    "law4_adafactor_mp": Law4AdafactorMPOptimizer,
}


__all__ = [
    "LAW4_ALT_FAMILIES",
    "LAW4_ALT_OPTIMIZER_CLASSES",
    "Law4AdafactorMPOptimizer",
    "Law4AdamEquilOptimizer",
    "Law4LionMPOptimizer",
    "Law4MuonMOptimizer",
    "Law4OptimisticEquilOptimizer",
    "Law4SignMPOptimizer",
]
