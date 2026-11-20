"""Round-4 Occam's Razor optimizer knockouts (`r4_mp_no_ns`, `r4_no_secant_band`).

Both subclass `UnifiedV4EquilibratedOptimizer` without modifying `particlegan/_unified_v4/`:
1. `R4MPNoNSOptimizer` (`family="r4_mp_no_ns"`): exact 1-factor knockout replacing Quintic
   Newton-Schulz polar orthogonalization on 2D dense and 4D convolutional layers with pure
   elementwise `O(params)` $10^{-38}$-safe `peak_rms` + Marchenko-Pastur RMS-to-RMS operator
   scaling (`t4_nash_v3` style; strictly `0` SVD, `0` eigh, `0` pinv, `0` NS, `0` autograd.grad).
2. `R4NoSecantBandOptimizer` (`family="r4_no_secant_band"`): exact 1-factor knockout disabling
   the realized function-space pushforward/logit secant trust band (`alpha_gain == 1.0`).
"""
from __future__ import annotations

from copy import copy
import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.optim import Optimizer

from .._unified_v4.optimizer import (
    UnifiedV4EquilibratedOptimizer,
    _band_scale,
    _eval_probe_forward,
    _smoothstep01,
)


class R4MPNoNSOptimizer(UnifiedV4EquilibratedOptimizer):
    """Exact `-NS` knockout: pure-elementwise $10^{-38}$-safe `peak_rms` + Marchenko-Pastur (`0` NS)."""

    family = "r4_mp_no_ns"
    recipe_optimizer_family = "r4_mp_no_ns"

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 0.014,
        betas: Tuple[float, float] = (0.0, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        smoothing: float = 0.0,
        settle_rel: float = 0.015,
        **options: Any,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
            smoothing=smoothing,
            **options,
        )
        self.family = "r4_mp_no_ns"
        self.recipe_optimizer_family = "r4_mp_no_ns"
        self.settle_rel = float(settle_rel)
        for grp in self.param_groups:
            if grp["algorithm"] == "unified_equil":
                grp["algorithm"] = "r4_mp_no_ns"

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.grad.is_complex():
                    raise ValueError("r4_mp_no_ns requires dense real gradients")
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
                        unit_r = torch.where(active_r, dir_r / denom_r, torch.zeros_like(dir_r))
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

                    # Underflow-safe (10^-38) dynamic scale-normalized moments + peak_rms settling
                    # for EVERY non-prior tensor (pure elementwise O(params), 0 Newton-Schulz!):
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    b1_use = max(b1_eff, 0.80) if is_scalar_1d else b1_eff
                    if "m_unit" not in st:
                        st["m_unit"] = torch.zeros_like(p, dtype=torch.float32)
                        st["m_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        st["v_rms"] = torch.zeros_like(p, dtype=torch.float32)
                        st["v_scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)

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

                    # Degree-0 scale-invariant peak_rms settling in FP32 (matches param dtype for exact checkpoint parity):
                    cur_rms = (v_hat.square().mean() + 1e-36).sqrt() * sc
                    peak_new = torch.maximum(st["peak_rms"] * 0.9995, cur_rms)
                    st["peak_rms"].copy_(torch.where(finite, peak_new, st["peak_rms"]))
                    peak_scaled = (peak_new / sc.clamp_min(1e-38)).clamp_max(1e6)

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

                    if rate_p <= 0.0:
                        continue

                    denom_elem = v_hat + (self.settle_rel * peak_scaled + 1e-8)
                    u_elem = torch.where(g_active, m_hat / denom_elem, torch.zeros_like(m_hat))

                    if p.ndim == 4 and (p.shape[2] > 1 or p.shape[3] > 1):
                        c0, c1, kh, kw = (int(s) for s in p.shape)
                        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
                        g_grp = max(1, int(groups_c))
                        if is_trans:
                            out_ch, in_ch = float(c1 * g_grp), float(c0)
                            k_eff = float(max(1, (kh * kw) // 4))
                        else:
                            out_ch, in_ch = float(c0), float(c1 * g_grp)
                            k_eff = float(max(1, kh * kw))
                        factor = math.sqrt(max(0.5, out_ch / max(1.0, in_ch)))
                        mp_denom = (math.sqrt(max(1.0, float(c0))) + math.sqrt(max(1.0, float(c1)))) * math.sqrt(k_eff)
                        if deep_gate > 0.0:
                            w_op = 1.0 / math.sqrt(float(math.sqrt(max(1.0, out_ch * in_ch)) * float(kh * kw)))
                            w_rms = p.float().square().mean().sqrt()
                            r_ratio = (w_rms / max(w_op, 1e-8)).clamp(c_zero, 2.5)
                            trust_mult = (1.0 - deep_gate) + deep_gate * r_ratio
                        else:
                            trust_mult = u_elem.new_ones(())
                        if s_eff > 0.0:
                            g_slice_rms = (
                                (m_norm * sc) / math.sqrt(max(1.0, min(out_ch, in_ch) * float(kh * kw)))
                            ).clamp_min(0.0)
                            conv_smooth = g_slice_rms / torch.hypot(
                                g_slice_rms, g_slice_rms.new_tensor(s_eff)
                            ).clamp_min(1e-30)
                            trust_mult = trust_mult * conv_smooth
                        step_mult = (-rate_p * (factor / mp_denom)) * coh * trust_mult
                        delta_4d = (u_elem * step_mult).to(p.dtype)
                        p.add_(delta_4d)
                        if track_secant:
                            applied_deltas.append((p, delta_4d))
                    elif p.ndim >= 2:
                        r_d = int(p.shape[0])
                        c_d = int(p.numel() // max(1, r_d))
                        factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
                        mp_denom = math.sqrt(max(1.0, float(r_d))) + math.sqrt(max(1.0, float(c_d)))
                        if deep_gate > 0.0:
                            w_op = 1.0 / math.sqrt(float(max(1, c_d)))
                            w_rms = p.float().square().mean().sqrt()
                            r_ratio = (w_rms / max(w_op, 1e-8)).clamp(c_zero, 2.5)
                            trust_mult = (1.0 - deep_gate) + deep_gate * r_ratio
                        else:
                            trust_mult = u_elem.new_ones(())
                        if s_eff > 0.0:
                            g_rms_2d = ((m_norm * sc) / math.sqrt(float(max(1, min(r_d, c_d))))).clamp_min(0.0)
                            s_scale = g_rms_2d / torch.hypot(g_rms_2d, g_rms_2d.new_tensor(s_eff)).clamp_min(1e-30)
                            trust_mult = trust_mult * s_scale
                        step_mult = (-rate_p * (factor / mp_denom)) * coh * trust_mult
                        delta_2d = (u_elem * step_mult).to(p.dtype)
                        p.add_(delta_2d)
                        if track_secant:
                            applied_deltas.append((p, delta_2d))
                    else:
                        direc = torch.where(g_active, (m_hat * sc) if s_eff > 0.0 else m_hat, torch.zeros_like(m_hat))
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

            # Preserve Law 4 realized function-space secant trust-band identically to unified_equil:
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

    def validate_state_dict(self, saved: Any) -> None:
        expected_keys = {"state", "param_groups", "unified_equil"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError("invalid R4MPNoNSOptimizer checkpoint schema")
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
            raise ValueError("incompatible R4MPNoNSOptimizer metadata")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) <= 0.0:
            raise ValueError("invalid R4MPNoNSOptimizer last_step_gain")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError("R4MPNoNSOptimizer group count mismatch")
        for cur_grp, sav_grp in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav_grp, dict) or len(sav_grp.get("params", ())) != len(cur_grp["params"]):
                raise ValueError("R4MPNoNSOptimizer group parameter mismatch")
            if sav_grp.get("role") != cur_grp["role"] or sav_grp.get("algorithm") != cur_grp["algorithm"]:
                raise ValueError("R4MPNoNSOptimizer group role/algorithm mismatch")
        if not isinstance(saved["state"], dict):
            raise ValueError("invalid R4MPNoNSOptimizer state map")
        for st_val in saved["state"].values():
            if not isinstance(st_val, dict):
                raise ValueError("invalid R4MPNoNSOptimizer parameter state")
            step_v = st_val.get("step", 0)
            if type(step_v) is not int or step_v < 0:
                raise ValueError("invalid R4MPNoNSOptimizer step counter")
            for t_key in ("m_unit", "m_scale", "v_rms", "v_scale", "peak_rms", "m_prior"):
                if t_key in st_val:
                    ten = st_val[t_key]
                    if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                        raise ValueError(f"non-finite or invalid tensor {t_key} in R4MPNoNSOptimizer state")
        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid R4MPNoNSOptimizer critic regularizer checkpoint")
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


class R4NoSecantBandOptimizer(UnifiedV4EquilibratedOptimizer):
    """Exact `-secant-band` knockout: disables realized function-space secant trust bands."""

    family = "r4_no_secant_band"
    recipe_optimizer_family = "r4_no_secant_band"

    def __init__(self, params: Iterable[Any], **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.family = "r4_no_secant_band"
        self.recipe_optimizer_family = "r4_no_secant_band"
        for grp in self.param_groups:
            if grp["algorithm"] == "unified_equil":
                grp["algorithm"] = "r4_no_secant_band"

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        rec = self._active_record()
        if rec is not None:
            rec._enable_secant_probe = False
            rec._g_probe_cache = None
            if hasattr(rec, "clear_step_probes"):
                rec.clear_step_probes()
        return super().step(closure=closure)


__all__ = [
    "R4MPNoNSOptimizer",
    "R4NoSecantBandOptimizer",
]
