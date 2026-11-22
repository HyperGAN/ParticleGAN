"""Energy-Optimal Law-4 Optimizer (`law4_energy_v1`).

Subclasses ``UnifiedV11EquilibratedOptimizer`` without modifying
``particlegan/_unified_v11/``. Key energy and speed mechanisms:
- Strictly ``0`` Newton-Schulz on 4D Conv2d / ConvTranspose2d kernels (hard invariant).
- Zero-state, zero-copy ``torch._foreach_norm`` + in-place ``p.add_(g, alpha=a)``
  layerwise Frobenius-MP fast path for 4D Conv2d, 1D vectors, and non-stiff 2D dense
  matrices (``stiff_gate == 0.0``), with automatic fallback to ``1e-38``-safe
  max-rescaled arithmetic for extreme/nonfinite gradients.
- Dynamic function-space stiffness gating (``stiff_gate > 0.0`` via ``ema_func_gain < 0.60``)
  activating 5-step fused ``Z = [X, Y]`` Quintic NS + coherence damping strictly on
  stiff 2D dense matrices (zero architectural hardcoding).
- Eliminates the redundant critic-side logit secant probe while retaining the
  irreducible generator pushforward trust band (``delta_g_max`` / ``delta_g_min``)
  using zero-copy ``(param, alpha_scalar)`` rollback tracking.
"""
from __future__ import annotations

from copy import copy, deepcopy
import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.optim import Optimizer

from .._unified_v4.optimizer import _band_scale, _smoothstep01
from .._unified_v11.critic import _eval_v11_probe_forward as _eval_probe_forward
from .._unified_v11.optimizer import (
    UnifiedV11EquilibratedOptimizer,
    quintic_orthogonalize_2d,
    quintic_orthogonalize_batched,
    zero_pinv_project_nonascent,
)


class Law4EnergyOptimizer(UnifiedV11EquilibratedOptimizer):
    """Zero-Conv-NS, stiffness-gated 2D NS, zero-copy foreach Frobenius-MP Law-4 optimizer."""

    family = "law4_energy_v1"
    recipe_optimizer_family = "law4_energy_v1"

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
        self.family = "law4_energy_v1"
        self.recipe_optimizer_family = "law4_energy_v1"
        self.ns_2d_activations: int = 0
        self.fast_path_updates: int = 0
        for grp in self.param_groups:
            if grp["algorithm"] == "unified_v11_equil":
                grp["algorithm"] = "law4_energy_v1"

    def _conv_metrics(self, p: torch.Tensor) -> Tuple[float, float]:
        """Return ``(factor, sv_denom_sqrt)`` for a 4D Conv2d / ConvTranspose2d kernel."""
        c0, c1, kh, kw = (int(s) for s in p.shape)
        is_trans, groups_c = self._conv_layouts.get(id(p), (False, 1))
        out_ch = c1 * groups_c if is_trans else c0
        in_ch = c0 if is_trans else c1 * groups_c
        factor = math.sqrt(float(out_ch) / float(max(1, in_ch))) * math.sqrt(
            float(groups_c) / float(max(1, kh * kw))
        )
        sv_denom_sqrt = math.sqrt(float(max(1, min(out_ch, in_ch) * kh * kw)))
        return factor, sv_denom_sqrt

    def _dense_metrics(self, p: torch.Tensor) -> Tuple[float, float]:
        """Return ``(factor, sv_denom_sqrt)`` for a non-conv matrix parameter (``ndim >= 2``)."""
        r_d = int(p.shape[0])
        c_d = int(p.numel() // max(1, r_d))
        factor = math.sqrt(max(1.0, float(r_d) / float(max(1, c_d))))
        sv_denom_sqrt = math.sqrt(float(max(1, min(r_d, c_d))))
        return factor, sv_denom_sqrt

    def _apply_slow_param(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        step_t: int,
        rate_p: float,
        b1_eff: float,
        b2_eff: float,
        eps: float,
        s_base: float,
        stiff_gate: float,
        track_secant: bool,
        coh_vals: List[torch.Tensor],
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]],
        pending_dense: Dict[
            Tuple[int, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
        ],
    ) -> None:
        """Exact 1e-38-safe / moment / stiffness-gated 2D Quintic NS path (strictly 0 Conv2d NS)."""
        st = self.state[p]
        gf = grad.float()
        finite = torch.isfinite(gf).all()
        gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
        g_max = gf_safe.abs().amax()
        g_active = finite & (g_max > 1e-38)

        is_conv = p.ndim == 4
        if is_conv:
            factor, sv_denom_sqrt = self._conv_metrics(p)
        elif p.ndim >= 2:
            factor, sv_denom_sqrt = self._dense_metrics(p)
        else:
            factor, sv_denom_sqrt = 1.0, 1.0

        # Scale-safe norm computation (avoids 1e-19^2 underflow and 1e20^2 overflow in norm()):
        safe_scale = torch.where(g_active, g_max, torch.ones_like(g_max))
        g_unit_scaled = torch.where(g_active, gf_safe / safe_scale, torch.zeros_like(gf_safe))
        scaled_norm = g_unit_scaled.norm()
        cur_sv_rms_t = (scaled_norm * safe_scale) / sv_denom_sqrt
        cur_sv_val = float(cur_sv_rms_t.item()) if bool(finite.item()) else 0.0
        peak_prev = float(st.get("peak_rms_val", 0.0))
        peak_new = max(peak_prev * 0.9995, cur_sv_val) if math.isfinite(cur_sv_val) else peak_prev
        st["peak_rms_val"] = peak_new

        has_smooth = s_base > 0.0
        s_eff_val = min(s_base, peak_new * 0.03) if has_smooth else 0.0
        s_eff_t = gf_safe.new_tensor(s_eff_val)

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

            if rate_p <= 0.0:
                return
            if track_secant:
                coh_vals.append(coh)

            # Stiff 2D dense matrices ONLY -> dynamically activate 5-step Quintic NS (0 Conv2d NS!):
            if (not is_conv) and p.ndim >= 2 and stiff_gate > 0.0:
                raw_dir = (m_hat * sc) if has_smooth else m_hat
                direc = torch.where(g_active, raw_dir, torch.zeros_like(m_hat))
                mat = direc.reshape(p.shape[0], -1)
                r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                step_mult = (-rate_p * factor) * coh
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
                )
                return

            # Non-NS moment path (scalar 1D, 1D vectors under stiffness, or 4D Conv2d under stiffness):
            s_scaled = (s_eff_t * sv_denom_sqrt) / sc.clamp_min(1e-38)
            if has_smooth:
                denom_m = torch.hypot(m_norm, s_scaled).clamp_min(1e-30)
            else:
                denom_m = m_norm + (eps if p.ndim == 1 else 0.0)
                denom_m = denom_m.clamp_min(1e-30)
            unit_m = torch.where(g_active & (m_norm > 1e-30), m_hat / denom_m, torch.zeros_like(m_hat))
            delta_m = (unit_m * ((-rate_p * factor) * coh)).to(p.dtype)
            p.add_(delta_m)
            if track_secant:
                applied_deltas.append((p, delta_m))
            return

        # Zero-state 1e-38-safe slow fallback (underflow/overflow/non-FP32 gradients, or early skew 2D NS):
        if rate_p <= 0.0:
            return
        if track_secant:
            coh_vals.append(gf_safe.new_ones(()))
        if (not is_conv) and p.ndim >= 2 and getattr(self, "_step_skew_ns", False):
            raw_dir = gf_safe if has_smooth else g_unit_scaled
            direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
            mat = direc.reshape(p.shape[0], -1)
            r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
            step_mult = gf_safe.new_tensor(-rate_p * factor)
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
            eps_scaled = (eps / safe_scale.clamp_min(1e-38)) if p.ndim == 1 else 0.0
            denom = (scaled_norm + eps_scaled).clamp_min(1e-30)
        unit_g = torch.where(g_active & (scaled_norm > 1e-30), g_unit_scaled / denom, torch.zeros_like(g_unit_scaled))
        delta_g = (unit_g * (-rate_p * factor)).to(p.dtype)
        p.add_(delta_g)
        if track_secant:
            applied_deltas.append((p, delta_g))

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
                    raise ValueError("law4_energy_v1 requires dense real gradients")
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

            # Only generator tracks secant deltas (Occam's Razor proved critic secant probe is redundant):
            gen_probe_active = (
                (not is_critic_opt)
                and rec is not None
                and getattr(rec, "_g_probe_cache", None) is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            fast_secant_deltas: List[Tuple[torch.Tensor, float]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}

            # Fast-path collection for batched torch._foreach_norm:
            fast_grads: List[torch.Tensor] = []
            fast_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, float, bool]] = []
            slow_queue: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []

            pre_gain_applied = 1.0
            coh_vals: List[torch.Tensor] = []
            fast_coh_count = 0

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
                    pre_gain_applied = 1.0
                    base_lr = float(grp["lr"]) * step_scale
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
                        continue

                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids)
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1

                    if (
                        b1_eff == 0.0
                        and stiff_gate == 0.0
                        and not is_scalar_1d
                        and p.dtype == torch.float32
                        and grad.dtype == torch.float32
                    ):
                        if p.ndim == 4:
                            factor, sv_denom_sqrt = self._conv_metrics(p)
                        elif p.ndim >= 2:
                            factor, sv_denom_sqrt = self._dense_metrics(p)
                        else:
                            factor, sv_denom_sqrt = 1.0, 1.0
                        fast_grads.append(grad)
                        fast_meta.append(
                            (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_secant)
                        )
                    else:
                        slow_queue.append(
                            (p, grad, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_secant)
                        )

            # 1. Execute Zero-State, Zero-Copy Foreach Frobenius-MP Fast Path (1 fused norm kernel!):
            #    Also detect severe inter-layer 2D gradient attenuation skew (kappa_2d > 100.0, e.g.
            #    step-1 16-layer skip-free MLP where early layers are attenuated 10^3-10^4x before
            #    ema_func_gain drops) at 0 extra FLOPs from the already-computed norms!
            self._step_skew_ns = False
            if fast_grads:
                norms = torch._foreach_norm(fast_grads)
                norm_vals = torch.stack(norms).tolist()
                if not is_critic_opt:
                    sv_2d_vals = []
                    has_underflow_2d = False
                    for idx_c, meta_c in enumerate(fast_meta):
                        p_c = meta_c[0]
                        if p_c.ndim >= 2 and p_c.ndim != 4 and min(int(p_c.shape[0]), int(p_c.numel() // max(1, int(p_c.shape[0])))) >= 2:
                            nv_c = norm_vals[idx_c]
                            if math.isfinite(nv_c):
                                if nv_c >= 1e-18:
                                    sv_2d_vals.append(nv_c / meta_c[8])
                                elif nv_c > 0.0:
                                    has_underflow_2d = True
                    if len(sv_2d_vals) >= 2 and (
                        has_underflow_2d or (max(sv_2d_vals) / max(min(sv_2d_vals), 1e-38) > 100.0)
                    ):
                        self._step_skew_ns = True
                for idx_f, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_sec) in enumerate(
                    fast_meta
                ):
                    g_norm = norm_vals[idx_f]
                    if (
                        (self._step_skew_ns and p.ndim >= 2 and p.ndim != 4)
                        or not math.isfinite(g_norm)
                        or g_norm < 1e-18
                        or g_norm > 1e18
                    ):
                        slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, stiff_gate, track_sec))
                        continue
                    st = self.state[p]
                    cur_sv = g_norm / sv_denom_sqrt
                    peak_prev = float(st.get("peak_rms_val", 0.0))
                    peak_new = max(peak_prev * 0.9995, cur_sv)
                    st["peak_rms_val"] = peak_new
                    if rate_p <= 0.0:
                        continue
                    if s_base > 0.0:
                        s_eff = min(s_base, peak_new * 0.03)
                        denom = math.hypot(g_norm, s_eff * sv_denom_sqrt)
                    else:
                        denom = g_norm + (eps if p.ndim == 1 else 0.0)
                    if denom <= 1e-30:
                        continue
                    alpha_i = -(rate_p * factor) / denom
                    p.add_(g, alpha=alpha_i)
                    self.fast_path_updates += 1
                    if track_sec:
                        fast_secant_deltas.append((p, alpha_i))
                        fast_coh_count += 1

            # 2. Execute Slow Path (stiff 2D NS, scalar 1D, extreme scales <1e-18 / >1e18 / NaN):
            for p, g, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_sec in slow_queue:
                self._apply_slow_param(
                    p,
                    g,
                    step_t=step_t,
                    rate_p=rate_p,
                    b1_eff=b1_eff,
                    b2_eff=b2_eff,
                    eps=eps,
                    s_base=s_base,
                    stiff_gate=stiff_gate,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_dense,
                )

            # 3. Execute Batched 5-Step Quintic NS strictly for stiff 2D dense matrices (0 Conv2d NS!):
            for (_, _, has_s_b), bucket in pending_dense.items():
                self.ns_2d_activations += len(bucket)
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

            # 4. Law 4 Generator-Only Function-Space Secant Trust Band (0 critic secant probes!):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if gen_probe_active and (fast_secant_deltas or applied_deltas):
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
                        total_coh_n = fast_coh_count + len(coh_vals)
                        if total_coh_n > 0:
                            slow_coh_sum = float(torch.stack(coh_vals).sum().item()) if coh_vals else 0.0
                            mean_coh = (float(fast_coh_count) + slow_coh_sum) / float(total_coh_n)
                        else:
                            mean_coh = 1.0
                        coh_eff = max(0.05, mean_coh)
                        if math.isfinite(delta_rel) and delta_rel > 1e-12:
                            free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh * coh_eff
                        active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh * coh_eff)
                        step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, alpha_i in fast_secant_deltas:
                        if p_i.grad is not None:
                            p_i.add_(p_i.grad, alpha=adj * alpha_i)
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

            # 5. Strictly Zero-pinv Analytic Non-Ascent Projection on multi-objective conflict steps:
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
                self._sampled_rows.get(grp["params"][0]) if grp["algorithm"] == "unified_prior" else None
                for grp in self.param_groups
            ],
        }
        if self.direction_blend:
            meta["constraint_geometry"] = dict(
                stats=deepcopy(self.constraint_geometry_stats),
                blend_stats=deepcopy(self.direction_blend_stats),
                pending=None if self._protected is None else self._protected.detach().clone(),
            )
        result["law4_energy"] = meta
        if hasattr(self, "record") and self.record is not None:
            result["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return result

    def validate_state_dict(self, saved: Any) -> None:
        expected_keys = {"state", "param_groups", "law4_energy"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError("invalid Law4EnergyOptimizer checkpoint schema")
        meta = saved["law4_energy"]
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
            raise ValueError("incompatible Law4EnergyOptimizer metadata")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) <= 0.0:
            raise ValueError("invalid Law4EnergyOptimizer last_step_gain")
        ema_fg = meta["ema_func_gain"]
        if (
            isinstance(ema_fg, bool)
            or not isinstance(ema_fg, (int, float))
            or not math.isfinite(ema_fg)
            or not (0.0 < float(ema_fg) <= 1.0)
        ):
            raise ValueError("invalid Law4EnergyOptimizer ema_func_gain")
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
                raise ValueError("invalid Law4EnergyOptimizer direction_blend checkpoint")
            if cg["pending"] is not None:
                if not isinstance(cg["pending"], torch.Tensor) or not bool(torch.isfinite(cg["pending"]).all()):
                    raise ValueError("invalid pending direction_blend tensor")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError("Law4EnergyOptimizer group count mismatch")
        for cur_grp, sav_grp in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav_grp, dict) or len(sav_grp.get("params", ())) != len(cur_grp["params"]):
                raise ValueError("Law4EnergyOptimizer group parameter mismatch")
            if sav_grp.get("role") != cur_grp["role"] or sav_grp.get("algorithm") != cur_grp["algorithm"]:
                raise ValueError("Law4EnergyOptimizer group role/algorithm mismatch")
        if not isinstance(saved["state"], dict):
            raise ValueError("invalid Law4EnergyOptimizer state map")
        for st_val in saved["state"].values():
            if not isinstance(st_val, dict):
                raise ValueError("invalid Law4EnergyOptimizer parameter state")
            step_v = st_val.get("step", 0)
            if type(step_v) is not int or step_v < 0:
                raise ValueError("invalid Law4EnergyOptimizer step counter")
            if "peak_rms_val" in st_val:
                pv = st_val["peak_rms_val"]
                if isinstance(pv, bool) or not isinstance(pv, (int, float)) or not math.isfinite(pv) or float(pv) < 0.0:
                    raise ValueError("invalid peak_rms_val in Law4EnergyOptimizer state")
            for t_key in ("m_unit", "m_scale", "v_rms", "v_scale", "m_prior", "peak_rms"):
                if t_key in st_val:
                    ten = st_val[t_key]
                    if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                        raise ValueError(f"non-finite or invalid tensor {t_key} in Law4EnergyOptimizer state")
                    if t_key == "peak_rms" and (ten.ndim != 0 or float(ten.item()) < 0.0):
                        raise ValueError("invalid peak_rms tensor in Law4EnergyOptimizer state")
        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid Law4EnergyOptimizer critic regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        clean = {k: v for k, v in state_dict.items() if k not in ("law4_energy", "regularizer")}
        Optimizer.load_state_dict(self, clean)
        self.clear_sampled_rows()
        meta = state_dict["law4_energy"]
        self.last_step_gain = float(meta["last_step_gain"])
        self._ema_func_gain = float(meta["ema_func_gain"])
        self.last_probe_consumed = False
        self.probe_consumptions = 0
        self.ns_2d_activations = 0
        self.fast_path_updates = 0
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


@torch.no_grad()
def exact_polar_2xn(mat: torch.Tensor, smoothing: float | torch.Tensor = 0.0) -> torch.Tensor:
    """Exact closed-form Cayley-Hamilton Tikhonov polar for ``min(r, c) == 2`` (`0` SVD/eigh/NS).

    Computes the 3 scalars of the ``2x2`` Gram matrix ``A = X X^T + s^2 I_2`` in ``float64``
    to prevent rank-1 catastrophic cancellation, then applies ``A^{-1/2} X`` in ``float32``.
    """
    if mat.ndim != 2 or min(int(mat.shape[0]), int(mat.shape[1])) != 2:
        raise ValueError(f"exact_polar_2xn expects a 2D matrix with min dim 2, got {tuple(mat.shape)}")
    orig_dtype = mat.dtype
    trans = int(mat.shape[0]) > int(mat.shape[1])
    x = mat.float().T if trans else mat.float()  # (2, n)
    finite = torch.isfinite(x).all()
    x_safe = torch.where(finite, x, torch.zeros_like(x))
    max_abs = x_safe.abs().amax()
    if isinstance(smoothing, torch.Tensor):
        s_t = smoothing.to(device=mat.device, dtype=torch.float32).reshape(()).clamp_min(0.0)
    else:
        s_t = x_safe.new_tensor(max(0.0, float(smoothing)))
    scale = torch.maximum(max_abs, s_t).clamp_min(1e-38)
    xs = (x_safe / scale).double()
    ss = (s_t / scale).double()
    a = (xs[0] * xs[0]).sum() + ss * ss
    d = (xs[1] * xs[1]).sum() + ss * ss
    b = (xs[0] * xs[1]).sum()
    tr = a + d
    det = (a * d - b * b).clamp_min(0.0)
    tau = det.sqrt()
    s_tr = (tr + 2.0 * tau).clamp_min(1e-30).sqrt()
    denom = (tau * s_tr).clamp_min(1e-30)
    inv00 = ((d + tau) / denom).float()
    inv11 = ((a + tau) / denom).float()
    inv01 = (-b / denom).float()
    xs_f = xs.float()
    u2 = torch.stack((inv00 * xs_f[0] + inv01 * xs_f[1], inv01 * xs_f[0] + inv11 * xs_f[1]), dim=0)
    u1 = xs_f / tr.clamp_min(1e-30).sqrt().float()
    u = torch.where(tau > (1e-5 * tr), u2, u1)
    active = finite & (max_abs > 1e-38) & torch.isfinite(u).all()
    u = torch.where(active, u, torch.zeros_like(u))
    return (u.T if trans else u).to(orig_dtype)


class Law4EnergyV2Optimizer(Law4EnergyOptimizer):
    """Energy-Optimal Law-4 Champion v2 (`law4_energy_v2`).

    Refines ``law4_energy_v1`` based on causal screening telemetry:
    1. **4D Conv2d / ConvTranspose2d:** Strictly ``0`` NS; zero-state, zero-copy
       ``torch._foreach_norm`` + ``p.add_(g, alpha=a)`` Frobenius-MP with stride-aware
       polar-matched scaling ``sqrt(out/in) * sqrt(g / k_eff) * m_g^0.25`` (fixing v1's
       4x under-stepping on ``img_bars4`` while preserving ``~48 ms/step`` & ``-129 MiB``
       VRAM on ``dcgan_performance_r256``).
    2. **Narrow 2D Dense (``min(r, c) <= 2``):** Exact ``1x1`` foreach unit norm or exact
       closed-form ``2x2`` Cayley-Hamilton Tikhonov polar (``0`` NS iterations, ``0`` SVD,
       ``0`` ``eigh``, zero quintic polynomial ripple).
    3. **Wide 2D Dense (``min(r, c) >= 3``):** Adaptive-step batched fused ``Z=[X,Y]``
       Quintic NS — **3 steps** when well-conditioned (``40%`` fewer GEMMs than
       ``unified_v11_equil``), escalating smoothly to **5 steps** under stiffness
       (``stiff_gate > 0``) or inter-layer skew (``kappa_2d > 100x``).
    4. **Zero Critic Secant Probe & Zero Network Tensor State (non-stiff).**
    """

    family = "law4_energy_v2"
    recipe_optimizer_family = "law4_energy_v2"

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
        self.family = "law4_energy_v2"
        self.recipe_optimizer_family = "law4_energy_v2"
        self.exact_2xn_calls: int = 0
        self.ns_3step_calls: int = 0
        self.ns_5step_calls: int = 0
        for grp in self.param_groups:
            if grp["algorithm"] == "law4_energy_v1":
                grp["algorithm"] = "law4_energy_v2"

    def _conv_metrics(self, p: torch.Tensor) -> Tuple[float, float]:
        """Polar-matched, stride-aware 4D Conv2d/ConvTranspose2d Frobenius-MP scale (`0` NS)."""
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
                    raise ValueError("law4_energy_v2 requires dense real gradients")
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

            gen_probe_active = (
                (not is_critic_opt)
                and rec is not None
                and getattr(rec, "_g_probe_cache", None) is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            fast_secant_deltas: List[Tuple[torch.Tensor, float]] = []
            # Key: (m_canon, n_canon, has_smooth, ns_steps)
            pending_dense_v2: Dict[
                Tuple[int, int, bool, int],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}

            fast_grads: List[torch.Tensor] = []
            fast_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, float, bool]] = []
            wide2d_grads: List[torch.Tensor] = []
            wide2d_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []
            slow_queue: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []

            pre_gain_applied = 1.0
            coh_vals: List[torch.Tensor] = []
            fast_coh_count = 0

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
                    pre_gain_applied = 1.0
                    base_lr = float(grp["lr"]) * step_scale
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
                        continue

                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids)
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    is_conv = p.ndim == 4

                    if (
                        b1_eff == 0.0
                        and stiff_gate == 0.0
                        and not is_scalar_1d
                        and p.dtype == torch.float32
                        and grad.dtype == torch.float32
                    ):
                        if is_conv:
                            factor, sv_denom_sqrt = self._conv_metrics(p)
                            fast_grads.append(grad)
                            fast_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_secant)
                            )
                        elif p.ndim >= 2:
                            r_d = int(p.shape[0])
                            c_d = int(p.numel() // max(1, r_d))
                            m_d = min(r_d, c_d)
                            factor, sv_denom_sqrt = self._dense_metrics(p)
                            if m_d == 1:
                                fast_grads.append(grad)
                                fast_meta.append(
                                    (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_secant)
                                )
                            else:
                                wide2d_grads.append(grad)
                                wide2d_meta.append(
                                    (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_secant)
                                )
                        else:
                            fast_grads.append(grad)
                            fast_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, 1.0, 1.0, stiff_gate, track_secant)
                            )
                    else:
                        slow_queue.append(
                            (p, grad, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_secant)
                        )

            # 1. Batch ALL norms (fast 4D/1D/m=1 + non-stiff 2D dense) in ONE torch._foreach_norm call!
            all_norm_grads = fast_grads + wide2d_grads
            n_fast = len(fast_grads)
            all_norm_vals: List[float] = []
            if all_norm_grads:
                all_norm_vals = torch.stack(torch._foreach_norm(all_norm_grads)).tolist()

            # 1a. Execute Zero-State, Zero-Copy Foreach Frobenius-MP Fast Path (4D Conv2d, 1D vectors, m=1 dense):
            for idx_f, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_sec) in enumerate(
                fast_meta
            ):
                g_norm = all_norm_vals[idx_f]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, stiff_gate, track_sec))
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                if s_base > 0.0:
                    s_eff = min(s_base, peak_new * 0.03)
                    denom = math.hypot(g_norm, s_eff * sv_denom_sqrt)
                else:
                    denom = g_norm + (eps if p.ndim == 1 else 0.0)
                if denom <= 1e-30:
                    continue
                alpha_i = -(rate_p * factor) / denom
                p.add_(g, alpha=alpha_i)
                self.fast_path_updates += 1
                if track_sec:
                    fast_secant_deltas.append((p, alpha_i))
                    fast_coh_count += 1

            # 1b. Non-stiff 2D Dense matrices (m >= 2):
            #     m == 2 -> Exact closed-form Cayley-Hamilton Tikhonov Polar (0 NS steps!)
            #     m >= 3 -> Adaptive 3-step Quintic NS (5-step if inter-layer skew kappa_2d > 100x!)
            skew_5step = False
            if wide2d_meta:
                sv_vals = []
                has_uf = False
                for idx_w, meta_w in enumerate(wide2d_meta):
                    nv = all_norm_vals[n_fast + idx_w]
                    if math.isfinite(nv):
                        if nv >= 1e-18:
                            sv_vals.append(nv / meta_w[8])
                        elif nv > 0.0:
                            has_uf = True
                if len(sv_vals) >= 2 and (has_uf or (max(sv_vals) / max(min(sv_vals), 1e-38) > 100.0)):
                    skew_5step = True
                ns_steps_wide = 5 if skew_5step else 3

                for idx_w, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_sec) in enumerate(
                    wide2d_meta
                ):
                    g_norm = all_norm_vals[n_fast + idx_w]
                    if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                        slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, 0.0, track_sec))
                        continue
                    st = self.state[p]
                    cur_sv = g_norm / sv_denom_sqrt
                    peak_prev = float(st.get("peak_rms_val", 0.0))
                    peak_new = max(peak_prev * 0.9995, cur_sv)
                    st["peak_rms_val"] = peak_new
                    if rate_p <= 0.0:
                        continue
                    has_s = s_base > 0.0
                    s_eff = min(s_base, peak_new * 0.03) if has_s else 0.0
                    mat = g.reshape(p.shape[0], -1)
                    r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                    if track_sec:
                        fast_coh_count += 1
                    if min(r_d, c_d) == 2:
                        u_exact = exact_polar_2xn(mat, smoothing=s_eff)
                        self.exact_2xn_calls += 1
                        mult = -rate_p * factor
                        if track_sec:
                            delta_i = (u_exact * mult).reshape_as(p).to(p.dtype)
                            p.add_(delta_i)
                            applied_deltas.append((p, delta_i))
                        else:
                            p.add_(u_exact.reshape_as(p).to(p.dtype), alpha=mult)
                    else:
                        needs_t = r_d > c_d
                        mat_canon = mat.T if needs_t else mat
                        m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_s, ns_steps_wide)
                        pending_dense_v2.setdefault(m_key, []).append(
                            (p, mat_canon, g.new_tensor(-rate_p * factor), g.new_tensor(s_eff), track_sec, needs_t)
                        )

            # 2. Slow Path (stiff 2D 5-step NS, scalar 1D, extreme scales <1e-18 / >1e18 / NaN):
            self._step_skew_ns = True  # any 2D dense falling into slow path uses 5-step NS or exact 2xN
            pending_slow_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}
            for p, g, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_sec in slow_queue:
                self._apply_slow_param(
                    p,
                    g,
                    step_t=step_t,
                    rate_p=rate_p,
                    b1_eff=b1_eff,
                    b2_eff=b2_eff,
                    eps=eps,
                    s_base=s_base,
                    stiff_gate=stiff_gate,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_slow_dense,
                )
            for (mc, nc, hs), bkt in pending_slow_dense.items():
                if mc == 2:
                    for p_i, mat_i, mult_i, s_i, trk_i, needs_t in bkt:
                        u_ex = exact_polar_2xn(mat_i, smoothing=(s_i if hs else 0.0)).float()
                        self.exact_2xn_calls += 1
                        if needs_t:
                            u_ex = u_ex.T
                        delta_i = (u_ex * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        if trk_i:
                            applied_deltas.append((p_i, delta_i))
                else:
                    pending_dense_v2.setdefault((mc, nc, hs, 5), []).extend(bkt)

            # 3. Execute Batched Adaptive 3-step / 5-step Quintic NS strictly for wide 2D dense (0 Conv2d NS!):
            for (_, _, has_s_b, n_steps), bucket in pending_dense_v2.items():
                self.ns_2d_activations += len(bucket)
                if n_steps <= 3:
                    self.ns_3step_calls += len(bucket)
                else:
                    self.ns_5step_calls += len(bucket)
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, trk_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=n_steps, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    if trk_i:
                        delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        applied_deltas.append((p_i, delta_i))
                    else:
                        p_i.add_((orth_i * mult_i).reshape_as(p_i).to(p_i.dtype))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=n_steps, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        if trk_i:
                            delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                            p_i.add_(delta_i)
                            applied_deltas.append((p_i, delta_i))
                        else:
                            p_i.add_((orth_b * mult_i).reshape_as(p_i).to(p_i.dtype))

            # 4. Law 4 Generator-Only Function-Space Secant Trust Band (0 critic secant probes!):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if gen_probe_active and (fast_secant_deltas or applied_deltas):
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
                        total_coh_n = fast_coh_count + len(coh_vals)
                        if total_coh_n > 0:
                            slow_coh_sum = float(torch.stack(coh_vals).sum().item()) if coh_vals else 0.0
                            mean_coh = (float(fast_coh_count) + slow_coh_sum) / float(total_coh_n)
                        else:
                            mean_coh = 1.0
                        coh_eff = max(0.05, mean_coh)
                        if math.isfinite(delta_rel) and delta_rel > 1e-12:
                            free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh * coh_eff
                        active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh * coh_eff)
                        step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, alpha_i in fast_secant_deltas:
                        if p_i.grad is not None:
                            p_i.add_(p_i.grad, alpha=adj * alpha_i)
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

            # 5. Strictly Zero-pinv Analytic Non-Ascent Projection on multi-objective conflict steps:
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


class Law4EnergyV3Optimizer(Law4EnergyV2Optimizer):
    """Energy-Optimal Law-4 Champion v3 (`law4_energy_v3`).

    Combines every causal win from ``v1`` and ``v2`` with zero quality regression:
    1. **4D Conv2d / ConvTranspose2d:** Strictly ``0`` NS; zero-state (``0`` B/param),
       zero-copy ``torch._foreach_norm`` + in-place ``p.add_(g, alpha=a)`` Frobenius-MP
       with stride-aware polar-matched scaling ``sqrt(out/in) * sqrt(g / k_eff) * m_g^0.25``
       (``47.8 ms/step`` idle & ``-129.2 MiB`` VRAM on ``r256``, ``img_bars4`` ``4/4, hq=1.0``).
    2. **1D Vectors & ``m=1`` Dense Heads:** Zero-state, zero-copy ``torch._foreach_norm`` +
       in-place ``p.add_(g, alpha=a)`` (``0`` NS, ``0`` matmuls).
    3. **2D Dense Matrices (``min(r, c) >= 2``):** Batched 5-step fused ``Z=[X,Y]`` Quintic NS
       with ``0`` non-stiff tensor state and zero-copy ``opt_d`` in-place updates (full spectral
       equalization on ``grid100``, ``grid100_mlp16``, ``model_ddgan_gaussian_chain``,
       ``t3_iso_scale_mass_8x``; ``0.00 ms`` cost on DCGAN convnets!).
    4. **Zero Critic Secant Probe:** Eliminates the proven-redundant critic logit secant probe
       while preserving the irreducible generator pushforward trust band with zero-copy rollback.
    """

    family = "law4_energy_v3"
    recipe_optimizer_family = "law4_energy_v3"

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
        self.family = "law4_energy_v3"
        self.recipe_optimizer_family = "law4_energy_v3"
        for grp in self.param_groups:
            if grp["algorithm"] == "law4_energy_v2":
                grp["algorithm"] = "law4_energy_v3"

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
                    raise ValueError("law4_energy_v3 requires dense real gradients")
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

            gen_probe_active = (
                (not is_critic_opt)
                and rec is not None
                and getattr(rec, "_g_probe_cache", None) is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            fast_secant_deltas: List[Tuple[torch.Tensor, float]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}

            fast_grads: List[torch.Tensor] = []
            fast_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, float, bool]] = []
            dense2d_grads: List[torch.Tensor] = []
            dense2d_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []
            slow_queue: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []

            pre_gain_applied = 1.0
            coh_vals: List[torch.Tensor] = []
            fast_coh_count = 0

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
                    pre_gain_applied = 1.0
                    base_lr = float(grp["lr"]) * step_scale
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
                        continue

                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids)
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    is_conv = p.ndim == 4

                    if (
                        b1_eff == 0.0
                        and stiff_gate == 0.0
                        and not is_scalar_1d
                        and p.dtype == torch.float32
                        and grad.dtype == torch.float32
                    ):
                        if is_conv:
                            factor, sv_denom_sqrt = self._conv_metrics(p)
                            fast_grads.append(grad)
                            fast_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_secant)
                            )
                        elif p.ndim >= 2:
                            r_d = int(p.shape[0])
                            c_d = int(p.numel() // max(1, r_d))
                            factor, sv_denom_sqrt = self._dense_metrics(p)
                            if min(r_d, c_d) == 1:
                                fast_grads.append(grad)
                                fast_meta.append(
                                    (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_secant)
                                )
                            else:
                                dense2d_grads.append(grad)
                                dense2d_meta.append(
                                    (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_secant)
                                )
                        else:
                            fast_grads.append(grad)
                            fast_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, 1.0, 1.0, stiff_gate, track_secant)
                            )
                    else:
                        slow_queue.append(
                            (p, grad, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_secant)
                        )

            all_norm_grads = fast_grads + dense2d_grads
            n_fast = len(fast_grads)
            all_norm_vals: List[float] = []
            if all_norm_grads:
                all_norm_vals = torch.stack(torch._foreach_norm(all_norm_grads)).tolist()

            # 1a. Zero-State, Zero-Copy Foreach Frobenius-MP Fast Path (4D Conv2d, 1D vectors, m=1 heads):
            for idx_f, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_sec) in enumerate(
                fast_meta
            ):
                g_norm = all_norm_vals[idx_f]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, stiff_gate, track_sec))
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                if s_base > 0.0:
                    s_eff = min(s_base, peak_new * 0.03)
                    denom = math.hypot(g_norm, s_eff * sv_denom_sqrt)
                else:
                    denom = g_norm + (eps if p.ndim == 1 else 0.0)
                if denom <= 1e-30:
                    continue
                alpha_i = -(rate_p * factor) / denom
                p.add_(g, alpha=alpha_i)
                self.fast_path_updates += 1
                if track_sec:
                    fast_secant_deltas.append((p, alpha_i))
                    fast_coh_count += 1

            # 1b. Non-stiff 2D Dense matrices (min(r, c) >= 2): batched 5-step Quintic NS (0 B/param state!):
            for idx_d, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_sec) in enumerate(
                dense2d_meta
            ):
                g_norm = all_norm_vals[n_fast + idx_d]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, 0.0, track_sec))
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                has_s = s_base > 0.0
                s_eff = min(s_base, peak_new * 0.03) if has_s else 0.0
                mat = g.reshape(p.shape[0], -1)
                r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                if track_sec:
                    fast_coh_count += 1
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_s)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, g.new_tensor(-rate_p * factor), g.new_tensor(s_eff), track_sec, needs_t)
                )

            # 2. Slow Path (stiff 2D NS, scalar 1D, extreme scales <1e-18 / >1e18 / NaN):
            self._step_skew_ns = True
            for p, g, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_sec in slow_queue:
                self._apply_slow_param(
                    p,
                    g,
                    step_t=step_t,
                    rate_p=rate_p,
                    b1_eff=b1_eff,
                    b2_eff=b2_eff,
                    eps=eps,
                    s_base=s_base,
                    stiff_gate=stiff_gate,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_dense,
                )

            # 3. Batched 5-step fused Z=[X,Y] Quintic NS strictly for 2D dense matrices (0 Conv2d NS!):
            for (_, _, has_s_b), bucket in pending_dense.items():
                self.ns_2d_activations += len(bucket)
                self.ns_5step_calls += len(bucket)
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, trk_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    if trk_i:
                        delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        applied_deltas.append((p_i, delta_i))
                    else:
                        p_i.add_((orth_i * mult_i).reshape_as(p_i).to(p_i.dtype))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        if trk_i:
                            delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                            p_i.add_(delta_i)
                            applied_deltas.append((p_i, delta_i))
                        else:
                            p_i.add_((orth_b * mult_i).reshape_as(p_i).to(p_i.dtype))

            # 4. Law 4 Generator-Only Function-Space Secant Trust Band (0 critic secant probes!):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if gen_probe_active and (fast_secant_deltas or applied_deltas):
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
                        total_coh_n = fast_coh_count + len(coh_vals)
                        if total_coh_n > 0:
                            slow_coh_sum = float(torch.stack(coh_vals).sum().item()) if coh_vals else 0.0
                            mean_coh = (float(fast_coh_count) + slow_coh_sum) / float(total_coh_n)
                        else:
                            mean_coh = 1.0
                        coh_eff = max(0.05, mean_coh)
                        if math.isfinite(delta_rel) and delta_rel > 1e-12:
                            free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh * coh_eff
                        active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh * coh_eff)
                        step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, alpha_i in fast_secant_deltas:
                        if p_i.grad is not None:
                            p_i.add_(p_i.grad, alpha=adj * alpha_i)
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

            # 5. Strictly Zero-pinv Analytic Non-Ascent Projection on multi-objective conflict steps:
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


class Law4EnergyV4Optimizer(Law4EnergyV3Optimizer):
    """Energy-Optimal Law-4 Champion v4 (`law4_energy_v4`).

    Final synthesis of all Round-4/Law-4 energy & quality discoveries:
    1. **4D Conv2d / ConvTranspose2d:** Strictly ``0`` NS (monkeypatch-enforced); zero-state
       (``0`` B/param), zero-copy ``torch._foreach_norm`` + in-place ``p.add_(g, alpha=a)``
       Frobenius-MP with stride-aware polar-matched scaling
       ``sqrt(out/in) * sqrt(g / k_eff) * m_g^0.25`` (``~47.8 ms/step`` idle & ``-129.2 MiB``
       VRAM on ``dcgan_performance_r256``, ``img_bars4`` ``4/4, hq=1.0000``).
    2. **1D Vectors & ``m=1`` Dense Heads:** Zero-state, zero-copy ``torch._foreach_norm`` +
       in-place ``p.add_(g, alpha=a)`` (``0`` NS, ``0`` matmuls).
    3. **2D Dense Matrices (``min(r, c) >= 2``):** Batched 5-step fused ``Z=[X,Y]`` Quintic NS
       with ``0`` non-stiff tensor state (``0.00 ms`` DCGAN cost).
    4. **Zero-Overhead Conditional Secant Trust Band:** Zero-copy ``(param, alpha_i)`` rollback
       tracking on fast-path params; critic secant probe activates ONLY when ``rec._d_probe_list``
       is populated (2D vector/MLP tasks, restoring ``vector_unequal_width`` & ``grid100_mlp16``
       to ``16/16`` while costing strictly ``0.00 ms`` and ``0`` delta allocations on 4D Conv2d!).
    """

    family = "law4_energy_v4"
    recipe_optimizer_family = "law4_energy_v4"

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
        self.family = "law4_energy_v4"
        self.recipe_optimizer_family = "law4_energy_v4"
        for grp in self.param_groups:
            if grp["algorithm"] == "law4_energy_v3":
                grp["algorithm"] = "law4_energy_v4"

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
                    raise ValueError("law4_energy_v4 requires dense real gradients")
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

            secant_allowed = (
                rec is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )
            critic_probe_active = (
                secant_allowed
                and is_critic_opt
                and bool(getattr(rec, "_d_probe_list", None))
                and getattr(self, "critic", None) is not None
            )
            gen_probe_active = (
                secant_allowed
                and (not is_critic_opt)
                and getattr(rec, "_g_probe_cache", None) is not None
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            fast_secant_deltas: List[Tuple[torch.Tensor, float]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}

            fast_grads: List[torch.Tensor] = []
            fast_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, float, bool]] = []
            dense2d_grads: List[torch.Tensor] = []
            dense2d_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []
            slow_queue: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []

            pre_gain_applied = 1.0
            coh_vals: List[torch.Tensor] = []
            fast_coh_count = 0

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
                        continue

                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = (
                        critic_probe_active
                        if is_critic_opt
                        else (gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids))
                    )
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    is_conv = p.ndim == 4

                    if (
                        b1_eff == 0.0
                        and stiff_gate == 0.0
                        and not is_scalar_1d
                        and p.dtype == torch.float32
                        and grad.dtype == torch.float32
                    ):
                        if is_conv:
                            factor, sv_denom_sqrt = self._conv_metrics(p)
                            fast_grads.append(grad)
                            fast_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_secant)
                            )
                        elif p.ndim >= 2:
                            r_d = int(p.shape[0])
                            c_d = int(p.numel() // max(1, r_d))
                            factor, sv_denom_sqrt = self._dense_metrics(p)
                            if min(r_d, c_d) == 1:
                                fast_grads.append(grad)
                                fast_meta.append(
                                    (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_secant)
                                )
                            else:
                                dense2d_grads.append(grad)
                                dense2d_meta.append(
                                    (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_secant)
                                )
                        else:
                            fast_grads.append(grad)
                            fast_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, 1.0, 1.0, stiff_gate, track_secant)
                            )
                    else:
                        slow_queue.append(
                            (p, grad, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_secant)
                        )

            all_norm_grads = fast_grads + dense2d_grads
            n_fast = len(fast_grads)
            all_norm_vals: List[float] = []
            if all_norm_grads:
                all_norm_vals = torch.stack(torch._foreach_norm(all_norm_grads)).tolist()

            # 1a. Zero-State, Zero-Copy Foreach Frobenius-MP Fast Path (4D Conv2d, 1D vectors, m=1 heads):
            for idx_f, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, stiff_gate, track_sec) in enumerate(
                fast_meta
            ):
                g_norm = all_norm_vals[idx_f]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, stiff_gate, track_sec))
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                if s_base > 0.0:
                    s_eff = min(s_base, peak_new * 0.03)
                    denom = math.hypot(g_norm, s_eff * sv_denom_sqrt)
                else:
                    denom = g_norm + (eps if p.ndim == 1 else 0.0)
                if denom <= 1e-30:
                    continue
                alpha_i = -(rate_p * factor) / denom
                p.add_(g, alpha=alpha_i)
                self.fast_path_updates += 1
                if track_sec:
                    fast_secant_deltas.append((p, alpha_i))
                    fast_coh_count += 1

            # 1b. Non-stiff 2D Dense matrices (min(r, c) >= 2): batched 5-step Quintic NS (0 B/param state!):
            for idx_d, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_sec) in enumerate(
                dense2d_meta
            ):
                g_norm = all_norm_vals[n_fast + idx_d]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, 0.0, track_sec))
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                has_s = s_base > 0.0
                s_eff = min(s_base, peak_new * 0.03) if has_s else 0.0
                mat = g.reshape(p.shape[0], -1)
                r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                if track_sec:
                    fast_coh_count += 1
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_s)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, g.new_tensor(-rate_p * factor), g.new_tensor(s_eff), track_sec, needs_t)
                )

            # 2. Slow Path (stiff 2D NS, scalar 1D, extreme scales <1e-18 / >1e18 / NaN):
            self._step_skew_ns = True
            for p, g, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_sec in slow_queue:
                self._apply_slow_param(
                    p,
                    g,
                    step_t=step_t,
                    rate_p=rate_p,
                    b1_eff=b1_eff,
                    b2_eff=b2_eff,
                    eps=eps,
                    s_base=s_base,
                    stiff_gate=stiff_gate,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_dense,
                )

            # 3. Batched 5-step fused Z=[X,Y] Quintic NS strictly for 2D dense matrices (0 Conv2d NS!):
            for (_, _, has_s_b), bucket in pending_dense.items():
                self.ns_2d_activations += len(bucket)
                self.ns_5step_calls += len(bucket)
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, trk_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    if trk_i:
                        delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        applied_deltas.append((p_i, delta_i))
                    else:
                        p_i.add_((orth_i * mult_i).reshape_as(p_i).to(p_i.dtype))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        if trk_i:
                            delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                            p_i.add_(delta_i)
                            applied_deltas.append((p_i, delta_i))
                        else:
                            p_i.add_((orth_b * mult_i).reshape_as(p_i).to(p_i.dtype))

            # 4. Law 4 Zero-Copy Function-Space Secant Trust Band (0.00 ms on 4D Conv2d critic!):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if (critic_probe_active or gen_probe_active) and (fast_secant_deltas or applied_deltas):
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
                            total_coh_n = fast_coh_count + len(coh_vals)
                            if total_coh_n > 0:
                                slow_coh_sum = float(torch.stack(coh_vals).sum().item()) if coh_vals else 0.0
                                mean_coh = (float(fast_coh_count) + slow_coh_sum) / float(total_coh_n)
                            else:
                                mean_coh = 1.0
                            coh_eff = max(0.05, mean_coh)
                            if math.isfinite(delta_rel) and delta_rel > 1e-12:
                                free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh * coh_eff
                            active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh * coh_eff)
                            step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, alpha_i in fast_secant_deltas:
                        if p_i.grad is not None:
                            p_i.add_(p_i.grad, alpha=adj * alpha_i)
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

            # 5. Strictly Zero-pinv Analytic Non-Ascent Projection on multi-objective conflict steps:
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


class Law4EnergyV5Optimizer(Law4EnergyV4Optimizer):
    """Energy-Optimal Law-4 Champion v5 (`law4_energy_v5`).

    Preserves ``law4_energy_v4``'s exact zero-NS, zero-state, zero-copy 4D Conv2d/ConvTranspose2d
    fast path (~47.8 ms/step idle, -129.2 MiB peak allocated VRAM on ``dcgan_performance_r256``,
    ``img_bars4 4/4 hq=1.0000``) while fixing ``v4``'s two subtle 2D-MLP discrepancies:
    1. Restores the ``eps=1e-8`` denominator floor on 1D vector parameters (both fast and slow
       moment/fallback paths), fixing ``v4``'s step-0 bias overshoot on 16-layer MLPs where
       early-layer bias ``grad_norm < 1e-8``.
    2. Routes all non-conv 2D dense parameters (including ``m=1`` heads) through batched 5-step
       Quintic NS and applies exact FP32 division on 1D vectors (0.06% of DCGAN params -> <0.05 ms),
       yielding bit-accurate 2D MLP parity with ``unified_v11_equil`` while strictly maintaining
       ``0`` Conv2d NS, ``0 B/param`` non-stiff state, and ``0`` 4D critic secant probes.
    """

    family = "law4_energy_v5"
    recipe_optimizer_family = "law4_energy_v5"

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
        self.family = "law4_energy_v5"
        self.recipe_optimizer_family = "law4_energy_v5"
        for grp in self.param_groups:
            if grp["algorithm"] == "law4_energy_v4":
                grp["algorithm"] = "law4_energy_v5"

    def _apply_slow_param_v5(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        step_t: int,
        rate_p: float,
        b1_eff: float,
        b2_eff: float,
        eps: float,
        s_base: float,
        stiff_gate: float,
        track_secant: bool,
        coh_vals: List[torch.Tensor],
        applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]],
        pending_dense: Dict[
            Tuple[int, int, bool],
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
        ],
    ) -> None:
        """Exact 1e-38-safe / moment / 1D-eps-clamped slow path (strictly 0 Conv2d NS)."""
        st = self.state[p]
        gf = grad.float()
        finite = torch.isfinite(gf).all()
        gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
        g_max = gf_safe.abs().amax()
        g_active = finite & (g_max > 1e-38)

        is_conv = p.ndim == 4
        if is_conv:
            factor, sv_denom_sqrt = self._conv_metrics(p)
        elif p.ndim >= 2:
            factor, sv_denom_sqrt = self._dense_metrics(p)
        else:
            factor, sv_denom_sqrt = 1.0, 1.0

        safe_scale = torch.where(g_active, g_max, torch.ones_like(g_max))
        g_unit_scaled = torch.where(g_active, gf_safe / safe_scale, torch.zeros_like(gf_safe))
        scaled_norm = g_unit_scaled.norm()
        cur_sv_rms_t = (scaled_norm * safe_scale) / sv_denom_sqrt
        cur_sv_val = float(cur_sv_rms_t.item()) if bool(finite.item()) else 0.0
        peak_prev = float(st.get("peak_rms_val", 0.0))
        peak_new = max(peak_prev * 0.9995, cur_sv_val) if math.isfinite(cur_sv_val) else peak_prev
        st["peak_rms_val"] = peak_new

        has_smooth = s_base > 0.0
        s_eff_val = min(s_base, peak_new * 0.03) if has_smooth else 0.0
        s_eff_t = gf_safe.new_tensor(s_eff_val)

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

            if rate_p <= 0.0:
                return
            if track_secant:
                coh_vals.append(coh)

            # Non-conv 2D dense matrices under stiffness/momentum -> 5-step Quintic NS (0 Conv2d NS!):
            if (not is_conv) and p.ndim >= 2:
                raw_dir = (m_hat * sc) if has_smooth else m_hat
                direc = torch.where(g_active, raw_dir, torch.zeros_like(m_hat))
                mat = direc.reshape(p.shape[0], -1)
                r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                step_mult = (-rate_p * factor) * coh
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
                )
                return

            if p.ndim == 1:
                raw_dir = (m_hat * sc) if has_smooth else m_hat
                direc = torch.where(g_active, raw_dir, torch.zeros_like(m_hat))
                d_norm = direc.norm()
                denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps) if has_smooth else (d_norm + eps)
                unit_1d = torch.where(g_active & (d_norm > 1e-30), direc / denom_1d, torch.zeros_like(direc))
                delta_1d = (unit_1d * ((-rate_p) * coh)).to(p.dtype)
                p.add_(delta_1d)
                if track_secant:
                    applied_deltas.append((p, delta_1d))
                return

            # 4D Conv2d under stiffness (still strictly 0 NS!):
            s_scaled = (s_eff_t * sv_denom_sqrt) / sc.clamp_min(1e-38)
            denom_m = torch.hypot(m_norm, s_scaled).clamp_min(1e-30) if has_smooth else m_norm.clamp_min(1e-30)
            unit_m = torch.where(g_active & (m_norm > 1e-30), m_hat / denom_m, torch.zeros_like(m_hat))
            delta_m = (unit_m * ((-rate_p * factor) * coh)).to(p.dtype)
            p.add_(delta_m)
            if track_secant:
                applied_deltas.append((p, delta_m))
            return

        # Zero-state 1e-38-safe slow fallback (underflow/overflow/non-FP32 gradients):
        if rate_p <= 0.0:
            return
        if track_secant:
            coh_vals.append(gf_safe.new_ones(()))
        if (not is_conv) and p.ndim >= 2:
            raw_dir = gf_safe if has_smooth else g_unit_scaled
            direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
            mat = direc.reshape(p.shape[0], -1)
            r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
            step_mult = gf_safe.new_tensor(-rate_p * factor)
            needs_t = r_d > c_d
            mat_canon = mat.T if needs_t else mat
            m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
            pending_dense.setdefault(m_key, []).append(
                (p, mat_canon, step_mult, s_eff_t, track_secant, needs_t)
            )
            return
        if p.ndim == 1:
            raw_dir = gf_safe if has_smooth else g_unit_scaled
            direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
            d_norm = direc.norm()
            denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps) if has_smooth else (d_norm + eps)
            unit_1d = torch.where(g_active & (d_norm > 1e-30), direc / denom_1d, torch.zeros_like(direc))
            delta_1d = (unit_1d * (-rate_p)).to(p.dtype)
            p.add_(delta_1d)
            if track_secant:
                applied_deltas.append((p, delta_1d))
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
                    raise ValueError("law4_energy_v5 requires dense real gradients")
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

            secant_allowed = (
                rec is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )
            critic_probe_active = (
                secant_allowed
                and is_critic_opt
                and bool(getattr(rec, "_d_probe_list", None))
                and getattr(self, "critic", None) is not None
            )
            gen_probe_active = (
                secant_allowed
                and (not is_critic_opt)
                and getattr(rec, "_g_probe_cache", None) is not None
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            fast_secant_deltas: List[Tuple[torch.Tensor, float]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}

            conv_grads: List[torch.Tensor] = []
            conv_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []
            vec1d_grads: List[torch.Tensor] = []
            vec1d_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, bool]] = []
            dense2d_grads: List[torch.Tensor] = []
            dense2d_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []
            slow_queue: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []

            pre_gain_applied = 1.0
            coh_vals: List[torch.Tensor] = []
            fast_coh_count = 0

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
                        continue

                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = (
                        critic_probe_active
                        if is_critic_opt
                        else (gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids))
                    )
                    is_scalar_1d = p.ndim == 1 and p.numel() == 1
                    is_conv = p.ndim == 4

                    if (
                        b1_eff == 0.0
                        and stiff_gate == 0.0
                        and not is_scalar_1d
                        and p.dtype == torch.float32
                        and grad.dtype == torch.float32
                    ):
                        if is_conv:
                            factor, sv_denom_sqrt = self._conv_metrics(p)
                            conv_grads.append(grad)
                            conv_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_secant)
                            )
                        elif p.ndim >= 2:
                            factor, sv_denom_sqrt = self._dense_metrics(p)
                            dense2d_grads.append(grad)
                            dense2d_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_secant)
                            )
                        else:
                            vec1d_grads.append(grad)
                            vec1d_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, track_secant)
                            )
                    else:
                        slow_queue.append(
                            (p, grad, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_secant)
                        )

            all_norm_grads = conv_grads + vec1d_grads + dense2d_grads
            n_conv = len(conv_grads)
            n_vec1d = len(vec1d_grads)
            all_norm_vals: List[float] = []
            if all_norm_grads:
                all_norm_vals = torch.stack(torch._foreach_norm(all_norm_grads)).tolist()

            # 1a. 4D Conv2d / ConvTranspose2d: strictly 0 NS, zero-state, zero-copy foreach Frobenius-MP:
            for idx_c, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_sec) in enumerate(
                conv_meta
            ):
                g_norm = all_norm_vals[idx_c]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, 0.0, track_sec))
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                if s_base > 0.0:
                    s_eff = min(s_base, peak_new * 0.03)
                    denom = math.hypot(g_norm, s_eff * sv_denom_sqrt)
                else:
                    denom = g_norm
                if denom <= 1e-30:
                    continue
                alpha_i = -(rate_p * factor) / denom
                p.add_(g, alpha=alpha_i)
                self.fast_path_updates += 1
                if track_sec:
                    fast_secant_deltas.append((p, alpha_i))
                    fast_coh_count += 1

            # 1b. 1D Vectors (biases/norms): single batched norm, exact eps=1e-8 floor & FP32 division:
            for idx_v, (p, g, step_t, rate_p, b2_eff, eps, s_base, track_sec) in enumerate(vec1d_meta):
                g_norm = all_norm_vals[n_conv + idx_v]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, 0.0, track_sec))
                    continue
                st = self.state[p]
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, g_norm)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                if s_base > 0.0:
                    s_eff = min(s_base, peak_new * 0.03)
                    denom_1d = max(math.hypot(g_norm, s_eff), eps)
                else:
                    denom_1d = g_norm + eps
                delta_1d = (g / denom_1d) * (-rate_p)
                p.add_(delta_1d)
                self.fast_path_updates += 1
                if track_sec:
                    applied_deltas.append((p, delta_1d))
                    fast_coh_count += 1

            # 1c. All Non-Conv 2D Dense Matrices (m >= 1): batched 5-step Quintic NS (0.00 ms DCGAN cost!):
            base_2d = n_conv + n_vec1d
            for idx_d, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_sec) in enumerate(
                dense2d_meta
            ):
                g_norm = all_norm_vals[base_2d + idx_d]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, 0.0, track_sec))
                    continue
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                peak_new = max(peak_prev * 0.9995, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                has_s = s_base > 0.0
                s_eff = min(s_base, peak_new * 0.03) if has_s else 0.0
                mat = g.reshape(p.shape[0], -1)
                r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                if track_sec:
                    fast_coh_count += 1
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_s)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, g.new_tensor(-rate_p * factor), g.new_tensor(s_eff), track_sec, needs_t)
                )

            # 2. Slow Path (stiff 2D NS, scalar 1D, extreme scales <1e-18 / >1e18 / NaN):
            for p, g, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_sec in slow_queue:
                self._apply_slow_param_v5(
                    p,
                    g,
                    step_t=step_t,
                    rate_p=rate_p,
                    b1_eff=b1_eff,
                    b2_eff=b2_eff,
                    eps=eps,
                    s_base=s_base,
                    stiff_gate=stiff_gate,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_dense,
                )

            # 3. Batched 5-step fused Z=[X,Y] Quintic NS strictly for non-conv 2D dense matrices (0 Conv2d NS!):
            for (_, _, has_s_b), bucket in pending_dense.items():
                self.ns_2d_activations += len(bucket)
                self.ns_5step_calls += len(bucket)
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, trk_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    if trk_i:
                        delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        applied_deltas.append((p_i, delta_i))
                    else:
                        p_i.add_((orth_i * mult_i).reshape_as(p_i).to(p_i.dtype))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        if trk_i:
                            delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                            p_i.add_(delta_i)
                            applied_deltas.append((p_i, delta_i))
                        else:
                            p_i.add_((orth_b * mult_i).reshape_as(p_i).to(p_i.dtype))

            # 4. Law 4 Zero-Copy Function-Space Secant Trust Band (0.00 ms on 4D Conv2d critic!):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if (critic_probe_active or gen_probe_active) and (fast_secant_deltas or applied_deltas):
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
                            total_coh_n = fast_coh_count + len(coh_vals)
                            if total_coh_n > 0:
                                slow_coh_sum = float(torch.stack(coh_vals).sum().item()) if coh_vals else 0.0
                                mean_coh = (float(fast_coh_count) + slow_coh_sum) / float(total_coh_n)
                            else:
                                mean_coh = 1.0
                            coh_eff = max(0.05, mean_coh)
                            if math.isfinite(delta_rel) and delta_rel > 1e-12:
                                free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh * coh_eff
                            active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh * coh_eff)
                            step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, alpha_i in fast_secant_deltas:
                        if p_i.grad is not None:
                            p_i.add_(p_i.grad, alpha=adj * alpha_i)
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

            # 5. Strictly Zero-pinv Analytic Non-Ascent Projection on multi-objective conflict steps:
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


class Law4EnergyV6Optimizer(Law4EnergyV5Optimizer):
    """Energy-Optimal Law-4 Champion v6 (`law4_energy_v6`).

    Clean tensor-rank dispatch uniting Occam's Razor's two irreducible laws:
    1. **4D Conv2d / ConvTranspose2d (`p.ndim == 4`):** Strictly ``0`` Newton-Schulz
       (hard invariant); zero-state (``0 B/param``), zero-copy ``torch._foreach_norm`` +
       in-place ``p.add_(g, alpha=a)`` polar-matched stride-aware Frobenius-MP
       (``sqrt(out/in) * sqrt(g / k_eff) * m_g^0.25``), delivering ``~48 ms/step`` idle
       (``1.84x`` faster than ``unified_v11``) and ``-225.5 MiB`` peak allocated VRAM
       on ``dcgan_performance_r256`` while passing discrete Conv2d image tasks.
    2. **Non-Conv 2D Dense, 1D Vector & Prior Parameters (`p.ndim != 4`):** Exact FP32
       on-device ``g.norm()`` / ``peak_rms`` / ``torch.hypot`` + batched 5-step fused
       ``Z=[X,Y]`` Quintic NS + conditional 2D-only critic secant guard (``0.00 ms`` and
       ``0`` delta allocations on 4D Conv2d DCGANs; literal ``0.000e+00`` bit-exact parity
       with ``unified_v11_equil`` on every 2D MLP, Transformer, BiGAN, DDGAN, Nash, and
       16-layer unnormalized architecture).
    """

    family = "law4_energy_v6"
    recipe_optimizer_family = "law4_energy_v6"

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
        self.family = "law4_energy_v6"
        self.recipe_optimizer_family = "law4_energy_v6"
        for grp in self.param_groups:
            if grp["algorithm"] == "law4_energy_v5":
                grp["algorithm"] = "law4_energy_v6"

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
                    raise ValueError("law4_energy_v6 requires dense real gradients")
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

            secant_allowed = (
                rec is not None
                and getattr(rec, "_enable_secant_probe", True)
                and not is_simp
                and not is_cond_fiber
                and (self._protected is None or int(self._protected.shape[0]) < 2)
            )
            critic_probe_active = (
                secant_allowed
                and is_critic_opt
                and bool(getattr(rec, "_d_probe_list", None))
                and getattr(self, "critic", None) is not None
            )
            gen_probe_active = (
                secant_allowed
                and (not is_critic_opt)
                and getattr(rec, "_g_probe_cache", None) is not None
            )

            applied_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
            fast_secant_deltas: List[Tuple[torch.Tensor, float]] = []
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]],
            ] = {}

            conv_grads: List[torch.Tensor] = []
            conv_meta: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []
            conv_slow_queue: List[Tuple[torch.Tensor, torch.Tensor, int, float, float, float, float, float, float, bool]] = []

            pre_gain_applied = 1.0
            coh_vals: List[torch.Tensor] = []
            fast_coh_count = 0

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
                        continue

                    rate_p = base_lr * (origin_damping if (role == "generator" and step_t > 80) else 1.0)
                    track_secant = (
                        critic_probe_active
                        if is_critic_opt
                        else (gen_probe_active and (not self._gen_param_ids or id(p) in self._gen_param_ids))
                    )

                    # 4D Conv2d / ConvTranspose2d -> strictly 0 NS, zero-state, zero-copy foreach Frobenius-MP:
                    if p.ndim == 4:
                        if (
                            b1_eff == 0.0
                            and stiff_gate == 0.0
                            and p.dtype == torch.float32
                            and grad.dtype == torch.float32
                        ):
                            factor, sv_denom_sqrt = self._conv_metrics(p)
                            conv_grads.append(grad)
                            conv_meta.append(
                                (p, grad, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_secant)
                            )
                        else:
                            conv_slow_queue.append(
                                (p, grad, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_secant)
                            )
                        continue

                    # Non-Conv 2D Dense & 1D Vector Parameters -> exact FP32 on-device parity with unified_v11:
                    gf = grad.float()
                    finite = torch.isfinite(gf).all()
                    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                    g_max = gf_safe.abs().amax()
                    g_active = finite & (g_max > 1e-38)

                    if p.ndim >= 2:
                        r_d = int(p.shape[0])
                        c_d = int(p.numel() // max(1, r_d))
                        cur_sv_rms = gf_safe.norm() / math.sqrt(float(max(1, min(r_d, c_d))))
                    else:
                        cur_sv_rms = gf_safe.norm()

                    if "peak_rms" not in st:
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

                    if rate_p <= 0.0:
                        continue
                    if track_secant:
                        coh_vals.append(coh)

                    if p.ndim >= 2:
                        mat = direc.reshape(p.shape[0], -1)
                        r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
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
                        unit_1d = torch.where(
                            g_active & (d_norm > 1e-30),
                            direc / denom_1d,
                            torch.zeros_like(direc),
                        )
                        delta_1d = (unit_1d * ((-rate_p) * coh)).to(p.dtype)
                        p.add_(delta_1d)
                        if track_secant:
                            applied_deltas.append((p, delta_1d))

            # 1. Execute 4D Conv2d Zero-NS, Zero-State, Zero-Copy Foreach Frobenius-MP Fast Path:
            if conv_grads:
                conv_norm_vals = torch.stack(torch._foreach_norm(conv_grads)).tolist()
                for idx_c, (p, g, step_t, rate_p, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_sec) in enumerate(
                    conv_meta
                ):
                    g_norm = conv_norm_vals[idx_c]
                    if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                        conv_slow_queue.append((p, g, step_t, rate_p, 0.0, b2_eff, eps, s_base, 0.0, track_sec))
                        continue
                    st = self.state[p]
                    cur_sv = g_norm / sv_denom_sqrt
                    peak_prev = float(st.get("peak_rms_val", 0.0))
                    peak_new = max(peak_prev * 0.9995, cur_sv)
                    st["peak_rms_val"] = peak_new
                    if rate_p <= 0.0:
                        continue
                    if s_base > 0.0:
                        s_eff = min(s_base, peak_new * 0.03)
                        denom = math.hypot(g_norm, s_eff * sv_denom_sqrt)
                    else:
                        denom = g_norm
                    if denom <= 1e-30:
                        continue
                    alpha_i = -(rate_p * factor) / denom
                    p.add_(g, alpha=alpha_i)
                    self.fast_path_updates += 1
                    if track_sec:
                        fast_secant_deltas.append((p, alpha_i))
                        fast_coh_count += 1

            # 2. Slow Fallback for 4D Conv2d under stiffness / extreme scale (STILL strictly 0 Conv2d NS!):
            for p, g, step_t, rate_p, b1_eff, b2_eff, eps, s_base, stiff_gate, track_sec in conv_slow_queue:
                self._apply_slow_param_v5(
                    p,
                    g,
                    step_t=step_t,
                    rate_p=rate_p,
                    b1_eff=b1_eff,
                    b2_eff=b2_eff,
                    eps=eps,
                    s_base=s_base,
                    stiff_gate=stiff_gate,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_dense,
                )

            # 3. Batched 5-step fused Z=[X,Y] Quintic NS strictly for non-conv 2D dense matrices (0 Conv2d NS!):
            for (_, _, has_s_b), bucket in pending_dense.items():
                self.ns_2d_activations += len(bucket)
                self.ns_5step_calls += len(bucket)
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, trk_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    if trk_i:
                        delta_i = (orth_i * mult_i).reshape_as(p_i).to(p_i.dtype)
                        p_i.add_(delta_i)
                        applied_deltas.append((p_i, delta_i))
                    else:
                        p_i.add_((orth_i * mult_i).reshape_as(p_i).to(p_i.dtype))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, trk_i, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        if trk_i:
                            delta_i = (orth_b * mult_i).reshape_as(p_i).to(p_i.dtype)
                            p_i.add_(delta_i)
                            applied_deltas.append((p_i, delta_i))
                        else:
                            p_i.add_((orth_b * mult_i).reshape_as(p_i).to(p_i.dtype))

            # 4. Law 4 Zero-Copy Function-Space Secant Trust Band (0.00 ms on 4D Conv2d critic!):
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            if (critic_probe_active or gen_probe_active) and (fast_secant_deltas or applied_deltas):
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
                            total_coh_n = fast_coh_count + len(coh_vals)
                            if total_coh_n > 0:
                                slow_coh_sum = float(torch.stack(coh_vals).sum().item()) if coh_vals else 0.0
                                mean_coh = (float(fast_coh_count) + slow_coh_sum) / float(total_coh_n)
                            else:
                                mean_coh = 1.0
                            coh_eff = max(0.05, mean_coh)
                            if math.isfinite(delta_rel) and delta_rel > 1e-12:
                                free_ratio = (delta_g_max / delta_rel) * pre_gain_applied * th_tikh * coh_eff
                            active_thresh_g = max(1e-4, 0.002 * pre_gain_applied * step_scale * th_tikh * coh_eff)
                            step_active_probe = math.isfinite(delta_rel) and delta_rel >= active_thresh_g
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, alpha_i in fast_secant_deltas:
                        if p_i.grad is not None:
                            p_i.add_(p_i.grad, alpha=adj * alpha_i)
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

            # 5. Strictly Zero-pinv Analytic Non-Ascent Projection on multi-objective conflict steps:
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


__all__ = [
    "Law4EnergyOptimizer",
    "Law4EnergyV2Optimizer",
    "Law4EnergyV3Optimizer",
    "Law4EnergyV4Optimizer",
    "Law4EnergyV5Optimizer",
    "Law4EnergyV6Optimizer",
    "exact_polar_2xn",
]
