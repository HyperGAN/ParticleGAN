"""Unified-v14 Energy-Optimal 4-Law Equilibrated Optimizer (`unified_v14_equil`).

Unifies ``Law4EnergyV11Optimizer`` (asymmetric 0-Conv-NS / 0-neural-G-Critic-NS
Frobenius-MP + generator-free 2D Critic NS restoration + 0 critic secant probes +
0 B/param non-stiff state + 0 ``new_tensor`` H2D copies + ``0.50 * peak_rms``
degree-0 Tikhonov smoothing) with:
1. **PR #612 Deep-Residual Coupling & Closed-Loop Period-2 Flip-Bifurcation Control:**
   - Pure monotonic overshoot ``current_stiff_gate() = smoothstep01((0.60 - ema_fg) / 0.30)``
     with ``coh`` unmasking and preserved prior particle smoothing ``s_prior_base = smoothing``.
   - Zero-B/param (268 B total), prime-67 CountSketch closed-loop Period-2
     Flip-Bifurcation controller: activates ONLY when ``ema_fg < 0.60 and s_therm >= 0.5``
     (``0.000 ms`` on DCGAN and healthy networks; ``0.000`` false triggers on deep ResBN
     where 0-NS Frob-MP critic keeps ``ema_fg == 1.0``). Uses symmetric ``0.90 / 0.10``
     EMA and constant ``flip_thresh = 0.45`` deadband to drive period-2 flipping networks
     smoothly down to equilibrium (solving ``grid100_res16_tanh`` while preserving
     ``grid100_mlp16`` deterministically).
   - Natural 0-NS Frobenius-MP Critic Spectral Control: 0-NS ``sqrt(m)`` Frobenius-MP
     inherently bounds deep ResBN critic singular values (``max_sv ~ 2.2-2.4``) and keeps
     deep ResBN ``ema_fg == 1.0`` without Schatten-6 decay or critic-throttling ``d_couple``
     (eliminating extra critic GEMMs and preserving DCGAN speed/energy).
2. **Zero-Sync V4 Tensor Opt-In (``_SUPPORTS_ZEROSYNC_V4_TENSORS = True``):**
   - Zero-sync ``sort + is_first`` sampled-row deduplication for particle priors.
   - GPU-resident ``_origin_norm_t``, ``_v4_g_scale``, and ``_v4_wakeup_boost`` packed
     into the single existing ``torch._foreach_norm`` / secant-probe sync (zero extra
     CUDA synchronizations) alongside pending ``UnifiedV11StepRecord`` EMA tensors.
3. **Zero-Hyperparameter Role-Level Conditional-Fiber Triangle-Inequality 2-Obj Radius:**
   - On cooled conditional-fiber 2-objective ``direction_blend`` steps
     (``is_cond_fiber`` and ``self._protected.shape[0] == 2``), compensates each shared
     role's unit-ball gradient-cancellation shrinkage by
     ``min(2.0, (||g0|| + ||g1||) / ||g0 + g1||)`` (with ``p_dec = 0.9995`` tracking
     decaying multi-timestep fiber gradients). Strictly symmetric in ``(L_0, L_1)``,
     degree-0 scale-invariant, group-split invariant, and identically ``1.0`` (with
     non-decaying ``p_dec = 1.0``) on all 1-objective tasks, unconditional tasks, and
     single-loss roles (e.g., encoder ``E``).
"""
from __future__ import annotations

from copy import copy, deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

_FLIP_SKETCH_DIM = 67  # Prime -> coprime to all power-of-2 layer widths/strides (268 B total, 0 B/param)


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


class UnifiedV14Optimizer(UnifiedV11EquilibratedOptimizer):
    """Energy-optimal, zero-sync, deep-residual-coupled 4-Law optimizer (`unified_v14_equil`)."""

    family = "unified_v14_equil"
    recipe_optimizer_family = "unified_v14_equil"
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
        self.family = "unified_v14_equil"
        self.recipe_optimizer_family = "unified_v14_equil"
        self.ns_2d_activations: int = 0
        self.ns_5step_calls: int = 0
        self.fast_path_updates: int = 0
        self.critic_ns_calls: int = 0
        self._generator_free: bool = bool(generator_free)
        self._ema_flip_cos: float = 0.0
        self._prev_flip_sketch: Optional[torch.Tensor] = None
        self._dense_metrics_cache: Dict[int, Tuple[int, int, int, float, float]] = {}
        self._conv_metrics_cache: Dict[Tuple[int, Tuple[bool, int]], Tuple[float, float]] = {}
        for grp in self.param_groups:
            if grp["algorithm"] == "unified_v11_equil":
                grp["algorithm"] = self.family

    def current_stiff_gate(self) -> float:
        """Pure monotonic stiffness gate (0.0 at ema_fg >= 0.60, 1.0 at ema_fg <= 0.30)."""
        return _smoothstep01((0.60 - float(self._ema_func_gain)) / 0.30)

    def _update_period2_flip_feedback(self, *, s_therm: float) -> None:
        """Zero-B/param (268 B total) prime-67 CountSketch closed-loop Period-2 Flip-Bifurcation controller.

        Runs strictly when ``ema_fg < 0.60 and s_therm >= 0.5`` (0.000 ms on DCGAN and healthy
        networks). Sustained step-to-step gradient sign reversal (``ema_flip_cos < -0.45`` while
        ``ema_fg < 0.55``) gently lowers ``_ema_func_gain`` until the period-2 limit cycle ceases.
        """
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
        """Role-level $K=2$ Triangle-Inequality ball-radius compensation on cooled conditional-fiber steps.

        Active strictly when ``is_cond_fiber`` and ``self._protected.shape[0] == 2``: accumulates
        ``(||g0||^2, ||g1||^2, <g0, g1>)`` per parameter ``role`` across shared parameters via ONE
        batched reduction sync, returning ``min(2.0, (||g0|| + ||g1||) / ||g0 + g1||)`` per shared
        role and ``1.0`` for single-loss roles (e.g., encoder ``E``). Strictly symmetric in
        ``(L_0, L_1)``, degree-0 scale-invariant, and group-split invariant.
        """
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
        eps: float,
        s_base: float,
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
    ) -> None:
        """Zero-sync device-tensor 1e-38-safe / stiff 2D NS path (strictly 0 Conv2d & 0 neural-G Critic NS)."""
        st = self.state[p]
        gf = grad.float()
        finite = torch.isfinite(gf).all()
        gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
        g_max = gf_safe.abs().amax()
        g_active = finite & (g_max > 1e-38)

        is_conv = p.ndim == 4
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
        cur_sv_rms_t = (scaled_norm * safe_scale) / sv_denom_sqrt
        if "peak_rms" not in st:
            init_peak = float(st.get("peak_rms_val", 0.0))
            st["peak_rms"] = gf_safe.new_full((), init_peak)
        peak_prev_t = st["peak_rms"]
        cur_valid = finite & torch.isfinite(cur_sv_rms_t)
        peak_new_t = torch.maximum(peak_prev_t * p_dec, torch.where(cur_valid, cur_sv_rms_t, peak_prev_t.new_zeros(())))
        st["peak_rms"].copy_(torch.where(torch.isfinite(peak_new_t), peak_new_t, peak_prev_t))

        has_smooth = s_base > 0.0
        if has_smooth:
            s_eff_t = torch.minimum(gf_safe.new_full((), s_base), peak_new_t * 0.50)
        else:
            s_eff_t = gf_safe.new_zeros(())

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

            if not rate_positive:
                return
            if track_secant:
                coh_vals.append(coh)

            # Stiff 2D dense matrices on GENERATOR ONLY (strictly 0 Conv2d NS & 0 neural-G Critic NS!):
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
            if has_smooth:
                denom_m = torch.hypot(m_norm, s_scaled).clamp_min(1e-30)
            else:
                denom_m = (m_norm + (eps if p.ndim == 1 else 0.0)).clamp_min(1e-30)
            unit_m = torch.where(g_active & (m_norm > 1e-30), m_hat / denom_m, torch.zeros_like(m_hat))
            delta_m = (unit_m * ((-rate_p * factor) * coh)).to(p.dtype)
            p.add_(delta_m)
            if track_secant:
                applied_deltas.append((p, delta_m))
            return

        # Zero-state 1e-38-safe slow fallback (extreme scales <1e-18 / >1e18 / NaN / non-FP32):
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
            eps_scaled = (eps / safe_scale.clamp_min(1e-38)) if p.ndim == 1 else 0.0
            denom = (scaled_norm + eps_scaled).clamp_min(1e-30)
        unit_g = torch.where(g_active & (scaled_norm > 1e-30), g_unit_scaled / denom, torch.zeros_like(g_unit_scaled))
        delta_g = (unit_g * (-rate_p * factor)).to(p.dtype)
        p.add_(delta_g)
        if track_secant:
            applied_deltas.append((p, delta_g))

    @torch.no_grad()
    def _step_generator_free_critic(self, closure: Any = None) -> Any:
        """Zero-sync FP32 device-tensor 5-step 2D Critic NS + Schatten-6 strictly for generator-free tasks."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.grad.is_complex():
                    raise ValueError(f"{self.family} requires dense real gradients")

        try:
            rec = self._active_record()
            step_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0
            s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            p_dec = 0.9995 if s_therm >= 0.5 else 1.0
            pending_dense: Dict[
                Tuple[int, int, bool],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]],
            ] = {}

            for grp in self.param_groups:
                b1, b2 = grp["betas"]
                eps = float(grp["eps"])
                s_base = self.smoothing
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

                    is_conv = p.ndim == 4
                    r_d = c_d = m_dim = 1
                    if is_conv:
                        factor, sv_denom_sqrt = self._cached_conv_metrics(p)
                        cur_sv_rms = gf_safe.norm() / sv_denom_sqrt
                    elif p.ndim >= 2:
                        r_d, c_d, m_dim, factor, sv_denom_sqrt = self._cached_dense_metrics(p)
                        cur_sv_rms = gf_safe.norm() / sv_denom_sqrt
                    else:
                        factor, sv_denom_sqrt = 1.0, 1.0
                        cur_sv_rms = gf_safe.norm()

                    if "peak_rms" not in st:
                        st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
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
                    use_moments = (b1_eff > 0.0) or is_scalar_1d
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
                        coh = raw_coh.clamp(0.35, 1.0) if is_scalar_1d else torch.ones_like(m_norm)
                        raw_dir = (m_hat * sc) if has_smooth else m_hat
                        direc = torch.where(g_active, raw_dir, torch.zeros_like(m_hat))
                    else:
                        raw_dir = gf_safe if has_smooth else (gf_safe / g_max.clamp_min(1e-38))
                        direc = torch.where(g_active, raw_dir, torch.zeros_like(gf_safe))
                        coh = gf_safe.new_ones(())

                    if rate_p <= 0.0:
                        continue

                    if is_conv:
                        # Hard invariant: strictly 0 Conv2d NS even on generator-free tasks!
                        d_norm = direc.norm()
                        denom_c = (
                            torch.hypot(d_norm, s_eff_t * sv_denom_sqrt).clamp_min(1e-30)
                            if has_smooth
                            else d_norm.clamp_min(1e-30)
                        )
                        unit_c = torch.where(g_active & (d_norm > 1e-30), direc / denom_c, torch.zeros_like(direc))
                        p.add_((unit_c * ((-rate_p * factor) * coh)).to(p.dtype))
                        self.fast_path_updates += 1
                    elif p.ndim >= 2:
                        mat = direc.reshape(r_d, c_d)
                        step_mult = (-rate_p * factor) * coh
                        needs_t = r_d > c_d
                        mat_canon = mat.T if needs_t else mat
                        m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_smooth)
                        pending_dense.setdefault(m_key, []).append((p, mat_canon, step_mult, s_eff_t, needs_t))
                    else:
                        d_norm = direc.norm()
                        denom_1d = torch.hypot(d_norm, s_eff_t).clamp_min(eps) if has_smooth else (d_norm + eps)
                        unit_1d = torch.where(
                            g_active & (d_norm > 1e-30),
                            direc / denom_1d,
                            torch.zeros_like(direc),
                        )
                        p.add_((unit_1d * ((-rate_p) * coh)).to(p.dtype))
                        self.fast_path_updates += 1

            for (m_r, m_c, has_s_b), bucket in pending_dense.items():
                if min(m_r, m_c) >= 2:
                    self.ns_2d_activations += len(bucket)
                    self.ns_5step_calls += len(bucket)
                    self.critic_ns_calls += len(bucket)
                else:
                    self.fast_path_updates += len(bucket)
                if len(bucket) == 1:
                    p_i, mat_i, mult_i, s_i, needs_t = bucket[0]
                    orth_i = quintic_orthogonalize_2d(mat_i, steps=5, smoothing=(s_i if has_s_b else 0.0)).float()
                    if needs_t:
                        orth_i = orth_i.T
                    p_i.add_((orth_i * mult_i).reshape_as(p_i).to(p_i.dtype))
                else:
                    mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                    s_batch = torch.stack([b_item[3] for b_item in bucket], dim=0) if has_s_b else 0.0
                    orths = quintic_orthogonalize_batched(mats, steps=5, smoothing=s_batch).float()
                    for idx_b, (p_i, _, mult_i, _, needs_t) in enumerate(bucket):
                        orth_b = orths[idx_b].T if needs_t else orths[idx_b]
                        p_i.add_((orth_b * mult_i).reshape_as(p_i).to(p_i.dtype))

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

            origin_damping = 1.0
            origin_damping_t: Optional[torch.Tensor] = None
            if self._origin_norm_t is not None:
                origin_damping_t = (self._origin_norm_t / 0.05).clamp_max(1.0)
            elif self._origin_norm is not None and self._origin_norm < 0.05:
                origin_damping = min(1.0, self._origin_norm / 0.05)

            if not is_critic_opt:
                self._update_period2_flip_feedback(s_therm=s_therm)

            ema_fg = float(self._ema_func_gain)
            stiff_gate_raw = self.current_stiff_gate()
            pre_gain_base = 1.0 if ema_fg >= 0.75 else ema_fg

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

            # Zero-hyperparameter role-level K=2 Triangle-Inequality compensation on cooled cond-fiber steps:
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

            # Fast-path queues (batched into ONE torch._foreach_norm + ONE packed sync with v4 tensor scalars):
            fast_grads: List[torch.Tensor] = []
            # (p, grad, step_t, lr_no_v4, is_gen_v4, apply_od, b2_eff, eps, s_base, factor, sv_denom_sqrt,
            #  stiff_gate, track_secant, is_critic_role, aspect_factor, r_d, c_d, m_dim)
            fast_meta: List[Tuple[Any, ...]] = []
            dense2d_grads: List[torch.Tensor] = []
            # (p, grad, step_t, lr_no_v4, is_gen_v4, apply_od, b2_eff, eps, s_base, factor, sv_denom_sqrt, track_secant)
            dense2d_meta: List[Tuple[Any, ...]] = []
            # (p, grad, step_t, lr_no_v4, is_gen_v4, apply_od, b1_eff, b2_eff, eps, s_base, stiff_gate,
            #  is_critic_role, track_secant)
            slow_queue: List[Tuple[Any, ...]] = []

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
                        if base_lr_prior <= 0.0:
                            continue
                        r_scales = self._row_scales.get(p)
                        sampled_req = bool(grp.get("sampled_rows_required", True))

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
                            g_max_r = gr_safe.abs().amax(dim=-1, keepdim=True)
                            g_ok_r = g_max_r > 1e-38
                            g_sc_r = torch.where(g_ok_r, g_max_r, torch.ones_like(g_max_r))
                            g_norm = (gr_safe / g_sc_r).norm(dim=-1, keepdim=True) * torch.where(
                                g_ok_r, g_max_r, torch.zeros_like(g_max_r)
                            )
                            if b1_eff > 0.0:
                                d_max_r = dir_r.abs().amax(dim=-1, keepdim=True)
                                d_ok_r = d_max_r > 1e-38
                                d_sc_r = torch.where(d_ok_r, d_max_r, torch.ones_like(d_max_r))
                                d_norm = (dir_r / d_sc_r).norm(dim=-1, keepdim=True) * torch.where(
                                    d_ok_r, d_max_r, torch.zeros_like(d_max_r)
                                )
                            else:
                                d_norm = g_norm
                            active_r = is_first.unsqueeze(1) & fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)

                            d_u = torch.where(is_first.unsqueeze(1), d_norm, torch.zeros_like(d_norm))
                            d_u_max = d_u.amax()
                            d_u_ok = torch.isfinite(d_u_max) & (d_u_max > 1e-38)
                            safe_d_u = torch.where(d_u_ok, d_u_max, d_u_max.new_ones(()))
                            cur_row_rms = torch.where(
                                d_u_ok,
                                ((d_u / safe_d_u).square().sum() / n_unique_t.clamp_min(1).float()).sqrt() * safe_d_u,
                                d_u_max.new_zeros(()),
                            )
                            if "peak_rms" not in st:
                                st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
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
                                s_p_row = torch.minimum(s_p_row, peak_r * 0.30)
                                denom_r = torch.hypot(d_norm, s_p_row).clamp_min(1e-36)
                            else:
                                denom_r = d_norm.clamp_min(1e-36)
                            unit_r = torch.where(active_r, dir_r / denom_r, torch.zeros_like(dir_r))
                            p.index_add_(0, sorted_r, unit_r.to(p.dtype), alpha=-base_lr_prior)
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
                        g_max_r = gr_safe.abs().amax(dim=-1, keepdim=True)
                        g_ok_r = g_max_r > 1e-38
                        g_sc_r = torch.where(g_ok_r, g_max_r, torch.ones_like(g_max_r))
                        g_norm = (gr_safe / g_sc_r).norm(dim=-1, keepdim=True) * torch.where(
                            g_ok_r, g_max_r, torch.zeros_like(g_max_r)
                        )
                        if b1_eff > 0.0:
                            d_max_r = dir_r.abs().amax(dim=-1, keepdim=True)
                            d_ok_r = d_max_r > 1e-38
                            d_sc_r = torch.where(d_ok_r, d_max_r, torch.ones_like(d_max_r))
                            d_norm = (dir_r / d_sc_r).norm(dim=-1, keepdim=True) * torch.where(
                                d_ok_r, d_max_r, torch.zeros_like(d_max_r)
                            )
                        else:
                            d_norm = g_norm
                        active_r = fin_row & (g_norm > 1e-36) & (d_norm > 1e-36)

                        d_u_max = d_norm.amax()
                        d_u_ok = torch.isfinite(d_u_max) & (d_u_max > 1e-38)
                        safe_d_u = torch.where(d_u_ok, d_u_max, d_u_max.new_ones(()))
                        cur_row_rms = torch.where(
                            d_u_ok,
                            (d_norm / safe_d_u).square().mean().sqrt() * safe_d_u,
                            d_u_max.new_zeros(()),
                        )
                        if "peak_rms" not in st:
                            st["peak_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        peak_r = torch.maximum(
                            st["peak_rms"] * p_dec,
                            torch.where(torch.isfinite(cur_row_rms), cur_row_rms, st["peak_rms"].new_zeros(())),
                        )
                        st["peak_rms"].copy_(torch.where(torch.isfinite(peak_r), peak_r, st["peak_rms"]))

                        if s_prior_base > 0.0:
                            s_cap = s_prior_base * (128.0 / float(len(rows))) if len(rows) > 128 else s_prior_base
                            s_p_row = torch.minimum(d_norm.new_full((), s_cap), peak_r * 0.30)
                            denom_r = torch.hypot(d_norm, s_p_row).clamp_min(1e-36)
                        else:
                            denom_r = d_norm.clamp_min(1e-36)
                        unit_r = torch.where(active_r, dir_r / denom_r, torch.zeros_like(dir_r))
                        if r_scales is not None and len(r_scales) == len(rows):
                            unit_r = unit_r * r_scales.unsqueeze(1)
                        p.index_add_(0, rows, unit_r.to(p.dtype), alpha=-base_lr_prior)
                        continue

                    pyth_m = pyth_param_mults.get(id(p), 1.0)
                    lr_p_no_v4 = lr_no_v4 * pyth_m
                    apply_od = role == "generator" and step_t > 80
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
                            factor, sv_denom_sqrt = self._cached_conv_metrics(p)
                            fast_grads.append(grad)
                            fast_meta.append(
                                (
                                    p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff, eps,
                                    s_base, factor, sv_denom_sqrt, stiff_gate, track_secant,
                                    is_critic_role, factor, 1, 1, 1,
                                )
                            )
                        elif p.ndim >= 2:
                            r_d, c_d, m_dim, aspect_factor, sv_denom_sqrt = self._cached_dense_metrics(p)
                            if m_dim == 1:
                                fast_grads.append(grad)
                                fast_meta.append(
                                    (
                                        p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff, eps,
                                        s_base, aspect_factor, sv_denom_sqrt, stiff_gate, track_secant,
                                        is_critic_role, aspect_factor, r_d, c_d, m_dim,
                                    )
                                )
                            elif is_critic_role:
                                # Law 4.4 Asymmetric 2D Spectral Effort: strictly 0 NS on neural-G critic D!
                                factor_d = aspect_factor * math.sqrt(float(m_dim))
                                fast_grads.append(grad)
                                fast_meta.append(
                                    (
                                        p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff, eps,
                                        s_base, factor_d, sv_denom_sqrt, stiff_gate, track_secant,
                                        True, aspect_factor, r_d, c_d, m_dim,
                                    )
                                )
                            else:
                                dense2d_grads.append(grad)
                                dense2d_meta.append(
                                    (
                                        p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff, eps,
                                        s_base, aspect_factor, sv_denom_sqrt, track_secant,
                                    )
                                )
                        else:
                            fast_grads.append(grad)
                            fast_meta.append(
                                (
                                    p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff, eps,
                                    s_base, 1.0, 1.0, stiff_gate, track_secant,
                                    is_critic_role, 1.0, 1, 1, 1,
                                )
                            )
                    else:
                        slow_queue.append(
                            (
                                p, grad, step_t, lr_p_no_v4, is_gen_v4, apply_od, b1_eff, b2_eff, eps,
                                s_base, stiff_gate, is_critic_role, track_secant,
                            )
                        )

            # 1. Single Batched Foreach-Norm + Zero-Extra-Sync V4 Scalar Packing:
            all_norm_grads = fast_grads + dense2d_grads
            n_fast = len(fast_grads)
            all_norm_vals: List[float] = []
            v4_g_scale_f: Optional[float] = None if isinstance(self._v4_g_scale, torch.Tensor) else float(self._v4_g_scale)
            v4_pre_g_f: Optional[float] = None if pre_g_tensor is not None else pre_g_float
            od_f: Optional[float] = None if origin_damping_t is not None else origin_damping

            if all_norm_grads:
                norms = torch._foreach_norm(all_norm_grads)
                ref_dev = all_norm_grads[0].device
                extra_t: List[torch.Tensor] = []
                idx_gs = idx_pg = idx_od = -1
                if isinstance(self._v4_g_scale, torch.Tensor):
                    idx_gs = len(norms) + len(extra_t)
                    extra_t.append(self._v4_g_scale.to(device=ref_dev, dtype=torch.float32))
                if pre_g_tensor is not None:
                    idx_pg = len(norms) + len(extra_t)
                    extra_t.append(pre_g_tensor.to(device=ref_dev, dtype=torch.float32))
                if origin_damping_t is not None:
                    idx_od = len(norms) + len(extra_t)
                    extra_t.append(origin_damping_t.to(device=ref_dev, dtype=torch.float32))
                packed = torch.stack((*norms, *extra_t)).tolist()
                all_norm_vals = packed[: len(norms)]
                if idx_gs >= 0:
                    v4_g_scale_f = float(packed[idx_gs])
                if idx_pg >= 0:
                    v4_pre_g_f = float(packed[idx_pg])
                    pre_gain_applied = v4_pre_g_f
                if idx_od >= 0:
                    od_f = float(packed[idx_od])

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

            # 1a. Zero-State, Zero-Copy Foreach Frobenius-MP Fast Path (4D Conv2d, Neural-G Critic D, 1D, m=1):
            for idx_f, (
                p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff, eps,
                s_base, factor, sv_denom_sqrt, stiff_gate, track_sec,
                is_critic_role, aspect_factor, r_d, c_d, m_dim,
            ) in enumerate(fast_meta):
                g_norm = all_norm_vals[idx_f]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append(
                        (
                            p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, 0.0, b2_eff, eps,
                            s_base, stiff_gate, is_critic_role, track_sec,
                        )
                    )
                    continue
                rate_p = lr_p_no_v4 * (gen_v4_mult_f if is_gen_v4 else 1.0) * (od_f if apply_od else 1.0)
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                if "peak_rms" in st:
                    peak_prev = max(peak_prev, float(st["peak_rms"].item()))
                    del st["peak_rms"]
                peak_new = max(peak_prev * p_dec, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                if s_base > 0.0:
                    s_eff = min(s_base, peak_new * 0.50)
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

            # 1b. Non-stiff Generator 2D Dense Matrices (min(r, c) >= 2 -> batched 5-step Quintic NS):
            for idx_d, (
                p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b2_eff, eps,
                s_base, factor, sv_denom_sqrt, track_sec,
            ) in enumerate(dense2d_meta):
                g_norm = all_norm_vals[n_fast + idx_d]
                if not math.isfinite(g_norm) or g_norm < 1e-18 or g_norm > 1e18:
                    slow_queue.append(
                        (
                            p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, 0.0, b2_eff, eps,
                            s_base, 0.0, False, track_sec,
                        )
                    )
                    continue
                rate_p = lr_p_no_v4 * (gen_v4_mult_f if is_gen_v4 else 1.0) * (od_f if apply_od else 1.0)
                st = self.state[p]
                cur_sv = g_norm / sv_denom_sqrt
                peak_prev = float(st.get("peak_rms_val", 0.0))
                if "peak_rms" in st:
                    peak_prev = max(peak_prev, float(st["peak_rms"].item()))
                    del st["peak_rms"]
                peak_new = max(peak_prev * p_dec, cur_sv)
                st["peak_rms_val"] = peak_new
                if rate_p <= 0.0:
                    continue
                has_s = s_base > 0.0
                s_eff = min(s_base, peak_new * 0.50) if has_s else 0.0
                mat = g.reshape(p.shape[0], -1)
                r_d, c_d = int(mat.shape[0]), int(mat.shape[1])
                needs_t = r_d > c_d
                mat_canon = mat.T if needs_t else mat
                if track_sec:
                    fast_coh_count += 1
                m_key = (int(mat_canon.shape[0]), int(mat_canon.shape[1]), has_s)
                pending_dense.setdefault(m_key, []).append(
                    (p, mat_canon, g.new_full((), -rate_p * factor), g.new_full((), s_eff), track_sec, needs_t)
                )

            # 2. Zero-Sync Slow Path (stiff 2D NS on G, scalar 1D, extreme scales <1e-18 / >1e18 / NaN / non-FP32):
            for (
                p, g, step_t, lr_p_no_v4, is_gen_v4, apply_od, b1_eff, b2_eff, eps,
                s_base, stiff_gate, is_critic_role, track_sec,
            ) in slow_queue:
                if is_gen_v4 and gen_v4_mult_f is None and gen_v4_mult_t is not None:
                    rate_val: float | torch.Tensor = lr_p_no_v4 * gen_v4_mult_t
                    if apply_od:
                        rate_val = rate_val * (origin_damping_t if origin_damping_t is not None else origin_damping)
                    rate_pos = lr_p_no_v4 > 0.0
                else:
                    base_r = lr_p_no_v4 * (gen_v4_mult_f if is_gen_v4 else 1.0)
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
                    eps=eps,
                    s_base=s_base,
                    s_therm=s_therm,
                    p_dec=p_dec,
                    stiff_gate=stiff_gate,
                    is_critic_role=is_critic_role,
                    track_secant=track_sec,
                    coh_vals=coh_vals,
                    applied_deltas=applied_deltas,
                    pending_dense=pending_dense,
                )

            # 3. Batched 5-step fused Z=[X,Y] Quintic NS strictly for Generator 2D dense matrices:
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

            # 4. Law 4 Generator-Only Function-Space Secant Trust Band (0 critic secant probes!) + PR #612 coupling:
            alpha_gain = 1.0
            probe_ran = False
            free_ratio = 1.0
            step_active_probe = False
            pre_gain_f = 1.0
            if gen_probe_active and (fast_secant_deltas or applied_deltas):
                probe_cache = getattr(rec, "_g_probe_cache", None)
                rec._g_probe_cache = None
                if probe_cache is not None:
                    g_mod, p_args, p_kwargs, y_pre = probe_cache
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
                        coh_eff = max(0.05, mean_coh)
                        if math.isfinite(delta_rel) and delta_rel > 1e-12:
                            free_ratio = (delta_g_max / delta_rel) * pre_gain_f * coh_eff
                        step_active_probe = math.isfinite(delta_rel) and delta_rel >= 0.012
                if alpha_gain != 1.0:
                    adj = alpha_gain - 1.0
                    for p_i, alpha_i in fast_secant_deltas:
                        if p_i.grad is not None:
                            p_i.add_(p_i.grad, alpha=adj * alpha_i)
                    for p_i, d_i in applied_deltas:
                        p_i.add_(d_i, alpha=adj)
                if probe_ran:
                    if alpha_gain < 1.0:
                        target_fg = max(0.05, min(1.0, pre_gain_f * alpha_gain))
                        self._ema_func_gain = float(0.70 * self._ema_func_gain + 0.30 * target_fg)
                    elif step_active_probe and free_ratio > 1.5:
                        target_fg = max(0.05, min(1.0, free_ratio))
                        if target_fg > self._ema_func_gain:
                            self._ema_func_gain = float(0.999 * self._ema_func_gain + 0.001 * target_fg)
                if rec is not None:
                    g_stiff_post = self.current_stiff_gate()
                    rec._g_func_gain = float((1.0 - g_stiff_post) * self._ema_func_gain + g_stiff_post)
            elif rec is not None and not is_critic_opt:
                rec._g_probe_cache = None

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
        meta: Dict[str, Any] = {
            "schema": 1,
            "family": self.family,
            "generator_free": bool(self._generator_free),
            "last_step_gain": float(self.last_step_gain),
            "ema_func_gain": float(self._ema_func_gain),
            "ema_flip_cos": float(self._ema_flip_cos),
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
        result["unified_v14_equil"] = meta
        if hasattr(self, "record") and self.record is not None:
            result["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return result

    def validate_state_dict(self, saved: Any) -> None:
        expected_keys = {"state", "param_groups", "unified_v14_equil"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError("invalid UnifiedV14Optimizer checkpoint schema")
        meta = saved["unified_v14_equil"]
        expected_meta_keys = {
            "schema",
            "family",
            "generator_free",
            "last_step_gain",
            "ema_func_gain",
            "ema_flip_cos",
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
            raise ValueError("incompatible UnifiedV14Optimizer metadata")
        gain = meta["last_step_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) <= 0.0:
            raise ValueError("invalid UnifiedV14Optimizer last_step_gain")
        ema_fg = meta["ema_func_gain"]
        if (
            isinstance(ema_fg, bool)
            or not isinstance(ema_fg, (int, float))
            or not math.isfinite(ema_fg)
            or not (0.0 < float(ema_fg) <= 1.0)
        ):
            raise ValueError("invalid UnifiedV14Optimizer ema_func_gain")
        ema_flip = meta["ema_flip_cos"]
        if (
            isinstance(ema_flip, bool)
            or not isinstance(ema_flip, (int, float))
            or not math.isfinite(ema_flip)
            or not (-1.0 <= float(ema_flip) <= 1.0)
        ):
            raise ValueError("invalid UnifiedV14Optimizer ema_flip_cos")
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
                raise ValueError("invalid UnifiedV14Optimizer prev_flip_sketch")
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
                raise ValueError("invalid UnifiedV14Optimizer direction_blend checkpoint")
            if cg["pending"] is not None:
                if not isinstance(cg["pending"], torch.Tensor) or not bool(torch.isfinite(cg["pending"]).all()):
                    raise ValueError("invalid pending direction_blend tensor")
        if not isinstance(saved["param_groups"], list) or len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError("UnifiedV14Optimizer group count mismatch")
        for cur_grp, sav_grp in zip(self.param_groups, saved["param_groups"]):
            if not isinstance(sav_grp, dict) or len(sav_grp.get("params", ())) != len(cur_grp["params"]):
                raise ValueError("UnifiedV14Optimizer group parameter mismatch")
            if sav_grp.get("role") != cur_grp["role"] or sav_grp.get("algorithm") != cur_grp["algorithm"]:
                raise ValueError("UnifiedV14Optimizer group role/algorithm mismatch")
        if not isinstance(saved["state"], dict):
            raise ValueError("invalid UnifiedV14Optimizer state map")
        for st_val in saved["state"].values():
            if not isinstance(st_val, dict):
                raise ValueError("invalid UnifiedV14Optimizer parameter state")
            step_v = st_val.get("step", 0)
            if type(step_v) is not int or step_v < 0:
                raise ValueError("invalid UnifiedV14Optimizer step counter")
            if "peak_rms_val" in st_val:
                pv = st_val["peak_rms_val"]
                if isinstance(pv, bool) or not isinstance(pv, (int, float)) or not math.isfinite(pv) or float(pv) < 0.0:
                    raise ValueError("invalid peak_rms_val in UnifiedV14Optimizer state")
            for t_key in ("m_unit", "m_scale", "v_rms", "v_scale", "m_prior", "peak_rms"):
                if t_key in st_val:
                    ten = st_val[t_key]
                    if not isinstance(ten, torch.Tensor) or not bool(torch.isfinite(ten).all()):
                        raise ValueError(f"non-finite or invalid tensor {t_key} in UnifiedV14Optimizer state")
                    if t_key == "peak_rms" and (ten.ndim != 0 or float(ten.item()) < 0.0):
                        raise ValueError("invalid peak_rms tensor in UnifiedV14Optimizer state")
        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid UnifiedV14Optimizer critic regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        meta = state_dict["unified_v14_equil"]
        clean = {k: v for k, v in state_dict.items() if k not in ("unified_v14_equil", "regularizer")}
        Optimizer.load_state_dict(self, clean)
        self.clear_sampled_rows()
        self.last_step_gain = float(meta["last_step_gain"])
        self._ema_func_gain = float(meta["ema_func_gain"])
        self._ema_flip_cos = float(meta["ema_flip_cos"])
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


__all__ = ["UnifiedV14Optimizer"]
