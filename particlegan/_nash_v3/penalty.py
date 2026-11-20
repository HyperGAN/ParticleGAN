"""Pure 1st-order fwAD & live-endpoint secant Born-Infeld critic penalty for T4NashV3.

Strictly 0 torch.autograd.grad, 0 SVD, 0 eigh, 0 pinv, and 0 Newton-Schulz calls:
1. Low-D continuous vectors & discrete simplexes: 1st-order forward-mode AD
   (``torch.autograd.forward_ad.dual_level`` with central-difference fallback for
   emulated MXFP8/NVFP4 ops) paired with intrinsic-scale e-means Sinkhorn chords.
2. High-res convolutional images: live-endpoint 1st-order logit reuse (0 extra
   endpoint forwards) + single BF16 micro-secant probe + free chord & cyclic secants.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .._empf.critic import compute_empf_step_scale, compute_thermal_scale
from .._empf.prototypes import _PHI1, _PHI2, update_emeans_prototypes
from .._empf.transport import (
    conditional_fiber_residual,
    conditional_joint_discrepancy,
    effective_sinkhorn_eps,
    unconditioned_chamfer_residual,
)
from .._stats import pairwise_sq_dist
from .._t2_critic.critic import _cond_signature, _tensor_identity_key
from .._t2_critic.penalty import (
    _eval_exact_lowd_forward_grad_sq,
    _eval_forward_tangent_sq,
    _format_logits_2d,
    _logit_diff_rms,
    _prepare_spatial_tangent_plan,
    _separable_image_unit_perturbation,
)
from ..grad_regularizers import CriticStepRecord
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output, _no_anchor
from .optim import NashV3CriticAdam
from .record import FROZEN_THERMAL_SCALE, NashV3StepRecord, _attach_nash_v3_critic_hook
from .transport import intrinsic_sinkhorn_coupling


class _NashV3PenaltyKernel:
    """Clock-free pure 1st-order T4NashV3 critic penalty kernel (0 autograd.grad, 0 SVD)."""

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        lambda_floor: float = 0.01,
        lambda_cap: float = 1.0,
        ebi_beta: float = 8.0,
        gamma_gauge: float = 0.060,
        chord_blend: float = 0.08,
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        max_prototypes: int = 256,
        alpha_base: float = 0.08,
        hit_exponent: float = 0.45,
        revival_tau: float = 0.05,
        tau_cooling: float = 0.020,
        s_min: float = 1e-5,
        thermal_decay: float = 0.92,
        secant_eps: float = 1e-5,
        input_domain: str = "auto",
        record: CriticStepRecord | None = None,
    ) -> None:
        for name, val, pos in (
            ("coeff", coeff, False),
            ("kappa", kappa, False),
            ("lambda_floor", lambda_floor, False),
            ("lambda_cap", lambda_cap, False),
            ("ebi_beta", ebi_beta, True),
            ("gamma_gauge", gamma_gauge, False),
            ("chord_blend", chord_blend, False),
            ("sinkhorn_eps", sinkhorn_eps, True),
            ("alpha_base", alpha_base, True),
            ("hit_exponent", hit_exponent, False),
            ("revival_tau", revival_tau, True),
            ("tau_cooling", tau_cooling, True),
            ("s_min", s_min, True),
            ("thermal_decay", thermal_decay, True),
            ("secant_eps", secant_eps, True),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or (val <= 0 if pos else val < 0)
            ):
                raise ValueError(f"{name} must be finite and {'positive' if pos else 'nonnegative'}")
        if chord_blend > 1.0:
            raise ValueError("chord_blend must be in [0, 1]")
        if revival_tau >= 1.0:
            raise ValueError("revival_tau must be in (0, 1)")
        if s_min > 1.0:
            raise ValueError("s_min must be <= 1.0")
        if thermal_decay >= 1.0:
            raise ValueError("thermal_decay must be < 1.0")
        if type(lazy_k) is not int or lazy_k < 1:
            raise ValueError("lazy_k must be a positive integer")
        if type(sinkhorn_iters) is not int or sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be a positive integer")
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an integer >= 2")
        if anchor_weight != 0.0:
            raise ValueError(
                "T4NashV3 uses a zero-mean logit gauge instead of an EMA critic anchor (require reg_anchor_weight=0)"
            )
        if not isinstance(input_domain, str) or input_domain not in ("auto", "continuous", "categorical"):
            raise ValueError("input_domain must be auto, continuous or categorical")

        self.arm = "t4_nash_v3"
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.anchor_weight = 0.0
        self.lambda_floor = float(lambda_floor)
        self.lambda_cap = float(lambda_cap)
        self.ebi_beta = float(ebi_beta)
        self.gamma_gauge = float(gamma_gauge)
        self.chord_blend = float(chord_blend)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.max_prototypes = int(max_prototypes)
        self.alpha_base = float(alpha_base)
        self.hit_exponent = float(hit_exponent)
        self.revival_tau = float(revival_tau)
        self.tau_cooling = float(tau_cooling)
        self.s_min = float(s_min)
        self.thermal_decay = float(thermal_decay)
        self.secant_eps = float(secant_eps)
        self.input_domain = input_domain
        self.record = NashV3StepRecord() if record is None else record

    def _observe(self, res: float, data_scale: float, *, ema_decay: float | None = None) -> None:
        decay = self.thermal_decay if ema_decay is None else float(ema_decay)
        if hasattr(self.record, "observe_transport"):
            self.record.observe_transport(res, data_scale, ema_decay=decay)

    def _lookup_live_endpoint(
        self,
        x_target: torch.Tensor,
        cond_sig: Tuple[Any, ...],
        *,
        exclude_out: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        captures = getattr(self.record, "_live_captures", None)
        if not captures:
            return None
        key = _tensor_identity_key(x_target)
        if key is None:
            return None
        cur_step = int(getattr(self.record, "observed_steps", 0))
        for step_tag, src_key, c_sig, x_act, raw_out, alive in reversed(captures):
            if not alive or step_tag != cur_step:
                continue
            if exclude_out is not None and raw_out is exclude_out:
                continue
            if src_key == key and c_sig == cond_sig and raw_out.requires_grad and raw_out.grad_fn is not None:
                return x_act, raw_out
        return None

    def penalty(
        self,
        live,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        is_conditioned: bool = False,
        cond_tensors: Sequence[torch.Tensor] = (),
        cond_sig: Tuple[Any, ...] = (),
        allow_live_reuse: bool = True,
        can_concat_batches: bool = False,
        live_rep: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del kwargs
        if self.input_domain == "categorical" and (x_real.ndim != 2 or x_fake.ndim != 2):
            raise ValueError("categorical T4NashV3 inputs require native [batch, categories] tensors")
        if self.coeff == 0.0 or (self.lazy_k > 1 and step % self.lazy_k != 0):
            self.record._last_d_logits = None
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        if isinstance(live, nn.Module) and getattr(live, "_nash_v3_record", None) is not self.record:
            _attach_nash_v3_critic_hook(live, self.record)

        self.record.calls += 1
        b_r, b_f = x_real.shape[0], x_fake.shape[0]
        dim = max(1, x_real[0].numel())

        is_image_tensor = x_real.ndim >= 3 and self.input_domain != "continuous"
        is_discrete_simplex = self.input_domain == "categorical" or (
            self.input_domain == "auto" and x_real.ndim == 2 and dim >= 170
        )
        use_chords = not is_discrete_simplex and not is_image_tensor and x_real.ndim == 2 and dim <= 64

        target_matched: Optional[torch.Tensor] = None
        alpha: Optional[torch.Tensor] = None
        local_clump_gain: float = 1.0
        step_scale_val: float = 1.0
        phase = float(getattr(self.record, "chord_phase", 0.0))

        try:
            if use_chords and not is_conditioned:
                p_sum = getattr(self.record, "proto_sum", None)
                p_count = getattr(self.record, "proto_count", None)
                rev_phase = float(getattr(self.record, "revival_phase", 0.0))
                rev_count = int(getattr(self.record, "revivals", 0))
                prev_prototypes: torch.Tensor | None = None
                if (
                    isinstance(p_sum, torch.Tensor)
                    and isinstance(p_count, torch.Tensor)
                    and p_sum.ndim == 2
                    and p_count.ndim == 1
                    and p_sum.shape[0] == p_count.shape[0]
                    and p_sum.shape[0] > 0
                    and p_sum.shape[1] == dim
                ):
                    prev_prototypes = (p_sum / p_count.clamp_min(1e-8).unsqueeze(1)).detach()

                prototypes, proto_weights, new_sum, new_count, new_rev_phase, new_revivals = update_emeans_prototypes(
                    x_real,
                    p_sum,
                    p_count,
                    rev_phase,
                    rev_count,
                    max_prototypes=self.max_prototypes,
                    alpha_base=self.alpha_base,
                    hit_exponent=self.hit_exponent,
                    revival_tau=self.revival_tau,
                )
                if hasattr(self.record, "update_prototypes"):
                    self.record.update_prototypes(new_sum, new_count, new_rev_phase, new_revivals)

                prev_res = float(self.record.ema_residual) if getattr(self.record, "initialized", False) else 1.0
                prev_scale = float(getattr(self.record, "_thermal_scale", 1.0))
                compressed_bank = bool(2 <= dim <= 4 and b_r >= max(1024, 3 * prototypes.shape[0]))
                eps_eff = effective_sinkhorn_eps(prev_res, self.sinkhorn_eps) if compressed_bank else self.sinkhorn_eps
                pi, _, res, data_scale = intrinsic_sinkhorn_coupling(
                    prototypes,
                    proto_weights,
                    x_fake,
                    x_real=x_real,
                    sinkhorn_eps=eps_eff,
                    sinkhorn_iters=self.sinkhorn_iters,
                )

                obs_res, obs_ds = torch.stack((res.detach(), data_scale.detach())).tolist()
                self._observe(float(obs_res), float(obs_ds))

                if 1 <= dim <= 4 and b_r >= 32 and b_f >= 32:
                    t_scale = compute_thermal_scale(
                        float(self.record.ema_residual),
                        b_r,
                        dim,
                        tau_cooling=self.tau_cooling,
                        s_min=self.s_min,
                    )
                    if dim == 1:
                        step_scale_val = compute_empf_step_scale(float(self.record.ema_residual), b_r, dim)
                else:
                    t_scale = 1.0

                if 2 <= dim <= 8 and b_r >= 64:
                    with torch.no_grad():
                        r_flat_clump = x_real.detach().flatten(1).to(dtype=torch.float32)
                        if b_r > 512:
                            slots_c = (torch.arange(512, device=x_real.device, dtype=torch.long) * b_r) // 512
                            r_flat_clump = r_flat_clump[slots_c]
                        n_c = int(r_flat_clump.shape[0])
                        d2_rr_clump = pairwise_sq_dist(r_flat_clump)
                        eye_clump = torch.eye(n_c, device=x_real.device, dtype=torch.bool)
                        d2_rr_masked_c = d2_rr_clump.masked_fill(eye_clump, float("inf"))
                        med_d_c = float(d2_rr_clump.masked_fill(eye_clump, 0.0).median().clamp_min(1e-12).sqrt().item())
                        k_nn_c = max(2, min(12, n_c // 32))
                        knn_d2_c = torch.topk(d2_rr_masked_c, k=k_nn_c, dim=1, largest=False).values[:, -1]
                        knn_med_c = float(knn_d2_c.median().clamp_min(1e-8).sqrt().item())
                        expected_homo_c = med_d_c * ((float(k_nn_c) / float(n_c)) ** (1.0 / float(dim)))
                        clump_ratio_c = max(1.0, min(8.0, expected_homo_c / max(knn_med_c, 1e-6)))
                        local_clump_gain = min(5.5, max(1.0, clump_ratio_c))

                if dim == 2 and 64 <= b_r <= 512 and b_f >= 64 and float(self.record.ema_residual) < 0.12:
                    with torch.no_grad():
                        r_flat_2d = x_real.detach().flatten(1)
                        f_flat_2d = x_fake.detach().flatten(1)
                        d2_rf = pairwise_sq_dist(r_flat_2d, f_flat_2d)
                        med_d2 = d2_rf.median().clamp_min(1e-12)
                        norm_d = (d2_rf / med_d2).sqrt()
                        f2r_q = float(norm_d.min(dim=0).values.quantile(0.95).item())
                        if prev_scale <= FROZEN_THERMAL_SCALE and prev_prototypes is not None:
                            p_prev_2d = prev_prototypes.to(device=r_flat_2d.device, dtype=r_flat_2d.dtype)
                            d2_rp = pairwise_sq_dist(r_flat_2d, p_prev_2d)
                            r2p_q = float(
                                (d2_rp / d2_rp.median().clamp_min(1e-12))
                                .sqrt()
                                .min(dim=1)
                                .values.quantile(0.95)
                                .item()
                            )
                            if max(r2p_q, f2r_q) < 0.055:
                                t_scale = self.s_min
                        else:
                            r2f_q = float(norm_d.min(dim=1).values.quantile(0.95).item())
                            thresh = 0.065 if prev_scale < 1.0 else 0.050
                            if max(r2f_q, f2r_q) < thresh:
                                t_scale = self.s_min
                elif compressed_bank and b_f >= 64:
                    with torch.no_grad():
                        r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
                        d2_rr = pairwise_sq_dist(r_flat)
                        eye_r = torch.eye(b_r, device=x_real.device, dtype=torch.bool)
                        d2_rr_masked = d2_rr.masked_fill(eye_r, float("inf"))
                        if t_scale < 0.15:
                            f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
                            d2_rf = pairwise_sq_dist(r_flat, f_flat)
                            r2r_nn = d2_rr_masked.min(dim=1).values.sqrt()
                            r2r_q90 = r2r_nn.quantile(0.90).clamp_min(1e-6)
                            r2r_q65 = r2r_nn.quantile(0.65).clamp_min(1e-6)
                            r2f_q90 = d2_rf.min(dim=1).values.sqrt().quantile(0.90)
                            f2r_q65 = d2_rf.min(dim=0).values.sqrt().quantile(0.65)
                            r2f_ratio = float((r2f_q90 / r2r_q90).item())
                            f2r_ratio = float((f2r_q65 / r2r_q65).item())
                            thresh = 1.25 if prev_scale < 0.005 else 1.12
                            if max(r2f_ratio, f2r_ratio) < thresh:
                                t_scale = self.s_min

                with torch.no_grad():
                    col_probs = (b_f * pi.t()).clamp_min(1e-12)
                    col_probs = col_probs / col_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    cdf = torch.cumsum(col_probs, dim=1)
                    j_idx = torch.arange(b_f, device=x_real.device, dtype=torch.float32)
                    q = torch.remainder((j_idx + 0.5) / max(1, b_f) + phase, 1.0).unsqueeze(1)
                    matched_idx = torch.searchsorted(cdf.float(), q).squeeze(1).clamp(0, prototypes.shape[0] - 1)
                    target_matched = prototypes[matched_idx].to(dtype=x_real.dtype).reshape((b_f,) + x_real.shape[1:])
                    u_w = torch.remainder((j_idx + 0.5) / max(1, b_f) + ((phase + _PHI2) % 1.0), 1.0).to(
                        dtype=x_real.dtype
                    )
                    alpha = u_w.reshape((b_f,) + (1,) * (x_real.ndim - 1))
            elif use_chords and is_conditioned:
                if cond_tensors and b_r == b_f and b_r >= 16:
                    u_signed, data_scale, _ = conditional_joint_discrepancy(x_real, x_fake, cond_tensors)
                    if hasattr(self.record, "observe_conditional_energy"):
                        self.record.observe_conditional_energy(u_signed, float(data_scale.item()))
                    else:
                        self._observe(math.sqrt(max(0.0, u_signed)), float(data_scale.item()))
                    cond_res = float(getattr(self.record, "ema_residual", 1.0))
                    u_c = max(0.0, min(1.0, (cond_res - 0.008) / (0.024 - 0.008)))
                    step_scale_val = float(0.85 + 0.15 * (u_c * u_c * (3.0 - 2.0 * u_c)))
                else:
                    res, data_scale = conditional_fiber_residual(x_real, x_fake)
                    self._observe(float(res.item()), float(data_scale.item()))
                t_scale = 1.0
                if b_r == b_f:
                    with torch.no_grad():
                        j_idx = torch.arange(b_f, device=x_real.device, dtype=x_real.dtype)
                        u_w = torch.remainder((j_idx + 0.5) / max(1, b_f) + phase, 1.0)
                        alpha = u_w.reshape((b_f,) + (1,) * (x_real.ndim - 1))
                        target_matched = x_real.detach()
                else:
                    use_chords = False
            elif is_discrete_simplex:
                if not is_conditioned and b_r >= 64 and b_f >= 64 and dim >= 170:
                    res, data_scale = unconditioned_chamfer_residual(x_real.detach(), x_fake.detach())
                    obs_res, obs_ds = torch.stack((res.detach(), data_scale.detach())).tolist()
                    self._observe(float(obs_res), float(obs_ds), ema_decay=0.96)
                    u_w = max(0.0, min(1.0, (float(self.record.ema_residual) - 0.064) / (0.084 - 0.064)))
                    smooth_w = u_w * u_w * (3.0 - 2.0 * u_w)
                    t_scale = float(0.0004 + 0.9996 * smooth_w)
                else:
                    t_scale = 1.0
            else:
                with torch.no_grad():
                    if is_image_tensor and x_real.ndim == 4 and (x_real.shape[2] > 16 or x_real.shape[3] > 16):
                        h_in, w_in = x_real.shape[2], x_real.shape[3]
                        r_obs = F.adaptive_avg_pool2d(x_real.detach().float(), 8)
                        f_obs = F.adaptive_avg_pool2d(x_fake.detach().float(), 8)
                        res, ds_pool = unconditioned_chamfer_residual(r_obs, f_obs)
                        spatial_gain = math.sqrt(float(h_in * w_in) / 64.0)
                        data_scale = ds_pool * spatial_gain
                    else:
                        res, data_scale = unconditioned_chamfer_residual(x_real.detach(), x_fake.detach())
                    obs_res, obs_ds = torch.stack((res.detach(), data_scale.detach())).tolist()
                self._observe(float(obs_res), float(obs_ds))
                t_scale = 1.0

            self.record.chord_phase = (phase + _PHI1) % 1.0
            self.record._thermal_scale = float(t_scale)
            self.record._step_scale = float(step_scale_val)

            scale_dim = 1.0 if (is_discrete_simplex or is_image_tensor or dim <= 4) else math.sqrt(dim)
            res_ratio = min(1.0, max(0.0, float(getattr(self.record, "ema_residual", 1.0))))
            if is_image_tensor:
                ref_scale = 4.899
                img_scale = min(1.0, max(0.15, float(getattr(self.record, "ema_data_scale", ref_scale)) / ref_scale))
                kappa_eff = (self.kappa / img_scale) * (0.6 + 1.10 * (1.0 - res_ratio))
            elif local_clump_gain > 1.0:
                settle_fine = min(1.0, max(0.0, 1.0 - res_ratio / 0.22))
                kappa_eff = self.kappa * (1.0 + (local_clump_gain - 1.0) * (settle_fine * settle_fine))
            else:
                kappa_eff = self.kappa

            reused_endpoints = False

            if is_image_tensor:
                hit_r = self._lookup_live_endpoint(x_real, cond_sig) if allow_live_reuse else None
                hit_f = (
                    self._lookup_live_endpoint(
                        x_fake,
                        cond_sig,
                        exclude_out=hit_r[1] if hit_r is not None else None,
                    )
                    if allow_live_reuse
                    else None
                )
                reused_endpoints = bool(hit_r is not None and hit_f is not None)

                self.record._in_penalty = True
                try:
                    if hit_r is not None:
                        z_r, d_r_raw = hit_r
                        z_r = z_r.to(dtype=x_real.dtype)
                    else:
                        self.record._last_probe_actual_input = None
                        x_r_det = x_real.detach()
                        d_r_raw = live(x_r_det)
                        act_r = getattr(self.record, "_last_probe_actual_input", None)
                        z_r = (
                            act_r.to(dtype=x_real.dtype)
                            if isinstance(act_r, torch.Tensor) and act_r.shape == x_r_det.shape
                            else x_r_det
                        )

                    if hit_f is not None:
                        z_f, d_f_raw = hit_f
                        z_f = z_f.to(dtype=x_fake.dtype)
                    else:
                        self.record._last_probe_actual_input = None
                        x_f_det = x_fake.detach()
                        d_f_raw = live(x_f_det)
                        act_f = getattr(self.record, "_last_probe_actual_input", None)
                        z_f = (
                            act_f.to(dtype=x_fake.dtype)
                            if isinstance(act_f, torch.Tensor) and act_f.shape == x_f_det.shape
                            else x_f_det
                        )

                    d_r_2d = _format_logits_2d(d_r_raw, b_r)
                    d_f_2d = _format_logits_2d(d_f_raw, b_f)

                    matched_idx_img = torch.arange(b_f, device=x_real.device, dtype=torch.long) % b_r
                    z_r_m = z_r[matched_idx_img].detach()
                    d_r_m = d_r_2d[matched_idx_img]

                    parity_mask = ((torch.arange(b_f, device=x_fake.device) + int(phase * 1024.0)) % 2) == 0
                    mask_nd = parity_mask.reshape((b_f,) + (1,) * (x_fake.ndim - 1))
                    z_base = torch.where(mask_nd, z_r_m, z_f.detach())
                    d_base = torch.where(parity_mask.unsqueeze(1), d_r_m, d_f_2d)

                    img_delta = 0.02
                    u_img = _separable_image_unit_perturbation(z_base, phase, salt=3)
                    x_probe = (z_base + img_delta * u_img).detach()

                    self.record._last_probe_actual_input = None
                    if is_conditioned and b_f != b_r:
                        d_p_2d = d_f_2d
                        z_p = z_f.detach()
                    else:
                        d_p_raw = live(x_probe)
                        act_p = getattr(self.record, "_last_probe_actual_input", None)
                        z_p = (
                            act_p.to(dtype=x_probe.dtype)
                            if isinstance(act_p, torch.Tensor) and act_p.shape == x_probe.shape
                            else x_probe
                        )
                        d_p_2d = _format_logits_2d(d_p_raw, b_f)
                finally:
                    self.record._in_penalty = False

                eps_sq = self.secant_eps * self.secant_eps
                step_sq_img = (z_p - z_base).flatten(1).float().square().sum(dim=1).add(eps_sq).to(dtype=d_f_2d.dtype)
                micro_gn_sq = float(dim) * (_logit_diff_rms(d_p_2d, d_base).square() / step_sq_img)

                dist_chord = (
                    (z_r_m - z_f.detach()).flatten(1).float().square().sum(dim=1).add(eps_sq).sqrt().to(dtype=d_f_2d.dtype)
                )
                s_chord = _logit_diff_rms(d_r_m, d_f_2d) / dist_chord

                s_candidates = [(micro_gn_sq + 1e-12).sqrt(), s_chord]
                if b_f >= 2:
                    zf_roll = torch.roll(z_f.detach(), shifts=1, dims=0)
                    df_roll = torch.roll(d_f_2d, shifts=1, dims=0)
                    dist_ff = (
                        (zf_roll - z_f.detach())
                        .flatten(1)
                        .float()
                        .square()
                        .sum(dim=1)
                        .add(eps_sq)
                        .sqrt()
                        .to(dtype=d_f_2d.dtype)
                    )
                    s_candidates.append(_logit_diff_rms(df_roll, d_f_2d) / dist_ff)
                if b_r == b_f and b_r >= 2:
                    zr_roll = torch.roll(z_r.detach(), shifts=1, dims=0)
                    dr_roll = torch.roll(d_r_2d, shifts=1, dims=0)
                    dist_rr = (
                        (zr_roll - z_r.detach())
                        .flatten(1)
                        .float()
                        .square()
                        .sum(dim=1)
                        .add(eps_sq)
                        .sqrt()
                        .to(dtype=d_r_2d.dtype)
                    )
                    s_candidates.append(_logit_diff_rms(dr_roll, d_r_2d) / dist_rr)

                norm_env = torch.stack(s_candidates, dim=0).amax(dim=0)
                settle_s = min(1.0, max(0.0, 1.0 - res_ratio / 0.16))
                b_frac = min(1.0, max(0.0, (float(b_r) - 64.0) / 64.0))
                cool_boost = 1.0 - float(t_scale)
                batch_gain = (2.0 + 3.0 * b_frac) + cool_boost * (4.5 + 12.5 * b_frac)
                eff_ebi_beta = self.ebi_beta + batch_gain * (settle_s * settle_s)

                cap_pen = self.lambda_cap * F.softplus(norm_env - kappa_eff, beta=eff_ebi_beta).square().mean()
                floor_pen = 0.5 * 0.0005 * (
                    (0.5 * micro_gn_sq.mean() + 0.5 * s_chord.square().mean()) / float(dim)
                )
                ebi_loss = floor_pen + cap_pen

                self.record._last_d_logits = (d_r_2d.reshape(-1).detach(), d_f_2d.reshape(-1).detach())
                mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                gauge_loss = 0.5 * 0.005 * mean_logit.square()
                norm_chord_stat = norm_env.detach().mean()
            else:
                rep_fn = live_rep if live_rep is not None else (lambda z_in, _m: live(z_in))
                if dim <= 4:
                    self.record._in_penalty = True
                    try:
                        d_r_2d, sq_real = _eval_exact_lowd_forward_grad_sq(
                            rep_fn, x_real, can_stack_basis=can_concat_batches, fd_delta=1e-3
                        )
                        d_f_2d, sq_fake = _eval_exact_lowd_forward_grad_sq(
                            rep_fn, x_fake, can_stack_basis=can_concat_batches, fd_delta=1e-3
                        )
                        sq_chord: Optional[torch.Tensor] = None
                        if use_chords and target_matched is not None and alpha is not None:
                            x_chord = ((1.0 - alpha) * x_fake.detach() + alpha * target_matched).detach()
                            u_c, recon_c_lowd = _prepare_spatial_tangent_plan(
                                x_chord, phase, salt=3, cond_tensors=cond_tensors
                            )
                            _, tau_c_list = _eval_forward_tangent_sq(
                                live, [x_chord], [u_c], can_concat=False, fd_delta=1e-3
                            )
                            sq_chord = recon_c_lowd(tau_c_list[0])
                    finally:
                        self.record._in_penalty = False
                else:
                    u_r, recon_r = _prepare_spatial_tangent_plan(
                        x_real.detach(), phase, salt=1, cond_tensors=cond_tensors
                    )
                    u_f, recon_f = _prepare_spatial_tangent_plan(
                        x_fake.detach(), phase, salt=2, cond_tensors=cond_tensors
                    )
                    x_eval_list: List[torch.Tensor] = [x_real, x_fake]
                    u_eval_list: List[torch.Tensor] = [u_r, u_f]
                    recon_c: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                    if use_chords and target_matched is not None and alpha is not None:
                        x_chord = ((1.0 - alpha) * x_fake.detach() + alpha * target_matched).detach()
                        u_c, recon_c = _prepare_spatial_tangent_plan(
                            x_chord, phase, salt=3, cond_tensors=cond_tensors
                        )
                        x_eval_list.append(x_chord)
                        u_eval_list.append(u_c)

                    self.record._in_penalty = True
                    try:
                        logits_list, tau_sq_list = _eval_forward_tangent_sq(
                            live,
                            x_eval_list,
                            u_eval_list,
                            can_concat=bool(
                                can_concat_batches and not is_conditioned and x_real.shape[1:] == x_fake.shape[1:]
                            ),
                        )
                    finally:
                        self.record._in_penalty = False

                    d_r_2d, d_f_2d = logits_list[0], logits_list[1]
                    sq_real = recon_r(tau_sq_list[0])
                    sq_fake = recon_f(tau_sq_list[1])
                    sq_chord = recon_c(tau_sq_list[2]) if (recon_c is not None and len(tau_sq_list) > 2) else None

                if is_discrete_simplex:
                    norm_real = (sq_real + 1e-12).sqrt()
                    norm_fake = (sq_fake + 1e-12).sqrt()
                    norm_chord = 0.5 * (norm_real + norm_fake)
                    floor_pen = x_real.new_zeros(())
                    cap_real = F.relu(norm_real - self.kappa).square().mean()
                    cap_fake = F.relu(norm_fake - self.kappa).square().mean()
                    ebi_loss = 0.5 * self.lambda_cap * (cap_real + cap_fake)
                    mean_logit = x_real.new_zeros(())
                    gauge_loss = x_real.new_zeros(())
                    self.record._last_d_logits = (d_r_2d.reshape(-1).detach(), d_f_2d.reshape(-1).detach())
                else:
                    settle_s = min(1.0, max(0.0, 1.0 - res_ratio / 0.16))
                    b_frac = min(1.0, max(0.0, (float(b_r) - 64.0) / 64.0))
                    cool_boost = 1.0 - float(t_scale)
                    batch_gain = (2.0 + 3.0 * b_frac) + cool_boost * (4.5 + 12.5 * b_frac)
                    eff_ebi_beta = self.ebi_beta + batch_gain * (settle_s * settle_s)

                    norm_real = (sq_real + 1e-12).sqrt() / scale_dim
                    norm_fake = (sq_fake + 1e-12).sqrt() / scale_dim
                    cap_real = F.softplus(norm_real - kappa_eff, beta=eff_ebi_beta).square().mean()
                    cap_fake = F.softplus(norm_fake - kappa_eff, beta=eff_ebi_beta).square().mean()

                    if sq_chord is not None:
                        norm_chord = (sq_chord + 1e-12).sqrt() / scale_dim
                        floor_pen = 0.5 * self.lambda_floor * (
                            (sq_real.mean() + sq_fake.mean() + sq_chord.mean()) / (3.0 * dim)
                        )
                        cap_chord = F.softplus(norm_chord - kappa_eff, beta=eff_ebi_beta).square().mean()
                        cap_pen = 0.5 * self.lambda_cap * (
                            cap_real + (1.0 - self.chord_blend) * cap_fake + self.chord_blend * cap_chord
                        )
                    else:
                        norm_chord = 0.5 * (norm_real + norm_fake)
                        floor_pen = 0.5 * self.lambda_floor * ((sq_real.mean() + sq_fake.mean()) / (2.0 * dim))
                        cap_pen = 0.5 * self.lambda_cap * (cap_real + cap_fake)
                    ebi_loss = floor_pen + cap_pen

                    self.record._last_d_logits = (d_r_2d.reshape(-1).detach(), d_f_2d.reshape(-1).detach())
                    mean_logit = 0.5 * (d_r_2d.mean() + d_f_2d.mean())
                    abs_mean = float(mean_logit.detach().abs().item())
                    if b_r < 16:
                        eff_gamma_gauge = min(self.gamma_gauge, 0.020)
                    else:
                        offset_boost = 0.14 * min(1.0, max(0.0, (abs_mean - 2.0) / 8.0))
                        eff_gamma_gauge = self.gamma_gauge + 0.10 * (1.0 - res_ratio) + offset_boost
                    gauge_loss = 0.5 * eff_gamma_gauge * mean_logit.square()

                norm_chord_stat = norm_chord.detach().mean()

            penalty = (self.coeff * self.lazy_k) * (ebi_loss + gauge_loss)
            if t_scale <= FROZEN_THERMAL_SCALE and penalty.requires_grad:
                penalty = penalty * t_scale + penalty.detach() * (1.0 - t_scale)

            if not collect_stats:
                return penalty, {}
            pen_v, grad_v, logit_v, ebi_v, gauge_v, floor_v = torch.stack(
                (
                    penalty.detach().float(),
                    norm_chord_stat.float(),
                    mean_logit.detach().float(),
                    ebi_loss.detach().float(),
                    gauge_loss.detach().float(),
                    floor_pen.detach().float(),
                )
            ).tolist()
            return penalty, {
                "applied": True,
                "pen": float(pen_v),
                "center": self.kappa,
                "phase": self.arm,
                "mean_grad_norm": float(grad_v),
                "mean_logit": float(logit_v),
                "ebi_loss": float(ebi_v),
                "gauge_loss": float(gauge_v),
                "floor_pen": float(floor_v),
                "s_thermal": float(t_scale),
                "step_scale": float(step_scale_val),
                "ema_residual": float(getattr(self.record, "ema_residual", 1.0)),
                "ema_data_scale": float(getattr(self.record, "ema_data_scale", 1.0)),
                "ema_margin": float(getattr(self.record, "ema_margin", 0.0)),
                "calibration_gate": float(getattr(self.record, "calibration_gate", 1.0)),
                "d_trust_scale": float(getattr(self.record, "d_trust_scale", 1.0)),
                "proto_revivals": int(getattr(self.record, "revivals", 0)),
                "live_endpoints_reused": bool(reused_endpoints),
            }
        finally:
            if hasattr(self.record, "clear_transient_captures"):
                self.record.clear_transient_captures()


class NashV3CriticPenalty(_K3PCriticPenalty):
    """Recipe-facing pure 1st-order T4NashV3 critic penalty."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides) -> None:
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (NashV3CriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_nash_v3_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _NashV3PenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = isinstance(self.critic, nn.Module) and any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
        )

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        cid = id(critic)
        if cid not in self._verified_critics:
            self._ema_view(critic)
            self._verified_critics.add(cid)
        from ..recipe_schedules import apply_penalty_schedule

        if getattr(self, "recipe", None) is not None:
            apply_penalty_schedule(self.optimizer.record.observed_steps, self.recipe, self)
        output = self.output

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))

        base_b = x_real.shape[0]

        def live_rep(x, m: int):
            if m <= 1 or (not condition and not condition_kwargs):
                return output(critic(x, *condition, **condition_kwargs))
            if x.shape[0] != m * base_b:
                return output(critic(x, *condition, **condition_kwargs))
            rep_args = tuple(
                t.repeat((m,) + (1,) * (t.ndim - 1))
                if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == base_b
                else t
                for t in condition
            )
            rep_kwargs = {
                k: (
                    v.repeat((m,) + (1,) * (v.ndim - 1))
                    if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == base_b
                    else v
                )
                for k, v in condition_kwargs.items()
            }
            return output(critic(x, *rep_args, **rep_kwargs))

        has_batch_state = isinstance(critic, torch.nn.Module) and any(
            isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == x_real.shape[0]
            for v in vars(critic).values()
        )
        uses_input_noise = float(getattr(critic, "std", 0.0)) > 0.0
        is_conditioned = bool(has_batch_state or condition or condition_kwargs)
        cond_tensors = [
            t.detach()
            for t in (*condition, *(condition_kwargs[k] for k in sorted(condition_kwargs)))
            if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == x_real.shape[0]
        ]
        cond_sig = _cond_signature((x_real, *condition), condition_kwargs)
        allow_live_reuse = (output is _first_output) and not has_batch_state
        can_concat_batches = (
            not self._has_batchnorm
            and not has_batch_state
            and not uses_input_noise
            and (output is _first_output)
        )
        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(
            live,
            x_real,
            x_fake,
            step,
            self.collect_stats,
            is_conditioned=is_conditioned,
            cond_tensors=cond_tensors,
            cond_sig=cond_sig,
            allow_live_reuse=allow_live_reuse,
            can_concat_batches=can_concat_batches,
            live_rep=live_rep,
            ema_critic=_no_anchor,
        )
        self.last_stats = stats
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            ema_residual=float(getattr(record, "ema_residual", 1.0)),
            ema_data_scale=float(getattr(record, "ema_data_scale", 1.0)),
            ema_margin=float(getattr(record, "ema_margin", 0.0)),
            calibration_gate=float(getattr(record, "calibration_gate", 1.0)),
            d_trust_scale=float(getattr(record, "d_trust_scale", 1.0)),
            s_thermal=float(getattr(record, "_thermal_scale", 1.0)),
            step_scale=float(getattr(record, "_step_scale", 1.0)),
            proto_revivals=int(getattr(record, "revivals", 0)),
        )
        return out


__all__ = [
    "NashV3CriticPenalty",
    "_NashV3PenaltyKernel",
]
