"""T4Nash critic penalty: intrinsic-scale e-means Sinkhorn chords, Born-Infeld slope cap, and adaptive logit gauge."""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

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
from ..grad_regularizers import CriticStepRecord, _score_scalar
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output, _no_anchor
from .optim import T4NashCriticAdam
from .record import FROZEN_THERMAL_SCALE, T4NashStepRecord, _attach_t4_critic_hook
from .transport import intrinsic_sinkhorn_coupling


class _T4NashPenaltyKernel:
    """Clock-free T4Nash critic penalty kernel paired with a ``T4NashStepRecord``."""

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
                "T4Nash uses a zero-mean logit gauge instead of an EMA critic anchor (require reg_anchor_weight=0)"
            )
        if not isinstance(input_domain, str) or input_domain not in ("auto", "continuous", "categorical"):
            raise ValueError("input_domain must be auto, continuous or categorical")

        self.arm = "t4_nash"
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
        self.input_domain = input_domain
        self.record = T4NashStepRecord() if record is None else record

    def _observe(self, res: float, data_scale: float, *, ema_decay: float | None = None) -> None:
        decay = self.thermal_decay if ema_decay is None else float(ema_decay)
        if hasattr(self.record, "observe_transport"):
            self.record.observe_transport(res, data_scale, ema_decay=decay)

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del kwargs
        if self.input_domain == "categorical" and (x_real.ndim != 2 or x_fake.ndim != 2):
            raise ValueError("categorical T4Nash inputs require native [batch, categories] tensors")
        if self.coeff == 0.0 or (self.lazy_k > 1 and step % self.lazy_k != 0):
            self.record._last_d_logits = None
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        if isinstance(live, nn.Module) and getattr(live, "_t4_record", None) is not self.record:
            _attach_t4_critic_hook(live, self.record)

        self.record.calls += 1
        b_r, b_f = x_real.shape[0], x_fake.shape[0]
        dim = max(1, x_real[0].numel())

        is_image_tensor = x_real.ndim >= 3 and self.input_domain != "continuous"
        is_discrete_simplex = self.input_domain == "categorical" or (
            self.input_domain == "auto" and x_real.ndim == 2 and dim >= 170
        )
        use_chords = not is_discrete_simplex and not is_image_tensor and x_real.ndim == 2 and dim <= 64

        target_matched: torch.Tensor | None = None
        alpha: torch.Tensor | None = None
        local_clump_gain: float = 1.0
        step_scale_val: float = 1.0

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
                    d2_rr_clump = pairwise_sq_dist(r_flat_clump)
                    eye_clump = torch.eye(b_r, device=x_real.device, dtype=torch.bool)
                    d2_rr_masked_c = d2_rr_clump.masked_fill(eye_clump, float("inf"))
                    med_d_c = float(d2_rr_clump.masked_fill(eye_clump, 0.0).median().clamp_min(1e-12).sqrt().item())
                    k_nn_c = max(2, min(12, b_r // 32))
                    knn_d2_c = torch.topk(d2_rr_masked_c, k=k_nn_c, dim=1, largest=False).values[:, -1]
                    knn_med_c = float(knn_d2_c.median().clamp_min(1e-8).sqrt().item())
                    expected_homo_c = med_d_c * ((float(k_nn_c) / float(b_r)) ** (1.0 / float(dim)))
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
                j_idx = torch.arange(b_f, device=x_real.device, dtype=x_real.dtype)
                phase = float(getattr(self.record, "chord_phase", 0.0))
                q = torch.remainder((j_idx + 0.5) / max(1, b_f) + phase, 1.0).unsqueeze(1)
                matched_idx = torch.searchsorted(cdf, q).squeeze(1).clamp(0, prototypes.shape[0] - 1)
                target_matched = prototypes[matched_idx].reshape((b_f,) + x_real.shape[1:])
                u = torch.remainder((j_idx + 0.5) / max(1, b_f) + ((phase + _PHI2) % 1.0), 1.0)
                alpha = u.reshape((b_f,) + (1,) * (x_real.ndim - 1))
                self.record.chord_phase = (phase + _PHI1) % 1.0
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
                    phase = float(getattr(self.record, "chord_phase", 0.0))
                    u = torch.remainder((j_idx + 0.5) / max(1, b_f) + phase, 1.0)
                    alpha = u.reshape((b_f,) + (1,) * (x_real.ndim - 1))
                    target_matched = x_real.detach()
                    self.record.chord_phase = (phase + _PHI1) % 1.0
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
            res, data_scale = unconditioned_chamfer_residual(x_real.detach(), x_fake.detach())
            obs_res, obs_ds = torch.stack((res.detach(), data_scale.detach())).tolist()
            self._observe(float(obs_res), float(obs_ds))
            t_scale = 1.0

        self.record._thermal_scale = float(t_scale)
        self.record._step_scale = float(step_scale_val)

        scale_dim = 1.0 if (is_discrete_simplex or is_image_tensor or dim <= 4) else math.sqrt(dim)

        self.record._in_penalty = True
        try:
            x_r_req = x_real.detach().clone().requires_grad_(True)
            d_r_out = live(x_r_req)
            x_f_req = x_fake.detach().clone().requires_grad_(True)
            d_f_out = live(x_f_req)

            g_chord: torch.Tensor | None = None
            if use_chords and target_matched is not None and alpha is not None:
                x_chord = ((1.0 - alpha) * x_fake.detach() + alpha * target_matched).requires_grad_(True)
                d_chord_out = live(x_chord)
                g_real, g_fake, g_chord = torch.autograd.grad(
                    outputs=(_score_scalar(d_r_out), _score_scalar(d_f_out), _score_scalar(d_chord_out)),
                    inputs=(x_r_req, x_f_req, x_chord),
                    create_graph=True,
                    only_inputs=True,
                )
            else:
                g_real, g_fake = torch.autograd.grad(
                    outputs=(_score_scalar(d_r_out), _score_scalar(d_f_out)),
                    inputs=(x_r_req, x_f_req),
                    create_graph=True,
                    only_inputs=True,
                )
        finally:
            self.record._in_penalty = False

        res_ratio = min(1.0, max(0.0, float(getattr(self.record, "ema_residual", 1.0))))
        kappa_eff: float | torch.Tensor
        if is_image_tensor:
            ref_scale = 4.899
            img_scale = min(1.0, max(0.15, float(getattr(self.record, "ema_data_scale", ref_scale)) / ref_scale))
            kappa_eff = (self.kappa / img_scale) * (0.6 + 1.10 * (1.0 - res_ratio))
        elif local_clump_gain > 1.0:
            settle_fine = min(1.0, max(0.0, 1.0 - res_ratio / 0.22))
            kappa_eff = self.kappa * (1.0 + (local_clump_gain - 1.0) * (settle_fine * settle_fine))
        else:
            kappa_eff = self.kappa

        if is_discrete_simplex:
            norm_real = (g_real.flatten(1).pow(2).sum(dim=1) + 1e-12).sqrt()
            norm_fake = (g_fake.flatten(1).pow(2).sum(dim=1) + 1e-12).sqrt()
            norm_chord = 0.5 * (norm_real + norm_fake)
            floor_pen = x_real.new_zeros(())
            cap_real = F.relu(norm_real - self.kappa).pow(2).mean()
            cap_fake = F.relu(norm_fake - self.kappa).pow(2).mean()
            ebi_loss = 0.5 * self.lambda_cap * (cap_real + cap_fake)
            mean_logit = x_real.new_zeros(())
            gauge_loss = x_real.new_zeros(())
            d_r = d_r_out.reshape(-1)
            d_f = d_f_out.reshape(-1)
            self.record._last_d_logits = (d_r.detach(), d_f.detach())
        else:
            settle_s = min(1.0, max(0.0, 1.0 - res_ratio / 0.16))
            b_frac = min(1.0, max(0.0, (float(b_r) - 64.0) / 64.0))
            cool_boost = 1.0 - float(t_scale)
            batch_gain = (2.0 + 3.0 * b_frac) + cool_boost * (4.5 + 12.5 * b_frac)
            eff_ebi_beta = self.ebi_beta + batch_gain * (settle_s * settle_s)
            sq_real = g_real.flatten(1).pow(2).sum(dim=1)
            norm_real = (sq_real + 1e-12).sqrt() / scale_dim
            sq_fake = g_fake.flatten(1).pow(2).sum(dim=1)
            norm_fake = (sq_fake + 1e-12).sqrt() / scale_dim
            cap_real = F.softplus(norm_real - kappa_eff, beta=eff_ebi_beta).pow(2).mean()
            cap_fake = F.softplus(norm_fake - kappa_eff, beta=eff_ebi_beta).pow(2).mean()

            eff_floor = 0.0005 if is_image_tensor else self.lambda_floor
            if g_chord is not None:
                sq_chord = g_chord.flatten(1).pow(2).sum(dim=1)
                norm_chord = (sq_chord + 1e-12).sqrt() / scale_dim
                floor_pen = 0.5 * eff_floor * ((sq_real.mean() + sq_fake.mean() + sq_chord.mean()) / (3.0 * dim))
                cap_chord = F.softplus(norm_chord - kappa_eff, beta=eff_ebi_beta).pow(2).mean()
                cap_pen = 0.5 * self.lambda_cap * (
                    cap_real + (1.0 - self.chord_blend) * cap_fake + self.chord_blend * cap_chord
                )
                ebi_loss = floor_pen + cap_pen
            else:
                norm_chord = 0.5 * (norm_real + norm_fake)
                floor_pen = 0.5 * eff_floor * ((sq_real.mean() + sq_fake.mean()) / (2.0 * dim))
                cap_pen = 0.5 * self.lambda_cap * (cap_real + cap_fake)
                ebi_loss = floor_pen + cap_pen

            d_r = d_r_out.reshape(-1)
            d_f = d_f_out.reshape(-1)
            self.record._last_d_logits = (d_r.detach(), d_f.detach())
            mean_logit = 0.5 * (d_r.mean() + d_f.mean())
            abs_mean = float(mean_logit.detach().abs().item())
            if is_image_tensor:
                eff_gamma_gauge = 0.005
            elif b_r < 16:
                eff_gamma_gauge = min(self.gamma_gauge, 0.020)
            else:
                offset_boost = 0.14 * min(1.0, max(0.0, (abs_mean - 2.0) / 8.0))
                eff_gamma_gauge = self.gamma_gauge + 0.10 * (1.0 - res_ratio) + offset_boost
            gauge_loss = 0.5 * eff_gamma_gauge * mean_logit.pow(2)

        penalty = (self.coeff * self.lazy_k) * (ebi_loss + gauge_loss)
        if t_scale <= FROZEN_THERMAL_SCALE and penalty.requires_grad:
            penalty = penalty * t_scale + penalty.detach() * (1.0 - t_scale)

        if not collect_stats:
            return penalty, {}
        pen_v, grad_v, logit_v, ebi_v, gauge_v, floor_v = torch.stack(
            (
                penalty.detach(),
                norm_chord.detach().mean(),
                mean_logit.detach(),
                ebi_loss.detach(),
                gauge_loss.detach(),
                floor_pen.detach(),
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
        }


class T4NashCriticPenalty(_K3PCriticPenalty):
    """Recipe-facing T4Nash critic penalty."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides) -> None:
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (T4NashCriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")

        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_t4_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _T4NashPenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        self._ema_view(critic)
        from ..recipe_schedules import apply_penalty_schedule

        if getattr(self, "recipe", None) is not None:
            apply_penalty_schedule(self.optimizer.record.observed_steps, self.recipe, self)
        output = self.output

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))

        is_conditioned = bool(condition or condition_kwargs)
        cond_tensors = [
            t.detach()
            for t in (*condition, *(condition_kwargs[k] for k in sorted(condition_kwargs)))
            if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == x_real.shape[0]
        ]
        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(
            live,
            x_real,
            x_fake,
            step,
            self.collect_stats,
            is_conditioned=is_conditioned,
            cond_tensors=cond_tensors,
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
