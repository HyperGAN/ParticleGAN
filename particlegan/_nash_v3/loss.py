"""Scale-protected relativistic rapidity, unbiased transport, and O(BKD) pullback covariance isometry loss for T4NashV3."""
from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import torch.nn.functional as F

from .._empf.transport import balanced_sinkhorn_coupling, effective_sinkhorn_eps
from ..gan_loss import GANLoss
from .record import FROZEN_THERMAL_SCALE, _smoothstep01
from .transport import born_infeld_debiased_transport_loss, pullback_covariance_isometry_loss


def _protected_softplus_neg(delta: torch.Tensor, *, tau_logit: float = 3.5) -> torch.Tensor:
    """Scale-protected relativistic softplus ``s * softplus(-delta / s)``."""
    d_f32 = delta.to(dtype=torch.float32)
    with torch.no_grad():
        rms_sq = d_f32.detach().square().mean()
        s = torch.sqrt(1.0 + rms_sq / (float(tau_logit) * float(tau_logit)))
    out = s * F.softplus(-d_f32 / s)
    return out.to(dtype=delta.dtype)


class NashV3Loss(GANLoss):
    """Two-player calibration-coupled adversarial + unbiased Born-Infeld transport + pullback covariance isometry."""

    def __init__(
        self,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        *,
        loss: str = "relativistic",
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        transport_far_weight: float = 0.40,
        transport_rescue_weight: float = 1.15,
        isometry_weight: float = 0.35,
        tau_logit: float = 3.5,
        nash_v3_record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.transport_far_weight = float(transport_far_weight)
        self.transport_rescue_weight = float(transport_rescue_weight)
        self.isometry_weight = float(isometry_weight)
        self.tau_logit = float(tau_logit)
        self.nash_v3_record = nash_v3_record

    def _resolve_record(self, *tensors: torch.Tensor | None) -> Any:
        if self.nash_v3_record is not None:
            return self.nash_v3_record
        for t in tensors:
            if t is not None:
                rec = getattr(t, "_nash_v3_record", None)
                if rec is not None:
                    return rec
        return None

    @staticmethod
    def _same_continuous_fiber(t1: torch.Tensor | None, t2: torch.Tensor | None) -> bool:
        if t1 is None or t2 is None:
            return False
        c1 = getattr(t1, "_nash_v3_cond_tensors", ())
        c2 = getattr(t2, "_nash_v3_cond_tensors", ())
        if c1 and c2:
            if len(c1) != len(c2):
                return False
            if not any(isinstance(a, torch.Tensor) and a.is_floating_point() for a in c1):
                return False
            return all(
                isinstance(a, torch.Tensor)
                and isinstance(b, torch.Tensor)
                and a.shape == b.shape
                and a.dtype == b.dtype
                and a.device == b.device
                and bool(torch.equal(a, b))
                for a, b in zip(c1, c2)
            )
        c_live = c1 or c2
        if not c_live:
            return False
        if hasattr(t1, "_nash_v3_conditioned") and hasattr(t2, "_nash_v3_conditioned"):
            return False
        return any(isinstance(a, torch.Tensor) and a.is_floating_point() for a in c_live)

    def _spatial_inputs(
        self,
        real_logits: torch.Tensor | None,
        fake_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if real_logits is None:
            return None
        x_r = getattr(real_logits, "_nash_v3_input", None)
        x_f = getattr(fake_logits, "_nash_v3_input", None)
        if not isinstance(x_r, torch.Tensor) or not isinstance(x_f, torch.Tensor):
            return None
        if x_r.ndim != 2 or x_f.ndim != 2 or x_r.shape[1] != x_f.shape[1]:
            return None
        b_r, b_f = int(x_r.shape[0]), int(x_f.shape[0])
        dim = int(x_r.shape[1])
        if not (1 <= dim <= 8 and b_r >= 32 and b_f >= 32 and max(b_r, b_f) <= 2048):
            return None
        if b_r != real_logits.reshape(-1).shape[0] or b_f != fake_logits.reshape(-1).shape[0]:
            return None
        return x_r, x_f

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        rec = self._resolve_record(fake_logits, real_logits)
        if (
            rec is not None
            and torch.is_grad_enabled()
            and not getattr(rec, "_in_protected_probe", False)
            and hasattr(rec, "observe_margin")
        ):
            margin = float((r.detach().to(torch.float32).mean() - f.detach().to(torch.float32).mean()).item())
            rec.observe_margin(margin)

        if self._same_continuous_fiber(real_logits, fake_logits) and r.shape == f.shape:
            return _protected_softplus_neg(r - f, tau_logit=self.tau_logit).mean()
        if self.loss == "relativistic":
            spatial = self._spatial_inputs(real_logits, fake_logits)
            if spatial is not None:
                x_r, x_f = spatial
                ema_res = float(getattr(rec, "ema_residual", 1.0)) if rec is not None else 1.0
                eps_eff = effective_sinkhorn_eps(ema_res, self.sinkhorn_eps)
                pi = balanced_sinkhorn_coupling(x_r, x_f, sinkhorn_eps=eps_eff, sinkhorn_iters=self.sinkhorn_iters)
                delta = r.unsqueeze(1) - f.unsqueeze(0)
                return (pi * _protected_softplus_neg(delta, tau_logit=self.tau_logit)).sum()
            return 0.5 * (
                _protected_softplus_neg(r - f.mean(), tau_logit=self.tau_logit).mean()
                + _protected_softplus_neg(r.mean() - f, tau_logit=self.tau_logit).mean()
            )
        if self.loss == "non_saturating":
            return 0.5 * (
                _protected_softplus_neg(r, tau_logit=self.tau_logit).mean()
                + _protected_softplus_neg(-f, tau_logit=self.tau_logit).mean()
            )
        return super().d_loss(real_logits, fake_logits)

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        f = fake_logits.reshape(-1)
        rec = self._resolve_record(fake_logits, real_logits)
        in_probe = bool(getattr(rec, "_in_protected_probe", False)) if rec is not None else False
        c_d = float(getattr(rec, "calibration_gate", 1.0)) if rec is not None else 1.0
        ema_res = float(getattr(rec, "ema_residual", 1.0)) if rec is not None else 1.0
        s_therm = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
        qmc_phase = float(getattr(rec, "chord_phase", 0.0)) if rec is not None else 0.0

        if real_logits is not None:
            r = real_logits.reshape(-1)
            if self._same_continuous_fiber(fake_logits, real_logits) and r.shape == f.shape:
                return _protected_softplus_neg(f - r, tau_logit=self.tau_logit).mean()
            if self.loss in ("relativistic", "non_saturating"):
                spatial = self._spatial_inputs(real_logits, fake_logits)
                if spatial is not None:
                    x_r, x_f = spatial
                    dim = int(x_f.shape[1])
                    eps_eff = effective_sinkhorn_eps(ema_res, self.sinkhorn_eps)
                    pi = balanced_sinkhorn_coupling(x_r, x_f, sinkhorn_eps=eps_eff, sinkhorn_iters=self.sinkhorn_iters)
                    delta = f.unsqueeze(1) - r.unsqueeze(0)
                    b_scale = float(x_f.shape[0])
                    adv = (b_scale * pi.t() * _protected_softplus_neg(delta, tau_logit=self.tau_logit)).sum()

                    if not in_probe and (x_f.requires_grad or x_r.requires_grad) and s_therm > FROZEN_THERMAL_SCALE:
                        therm_aux = math.sqrt(s_therm) if dim == 1 else max(0.15, math.sqrt(s_therm))
                        near_snap = 0.22 if dim >= 2 else 0.45
                        far_scaffold = _smoothstep01((ema_res - 0.16) / 0.22)
                        uncal_rescue = 1.0 - c_d
                        w_ot = (
                            near_snap
                            + self.transport_far_weight * far_scaffold
                            + self.transport_rescue_weight * uncal_rescue
                        ) * therm_aux * b_scale
                        w_iso = self.isometry_weight * (2.60 if dim >= 2 else 0.70) * therm_aux * b_scale

                        protos = None
                        p_weights = None
                        if (
                            rec is not None
                            and getattr(rec, "proto_initialized", False)
                            and isinstance(rec.proto_sum, torch.Tensor)
                            and isinstance(rec.proto_count, torch.Tensor)
                        ):
                            p_c = rec.proto_count.to(device=x_f.device, dtype=torch.float32).clamp_min(1e-8)
                            protos = (rec.proto_sum.to(device=x_f.device, dtype=torch.float32) / p_c.unsqueeze(1)).detach()
                            p_weights = (p_c / p_c.sum().clamp_min(1e-12)).detach()

                        aux = x_f.new_zeros(())
                        if x_f.requires_grad:
                            if w_ot > 1e-6:
                                aux = aux + w_ot * born_infeld_debiased_transport_loss(
                                    x_f,
                                    x_r,
                                    prototypes=protos,
                                    proto_weights=p_weights,
                                    ema_residual=ema_res,
                                    calibration_gate=c_d,
                                    qmc_phase=qmc_phase,
                                )
                            if w_iso > 1e-6:
                                aux = aux + w_iso * pullback_covariance_isometry_loss(
                                    x_f,
                                    x_r,
                                    prototypes=protos,
                                    ema_residual=ema_res,
                                    qmc_phase=qmc_phase,
                                )
                        # Symmetric joint BiGAN support: when real branch carries encoder gradients (x_r = [x, E(x)]).
                        if x_r.requires_grad:
                            if w_ot > 1e-6:
                                aux = aux + w_ot * born_infeld_debiased_transport_loss(
                                    x_r,
                                    x_f.detach(),
                                    prototypes=None,
                                    proto_weights=None,
                                    ema_residual=ema_res,
                                    calibration_gate=c_d,
                                    qmc_phase=qmc_phase,
                                )
                            if w_iso > 1e-6:
                                aux = aux + w_iso * pullback_covariance_isometry_loss(
                                    x_r,
                                    x_f.detach(),
                                    prototypes=None,
                                    ema_residual=ema_res,
                                    qmc_phase=qmc_phase,
                                )
                        return c_d * adv + aux
                    return c_d * adv
                if self.loss == "relativistic":
                    return 0.5 * (
                        _protected_softplus_neg(f - r.mean(), tau_logit=self.tau_logit).mean()
                        + _protected_softplus_neg(f.mean() - r, tau_logit=self.tau_logit).mean()
                    )
                return _protected_softplus_neg(f, tau_logit=self.tau_logit).mean()
        elif self.loss in ("relativistic", "non_saturating"):
            return _protected_softplus_neg(f, tau_logit=self.tau_logit).mean()
        return super().g_loss(fake_logits, real_logits)


__all__ = ["NashV3Loss", "_protected_softplus_neg"]
