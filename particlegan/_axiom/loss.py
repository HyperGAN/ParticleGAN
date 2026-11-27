"""Axiom L2 GANLoss Adapter composing ``state`` Captures and ``AxiomTransportLoss``."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .state import (
    AxiomStepRecord,
    CaptureEntry,
    check_scalar,
    format_logits_2d,
    lookup_axiom_capture,
)
from .transport import (
    AxiomTransportLoss,
    compute_contrast_confidence,
    project_features,
)


def _is_shared_context_part(pf: torch.Tensor, pr: torch.Tensor) -> bool:
    if pf.requires_grad or pf.shape != pr.shape or pf.dtype != pr.dtype:
        return False
    return bool(torch.equal(pf.detach(), pr.detach()))


def _cat_parts(parts: list[torch.Tensor]) -> torch.Tensor:
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def _split_shared_conditional_parts(
    parts_f: Optional[Tuple[torch.Tensor, ...]],
    parts_r: Optional[Tuple[torch.Tensor, ...]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Split multi-tensor critic inputs into ``(x_fake_sample, x_real_sample, shared_c_cont)``."""
    if not parts_f or not parts_r:
        return None, None, None
    if len(parts_f) != len(parts_r) or len(parts_f) < 2:
        return None, None, None
    bn = int(parts_f[0].shape[0])
    if bn <= 0 or int(parts_r[0].shape[0]) != bn:
        return None, None, None
    sample_f, sample_r, ctx = [], [], []
    for pf, pr in zip(parts_f, parts_r):
        if _is_shared_context_part(pf, pr):
            ctx.append(pf.detach())
        else:
            sample_f.append(pf)
            sample_r.append(pr)
    if not ctx or not sample_f:
        return None, None, None
    x_sf, x_sr = _cat_parts(sample_f), _cat_parts(sample_r)
    if x_sf.shape != x_sr.shape:
        return None, None, None
    return x_sf, x_sr, _cat_parts(ctx)


def _scalar_pair_rel(
    pos: torch.Tensor, neg: torch.Tensor, *, detach_neg: bool = False
) -> Optional[torch.Tensor]:
    bp, bn = int(pos.shape[0]), int(neg.shape[0])
    if bp != bn or bp < 1:
        return None
    p_flat = format_logits_2d(pos, bp).mean(dim=1)
    n_src = neg.detach() if detach_neg else neg
    n_flat = format_logits_2d(n_src, bn).mean(dim=1)
    return F.softplus(n_flat - p_flat).mean()


def _same_discrete_cond(
    cond_f: Optional[torch.Tensor], cond_r: Optional[torch.Tensor]
) -> bool:
    if cond_f is None and cond_r is None:
        return True
    if cond_f is None or cond_r is None or cond_f.shape != cond_r.shape:
        return False
    return bool(torch.equal(cond_f, cond_r))


def _can_mutate(record: Any) -> bool:
    if not torch.is_grad_enabled():
        return False
    return not (
        getattr(record, "_in_probe", False)
        or getattr(record, "_in_protected_probe", False)
    )


class AxiomGANLoss(GANLoss):
    """Thin Axiom v6 ``GANLoss`` adapter for GAN, BiGAN, AE/VAE-GAN, DDGAN & ByteGAN."""

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        weight: float = 1.0,
        rho: float = 3.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        chart_capacity: int = 256,
        chart_weight: float = 1.4,
        halo_mult: float = 2.1,
        max_spatial: int = 8,
        fibre_rel_blend: float = 0.50,
        cond_ot_weight: float = 1.5,
        mean_op_weight: float = 0.36,
        cov_op_weight: float = 0.35,
        imq_active_scale: float = 0.40,
        bank_mean_blend: float = 0.75,
        record: Optional[AxiomStepRecord] = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        w_val = check_scalar(weight, "weight", lo=0.0)
        self.cond_ot_weight = check_scalar(cond_ot_weight, "cond_ot_weight", lo=0.0)
        self.fibre_rel_blend = check_scalar(
            fibre_rel_blend, "fibre_rel_blend", lo=0.0, hi=1.0
        )
        self.record: Optional[AxiomStepRecord] = record
        self.transport = AxiomTransportLoss(
            weight=w_val,
            rho=float(rho),
            tau=float(tau),
            eps=float(eps),
            iters=int(iters),
            chart_capacity=int(chart_capacity),
            chart_weight=float(chart_weight),
            halo_mult=float(halo_mult),
            max_spatial=int(max_spatial),
            mean_op_weight=float(mean_op_weight),
            cov_op_weight=float(cov_op_weight),
            imq_active_scale=float(imq_active_scale),
            bank_mean_blend=float(bank_mean_blend),
        )
        self.t1c = self.transport
        self.last_metrics: dict[str, float] = {}

    @property
    def axiom_record(self) -> Optional[AxiomStepRecord]:
        return self.record

    @axiom_record.setter
    def axiom_record(self, value: Optional[AxiomStepRecord]) -> None:
        self.record = value

    def _resolve_pair(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> Tuple[Optional[CaptureEntry], Optional[CaptureEntry], Any]:
        ef = lookup_axiom_capture(fake_logits, self.record)
        rec_hint = (
            ef.record if (ef is not None and ef.record is not None) else self.record
        )
        er = lookup_axiom_capture(real_logits, rec_hint)
        rec = (
            rec_hint
            if rec_hint is not None
            else (er.record if er is not None else None)
        )
        return ef, er, rec

    def _blended_g(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        base_g = super().g_loss(fake_logits, real_logits)
        if self.loss == "non_saturating" and self.fibre_rel_blend > 0.0:
            pair_g = _scalar_pair_rel(fake_logits, real_logits, detach_neg=True)
            if pair_g is not None:
                b = self.fibre_rel_blend
                return self.transport.weight * ((1.0 - b) * base_g + b * pair_g)
        return self.transport.weight * base_g

    def _conditional_g_loss(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor,
        ef: CaptureEntry,
        er: CaptureEntry,
        rec: Any,
    ) -> torch.Tensor:
        if int(fake_logits.shape[0]) != int(real_logits.shape[0]):
            return self.transport.weight * super().g_loss(fake_logits, real_logits)
        rel_term = _scalar_pair_rel(fake_logits, real_logits, detach_neg=True)
        if rel_term is None:
            return self.transport.weight * super().g_loss(fake_logits, real_logits)
        if self.cond_ot_weight <= 0.0 or not _same_discrete_cond(ef.cond, er.cond):
            return self.transport.weight * rel_term
        x_sf, x_sr, c_cont = _split_shared_conditional_parts(ef.parts, er.parts)
        if x_sf is None or x_sr is None or c_cont is None:
            return self.transport.weight * rel_term
        cond_ot = self.transport.conditional_fiber_loss(
            x_sf,
            x_sr,
            c_cont,
            cond_disc=ef.cond,
            record=rec,
            allow_state_update=_can_mutate(rec),
        )
        if rec is not None and hasattr(rec, "_simplicial_step"):
            rec._simplicial_step = False
        s_th = float(getattr(rec, "thermal_scale", 1.0)) if rec is not None else 1.0
        total = rel_term + self.cond_ot_weight * cond_ot
        if s_th < 1.0 and total.requires_grad:
            if ef.thermally_scaled:
                scaled_ot = cond_ot * s_th + cond_ot.detach() * (1.0 - s_th)
                total = rel_term + self.cond_ot_weight * scaled_ot
            else:
                total = total * s_th + total.detach() * (1.0 - s_th)
        return self.transport.weight * total

    def _d_rel_mix(
        self, er: Optional[CaptureEntry], captured: bool, has_cc: bool
    ) -> float:
        if not captured or has_cc or er is None:
            return self.fibre_rel_blend
        yp = project_features(
            er.x_in.detach(),
            self.transport.max_spatial,
            self.transport.max_flat_dim,
        )
        kappa = float(compute_contrast_confidence(yp).item())
        return self.fibre_rel_blend * (1.0 - kappa)

    def d_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        """Critic loss with smooth ``kappa_struct`` relativistic blend and scale observation."""
        base = super().d_loss(real_logits, fake_logits)
        ef, er, rec = self._resolve_pair(fake_logits, real_logits)
        captured = ef is not None and er is not None
        has_cc = bool(captured and (ef.has_cont_cond or er.has_cont_cond))
        if rec is not None and hasattr(rec, "observe_margin_and_scale") and not has_cc:
            if captured:
                rec.observe_margin_and_scale(real_logits, fake_logits, er.x_in, ef.x_in)
            else:
                rec.observe_margin_and_scale(real_logits, fake_logits)
        if (
            self.loss != "non_saturating"
            or self.transport.weight == 0.0
            or self.fibre_rel_blend <= 0.0
        ):
            return base
        mix = self._d_rel_mix(er, captured, has_cc)
        if mix <= 1e-4:
            return base
        pair_d = _scalar_pair_rel(real_logits, fake_logits)
        return ((1.0 - mix) * base + mix * pair_d) if pair_d is not None else base

    def g_loss(
        self, fake_logits: torch.Tensor, real_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generator loss dispatching conditional fiber or dimension-free Law 3A transport."""
        if real_logits is None:
            if self.transport.weight == 0.0:
                return super().g_loss(fake_logits, None)
            raise ValueError(
                "AxiomGANLoss requires real_logits in g_loss when transport weight > 0"
            )
        if self.transport.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)
        ef, er, rec = self._resolve_pair(fake_logits, real_logits)
        if ef is None or er is None:
            return self._blended_g(fake_logits, real_logits)
        if ef.has_cont_cond or er.has_cont_cond:
            return self._conditional_g_loss(fake_logits, real_logits, ef, er, rec)
        return self.transport(
            ef.x_in,
            er.x_in,
            fake_logits,
            real_logits,
            record=rec,
            cond_fake=ef.cond,
            cond_real=er.cond,
            allow_state_update=_can_mutate(rec),
            thermally_scaled=ef.thermally_scaled,
        )

    def joint_g_loss(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Joint BiGAN / ByteGAN generator+encoder transport loss."""
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError(
                "joint_g_loss requires gradients on both generator and encoder streams"
            )
        if self.transport.weight == 0.0:
            return super().joint_g_loss(fake_logits, real_logits)
        ef, er, rec = self._resolve_pair(fake_logits, real_logits)
        if ef is None or er is None:
            return super().joint_g_loss(fake_logits, real_logits)
        return self.transport.joint_relativistic_loss(
            ef.x_in,
            er.x_in,
            fake_logits,
            real_logits,
            record=rec,
            cond_fake=ef.cond,
            cond_real=er.cond,
            allow_state_update=_can_mutate(rec),
            thermally_scaled=ef.thermally_scaled,
        )


__all__ = ["AxiomGANLoss"]
