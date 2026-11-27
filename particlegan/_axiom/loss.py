"""Axiom L2 GANLoss adapter composing ``state`` captures and ``AxiomTransportLoss``."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .state import AxiomStepRecord, check_scalar, format_logits_2d, lookup_axiom_capture
from .transport import AxiomTransportLoss, compute_contrast_confidence, project_features


def _eq_t(a, b):
    return (
        a is not None
        and b is not None
        and a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.detach(), b.detach())
    )


def _same_opt(a, b):
    return (a is None and b is None) or _eq_t(a, b)


def _split_cond(pfa, pra):
    if not pfa or not pra or len(pfa) != len(pra) or len(pfa) < 2:
        return None, None, None
    bn = pfa[0].shape[0]
    if bn <= 0 or pra[0].shape[0] != bn:
        return None, None, None
    sf, sr, ctx = [], [], []
    for pf, pr in zip(pfa, pra):
        if not pf.requires_grad and _eq_t(pf, pr):
            ctx.append(pf.detach())
        else:
            sf.append(pf)
            sr.append(pr)
    if not ctx or not sf:
        return None, None, None

    def c(xs):
        return xs[0] if len(xs) == 1 else torch.cat(xs, -1)

    xf, xr = c(sf), c(sr)
    return (xf, xr, c(ctx)) if xf.shape == xr.shape else (None, None, None)


def _scalar_pair_rel(pos, neg, *, detach_neg=False):
    bp, bn = pos.shape[0], neg.shape[0]
    if bp != bn or bp < 1:
        return None
    p = format_logits_2d(pos, bp).mean(1)
    n = format_logits_2d(neg.detach() if detach_neg else neg, bn).mean(1)
    return F.softplus(n - p).mean()


def _mut(rec):
    return torch.is_grad_enabled() and not (
        getattr(rec, "_in_probe", False) or getattr(rec, "_in_protected_probe", False)
    )


class AxiomGANLoss(GANLoss):
    """Thin Axiom v6 ``GANLoss`` adapter."""

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: tuple[float, float, float] = (0.0, 1.0, 1.0),
        weight: float = 1.0,
        fibre_rel_blend: float = 0.50,
        cond_ot_weight: float = 1.5,
        record: AxiomStepRecord | None = None,
        **transport_kw: Any,
    ) -> None:
        super().__init__(loss, labels=labels)
        w = check_scalar(weight, "weight", lo=0)
        self.cond_ot_weight = check_scalar(cond_ot_weight, "cond_ot_weight", lo=0)
        self.fibre_rel_blend = check_scalar(fibre_rel_blend, "fibre_rel_blend", lo=0, hi=1)
        self.record: AxiomStepRecord | None = record
        self.transport = self.t1c = AxiomTransportLoss(weight=w, **transport_kw)
        self.last_metrics: dict[str, float] = {}

    @property
    def axiom_record(self) -> AxiomStepRecord | None:
        return self.record

    @axiom_record.setter
    def axiom_record(self, value: AxiomStepRecord | None) -> None:
        self.record = value

    def _pair(self, f, r):
        ef = lookup_axiom_capture(f, self.record)
        h = ef.record if (ef is not None and ef.record is not None) else self.record
        er = lookup_axiom_capture(r, h)
        return ef, er, (h if h is not None else (er.record if er is not None else None))

    def _cond_g(self, f, r, ef, er, rec):
        w, rel = self.transport.weight, _scalar_pair_rel(f, r, detach_neg=True) if f.shape[0] == r.shape[0] else None
        if rel is None:
            return w * super().g_loss(f, r)
        if self.cond_ot_weight <= 0 or not _same_opt(ef.cond, er.cond):
            return w * rel
        xf, xr, cc = _split_cond(ef.parts, er.parts)
        if xf is None or xr is None or cc is None:
            return w * rel
        ot = self.transport.conditional_fiber_loss(
            xf, xr, cc, cond_disc=ef.cond, record=rec, allow_state_update=_mut(rec)
        )
        if rec is not None and hasattr(rec, "_simplicial_step"):
            rec._simplicial_step = False
        s = float(getattr(rec, "thermal_scale", 1.0)) if rec is not None else 1.0
        tot = rel + self.cond_ot_weight * ot
        if s < 1 and tot.requires_grad:
            tot = (
                (rel + self.cond_ot_weight * (ot * s + ot.detach() * (1 - s)))
                if ef.thermally_scaled
                else (tot * s + tot.detach() * (1 - s))
            )
        return w * tot

    def _call(self, fn, f, r, ef, er, rec):
        return fn(
            ef.x_in,
            er.x_in,
            f,
            r,
            record=rec,
            cond_fake=ef.cond,
            cond_real=er.cond,
            allow_state_update=_mut(rec),
            thermally_scaled=ef.thermally_scaled,
        )

    def d_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        """Critic loss with sync-free ``kappa_struct`` relativistic blend."""
        base = super().d_loss(real_logits, fake_logits)
        ef, er, rec = self._pair(fake_logits, real_logits)
        cap = ef is not None and er is not None
        hcc = cap and (ef.has_cont_cond or er.has_cont_cond)
        if rec is not None and hasattr(rec, "observe_margin_and_scale") and not hcc:
            rec.observe_margin_and_scale(real_logits, fake_logits, *((er.x_in, ef.x_in) if cap else ()))
        if self.loss != "non_saturating" or self.transport.weight == 0 or self.fibre_rel_blend <= 0:
            return base
        pair = _scalar_pair_rel(real_logits, fake_logits)
        if pair is None:
            return base
        if not cap or hcc:
            mix: float | Tensor = self.fibre_rel_blend
        else:
            yp = project_features(er.x_in.detach(), self.transport.max_spatial, self.transport.max_flat_dim)
            mix = (self.fibre_rel_blend * (1 - compute_contrast_confidence(yp))).to(base.dtype)
        return (1 - mix) * base + mix * pair

    def g_loss(self, fake_logits: Tensor, real_logits: Tensor | None = None) -> Tensor:
        """Generator loss dispatching conditional fiber or Law 3A transport."""
        w = self.transport.weight
        if real_logits is None:
            if w == 0:
                return super().g_loss(fake_logits, None)
            raise ValueError("real_logits required")
        if w == 0:
            return super().g_loss(fake_logits, real_logits)
        ef, er, rec = self._pair(fake_logits, real_logits)
        if ef is None or er is None:
            base = super().g_loss(fake_logits, real_logits)
            pair = (
                _scalar_pair_rel(fake_logits, real_logits, detach_neg=True)
                if (self.loss == "non_saturating" and self.fibre_rel_blend > 0)
                else None
            )
            b = self.fibre_rel_blend
            return w * (((1 - b) * base + b * pair) if pair is not None else base)
        if ef.has_cont_cond or er.has_cont_cond:
            return self._cond_g(fake_logits, real_logits, ef, er, rec)
        return self._call(self.transport, fake_logits, real_logits, ef, er, rec)

    def joint_g_loss(self, fake_logits: Tensor, real_logits: Tensor) -> Tensor:
        """Joint BiGAN / ByteGAN generator+encoder transport loss."""
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires grads")
        if self.transport.weight == 0:
            return super().joint_g_loss(fake_logits, real_logits)
        ef, er, rec = self._pair(fake_logits, real_logits)
        if ef is None or er is None:
            return super().joint_g_loss(fake_logits, real_logits)
        return self._call(self.transport.joint_relativistic_loss, fake_logits, real_logits, ef, er, rec)


__all__ = ["AxiomGANLoss"]
