"""Axiom L2 GANLoss adapter composing ``state`` captures and ``AxiomTransportLoss``."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .state import AxiomStepRecord, check_scalar, format_logits_2d, lookup_axiom_capture
from .transport import AxiomTransportLoss


def _tensors_equal(first: Tensor | None, second: Tensor | None) -> bool:
    return (
        first is not None
        and second is not None
        and first.shape == second.shape
        and first.dtype == second.dtype
        and torch.equal(first.detach(), second.detach())
    )


def _optional_tensors_equal(first: Tensor | None, second: Tensor | None) -> bool:
    return (first is None and second is None) or _tensors_equal(first, second)


def _concat_parts(parts: list[Tensor]) -> Tensor:
    return parts[0] if len(parts) == 1 else torch.cat(parts, -1)


def _split_conditional_parts(
    fake_parts: tuple[Tensor, ...] | None,
    real_parts: tuple[Tensor, ...] | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    if not fake_parts or not real_parts or len(fake_parts) != len(real_parts) or len(fake_parts) < 2:
        return None, None, None
    batch_n = fake_parts[0].shape[0]
    if batch_n <= 0 or real_parts[0].shape[0] != batch_n:
        return None, None, None
    sample_fake, sample_real, context_parts = [], [], []
    for part_fake, part_real in zip(fake_parts, real_parts):
        if not part_fake.requires_grad and _tensors_equal(part_fake, part_real):
            context_parts.append(part_fake.detach())
        else:
            sample_fake.append(part_fake)
            sample_real.append(part_real)
    if not context_parts or not sample_fake:
        return None, None, None
    x_fake, x_real = _concat_parts(sample_fake), _concat_parts(sample_real)
    return (x_fake, x_real, _concat_parts(context_parts)) if x_fake.shape == x_real.shape else (None, None, None)


def _paired_relativistic_loss(pos: Tensor, neg: Tensor, *, detach_neg: bool = False) -> Tensor | None:
    batch_pos, batch_neg = pos.shape[0], neg.shape[0]
    if batch_pos != batch_neg or batch_pos < 1:
        return None
    pos_flat = format_logits_2d(pos, batch_pos).mean(1)
    neg_flat = format_logits_2d(neg.detach() if detach_neg else neg, batch_neg).mean(1)
    return F.softplus(neg_flat - pos_flat).mean()


def _can_mutate_record(record: AxiomStepRecord | None) -> bool:
    return torch.is_grad_enabled() and not (
        getattr(record, "_in_probe", False) or getattr(record, "_in_protected_probe", False)
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
        checked_weight = check_scalar(weight, "weight", lo=0)
        self.cond_ot_weight = check_scalar(cond_ot_weight, "cond_ot_weight", lo=0)
        self.fibre_rel_blend = check_scalar(fibre_rel_blend, "fibre_rel_blend", lo=0, hi=1)
        self.record: AxiomStepRecord | None = record
        self.transport = self.t1c = AxiomTransportLoss(weight=checked_weight, **transport_kw)
        self.last_metrics: dict[str, float] = {}

    @property
    def axiom_record(self) -> AxiomStepRecord | None:
        return self.record

    @axiom_record.setter
    def axiom_record(self, value: AxiomStepRecord | None) -> None:
        self.record = value

    def _resolve_capture_pair(self, fake_logits: Tensor, real_logits: Tensor):
        fake_entry = lookup_axiom_capture(fake_logits, self.record)
        hint_record = fake_entry.record if (fake_entry is not None and fake_entry.record is not None) else self.record
        real_entry = lookup_axiom_capture(real_logits, hint_record)
        resolved = hint_record if hint_record is not None else (real_entry.record if real_entry is not None else None)
        return fake_entry, real_entry, resolved

    def _conditional_g_loss(
        self, fake_logits: Tensor, real_logits: Tensor, fake_entry: Any, real_entry: Any, record: Any
    ) -> Tensor:
        weight = self.transport.weight
        rel = (
            _paired_relativistic_loss(fake_logits, real_logits, detach_neg=True)
            if fake_logits.shape[0] == real_logits.shape[0]
            else None
        )
        if rel is None:
            return weight * super().g_loss(fake_logits, real_logits)
        if self.cond_ot_weight <= 0 or not _optional_tensors_equal(fake_entry.cond, real_entry.cond):
            return weight * rel
        x_fake, x_real, context = _split_conditional_parts(fake_entry.parts, real_entry.parts)
        if x_fake is None or x_real is None or context is None:
            return weight * rel
        fiber = self.transport.conditional_fiber_loss(
            x_fake,
            x_real,
            context,
            cond_disc=fake_entry.cond,
            record=record,
            allow_state_update=_can_mutate_record(record),
        )
        if record is not None and hasattr(record, "_simplicial_step"):
            record._simplicial_step = False
        thermal_scale = float(getattr(record, "thermal_scale", 1.0)) if record is not None else 1.0
        total = rel + self.cond_ot_weight * fiber
        if thermal_scale < 1 and total.requires_grad:
            if fake_entry.thermally_scaled:
                total = rel + self.cond_ot_weight * (fiber * thermal_scale + fiber.detach() * (1 - thermal_scale))
            else:
                total = total * thermal_scale + total.detach() * (1 - thermal_scale)
        return weight * total

    def _invoke_transport(
        self, transport_fn: Any, fake_logits: Tensor, real_logits: Tensor, fake_entry: Any, real_entry: Any, record: Any
    ) -> Tensor:
        return transport_fn(
            fake_entry.x_in,
            real_entry.x_in,
            fake_logits,
            real_logits,
            record=record,
            cond_fake=fake_entry.cond,
            cond_real=real_entry.cond,
            allow_state_update=_can_mutate_record(record),
            thermally_scaled=fake_entry.thermally_scaled,
        )

    def d_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        """Critic loss with sync-free ``kappa_struct`` relativistic blend."""
        base = super().d_loss(real_logits, fake_logits)
        fake_entry, real_entry, record = self._resolve_capture_pair(fake_logits, real_logits)
        has_capture = fake_entry is not None and real_entry is not None
        has_cont_cond = has_capture and (fake_entry.has_cont_cond or real_entry.has_cont_cond)
        if record is not None and hasattr(record, "observe_margin_and_scale") and not has_cont_cond:
            record.observe_margin_and_scale(
                real_logits, fake_logits, *((real_entry.x_in, fake_entry.x_in) if has_capture else ())
            )
        if self.loss != "non_saturating" or self.transport.weight == 0 or self.fibre_rel_blend <= 0:
            return base
        pair = _paired_relativistic_loss(real_logits, fake_logits)
        if pair is None:
            return base
        if not has_capture or has_cont_cond:
            mix: float | Tensor = self.fibre_rel_blend
        else:
            _, kappa = self.transport._cached_contrast(real_entry.x_in)
            mix = (self.fibre_rel_blend * (1 - kappa)).to(base.dtype)
        return (1 - mix) * base + mix * pair

    def g_loss(self, fake_logits: Tensor, real_logits: Tensor | None = None) -> Tensor:
        """Generator loss dispatching conditional fiber or Law 3A transport."""
        weight = self.transport.weight
        if real_logits is None:
            if weight == 0:
                return super().g_loss(fake_logits, None)
            raise ValueError("real_logits required")
        if weight == 0:
            return super().g_loss(fake_logits, real_logits)
        fake_entry, real_entry, record = self._resolve_capture_pair(fake_logits, real_logits)
        if fake_entry is None or real_entry is None:
            base = super().g_loss(fake_logits, real_logits)
            pair = (
                _paired_relativistic_loss(fake_logits, real_logits, detach_neg=True)
                if (self.loss == "non_saturating" and self.fibre_rel_blend > 0)
                else None
            )
            blend = self.fibre_rel_blend
            return weight * (((1 - blend) * base + blend * pair) if pair is not None else base)
        if fake_entry.has_cont_cond or real_entry.has_cont_cond:
            return self._conditional_g_loss(fake_logits, real_logits, fake_entry, real_entry, record)
        return self._invoke_transport(self.transport, fake_logits, real_logits, fake_entry, real_entry, record)

    def joint_g_loss(self, fake_logits: Tensor, real_logits: Tensor) -> Tensor:
        """Joint BiGAN / ByteGAN generator+encoder transport loss."""
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires grads")
        if self.transport.weight == 0:
            return super().joint_g_loss(fake_logits, real_logits)
        fake_entry, real_entry, record = self._resolve_capture_pair(fake_logits, real_logits)
        if fake_entry is None or real_entry is None:
            return super().joint_g_loss(fake_logits, real_logits)
        return self._invoke_transport(
            self.transport.joint_relativistic_loss, fake_logits, real_logits, fake_entry, real_entry, record
        )


__all__ = ["AxiomGANLoss"]
