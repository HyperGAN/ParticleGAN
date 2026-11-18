"""Domain- and fiber-aware adversarial loss for Track 2 ``t2_secant``."""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from .._empf.loss import EMPFSinkhornRelativisticLoss


class T2SecantGANLoss(EMPFSinkhornRelativisticLoss):
    """Domain-aware adversarial objective paired with ``T2SecantCriticPenalty``."""

    def __init__(
        self,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        *,
        loss: str = "relativistic",
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        t2_record: Any = None,
    ) -> None:
        super().__init__(
            labels=labels,
            loss=loss,
            sinkhorn_eps=sinkhorn_eps,
            sinkhorn_iters=sinkhorn_iters,
            empf_record=t2_record,
        )
        self.t2_record = t2_record

    def _resolve_record(self, *tensors: torch.Tensor | None) -> Any:
        if self.t2_record is not None:
            return self.t2_record
        for t in tensors:
            if t is not None:
                rec = getattr(t, "_t2_record", getattr(t, "_empf_record", None))
                if rec is not None:
                    return rec
        return super()._resolve_record(*tensors)

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.loss != "relativistic":
            from ..gan_loss import GANLoss
            return GANLoss.d_loss(self, real_logits, fake_logits)
        return super().d_loss(real_logits, fake_logits)

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        if self.loss != "relativistic":
            from ..gan_loss import GANLoss
            return GANLoss.g_loss(self, fake_logits, real_logits)
        return super().g_loss(fake_logits, real_logits)


__all__ = ["T2SecantGANLoss"]
