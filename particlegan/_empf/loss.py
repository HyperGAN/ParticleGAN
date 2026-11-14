"""Domain-aware Sinkhorn & fiber-paired adversarial loss for EMPF."""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .transport import balanced_sinkhorn_coupling, effective_sinkhorn_eps


class EMPFSinkhornRelativisticLoss(GANLoss):
    """Domain-aware EMPF adversarial objective compatible with ``GANLoss``.

    1. Row-aligned conditioned fibers (``_empf_conditioned == True``): pairs
       ``real_logits[i] - fake_logits[i]`` on the shared conditioning context
       so arbitrary baseline offsets ``b(cond[i])`` cancel exactly.
    2. Unconditioned low-dimensional continuous clouds (``1 <= dim <= 4``,
       ``batch >= 32``): pairs real and fake logits locally via state-settling
       doubly-stochastic Sinkhorn coupling ``Pi`` so each mode compares against
       generated samples inside its own Voronoi basin.
    3. Discrete simplexes, images, and general hosts: preserves ``GANLoss``
       (``non_saturating`` or centered ``relativistic``) identically.
    """

    def __init__(
        self,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        *,
        loss: str = "relativistic",
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        empf_record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.empf_record = empf_record

    def _resolve_record(self, *tensors: torch.Tensor | None) -> Any:
        if self.empf_record is not None:
            return self.empf_record
        for t in tensors:
            if t is not None:
                rec = getattr(t, "_empf_record", None)
                if rec is not None:
                    return rec
        return None

    @staticmethod
    def _same_continuous_fiber(t1: torch.Tensor | None, t2: torch.Tensor | None) -> bool:
        if t1 is None or t2 is None:
            return False
        c1 = getattr(t1, "_empf_cond_tensors", ())
        c2 = getattr(t2, "_empf_cond_tensors", ())
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
        # Support callers that pass .detach() on the reference branch (e.g. d_loss(r, f.detach())
        # or g_loss(f, r.detach())), which strips custom attributes on the detached wrapper.
        c_live = c1 or c2
        if not c_live:
            return False
        if hasattr(t1, "_empf_conditioned") and hasattr(t2, "_empf_conditioned"):
            return False
        return any(isinstance(a, torch.Tensor) and a.is_floating_point() for a in c_live)

    def _spatial_inputs(
        self,
        real_logits: torch.Tensor | None,
        fake_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if real_logits is None:
            return None
        x_r = getattr(real_logits, "_empf_input", None)
        x_f = getattr(fake_logits, "_empf_input", None)
        if not isinstance(x_r, torch.Tensor) or not isinstance(x_f, torch.Tensor):
            return None
        if x_r.ndim != 2 or x_f.ndim != 2 or x_r.shape[1] != x_f.shape[1]:
            return None
        dim = x_r.shape[1]
        b_r, b_f = x_r.shape[0], x_f.shape[0]
        if not (1 <= dim <= 4 and b_r >= 32 and b_f >= 32):
            return None
        if b_r != real_logits.reshape(-1).shape[0] or b_f != fake_logits.reshape(-1).shape[0]:
            return None
        return x_r, x_f

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        if self._same_continuous_fiber(real_logits, fake_logits) and r.shape == f.shape:
            return F.softplus(-(r - f)).mean()
        if self.loss == "relativistic":
            spatial = self._spatial_inputs(real_logits, fake_logits)
            if spatial is not None:
                x_r, x_f = spatial
                rec = self._resolve_record(fake_logits, real_logits)
                ema_res = float(getattr(rec, "ema_residual", 1.0)) if rec is not None else 1.0
                eps_eff = effective_sinkhorn_eps(ema_res, self.sinkhorn_eps)
                pi = balanced_sinkhorn_coupling(x_r, x_f, sinkhorn_eps=eps_eff, sinkhorn_iters=self.sinkhorn_iters)
                delta = r.unsqueeze(1) - f.unsqueeze(0)
                return (float(r.shape[0]) * pi * F.softplus(-delta)).sum()
            return 0.5 * (F.softplus(-(r - f.mean())).mean() + F.softplus(f - r.mean()).mean())
        return super().d_loss(real_logits, fake_logits)

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        f = fake_logits.reshape(-1)
        if real_logits is not None:
            r = real_logits.reshape(-1)
            if self._same_continuous_fiber(fake_logits, real_logits) and r.shape == f.shape:
                return F.softplus(-(f - r)).mean()
            if self.loss == "relativistic":
                spatial = self._spatial_inputs(real_logits, fake_logits)
                if spatial is not None:
                    x_r, x_f = spatial
                    rec = self._resolve_record(fake_logits, real_logits)
                    ema_res = float(getattr(rec, "ema_residual", 1.0)) if rec is not None else 1.0
                    eps_eff = effective_sinkhorn_eps(ema_res, self.sinkhorn_eps)
                    pi = balanced_sinkhorn_coupling(x_r, x_f, sinkhorn_eps=eps_eff, sinkhorn_iters=self.sinkhorn_iters)
                    delta = r.unsqueeze(1) - f.unsqueeze(0)
                    return (float(r.shape[0]) * pi * F.softplus(delta)).sum()
                return 0.5 * (F.softplus(-(f - r.mean())).mean() + F.softplus(r - f.mean()).mean())
        elif self.loss == "relativistic":
            return F.softplus(-f).mean()
        return super().g_loss(fake_logits, real_logits)
