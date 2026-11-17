"""Scale- & distortion-aware SMPF adversarial + s-means prototype transport loss."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .critic import _lookup_smpf_capture, read_smpf_thermal
from .transport import smeans_conditional_fiber_loss, smeans_debiased_generator_loss


def _same_tensor_fast(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Zero-sync fast path when conditioned critic pairs share the exact context tensor."""
    if a is b:
        return True
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    if (
        a.data_ptr() == b.data_ptr()
        and a.storage_offset() == b.storage_offset()
        and a.stride() == b.stride()
    ):
        return True
    return bool(torch.equal(a.detach(), b.detach()))


def _split_shared_conditional_parts(
    parts_f: Optional[Tuple[torch.Tensor, ...]],
    parts_r: Optional[Tuple[torch.Tensor, ...]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Separate sample tensors from shared continuous context tensors in conditioned critic calls."""
    if not parts_f or not parts_r or len(parts_f) != len(parts_r) or len(parts_f) < 2:
        return None, None, None
    b_n = int(parts_f[0].shape[0])
    if b_n <= 0 or parts_r[0].shape[0] != b_n:
        return None, None, None
    sample_f = []
    sample_r = []
    ctx_parts = []
    for pf, pr in zip(parts_f, parts_r):
        if not pf.requires_grad and _same_tensor_fast(pf, pr):
            ctx_parts.append(pf.detach())
        else:
            sample_f.append(pf)
            sample_r.append(pr)
    if not ctx_parts or not sample_f:
        return None, None, None
    x_sf = sample_f[0] if len(sample_f) == 1 else torch.cat(sample_f, dim=-1)
    x_sr = sample_r[0] if len(sample_r) == 1 else torch.cat(sample_r, dim=-1)
    c_cont = ctx_parts[0] if len(ctx_parts) == 1 else torch.cat(ctx_parts, dim=-1)
    if x_sf.shape != x_sr.shape:
        return None, None, None
    return x_sf, x_sr, c_cont


class SMPFGANLoss(GANLoss):
    """``GANLoss`` subclass pairing calibrated critic log-odds with s-means prototype transport.

    1. Conditioned continuous fibers (e.g. DDGAN / AE-DDGAN / student heads):
       pairs row-local relativistic log-odds on the shared context (surviving
       ``.detach()`` via the instance-local capture ring) with affine-detrended,
       coordinate-whitened multi-scale IMQ conditional fiber transport.
    2. Unconditioned vector and procedural-image domains (``1 <= dim <= 169``,
       or image tensors up to ``dim <= 256``, ``batch >= 16``): adds thermally
       matched, debiased physical Born-Infeld s-means prototype Sinkhorn
       divergence + per-prototype RMS distortion shell regularization.
    3. Discrete simplexes (``dim >= 170``), joint multi-stream BiGAN tuples,
       and tiny behavioral batches (``batch < 16``): preserves calibrated
       ``GANLoss`` identically.
    """

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        proto_weight: float = 0.85,
        cond_weight: float = 1.35,
        shell_weight: float = 0.25,
        sinkhorn_eps: float = 0.08,
        sinkhorn_iters: int = 7,
        smpf_record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        self.proto_weight = float(proto_weight)
        self.cond_weight = float(cond_weight)
        self.shell_weight = float(shell_weight)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.smpf_record = smpf_record

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        base = super().d_loss(real_logits, fake_logits)
        x_r, rec, is_cond_r, _, cd_r, parts_r, _ = _lookup_smpf_capture(real_logits, self.smpf_record)
        x_f, _, is_cond_f, _, cd_f, parts_f, _ = _lookup_smpf_capture(fake_logits, rec)
        if is_cond_r and is_cond_f and parts_r is not None and parts_f is not None:
            # Row-aligned continuous fiber critic pairing cancels arbitrary context baseline offsets b(c).
            same_disc = (cd_r is None and cd_f is None) or (
                cd_r is not None and cd_f is not None and _same_tensor_fast(cd_r, cd_f)
            )
            if same_disc and real_logits.shape == fake_logits.shape:
                x_sf, x_sr, c_cont = _split_shared_conditional_parts(parts_f, parts_r)
                if x_sf is not None and x_sr is not None and c_cont is not None:
                    r_flat = real_logits.reshape(-1)
                    f_flat = fake_logits.reshape(-1)
                    pair_d = F.softplus(-(r_flat - f_flat)).mean()
                    return 0.5 * base + 0.5 * pair_d
        return base

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        base = super().g_loss(fake_logits, real_logits)
        if real_logits is None:
            return base

        x_f, rec, is_cond_f, _, cd_f, parts_f, f_therm_scaled = _lookup_smpf_capture(fake_logits, self.smpf_record)
        x_r, rec_r, is_cond_r, _, cd_r, parts_r, _ = _lookup_smpf_capture(
            real_logits, rec if rec is not None else self.smpf_record
        )
        if rec is None:
            rec = rec_r

        # 1. Shared continuous-conditioned fiber (e.g. DDGAN / AE-DDGAN / conditioned residual).
        if is_cond_f and is_cond_r and parts_f is not None and parts_r is not None:
            same_disc = (cd_f is None and cd_r is None) or (
                cd_f is not None and cd_r is not None and _same_tensor_fast(cd_f, cd_r)
            )
            if same_disc and fake_logits.shape[0] == real_logits.shape[0]:
                x_sf, x_sr, c_cont = _split_shared_conditional_parts(parts_f, parts_r)
                if x_sf is not None and x_sr is not None and c_cont is not None and x_sf.requires_grad:
                    b_f = int(fake_logits.shape[0])
                    f_flat = fake_logits.reshape(b_f, -1).mean(dim=1) if fake_logits.ndim > 1 else fake_logits.reshape(b_f)
                    r_flat = (
                        real_logits.detach().reshape(b_f, -1).mean(dim=1)
                        if real_logits.ndim > 1
                        else real_logits.detach().reshape(b_f)
                    )
                    rel_adv = F.softplus(-(f_flat - r_flat)).mean()
                    if self.cond_weight > 0.0 and b_f >= 32 and 1 <= int(x_sf.shape[1]) <= 16:
                        cond_ot = smeans_conditional_fiber_loss(
                            x_sf,
                            x_sr,
                            c_cont,
                            cond_disc=cd_f,
                        )
                        s_th = read_smpf_thermal(rec, default=1.0)
                        if s_th < 1.0 and cond_ot.requires_grad:
                            cond_ot = cond_ot * s_th + cond_ot.detach() * (1.0 - s_th)
                        return rel_adv + self.cond_weight * cond_ot
                    return rel_adv
            return base

        # 2. Unconditioned vector or procedural-image domain with active s-means prototype bank.
        if (
            self.proto_weight <= 0.0
            or is_cond_f
            or is_cond_r
            or x_f is None
            or not isinstance(x_f, torch.Tensor)
            or not x_f.requires_grad
            or rec is None
            or not hasattr(rec, "readout_eb_prototypes")
        ):
            return base

        b_f = int(x_f.shape[0])
        if b_f < 16:
            return base
        dim = max(1, int(x_f[0].numel()))
        is_image = x_f.ndim >= 3
        if (not is_image and dim >= 170) or (is_image and dim > 256):
            return base

        eb = rec.readout_eb_prototypes(dtype=x_f.dtype, device=x_f.device)
        if eb is None:
            return base
        c_eb, weights, ell_proto, _ = eb
        if int(c_eb.shape[1]) != dim:
            return base

        discrete_prior = bool(getattr(rec, "_discrete_prior_step", False))
        phase = float(getattr(rec, "chord_phase", 0.0))
        ot_loss = smeans_debiased_generator_loss(
            x_f,
            c_eb,
            weights,
            ell_proto,
            sinkhorn_eps=self.sinkhorn_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            shell_weight=self.shell_weight,
            discrete_prior=discrete_prior,
            phase=phase,
        )
        s_th = read_smpf_thermal(rec, default=1.0)
        if s_th < 1.0 and ot_loss.requires_grad:
            ot_loss = ot_loss * s_th + ot_loss.detach() * (1.0 - s_th)

        # Shift-invariant centered generator log-odds prevents mean-logit bias from pulling modes off center.
        f_vec = fake_logits.reshape(-1)
        r_vec = real_logits.detach().reshape(-1)
        r_bar = r_vec.float().mean().to(dtype=f_vec.dtype)
        if not f_therm_scaled and s_th < 1.0 and f_vec.requires_grad:
            f_eff = f_vec * s_th + f_vec.detach() * (1.0 - s_th)
        else:
            f_eff = f_vec
        adv_centered = F.softplus(-(f_eff - r_bar)).mean() if self.loss == "non_saturating" else base
        return adv_centered + self.proto_weight * ot_loss
