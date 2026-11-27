"""Axiom 4-Law GANLoss adapter composing AxiomTransportLoss & AxiomCondFiberBank."""
from __future__ import annotations

import math, weakref
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .transport import AxiomTransportLoss, project_transport_features

_SPATIAL_CLUSTER_THRESH = 0.12


@torch.no_grad()
def _spatial_cluster_ratio(x: torch.Tensor, *, max_spatial: int = 8, max_b: int = 64) -> float:
    b = int(x.shape[0])
    if b < 4: return 1.0
    xd = x.detach()
    z = project_transport_features(xd - xd.mean(dim=0, keepdim=True), max_spatial=max_spatial).float()
    sc = z.abs().amax(); zu = z / sc.clamp_min(1e-30)
    if b > max_b:
        w = torch.remainder((torch.arange(zu.shape[1], device=zu.device, dtype=torch.float32) + 0.5) * 0.6180339887498949, 1.0) - 0.5
        order = torch.argsort(zu @ (w / w.norm().clamp_min(1e-12)), stable=True)
        zu = zu.index_select(0, order.index_select(0, (torch.arange(max_b, device=zu.device, dtype=torch.long) * b) // max_b))
    dmat, off = torch.cdist(zu, zu, p=2.0), ~torch.eye(int(zu.shape[0]), dtype=torch.bool, device=zu.device)
    med_all = torch.quantile(dmat[off], 0.50).clamp_min(1e-12)
    med_nn = torch.quantile(dmat.masked_fill(~off, float("inf")).amin(dim=1), 0.50)
    return float(torch.where(sc > 1e-30, med_nn / med_all, med_nn.new_zeros(())).item())


def _lookup_capture(
    logits: torch.Tensor, fallback_record: Any = None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Any, bool, Optional[Tuple[torch.Tensor, ...]]]:
    rec = getattr(logits, "_axiom_record", None)
    rec_use = fallback_record if rec is None else rec
    x_in = getattr(logits, "_axiom_input", None)
    if x_in is not None:
        return x_in, getattr(logits, "_axiom_cond", None), rec_use, bool(getattr(logits, "_axiom_cont_cond", False)), getattr(logits, "_axiom_parts", None)
    if rec_use is not None and hasattr(rec_use, "lookup_capture"):
        hit = rec_use.lookup_capture(logits)
        if hit is not None:
            if len(hit) == 5:
                x_h, c_h, r_h, cc_h, p_h = hit
                return x_h, c_h, (rec_use if r_h is None else r_h), bool(cc_h), p_h
            x_h, c_h, cc_h, p_h = hit
            return x_h, c_h, rec_use, bool(cc_h), p_h
    return None, None, rec_use, False, None


def _split_shared_conditional_parts(
    parts_f: Optional[Tuple[torch.Tensor, ...]],
    parts_r: Optional[Tuple[torch.Tensor, ...]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not parts_f or not parts_r or len(parts_f) != len(parts_r) or len(parts_f) < 2:
        return None, None, None
    b_n = int(parts_f[0].shape[0])
    if b_n <= 0 or int(parts_r[0].shape[0]) != b_n:
        return None, None, None
    sample_f, sample_r, ctx = [], [], []
    for pf, pr in zip(parts_f, parts_r):
        if not pf.requires_grad and pf.shape == pr.shape and pf.dtype == pr.dtype and torch.equal(pf.detach(), pr.detach()):
            ctx.append(pf.detach())
        else:
            sample_f.append(pf)
            sample_r.append(pr)
    if not ctx or not sample_f:
        return None, None, None
    x_sf = sample_f[0] if len(sample_f) == 1 else torch.cat(sample_f, dim=-1)
    x_sr = sample_r[0] if len(sample_r) == 1 else torch.cat(sample_r, dim=-1)
    c_cont = ctx[0] if len(ctx) == 1 else torch.cat(ctx, dim=-1)
    return (x_sf, x_sr, c_cont) if x_sf.shape == x_sr.shape else (None, None, None)


def _scalar_pair_rel(pos: torch.Tensor, neg: torch.Tensor, *, detach_neg: bool = False) -> Optional[torch.Tensor]:
    b_p, b_n = int(pos.shape[0]), int(neg.shape[0])
    if b_p != b_n or b_p < 1:
        return None
    p_flat = pos.reshape(b_p, -1).mean(dim=1) if pos.ndim > 1 else pos.reshape(b_p)
    n_src = neg.detach() if detach_neg else neg
    n_flat = n_src.reshape(b_n, -1).mean(dim=1) if n_src.ndim > 1 else n_src.reshape(b_n)
    return F.softplus(n_flat - p_flat).mean()


class AxiomGANLoss(GANLoss):
    """Unified Axiom 4-Law ``GANLoss`` supporting GAN, BiGAN, AE/VAE-GAN, DDGAN & ByteGAN."""

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
        bank_mean_blend: float = 0.85,
        record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        for name, val in (("weight", weight), ("cond_ot_weight", cond_ot_weight)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(fibre_rel_blend, bool) or not isinstance(fibre_rel_blend, (int, float)) or not math.isfinite(fibre_rel_blend) or not (0.0 <= float(fibre_rel_blend) <= 1.0):
            raise ValueError("fibre_rel_blend must be finite and in [0, 1]")
        self.fibre_rel_blend, self.cond_ot_weight, self.record = float(fibre_rel_blend), float(cond_ot_weight), record
        self.transport = AxiomTransportLoss(
            weight=float(weight), rho=float(rho), tau=float(tau), eps=float(eps), iters=int(iters),
            chart_capacity=int(chart_capacity), chart_weight=float(chart_weight), halo_mult=float(halo_mult),
            max_spatial=int(max_spatial), mean_op_weight=float(mean_op_weight), cov_op_weight=float(cov_op_weight),
            imq_active_scale=float(imq_active_scale), bank_mean_blend=float(bank_mean_blend),
        )
        self.t1c, self.last_metrics, self._sc_cache = self.transport, {}, None

    def _skip_spatial_transport(self, x_r: torch.Tensor) -> bool:
        if x_r.ndim < 2 or x_r.shape[0] <= 0: return False
        dim = int(x_r[0].numel())
        if x_r.ndim < 4 or not any(int(s) > 1 for s in x_r.shape[2:]): return dim > 1024
        if dim <= 1024 or dim > 4096: return dim > 4096
        key, c = (id(x_r), int(x_r.data_ptr()), int(x_r._version), tuple(x_r.shape), str(x_r.device)), self._sc_cache
        if c is not None and c[0] == key and c[2]() is x_r: return c[1]
        skip = _spatial_cluster_ratio(x_r, max_spatial=8) >= _SPATIAL_CLUSTER_THRESH
        self._sc_cache = (key, skip, weakref.ref(x_r)); return skip

    @property
    def axiom_record(self) -> Any: return self.record
    @axiom_record.setter
    def axiom_record(self, value: Any) -> None: self.record = value

    def _resolve_pair(self, fake_logits: torch.Tensor, real_logits: torch.Tensor):
        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_capture(fake_logits, self.record)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_capture(real_logits, rec_f or self.record)
        rec = rec_f if rec_f is not None else (rec_r if rec_r is not None else self.record)
        if x_f is None or x_r is None:
            return fake_logits.reshape(fake_logits.shape[0], -1), real_logits.reshape(real_logits.shape[0], -1), None, None, rec, False, None, None, False
        return x_f, x_r, cond_f, cond_r, rec, bool(cc_f or cc_r), parts_f, parts_r, True

    def _blended_g(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        base_g = super().g_loss(fake_logits, real_logits)
        if self.loss == "non_saturating" and self.fibre_rel_blend > 0.0:
            pair_g = _scalar_pair_rel(fake_logits, real_logits, detach_neg=True)
            if pair_g is not None:
                return self.transport.weight * ((1.0 - self.fibre_rel_blend) * base_g + self.fibre_rel_blend * pair_g)
        return self.transport.weight * base_g

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        base = super().d_loss(real_logits, fake_logits)
        x_f, x_r, _, _, rec, has_cc, _, _, captured = self._resolve_pair(fake_logits, real_logits)
        if rec is not None and hasattr(rec, "observe_margin_and_scale") and not has_cc:
            if captured:
                rec.observe_margin_and_scale(real_logits, fake_logits, x_r, x_f)
            else:
                rec.observe_margin_and_scale(real_logits, fake_logits)
        if self.loss != "non_saturating" or self.transport.weight == 0.0 or self.fibre_rel_blend <= 0.0:
            return base
        if (not captured) or has_cc or self._skip_spatial_transport(x_r):
            pair_d = _scalar_pair_rel(real_logits, fake_logits)
            if pair_d is not None:
                return (1.0 - self.fibre_rel_blend) * base + self.fibre_rel_blend * pair_d
        return base

    def g_loss(self, fake_logits: torch.Tensor, real_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        if real_logits is None:
            if self.transport.weight == 0.0:
                return super().g_loss(fake_logits, None)
            raise ValueError("AxiomGANLoss requires real_logits in g_loss when transport weight > 0")
        if self.transport.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, has_cc, parts_f, parts_r, captured = self._resolve_pair(fake_logits, real_logits)
        if not captured:
            return self._blended_g(fake_logits, real_logits)
        b_f, b_r = int(fake_logits.shape[0]), int(real_logits.shape[0])
        if has_cc:
            if b_f != b_r:
                return self.transport.weight * super().g_loss(fake_logits, real_logits)
            rel_term = _scalar_pair_rel(fake_logits, real_logits, detach_neg=True)
            assert rel_term is not None
            same_disc = (cond_f is None and cond_r is None) or (cond_f is not None and cond_r is not None and cond_f.shape == cond_r.shape and torch.equal(cond_f, cond_r))
            if self.cond_ot_weight > 0.0 and same_disc:
                x_sf, x_sr, c_cont = _split_shared_conditional_parts(parts_f, parts_r)
                if x_sf is not None and x_sr is not None and c_cont is not None:
                    cond_ot = self.transport.conditional_fiber_loss(x_sf, x_sr, c_cont, cond_disc=cond_f, record=rec)
                    if rec is not None and hasattr(rec, "_simplicial_step"):
                        rec._simplicial_step = False
                    s_th = float(getattr(rec, "thermal_scale", 1.0)) if rec is not None else 1.0
                    total = rel_term + self.cond_ot_weight * cond_ot
                    if s_th < 1.0 and total.requires_grad:
                        if getattr(fake_logits, "_axiom_thermally_scaled", False):
                            total = rel_term + self.cond_ot_weight * (cond_ot * s_th + cond_ot.detach() * (1.0 - s_th))
                        else:
                            total = total * s_th + total.detach() * (1.0 - s_th)
                    return self.transport.weight * total
            return self.transport.weight * rel_term
        if self._skip_spatial_transport(x_r):
            self.transport.bind_discrete_prior_step(False, simplicial_active=False)
            if rec is not None:
                if hasattr(rec, "_disc_gate"): rec._disc_gate = 0.0
                if hasattr(rec, "_simplicial_step"): rec._simplicial_step = False
            return self._blended_g(fake_logits, real_logits)
        allow = bool(torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False) and not getattr(rec, "_in_probe", False))
        prev_simp = bool(self.transport._simplicial_step)
        if prev_simp and b_r >= 4:
            _, g_mac_fwd, _, _ = self.transport._cached_manifold_gates(project_transport_features(x_r.detach(), self.transport.max_spatial))
            eff_simp = bool(g_mac_fwd > 1e-4)
            self.transport._simplicial_step = eff_simp
            if rec is not None and hasattr(rec, "_simplicial_step"):
                rec._simplicial_step = eff_simp
        try:
            return self.transport(x_f, x_r, fake_logits, real_logits, record=rec, cond_fake=cond_f, cond_real=cond_r, allow_state_update=allow)
        finally:
            self.transport._simplicial_step = prev_simp

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        if self.transport.weight == 0.0:
            return super().joint_g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, _, _, _, captured = self._resolve_pair(fake_logits, real_logits)
        if not captured:
            return super().joint_g_loss(fake_logits, real_logits)
        allow = bool(torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False) and not getattr(rec, "_in_probe", False))
        return self.transport.joint_relativistic_loss(x_f, x_r, fake_logits, real_logits, record=rec, cond_fake=cond_f, cond_real=cond_r, allow_state_update=allow)


__all__ = ["AxiomGANLoss"]
