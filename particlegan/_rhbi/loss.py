"""Relativistic Hellinger-Born-Infeld (RHBI) GAN loss adapter."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .critic import _lookup_rhbi_capture_parts
from .primitive import RHBILoss


def _split_shared_conditional_parts(
    parts_f: Optional[Tuple[torch.Tensor, ...]],
    parts_r: Optional[Tuple[torch.Tensor, ...]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Separate sample tensors from shared continuous context tensors in multi-arg critic tuples."""
    if not parts_f or not parts_r or len(parts_f) != len(parts_r) or len(parts_f) < 2:
        return None, None, None
    b_n = parts_f[0].shape[0]
    if b_n <= 0 or parts_r[0].shape[0] != b_n:
        return None, None, None
    sample_f = []
    sample_r = []
    ctx_parts = []
    for pf, pr in zip(parts_f, parts_r):
        if (
            not pf.requires_grad
            and pf.shape == pr.shape
            and pf.dtype == pr.dtype
            and torch.equal(pf.detach(), pr.detach())
        ):
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


class RHBIGANLoss(GANLoss):
    """``GANLoss`` subclass pairing calibrated log-odds critic training with ``RHBILoss``."""

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        weight: float = 1.0,
        rho: float = 2.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        critic_weight: float = 0.0,
        self_coupling: str = "hollow",
        relativistic_weight: float = 0.0,
        rapidity_screening: bool = True,
        scale_log_odds: bool = False,
        conditional_fibre: str = "joint",
        akt_horizon: bool = False,
        akt_mode: str = "sqrt",
        slack_gain: float = 0.0,
        thermal_tau: float = 0.0,
        thermal_shock_mu: float = 0.15,
        fibre_rel_blend: float = 0.0,
        cond_ot_weight: float = 0.0,
        cond_ot_mode: str = "kernel_ed",
        cvx_mu_gain: float = 0.40,
        tail_spill_gain: float = 0.0,
        simplex_macro_settle: bool = False,
        record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        if (
            isinstance(critic_weight, bool)
            or not isinstance(critic_weight, (int, float))
            or not math.isfinite(critic_weight)
            or critic_weight < 0.0
        ):
            raise ValueError("critic_weight must be finite and nonnegative")
        if (
            isinstance(fibre_rel_blend, bool)
            or not isinstance(fibre_rel_blend, (int, float))
            or not math.isfinite(fibre_rel_blend)
            or fibre_rel_blend < 0.0
            or fibre_rel_blend > 1.0
        ):
            raise ValueError("fibre_rel_blend must be finite and in [0, 1]")
        if (
            isinstance(cond_ot_weight, bool)
            or not isinstance(cond_ot_weight, (int, float))
            or not math.isfinite(cond_ot_weight)
            or cond_ot_weight < 0.0
        ):
            raise ValueError("cond_ot_weight must be finite and nonnegative")
        if type(cond_ot_mode) is not str or cond_ot_mode not in ("kernel_ed", "hollow_sinkhorn", "imq_cmmd"):
            raise ValueError("cond_ot_mode must be 'kernel_ed', 'hollow_sinkhorn' or 'imq_cmmd'")
        if type(conditional_fibre) is not str or conditional_fibre not in ("joint", "slack", "calibrated"):
            raise ValueError("conditional_fibre must be 'joint', 'slack', or 'calibrated'")
        self.critic_weight = float(critic_weight)
        self.conditional_fibre = conditional_fibre
        self.fibre_rel_blend = float(fibre_rel_blend)
        self.cond_ot_weight = float(cond_ot_weight)
        self.cond_ot_mode = cond_ot_mode
        self.record = record
        self.rhbi = RHBILoss(
            weight=weight,
            rho=rho,
            tau=tau,
            eps=eps,
            iters=iters,
            c_max=c_max,
            beta_ell=beta_ell,
            nn_quantile=nn_quantile,
            rapidity_screening=rapidity_screening,
            self_coupling=self_coupling,
            relativistic_weight=relativistic_weight,
            scale_log_odds=scale_log_odds,
            akt_horizon=akt_horizon,
            akt_mode=akt_mode,
            slack_gain=slack_gain,
            thermal_tau=thermal_tau,
            thermal_shock_mu=thermal_shock_mu,
            cvx_mu_gain=cvx_mu_gain,
            tail_spill_gain=tail_spill_gain,
            simplex_macro_settle=simplex_macro_settle,
        )
        self.last_metrics: Dict[str, float] = {}

    def _resolve_pair_parts(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Any,
        Any,
        Any,
        bool,
        Optional[Tuple[torch.Tensor, ...]],
        Optional[Tuple[torch.Tensor, ...]],
    ]:
        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_rhbi_capture_parts(fake_logits, self.record)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_rhbi_capture_parts(
            real_logits, rec_f if rec_f is not None else self.record
        )
        rec = rec_f if rec_f is not None else (rec_r if rec_r is not None else self.record)
        has_cont_cond = bool(cc_f or cc_r)
        if x_f is None or x_r is None:
            x_f = fake_logits.reshape(fake_logits.shape[0], -1)
            x_r = real_logits.reshape(real_logits.shape[0], -1)
            cond_f = None
            cond_r = None
            has_cont_cond = False
            parts_f = None
            parts_r = None
        return x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r

    def _resolve_pair_full(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, bool]:
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, _, _ = self._resolve_pair_parts(fake_logits, real_logits)
        return x_f, x_r, cond_f, cond_r, rec, has_cont_cond

    def _resolve_pair(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any]:
        x_f, x_r, cond_f, cond_r, rec, _ = self._resolve_pair_full(fake_logits, real_logits)
        return x_f, x_r, cond_f, cond_r, rec

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        base = super().d_loss(real_logits, fake_logits)
        if self.critic_weight == 0.0 and self.fibre_rel_blend == 0.0:
            return base
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond = self._resolve_pair_full(fake_logits, real_logits)
        if self.conditional_fibre == "slack" and has_cont_cond:
            if self.fibre_rel_blend > 0.0:
                b_r = real_logits.shape[0]
                b_f = fake_logits.shape[0]
                if b_r == b_f:
                    r_flat = (
                        real_logits.reshape(b_r, -1).mean(dim=1)
                        if real_logits.ndim > 1
                        else real_logits.reshape(b_r)
                    )
                    f_flat = (
                        fake_logits.reshape(b_f, -1).mean(dim=1)
                        if fake_logits.ndim > 1
                        else fake_logits.reshape(b_f)
                    )
                    pair_d = F.softplus(f_flat - r_flat).mean()
                    return (1.0 - self.fibre_rel_blend) * base + self.fibre_rel_blend * pair_d
            return base
        if self.conditional_fibre == "calibrated" and has_cont_cond:
            return base
        if self.critic_weight == 0.0:
            return base
        slack_reg = self.rhbi.critic_slack_loss(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            critic_weight=self.critic_weight,
            record=rec,
            cond_fake=cond_f,
            cond_real=cond_r,
        )
        return base + slack_reg

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        if real_logits is None:
            if self.rhbi.weight == 0.0:
                return super().g_loss(fake_logits, None)
            raise ValueError("RHBIGANLoss requires real_logits in g_loss when rhbi_weight > 0")
        if self.rhbi.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if self.conditional_fibre == "calibrated" and has_cont_cond:
            return self.rhbi.weight * super().g_loss(fake_logits, real_logits)
        if self.conditional_fibre == "slack" and has_cont_cond:
            b_f = fake_logits.shape[0]
            b_r = real_logits.shape[0]
            if b_f == b_r:
                f_flat = fake_logits.reshape(b_f, -1).mean(dim=1) if fake_logits.ndim > 1 else fake_logits.reshape(b_f)
                r_flat = (
                    real_logits.detach().reshape(b_r, -1).mean(dim=1)
                    if real_logits.ndim > 1
                    else real_logits.detach().reshape(b_r)
                )
                rel_term = F.softplus(r_flat - f_flat).mean()
                if self.cond_ot_weight > 0.0:
                    same_disc = (cond_f is None and cond_r is None) or (
                        cond_f is not None
                        and cond_r is not None
                        and cond_f.shape == cond_r.shape
                        and torch.equal(cond_f, cond_r)
                    )
                    if same_disc:
                        x_sf, x_sr, c_cont = _split_shared_conditional_parts(parts_f, parts_r)
                        if x_sf is not None and x_sr is not None and c_cont is not None:
                            cond_ot = self.rhbi.conditional_fiber_transport_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
                                mode=self.cond_ot_mode,
                            )
                            s_th = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
                            total_cond = rel_term + self.cond_ot_weight * cond_ot
                            if s_th < 1.0 and total_cond.requires_grad:
                                if getattr(fake_logits, "_rhbi_thermally_scaled", False):
                                    cond_scaled = cond_ot * s_th + cond_ot.detach() * (1.0 - s_th)
                                    total_cond = rel_term + self.cond_ot_weight * cond_scaled
                                else:
                                    total_cond = total_cond * s_th + total_cond.detach() * (1.0 - s_th)
                            return self.rhbi.weight * total_cond
                return self.rhbi.weight * rel_term
            return self.rhbi.weight * super().g_loss(fake_logits, real_logits)
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        loss_val, _ = self.rhbi(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            record=rec,
            symmetric=False,
            allow_ell_update=allow_ell,
            cond_fake=cond_f,
            cond_real=cond_r,
            return_metrics=False,
        )
        return loss_val

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        if self.rhbi.weight == 0.0:
            return super().joint_g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, _rec, _has_cont_cond = self._resolve_pair_full(fake_logits, real_logits)
        allow_ell = torch.is_grad_enabled() and not getattr(_rec, "_in_protected_probe", False)
        loss_val, _ = self.rhbi.joint_relativistic_loss(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            record=_rec,
            allow_ell_update=allow_ell,
            cond_fake=cond_f,
            cond_real=cond_r,
            return_metrics=False,
        )
        return loss_val
