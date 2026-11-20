"""Track 1B Deconflicted Piecewise Coupling GANLoss adapter."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .._rhbi.loss import _split_shared_conditional_parts
from ..gan_loss import GANLoss
from .critic import _lookup_t1c_capture_parts
from .primitive import T1CouplingLoss


class T1CouplingGANLoss(GANLoss):
    """``GANLoss`` subclass pairing calibrated log-odds critic training with ``T1CouplingLoss``."""

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        weight: float = 1.0,
        rho: float = 3.0,
        tau: float = 2.0,
        eps: float = 0.05,
        iters: int = 8,
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        critic_weight: float = 0.0,
        self_coupling: str = "feydy_discrete",
        relativistic_weight: float = 0.5,
        rapidity_screening: bool = True,
        scale_log_odds: bool = True,
        conditional_fibre: str = "slack",
        akt_horizon: bool = True,
        akt_mode: str = "weibull",
        slack_gain: float = 2.5,
        thermal_tau: float = 0.0,
        thermal_shock_mu: float = 0.15,
        fibre_rel_blend: float = 0.65,
        cond_ot_weight: float = 3.0,
        cond_ot_mode: str = "diag_imq_cmmd",
        cvx_mu_gain: float = 0.80,
        tail_spill_gain: float = 1.20,
        simplex_macro_settle: bool = True,
        simplex_curve_settle: bool = True,
        support_completion: bool = True,
        chart_capacity: int = 256,
        chart_weight: float = 0.0,
        halo_mult: float = 2.1,
        multiscale_horizon: bool = False,
        cond_drift_deconflict: bool = False,
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
        if type(cond_ot_mode) is not str or cond_ot_mode not in ("imq_cmmd", "diag_imq_cmmd"):
            raise ValueError("cond_ot_mode must be 'imq_cmmd' or 'diag_imq_cmmd'")
        if type(conditional_fibre) is not str or conditional_fibre not in ("joint", "slack", "calibrated"):
            raise ValueError("conditional_fibre must be 'joint', 'slack', or 'calibrated'")

        self.critic_weight = float(critic_weight)
        self.conditional_fibre = conditional_fibre
        self.fibre_rel_blend = float(fibre_rel_blend)
        self.cond_ot_weight = float(cond_ot_weight)
        self.cond_ot_mode = cond_ot_mode
        self.record = record
        self.t1c = T1CouplingLoss(
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
            simplex_curve_settle=simplex_curve_settle,
            support_completion=support_completion,
            chart_capacity=chart_capacity,
            chart_weight=chart_weight,
            halo_mult=halo_mult,
            multiscale_horizon=multiscale_horizon,
            cond_drift_deconflict=cond_drift_deconflict,
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
        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_t1c_capture_parts(fake_logits, self.record)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_t1c_capture_parts(
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
        slack_reg = self.t1c.critic_slack_loss(
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
            if self.t1c.weight == 0.0:
                return super().g_loss(fake_logits, None)
            raise ValueError("T1CouplingGANLoss requires real_logits in g_loss when t1c_weight > 0")
        if self.t1c.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if self.conditional_fibre == "calibrated" and has_cont_cond:
            return self.t1c.weight * super().g_loss(fake_logits, real_logits)
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
                            cond_ot = self.t1c.conditional_fiber_transport_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
                                mode=self.cond_ot_mode,
                                record=rec,
                            )
                            s_th = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
                            total_cond = rel_term + self.cond_ot_weight * cond_ot
                            if s_th < 1.0 and total_cond.requires_grad:
                                if getattr(fake_logits, "_t1c_thermally_scaled", False):
                                    cond_scaled = cond_ot * s_th + cond_ot.detach() * (1.0 - s_th)
                                    total_cond = rel_term + self.cond_ot_weight * cond_scaled
                                else:
                                    total_cond = total_cond * s_th + total_cond.detach() * (1.0 - s_th)
                            return self.t1c.weight * total_cond
                return self.t1c.weight * rel_term
            return self.t1c.weight * super().g_loss(fake_logits, real_logits)
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        loss_val, _ = self.t1c(
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
        if self.t1c.weight == 0.0:
            return super().joint_g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, _has_cont_cond = self._resolve_pair_full(fake_logits, real_logits)
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        loss_val, _ = self.t1c.joint_relativistic_loss(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            record=rec,
            allow_ell_update=allow_ell,
            cond_fake=cond_f,
            cond_real=cond_r,
            return_metrics=False,
        )
        return loss_val
