"""Riemannian Gauge-Deconflicted Transport (RGDT) GAN loss and connected-manifold chart operator."""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .critic import _lookup_rgdt_capture_parts
from .metric import (
    _PHI1,
    bipartite_deficit_macro_loss,
    born_infeld_aligned,
    born_infeld_c_hat_detached,
    born_infeld_pair,
    conformal_horizon_pair,
    discrete_pair_mask,
    pooled_tangent_conditional_loss,
    sinkhorn_slack,
    split_two_sample_residuals,
    sq_euclidean_f64,
    update_scalar_ell,
)


def _split_shared_conditional_parts(
    parts_f: Optional[Tuple[torch.Tensor, ...]],
    parts_r: Optional[Tuple[torch.Tensor, ...]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Separate generated/real sample tensors from shared continuous context tensors."""
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


def _centered_log_odds(
    d_fake: torch.Tensor,
    d_real: torch.Tensor,
    *,
    c_max: float = 20.0,
    detach_real: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b_f = d_fake.shape[0]
    b_r = d_real.shape[0]
    f_flat = d_fake.reshape(b_f, -1).mean(dim=1) if d_fake.ndim > 1 else d_fake.reshape(b_f)
    r_src = d_real.detach() if detach_real else d_real
    r_flat = r_src.reshape(b_r, -1).mean(dim=1) if r_src.ndim > 1 else r_src.reshape(b_r)
    d_bar = 0.5 * (f_flat.detach().double().mean() + r_flat.detach().double().mean())
    f_centered = (f_flat.double() - d_bar).to(f_flat.dtype)
    r_centered = (r_flat.double() - d_bar).to(r_flat.dtype)
    return (
        float(c_max) * torch.tanh(f_centered / float(c_max)),
        float(c_max) * torch.tanh(r_centered / float(c_max)),
    )


@torch.no_grad()
def _connected_manifold_gates(y_flat: torch.Tensor) -> Tuple[float, float, float, float, torch.Tensor]:
    """Return ``(g_conn, g_macro, g_curve, q90_y_u, y_scale)`` via 8-step minimax persistence."""
    b_r = y_flat.shape[0]
    y_scale = y_flat.float().abs().amax().clamp_min(1.0)
    if b_r < 4:
        return 0.0, 0.0, 0.0, 0.1, y_scale
    y_u = y_flat.float() / y_scale
    d_yy = torch.cdist(y_u, y_u, p=2.0)
    off_y = ~torch.eye(b_r, dtype=torch.bool, device=y_u.device)
    nn_y = d_yy.masked_fill(~off_y, float("inf")).amin(dim=-1)
    q90_t = torch.quantile(nn_y, 0.90).clamp_min(1e-8)
    q90_y_u = float(q90_t.clamp_min(1e-4).item())
    span_y = torch.quantile(d_yy[off_y], 0.90) / q90_t

    m_path = d_yy.clone()
    for _ in range(8):
        m_path = torch.minimum(
            m_path, torch.maximum(m_path.unsqueeze(2), m_path.unsqueeze(0)).amin(dim=1)
        )
    kappa_h0 = m_path.amax() / q90_t
    u_conn = ((10.0 - kappa_h0) / 3.0).clamp(0.0, 1.0)
    u_span = ((span_y - 1.15) / 0.25).clamp(0.0, 1.0)
    u_macro = ((span_y - 3.80) / 1.00).clamp(0.0, 1.0)
    s_conn = u_conn * u_conn * (3.0 - 2.0 * u_conn)
    s_span = u_span * u_span * (3.0 - 2.0 * u_span)
    s_macro = u_macro * u_macro * (3.0 - 2.0 * u_macro)
    g_conn = float((s_conn * s_span).item())
    g_macro = float((s_conn * s_span * s_macro).item())

    g_curve = 0.0
    if g_conn > 1e-4 and b_r >= 12:
        k_loc = min(9, b_r)
        idx_k = torch.topk(d_yy, k=k_loc, dim=-1, largest=False).indices
        patches = y_u[idx_k].double()
        centered = patches - patches.mean(dim=1, keepdim=True)
        gram = torch.bmm(centered, centered.transpose(1, 2))
        evals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        lam1 = evals[:, -1]
        lam2 = evals[:, -2] if k_loc >= 2 else torch.zeros_like(lam1)
        ratio_21 = torch.median(lam2 / lam1.clamp_min(1e-12))
        u_1d = ((0.11 - ratio_21) / 0.06).clamp(0.0, 1.0)
        s_1d = u_1d * u_1d * (3.0 - 2.0 * u_1d)
        g_curve = float((g_conn * s_1d).item())
    return g_conn, g_macro, g_curve, q90_y_u, y_scale


class RGDTGANLoss(GANLoss):
    """``GANLoss`` subclass implementing Riemannian Gauge-Deconflicted Transport (RGDT)."""

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        conformal_horizon: bool = True,
        macro_slack_weight: float = 1.0,
        fiber_residual_weight: float = 1.5,
        manifold_chart_weight: float = 0.35,
        rho: float = 3.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        slack_gain: float = 0.35,
        thermal_tau: float = 0.084,
        record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        if type(conformal_horizon) is not bool:
            raise ValueError("conformal_horizon must be a boolean")
        for name, val in (
            ("macro_slack_weight", macro_slack_weight),
            ("fiber_residual_weight", fiber_residual_weight),
            ("manifold_chart_weight", manifold_chart_weight),
            ("slack_gain", slack_gain),
            ("thermal_tau", thermal_tau),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.conformal_horizon = conformal_horizon
        self.macro_slack_weight = float(macro_slack_weight)
        self.fiber_residual_weight = float(fiber_residual_weight)
        self.manifold_chart_weight = float(manifold_chart_weight)
        self.rho = float(rho)
        self.tau = float(tau)
        self.eps = float(eps)
        self.iters = int(iters)
        self.beta_ell = float(beta_ell)
        self.nn_quantile = float(nn_quantile)
        self.slack_gain = float(slack_gain)
        self.thermal_tau = float(thermal_tau)
        self.record = record
        self._discrete_prior_step: bool = False
        self._manifold_step_active: bool = False
        self._fallback_ell = torch.tensor(0.0, dtype=torch.float32)

    def bind_discrete_prior_step(self, is_discrete: bool, *, manifold_active: bool = False) -> None:
        self._discrete_prior_step = bool(is_discrete)
        self._manifold_step_active = bool(is_discrete and manifold_active)

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
        bool,
    ]:
        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_rgdt_capture_parts(fake_logits, self.record)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_rgdt_capture_parts(
            real_logits, rec_f if rec_f is not None else self.record
        )
        rec = rec_f if rec_f is not None else (rec_r if rec_r is not None else self.record)
        has_cont_cond = bool(cc_f or cc_r)
        captured = x_f is not None and x_r is not None
        if not captured:
            x_f = fake_logits.reshape(fake_logits.shape[0], -1)
            x_r = real_logits.reshape(real_logits.shape[0], -1)
            cond_f = None
            cond_r = None
            has_cont_cond = False
            parts_f = None
            parts_r = None
        return x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r, captured

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return super().d_loss(real_logits, fake_logits)

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        discrete_step = bool(self._discrete_prior_step)
        manifold_step = bool(self._manifold_step_active)
        self._discrete_prior_step = False
        self._manifold_step_active = False

        if real_logits is None:
            return super().g_loss(fake_logits, None)

        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if not captured:
            return super().g_loss(fake_logits, real_logits)

        b_f = fake_logits.shape[0]
        b_r = real_logits.shape[0]

        # 1. Continuous-conditioned fibers (e.g. DDGAN / AE-DDGAN reverse transitions):
        if has_cont_cond:
            if b_f == b_r:
                f_flat = fake_logits.reshape(b_f, -1).mean(dim=1) if fake_logits.ndim > 1 else fake_logits.reshape(b_f)
                r_flat = (
                    real_logits.detach().reshape(b_r, -1).mean(dim=1)
                    if real_logits.ndim > 1
                    else real_logits.detach().reshape(b_r)
                )
                rel_term = F.softplus(r_flat - f_flat).mean()
                if self.fiber_residual_weight > 0.0:
                    same_disc = (cond_f is None and cond_r is None) or (
                        cond_f is not None
                        and cond_r is not None
                        and cond_f.shape == cond_r.shape
                        and torch.equal(cond_f, cond_r)
                    )
                    if same_disc:
                        x_sf, x_sr, c_cont = _split_shared_conditional_parts(parts_f, parts_r)
                        if x_sf is not None and x_sr is not None and c_cont is not None:
                            cond_ot = pooled_tangent_conditional_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
                                record=rec,
                            )
                            s_th = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
                            total_cond = rel_term + self.fiber_residual_weight * cond_ot
                            if s_th < 1.0 and total_cond.requires_grad:
                                if getattr(fake_logits, "_rgdt_thermally_scaled", False):
                                    cond_scaled = cond_ot * s_th + cond_ot.detach() * (1.0 - s_th)
                                    total_cond = rel_term + self.fiber_residual_weight * cond_scaled
                                else:
                                    total_cond = total_cond * s_th + total_cond.detach() * (1.0 - s_th)
                            return total_cond
                return rel_term
            return super().g_loss(fake_logits, real_logits)

        dim = max(1, x_r[0].numel()) if b_r > 0 else 0
        # 2. High-dimensional convolutional RGB domains (dim > 512, e.g. DCGAN 32^2..256^2):
        #    Avoid O(B^2 D) pixel-space L2 matrices; rely directly on BF16 feature critic D(x).
        if dim > 512:
            return super().g_loss(fake_logits, real_logits)

        # 3. Low/medium-dimensional ambient spaces (dim <= 512):
        #    Symmetric Conformal Born-Infeld Feydy OT + Bipartite Monge Deficit-to-Surplus Macro-Transport.
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        ell_scalar = update_scalar_ell(
            x_r,
            record=rec,
            fallback_ell=self._fallback_ell,
            beta_ell=self.beta_ell,
            nn_quantile=self.nn_quantile,
            allow_update=allow_ell,
        )
        if rec is None:
            self._fallback_ell = ell_scalar.detach().float().cpu()

        defer_thermal = False
        g_conn_fwd = 0.0
        if (
            (self.macro_slack_weight > 0.0 or (manifold_step and self.manifold_chart_weight > 0.0 and allow_ell))
            and cond_f is None
            and cond_r is None
            and dim <= 256
            and 4 <= b_r <= 256
        ):
            g_conn_fwd, g_macro_fwd, _, _, _ = _connected_manifold_gates(x_r.detach().reshape(b_r, -1))
            if manifold_step and self.manifold_chart_weight > 0.0 and allow_ell and g_macro_fwd > 1e-4:
                defer_thermal = True

        x_aug = x_f.reshape(b_f, -1)
        y_aug = x_r.detach().reshape(b_r, -1)
        cond_present = cond_f is not None or cond_r is not None
        ell_xy, ell_xx, _ = conformal_horizon_pair(
            x_aug,
            y_aug,
            ell_scalar,
            enabled=self.conformal_horizon,
            cond_present=cond_present,
        )

        c_phys_xy, c_hat_xy = born_infeld_pair(x_aug, y_aug, ell_xy)
        c_phys_xx, c_hat_xx = born_infeld_pair(x_aug, x_aug, ell_xx)

        d_fake_tilde, d_real_tilde = _centered_log_odds(fake_logits, real_logits, detach_real=True)
        mask_xy = discrete_pair_mask(cond_f, cond_r, b_f, b_r)
        mask_xx = discrete_pair_mask(cond_f, cond_f, b_f, b_f)

        P_xy = sinkhorn_slack(
            c_hat_xy,
            d_fake_tilde.detach(),
            d_real_tilde.detach(),
            rho=self.rho,
            tau=self.tau,
            eps=self.eps,
            iters=self.iters,
            hollow=False,
            pair_mask=mask_xy,
        )
        eff_mask_xx = mask_xx
        if discrete_step and b_f > 1:
            diag_f = torch.eye(b_f, dtype=torch.bool, device=x_aug.device)
            unique_xx = (c_hat_xx > 1e-12) | diag_f
            eff_mask_xx = unique_xx if mask_xx is None else (mask_xx & unique_xx)
        zeros_f = d_fake_tilde.new_zeros(b_f)
        P_xx = sinkhorn_slack(
            c_hat_xx,
            zeros_f,
            zeros_f,
            rho=self.rho,
            tau=self.tau,
            eps=self.eps,
            iters=self.iters,
            hollow=False,
            pair_mask=eff_mask_xx,
        )

        P_xy_act = P_xy[:b_f, :b_r]
        P_xy_row_slack = P_xy[:b_f, b_r]
        P_xy_col_slack = P_xy[b_f, :b_r]
        P_xx_act = P_xx[:b_f, :b_f]
        P_xx_act = 0.5 * (P_xx_act + P_xx_act.T)
        P_xx_act.fill_diagonal_(0.0)

        phase = float(getattr(rec, "rgdt_chord_phase", 0.0)) if rec is not None else 0.0
        macro_term = None
        if self.macro_slack_weight > 0.0 and not cond_present:
            macro_loss, explorer_mask = bipartite_deficit_macro_loss(
                x_aug,
                y_aug,
                c_hat_xy,
                P_xy_row_slack,
                P_xy_col_slack,
                rho=self.rho,
                phase=phase,
                g_conn=g_conn_fwd,
                cond_fake=cond_f,
                cond_real=cond_r,
                return_explorer_mask=True,
            )
            if explorer_mask is not None:
                P_xy_act = P_xy_act * (1.0 - explorer_mask.unsqueeze(1))
                macro_term = self.macro_slack_weight * macro_loss
        if rec is not None and allow_ell:
            rec.rgdt_chord_phase = (phase + _PHI1) % 1.0

        a_f = (float(b_f) * P_xy_act.sum(dim=1)).clamp(0.0, 1.0)
        P_xx_eff = P_xx_act * torch.sqrt(a_f.unsqueeze(1) * a_f.unsqueeze(0))

        transport_xy = (P_xy_act * c_phys_xy).sum()
        debias_xx = -0.5 * (P_xx_eff * c_phys_xx).sum()
        coord_total = transport_xy + debias_xx
        if macro_term is not None:
            coord_total = coord_total + macro_term

        s_f = (float(b_f) * P_xy_row_slack).clamp(0.0, 1.0)
        p_slack_eff = P_xy_row_slack * (1.0 + self.slack_gain * s_f) if self.slack_gain > 0.0 else P_xy_row_slack
        critic_total = -(p_slack_eff * (0.5 * self.tau * d_fake_tilde)).sum()
        total = coord_total + critic_total

        if self.thermal_tau > 0.0 and rec is not None:
            if (
                allow_ell
                and not defer_thermal
                and not cond_present
                and b_r >= 4
                and b_f >= 4
                and hasattr(rec, "observe_rgdt_thermal")
            ):
                r_ed, r_mu_sq, r_spread, r_q90, r_hmax, r_tail = split_two_sample_residuals(
                    x_aug,
                    y_aug,
                    ell=ell_scalar,
                    rho=self.rho,
                    is_discrete_prior=discrete_step,
                )
                rec.observe_rgdt_thermal(
                    r_ed,
                    r_mu_sq,
                    r_spread,
                    r_q90,
                    r_hmax,
                    self.thermal_tau,
                    r_tail=r_tail,
                )
                if dim == 1:
                    inv_sqrt_b = 1.0 / math.sqrt(float(max(16, b_r)))
                    u_1d = min(1.0, max(0.0, (rec.rgdt_ema_residual - 1.52 * inv_sqrt_b) / max(0.73 * inv_sqrt_b, 1e-8)))
                    rec._step_scale = float(0.48 + 0.52 * (u_1d * u_1d * (3.0 - 2.0 * u_1d)))
                else:
                    rec._step_scale = 1.0

            s_therm = float(getattr(rec, "_thermal_scale", 1.0))
            if s_therm < 1.0 and total.requires_grad:
                if getattr(fake_logits, "_rgdt_thermally_scaled", False):
                    coord_scaled = coord_total * s_therm + coord_total.detach() * (1.0 - s_therm)
                    total = coord_scaled + critic_total
                else:
                    total = total * s_therm + total.detach() * (1.0 - s_therm)

        return total

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        """Slack-screened relativistic BiGAN loss on ``(G(z), z)`` vs ``(x, E(x))``."""
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        x_f, x_r, cond_f, cond_r, rec, _, _, _, captured = self._resolve_pair_parts(fake_logits, real_logits)
        if not captured:
            return super().joint_g_loss(fake_logits, real_logits)

        b_f = fake_logits.shape[0]
        b_r = real_logits.shape[0]
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        ell = update_scalar_ell(
            x_r,
            record=rec,
            fallback_ell=self._fallback_ell,
            beta_ell=self.beta_ell,
            nn_quantile=self.nn_quantile,
            allow_update=allow_ell,
        )
        with torch.no_grad():
            c_hat_xy = born_infeld_c_hat_detached(x_f, x_r, ell)
            df_sg, dr_sg = _centered_log_odds(fake_logits.detach(), real_logits.detach(), detach_real=True)
            mask_xy = discrete_pair_mask(cond_f, cond_r, b_f, b_r)
            P_xy = sinkhorn_slack(
                c_hat_xy,
                df_sg,
                dr_sg,
                rho=self.rho,
                tau=self.tau,
                eps=self.eps,
                iters=self.iters,
                hollow=False,
                pair_mask=mask_xy,
            )
            P_xy_act = P_xy[:b_f, :b_r]
            P_xy_row_slack = P_xy[:b_f, b_r]
            P_xy_col_slack = P_xy[b_f, :b_r]

        f_flat = fake_logits.reshape(b_f, -1).mean(dim=1) if fake_logits.ndim > 1 else fake_logits.reshape(b_f)
        r_flat = real_logits.reshape(b_r, -1).mean(dim=1) if real_logits.ndim > 1 else real_logits.reshape(b_r)
        d_bar = 0.5 * (f_flat.detach().mean() + r_flat.detach().mean())

        active_rel = (P_xy_act * F.softplus(r_flat.unsqueeze(0) - f_flat.unsqueeze(1))).sum()
        slack_fake = (P_xy_row_slack * F.softplus(d_bar - f_flat)).sum()
        slack_real = (P_xy_col_slack * F.softplus(r_flat - d_bar)).sum()
        return active_rel + slack_fake + slack_real

    def manifold_chart_loss(
        self,
        z_table: torch.Tensor,
        gen_fn: Any,
        y_real: torch.Tensor,
        *,
        record: Any = None,
        chart_scale: float = 4.0,
    ) -> torch.Tensor:
        """Connected-manifold-gated simplicial chart regularizer for zero-sigma discrete priors.

        Strictly gated by ``H_0`` persistence connectivity ``g_conn > 1e-4`` and low ambient
        dimension ``dim <= 256``:
        - Returns exact zero with ZERO generator forward passes on disconnected multi-mode
          targets (e.g. ``img_bars4``) and high-dimensional CNNs (e.g. ``dcgan_performance``),
          avoiding both anti-Theory-1 inter-mode chord penalties and CNN BatchNorm/compute overhead.
        - Activates smoothly on connected low-dimensional curves/surfaces (e.g. ``interp_sparse_arc``,
          ``interp_saddle_barycentric``, ``interp_img_patch_shift``).
        """
        m_p = z_table.shape[0]
        b_r = y_real.shape[0]
        dim = max(1, y_real[0].numel()) if b_r > 0 else 0
        if self.manifold_chart_weight <= 0.0 or m_p < 2 or m_p > 128 or b_r < 4 or b_r > 256 or dim > 256:
            return y_real[:0].sum()

        y_flat = y_real.detach().reshape(b_r, -1)
        g_conn, g_macro, g_curve, q90_y_u, y_scale = _connected_manifold_gates(y_flat)
        if g_conn <= 1e-4:
            return y_real[:0].sum()

        phi = float(getattr(record, "rgdt_chord_phase", 0.0)) if record is not None else 0.0
        with torch.no_grad():
            z_det = z_table.detach()
            ell = update_scalar_ell(
                y_real,
                record=record,
                fallback_ell=self._fallback_ell,
                beta_ell=self.beta_ell,
                nn_quantile=self.nn_quantile,
                allow_update=False,
            )
            d_zz = torch.cdist(z_det.double(), z_det.double(), p=2.0)
            d_zz = d_zz + 1e-12 * torch.arange(m_p, device=z_det.device, dtype=torch.float64).unsqueeze(0)
            d_zz.fill_diagonal_(float("inf"))
            k_nn = min(4, m_p - 1)
            nbrs = d_zz.topk(k_nn, dim=-1, largest=False).indices
            y_u = y_flat.float() / y_scale

            n_q = max(4, min(b_r, 64))
            n_edge_base = (n_q // 2) if (m_p >= 3 and k_nn >= 2 and n_q >= 4) else n_q
            n_tri = int(round((1.0 - g_curve) * float(n_q - n_edge_base)))
            n_edge = n_q - n_tri

            k_edge = min(3, k_nn)
            r_e = torch.arange(n_edge, device=z_det.device)
            if g_curve > 0.5:
                n_dir = max(1, m_p * k_edge)
                pair_idx = (r_e + int(phi * float(n_dir))) % n_dir
                v0_e = pair_idx // k_edge
                v1_e = nbrs[v0_e, pair_idx % k_edge]
                q_idx = r_e // n_dir
                n_layers = max(1, (n_edge + n_dir - 1) // n_dir)
                u_e = (((q_idx.float() + 0.5) / float(n_layers)) + phi) % 1.0
            else:
                v0_e = (r_e + int(phi * m_p)) % m_p
                v1_e = nbrs[v0_e, r_e % k_edge]
                u_e = (((r_e.float() + 0.5) / float(max(1, n_edge))) + phi) % 1.0
            t_e = (0.20 + 0.60 * u_e).to(dtype=z_det.dtype)
            z_edge = (1.0 - t_e.unsqueeze(1)) * z_det[v0_e] + t_e.unsqueeze(1) * z_det[v1_e]

            if n_tri > 0:
                r_t = torch.arange(n_tri, device=z_det.device)
                phi_t = (phi * 0.7548776662466927) % 1.0
                v0_t = (r_t + int(phi_t * m_p)) % m_p
                c1_t = r_t % k_nn
                c2_t = (c1_t + 1 + ((r_t // k_nn) % max(1, k_nn - 1))) % k_nn
                v1_t = nbrs[v0_t, c1_t]
                v2_t = nbrs[v0_t, c2_t]
                u_t = (((r_t.float() + 0.5) / float(max(1, n_tri))) + phi_t) % 1.0
                ang = 2.0 * math.pi * u_t
                is_ring = (r_t % 2 == 1).to(dtype=z_det.dtype)
                rad_w = 0.12 * is_ring
                w0 = (1.0 / 3.0 + rad_w * torch.cos(ang).to(dtype=z_det.dtype)).unsqueeze(1)
                w1 = (1.0 / 3.0 + rad_w * torch.cos(ang + (2.0 * math.pi / 3.0)).to(dtype=z_det.dtype)).unsqueeze(1)
                w2 = 1.0 - w0 - w1
                z_tri = w0 * z_det[v0_t] + w1 * z_det[v1_t] + w2 * z_det[v2_t]
                z_simp = torch.cat([z_edge, z_tri], dim=0)
            else:
                z_simp = z_edge

        x_verts_live = gen_fn(z_table).reshape(m_p, -1)
        x_verts = x_verts_live.detach()
        x_simp = gen_fn(z_simp).reshape(n_q, -1)

        if (
            g_macro > 1e-4
            and self.thermal_tau > 0.0
            and record is not None
            and torch.is_grad_enabled()
            and not getattr(record, "_in_protected_probe", False)
            and hasattr(record, "observe_rgdt_thermal")
        ):
            with torch.no_grad():
                x_man = torch.cat([x_verts, x_simp.detach()], dim=0)
                r_ed, r_mu_sq, r_spread, r_q90, r_hmax, _ = split_two_sample_residuals(
                    x_man, y_flat, ell=ell, rho=self.rho, is_discrete_prior=True
                )
                r_q90_eff = (1.0 - 0.50 * g_curve) * r_q90 if g_curve > 1e-4 else r_q90
                record.observe_rgdt_thermal(
                    r_ed,
                    r_mu_sq,
                    r_spread,
                    r_q90_eff,
                    r_hmax,
                    self.thermal_tau * (1.0 + 0.25 * g_macro + 0.60 * g_curve),
                    shock_mu_ratio=0.35 + 0.55 * g_curve,
                    shock_ed_floor=0.30 + 0.50 * g_curve,
                    lock_hysteresis=0.22 * g_curve,
                    r_tail=0.0,
                )

        c_phys_sr, c_hat_sr = born_infeld_pair(x_simp, y_flat, ell)
        w_sr = torch.softmax(-c_hat_sr.float() / self.eps, dim=1).to(c_phys_sr.dtype)
        proj_loss = (w_sr * c_phys_sr).sum(dim=1).mean()

        eff_iso = (1.0 - g_macro) * 0.45 + g_macro * 0.25
        x_e = x_simp[:n_edge]
        d0_e = born_infeld_aligned(x_e, x_verts[v0_e], ell)
        d1_e = born_infeld_aligned(x_e, x_verts[v1_e], ell)
        iso_edge = ((1.0 - t_e.to(d0_e.dtype)) * d0_e - t_e.to(d0_e.dtype) * d1_e).abs().mean()
        if n_tri > 0:
            x_t = x_simp[n_edge:]
            d0_t = born_infeld_aligned(x_t, x_verts[v0_t], ell)
            d1_t = born_infeld_aligned(x_t, x_verts[v1_t], ell)
            d2_t = born_infeld_aligned(x_t, x_verts[v2_t], ell)
            with torch.no_grad():
                vx0, vx1, vx2 = x_verts[v0_t].double(), x_verts[v1_t].double(), x_verts[v2_t].double()
                ea = torch.linalg.vector_norm(vx0 - vx1, ord=2, dim=-1)
                eb = torch.linalg.vector_norm(vx1 - vx2, ord=2, dim=-1)
                ec = torch.linalg.vector_norm(vx2 - vx0, ord=2, dim=-1)
                s_p = 0.5 * (ea + eb + ec)
                area_sq = (
                    s_p * (s_p - ea).clamp_min(0.0) * (s_p - eb).clamp_min(0.0) * (s_p - ec).clamp_min(0.0)
                ).clamp_min(0.0)
                r_in = torch.sqrt(area_sq) / s_p.clamp_min(1e-8)
                ell64 = ell.detach().clamp_min(1e-6).double()
                margin = (0.70 * (torch.sqrt(ell64.square() + r_in.square()) - ell64)).to(d0_t.dtype)
            iso_tri = (F.relu(margin - d0_t) + F.relu(margin - d1_t) + F.relu(margin - d2_t)).mean() / 3.0
            iso_loss = 0.5 * (iso_edge + iso_tri)
        else:
            iso_loss = iso_edge

        total = proj_loss + eff_iso * iso_loss

        c_phys_vy, _ = born_infeld_pair(x_verts_live, y_flat, ell)
        with torch.no_grad():
            c_mean = c_phys_vy.detach().float().mean().clamp_min(1e-6)
            L_vy = -(c_phys_vy.detach().float() / c_mean) / 0.08
            log_a = L_vy.new_full((m_p,), -math.log(float(m_p)))
            log_b = L_vy.new_full((b_r,), -math.log(float(b_r)))
            u_v = torch.zeros_like(log_a)
            v_v = torch.zeros_like(log_b)
            for _ in range(10):
                v_v = log_b - torch.logsumexp(L_vy + u_v.unsqueeze(1), dim=0)
                u_v = log_a - torch.logsumexp(L_vy + v_v.unsqueeze(0), dim=1)
            P_vy = torch.exp(L_vy + u_v.unsqueeze(1) + v_v.unsqueeze(0)).to(c_phys_vy.dtype)
        vert_ot_loss = (P_vy * c_phys_vy).sum()
        if g_curve > 1e-4:
            with torch.no_grad():
                d_yy_m = torch.cdist(y_u, y_u, p=2.0)
                medoid_idx = torch.matmul(P_vy.float(), d_yy_m).argmin(dim=1)
                y_medoids = y_flat[medoid_idx]
            vert_medoid_loss = born_infeld_aligned(x_verts_live, y_medoids, 0.50 * ell).mean()
            total = total + float(g_conn) * (1.10 * vert_ot_loss + 0.65 * g_curve * vert_medoid_loss)
        else:
            total = total + (1.10 * float(g_conn)) * vert_ot_loss

        if z_table.requires_grad:
            with torch.no_grad():
                v_cloud = torch.cat([x_verts.float() / y_scale, y_u], dim=0)
                d_v = torch.cdist(v_cloud, v_cloud, p=2.0)
                h_step = max(1e-4, 1.35 * q90_y_u)
                g_mat = d_v * (1.0 + (d_v / h_step).square())
                for _ in range(6):
                    g_mat = torch.minimum(g_mat, (g_mat.unsqueeze(2) + g_mat.unsqueeze(0)).amin(dim=1))
                g_vv = g_mat[:m_p, :m_p].double()
                off_mask = ~torch.eye(m_p, dtype=torch.bool, device=z_table.device)
                g_rms = torch.sqrt(g_vv[off_mask].square().mean().clamp_min(1e-12))
                g_hat = g_vv / g_rms
                w_loc = torch.where(off_mask, (0.20 + g_hat.square()).pow(-2.0), 0.0)
                w_loc = w_loc / w_loc.sum().clamp_min(1e-12)
                if g_curve > 1e-4:
                    w_glob = torch.where(off_mask, 1.0 + 2.0 * g_hat.square(), 0.0)
                    w_glob = w_glob / w_glob.sum().clamp_min(1e-12)
                    w_chart = (1.0 - g_curve) * w_loc + g_curve * w_glob
                else:
                    w_chart = w_loc
            dz_sq = sq_euclidean_f64(z_table, z_table)
            dz_rms = torch.sqrt(dz_sq[off_mask].mean().clamp_min(1e-12))
            dz_hat = torch.sqrt(dz_sq + 1e-12) / dz_rms
            chart_stress = (w_chart * (dz_hat - g_hat).square()).sum().to(total.dtype)
            chart_mult = float(chart_scale) * (1.0 + 18.0 * g_curve)
            total = total + (float(g_conn) * chart_mult) * ell.detach().to(total.dtype) * chart_stress
            if g_curve > 1e-4 and z_table.shape[1] >= 2:
                z_c64 = z_table.double() - z_table.double().mean(dim=0, keepdim=True)
                frob_sq = z_c64.square().sum().clamp_min(1e-12)
                spec_sq = torch.linalg.matrix_norm(z_c64, ord=2).square()
                collinear_loss = ((frob_sq - spec_sq).clamp_min(0.0) / frob_sq).to(total.dtype)
                total = total + (3.2 * float(g_conn) * g_curve * float(chart_scale)) * ell.detach().to(total.dtype) * collinear_loss

        s_therm = float(getattr(record, "_thermal_scale", 1.0)) if record is not None else 1.0
        if s_therm < 1.0 and total.requires_grad:
            total = total * s_therm + total.detach() * (1.0 - s_therm)
        return self.manifold_chart_weight * total
