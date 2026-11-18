"""Unified Synthesis GANLoss: Deconflicted Piecewise Chart Coupling + Zero-Shrinkage Pooled Conditional Residualization."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .._rhbi.loss import _split_shared_conditional_parts
from .._t1_coupling.critic import _lookup_t1c_capture_parts
from .._t1_coupling.primitive import T1CouplingLoss
from .._t3_dynamics.metric import discrete_pair_mask
from ..gan_loss import GANLoss


@torch.no_grad()
def _zero_shrinkage_pooled_mean(
    c_f32: torch.Tensor,
    z_pool32: torch.Tensor,
    cond_disc: Optional[torch.Tensor],
) -> torch.Tensor:
    """Zero-sync FP32 exchangeability-invariant two-stage pooled conditional mean ``m_hat(c)`` on ``0.5*(X+Y)``.

    Uses non-blocking ``torch.linalg.cholesky_ex(..., check_errors=False)`` + ``torch.cholesky_solve`` in FP32
    (zero CUDA-to-CPU stream synchronizations!) to remove exact affine conditional drift first (Tikhonov ``1e-4``)
    followed by smooth RBF residual regression (ridge ``0.12 * n_g``).
    """
    b_n, d_c = c_f32.shape
    d_x = z_pool32.shape[1]
    orig_dev = c_f32.device
    if orig_dev.type == "cuda" and b_n <= 128 and d_c <= 16 and d_x <= 16:
        # Execute tiny (<=128x16) grouped Cholesky solves in 1 CPU transfer instead of ~20 tiny GPU kernels!
        c_cpu = c_f32.cpu()
        z_cpu = z_pool32.cpu()
        cd_cpu = None if cond_disc is None else cond_disc.detach().cpu()
        return _zero_shrinkage_pooled_mean(c_cpu, z_cpu, cd_cpu).to(device=orig_dev)

    m_hat = torch.zeros((b_n, d_x), dtype=torch.float32, device=c_f32.device)

    if cond_disc is None:
        groups = [torch.ones(b_n, dtype=torch.bool, device=c_f32.device)]
    else:
        cd = cond_disc.detach().reshape(b_n, -1)
        if cd.shape[1] == 1 and cd.dtype in (torch.int32, torch.int64) and b_n <= 1024:
            # Fast integer stratum scan for small discrete step indices (e.g. t in 1..4) without CPU sync loops
            unique_t = torch.unique(cd[:, 0])
            groups = [cd[:, 0] == unique_t[g] for g in range(unique_t.shape[0])]
        else:
            unique_t = torch.unique(cd, dim=0)
            groups = [(cd == unique_t[g].unsqueeze(0)).all(dim=-1) for g in range(unique_t.shape[0])]

    eye_c = torch.eye(d_c, dtype=torch.float32, device=c_f32.device)

    for mask_g in groups:
        idx_g = torch.nonzero(mask_g, as_tuple=False).squeeze(-1)
        n_g = int(idx_g.shape[0])
        if n_g == 0:
            continue
        z_g = z_pool32[idx_g]
        z_mean = z_g.mean(dim=0, keepdim=True)
        if n_g < max(4, d_c + 2):
            m_hat[idx_g] = z_mean
            continue
        c_g = c_f32[idx_g]
        dc = c_g - c_g.mean(dim=0, keepdim=True)
        dz = z_g - z_mean
        cov_tr = dc.square().sum().clamp_min(1e-8)

        # Stage 1: Nearly unregularized affine projection via zero-sync SPD Cholesky:
        gram_c = (dc.T @ dc) + (1e-4 * cov_tr / float(max(1, d_c))) * eye_c
        l_c, _ = torch.linalg.cholesky_ex(gram_c, check_errors=False)
        beta_ols = torch.cholesky_solve(dc.T @ dz, l_c)
        dz_lin = dc @ beta_ols
        res_nonlin = dz - dz_lin

        # Stage 2: Smooth RBF kernel ridge regression on nonlinear residual via zero-sync SPD Cholesky:
        dc_norm = dc * torch.rsqrt(cov_tr / float(n_g))
        d2_cc = torch.cdist(dc_norm, dc_norm, p=2.0).square()
        off_g = ~torch.eye(n_g, dtype=torch.bool, device=c_f32.device)
        med_d2 = torch.quantile(d2_cc[off_g], 0.50).clamp_min(1e-3) if n_g > 1 else d2_cc.new_tensor(1.0)
        k_rbf = torch.exp(-0.5 * d2_cc / med_d2)
        k_reg = k_rbf + (0.12 * float(n_g)) * torch.eye(n_g, dtype=torch.float32, device=c_f32.device)
        l_rbf, _ = torch.linalg.cholesky_ex(k_reg, check_errors=False)
        alpha_rbf = torch.cholesky_solve(res_nonlin, l_rbf)
        m_hat[idx_g] = z_mean + dz_lin + (k_rbf @ alpha_rbf)
    return m_hat


def unified_pooled_conditional_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
    record: Any = None,
) -> torch.Tensor:
    """Exchangeability-invariant zero-shrinkage pooled-residual conditional IMQ discrepancy on ``(x | c)``."""
    b_n = x_fake.shape[0]
    if b_n <= 1 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
        return x_fake[:0].sum()

    x_flat = x_fake.reshape(b_n, -1)
    y_flat = y_real.detach().reshape(b_n, -1)
    dim = max(1, x_flat.shape[-1])
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))

    with torch.no_grad():
        same_t = discrete_pair_mask(cond_disc, cond_disc, b_n, b_n)
        if same_t is None:
            same_t = torch.ones((b_n, b_n), dtype=torch.bool, device=x_fake.device)
        m_hollow = same_t.clone()
        m_hollow.fill_diagonal_(False)
        valid_row = m_hollow.any(dim=1)
        m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)
        if not bool(m_valid.any()):
            return x_fake[:0].sum()

        c_f32 = c_cont.detach().reshape(b_n, -1).float()
        z_pool32 = 0.5 * (x_flat.detach().float() + y_flat.float())
        m_hat32 = _zero_shrinkage_pooled_mean(c_f32, z_pool32, cond_disc)

        c_scale = c_f32.abs().amax().clamp_min(1.0)
        d_c = torch.cdist(c_f32 / c_scale, c_f32 / c_scale, p=2.0)
        d_c_max = d_c.amax()
        d_c_masked = d_c.masked_fill(~m_valid, float("inf"))
        d_sorted, sort_idx = torch.sort(d_c_masked, dim=1)
        nn_idx = sort_idx[:, 0]

        n_i = m_valid.sum(dim=1)
        k_idx = torch.minimum(torch.full_like(n_i, 3), n_i.clamp_min(1)) - 1
        sigma_raw = d_sorted.gather(1, k_idx.unsqueeze(1)).squeeze(1)
        sigma_floor = 1e-3 * d_c_max + 1e-6
        sigma_i = torch.where(
            valid_row,
            torch.where(torch.isfinite(sigma_raw), sigma_raw, torch.zeros_like(sigma_raw)).clamp_min(0.0)
            + sigma_floor,
            torch.ones_like(sigma_raw),
        )

        y_res = y_flat.float() - m_hat32
        y_sc = y_res.abs().amax().clamp_min(1e-3)
        innov_d = (y_res - y_res[nn_idx]) * (1.0 / math.sqrt(2.0))
        same_valid = same_t & valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
        f_mask = same_valid.float()
        f_cnt = f_mask.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        ell_d_sq_raw = (f_mask @ innov_d.square()) / f_cnt
        ell_iso_sq = ell_d_sq_raw.mean(dim=-1, keepdim=True)
        ell_d_sq = 0.50 * ell_iso_sq + 0.50 * ell_d_sq_raw
        ell_d_floor = (1e-2 * y_sc / math.sqrt(float(dim))) + 1e-5
        ell_d = torch.where(
            valid_row.unsqueeze(1),
            ell_d_sq.sqrt().clamp_min(ell_d_floor),
            torch.ones_like(ell_d_sq),
        )
        ell_rms = ell_d.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        w_coord = ell_rms / ell_d
        w_coord = w_coord / w_coord.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        ell_f = ell_rms * math.sqrt(float(dim))

        denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
        k_c = torch.where(m_valid, torch.exp(-d_c.square() / denom_2s), 0.0)
        deg_c = 1.0 + k_c.sum(dim=1)
        inv_sqrt_deg = torch.rsqrt(deg_c)
        w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
        w_imq_sym = 0.5 * (w_imq + w_imq.T)
        w_imq_dt = w_imq_sym.to(x_flat.dtype)

    xw = (x_flat.float() - m_hat32) * w_coord
    yw = (y_flat.float() - m_hat32) * w_coord
    sq_w_xy = (
        xw.square().sum(dim=-1, keepdim=True)
        + yw.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (xw @ yw.T)
    )
    sq_w_xy = sq_w_xy + (sq_w_xy.detach().clamp_min(0.0) - sq_w_xy.detach())
    sq_w_xx = (
        xw.square().sum(dim=-1, keepdim=True)
        + xw.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (xw @ xw.T)
    )
    sq_w_xx = sq_w_xx + (sq_w_xx.detach().clamp_min(0.0) - sq_w_xx.detach())

    psi_xy = torch.zeros_like(sq_w_xy)
    psi_xx = torch.zeros_like(sq_w_xx)
    for s_mult in (0.35, 1.0, 2.8):
        ell_s = ell_f * s_mult
        ell_s2 = ell_s.square()
        rad_s_xy = torch.sqrt(ell_s2 + sq_w_xy)
        psi_xy = psi_xy + (ell_s * sq_w_xy) / (rad_s_xy * (ell_s + rad_s_xy))
        rad_s_xx = torch.sqrt(ell_s2 + sq_w_xx)
        psi_xx = psi_xx + (ell_s * sq_w_xx) / (rad_s_xx * (ell_s + rad_s_xx))
    psi_xy_t = (psi_xy * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)
    psi_xx_t = (psi_xx * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)

    if (
        record is not None
        and hasattr(record, "t1c_cond_initialized")
        and torch.is_grad_enabled()
        and not getattr(record, "_in_protected_probe", False)
    ):
        with torch.no_grad():
            yw_det = yw.detach()
            sq_w_yy = (
                yw_det.square().sum(dim=-1, keepdim=True)
                + yw_det.square().sum(dim=-1).unsqueeze(0)
                - 2.0 * (yw_det @ yw_det.T)
            ).clamp_min(0.0)
            psi_yy = torch.zeros_like(sq_w_yy)
            for s_mult in (0.35, 1.0, 2.8):
                ell_s = ell_f * s_mult
                ell_s2 = ell_s.square()
                rad_s_yy = torch.sqrt(ell_s2 + sq_w_yy)
                psi_yy = psi_yy + (ell_s * sq_w_yy) / (rad_s_yy * (rad_s_yy + ell_s))
            psi_yy = psi_yy * (1.0 / 3.0)
            psi_xy_d = psi_xy.detach() * (1.0 / 3.0)
            psi_xx_d = psi_xx.detach() * (1.0 / 3.0)
            num_c = (w_imq_sym * (2.0 * psi_xy_d - psi_xx_d - psi_yy)).sum()
            den_c = (w_imq_sym * psi_yy).sum().clamp_min(1e-8)
            r_cond = max(-0.5, min(0.5, float((num_c / den_c).item())))
            if not record.t1c_cond_initialized:
                record.t1c_cond_ema = max(0.18, r_cond)
                record.t1c_cond_initialized = True
            else:
                record.t1c_cond_ema = 0.98 * float(record.t1c_cond_ema) + 0.02 * r_cond
            u_cond = min(1.0, max(0.0, (0.028 - max(0.0, float(record.t1c_cond_ema))) / 0.016))
            s_cond = u_cond * u_cond * (3.0 - 2.0 * u_cond)
            if hasattr(record, "_step_scale"):
                record._step_scale = float(1.0 - 0.68 * s_cond)

    return (w_imq_dt * psi_xy_t).sum() - 0.5 * (w_imq_dt * psi_xx_t).sum()


class UnifiedGANLoss(GANLoss):
    """Unified GANLoss combining Track 1B chart coupling, Track 2 high-res fast path, and Track 3 conditional residualization."""

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
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        critic_weight: float = 0.0,
        self_coupling: str = "feydy_discrete",
        relativistic_weight: float = 0.0,
        rapidity_screening: bool = False,
        scale_log_odds: bool = False,
        conditional_fibre: str = "slack",
        akt_horizon: bool = True,
        akt_mode: str = "weibull",
        slack_gain: float = 0.35,
        thermal_tau: float = 0.084,
        thermal_shock_mu: float = 0.35,
        fibre_rel_blend: float = 0.50,
        cond_ot_weight: float = 1.5,
        cond_ot_mode: str = "diag_imq_cmmd",
        cvx_mu_gain: float = 0.58,
        tail_spill_gain: float = 0.95,
        simplex_macro_settle: bool = True,
        simplex_curve_settle: bool = True,
        support_completion: bool = True,
        chart_capacity: int = 256,
        chart_weight: float = 1.0,
        halo_mult: float = 2.1,
        multiscale_horizon: bool = True,
        cond_drift_deconflict: bool = True,
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
        bool,
    ]:
        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_t1c_capture_parts(fake_logits, self.record)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_t1c_capture_parts(
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
        base = super().d_loss(real_logits, fake_logits)
        if self.critic_weight == 0.0 and self.fibre_rel_blend == 0.0:
            return base
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, _, _, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if not captured:
            return base
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
            raise ValueError("UnifiedGANLoss requires real_logits in g_loss when t1c_weight > 0")
        if self.t1c.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)

        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if not captured:
            return self.t1c.weight * super().g_loss(fake_logits, real_logits)

        b_f = fake_logits.shape[0]
        b_r = real_logits.shape[0]

        # 1. Continuous-conditioned fibers (DDGAN / AE-DDGAN reverse chains):
        if self.conditional_fibre == "calibrated" and has_cont_cond:
            return self.t1c.weight * super().g_loss(fake_logits, real_logits)
        if self.conditional_fibre == "slack" and has_cont_cond:
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
                            cond_ot = unified_pooled_conditional_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
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

        # 2. High-resolution RGB image domains (dim > 1024, e.g. 32x32..256x256 DCGAN):
        #    Avoid pixel-space pooling/Sinkhorn overhead; rely directly on 1st-order feature critic logits.
        dim = max(1, x_r[0].numel()) if b_r > 0 else 0
        if x_r.ndim >= 4 and dim > 1024:
            self.t1c._discrete_prior_step = False
            self.t1c._simplicial_active_step = False
            if rec is not None and hasattr(rec, "_t1c_disc_gate"):
                rec._t1c_disc_gate = 0.0
            return super().g_loss(fake_logits, real_logits)

        # 3. Deconflicted piecewise chart coupling:
        #    Suppress discrete-island chart snapping when simplicial manifold regularization is active on
        #    zero-sigma discrete manifolds (e.g. interp_sparse_arc, interp_saddle_barycentric, interp_img_patch_shift)
        #    unless x_f is a 4D disconnected image grid where g_conn == 0.
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        is_simplicial = bool(self.t1c._simplicial_active_step)
        saved_cw = self.t1c.chart_weight
        if is_simplicial and x_f.ndim == 2:
            self.t1c.chart_weight = 0.0
        try:
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
            if is_simplicial and rec is not None and hasattr(rec, "_t1c_disc_gate"):
                rec._t1c_disc_gate = 0.0
        finally:
            self.t1c.chart_weight = saved_cw
        return loss_val

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        if self.t1c.weight == 0.0:
            return super().joint_g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, _, _, _, captured = self._resolve_pair_parts(fake_logits, real_logits)
        if not captured:
            return super().joint_g_loss(fake_logits, real_logits)
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


__all__ = [
    "UnifiedGANLoss",
    "unified_pooled_conditional_loss",
]
