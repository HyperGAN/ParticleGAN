"""Unified-v15 4-Law + Zero-`eigh` CondFiber GANLoss & Coupling Primitive.

Self-contained Law 3 coupling and loss module importing directly from
``particlegan._unified_v15.atomic`` and ``particlegan._unified_v15.charts``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._bispf.thermal import _publish_thermal
from .._rhbi.loss import _split_shared_conditional_parts
from .._t1_coupling.critic import _lookup_t1c_capture_parts
from .._t3_dynamics.metric import born_infeld_aligned, born_infeld_c_hat_detached, discrete_pair_mask, sinkhorn_slack
from .._unified.loss import _zero_shrinkage_pooled_mean
from .._unified_v11.bank import UnifiedV11CondFiberBank, sym_eigensplit_zero_eigh
from ..gan_loss import GANLoss
from .atomic import (
    compute_discrete_atomic_gate,
    match_inverse_atomic_targets,
    split_joint_table_pair,
    update_atomic_ledger,
)
from .charts import (
    compute_chart_valley_gate,
    deconflicted_chart_routing,
    deterministic_spatial_pool2d,
    project_chart_features,
    project_transport_features,
    update_streaming_charts,
)


class _UnifiedV15CouplingBase(nn.Module):
    """Deconflicted piecewise Born-Infeld coupling with persistent Laguerre chart commitment.

    Combines:
    1. Pure-FP32 overflow-safe debiased Born-Infeld local transport (with energy-preserving
       8x8 spatial projection on >8x8 images for EMPF-class throughput).
    2. Persistent multi-batch streaming chart atlas ``(K_max >= 128)`` breaking the
       ``B_r < M`` horizon (e.g. ``grid100`` at ``B=64 < M=100``) paired with balanced
       Laguerre hard Monge chart commitment and a C^1 intra-mode halo dead-zone
       (eliminating 0.5-ghost barycentric averaging and false minibatch-absence kicks).
    3. Position-symmetric gap-guarded conformal multi-scale horizon ``ell(u, v) =
       sqrt(ell_tilde(u) * ell_tilde(v))`` (eliminating narrow-vs-wide mode conflation
       on ``vector_unequal_width`` while preserving exact zero Feydy self-coupling bias).
    4. Exchangeable midpoint ridge drift deconfliction on continuous conditioned fibers
       (eliminating off-diagonal context drift saturation on ``model_ddgan_gaussian_chain``).
    """

    def __init__(
        self,
        weight: float = 1.0,
        rho: float = 3.0,
        tau: float = 2.0,
        eps: float = 0.05,
        iters: int = 8,
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        *,
        rapidity_screening: bool = True,
        self_coupling: str = "feydy_discrete",
        relativistic_weight: float = 0.5,
        scale_log_odds: bool = True,
        akt_horizon: bool = True,
        akt_mode: str = "weibull",
        slack_gain: float = 2.5,
        thermal_tau: float = 0.0,
        thermal_shock_mu: float = 0.15,
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
        max_spatial: int = 16,
    ) -> None:
        super().__init__()
        for name, val, positive in (
            ("weight", weight, False),
            ("rho", rho, True),
            ("tau", tau, False),
            ("eps", eps, True),
            ("c_max", c_max, True),
            ("beta_ell", beta_ell, True),
            ("relativistic_weight", relativistic_weight, False),
            ("slack_gain", slack_gain, False),
            ("thermal_tau", thermal_tau, False),
            ("thermal_shock_mu", thermal_shock_mu, False),
            ("cvx_mu_gain", cvx_mu_gain, True),
            ("tail_spill_gain", tail_spill_gain, False),
            ("chart_weight", chart_weight, False),
            ("halo_mult", halo_mult, True),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or (
                val <= 0.0 if positive else val < 0.0
            ):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        if beta_ell > 1.0:
            raise ValueError("beta_ell must be in (0, 1]")
        if isinstance(nn_quantile, bool) or not isinstance(nn_quantile, (int, float)) or not (0.0 < nn_quantile < 1.0):
            raise ValueError("nn_quantile must be in (0, 1)")
        if type(iters) is not int or iters < 1:
            raise ValueError("iters must be a positive integer")
        if type(chart_capacity) is not int or chart_capacity < 2:
            raise ValueError("chart_capacity must be an integer >= 2")
        if type(max_spatial) is not int or max_spatial < 2:
            raise ValueError("max_spatial must be an integer >= 2")
        for bname, bval in (
            ("rapidity_screening", rapidity_screening),
            ("scale_log_odds", scale_log_odds),
            ("akt_horizon", akt_horizon),
            ("simplex_macro_settle", simplex_macro_settle),
            ("simplex_curve_settle", simplex_curve_settle),
            ("support_completion", support_completion),
            ("multiscale_horizon", multiscale_horizon),
            ("cond_drift_deconflict", cond_drift_deconflict),
        ):
            if type(bval) is not bool:
                raise ValueError(f"{bname} must be a boolean")
        if type(akt_mode) is not str or akt_mode not in ("sqrt", "weibull"):
            raise ValueError("akt_mode must be 'sqrt' or 'weibull'")
        if type(self_coupling) is not str or self_coupling not in (
            "hollow",
            "feydy",
            "feydy_unique",
            "feydy_discrete",
            "feydy_atomic",
            "feydy_atomic_v15",
        ):
            raise ValueError(
                "self_coupling must be 'hollow', 'feydy', 'feydy_unique', 'feydy_discrete', 'feydy_atomic', or 'feydy_atomic_v15'"
            )

        self.weight = float(weight)
        self.rho = float(rho)
        self.tau = float(tau)
        self.eps = float(eps)
        self.iters = int(iters)
        self.c_max = float(c_max)
        self.beta_ell = float(beta_ell)
        self.nn_quantile = float(nn_quantile)
        self.rapidity_screening = rapidity_screening
        self.self_coupling = self_coupling
        self.relativistic_weight = float(relativistic_weight)
        self.scale_log_odds = scale_log_odds
        self.akt_horizon = akt_horizon
        self.akt_mode = akt_mode
        self.slack_gain = float(slack_gain)
        self.thermal_tau = float(thermal_tau)
        self.thermal_shock_mu = float(thermal_shock_mu)
        self.cvx_mu_gain = float(cvx_mu_gain)
        self.tail_spill_gain = float(tail_spill_gain)
        self.simplex_macro_settle = simplex_macro_settle
        self.simplex_curve_settle = simplex_curve_settle
        self.support_completion = support_completion
        self.chart_capacity = int(chart_capacity)
        self.chart_weight = float(chart_weight)
        self.halo_mult = float(halo_mult)
        self.multiscale_horizon = multiscale_horizon
        self.cond_drift_deconflict = cond_drift_deconflict
        self.max_spatial = int(max_spatial)

        self.register_buffer("ell_ema", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ell_initialized", torch.tensor(False, dtype=torch.bool))
        self._ell_init_py: bool = False
        self.last_p_xy_slack: Optional[torch.Tensor] = None
        self._discrete_prior_step: bool = False
        self._simplicial_active_step: bool = False

    def bind_discrete_prior_step(self, active: bool, *, simplicial_active: bool = False) -> None:
        if type(simplicial_active) is not bool:
            raise ValueError("simplicial_active must be a boolean")
        self._discrete_prior_step = bool(active)
        self._simplicial_active_step = bool(active and simplicial_active)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ell_init_py = bool(self.ell_initialized.item())

    @staticmethod
    def _born_infeld_pair_fp32(
        x: torch.Tensor,
        y: torch.Tensor,
        ell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure-FP32 scale-normalized overflow-safe C^infinity Born-Infeld pairwise costs.

        Normalizes ``(x, y, ell)`` by detached ``max(|x|, |y|, ell, 1)`` so inputs up to
        ``FLT_MAX`` never overflow FP32 GEMMs while preserving exact analytical gradients
        ``(x_i - y_j) / sqrt(ell^2 + ||x_i - y_j||_2^2)`` everywhere, including ``x_i == y_j``.
        """
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dim = max(1, x_flat.shape[-1])
        ell_f = ell.detach().float().clamp_min(1e-6)

        with torch.no_grad():
            scale = torch.maximum(
                torch.maximum(x_flat.detach().float().abs().amax(), y_flat.detach().float().abs().amax()),
                ell_f.amax(),
            ).clamp_min(1.0)

        x_s = x_flat.float() / scale
        y_s = y_flat.float() / scale
        ell_s = (ell_f / scale).clamp_min(1e-20)

        sq_s = (
            x_s.square().sum(dim=-1, keepdim=True)
            + y_s.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * (x_s @ y_s.T)
        )
        sq_s = sq_s + (sq_s.detach().clamp_min(0.0) - sq_s.detach())
        rad_s = torch.sqrt(ell_s.square() + sq_s)
        val_s = sq_s / (rad_s + ell_s).clamp_min(1e-30)
        c_s = (rad_s - ell_s) + (val_s - (rad_s - ell_s)).detach()

        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
        c_phys = (c_s * scale).clamp_max(fmax).to(x_flat.dtype)

        with torch.no_grad():
            val_hat = val_s.detach() / ell_s
            ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
            c_hat = val_hat.clamp_max(ratio_max).to(x_flat.dtype)
        return c_phys, c_hat

    @staticmethod
    @torch.no_grad()
    def _born_infeld_c_hat_detached(
        x: torch.Tensor,
        y: torch.Tensor,
        ell: torch.Tensor,
        *,
        max_spatial: int = 8,
    ) -> torch.Tensor:
        """Fast pure-FP32 overflow-safe detached dimensionless cost ``c_hat``."""
        x_flat = project_transport_features(x.detach(), max_spatial=max_spatial).float()
        y_flat = project_transport_features(y.detach(), max_spatial=max_spatial).float()
        dim = max(1, x_flat.shape[-1])
        ell_f = ell.detach().float().clamp_min(1e-6)
        scale = torch.maximum(
            torch.maximum(x_flat.abs().amax(), y_flat.abs().amax()),
            ell_f.amax(),
        ).clamp_min(1.0)
        dist_scaled = torch.cdist(x_flat / scale, y_flat / scale, p=2.0)
        inv_u = (ell_f / scale).clamp_min(1e-20)
        rad = torch.sqrt(inv_u.square() + dist_scaled.square())
        val_hat = dist_scaled.square() / ((rad + inv_u) * inv_u).clamp_min(1e-38)
        ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
        return val_hat.clamp_max(ratio_max).to(x.dtype)

    @staticmethod
    def _born_infeld_aligned_fp32(x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        """Row-wise aligned pure-FP32 physical Born-Infeld distance ``[N]``."""
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dim = max(1, x_flat.shape[-1])
        ell_f = ell.detach().float().clamp_min(1e-6)
        with torch.no_grad():
            scale = torch.maximum(
                torch.maximum(x_flat.detach().float().abs().amax(), y_flat.detach().float().abs().amax()),
                ell_f.amax(),
            ).clamp_min(1.0)
        diff_s = (x_flat.float() - y_flat.float()) / scale
        ell_s = (ell_f / scale).clamp_min(1e-20)
        sq_s = diff_s.square().sum(dim=-1)
        rad_s = torch.sqrt(ell_s.square() + sq_s)
        val_s = sq_s / (rad_s + ell_s).clamp_min(1e-30)
        c_s = (rad_s - ell_s) + (val_s - (rad_s - ell_s)).detach()
        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
        return (c_s * scale).clamp_max(fmax).to(x_flat.dtype)

    @staticmethod
    @torch.no_grad()
    def _batch_ell_quantile_tensor(
        y_real: torch.Tensor,
        nn_quantile: float = 0.25,
        *,
        akt_horizon: bool = True,
        akt_mode: str = "weibull",
        max_spatial: int = 8,
    ) -> torch.Tensor:
        """Zero-sync scale-equivariant nearest-neighbor horizon quantile on device."""
        b_y = y_real.shape[0]
        if b_y <= 1:
            return y_real.new_tensor(1.0, dtype=torch.float32)
        y_flat = project_transport_features(y_real.detach(), max_spatial=max_spatial).float()
        dim = max(1, y_flat.shape[-1])
        akt_exp = max(0.0, (1.0 / float(dim)) - 0.5)
        if akt_horizon and akt_exp > 0.0:
            if akt_mode == "weibull":
                q_rate = max(1e-6, -math.log(max(1e-6, 1.0 - float(nn_quantile))))
                akt_factor = ((2.0 * math.e * float(b_y)) / q_rate) ** akt_exp
            else:
                akt_factor = float(b_y) ** akt_exp
        else:
            akt_factor = 1.0
        scale = y_flat.abs().amax().clamp_min(1.0)
        y_unit = y_flat / scale
        dist_unit = torch.cdist(y_unit, y_unit, p=2.0)
        dist_unit.fill_diagonal_(0.0)
        d_max = dist_unit.amax()
        thresh = 1e-5 * d_max
        pos_min = torch.where(dist_unit > thresh, dist_unit, float("inf")).amin()

        dist_unit.fill_diagonal_(float("inf"))
        k_nn = min(b_y - 1, max(1, b_y // 200))
        if k_nn == 1:
            nn_dist = dist_unit.amin(dim=-1)
        else:
            nn_dist = torch.topk(dist_unit, k=k_nn, dim=-1, largest=False).values[:, -1]
        q_raw = torch.quantile(nn_dist, nn_quantile)
        rms_unit = y_unit.square().mean().sqrt()

        tie_base = torch.where(torch.isfinite(pos_min), 0.15 * pos_min, 0.05 * d_max)
        raw_unit = torch.where(q_raw <= thresh, tie_base, q_raw)
        nonzero_val = scale * akt_factor * raw_unit + 1e-6
        dirac_val = 0.05 * torch.clamp_min(scale * rms_unit, 1.0) + 1e-6
        return torch.where(d_max == 0.0, dirac_val, nonzero_val)

    def _get_ell_tensor(self, device: torch.device, record: Any = None) -> Tuple[torch.Tensor, bool]:
        if record is not None and hasattr(record, "t1c_ell_initialized"):
            init_cur = bool(record.t1c_ell_initialized)
            ell_t = getattr(record, "_t1c_ell_t", None)
            if ell_t is not None and ell_t.device == device:
                return ell_t, init_cur
            val = float(record._t1c_ell_scalar) if hasattr(record, "_t1c_ell_scalar") else float(record.t1c_ell_ema)
            t = torch.tensor(val, dtype=torch.float32, device=device)
            if hasattr(record, "_t1c_ell_t"):
                record._t1c_ell_t = t
            return t, init_cur
        return self.ell_ema.to(device=device, dtype=torch.float32), self._ell_init_py

    def _set_ell_tensor(self, ell_f32: torch.Tensor, record: Any = None) -> None:
        if self.ell_ema.device != ell_f32.device:
            self.ell_ema = ell_f32.detach().clone()
            self.ell_initialized = torch.tensor(True, dtype=torch.bool, device=ell_f32.device)
        else:
            self.ell_ema.copy_(ell_f32.detach())
            if not self._ell_init_py:
                self.ell_initialized.fill_(True)
        self._ell_init_py = True
        if record is not None and hasattr(record, "t1c_ell_initialized"):
            record.t1c_ell_initialized = True
            if hasattr(record, "_t1c_ell_t"):
                record._t1c_ell_t = ell_f32.detach()
            else:
                record.t1c_ell_ema = float(ell_f32.item())

    @torch.no_grad()
    def _update_ell(
        self,
        y_real: torch.Tensor,
        *,
        record: Any = None,
        allow_update: bool = True,
    ) -> torch.Tensor:
        b_y = y_real.shape[0]
        ell_cur_f32, init_cur = self._get_ell_tensor(y_real.device, record)
        if not allow_update:
            out = ell_cur_f32.clamp_min(1e-6) if init_cur else y_real.new_tensor(1.0, dtype=torch.float32)
            return out.to(dtype=y_real.dtype)
        if b_y <= 1:
            out = ell_cur_f32.clamp_min(1.0) if init_cur else y_real.new_tensor(1.0, dtype=torch.float32)
            return out.to(dtype=y_real.dtype)

        q_f32 = self._batch_ell_quantile_tensor(
            y_real,
            self.nn_quantile,
            akt_horizon=self.akt_horizon,
            akt_mode=self.akt_mode,
            max_spatial=self.max_spatial,
        )
        new_ell_f32 = q_f32 if not init_cur else ((1.0 - self.beta_ell) * ell_cur_f32 + self.beta_ell * q_f32)
        self._set_ell_tensor(new_ell_f32, record)
        return new_ell_f32.clamp_min(1e-6).to(dtype=y_real.dtype)

    @torch.no_grad()
    def _conformal_local_horizons(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        ell_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Position-symmetric gap-guarded conformal local horizons ``(ell_x, ell_y, d_yy)``.

        Evaluates the exact same detached target-scale field ``ell_tilde(.)`` on both
        ``x_t`` and ``y_t``, guaranteeing ``ell_conf(X, X) == ell_conf(X, Y)`` bit-for-bit
        at ``X == Y`` (exact zero Feydy self-coupling bias) while scaling local screening
        horizons linearly across narrow and wide modes (e.g. ``vector_unequal_width``).
        """
        xf = x_t.detach().float()
        yf = y_t.detach().float()
        b_f = xf.shape[0]
        b_r = yf.shape[0]
        ell_g = ell_global.detach().float().clamp_min(1e-6)
        scale = torch.maximum(torch.maximum(xf.abs().amax(), yf.abs().amax()), ell_g).clamp_min(1.0)
        y_u = yf / scale
        x_u = xf / scale
        d_yy = torch.cdist(y_u, y_u, p=2.0) * scale
        d_yy_nn = d_yy.clone()
        d_yy_nn.fill_diagonal_(float("inf"))

        k_loc = min(3, b_r - 1)
        vals = torch.topk(d_yy_nn, k=k_loc, dim=1, largest=False).values
        d1 = vals[:, 0].clamp_min(1e-6)
        d2 = vals[:, min(1, k_loc - 1)].clamp_min(1e-6)
        d3 = vals[:, min(2, k_loc - 1)].clamp_min(1e-6)
        # Gap-guard: wide continuous modes satisfy d2 <= 2.8*d1 and d3 <= 2.8*d2,
        # whereas 1-or-2 point rare clusters jump across valleys and stay at d1/ell_g.
        smooth_mode = (d2 <= 2.8 * d1) & (d3 <= 2.8 * d2)
        r_pt = torch.where(smooth_mode, d2, d1)

        # Patch-smooth within each mode (k_patch <= 8 intra-mode neighbors within 3.2*d2)
        # so single-Gaussian Weibull tail variance averages out and homogeneous-width
        # targets (two_pole, ring16, grid100, ae_gan_hold) stay strictly at ell_g!
        k_patch = min(8, b_r)
        p_dists, p_idx = torch.topk(d_yy, k=k_patch, dim=1, largest=False)
        in_mode = (p_dists <= (3.2 * d2).unsqueeze(1)).to(dtype=torch.float32)
        in_mode[:, 0] = 1.0
        r_patch = (r_pt[p_idx] * in_mode).sum(dim=1) / in_mode.sum(dim=1).clamp_min(1.0)

        q15 = torch.quantile(r_patch, 0.15).clamp_min(1e-6)
        q85 = torch.quantile(r_patch, 0.85)
        kappa_width = float((q85 / q15).item())
        u_width = min(1.0, max(0.0, (kappa_width - 2.85) / 0.85))
        g_width = u_width * u_width * (3.0 - 2.0 * u_width)
        if g_width <= 1e-4:
            return ell_g.expand(b_f).clone(), ell_g.expand(b_r).clone(), d_yy

        q_ref = torch.quantile(r_patch, self.nn_quantile).clamp_min(1e-6)
        s_target = (ell_g * (r_patch / q_ref)).clamp(0.70 * ell_g, 4.50 * ell_g)
        s_raw = (1.0 - g_width) * ell_g + g_width * s_target

        bw_u = (2.4 * s_raw / scale).clamp_min(1e-5)
        two_bw_sq = 2.0 * bw_u.unsqueeze(0).square()

        d2_yy_u = (d_yy / scale).square()
        w_yy = torch.softmax(-d2_yy_u / two_bw_sq, dim=1)
        ell_y = (w_yy @ s_raw).clamp_min(1e-6)

        d2_xy_u = torch.cdist(x_u, y_u, p=2.0).square()
        w_xy = torch.softmax(-d2_xy_u / two_bw_sq, dim=1)
        ell_x = (w_xy @ s_raw).clamp_min(1e-6)
        return ell_x, ell_y, d_yy

    @torch.no_grad()
    def _screening_energy(self, c_hat: torch.Tensor) -> torch.Tensor:
        c_f = c_hat.float().clamp_min(0.0)
        if self.rapidity_screening:
            two_rho = 2.0 * self.rho
            return two_rho * torch.log1p(c_f / two_rho)
        return c_f.clamp_max(1e6)

    @torch.no_grad()
    def _sinkhorn_slack(
        self,
        c_hat: torch.Tensor,
        d_row: torch.Tensor,
        d_col: torch.Tensor,
        hollow: bool = False,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b_r, b_c = c_hat.shape
        energy = self._screening_energy(c_hat)
        L = c_hat.new_zeros((b_r + 1, b_c + 1), dtype=torch.float32)
        active = -energy / self.eps
        if pair_mask is not None:
            active = active.masked_fill(~pair_mask, float("-inf"))
        if hollow and b_r == b_c:
            active.fill_diagonal_(float("-inf"))
        L[:b_r, :b_c] = active
        L[:b_r, b_c] = (-0.5 * self.rho + 0.5 * self.tau * d_row.float()) / self.eps
        L[b_r, :b_c] = (-0.5 * self.rho - 0.5 * self.tau * d_col.float()) / self.eps

        log_row = L.new_full((b_r + 1,), -math.log(max(1, b_r)))
        log_row[b_r] = math.log(2.0)
        log_col = L.new_full((b_c + 1,), -math.log(max(1, b_c)))
        log_col[b_c] = math.log(2.0)

        u = torch.zeros_like(log_row)
        v = torch.zeros_like(log_col)
        for _ in range(self.iters):
            v = log_col - torch.logsumexp(L + u.unsqueeze(1), dim=0)
            u = log_row - torch.logsumexp(L + v.unsqueeze(0), dim=1)
        return torch.exp(L + u.unsqueeze(1) + v.unsqueeze(0)).to(c_hat.dtype)

    @staticmethod
    def _discrete_pair_mask(
        cond_row: Optional[torch.Tensor],
        cond_col: Optional[torch.Tensor],
        b_r: int,
        b_c: int,
    ) -> Optional[torch.Tensor]:
        if cond_row is None or cond_col is None:
            return None
        if cond_row.shape[0] != b_r or cond_col.shape[0] != b_c:
            return None
        r_flat = cond_row.detach().reshape(b_r, -1)
        c_flat = cond_col.detach().reshape(b_c, -1)
        if r_flat.shape[1] != c_flat.shape[1]:
            return None
        return (r_flat.unsqueeze(1) == c_flat.unsqueeze(0)).all(dim=-1)

    def _centered_log_odds(
        self,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        ell: Optional[torch.Tensor] = None,
        detach_real: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_f = d_fake.shape[0]
        b_r = d_real.shape[0]
        f_flat = d_fake.reshape(b_f, -1).mean(dim=1) if d_fake.ndim > 1 else d_fake.reshape(b_f)
        r_src = d_real.detach() if detach_real else d_real
        r_flat = r_src.reshape(b_r, -1).mean(dim=1) if r_src.ndim > 1 else r_src.reshape(b_r)
        # Scalar log-odds mean in double avoids cancellation on huge constant critic shifts (+1000).
        d_bar = 0.5 * (f_flat.detach().double().mean() + r_flat.detach().double().mean())
        if self.scale_log_odds and ell is not None:
            inv_ell = 1.0 / ell.detach().double().clamp_min(1e-6)
            f_centered = ((f_flat.double() - d_bar) * inv_ell).to(f_flat.dtype)
            r_centered = ((r_flat.double() - d_bar) * inv_ell).to(r_flat.dtype)
        else:
            f_centered = (f_flat.double() - d_bar).to(f_flat.dtype)
            r_centered = (r_flat.double() - d_bar).to(r_flat.dtype)
        return self.c_max * torch.tanh(f_centered / self.c_max), self.c_max * torch.tanh(r_centered / self.c_max)

    def critic_slack_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        critic_weight: float = 0.0,
        record: Any = None,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if critic_weight == 0.0:
            return d_real.new_zeros(())
        b_f = x_fake.shape[0]
        b_r = y_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=False)
        c_hat_xy = self._born_infeld_c_hat_detached(
            x_fake, y_real, ell, max_spatial=self.max_spatial
        )
        d_fake_tilde, d_real_tilde = self._centered_log_odds(
            d_fake, d_real, ell=ell, detach_real=False
        )
        mask_xy = self._discrete_pair_mask(cond_fake, cond_real, b_f, b_r)
        P_xy = self._sinkhorn_slack(
            c_hat_xy, d_fake_tilde.detach(), d_real_tilde.detach(), hollow=False, pair_mask=mask_xy
        )
        p_row_slack = P_xy[:b_f, b_r]
        p_col_slack = P_xy[b_f, :b_r]
        slack_scale = ell.detach() if self.scale_log_odds else 1.0
        return float(critic_weight) * slack_scale * (
            -(p_col_slack * (0.5 * self.tau * d_real_tilde)).sum()
            + (p_row_slack * (0.5 * self.tau * d_fake_tilde)).sum()
        )

    @staticmethod
    @torch.no_grad()
    def _minimax_persistence_max(d_yy: torch.Tensor, steps: int = 8) -> torch.Tensor:
        n = d_yy.shape[0]
        if n <= 512:
            m_path = d_yy.clone()
            for _ in range(steps):
                m_path = torch.minimum(
                    m_path, torch.maximum(m_path.unsqueeze(2), m_path.unsqueeze(0)).amin(dim=1)
                )
            return m_path.amax()
        in_tree = torch.zeros(n, dtype=torch.bool, device=d_yy.device)
        in_tree[0] = True
        min_edge = d_yy[0].clone()
        max_weight = d_yy.new_zeros(())
        for _ in range(n - 1):
            cand = min_edge.masked_fill(in_tree, float("inf"))
            weight, nxt = cand.min(dim=0)
            max_weight = torch.maximum(max_weight, weight)
            in_tree[nxt] = True
            min_edge = torch.minimum(min_edge, d_yy[nxt])
        return max_weight

    @staticmethod
    def _weyl_strided_indices(n: int, cap: int, phi: float, device: torch.device) -> torch.Tensor:
        offset = int(float(phi) * float(n)) % n
        return ((torch.arange(cap, device=device, dtype=torch.long) * n) // cap + offset) % n

    @staticmethod
    @torch.no_grad()
    def _compute_manifold_gates(y_flat: torch.Tensor) -> Tuple[float, float, float, torch.Tensor]:
        b_r = y_flat.shape[0]
        y_scale = y_flat.float().abs().amax().clamp_min(1.0)
        if b_r < 4:
            return 0.0, 0.0, 0.1, y_scale
        y_u = y_flat.float() / y_scale
        d_yy = torch.cdist(y_u, y_u, p=2.0)
        off_y = ~torch.eye(b_r, dtype=torch.bool, device=y_u.device)
        nn_y = d_yy.masked_fill(~off_y, float("inf")).amin(dim=-1)
        q90_t = torch.quantile(nn_y, 0.90).clamp_min(1e-8)
        q90_y_u = float(q90_t.clamp_min(1e-4).item())
        span_y = torch.quantile(d_yy[off_y], 0.90) / q90_t
        kappa_h0 = _UnifiedV15CouplingBase._minimax_persistence_max(d_yy, steps=8) / q90_t
        u_conn = ((10.0 - kappa_h0) / 3.0).clamp(0.0, 1.0)
        u_span = ((span_y - 1.15) / 0.25).clamp(0.0, 1.0)
        u_macro = ((span_y - 3.80) / 1.00).clamp(0.0, 1.0)
        s_conn = u_conn * u_conn * (3.0 - 2.0 * u_conn)
        s_span = u_span * u_span * (3.0 - 2.0 * u_span)
        s_macro = u_macro * u_macro * (3.0 - 2.0 * u_macro)
        return float((s_conn * s_span).item()), float((s_conn * s_span * s_macro).item()), q90_y_u, y_scale

    @staticmethod
    @torch.no_grad()
    def _record_prev_charts(record: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if record is None or not bool(getattr(record, "chart_initialized", False)):
            return None, None
        csum = getattr(record, "chart_sum", None)
        ccnt = getattr(record, "chart_count", None)
        csq = getattr(record, "chart_sq_dev", None)
        if csum is None or ccnt is None or csq is None or csum.shape[0] == 0:
            return None, None
        denom = ccnt.detach().float().clamp_min(1e-8)
        charts = csum.detach().float() / denom.unsqueeze(1)
        radii = (csq.detach().float() / denom).clamp_min(1e-12).sqrt()
        return charts, radii

    @torch.no_grad()
    def _compute_manifold_gates_eff(
        self,
        y_flat: torch.Tensor,
        *,
        record: Any = None,
    ) -> Tuple[float, float, float, torch.Tensor]:
        g_conn, g_macro, q90_y_u, y_scale = self._compute_manifold_gates(y_flat)
        if self.self_coupling in ("feydy_atomic", "feydy_atomic_v15") and (g_conn > 0.0 or g_macro > 0.0):
            prev_c, prev_r = self._record_prev_charts(record)
            g_atom = compute_discrete_atomic_gate(y_flat, charts=prev_c, radii=prev_r)
            if g_atom > 0.0:
                scale_a = 1.0 - float(g_atom)
                g_conn = float(g_conn * scale_a)
                g_macro = float(g_macro * scale_a)
        return g_conn, g_macro, q90_y_u, y_scale

    @staticmethod
    @torch.no_grad()
    def _compute_curve_gate(y_flat: torch.Tensor, g_conn: float) -> float:
        b_r = y_flat.shape[0]
        if g_conn <= 1e-4 or b_r < 12:
            return 0.0
        y_scale = y_flat.float().abs().amax().clamp_min(1.0)
        y_u = y_flat.float() / y_scale
        d_yy = torch.cdist(y_u, y_u, p=2.0)
        k_loc = min(9, b_r)
        idx_k = torch.topk(d_yy, k=k_loc, dim=-1, largest=False).indices
        patches = y_u[idx_k]
        centered = patches - patches.mean(dim=1, keepdim=True)
        gram = torch.bmm(centered, centered.transpose(1, 2))
        evals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        lam1 = evals[:, -1]
        lam2 = evals[:, -2] if k_loc >= 2 else torch.zeros_like(lam1)
        ratio_21 = torch.median(lam2 / lam1.clamp_min(1e-12))
        u_1d = ((0.11 - ratio_21) / 0.06).clamp(0.0, 1.0)
        s_1d = u_1d * u_1d * (3.0 - 2.0 * u_1d)
        return float((float(g_conn) * s_1d).item())

    @staticmethod
    @torch.no_grad()
    def _compute_valley_gate(y_flat: torch.Tensor, ell: torch.Tensor, rho: float) -> float:
        b_r = y_flat.shape[0]
        if b_r < 8:
            return 0.0
        scale = torch.maximum(y_flat.float().abs().amax(), ell.detach().float()).clamp_min(1.0)
        yr = y_flat.float() / scale
        off = ~torch.eye(b_r, dtype=torch.bool, device=yr.device)
        d_yy = torch.cdist(yr, yr, p=2.0)
        yy_q75 = torch.quantile(d_yy.masked_fill(~off, float("inf")).amin(dim=1), 0.75) * scale
        y_mid = 0.5 * (yr + torch.roll(yr, shifts=1, dims=0))
        d_my = torch.cdist(y_mid, yr, p=2.0)
        idx = torch.arange(b_r, device=yr.device)
        d_my[idx, idx] = float("inf")
        d_my[idx, (idx - 1) % b_r] = float("inf")
        mid_q75 = torch.quantile(d_my.amin(dim=1), 0.75) * scale
        denom_h = torch.maximum(yy_q75, ell.detach().float()).clamp_min(1e-8)
        nu_val = mid_q75 / denom_h
        u_val = ((nu_val - float(rho)) / 2.0).clamp(0.0, 1.0)
        return float((u_val * u_val * (3.0 - 2.0 * u_val)).item())

    @staticmethod
    @torch.no_grad()
    def _split_two_sample_residuals(
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        ell: Optional[torch.Tensor] = None,
        *,
        cvx_mu_gain: float = 0.80,
    ) -> Tuple[float, float, float, float, float]:
        xf = x_flat.detach().float()
        yf = y_flat.detach().float()
        bx, by = xf.shape[0], yf.shape[0]
        scale = torch.maximum(xf.abs().amax(), yf.abs().amax()).clamp_min(1.0)
        xu = xf / scale
        yu = yf / scale
        d_xy_mat = torch.cdist(xu, yu, p=2)
        d_xx_mat = torch.cdist(xu, xu, p=2)
        d_yy_mat = torch.cdist(yu, yu, p=2)

        d_xy = d_xy_mat.mean()
        d_xx = d_xx_mat.sum() / max(1, bx * (bx - 1))
        d_yy = d_yy_mat.sum() / max(1, by * (by - 1))
        denom_ed = (0.5 * (d_xy + 0.5 * (d_xx + d_yy))).clamp_min(1e-8)
        r_ed_t = (2.0 * d_xy - d_xx - d_yy) / denom_ed

        mu_diff_sq = (xu.mean(dim=0) - yu.mean(dim=0)).square().sum()
        var_x = xu.var(dim=0, unbiased=True).sum() if bx > 1 else mu_diff_sq.new_zeros(())
        var_y = yu.var(dim=0, unbiased=True).sum() if by > 1 else mu_diff_sq.new_zeros(())
        r_mu_sq_raw = (mu_diff_sq - var_x / max(1, bx) - var_y / max(1, by)) / denom_ed.square()
        r_spread_t = (d_xx - d_yy) / denom_ed

        b_min = min(bx, by)
        if b_min >= 4:
            xy_sub = d_xy_mat[:b_min, :b_min].clone()
            yy_sub = d_yy_mat[:b_min, :b_min].clone()
            xy_sub.fill_diagonal_(float("inf"))
            yy_sub.fill_diagonal_(float("inf"))
            xy_row_nn = xy_sub.amin(dim=1)
            xy_col_nn = xy_sub.amin(dim=0)
            yy_loo_nn = yy_sub.amin(dim=1)

            ell_ref = (
                (ell.detach().float() / scale).clamp_min(1e-8)
                if ell is not None
                else torch.quantile(yy_loo_nn, 0.25).clamp_min(1e-8)
            )
            psi_base = (0.15 + 5.5 * (ell_ref / denom_ed)).clamp(0.15, 0.40)
            r_mu_base = psi_base.square() * r_mu_sq_raw
            if cvx_mu_gain != 0.40 and by >= 4:
                d_yy_full = d_yy_mat.clone()
                d_yy_full.fill_diagonal_(float("inf"))
                nn_y_mean = d_yy_full.amin(dim=1).mean()
                if float(nn_y_mean.item()) > 1e-8:
                    y_mid = 0.5 * (yu + yu.roll(1, dims=0))
                    d_my = torch.cdist(y_mid, yu, p=2)
                    idx = torch.arange(by, device=yu.device)
                    d_my[idx, idx] = float("inf")
                    d_my[idx, (idx + 1) % by] = float("inf")
                    rho_cvx = d_my.amin(dim=1).mean() / nn_y_mean
                    u_cvx = ((1.15 - rho_cvx) / 0.20).clamp(0.0, 1.0)
                    w_cvx = float((u_cvx * u_cvx * (3.0 - 2.0 * u_cvx)).item())
                    psi_hi = 0.40 + w_cvx * (float(cvx_mu_gain) - 0.40)
                    psi_cvx = (0.15 + 5.5 * (ell_ref / denom_ed)).clamp(0.15, psi_hi)
                    r_mu_cvx = psi_cvx.square() * r_mu_sq_raw
                    r_mu_sq_t = torch.where(
                        r_mu_sq_raw >= 0.0,
                        torch.maximum(r_mu_base, r_mu_cvx.clamp_max(0.045)),
                        r_mu_cvx,
                    )
                else:
                    r_mu_sq_t = r_mu_base
            else:
                r_mu_sq_t = r_mu_base

            prec_q75 = torch.quantile(xy_row_nn, 0.75) - torch.quantile(yy_loo_nn, 0.75)
            r_q90_t = 10.2 * prec_q75 / denom_ed
            cov90 = torch.quantile(xy_col_nn, 0.90) - torch.quantile(yy_loo_nn, 0.90)
            r_hmax_t = cov90 / denom_ed
        else:
            r_mu_sq_t = 0.16 * r_mu_sq_raw
            r_q90_t = r_ed_t
            r_hmax_t = r_ed_t

        vals = torch.stack((r_ed_t, r_mu_sq_t, r_spread_t, r_q90_t, r_hmax_t)).tolist()
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])

    def forward(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        feat_fake: Optional[torch.Tensor] = None,
        feat_real: Optional[torch.Tensor] = None,
        *,
        record: Any = None,
        symmetric: bool = False,
        allow_ell_update: bool = True,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        discrete_step = bool(self._discrete_prior_step)
        simplicial_step = bool(self._simplicial_active_step)
        self._discrete_prior_step = False
        self._simplicial_active_step = False

        if self.weight == 0.0:
            zero = torch.zeros((), device=x_fake.device, dtype=x_fake.dtype)
            if not return_metrics:
                return zero, {}
            ell_cur = (
                float(record.t1c_ell_ema)
                if record is not None and hasattr(record, "t1c_ell_initialized")
                else float(self.ell_ema.item())
            )
            return zero, {"t1c_active_rate": 0.0, "t1c_ell": ell_cur}

        b_f = x_fake.shape[0]
        b_r = y_real.shape[0]
        grad_enabled = torch.is_grad_enabled()
        in_probe = bool(getattr(record, "_in_protected_probe", False)) if record is not None else False
        can_mutate = bool(allow_ell_update and grad_enabled and not in_probe)

        ell = self._update_ell(y_real, record=record, allow_update=allow_ell_update)

        y_target = y_real if symmetric else y_real.detach()
        x_proj = project_transport_features(x_fake, max_spatial=self.max_spatial)
        y_proj = project_transport_features(y_target, max_spatial=self.max_spatial)

        defer_thermal = False
        if (
            simplicial_step
            and self.simplex_macro_settle
            and allow_ell_update
            and cond_fake is None
            and cond_real is None
            and b_r >= 4
        ):
            _, g_macro_fwd, _, _ = self._compute_manifold_gates_eff(y_proj.detach(), record=record)
            if g_macro_fwd > 1e-4:
                defer_thermal = True

        if feat_fake is not None and feat_real is not None:
            f_real = feat_real if symmetric else feat_real.detach()
            x_aug = torch.cat([x_proj, feat_fake.reshape(b_f, -1) * ell], dim=-1)
            y_aug = torch.cat([y_proj, f_real.reshape(b_r, -1) * ell], dim=-1)
        else:
            x_aug = x_proj
            y_aug = y_proj

        uncond_pair = (not symmetric) and (cond_fake is None) and (cond_real is None)

        # 1. Persistent Multi-Batch Chart Update & Deconflicted Laguerre Routing
        absent_shield: Optional[torch.Tensor] = None
        chart_route_weight: Optional[torch.Tensor] = None
        chart_targets: Optional[torch.Tensor] = None
        chart_halos: Optional[torch.Tensor] = None
        w_valley_chart = 0.0
        atomic_active_fwd = False
        r_atomic_fwd = 0.0

        g_conn_fwd = 0.0
        g_atom_fwd = 0.0
        if self.chart_weight > 0.0 and uncond_pair and b_f >= 4 and b_r >= 4:
            # Compute spatial area ratio between transport coordinates and raw chart features.
            if y_real.ndim >= 4 and (y_real.shape[-2] > self.max_spatial or y_real.shape[-1] > self.max_spatial):
                h_c = min(self.max_spatial, int(y_real.shape[-2]))
                w_c = min(self.max_spatial, int(y_real.shape[-1]))
                area_ratio = math.sqrt(float(y_real.shape[-2] * y_real.shape[-1]) / float(max(1, h_c * w_c)))
            else:
                area_ratio = 1.0
            ell_chart = ell.detach().float() / area_ratio

            rec_csum = getattr(record, "chart_sum", None) if record is not None else None
            rec_ccnt = getattr(record, "chart_count", None) if record is not None else None
            rec_csq = getattr(record, "chart_sq_dev", None) if record is not None else None
            rec_cfm = getattr(record, "chart_fake_mass", None) if record is not None else None
            rec_phase = float(getattr(record, "revival_phase", 0.0)) if record is not None else 0.0
            rec_revs = int(getattr(record, "revivals", 0)) if record is not None else 0

            g_atom_fwd = 0.0
            if self.self_coupling in ("feydy_atomic", "feydy_atomic_v15"):
                y_cf_pre = project_chart_features(y_real.detach(), max_spatial=self.max_spatial)
                prev_c_pre, prev_r_pre = self._record_prev_charts(record)
                g_atom_fwd = compute_discrete_atomic_gate(y_cf_pre, charts=prev_c_pre, radii=prev_r_pre)

            if can_mutate or rec_csum is None or rec_ccnt is None or rec_csq is None or rec_cfm is None:
                (
                    charts_c,
                    w_real_c,
                    radii_c,
                    batch_real_cnt,
                    next_sum,
                    next_cnt,
                    next_sq,
                    cur_fm,
                    next_phase,
                    next_revs,
                ) = update_streaming_charts(
                    y_real,
                    rec_csum,
                    rec_ccnt,
                    rec_csq,
                    rec_cfm,
                    rec_phase,
                    rec_revs,
                    max_charts=self.chart_capacity,
                    max_spatial=self.max_spatial,
                    allow_dead_revival=(g_atom_fwd <= 0.5),
                )
            else:
                y_cf_tmp = project_chart_features(y_real.detach(), max_spatial=self.max_spatial)
                charts_c = rec_csum.to(device=y_real.device, dtype=torch.float32) / rec_ccnt.to(
                    device=y_real.device, dtype=torch.float32
                ).clamp_min(1e-8).unsqueeze(1)
                w_real_c = rec_ccnt.to(device=y_real.device, dtype=torch.float32)
                w_real_c = w_real_c / w_real_c.sum().clamp_min(1e-8)
                radii_c = (
                    rec_csq.to(device=y_real.device, dtype=torch.float32)
                    / rec_ccnt.to(device=y_real.device, dtype=torch.float32).clamp_min(1e-8)
                ).clamp_min(1e-12).sqrt()
                sc_tmp = torch.maximum(y_cf_tmp.abs().amax(), charts_c.abs().amax()).clamp_min(1.0)
                asg_tmp = torch.cdist(y_cf_tmp / sc_tmp, charts_c / sc_tmp, p=2.0).argmin(dim=1)
                batch_real_cnt = F.one_hot(asg_tmp, num_classes=charts_c.shape[0]).to(dtype=torch.float32).sum(dim=0)
                next_sum, next_cnt, next_sq = rec_csum, rec_ccnt, rec_csq
                cur_fm = rec_cfm.to(device=y_real.device, dtype=torch.float32)
                next_phase, next_revs = rec_phase, rec_revs

            y_cf_det = project_chart_features(y_real.detach(), max_spatial=self.max_spatial)
            x_cf_det = project_chart_features(x_fake.detach(), max_spatial=self.max_spatial)
            g_conn_fwd, _, _, _ = self._compute_manifold_gates_eff(y_cf_det, record=record)
            w_valley_raw = compute_chart_valley_gate(y_cf_det, charts_c, ell_chart, self.rho)
            # Connected curved manifolds (e.g. circular arcs / saddle sheets) have g_conn -> 1
            # even when Euclidean chords cut across ambient space; gate strictly by (1 - g_conn).
            w_valley_chart = float(w_valley_raw * (1.0 - g_conn_fwd))
            if g_atom_fwd > 0.5:
                w_valley_chart = max(w_valley_chart, float(g_atom_fwd))
            if can_mutate and record is not None:
                record._t1c_disc_gate = float(w_valley_chart) if charts_c.shape[0] >= 3 else 0.0
            s_th_cur = float(getattr(record, "_thermal_scale", 1.0)) if record is not None else 1.0

            prior_table_act = getattr(record, "_active_prior_table", None) if record is not None else None
            batch_rows_act = getattr(record, "_active_batch_rows", None) if record is not None else None
            use_atomic_ledger = (
                self.self_coupling in ("feydy_atomic", "feydy_atomic_v15")
                and g_atom_fwd > 0.5
                and charts_c.shape[0] >= 3
                and prior_table_act is not None
                and batch_rows_act is not None
                and isinstance(prior_table_act, torch.Tensor)
                and isinstance(batch_rows_act, torch.Tensor)
                and prior_table_act.ndim == 2
                and batch_rows_act.ndim == 1
                and int(batch_rows_act.shape[0]) == b_f
            )
            if use_atomic_ledger:
                assert prior_table_act is not None and batch_rows_act is not None and record is not None
                (
                    atomic_tgt_idx,
                    _nn_cc_a,
                    next_rc,
                    next_rd,
                    next_rt,
                    r_atomic_fwd,
                ) = update_atomic_ledger(
                    x_cf_det,
                    batch_rows_act,
                    int(prior_table_act.shape[0]),
                    charts_c,
                    w_real_c,
                    getattr(record, "atomic_row_chart", None),
                    getattr(record, "atomic_row_dist", None),
                    getattr(record, "atomic_row_target", None),
                    allow_update=can_mutate,
                )
                if can_mutate and hasattr(record, "update_atomic_ledger_state"):
                    record.update_atomic_ledger_state(next_rc, next_rd, next_rt)
                chart_targets = charts_c[atomic_tgt_idx]
                chart_halos = torch.zeros(b_f, device=x_cf_det.device, dtype=torch.float32)
                chart_route_weight = torch.full(
                    (b_f,), float(g_atom_fwd), device=x_cf_det.device, dtype=torch.float32
                )
                absent_shield = (
                    (batch_real_cnt <= 0.0).to(dtype=torch.float32)[atomic_tgt_idx] * float(g_atom_fwd)
                )
                k_a = int(charts_c.shape[0])
                batch_fk_a = F.one_hot(atomic_tgt_idx, num_classes=k_a).to(dtype=torch.float32).sum(dim=0) / float(
                    max(1, b_f)
                )
                next_fm = (0.90 * cur_fm.float() + 0.10 * batch_fk_a) if can_mutate else cur_fm.float()
                atomic_active_fwd = True
            else:
                (
                    chart_targets,
                    chart_halos,
                    chart_route_weight,
                    absent_shield,
                    next_fm,
                ) = deconflicted_chart_routing(
                    x_cf_det,
                    charts_c,
                    w_real_c,
                    radii_c,
                    batch_real_cnt,
                    cur_fm,
                    ell_chart,
                    w_valley=w_valley_chart,
                    halo_mult=self.halo_mult,
                    s_thermal=s_th_cur,
                    allow_update=can_mutate,
                )
            if can_mutate and record is not None and hasattr(record, "update_charts"):
                d_scale = float(y_cf_det.abs().amax().clamp_min(1e-6).item())
                record.update_charts(
                    next_sum, next_cnt, next_sq, next_fm, next_phase, next_revs, d_scale
                )

        # 2. Position-Symmetric Conformal Multi-Scale Horizon & Scale-Invariant Support Completion
        ell_xy = ell
        ell_xx = ell
        ell_y_loc = ell
        g_width_active = False
        g_het = 0.0
        nn1_y_sup: Optional[torch.Tensor] = None
        s_near_sup: Optional[torch.Tensor] = None
        d_yy_cached: Optional[torch.Tensor] = None

        if self.multiscale_horizon and (cond_fake is None) and (cond_real is None) and b_f >= 12 and b_r >= 12:
            ell_x_ms, ell_y_ms, d_yy_cached = self._conformal_local_horizons(x_aug, y_aug, ell)
            if not bool(torch.allclose(ell_y_ms, ell.detach().float().clamp_min(1e-6).expand_as(ell_y_ms))):
                g_width_active = True
                ell_xy = torch.sqrt(ell_x_ms.unsqueeze(1) * ell_y_ms.unsqueeze(0)).to(ell.dtype)
                ell_xx = torch.sqrt(ell_x_ms.unsqueeze(1) * ell_x_ms.unsqueeze(0)).to(ell.dtype)
                ell_y_loc = ell_y_ms.to(ell.dtype)

        use_support = self.support_completion and uncond_pair and b_f >= 8 and b_r >= 8
        if use_support:
            with torch.no_grad():
                yf_sup = y_proj.detach().float()
                xf_sup = x_proj.detach().float()
                if d_yy_cached is not None:
                    d_yy_sup = d_yy_cached.clone()
                else:
                    sc_sup = yf_sup.abs().amax().clamp_min(1.0)
                    d_yy_sup = torch.cdist(yf_sup / sc_sup, yf_sup / sc_sup, p=2.0) * sc_sup
                d_yy_sup.fill_diagonal_(0.0)
                ell_f_base = ell.detach().float().clamp_min(1e-6)
                wv_pt = self._compute_valley_gate(yf_sup, ell, self.rho) * (1.0 - g_conn_fwd)
                wv_sup = max(w_valley_chart, wv_pt)
                pos_mask = d_yy_sup > 0.0
                med_d = (
                    torch.quantile(d_yy_sup[pos_mask], 0.25).clamp_min(1e-5)
                    if bool(pos_mask.any())
                    else ell_f_base
                )
                if g_width_active and ell_y_loc.ndim == 1:
                    bw_kde = (1.15 * med_d * (ell_y_loc.detach().float() / ell_f_base)).clamp_min(1e-5)
                    denom_kde = (2.0 * bw_kde.unsqueeze(1).square()).clamp_min(1e-10)
                else:
                    denom_kde = (2.0 * (1.15 * med_d).square()).clamp_min(1e-10)
                rho_kde = torch.exp(-d_yy_sup.square() / denom_kde).sum(dim=1)
                eta_kde = float(
                    (torch.quantile(rho_kde, 0.85) / torch.quantile(rho_kde, 0.03).clamp_min(1.0)).item()
                )
                eta_cap = min(20.0, max(0.0, wv_sup * eta_kde))
                if record is not None and hasattr(record, "t1c_het_initialized"):
                    if can_mutate:
                        if not record.t1c_het_initialized:
                            record.t1c_het_ema = min(eta_cap, 3.0)
                            record.t1c_het_initialized = True
                        else:
                            record.t1c_het_ema = 0.92 * float(record.t1c_het_ema) + 0.08 * eta_cap
                    het_val = float(record.t1c_het_ema) if record.t1c_het_initialized else min(eta_cap, 3.0)
                else:
                    het_val = eta_cap
                u_het = min(1.0, max(0.0, (het_val - 3.4) / 1.0))
                g_het = u_het * u_het * (3.0 - 2.0 * u_het)
                if g_het > 1e-4:
                    d_yy_sup.fill_diagonal_(float("inf"))
                    nn1_y_sup = d_yy_sup.amin(dim=1)
                    nn2_y_sup = torch.topk(
                        d_yy_sup, k=min(2, b_r - 1), dim=1, largest=False
                    ).values[:, -1]
                    u_sp = ((nn2_y_sup / ell_f_base - 2.8) / 1.4).clamp(0.0, 1.0)
                    s_sp = float(g_het) * (u_sp * u_sp * (3.0 - 2.0 * u_sp))
                    if bool((s_sp > 1e-4).any()):
                        base_y = ell_y_loc.detach().float() if ell_y_loc.ndim == 1 else ell_f_base.expand(b_r)
                        ell_y_f = torch.maximum(
                            base_y,
                            ell_f_base + s_sp * (0.64 * nn2_y_sup - ell_f_base).clamp(min=0.0, max=3.4 * ell_f_base),
                        )
                        ell_y_loc = ell_y_f.to(ell.dtype)
                        sc_xy_s = torch.maximum(xf_sup.abs().amax(), yf_sup.abs().amax()).clamp_min(1.0)
                        d_xy_sup = torch.cdist(xf_sup / sc_xy_s, yf_sup / sc_xy_s, p=2.0) * sc_xy_s
                        near_d, near_y_idx = d_xy_sup.min(dim=1)
                        near_ell = ell_y_f[near_y_idx]
                        near_s_sp = s_sp[near_y_idx]
                        u_near = ((2.8 * near_ell - near_d) / (1.2 * near_ell)).clamp(0.0, 1.0)
                        s_near = near_s_sp * (u_near * u_near * (3.0 - 2.0 * u_near))
                        s_near_sup = s_near
                        base_x = (
                            ell_x_ms
                            if (self.multiscale_horizon and b_f >= 12 and b_r >= 12)
                            else ell_f_base.expand(b_f)
                        )
                        ell_x_loc = torch.maximum(
                            base_x,
                            ell_f_base + s_near * (near_ell - ell_f_base),
                        ).to(ell.dtype)
                        ell_xy = torch.sqrt(ell_x_loc.unsqueeze(1) * ell_y_loc.unsqueeze(0))
                        ell_xx = torch.sqrt(ell_x_loc.unsqueeze(1) * ell_x_loc.unsqueeze(0))

        c_phys_xy, c_hat_xy = self._born_infeld_pair_fp32(x_aug, y_aug, ell_xy)
        c_phys_xx, c_hat_xx = self._born_infeld_pair_fp32(x_aug, x_aug, ell_xx)

        d_fake_tilde, d_real_tilde = self._centered_log_odds(
            d_fake, d_real, ell=ell, detach_real=not symmetric
        )

        mask_xy = self._discrete_pair_mask(cond_fake, cond_real, b_f, b_r)
        mask_xx = self._discrete_pair_mask(cond_fake, cond_fake, b_f, b_f)
        P_xy = self._sinkhorn_slack(
            c_hat_xy, d_fake_tilde.detach(), d_real_tilde.detach(), hollow=False, pair_mask=mask_xy
        )
        use_hollow = self.self_coupling == "hollow" or (
            x_fake.ndim >= 2
            and not _has_spatial_dims(x_fake)
            and int(x_fake[0].numel()) >= 256
            and g_atom_fwd <= 0.1
        )
        dedup_xx = self.self_coupling == "feydy_unique" or (
            self.self_coupling in ("feydy_discrete", "feydy_atomic", "feydy_atomic_v15") and discrete_step
        )
        eff_mask_xx = mask_xx
        if dedup_xx and b_f > 1:
            diag_f = torch.eye(b_f, dtype=torch.bool, device=x_fake.device)
            unique_xx = (c_hat_xx > 1e-12) | diag_f
            eff_mask_xx = unique_xx if mask_xx is None else (mask_xx & unique_xx)
        zeros_f = d_fake_tilde.new_zeros(b_f)
        P_xx = self._sinkhorn_slack(
            c_hat_xx, zeros_f, zeros_f, hollow=use_hollow, pair_mask=eff_mask_xx
        )

        P_xy_act = P_xy[:b_f, :b_r]
        P_xy_slack = P_xy[:b_f, b_r]
        self.last_p_xy_slack = P_xy_slack.detach()
        P_xx_act = P_xx[:b_f, :b_f].clone()
        P_xx_act.fill_diagonal_(0.0)

        a_f = (float(b_f) * P_xy_act.sum(dim=1)).clamp(0.0, 1.0)
        if s_near_sup is not None:
            a_f = (a_f + 0.50 * s_near_sup.to(a_f.dtype)).clamp(0.0, 1.0)
        P_xx_eff = P_xx_act * torch.sqrt(a_f.unsqueeze(1) * a_f.unsqueeze(0))

        slack_scale = ell.detach() if self.scale_log_odds else 1.0
        transport_xy = (P_xy_act * c_phys_xy).sum()
        s_f = (float(b_f) * P_xy_slack).clamp(0.0, 1.0)
        p_slack_eff = (
            P_xy_slack * (1.0 + self.slack_gain * s_f) if self.slack_gain > 0.0 else P_xy_slack
        )
        if absent_shield is not None:
            p_slack_eff = p_slack_eff * (1.0 - absent_shield.to(p_slack_eff.dtype))

        if self.relativistic_weight > 0.0:
            active_rel = (
                P_xy_act * F.softplus(d_real_tilde.unsqueeze(0) - d_fake_tilde.unsqueeze(1))
            ).sum()
            slack_rel = (p_slack_eff * F.softplus(-d_fake_tilde)).sum()
            local_critic_slack = self.relativistic_weight * slack_scale * (active_rel + slack_rel)
        elif self.tau > 0.0:
            active_rel = (
                P_xy_act * (F.softplus(d_real_tilde.unsqueeze(0) - d_fake_tilde.unsqueeze(1)) - math.log(2.0))
            ).sum()
            local_critic_slack = (
                -(p_slack_eff * (0.5 * self.tau * slack_scale * d_fake_tilde)).sum()
                + (0.03 * self.tau * slack_scale) * active_rel
            )
        else:
            local_critic_slack = d_fake_tilde.new_zeros(())
        debias_xx = -0.5 * (P_xx_eff * c_phys_xx).sum()

        coord_total = transport_xy + debias_xx

        # Differentiable C^1 halo-screened single-chart Laguerre commitment
        if (
            self.chart_weight > 0.0
            and chart_targets is not None
            and chart_halos is not None
            and chart_route_weight is not None
            and bool((chart_route_weight > 1e-5).any())
        ):
            x_cf_live, area_ratio_x = deterministic_spatial_pool2d(x_fake, max_spatial=self.max_spatial)
            ell_cf = (ell.detach().float() / area_ratio_x).clamp_min(1e-6)
            diff_cf = x_cf_live - chart_targets.detach().float()
            sc_cf = torch.maximum(diff_cf.detach().abs().amax(), ell_cf).clamp_min(1.0)
            dist_cf = torch.sqrt((diff_cf / sc_cf).square().sum(dim=-1) + 1e-12) * sc_cf
            u_halo = F.relu(dist_cf - chart_halos.detach().float())
            rad_halo = torch.sqrt(ell_cf.square() + u_halo.square())
            bi_halo = (u_halo.square() / (rad_halo + ell_cf).clamp_min(1e-30)) * area_ratio_x
            coord_total = coord_total + self.chart_weight * (
                (chart_route_weight.to(coord_total.dtype) / float(b_f)) * bi_halo.to(coord_total.dtype)
            ).sum()

        if use_support and g_het > 1e-4 and nn1_y_sup is not None:
            with torch.no_grad():
                act_rate_sup = float(P_xy_act.detach().float().sum().item())
                u_bulk = min(1.0, max(0.0, (act_rate_sup - 0.72) / 0.12))
                g_bulk = u_bulk * u_bulk * (3.0 - 2.0 * u_bulk)
                w_orph: Optional[torch.Tensor] = None
                if g_bulk > 1e-4:
                    ell_yy_f = ell_y_loc.detach().float().clamp_min(1e-6)
                    c_hat_yy_nn = (torch.sqrt(ell_yy_f.square() + nn1_y_sup.square()) - ell_yy_f) / ell_yy_f
                    q98_yy = torch.quantile(c_hat_yy_nn, 0.98)
                    tau_gap = torch.maximum(q98_yy.new_tensor(4.2 * self.rho), 2.10 * q98_yy)
                    c_hat_xy_f = c_hat_xy.detach().float()
                    row_nn = c_hat_xy_f.amin(dim=1)
                    u_row = ((row_nn - tau_gap) / (0.40 * tau_gap)).clamp(0.0, 1.0)
                    c_valley = torch.sigmoid((-d_fake_tilde.detach().float() - 0.35) / 0.20)
                    sf_f = s_f.detach().float()
                    sf_gate = sf_f * sf_f * sf_f * sf_f
                    o_row = (
                        float(g_het * g_bulk)
                        * (u_row * u_row * (3.0 - 2.0 * u_row))
                        * c_valley
                        * sf_gate
                    )
                    if bool((o_row > 1e-4).any()):
                        w_orph = torch.softmax(
                            -(c_hat_xy_f - row_nn.unsqueeze(1)) / 0.25, dim=1
                        ).to(c_phys_xy.dtype)
            if w_orph is not None:
                coord_total = coord_total + 2.2 * (
                    (o_row.to(c_phys_xy.dtype) / float(b_f)).unsqueeze(1) * w_orph * c_phys_xy
                ).sum()

        critic_total = local_critic_slack
        if symmetric:
            ell_yy_sym = (
                torch.sqrt(ell_y_loc.unsqueeze(1) * ell_y_loc.unsqueeze(0))
                if ell_y_loc.ndim == 1
                else ell
            )
            c_phys_yy, c_hat_yy = self._born_infeld_pair_fp32(y_aug, y_aug, ell_yy_sym)
            mask_yy = self._discrete_pair_mask(cond_real, cond_real, b_r, b_r)
            eff_mask_yy = mask_yy
            if self.self_coupling == "feydy_unique" and b_r > 1:
                diag_r = torch.eye(b_r, dtype=torch.bool, device=y_target.device)
                unique_yy = (c_hat_yy > 1e-12) | diag_r
                eff_mask_yy = unique_yy if mask_yy is None else (mask_yy & unique_yy)
            zeros_r = d_real_tilde.new_zeros(b_r)
            P_yy = self._sinkhorn_slack(
                c_hat_yy, zeros_r, zeros_r, hollow=use_hollow, pair_mask=eff_mask_yy
            )
            P_yy_act = P_yy[:b_r, :b_r].clone()
            P_yy_act.fill_diagonal_(0.0)
            a_r = (float(b_r) * P_xy_act.sum(dim=0)).clamp(0.0, 1.0)
            P_yy_eff = P_yy_act * torch.sqrt(a_r.unsqueeze(1) * a_r.unsqueeze(0))
            P_xy_col_slack = P_xy[b_f, :b_r]
            s_r = (float(b_r) * P_xy_col_slack).clamp(0.0, 1.0)
            p_col_slack_eff = (
                P_xy_col_slack * (1.0 + self.slack_gain * s_r)
                if self.slack_gain > 0.0
                else P_xy_col_slack
            )
            if self.relativistic_weight > 0.0:
                local_real_slack = self.relativistic_weight * slack_scale * (
                    p_col_slack_eff * F.softplus(d_real_tilde)
                ).sum()
            else:
                local_real_slack = (p_col_slack_eff * (0.5 * self.tau * slack_scale * d_real_tilde)).sum()
            debias_yy = -0.5 * (P_yy_eff * c_phys_yy).sum()
            coord_total = coord_total + debias_yy
            critic_total = critic_total + local_real_slack

        total = coord_total + critic_total
        if self.thermal_tau > 0.0 and record is not None:
            if (
                can_mutate
                and not defer_thermal
                and cond_fake is None
                and cond_real is None
                and b_r >= 4
                and b_f >= 4
                and hasattr(record, "observe_t1c_thermal")
            ):
                r_ed, r_mu_sq, r_spread, r_q90, r_hmax = self._split_two_sample_residuals(
                    x_proj, y_proj, ell=ell, cvx_mu_gain=self.cvx_mu_gain
                )
                r_tail = 0.0
                if (
                    not discrete_step
                    and not symmetric
                    and self.tail_spill_gain > 0.0
                    and b_f >= 8
                    and b_r >= 8
                ):
                    wv_tail = max(
                        w_valley_chart,
                        self._compute_valley_gate(y_proj.detach(), ell, self.rho) * (1.0 - g_conn_fwd),
                    )
                    if wv_tail > 0.0:
                        with torch.no_grad():
                            xf_t = x_proj.detach().float()
                            yr_t = y_proj.detach().float()
                            sc_t = torch.maximum(xf_t.abs().amax(), yr_t.abs().amax()).clamp_min(1.0)
                            d_xy_t = torch.cdist(xf_t / sc_t, yr_t / sc_t, p=2.0)
                            d_yy_t = torch.cdist(yr_t / sc_t, yr_t / sc_t, p=2.0)
                            off_yy = ~torch.eye(b_r, dtype=torch.bool, device=yr_t.device)
                            denom_ed = d_yy_t[off_yy].mean().clamp_min(1e-6)
                            xy_row_nn = d_xy_t.amin(dim=1)
                            yy_loo_nn = d_yy_t.masked_fill(~off_yy, float("inf")).amin(dim=1)
                            prec_tail = torch.quantile(xy_row_nn, 0.96) - torch.quantile(yy_loo_nn, 0.96)
                            r_tail = float((wv_tail * float(self.tail_spill_gain) * prec_tail / denom_ed).item())
                if atomic_active_fwd:
                    r_tail = max(r_tail, float(r_atomic_fwd))
                s_obs = record.observe_t1c_thermal(
                    r_ed,
                    r_mu_sq,
                    r_spread,
                    r_q90,
                    r_hmax,
                    self.thermal_tau,
                    shock_mu_ratio=self.thermal_shock_mu,
                    r_tail=r_tail,
                )
                _publish_thermal(record, x_fake.device, s_obs)
            s_therm = float(getattr(record, "_thermal_scale", 1.0))
            if s_therm < 1.0 and total.requires_grad:
                if getattr(d_fake, "_t1c_thermally_scaled", False):
                    coord_scaled = coord_total * s_therm + coord_total.detach() * (1.0 - s_therm)
                    total = coord_scaled + critic_total
                else:
                    total = total * s_therm + total.detach() * (1.0 - s_therm)

        loss = self.weight * total
        if not return_metrics:
            return loss, {}

        with torch.no_grad():
            m_vals = torch.stack([
                P_xy_act.float().sum(),
                (P_xy_act * (c_hat_xy > self.rho).to(P_xy_act.dtype)).float().sum(),
                ell.float(),
            ]).tolist()
        active_rate = float(m_vals[0])
        return loss, {
            "t1c_active_rate": active_rate,
            "t1c_slack_rate": 1.0 - active_rate,
            "t1c_tunnel_flux": float(m_vals[1]),
            "t1c_ell": float(m_vals[2]),
        }

    def joint_relativistic_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        record: Any = None,
        allow_ell_update: bool = True,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Slack-screened relativistic joint BiGAN loss on ``(G(z), z)`` vs ``(x, E(x))``."""
        if self.weight == 0.0:
            zero = torch.zeros((), device=d_fake.device, dtype=d_fake.dtype)
            if not return_metrics:
                return zero, {}
            ell_cur = (
                float(record.t1c_ell_ema)
                if record is not None and hasattr(record, "t1c_ell_initialized")
                else float(self.ell_ema.item())
            )
            return zero, {"t1c_active_rate": 0.0, "t1c_ell": ell_cur}

        b_f = d_fake.shape[0]
        b_r = d_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=allow_ell_update)

        with torch.no_grad():
            c_hat_xy = self._born_infeld_c_hat_detached(
                x_fake, y_real, ell, max_spatial=self.max_spatial
            )
            df_sg, dr_sg = self._centered_log_odds(
                d_fake.detach(), d_real.detach(), ell=ell, detach_real=True
            )
            mask_xy = self._discrete_pair_mask(cond_fake, cond_real, b_f, b_r)
            P_xy = self._sinkhorn_slack(c_hat_xy, df_sg, dr_sg, hollow=False, pair_mask=mask_xy)
            P_xy_act = P_xy[:b_f, :b_r]
            P_xy_row_slack = P_xy[:b_f, b_r]
            P_xy_col_slack = P_xy[b_f, :b_r]
            self.last_p_xy_slack = P_xy_row_slack

        f_flat = d_fake.reshape(b_f, -1).mean(dim=1) if d_fake.ndim > 1 else d_fake.reshape(b_f)
        r_flat = d_real.reshape(b_r, -1).mean(dim=1) if d_real.ndim > 1 else d_real.reshape(b_r)
        d_bar = 0.5 * (f_flat.detach().mean() + r_flat.detach().mean())

        active_rel = (P_xy_act * F.softplus(r_flat.unsqueeze(0) - f_flat.unsqueeze(1))).sum()
        slack_fake = (P_xy_row_slack * F.softplus(d_bar - f_flat)).sum()
        slack_real = (P_xy_col_slack * F.softplus(r_flat - d_bar)).sum()
        loss = self.weight * (active_rel + slack_fake + slack_real)

        if self.self_coupling in ("feydy_atomic", "feydy_atomic_v15") and record is not None and cond_fake is None and cond_real is None:
            grad_enabled_j = torch.is_grad_enabled()
            in_probe_j = bool(getattr(record, "_in_protected_probe", False))
            can_mutate_j = bool(allow_ell_update and grad_enabled_j and not in_probe_j)
            prior_table_j = getattr(record, "_active_prior_table", None)
            batch_rows_j = getattr(record, "_active_batch_rows", None)
            split = split_joint_table_pair(x_fake, y_real, prior_table_j, batch_rows_j)
            if split is not None:
                fake_x, _fake_z, real_x, real_z = split
                prev_c_j, prev_r_j = self._record_prev_charts(record)
                g_atom_j = compute_discrete_atomic_gate(real_x.detach(), charts=prev_c_j, radii=prev_r_j)
                if g_atom_j > 0.5:
                    rec_csum = getattr(record, "chart_sum", None)
                    rec_ccnt = getattr(record, "chart_count", None)
                    rec_csq = getattr(record, "chart_sq_dev", None)
                    rec_cfm = getattr(record, "chart_fake_mass", None)
                    rec_phase = float(getattr(record, "revival_phase", 0.0))
                    rec_revs = int(getattr(record, "revivals", 0))
                    if can_mutate_j or rec_csum is None or rec_ccnt is None or rec_csq is None or rec_cfm is None:
                        (
                            charts_j,
                            w_real_j,
                            _radii_j,
                            _b_cnt_j,
                            next_sum_j,
                            next_cnt_j,
                            next_sq_j,
                            cur_fm_j,
                            next_phase_j,
                            next_revs_j,
                        ) = update_streaming_charts(
                            real_x.detach(),
                            rec_csum,
                            rec_ccnt,
                            rec_csq,
                            rec_cfm,
                            rec_phase,
                            rec_revs,
                            max_charts=self.chart_capacity,
                            max_spatial=self.max_spatial,
                            allow_dead_revival=False,
                        )
                    else:
                        charts_j = rec_csum.to(device=real_x.device, dtype=torch.float32) / rec_ccnt.to(
                            device=real_x.device, dtype=torch.float32
                        ).clamp_min(1e-8).unsqueeze(1)
                        w_real_j = rec_ccnt.to(device=real_x.device, dtype=torch.float32)
                        w_real_j = w_real_j / w_real_j.sum().clamp_min(1e-8)
                        next_sum_j, next_cnt_j, next_sq_j = rec_csum, rec_ccnt, rec_csq
                        cur_fm_j = rec_cfm.to(device=real_x.device, dtype=torch.float32)
                        next_phase_j, next_revs_j = rec_phase, rec_revs

                    if charts_j.shape[0] >= 3:
                        assert prior_table_j is not None and batch_rows_j is not None
                        (
                            tgt_k_j,
                            nn_cc_j,
                            next_rc_j,
                            next_rd_j,
                            next_rt_j,
                            r_fwd_j,
                        ) = update_atomic_ledger(
                            fake_x.detach(),
                            batch_rows_j,
                            int(prior_table_j.shape[0]),
                            charts_j,
                            w_real_j,
                            getattr(record, "atomic_row_chart", None),
                            getattr(record, "atomic_row_dist", None),
                            getattr(record, "atomic_row_target", None),
                            allow_update=can_mutate_j,
                        )
                        if can_mutate_j:
                            if hasattr(record, "update_atomic_ledger_state"):
                                record.update_atomic_ledger_state(next_rc_j, next_rd_j, next_rt_j)
                            if hasattr(record, "update_charts"):
                                k_j = int(charts_j.shape[0])
                                bfk_j = F.one_hot(tgt_k_j, num_classes=k_j).to(dtype=torch.float32).sum(dim=0) / float(
                                    max(1, b_f)
                                )
                                next_fm_j = 0.90 * cur_fm_j.float() + 0.10 * bfk_j
                                d_scale_j = float(real_x.detach().abs().amax().clamp_min(1e-6).item())
                                record.update_charts(
                                    next_sum_j, next_cnt_j, next_sq_j, next_fm_j, next_phase_j, next_revs_j, d_scale_j
                                )

                        ell_x_j = (0.15 * torch.median(nn_cc_j)).clamp_min(1e-4).to(dtype=fake_x.dtype)
                        c_tgt_j = charts_j[tgt_k_j].detach().to(dtype=fake_x.dtype)
                        l_fwd_j = self._born_infeld_aligned_fp32(fake_x, c_tgt_j, ell_x_j).mean()

                        inv_z_tgt, has_inv, ell_z_j, inv_margin = match_inverse_atomic_targets(
                            real_x.detach(),
                            real_z.detach(),
                            charts_j,
                            next_rt_j,
                            prior_table_j.detach(),
                        )
                        if bool(has_inv.any()):
                            bi_inv = self._born_infeld_aligned_fp32(
                                real_z, inv_z_tgt.to(dtype=real_z.dtype), ell_z_j.to(dtype=real_z.dtype)
                            )
                            w_inv_m = has_inv.to(dtype=bi_inv.dtype)
                            l_inv_j = (w_inv_m * bi_inv).sum() / w_inv_m.sum().clamp_min(1.0)
                            with torch.no_grad():
                                inv_d = (real_z.detach().float() - inv_z_tgt.detach().float()).norm(dim=-1)
                                inv_rel = (inv_d / inv_margin.clamp_min(1e-4)).clamp_max(10.0)
                                r_inv_crisp = 1.6 * float(inv_rel[has_inv].mean().item())
                                inv_unsettled = bool((inv_rel[has_inv] > 0.12).any().item()) or (
                                    not bool(has_inv.all().item())
                                )
                                r_inv_cov = 0.20 if inv_unsettled else 0.0
                        else:
                            l_inv_j = real_z.new_zeros(())
                            r_inv_crisp = 0.0
                            r_inv_cov = 0.25

                        if can_mutate_j and self.thermal_tau > 0.0 and hasattr(record, "observe_t1c_thermal"):
                            r_joint = max(float(r_fwd_j), r_inv_crisp, r_inv_cov)
                            s_obs_j = record.observe_t1c_thermal(
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                self.thermal_tau,
                                shock_mu_ratio=self.thermal_shock_mu,
                                r_tail=r_joint,
                            )
                            _publish_thermal(record, x_fake.device, s_obs_j)

                        w_cb = max(float(self.chart_weight), 1.0)
                        cb_loss = self.weight * w_cb * (l_fwd_j + l_inv_j)
                        s_therm_j = float(getattr(record, "_thermal_scale", 1.0))
                        if s_therm_j < 1.0 and cb_loss.requires_grad:
                            cb_loss = cb_loss * s_therm_j + cb_loss.detach() * (1.0 - s_therm_j)
                        if s_therm_j < 1.0 and loss.requires_grad and not getattr(d_fake, "_t1c_thermally_scaled", False):
                            loss = loss * s_therm_j + loss.detach() * (1.0 - s_therm_j)
                        loss = loss + cb_loss

        if not return_metrics:
            return loss, {}

        with torch.no_grad():
            m_vals = torch.stack([
                P_xy_act.float().sum(),
                (P_xy_act * (c_hat_xy > self.rho).to(P_xy_act.dtype)).float().sum(),
                ell.float(),
            ]).tolist()
        active_rate = float(m_vals[0])
        return loss, {
            "t1c_active_rate": active_rate,
            "t1c_slack_rate": 1.0 - active_rate,
            "t1c_tunnel_flux": float(m_vals[1]),
            "t1c_ell": float(m_vals[2]),
        }

    def conditional_fiber_transport_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        c_cont: torch.Tensor,
        *,
        cond_disc: Optional[torch.Tensor] = None,
        mode: str = "diag_imq_cmmd",
        record: Any = None,
    ) -> torch.Tensor:
        """Unbiased hollow context-screened conditional Born-Infeld discrepancy on ``(x | c)``.

        When ``self.cond_drift_deconflict`` is True, subtracts the exchangeable within-fiber
        midpoint ridge regression ``B_hat_t c_i`` fit on ``M = 0.5 * (X_det + Y_det)``.
        Because ``M_m`` is invariant under pair-swaps ``(X_m, Y_m) <-> (Y_m, X_m)`` at
        ``q_theta == p(. | c)``, ``B_hat_t`` is strictly swap-invariant, guaranteeing exact
        zero finite-sample bias for every ``B >= 2`` while preventing cross-context mean
        drift ``B_t (c_i - c_k)`` from saturating the concave Born-Infeld / IMQ kernel.
        """
        b_n = x_fake.shape[0]
        if b_n <= 1 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
            return x_fake[:0].sum()

        x_flat = x_fake.reshape(b_n, -1)
        y_flat = y_real.detach().reshape(b_n, -1)
        dim = max(1, x_flat.shape[-1])
        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))

        with torch.no_grad():
            same_t = self._discrete_pair_mask(cond_disc, cond_disc, b_n, b_n)
            if same_t is None:
                same_t = torch.ones((b_n, b_n), dtype=torch.bool, device=x_fake.device)
            m_hollow = same_t.clone()
            m_hollow.fill_diagonal_(False)
            valid_row = m_hollow.any(dim=1)
            m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)

            c_f32 = c_cont.detach().reshape(b_n, -1).float()
            d_ctx = c_f32.shape[1]

            drift_offset: Optional[torch.Tensor] = None
            if self.cond_drift_deconflict and 1 <= d_ctx <= 32 and b_n >= 6:
                x_det_f = x_flat.detach().float()
                y_det_f = y_flat.float()
                mid_f = 0.5 * (x_det_f + y_det_f)
                drift_buf = torch.zeros_like(y_det_f)
                eye_c = torch.eye(d_ctx, device=x_fake.device, dtype=torch.float32)

                if cond_disc is None:
                    fiber_masks = [valid_row]
                else:
                    disc_flat = cond_disc.detach().reshape(b_n, -1)
                    uniq_disc = torch.unique(disc_flat, dim=0)
                    fiber_masks = [
                        (disc_flat == uniq_disc[u_i].unsqueeze(0)).all(dim=-1) & valid_row
                        for u_i in range(uniq_disc.shape[0])
                    ]
                for f_mask in fiber_masks:
                    n_f = int(f_mask.sum().item())
                    if n_f < max(6, d_ctx + 2):
                        continue
                    c_sub = c_f32[f_mask]
                    m_sub = mid_f[f_mask]
                    c_cen = c_sub - c_sub.mean(dim=0, keepdim=True)
                    m_cen = m_sub - m_sub.mean(dim=0, keepdim=True)
                    gram_c = (c_cen.mT @ c_cen) / float(n_f)
                    mean_var = torch.diagonal(gram_c).mean()
                    if float(mean_var.item()) <= 1e-10 or not bool(torch.isfinite(mean_var)):
                        continue
                    reg_mat = gram_c + (0.10 * mean_var) * eye_c
                    rhs_mat = (c_cen.mT @ m_cen) / float(n_f)
                    b_hat = torch.linalg.solve(reg_mat, rhs_mat)
                    if bool(torch.isfinite(b_hat).all()):
                        drift_buf[f_mask] = c_cen @ b_hat
                drift_offset = drift_buf.to(dtype=x_flat.dtype)

        if drift_offset is not None:
            x_eff = x_flat - drift_offset
            y_eff = y_flat - drift_offset
        else:
            x_eff = x_flat
            y_eff = y_flat

        with torch.no_grad():
            c_scale = c_f32.abs().amax().clamp_min(1.0)
            c_unit = c_f32 / c_scale
            d_c = torch.cdist(c_unit, c_unit, p=2.0)
            d_c_max = d_c.amax()
            d_c_masked = d_c.masked_fill(~m_valid, float("inf"))
            d_sorted, sort_idx = torch.sort(d_c_masked, dim=1)
            nn_idx = sort_idx[:, 0]

            n_i = m_valid.sum(dim=1)
            k_idx = torch.minimum(torch.full_like(n_i, 2), n_i.clamp_min(1)) - 1
            sigma_raw = d_sorted.gather(1, k_idx.unsqueeze(1)).squeeze(1)
            sigma_floor = 1e-3 * d_c_max + 1e-6
            sigma_i = torch.where(
                valid_row,
                torch.where(torch.isfinite(sigma_raw), sigma_raw, torch.zeros_like(sigma_raw)).clamp_min(0.0)
                + sigma_floor,
                torch.ones_like(sigma_raw),
            )

            y_f32 = y_eff.detach().float()
            y_scale = y_f32.abs().amax().clamp_min(1.0)
            y_unit = y_f32 / y_scale
            delta_nn = torch.linalg.vector_norm(y_unit - y_unit[nn_idx], ord=2, dim=-1) * (
                y_scale / math.sqrt(2.0)
            )
            same_valid = same_t & valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
            fiber_cnt = same_valid.float().sum(dim=1).clamp_min(1.0)
            ell_fiber = ((same_valid.float() @ delta_nn) / fiber_cnt).clamp_min(1e-5)
            ell_fiber = torch.where(valid_row, ell_fiber, torch.ones_like(ell_fiber))

        if mode == "diag_imq_cmmd":
            with torch.no_grad():
                innov_d = (y_unit - y_unit[nn_idx]) * (y_scale / math.sqrt(2.0))
                f_mask_f = same_valid.float()
                f_cnt_f = f_mask_f.sum(dim=1).clamp_min(1.0).unsqueeze(1)
                ell_d_sq_raw = (f_mask_f @ innov_d.square()) / f_cnt_f
                ell_iso_sq = ell_d_sq_raw.mean(dim=-1, keepdim=True)
                ell_d_sq = 0.50 * ell_iso_sq + 0.50 * ell_d_sq_raw
                ell_d_floor = (1e-2 * y_scale / math.sqrt(float(dim))) + 1e-5
                ell_d = torch.where(
                    valid_row.unsqueeze(1),
                    ell_d_sq.sqrt().clamp_min(ell_d_floor),
                    torch.ones_like(ell_d_sq),
                )
                ell_rms = ell_d.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
                w_coord = ell_rms / ell_d
                w_coord = w_coord / w_coord.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
                ell_d_tot = ell_rms * math.sqrt(float(dim))

                denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
                k_c = torch.where(m_valid, torch.exp(-d_c.square() / denom_2s), 0.0)
                deg_c = 1.0 + k_c.sum(dim=1)
                inv_sqrt_deg = torch.rsqrt(deg_c)
                w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
                w_imq_sym = 0.5 * (w_imq + w_imq.T)
                w_imq_dt = w_imq_sym.to(x_flat.dtype)

                sc_w = torch.maximum(
                    torch.maximum(
                        (x_eff.detach().float() * w_coord).abs().amax(),
                        (y_eff.detach().float() * w_coord).abs().amax(),
                    ),
                    ell_d_tot.amax(),
                ).clamp_min(1.0)

            xw = (x_eff.float() * w_coord) / sc_w
            yw = (y_eff.float() * w_coord) / sc_w
            ell_u = (ell_d_tot / sc_w).clamp_min(1e-20)

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
            for s_mult in (0.25, 1.0, 4.0):
                ell_s = ell_u * s_mult
                ell_s2 = ell_s.square()
                rad_s_xy = torch.sqrt(ell_s2 + sq_w_xy)
                psi_xy = psi_xy + (ell_s * sq_w_xy) / (rad_s_xy * (ell_s + rad_s_xy)).clamp_min(1e-30)
                rad_s_xx = torch.sqrt(ell_s2 + sq_w_xx)
                psi_xx = psi_xx + (ell_s * sq_w_xx) / (rad_s_xx * (ell_s + rad_s_xx)).clamp_min(1e-30)
            psi_xy_t = ((psi_xy * (1.0 / 3.0)) * sc_w).clamp_max(fmax).to(x_flat.dtype)
            psi_xx_t = ((psi_xx * (1.0 / 3.0)) * sc_w).clamp_max(fmax).to(x_flat.dtype)

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
                    for s_mult in (0.25, 1.0, 4.0):
                        ell_s = ell_u * s_mult
                        ell_s2 = ell_s.square()
                        rad_s_yy = torch.sqrt(ell_s2 + sq_w_yy)
                        psi_yy = psi_yy + (ell_s * sq_w_yy) / (rad_s_yy * (ell_s + rad_s_yy)).clamp_min(1e-30)
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

            return (w_imq_dt * psi_xy_t).sum() - 0.5 * (w_imq_dt * psi_xx_t).sum()

        c_xy, _ = self._born_infeld_pair_fp32(x_eff, y_eff, ell_fiber.unsqueeze(1))
        c_xx, _ = self._born_infeld_pair_fp32(x_eff, x_eff, ell_fiber.unsqueeze(1))
        with torch.no_grad():
            denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
            k_c = torch.where(m_valid, torch.exp(-d_c.square() / denom_2s), 0.0)
            deg_c = 1.0 + k_c.sum(dim=1)
            inv_sqrt_deg = torch.rsqrt(deg_c)
            w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
            w_dt = (0.5 * (w_imq + w_imq.T)).to(x_flat.dtype)
        return (w_dt * c_xy).sum() - 0.5 * (w_dt * c_xx).sum()

    def simplicial_manifold_loss(
        self,
        z_table: torch.Tensor,
        gen_fn: Any,
        y_real: torch.Tensor,
        *,
        record: Any = None,
        iso_weight: float = 0.45,
        chart_weight: float = 0.0,
        macro_settle: Optional[bool] = None,
        curve_settle: Optional[bool] = None,
    ) -> torch.Tensor:
        """Local k-NN 1-simplex and 2-simplex manifold chart regularizer for discrete ParticlePriors."""
        if (
            isinstance(chart_weight, bool)
            or not isinstance(chart_weight, (int, float))
            or not math.isfinite(chart_weight)
            or chart_weight < 0.0
        ):
            raise ValueError("chart_weight must be finite and nonnegative")
        use_macro = self.simplex_macro_settle if macro_settle is None else macro_settle
        if type(use_macro) is not bool:
            raise ValueError("macro_settle must be a boolean or None")
        use_curve = self.simplex_curve_settle if curve_settle is None else curve_settle
        if type(use_curve) is not bool:
            raise ValueError("curve_settle must be a boolean or None")
        m_p = z_table.shape[0]
        b_r = y_real.shape[0]
        if self.weight == 0.0 or m_p < 2 or b_r < 1:
            return y_real[:0].sum()

        phi = float(getattr(record, "t1c_chord_phase", 0.0)) if record is not None else 0.0
        if m_p > 512:
            idx_p = self._weyl_strided_indices(m_p, 128, phi, z_table.device)
            z_table = z_table[idx_p]
            m_p = 128

        y_flat = project_transport_features(y_real.detach(), max_spatial=self.max_spatial)
        g_macro = 0.0
        g_conn = 0.0
        q90_y_u = 0.1
        if b_r >= 4:
            g_conn_m, g_macro_m, q90_m, y_scale = self._compute_manifold_gates_eff(y_flat, record=record)
            if g_conn_m <= 1e-4:
                # Theory 1 disconnected-target screen: when Vietoris-Rips H_0
                # persistence indicates disconnected target components (or an
                # unspanned bulk), do not force continuous latent-chord
                # interpolation across valleys and skip extra generator passes.
                return y_real[:0].sum()
            q90_y_u = q90_m
            if chart_weight > 0.0:
                g_conn = g_conn_m
                if use_macro:
                    g_macro = g_macro_m
        else:
            y_scale = y_flat.float().abs().amax().clamp_min(1.0)

        with torch.no_grad():
            z_det = z_table.detach()
            ell = self._update_ell(y_real, record=record, allow_update=False)
            z_f = z_det.float()
            z_sc = z_f.abs().amax().clamp_min(1.0)
            d_zz = torch.cdist(z_f / z_sc, z_f / z_sc, p=2.0)
            d_zz = d_zz + 1e-7 * torch.arange(m_p, device=z_det.device, dtype=torch.float32).unsqueeze(0)
            d_zz.fill_diagonal_(float("inf"))
            k_nn = min(4, m_p - 1)
            nbrs = d_zz.topk(k_nn, dim=-1, largest=False).indices
            y_u = y_flat.float() / y_scale

            g_curve = self._compute_curve_gate(y_flat, g_conn) if (use_curve and chart_weight > 0.0) else 0.0
            n_q = max(4, b_r if b_r <= 512 else 128)
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

        if g_conn > 1e-4 and chart_weight > 0.0:
            x_verts_live = project_transport_features(gen_fn(z_table), max_spatial=self.max_spatial)
            x_verts = x_verts_live.detach()
        else:
            x_verts_live = None
            with torch.no_grad():
                x_verts = project_transport_features(gen_fn(z_det).detach(), max_spatial=self.max_spatial)

        x_simp = project_transport_features(gen_fn(z_simp), max_spatial=self.max_spatial)

        if (
            g_macro > 1e-4
            and self.thermal_tau > 0.0
            and record is not None
            and torch.is_grad_enabled()
            and not getattr(record, "_in_protected_probe", False)
            and b_r >= 4
            and hasattr(record, "observe_t1c_thermal")
        ):
            with torch.no_grad():
                x_man = torch.cat([x_verts, x_simp.detach()], dim=0)
                r_ed, r_mu_sq, r_spread, r_q90, r_hmax = self._split_two_sample_residuals(
                    x_man, y_flat, ell=ell, cvx_mu_gain=self.cvx_mu_gain
                )
                r_q90_eff = (1.0 - 0.50 * g_curve) * r_q90 if g_curve > 1e-4 else r_q90
                s_obs = record.observe_t1c_thermal(
                    r_ed,
                    r_mu_sq,
                    r_spread,
                    r_q90_eff,
                    r_hmax,
                    self.thermal_tau * (1.0 + 0.25 * g_macro + 0.60 * g_curve),
                    shock_mu_ratio=self.thermal_shock_mu + 0.55 * g_curve,
                    shock_ed_floor=0.30 + 0.50 * g_curve,
                    lock_hysteresis=0.22 * g_curve,
                    r_tail=0.0,
                )
                _publish_thermal(record, y_real.device, s_obs)

        c_phys_sr, c_hat_sr = self._born_infeld_pair_fp32(x_simp, y_flat, ell)
        w_sr = torch.softmax(-c_hat_sr.float() / self.eps, dim=1).to(c_phys_sr.dtype)
        proj_loss = (w_sr * c_phys_sr).sum(dim=1).mean()

        eff_iso = ((1.0 - g_macro) * float(iso_weight) + g_macro * 0.25) if g_macro > 1e-4 else float(iso_weight)
        iso_loss = proj_loss.new_zeros(())
        if eff_iso > 0.0:
            x_e = x_simp[:n_edge]
            d0_e = self._born_infeld_aligned_fp32(x_e, x_verts[v0_e], ell)
            d1_e = self._born_infeld_aligned_fp32(x_e, x_verts[v1_e], ell)
            iso_edge = ((1.0 - t_e.to(d0_e.dtype)) * d0_e - t_e.to(d0_e.dtype) * d1_e).abs().mean()
            if n_tri > 0:
                x_t = x_simp[n_edge:]
                d0_t = self._born_infeld_aligned_fp32(x_t, x_verts[v0_t], ell)
                d1_t = self._born_infeld_aligned_fp32(x_t, x_verts[v1_t], ell)
                d2_t = self._born_infeld_aligned_fp32(x_t, x_verts[v2_t], ell)
                with torch.no_grad():
                    if g_conn > 1e-4 and chart_weight > 0.0:
                        vx0 = x_verts[v0_t].float()
                        vx1 = x_verts[v1_t].float()
                        vx2 = x_verts[v2_t].float()
                        ea = torch.linalg.vector_norm(vx0 - vx1, ord=2, dim=-1)
                        eb = torch.linalg.vector_norm(vx1 - vx2, ord=2, dim=-1)
                        ec = torch.linalg.vector_norm(vx2 - vx0, ord=2, dim=-1)
                        s_p = 0.5 * (ea + eb + ec)
                        area_sq = (
                            s_p
                            * (s_p - ea).clamp_min(0.0)
                            * (s_p - eb).clamp_min(0.0)
                            * (s_p - ec).clamp_min(0.0)
                        ).clamp_min(0.0)
                        r_in = torch.sqrt(area_sq) / s_p.clamp_min(1e-8)
                        ell_f = ell.detach().clamp_min(1e-6).float()
                        margin_raw = 0.70 * (torch.sqrt(ell_f.square() + r_in.square()) - ell_f)
                        if g_macro > 1e-4:
                            zv0 = z_det[v0_t].float()
                            zv1 = z_det[v1_t].float()
                            zv2 = z_det[v2_t].float()
                            zt_f = z_tri.detach().float()
                            z_diam = torch.maximum(
                                torch.maximum(
                                    torch.linalg.vector_norm(zv0 - zv1, ord=2, dim=-1),
                                    torch.linalg.vector_norm(zv1 - zv2, ord=2, dim=-1),
                                ),
                                torch.linalg.vector_norm(zv2 - zv0, ord=2, dim=-1),
                            ).clamp_min(1e-8)
                            z_min_d = torch.minimum(
                                torch.minimum(
                                    torch.linalg.vector_norm(zt_f - zv0, ord=2, dim=-1),
                                    torch.linalg.vector_norm(zt_f - zv1, ord=2, dim=-1),
                                ),
                                torch.linalg.vector_norm(zt_f - zv2, ord=2, dim=-1),
                            )
                            u_fat = ((z_min_d / z_diam - 0.12) / 0.10).clamp(0.0, 1.0)
                            eta_fat = u_fat * u_fat * (3.0 - 2.0 * u_fat)
                            margin = ((1.0 - g_macro + g_macro * eta_fat) * margin_raw).to(d0_t.dtype)
                        else:
                            margin = margin_raw.to(d0_t.dtype)
                    else:
                        e01 = self._born_infeld_aligned_fp32(x_verts[v0_t], x_verts[v1_t], ell)
                        e12 = self._born_infeld_aligned_fp32(x_verts[v1_t], x_verts[v2_t], ell)
                        e20 = self._born_infeld_aligned_fp32(x_verts[v2_t], x_verts[v0_t], ell)
                        margin = 0.20 * torch.minimum(torch.minimum(e01, e12), e20)
                iso_tri = (
                    F.relu(margin - d0_t) + F.relu(margin - d1_t) + F.relu(margin - d2_t)
                ).mean() / 3.0
                iso_loss = 0.5 * (iso_edge + iso_tri)
            else:
                iso_loss = iso_edge

        total = proj_loss + eff_iso * iso_loss

        z_chart_total = total.new_zeros(())
        if g_conn > 1e-4 and chart_weight > 0.0:
            if x_verts_live is not None:
                c_phys_vy, _ = self._born_infeld_pair_fp32(x_verts_live, y_flat, ell)
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
                    vert_medoid_loss = self._born_infeld_aligned_fp32(
                        x_verts_live, y_medoids, 0.50 * ell
                    ).mean()
                    total = total + g_conn * (
                        1.10 * vert_ot_loss + 0.65 * g_curve * vert_medoid_loss
                    )
                else:
                    total = total + (1.10 * g_conn) * vert_ot_loss

            if z_table.requires_grad:
                with torch.no_grad():
                    y_chart = (
                        y_u
                        if (m_p + b_r) <= 512
                        else y_u[self._weyl_strided_indices(b_r, min(b_r, 128), phi, y_u.device)]
                    )
                    v_cloud = torch.cat([x_verts.float() / y_scale, y_chart], dim=0)
                    d_v = torch.cdist(v_cloud, v_cloud, p=2.0)
                    h_step = max(1e-4, 1.35 * q90_y_u)
                    g_mat = d_v * (1.0 + (d_v / h_step).square())
                    for _ in range(6):
                        g_mat = torch.minimum(g_mat, (g_mat.unsqueeze(2) + g_mat.unsqueeze(0)).amin(dim=1))
                    g_vv = g_mat[:m_p, :m_p].float()
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
                    z_sc_c = z_table.detach().float().abs().amax().clamp_min(1.0)
                z_u_live = z_table.float() / z_sc_c
                dz_sq = (
                    z_u_live.square().sum(dim=-1, keepdim=True)
                    + z_u_live.square().sum(dim=-1).unsqueeze(0)
                    - 2.0 * (z_u_live @ z_u_live.T)
                )
                dz_sq = dz_sq + (dz_sq.detach().clamp_min(0.0) - dz_sq.detach())
                dz_rms = torch.sqrt(dz_sq[off_mask].mean().clamp_min(1e-12))
                dz_hat = torch.sqrt(dz_sq + 1e-12) / dz_rms
                chart_stress = (w_chart * (dz_hat - g_hat).square()).sum().to(total.dtype)
                chart_mult = float(chart_weight) * (1.0 + 18.0 * g_curve)
                z_chart_total = z_chart_total + (
                    g_conn * chart_mult
                ) * ell.detach().to(total.dtype) * chart_stress
                if g_curve > 1e-4 and z_table.shape[1] >= 2:
                    z_cf = z_table.float() - z_table.float().mean(dim=0, keepdim=True)
                    frob_sq = z_cf.square().sum().clamp_min(1e-12)
                    spec_sq = torch.linalg.matrix_norm(z_cf, ord=2).square()
                    collinear_loss = ((frob_sq - spec_sq).clamp_min(0.0) / frob_sq).to(total.dtype)
                    z_chart_total = z_chart_total + (
                        3.2 * g_conn * g_curve * float(chart_weight)
                    ) * ell.detach().to(total.dtype) * collinear_loss

        total = total + z_chart_total
        s_therm = float(getattr(record, "_thermal_scale", 1.0)) if record is not None else 1.0
        if s_therm < 1.0 and total.requires_grad:
            total = total * s_therm + total.detach() * (1.0 - s_therm)
        return self.weight * total


_PHI1 = 0.6180339887498949
_PHI2 = 0.7548776662466927


@torch.no_grad()
def _principal_axis_power(gram: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Deterministic zero-RNG power-iteration top eigenvector on symmetric PSD ``[D, D]`` (`0` `eigh`/`svd`)."""
    d = int(gram.shape[0])
    idx = torch.arange(1, d + 1, device=gram.device, dtype=gram.dtype)
    v = torch.remainder(idx * _PHI1 + 0.25, 1.0) - 0.5
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(iters):
        gv = gram @ v
        nrm = gv.norm()
        v = torch.where(nrm > 1e-20, gv / nrm.clamp_min(1e-20), v)
    nz = v.abs() > 1e-6
    if bool(nz.any()):
        first = v[torch.nonzero(nz, as_tuple=False)[0, 0]]
        v = torch.where(first < 0, -v, v)
    return v


def _rayleigh_spec_sq(z_cf: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Differentiable top squared singular value ``||z_cf||_2^2`` via detached power iteration + Rayleigh quotient (`0` SVD)."""
    if z_cf.shape[1] == 2:
        # Exact closed-form 2x2 symmetric quadratic formula (zero iterations, machine precision):
        g00 = (z_cf[:, 0] * z_cf[:, 0]).sum()
        g01 = (z_cf[:, 0] * z_cf[:, 1]).sum()
        g11 = (z_cf[:, 1] * z_cf[:, 1]).sum()
        tr = g00 + g11
        disc = torch.sqrt((g00 - g11).square() + 4.0 * g01.square() + 1e-20)
        return 0.5 * (tr + disc)
    with torch.no_grad():
        zd = z_cf.detach()
        gram = zd.T @ zd
        v_top = _principal_axis_power(gram, iters=iters)
    return (z_cf @ v_top).square().sum()


class _UnifiedV15CouplingV4Base(_UnifiedV15CouplingBase):
    """Zero-`eigvalsh` / zero-`svdvals` subclass of `T1CouplingLoss`."""

    @staticmethod
    @torch.no_grad()
    def _compute_curve_gate(y_flat: torch.Tensor, g_conn: float) -> float:
        """Compute local 1D-curve intrinsic dimension gate via batched power iteration + deflation (`0` `eigvalsh`)."""
        b_r = y_flat.shape[0]
        if g_conn <= 1e-4 or b_r < 12:
            return 0.0
        y_scale = y_flat.float().abs().amax().clamp_min(1.0)
        y_u = y_flat.float() / y_scale
        d_yy = torch.cdist(y_u, y_u, p=2.0)
        k_loc = min(9, b_r)
        if k_loc < 2:
            return float(g_conn)
        idx_k = torch.topk(d_yy, k=k_loc, dim=-1, largest=False).indices
        patches = y_u[idx_k]
        centered = patches - patches.mean(dim=1, keepdim=True)
        gram = torch.bmm(centered, centered.transpose(1, 2))  # [B, k_loc, k_loc]

        ks = torch.arange(1, k_loc + 1, device=gram.device, dtype=torch.float32)
        v1 = (torch.remainder(ks * _PHI1 + 0.13, 1.0) - 0.5).view(1, k_loc, 1).expand(b_r, k_loc, 1).clone()
        v1 = v1 / v1.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for _ in range(6):
            gv = torch.bmm(gram, v1)
            nrm = gv.norm(dim=1, keepdim=True)
            v1 = torch.where(nrm > 1e-18, gv / nrm.clamp_min(1e-18), v1)
        lam1 = (v1 * torch.bmm(gram, v1)).sum(dim=(1, 2)).clamp_min(0.0)

        gram_def = gram - lam1.view(b_r, 1, 1) * torch.bmm(v1, v1.transpose(1, 2))
        v2 = (torch.remainder(ks * _PHI2 + 0.37, 1.0) - 0.5).view(1, k_loc, 1).expand(b_r, k_loc, 1).clone()
        v2 = v2 - (v2 * v1).sum(dim=1, keepdim=True) * v1
        v2 = v2 / v2.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for _ in range(6):
            gv2 = torch.bmm(gram_def, v2)
            gv2 = gv2 - (gv2 * v1).sum(dim=1, keepdim=True) * v1
            nrm2 = gv2.norm(dim=1, keepdim=True)
            v2 = torch.where(nrm2 > 1e-18, gv2 / nrm2.clamp_min(1e-18), v2)
        lam2 = (v2 * torch.bmm(gram_def, v2)).sum(dim=(1, 2)).clamp_min(0.0)

        ratio_21 = torch.median(lam2 / lam1.clamp_min(1e-12))
        u_1d = ((0.11 - ratio_21) / 0.06).clamp(0.0, 1.0)
        s_1d = u_1d * u_1d * (3.0 - 2.0 * u_1d)
        return float((float(g_conn) * s_1d).item())

    def simplicial_manifold_loss(
        self,
        z_table: torch.Tensor,
        gen_fn: Any,
        y_real: torch.Tensor,
        *,
        record: Any = None,
        iso_weight: float = 0.45,
        chart_weight: float = 0.0,
        macro_settle: Optional[bool] = None,
        curve_settle: Optional[bool] = None,
    ) -> torch.Tensor:
        """Zero-SVD / zero-`eigvalsh` simplicial manifold chart regularizer for discrete `ParticlePrior`s."""
        if (
            isinstance(chart_weight, bool)
            or not isinstance(chart_weight, (int, float))
            or not math.isfinite(chart_weight)
            or chart_weight < 0.0
        ):
            raise ValueError("chart_weight must be finite and nonnegative")
        use_macro = self.simplex_macro_settle if macro_settle is None else macro_settle
        if type(use_macro) is not bool:
            raise ValueError("macro_settle must be a boolean or None")
        use_curve = self.simplex_curve_settle if curve_settle is None else curve_settle
        if type(use_curve) is not bool:
            raise ValueError("curve_settle must be a boolean or None")
        m_p = z_table.shape[0]
        b_r = y_real.shape[0]
        if self.weight == 0.0 or m_p < 2 or b_r < 1:
            return y_real[:0].sum()

        phi = float(getattr(record, "t1c_chord_phase", 0.0)) if record is not None else 0.0
        if m_p > 512:
            idx_p = self._weyl_strided_indices(m_p, 128, phi, z_table.device)
            z_table = z_table[idx_p]
            m_p = 128

        y_flat = project_transport_features(y_real.detach(), max_spatial=self.max_spatial)
        g_macro = 0.0
        g_conn = 0.0
        q90_y_u = 0.1
        if b_r >= 4:
            g_conn_m, g_macro_m, q90_m, y_scale = self._compute_manifold_gates_eff(y_flat, record=record)
            if g_conn_m <= 1e-4:
                return y_real[:0].sum()
            q90_y_u = q90_m
            if chart_weight > 0.0:
                g_conn = g_conn_m
                if use_macro:
                    g_macro = g_macro_m
        else:
            y_scale = y_flat.float().abs().amax().clamp_min(1.0)

        with torch.no_grad():
            z_det = z_table.detach()
            ell = self._update_ell(y_real, record=record, allow_update=False)
            z_f = z_det.float()
            z_sc = z_f.abs().amax().clamp_min(1.0)
            d_zz = torch.cdist(z_f / z_sc, z_f / z_sc, p=2.0)
            d_zz = d_zz + 1e-7 * torch.arange(m_p, device=z_det.device, dtype=torch.float32).unsqueeze(0)
            d_zz.fill_diagonal_(float("inf"))
            k_nn = min(4, m_p - 1)
            nbrs = d_zz.topk(k_nn, dim=-1, largest=False).indices
            y_u = y_flat.float() / y_scale

            g_curve = self._compute_curve_gate(y_flat, g_conn) if (use_curve and chart_weight > 0.0) else 0.0
            n_q = max(4, b_r if b_r <= 512 else 128)
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
                phi_t = (phi * _PHI2) % 1.0
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

        if g_conn > 1e-4 and chart_weight > 0.0:
            x_verts_live = project_transport_features(gen_fn(z_table), max_spatial=self.max_spatial)
            x_verts = x_verts_live.detach()
        else:
            x_verts_live = None
            with torch.no_grad():
                x_verts = project_transport_features(gen_fn(z_det).detach(), max_spatial=self.max_spatial)

        x_simp = project_transport_features(gen_fn(z_simp), max_spatial=self.max_spatial)

        if (
            g_macro > 1e-4
            and self.thermal_tau > 0.0
            and record is not None
            and torch.is_grad_enabled()
            and not getattr(record, "_in_protected_probe", False)
            and b_r >= 4
            and hasattr(record, "observe_t1c_thermal")
        ):
            with torch.no_grad():
                x_man = torch.cat([x_verts, x_simp.detach()], dim=0)
                r_ed, r_mu_sq, r_spread, r_q90, r_hmax = self._split_two_sample_residuals(
                    x_man, y_flat, ell=ell, cvx_mu_gain=self.cvx_mu_gain
                )
                r_q90_eff = (1.0 - 0.50 * g_curve) * r_q90 if g_curve > 1e-4 else r_q90
                s_obs = record.observe_t1c_thermal(
                    r_ed,
                    r_mu_sq,
                    r_spread,
                    r_q90_eff,
                    r_hmax,
                    self.thermal_tau * (1.0 + 0.25 * g_macro + 0.60 * g_curve),
                    shock_mu_ratio=self.thermal_shock_mu + 0.55 * g_curve,
                    shock_ed_floor=0.30 + 0.50 * g_curve,
                    lock_hysteresis=0.22 * g_curve,
                    r_tail=0.0,
                )
                _publish_thermal(record, y_real.device, s_obs)

        c_phys_sr, c_hat_sr = self._born_infeld_pair_fp32(x_simp, y_flat, ell)
        w_sr = torch.softmax(-c_hat_sr.float() / self.eps, dim=1).to(c_phys_sr.dtype)
        proj_loss = (w_sr * c_phys_sr).sum(dim=1).mean()

        eff_iso = ((1.0 - g_macro) * float(iso_weight) + g_macro * 0.25) if g_macro > 1e-4 else float(iso_weight)
        iso_loss = proj_loss.new_zeros(())
        if eff_iso > 0.0:
            x_e = x_simp[:n_edge]
            d0_e = self._born_infeld_aligned_fp32(x_e, x_verts[v0_e], ell)
            d1_e = self._born_infeld_aligned_fp32(x_e, x_verts[v1_e], ell)
            iso_edge = ((1.0 - t_e.to(d0_e.dtype)) * d0_e - t_e.to(d0_e.dtype) * d1_e).abs().mean()
            if n_tri > 0:
                x_t = x_simp[n_edge:]
                d0_t = self._born_infeld_aligned_fp32(x_t, x_verts[v0_t], ell)
                d1_t = self._born_infeld_aligned_fp32(x_t, x_verts[v1_t], ell)
                d2_t = self._born_infeld_aligned_fp32(x_t, x_verts[v2_t], ell)
                with torch.no_grad():
                    if g_conn > 1e-4 and chart_weight > 0.0:
                        vx0 = x_verts[v0_t].float()
                        vx1 = x_verts[v1_t].float()
                        vx2 = x_verts[v2_t].float()
                        ea = torch.linalg.vector_norm(vx0 - vx1, ord=2, dim=-1)
                        eb = torch.linalg.vector_norm(vx1 - vx2, ord=2, dim=-1)
                        ec = torch.linalg.vector_norm(vx2 - vx0, ord=2, dim=-1)
                        s_p = 0.5 * (ea + eb + ec)
                        area_sq = (
                            s_p
                            * (s_p - ea).clamp_min(0.0)
                            * (s_p - eb).clamp_min(0.0)
                            * (s_p - ec).clamp_min(0.0)
                        ).clamp_min(0.0)
                        r_in = torch.sqrt(area_sq) / s_p.clamp_min(1e-8)
                        ell_f = ell.detach().clamp_min(1e-6).float()
                        margin_raw = 0.70 * (torch.sqrt(ell_f.square() + r_in.square()) - ell_f)
                        if g_macro > 1e-4:
                            zv0 = z_det[v0_t].float()
                            zv1 = z_det[v1_t].float()
                            zv2 = z_det[v2_t].float()
                            zt_f = z_tri.detach().float()
                            z_diam = torch.maximum(
                                torch.maximum(
                                    torch.linalg.vector_norm(zv0 - zv1, ord=2, dim=-1),
                                    torch.linalg.vector_norm(zv1 - zv2, ord=2, dim=-1),
                                ),
                                torch.linalg.vector_norm(zv2 - zv0, ord=2, dim=-1),
                            ).clamp_min(1e-8)
                            z_min_d = torch.minimum(
                                torch.minimum(
                                    torch.linalg.vector_norm(zt_f - zv0, ord=2, dim=-1),
                                    torch.linalg.vector_norm(zt_f - zv1, ord=2, dim=-1),
                                ),
                                torch.linalg.vector_norm(zt_f - zv2, ord=2, dim=-1),
                            )
                            u_fat = ((z_min_d / z_diam - 0.12) / 0.10).clamp(0.0, 1.0)
                            eta_fat = u_fat * u_fat * (3.0 - 2.0 * u_fat)
                            margin = ((1.0 - g_macro + g_macro * eta_fat) * margin_raw).to(d0_t.dtype)
                        else:
                            margin = margin_raw.to(d0_t.dtype)
                    else:
                        e01 = self._born_infeld_aligned_fp32(x_verts[v0_t], x_verts[v1_t], ell)
                        e12 = self._born_infeld_aligned_fp32(x_verts[v1_t], x_verts[v2_t], ell)
                        e20 = self._born_infeld_aligned_fp32(x_verts[v2_t], x_verts[v0_t], ell)
                        margin = 0.20 * torch.minimum(torch.minimum(e01, e12), e20)
                iso_tri = (
                    F.relu(margin - d0_t) + F.relu(margin - d1_t) + F.relu(margin - d2_t)
                ).mean() / 3.0
                iso_loss = 0.5 * (iso_edge + iso_tri)
            else:
                iso_loss = iso_edge

        total = proj_loss + eff_iso * iso_loss

        z_chart_total = total.new_zeros(())
        if g_conn > 1e-4 and chart_weight > 0.0:
            if x_verts_live is not None:
                c_phys_vy, _ = self._born_infeld_pair_fp32(x_verts_live, y_flat, ell)
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
                    vert_medoid_loss = self._born_infeld_aligned_fp32(
                        x_verts_live, y_medoids, 0.50 * ell
                    ).mean()
                    total = total + g_conn * (
                        1.10 * vert_ot_loss + 0.65 * g_curve * vert_medoid_loss
                    )
                else:
                    total = total + (1.10 * g_conn) * vert_ot_loss

            if z_table.requires_grad:
                with torch.no_grad():
                    y_chart = (
                        y_u
                        if (m_p + b_r) <= 512
                        else y_u[self._weyl_strided_indices(b_r, min(b_r, 128), phi, y_u.device)]
                    )
                    v_cloud = torch.cat([x_verts.float() / y_scale, y_chart], dim=0)
                    d_v = torch.cdist(v_cloud, v_cloud, p=2.0)
                    h_step = max(1e-4, 1.35 * q90_y_u)
                    g_mat = d_v * (1.0 + (d_v / h_step).square())
                    for _ in range(6):
                        g_mat = torch.minimum(g_mat, (g_mat.unsqueeze(2) + g_mat.unsqueeze(0)).amin(dim=1))
                    g_vv = g_mat[:m_p, :m_p].float()
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
                    z_sc_c = z_table.detach().float().abs().amax().clamp_min(1.0)
                z_u_live = z_table.float() / z_sc_c
                dz_sq = (
                    z_u_live.square().sum(dim=-1, keepdim=True)
                    + z_u_live.square().sum(dim=-1).unsqueeze(0)
                    - 2.0 * (z_u_live @ z_u_live.T)
                )
                dz_sq = dz_sq + (dz_sq.detach().clamp_min(0.0) - dz_sq.detach())
                dz_rms = torch.sqrt(dz_sq[off_mask].mean().clamp_min(1e-12))
                dz_hat = torch.sqrt(dz_sq + 1e-12) / dz_rms
                chart_stress = (w_chart * (dz_hat - g_hat).square()).sum().to(total.dtype)
                chart_mult = float(chart_weight) * (1.0 + 18.0 * g_curve)
                z_chart_total = z_chart_total + (
                    g_conn * chart_mult
                ) * ell.detach().to(total.dtype) * chart_stress
                if g_curve > 1e-4 and z_table.shape[1] >= 2:
                    z_cf = z_table.float() - z_table.float().mean(dim=0, keepdim=True)
                    frob_sq = z_cf.square().sum().clamp_min(1e-12)
                    spec_sq = _rayleigh_spec_sq(z_cf, iters=6)
                    collinear_loss = ((frob_sq - spec_sq).clamp_min(0.0) / frob_sq).to(total.dtype)
                    z_chart_total = z_chart_total + (
                        3.2 * g_conn * g_curve * float(chart_weight)
                    ) * ell.detach().to(total.dtype) * collinear_loss

        total = total + z_chart_total
        s_therm = float(getattr(record, "_thermal_scale", 1.0)) if record is not None else 1.0
        if s_therm < 1.0 and total.requires_grad:
            total = total * s_therm + total.detach() * (1.0 - s_therm)
        return self.weight * total


_POWER_INIT_CACHE: Dict[Tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_power_init_helpers(d: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (d, device, dtype)
    cached = _POWER_INIT_CACHE.get(key)
    if cached is None:
        idx = torch.arange(1, d + 1, device=device, dtype=dtype)
        v0 = torch.remainder(idx * _PHI1 + 0.25, 1.0) - 0.5
        v0 = (v0 / v0.norm().clamp_min(1e-12)).unsqueeze(1)
        idx0 = torch.arange(d, device=device)
        cached = (v0, idx0)
        if len(_POWER_INIT_CACHE) < 64:
            _POWER_INIT_CACHE[key] = cached
    return cached


@torch.no_grad()
def _principal_axis_power_zerosync(
    gram: torch.Tensor,
    iters: int = 6,
    *,
    canonicalize_sign: bool = True,
) -> torch.Tensor:
    """Zero-sync deterministic power-iteration top eigenvector on symmetric PSD ``[D, D]``."""
    d = int(gram.shape[0])
    v0_col, idx0 = _get_power_init_helpers(d, gram.device, gram.dtype)
    if iters <= 0:
        v = v0_col.squeeze(1)
    else:
        # For any symmetric G, ||G v_t|| is monotonically non-decreasing in t (Cauchy-Schwarz),
        # so checking ||G v_0|| > 1e-20 once is mathematically equivalent to per-iteration checks.
        gv = torch.mm(gram, v0_col)
        nrm0 = gv.norm()
        active = nrm0 > 1e-20
        v_col = gv / nrm0.clamp_min(1e-20)
        for _ in range(iters - 1):
            gv = torch.mm(gram, v_col)
            v_col = gv / gv.norm().clamp_min(1e-20)
        v = torch.where(active, v_col, v0_col).squeeze(1)
    if not canonicalize_sign:
        return v
    nz = v.abs() > 1e-6
    pos = torch.where(nz, idx0, d)
    first = v.index_select(0, pos.argmin().unsqueeze(0)).squeeze(0)
    return torch.where(nz.any() & (first < 0), -v, v)


class UnifiedV15CouplingLoss(_UnifiedV15CouplingV4Base):
    """Zero-`eigvalsh` / zero-`svdvals` coupling primitive for Unified-v15."""

    @torch.no_grad()
    def _sinkhorn_slack(
        self,
        c_hat: torch.Tensor,
        d_row: torch.Tensor,
        d_col: torch.Tensor,
        hollow: bool = False,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bit-exact 1-buffer in-place Sinkhorn-slack fastpath (eliminates 2*iters logsumexp allocations)."""
        if hollow or pair_mask is not None or self.iters < 1:
            return super()._sinkhorn_slack(c_hat, d_row, d_col, hollow=hollow, pair_mask=pair_mask)
        b_r, b_c = c_hat.shape
        energy = self._screening_energy(c_hat)
        L = c_hat.new_zeros((b_r + 1, b_c + 1), dtype=torch.float32)
        L[:b_r, :b_c] = -energy / self.eps
        L[:b_r, b_c] = (-0.5 * self.rho + 0.5 * self.tau * d_row.float()) / self.eps
        L[b_r, :b_c] = (-0.5 * self.rho - 0.5 * self.tau * d_col.float()) / self.eps

        log_row = L.new_full((b_r + 1,), -math.log(max(1, b_r)))
        log_row[b_r] = math.log(2.0)
        log_col = L.new_full((b_c + 1,), -math.log(max(1, b_c)))
        log_col[b_c] = math.log(2.0)

        buf = torch.empty_like(L)
        m0 = L.amax(dim=0, keepdim=True)
        torch.sub(L, m0, out=buf)
        buf.exp_()
        v = log_col - (m0.squeeze(0) + buf.sum(dim=0).log_())
        torch.add(L, v.unsqueeze(0), out=buf)
        m1 = buf.amax(dim=1, keepdim=True)
        buf.sub_(m1).exp_()
        u = log_row - (m1.squeeze(1) + buf.sum(dim=1).log_())
        for _ in range(self.iters - 1):
            torch.add(L, u.unsqueeze(1), out=buf)
            m0 = buf.amax(dim=0, keepdim=True)
            buf.sub_(m0).exp_()
            v = log_col - (m0.squeeze(0) + buf.sum(dim=0).log_())
            torch.add(L, v.unsqueeze(0), out=buf)
            m1 = buf.amax(dim=1, keepdim=True)
            buf.sub_(m1).exp_()
            u = log_row - (m1.squeeze(1) + buf.sum(dim=1).log_())
        torch.add(L, u.unsqueeze(1), out=buf)
        buf.add_(v.unsqueeze(0)).exp_()
        return buf.to(c_hat.dtype)


def v15_zero_eigh_bipartite_deficit_macro_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_hat_xy: torch.Tensor,
    p_row_slack: torch.Tensor,
    p_col_slack: torch.Tensor,
    *,
    rho: float = 3.0,
    g_conn: float = 0.0,
) -> torch.Tensor:
    """Scale-invariant local-NN & Poisson-deficit-gated zero-`eigh` bipartite Monge macro-transport.

    Combines Born-Infeld proximity screening with the scale-invariant local-NN orphan ratio
    ``r_loc(y_j) = min_i ||x_i - y_j|| / NN_4(y_j; Y_real)`` so covered broad clusters next to
    narrow cores (``t3_iso_scale_mass_8x``, ``r_loc <= 1.0``) stay strictly zero while genuinely
    orphaned/shifted modes (``t3_iso_horizon_saddle_shift``, ``r_loc >> 3``) trigger full Monge rescue.
    """
    b_f = x_fake.shape[0]
    b_r = y_real.shape[0]
    zero = x_fake[:0].sum()
    if b_f < 2 or b_r < 2 or g_conn > 0.05:
        return zero

    with torch.no_grad():
        c_f = c_hat_xy.detach().float()
        col_min_c = c_f.amin(dim=0)
        # D-08: align proximity onset directly with Sinkhorn slack screening crossover (1.65*rho,
        # full at 3.00*rho) so no screening dead zone exists between local OT screening and macro
        # orphan rescue while arriving modes settle cleanly; broad covered modes (r_loc <= 1.0)
        # remain strictly gated off by g_loc below.
        u_orph = ((col_min_c - 1.65 * float(rho)) / max(1.35 * float(rho), 1e-6)).clamp(0.0, 1.0)
        g_prox = u_orph * u_orph * (3.0 - 2.0 * u_orph)

        xf = x_fake.detach().reshape(b_f, -1).float()
        yf = y_real.detach().reshape(b_r, -1).float()
        scale = torch.maximum(xf.abs().amax(), yf.abs().amax()).clamp_min(1.0)
        d_xy = torch.cdist(xf / scale, yf / scale, p=2.0) * scale

        g_orph = g_prox
        if b_r >= 8 and b_f >= 8:
            yu = yf / scale
            d_yy = torch.cdist(yu, yu, p=2.0) * scale
            d_yy_inf = d_yy.clone()
            d_yy_inf.fill_diagonal_(float("inf"))

            # Scale-invariant k-NN density-ratio orphan gate (k=4): zero on covered broad clusters
            # of arbitrary bandwidth (r_loc ~= 1.0), active on under-populated shifted modes until mass arrives:
            k_loc = min(4, b_r - 1, b_f)
            nn_loc = torch.topk(d_yy_inf, k=k_loc, dim=1, largest=False).values[:, -1].clamp_min(1e-6)
            d_xy_k = torch.topk(d_xy, k=k_loc, dim=0, largest=False).values[-1]
            r_loc = d_xy_k / nn_loc
            u_loc = ((r_loc - 3.52) / 1.30).clamp(0.0, 1.0)
            g_loc = u_loc * u_loc * (3.0 - 2.0 * u_loc)
            g_prox = g_prox * g_loc

            med_d1 = torch.quantile(d_yy_inf.amin(dim=1), 0.50).clamp_min(1e-6)
            k_far_m = min(b_r - 1, 2)
            j_far_m = torch.topk(d_yy, k=k_far_m, dim=-1, largest=True).indices[:, -1]
            y_mid = 0.5 * (yu + yu[j_far_m])
            d_my = torch.cdist(y_mid, yu, p=2.0) * scale
            idx_r = torch.arange(b_r, device=yf.device)
            d_my[idx_r, idx_r] = float("inf")
            d_my[idx_r, j_far_m] = float("inf")
            valley_q85 = torch.quantile(d_my.amin(dim=1), 0.85)
            u_val = ((valley_q85 / med_d1 - 3.20) / 1.80).clamp(0.0, 1.0)
            w_val = float((u_val * u_val * (3.0 - 2.0 * u_val)).item())
            if w_val > 1e-4:
                h_bas = (0.38 * valley_q85).clamp_min(1e-4)
                a_yy = torch.exp(-0.5 * (d_yy / h_bas).square())
                pi_y = a_yy.mean(dim=1).clamp(1e-4, 1.0)
                a_xy = torch.exp(-0.5 * (d_xy / h_bas).square())
                pi_x = a_xy.mean(dim=0)
                u_zd = ((float(b_f) * pi_y - 4.2) / 2.3).clamp(0.0, 1.0)
                s_zd = u_zd * u_zd * (3.0 - 2.0 * u_zd)
                u_starv = ((0.45 - pi_x / pi_y) / 0.20).clamp(0.0, 1.0)
                s_starv = u_starv * u_starv * (3.0 - 2.0 * u_starv)
                g_orph = g_prox * ((1.0 - w_val) + w_val * s_zd * s_starv)
            else:
                g_orph = g_prox

        s_col = (float(b_r) * p_col_slack.detach().float()).clamp(0.0, 1.0)
        q_raw = s_col * g_orph
        m_orph_val = float(q_raw.mean().item())
        if m_orph_val <= 1e-4:
            return zero

        s_row = (float(b_f) * p_row_slack.detach().float()).clamp(0.0, 1.0)
        med_d = torch.quantile(d_xy.reshape(-1), 0.50).clamp_min(1e-4)
        ell_macro = (0.25 * med_d).clamp_min(1e-5)

        idx_f = torch.arange(b_f, device=x_fake.device, dtype=torch.float32)
        det_tie = 1e-4 * torch.remainder((idx_f + 0.5) * _PHI1, 1.0)

        if m_orph_val >= 0.85:
            k_exp = b_f
            exp_idx = torch.arange(b_f, device=x_fake.device, dtype=torch.long)
        else:
            k_exp = min(b_f - 1, int(round(float(b_f) * m_orph_val)))
            if k_exp < 1:
                return zero
            orph_active = (g_orph > 0.05).float()
            has_orph = (orph_active > 0.0).any()
            d_orph_min = torch.where(
                has_orph,
                (d_xy + (1.0 - orph_active).unsqueeze(0) * (1e6 * med_d)).amin(dim=1),
                d_xy.amin(dim=1),
            )
            score = 1.50 * s_row - 0.65 * (d_orph_min / med_d).clamp_max(4.0) + det_tie
            exp_idx = torch.sort(torch.topk(score, k=k_exp, largest=True).indices).values

        p_marg = torch.full((k_exp,), 1.0 / float(k_exp), device=x_fake.device, dtype=torch.float32)
        q_marg = q_raw + 1e-7
        q_marg = q_marg / q_marg.sum().clamp_min(1e-12)

        if yf.shape[1] >= 2 and b_r >= 2:
            y_mean_c = (q_marg.unsqueeze(1) * yf).sum(dim=0, keepdim=True)
            yc_c = (yf - y_mean_c) * q_marg.sqrt().unsqueeze(1)
            gram_c = (yc_c.T @ yc_c).double()
            axis_c = _principal_axis_power_zerosync(gram_c, iters=6).float()
            col_order = torch.argsort(yf @ axis_c)
        else:
            col_order = torch.argsort(yf[:, 0])
        q_ord = q_marg[col_order]

        cost_norm = d_xy[exp_idx][:, col_order] / med_d
        L_mac = -cost_norm / 0.03
        log_p = p_marg.log()
        log_q = q_ord.clamp_min(1e-12).log()
        u = torch.zeros_like(log_p)
        v = torch.zeros_like(log_q)
        for _ in range(10):
            v = log_q - torch.logsumexp(L_mac + u.unsqueeze(1), dim=0)
            u = log_p - torch.logsumexp(L_mac + v.unsqueeze(0), dim=1)
        pi_mac = torch.exp(L_mac + u.unsqueeze(1) + v.unsqueeze(0))

        row_cond = pi_mac / pi_mac.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cdf = torch.cumsum(row_cond, dim=1)
        cdf[:, -1] = 1.0
        idx_e_long = torch.arange(k_exp, device=x_fake.device, dtype=torch.long)
        half = (k_exp + 1) // 2
        slot = torch.where(idx_e_long % 2 == 0, idx_e_long // 2, half + (idx_e_long // 2))
        q_u = ((slot.float() + 0.5) / float(k_exp)).clamp(1e-6, 1.0 - 1e-6).unsqueeze(1)
        j_ord = torch.searchsorted(cdf, q_u).squeeze(1).clamp(0, b_r - 1)
        y_matched = y_real.detach().reshape(b_r, -1)[col_order[j_ord]]

    chord_cost = born_infeld_aligned(x_fake.reshape(b_f, -1)[exp_idx], y_matched, ell_macro)
    return chord_cost.sum() / float(b_f)


def _odd_radial_saturate(v: torch.Tensor, tau: float) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (tau * torch.tanh(norm / tau)) / norm
    return v * scale


def _odd_frob_saturate(mat: torch.Tensor, tau: float) -> torch.Tensor:
    norm = mat.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    scale = (tau * torch.tanh(norm / tau)) / norm
    return mat * scale


def _group_strata_cpu(
    b_n: int,
    cond_disc: Optional[torch.Tensor],
) -> List[Tuple[Tuple[int, ...], torch.Tensor]]:
    cpu = torch.device("cpu")
    if cond_disc is None:
        return [((0,), torch.arange(b_n, dtype=torch.long, device=cpu))]
    cd_cpu = cond_disc.detach().reshape(b_n, -1).to(device=cpu, dtype=torch.int64)
    buckets: Dict[Tuple[int, ...], List[int]] = {}
    rows = cd_cpu.tolist()
    for i, row in enumerate(rows):
        key = tuple(int(v) for v in row)
        if key not in buckets:
            buckets[key] = [i]
        else:
            buckets[key].append(i)
    return [(key, torch.tensor(buckets[key], dtype=torch.long, device=cpu)) for key in sorted(buckets)]


def unified_v15_conditional_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
    record: Any = None,
    mean_op_weight: float = 0.36,
    cov_op_weight: float = 0.30,
    imq_active_scale: float = 0.40,
    bank_mean_blend: float = 0.85,
) -> torch.Tensor:
    """Zero-`eigh` cross-batch paired-ring + sufficient-statistics conditional fiber discrepancy."""
    b_n = x_fake.shape[0]
    if b_n <= 1 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
        return x_fake[:0].sum()

    x_flat = x_fake.reshape(b_n, -1)
    y_flat = y_real.detach().reshape(b_n, -1)
    orig_dev = x_flat.device
    dim = max(1, x_flat.shape[-1])
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))

    bank: Optional[UnifiedV11CondFiberBank] = (
        getattr(record, "cond_fiber_bank", None) if record is not None else None
    )
    allow_state_update = (
        record is not None
        and torch.is_grad_enabled()
        and not getattr(record, "_in_protected_probe", False)
        and not getattr(record, "_in_probe", False)
    )

    with torch.no_grad():
        same_t = discrete_pair_mask(cond_disc, cond_disc, b_n, b_n)
        if same_t is None:
            same_t = torch.ones((b_n, b_n), dtype=torch.bool, device=orig_dev)
        m_hollow = same_t.clone()
        m_hollow.fill_diagonal_(False)
        valid_row = m_hollow.any(dim=1)
        m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)
        if not bool(m_valid.any()):
            return x_fake[:0].sum()

        c_f32 = c_cont.detach().reshape(b_n, -1).float()
        x_det32 = x_flat.detach().float()
        y_det32 = y_flat.float()
        d_c = int(c_f32.shape[1])

        cpu = torch.device("cpu")
        strata_groups = _group_strata_cpu(b_n, cond_disc)
        cxy_cpu64 = torch.cat((c_f32, x_det32, y_det32), dim=1).to(device=cpu, dtype=torch.float64)
        c_cpu64 = cxy_cpu64[:, :d_c]
        x_cpu64 = cxy_cpu64[:, d_c : d_c + dim]
        y_cpu64 = cxy_cpu64[:, d_c + dim :]

        has_op_row = torch.zeros(b_n, dtype=torch.bool, device=cpu)
        op_w_mat = torch.eye(dim, dtype=torch.float32, device=cpu).unsqueeze(0).expand(b_n, dim, dim).clone()
        op_dm_w = torch.zeros((b_n, dim), dtype=torch.float32, device=cpu)
        op_mx = torch.zeros((b_n, dim), dtype=torch.float32, device=cpu)
        op_ecov_w = torch.zeros((b_n, dim, dim), dtype=torch.float32, device=cpu)
        op_row_w = torch.zeros(b_n, dtype=torch.float32, device=cpu)
        m_pool_cpu = torch.zeros((b_n, dim), dtype=torch.float32, device=cpu)
        m_pool_gate = torch.zeros((b_n, 1), dtype=torch.float32, device=cpu)

        pending_updates: List[Tuple[Tuple[int, ...], torch.Tensor, float]] = []
        r_op_worst = 0.0

        if bank is not None:
            for key, idx_cpu in strata_groups:
                n_g = int(idx_cpu.shape[0])
                if n_g == 0:
                    continue
                c_g = c_cpu64[idx_cpu]
                x_g = x_cpu64[idx_cpu]
                y_g = y_cpu64[idx_cpu]
                q = bank.query_stratum(key, c_g, x_g, y_g)
                z_innov = 0.0
                if q is not None:
                    ring_len = float(q["ring_len"])
                    w_mat_gate = ring_len / (ring_len + 32.0)
                    m_pool_cpu[idx_cpu] = q["m_pool"].float()
                    m_pool_gate[idx_cpu] = float(w_mat_gate)

                    w_mat = q["w_mat"]
                    dm_blend = (1.0 - bank_mean_blend) * q["delta_m_loo"] + bank_mean_blend * q["delta_m_bank"]
                    dm_w = _odd_radial_saturate(dm_blend @ w_mat, tau=3.0).float()

                    w_mat32 = w_mat.float()
                    dcov32 = q["delta_cov_loc"].float()
                    ecov_w = torch.matmul(w_mat32.unsqueeze(0), torch.matmul(dcov32, w_mat32.unsqueeze(0)))
                    ecov_w = _odd_frob_saturate(0.5 * (ecov_w + ecov_w.transpose(-1, -2)), tau=1.5)

                    omega_g = q["omega"].float()
                    omega_norm = omega_g / omega_g.mean().clamp_min(1e-6)

                    has_op_row[idx_cpu] = True
                    op_w_mat[idx_cpu] = w_mat32.unsqueeze(0)
                    op_dm_w[idx_cpu] = dm_w
                    op_mx[idx_cpu] = q["m_x"].float()
                    op_ecov_w[idx_cpu] = ecov_w
                    op_row_w[idx_cpu] = float(w_mat_gate) * omega_norm
                    z_innov = float(q["z_innov"])

                    r_stratum = float(q["mean_err_sq"])
                    if r_stratum > r_op_worst:
                        r_op_worst = r_stratum
                pending_updates.append((key, idx_cpu, z_innov))

        if bool(has_op_row.all()):
            m_hat32 = m_pool_cpu.to(device=orig_dev)
        else:
            z_pool32 = 0.5 * (x_det32 + y_det32)
            m_hat_fallback32 = _zero_shrinkage_pooled_mean(c_f32, z_pool32, cond_disc)
            if bool(has_op_row.any()):
                gate_dev = m_pool_gate.to(device=orig_dev)
                m_hat32 = gate_dev * m_pool_cpu.to(device=orig_dev) + (1.0 - gate_dev) * m_hat_fallback32
            else:
                m_hat32 = m_hat_fallback32

        if bank is not None and allow_state_update:
            for key, idx_cpu, z_innov in pending_updates:
                bank.update_stratum(
                    key,
                    c_cpu64[idx_cpu],
                    x_cpu64[idx_cpu],
                    y_cpu64[idx_cpu],
                    z_innov=z_innov,
                )

        c_scale = c_f32.abs().amax().clamp_min(1.0)
        d_c_mat = torch.cdist(c_f32 / c_scale, c_f32 / c_scale, p=2.0)
        d_c_max = d_c_mat.amax()
        d_c_masked = d_c_mat.masked_fill(~m_valid, float("inf"))
        k_top = min(4, b_n)
        d_sorted, sort_idx = torch.topk(d_c_masked, k=k_top, dim=1, largest=False)
        nn_idx = sort_idx[:, 0]

        n_i = m_valid.sum(dim=1)
        k_idx = torch.minimum(torch.full_like(n_i, min(3, k_top)), n_i.clamp_min(1)) - 1
        sigma_raw = d_sorted.gather(1, k_idx.unsqueeze(1)).squeeze(1)
        sigma_floor = 1e-3 * d_c_max + 1e-6
        sigma_i = torch.where(
            valid_row,
            torch.where(torch.isfinite(sigma_raw), sigma_raw, torch.zeros_like(sigma_raw)).clamp_min(0.0)
            + sigma_floor,
            torch.ones_like(sigma_raw),
        )

        y_res = y_det32 - m_hat32
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
        k_c = torch.where(m_valid, torch.exp(-d_c_mat.square() / denom_2s), 0.0)
        deg_c = 1.0 + k_c.sum(dim=1)
        inv_sqrt_deg = torch.rsqrt(deg_c)
        w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
        w_imq_sym = 0.5 * (w_imq + w_imq.T)
        w_imq_dt = w_imq_sym.to(x_flat.dtype)

        has_any_op = bool(has_op_row.any())
        if has_any_op:
            if cov_op_weight > 0.0:
                op_ecov_dev = op_ecov_w.to(device=orig_dev)
                op_epos_dev, op_eneg_dev = sym_eigensplit_zero_eigh(op_ecov_dev)
            op_w_mat_dev = op_w_mat.to(device=orig_dev)
            op_dm_w_dev = op_dm_w.to(device=orig_dev)
            op_mx_dev = op_mx.to(device=orig_dev)
            op_row_w_dev = op_row_w.to(device=orig_dev)

    x_f32 = x_flat.float()
    xw = (x_f32 - m_hat32) * w_coord
    yw = (y_det32 - m_hat32) * w_coord
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
    loss_imq = (w_imq_dt * psi_xy_t).sum() - 0.5 * (w_imq_dt * psi_xx_t).sum()

    loss_op = x_flat.new_zeros(())
    if has_any_op and (mean_op_weight > 0.0 or cov_op_weight > 0.0):
        inv_dim = 1.0 / float(dim)
        x_w = torch.einsum("nij,nj->ni", op_w_mat_dev, x_f32)
        if mean_op_weight > 0.0:
            mean_inner = (x_w * op_dm_w_dev).sum(dim=-1) * inv_dim
            loss_mean = (op_row_w_dev * mean_inner).mean()
            loss_op = loss_op + float(mean_op_weight) * loss_mean.to(dtype=x_flat.dtype)
        if cov_op_weight > 0.0:
            rx_w = torch.einsum("nij,nj->ni", op_w_mat_dev, x_f32 - op_mx_dev)
            cov_pos = 0.5 * torch.einsum("ni,nij,nj->n", rx_w, op_epos_dev, rx_w) * inv_dim
            rx_norm_sq = rx_w.square().sum(dim=-1, keepdim=True) / float(dim)
            rx_sat = rx_w * torch.rsqrt(1.0 + 0.25 * rx_norm_sq)
            cov_neg = 0.5 * torch.einsum("ni,nij,nj->n", rx_sat, op_eneg_dev, rx_sat) * inv_dim
            loss_cov = (op_row_w_dev * (cov_pos + cov_neg)).mean()
            loss_op = loss_op + float(cov_op_weight) * loss_cov.to(dtype=x_flat.dtype)

    if (
        record is not None
        and hasattr(record, "t1c_cond_initialized")
        and allow_state_update
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
            r_imq = max(-0.5, min(0.5, float((num_c / den_c).item())))
            r_cond = r_imq + min(0.5, 0.11 * max(0.0, float(r_op_worst) - 0.10))
            if not record.t1c_cond_initialized:
                record.t1c_cond_ema = max(0.18, r_cond)
                record.t1c_cond_initialized = True
            else:
                record.t1c_cond_ema = 0.99 * float(record.t1c_cond_ema) + 0.01 * r_cond
            u_cond = min(1.0, max(0.0, (0.050 - max(0.0, float(record.t1c_cond_ema))) / 0.024))
            s_cond = u_cond * u_cond * (3.0 - 2.0 * u_cond)
            if hasattr(record, "_step_scale"):
                record._step_scale = float(1.0 - 0.85 * s_cond)

    imq_scale = float(imq_active_scale) if (has_any_op and (mean_op_weight > 0.0 or cov_op_weight > 0.0)) else 1.0
    return (imq_scale * loss_imq + loss_op).clamp(-fmax, fmax).to(x_flat.dtype)


@torch.no_grad()
def _highd_bank_assign_and_update(
    bank: Any,
    x_det: torch.Tensor,
    y_det: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Update checkpointed high-D streaming basin statistics (0 eigh, Haar-O(d) invariant)."""
    dev = y_det.device
    yf = y_det.double()
    xf = x_det.double()
    b_r, d = yf.shape
    if bank.k > 0 and bank.dim != d:
        bank.reset()
    bank.dim = int(d)

    d_yy = torch.cdist(yf, yf, p=2.0)
    d_yy_inf = d_yy.clone()
    d_yy_inf.fill_diagonal_(float("inf"))
    k_loc = min(3, max(1, b_r - 1))
    r_loc = torch.topk(d_yy_inf, k=k_loc, dim=1, largest=False).values[:, -1].clamp_min(1e-6)

    if bank.k == 0 or bank.mu_y is None:
        first = int(r_loc.argmin().item())
        seeds = [first]
        seed_r = [float(r_loc[first].item())]
        while len(seeds) < bank.max_basins:
            ratios = [
                d_yy[:, s_idx] / torch.maximum(r_loc, torch.full_like(r_loc, sr))
                for s_idx, sr in zip(seeds, seed_r)
            ]
            far_val, far_idx = torch.stack(ratios, dim=1).amin(dim=1).max(dim=0)
            if float(far_val.item()) <= 1.85:
                break
            fi = int(far_idx.item())
            seeds.append(fi)
            seed_r.append(float(r_loc[fi].item()))
        bank.k = len(seeds)
        centers = yf[seeds].clone()
        seed_r_t = torch.tensor(seed_r, dtype=torch.float64, device=dev)
        d_yc0 = torch.cdist(yf, centers, p=2.0)
        denom0 = torch.maximum(r_loc.unsqueeze(1), seed_r_t.unsqueeze(0)).clamp_min(1e-6)
        min_r0, asg_y = (d_yc0 / denom0).min(dim=1)
        overflow = (bank.k >= bank.max_basins) and (int((min_r0 > 1.95).sum().item()) >= 3)
        bank.w_y = torch.zeros(bank.k, dtype=torch.float64, device=dev)
        bank.mu_y = centers.clone()
        bank.cov_y = torch.zeros((bank.k, d, d), dtype=torch.float64, device=dev)
        bank.rad_y = seed_r_t.clone()
        bank.w_x = torch.zeros(bank.k, dtype=torch.float64, device=dev)
        bank.mu_x = centers.clone()
        bank.cov_x = torch.zeros((bank.k, d, d), dtype=torch.float64, device=dev)
        for c in range(bank.k):
            mc = asg_y == c
            nc = int(mc.sum().item())
            if nc >= 1:
                yc = yf[mc]
                mu = yc.mean(dim=0)
                bank.w_y[c] = float(nc)
                bank.mu_y[c] = mu
                ry = yc - mu
                bank.cov_y[c] = (ry.T @ ry) / float(nc)
                bank.rad_y[c] = r_loc[mc].median().clamp_min(1e-6)
    else:
        if bank.mu_y.device != dev:
            bank.restore_device(dev)
        d_yc = torch.cdist(yf, bank.mu_y, p=2.0)
        denom_yc = torch.maximum(r_loc.unsqueeze(1), bank.rad_y.unsqueeze(0)).clamp_min(1e-6)
        min_ratio, asg_y = (d_yc / denom_yc).min(dim=1)
        far_mask = min_ratio > 1.95
        overflow = (bank.k >= bank.max_basins) and (int(far_mask.sum().item()) >= 3)
        if bank.k < bank.max_basins and int(far_mask.sum().item()) >= 3:
            far_idx_all = far_mask.nonzero(as_tuple=False).squeeze(-1)
            fi = int(far_idx_all[r_loc[far_idx_all].argmin()].item())
            new_c = yf[fi : fi + 1]
            new_r = r_loc[fi : fi + 1]
            bank.k += 1
            bank.mu_y = torch.cat([bank.mu_y, new_c], dim=0)
            bank.rad_y = torch.cat([bank.rad_y, new_r], dim=0)
            bank.w_y = torch.cat([bank.w_y, torch.zeros(1, dtype=torch.float64, device=dev)], dim=0)
            bank.cov_y = torch.cat([bank.cov_y, torch.zeros((1, d, d), dtype=torch.float64, device=dev)], dim=0)
            bank.mu_x = torch.cat([bank.mu_x, new_c.clone()], dim=0)
            bank.w_x = torch.cat([bank.w_x, torch.zeros(1, dtype=torch.float64, device=dev)], dim=0)
            bank.cov_x = torch.cat([bank.cov_x, torch.zeros((1, d, d), dtype=torch.float64, device=dev)], dim=0)
            d_yc = torch.cdist(yf, bank.mu_y, p=2.0)
            denom_yc = torch.maximum(r_loc.unsqueeze(1), bank.rad_y.unsqueeze(0)).clamp_min(1e-6)
            min_ratio, asg_y = (d_yc / denom_yc).min(dim=1)

        bank.w_y.mul_(bank.gamma)
        for c in range(bank.k):
            mc = (asg_y == c) & (min_ratio <= 1.95)
            nc = int(mc.sum().item())
            if nc >= 1:
                yc = yf[mc]
                w_new = float(bank.w_y[c].item()) + float(nc)
                alpha = float(nc) / max(w_new, 1e-8)
                mu_b = yc.mean(dim=0)
                delta = mu_b - bank.mu_y[c]
                ry = yc - mu_b
                cov_b = (ry.T @ ry) / float(nc)
                bank.cov_y[c] = (
                    (1.0 - alpha) * bank.cov_y[c]
                    + alpha * cov_b
                    + alpha * (1.0 - alpha) * torch.outer(delta, delta)
                )
                bank.mu_y[c] = bank.mu_y[c] + alpha * delta
                bank.w_y[c] = w_new
                bank.rad_y[c] = (1.0 - alpha) * bank.rad_y[c] + alpha * r_loc[mc].median()

    d_xc = torch.cdist(xf, bank.mu_y, p=2.0)
    min_rx, asg_x = (d_xc / bank.rad_y.unsqueeze(0).clamp_min(1e-6)).min(dim=1)
    in_basin_x = min_rx <= 2.2
    bank.w_x.mul_(bank.gamma)
    for c in range(bank.k):
        mc = (asg_x == c) & in_basin_x
        nc = int(mc.sum().item())
        if nc >= 1:
            xc = xf[mc]
            w_old = float(bank.w_x[c].item())
            w_new = w_old + float(nc)
            alpha = float(nc) / max(w_new, 1e-8)
            mu_b = xc.mean(dim=0)
            rx = xc - mu_b
            cov_b = (rx.T @ rx) / float(nc)
            if w_old < 1e-6:
                bank.mu_x[c] = mu_b
                bank.cov_x[c] = cov_b
            else:
                delta = mu_b - bank.mu_x[c]
                bank.cov_x[c] = (
                    (1.0 - alpha) * bank.cov_x[c]
                    + alpha * cov_b
                    + alpha * (1.0 - alpha) * torch.outer(delta, delta)
                )
                bank.mu_x[c] = bank.mu_x[c] + alpha * delta
            bank.w_x[c] = w_new
    return d_xc, min_rx, asg_x, overflow


def _has_spatial_dims(x: torch.Tensor) -> bool:
    """Return True iff ``x`` has non-singleton spatial dimensions beyond ``(B, C)``."""
    return x.ndim >= 3 and any(int(s) > 1 for s in x.shape[2:])


def unified_v15_highd_basin_stein_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    *,
    record: Any = None,
    generator: Optional[nn.Module] = None,
    z_batch: Optional[torch.Tensor] = None,
    allow_update: bool = True,
    w_jac: float = 0.90,
    w_x: float = 0.85,
    w_mu: float = 1.45,
    w_bridge: float = 1.55,
) -> Tuple[torch.Tensor, float]:
    """Zero-`eigh`, Haar-O(d)-invariant, scale-equivariant High-D Basin Wiener-Stein & shell-bridge operator (L3-7)."""
    zero = x_fake[:0].sum()
    bank = getattr(record, "highd_basin_bank", None) if record is not None else None
    if bank is None or x_fake.ndim < 2 or y_real.ndim < 2 or _has_spatial_dims(x_fake) or _has_spatial_dims(y_real):
        return zero, 0.0
    x_fake = x_fake.reshape(int(x_fake.shape[0]), -1)
    y_real = y_real.reshape(int(y_real.shape[0]), -1)
    b_f, d = int(x_fake.shape[0]), int(x_fake.shape[1])
    b_r = int(y_real.shape[0])
    if b_f < 8 or b_r < 8 or d <= 16 or d > 256:
        return zero, 0.0

    z_tail: Tuple[int, ...] = ()
    z_2d: Optional[torch.Tensor] = None
    if z_batch is not None and z_batch.ndim >= 2 and not _has_spatial_dims(z_batch):
        z_tail = tuple(z_batch.shape[2:])
        z_2d = z_batch.reshape(int(z_batch.shape[0]), -1)

    with torch.no_grad():
        if allow_update:
            d_xc, min_rx, asg_x, overflow = _highd_bank_assign_and_update(bank, x_fake.detach(), y_real.detach())
            if overflow:
                return zero, 0.0
            if z_2d is not None and z_2d.shape[0] >= 4 and z_2d.shape[1] >= 1:
                zf = z_2d.detach().float()
                d_zz = torch.cdist(zf, zf, p=2.0)
                d_zz.fill_diagonal_(float("inf"))
                est_sig = max(1e-4, float(torch.quantile(d_zz.amin(dim=1), 0.10).item()) / math.sqrt(2.0 * float(zf.shape[1])))
                bank.sigma_z_ema = est_sig if not bank.sigma_z_init else (0.90 * bank.sigma_z_ema + 0.10 * est_sig)
                bank.sigma_z_init = True
        else:
            if bank.k == 0 or bank.mu_y is None or bank.dim != d:
                return zero, 0.0
            if bank.mu_y.device != x_fake.device:
                bank.restore_device(x_fake.device)
            if bank.k >= bank.max_basins:
                d_yc_ro = torch.cdist(y_real.detach().double(), bank.mu_y, p=2.0)
                min_ry_ro = (d_yc_ro / bank.rad_y.unsqueeze(0).clamp_min(1e-6)).amin(dim=1)
                if int((min_ry_ro > 1.95).sum().item()) >= 3:
                    return zero, 0.0
            d_xc = torch.cdist(x_fake.detach().double(), bank.mu_y, p=2.0)
            min_rx, asg_x = (d_xc / bank.rad_y.unsqueeze(0).clamp_min(1e-6)).min(dim=1)

        # Dimension-derived chi_d 3-sigma shell boundary: 1 + 3 / sqrt(2*d)
        r_shell = 1.0 + 3.0 / math.sqrt(2.0 * float(d))
        in_basin_x = min_rx <= (r_shell + 0.65)
        mature_y = bank.w_y >= float(d) * 0.4
        if not bool(mature_y.any()):
            return zero, 0.0
        min_cov_ratio = float((bank.w_x[mature_y] / bank.w_y[mature_y].clamp_min(1e-6)).amin().item())
        u_cov = max(0.0, min(1.0, (min_cov_ratio - 0.50) / 0.25))
        cov_gate = u_cov * u_cov * (3.0 - 2.0 * u_cov)
        if cov_gate <= 1e-4:
            return zero, 1.0
        euclid_asg = d_xc.argmin(dim=1)

    total_loss = zero
    if w_bridge > 0.0 and bank.k >= 1:
        u_br = ((min_rx.float() - r_shell) / 0.55).clamp(0.0, 1.0)
        g_br = torch.where(mature_y[euclid_asg], u_br * u_br * (3.0 - 2.0 * u_br), torch.zeros_like(u_br))
        if bool((g_br > 1e-4).any()):
            mu_tgt = bank.mu_y.float()[euclid_asg]
            rad_tgt = bank.rad_y.float()[euclid_asg].clamp_min(1e-5)
            bi_dist = (rad_tgt.square() + (x_fake.float() - mu_tgt).square().sum(dim=-1)).sqrt() - rad_tgt
            total_loss = total_loss + (float(w_bridge) * float(cov_gate)) * (
                (g_br * bi_dist).sum() / float(b_f)
            ).to(dtype=x_fake.dtype)

    has_jac_inputs = (
        generator is not None
        and z_2d is not None
        and z_2d.shape[0] == b_f
        and 1 <= z_2d.shape[1] <= 128
        and bank.sigma_z_init
        and w_jac > 0.0
    )
    with torch.no_grad():
        eye = torch.eye(d, dtype=torch.float64, device=x_fake.device)
        op_x = torch.zeros((bank.k, d, d), dtype=torch.float32, device=x_fake.device)
        mu_force = torch.zeros((bank.k, d), dtype=torch.float32, device=x_fake.device)
        sig_y_vec = torch.zeros(bank.k, dtype=torch.float32, device=x_fake.device)
        valid_c = torch.zeros(bank.k, dtype=torch.bool, device=x_fake.device)
        anchors: List[torch.Tensor] = []
        t_mats: List[torch.Tensor] = []
        t_invs: List[torch.Tensor] = []
        anc_sigs: List[float] = []
        worst_def = 0.0
        eps_x = 0.12

        for c in range(bank.k):
            if float(bank.w_y[c].item()) < float(d) * 0.4 or float(bank.w_x[c].item()) < 4.0:
                continue
            cy = bank.cov_y[c]
            var_y = float(torch.diagonal(cy).sum().item()) / float(d)
            if var_y <= 1e-10:
                continue
            sigma_y = math.sqrt(var_y)
            shrink = min(0.25, float(d) / (float(d) + 4.0 * float(bank.w_y[c].item())))
            cy_reg = (1.0 - shrink) * cy + (shrink * var_y) * eye
            Ly, info_y = torch.linalg.cholesky_ex(0.5 * (cy_reg + cy_reg.T))
            if int(info_y.item()) != 0:
                continue
            cy_inv = torch.cholesky_inverse(Ly)

            cx = 0.5 * (bank.cov_x[c] + bank.cov_x[c].T)
            var_x = float(torch.diagonal(cx).sum().item()) / float(d)
            # Symmetric shrinkage on cx (exact cx_reg == cy_reg -> op_x == 0 at equilibrium!)
            cx_reg = (1.0 - shrink) * cx + (shrink * min(var_x, var_y)) * eye
            Lx_x, info_x = torch.linalg.cholesky_ex(cx_reg + eps_x * cy_reg)
            if int(info_x.item()) != 0:
                continue
            cx_inv = torch.cholesky_inverse(Lx_x)
            Lx_ref, info_xr = torch.linalg.cholesky_ex(cx_reg + 0.01 * cy_reg)
            if int(info_xr.item()) == 0:
                ld_gap = float((2.0 * torch.diagonal(Ly).log().sum() - 2.0 * torch.diagonal(Lx_ref).log().sum()).item()) / float(d)
                if ld_gap > worst_def:
                    worst_def = ld_gap

            op_x[c] = ((1.0 / (1.0 + eps_x)) * cy_inv - cx_inv).float()
            dm_c = bank.mu_x[c] - bank.mu_y[c]
            mf_c = cy_inv @ dm_c
            dm_w_norm = math.sqrt(max(0.0, float((dm_c * mf_c).sum().item())) / float(d))
            if dm_w_norm > 2.0:
                mf_c = mf_c * (2.0 / dm_w_norm)
            mu_force[c] = mf_c.float()
            sig_y_vec[c] = float(sigma_y)
            valid_c[c] = True

            if has_jac_inputs:
                mc = (asg_x == c) & (min_rx <= 2.4)
                if bool(mc.any()):
                    # Golden-Ratio Wiener matrix residual target: T_c^{-1} = C_y^{-1} + C_y^{-1} C_x C_y^{-1}
                    # Strictly SPD, zero-eigh, Haar-O(d) & scale equivariant: expands collapsed eigendirections
                    # (tau -> 1) while suppressing local Jacobian puffing on already-wide eigendirections!
                    t_inv = cy_inv + (cy_inv @ cx_reg @ cy_inv)
                    t_inv = 0.5 * (t_inv + t_inv.T)
                    L_ti, info_ti = torch.linalg.cholesky_ex(t_inv)
                    if int(info_ti.item()) == 0:
                        t_c = torch.cholesky_inverse(L_ti)
                        cand_idx = mc.nonzero(as_tuple=False).squeeze(-1)
                        order_c = torch.argsort(d_xc[cand_idx, c])
                        nc = len(order_c)
                        pick_pos = [0] if nc == 1 else ([0, 1] if nc == 2 else [0, nc // 3, (2 * nc) // 3])
                        for pp in pick_pos:
                            anchors.append(z_2d.detach()[int(cand_idx[int(order_c[pp].item())].item())])
                            t_mats.append(t_c)
                            t_invs.append(t_inv)
                            anc_sigs.append(sigma_y)

    if not bool(valid_c.any()):
        return total_loss, worst_def

    if anchors and generator is not None and z_2d is not None:
        m_act = len(anchors)
        d_z = int(z_2d.shape[1])
        z_anc = torch.stack(anchors, dim=0)
        eye_z = torch.eye(d_z, dtype=z_anc.dtype, device=z_anc.device) * float(bank.sigma_z_ema)
        z_all = torch.cat(
            [
                (z_anc.unsqueeze(1) + eye_z.unsqueeze(0)).reshape(m_act * d_z, d_z),
                (z_anc.unsqueeze(1) - eye_z.unsqueeze(0)).reshape(m_act * d_z, d_z),
            ],
            dim=0,
        )
        if z_tail:
            z_all = z_all.reshape(2 * m_act * d_z, d_z, *z_tail)
        prev_probe = bool(getattr(record, "_in_probe", False)) if record is not None else False
        bn_modes = []
        for mod in generator.modules():
            if isinstance(mod, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)):
                bn_modes.append((mod, mod.training))
                mod.eval()
        if record is not None:
            record._in_probe = True
        try:
            y_out = generator(z_all)
        finally:
            if record is not None:
                record._in_probe = prev_probe
            for mod, flag in bn_modes:
                mod.training = flag
        y_all = y_out[0] if isinstance(y_out, (tuple, list)) and y_out else y_out
        if isinstance(y_all, torch.Tensor) and y_all.shape[0] == 2 * m_act * d_z:
            y_p = y_all[: m_act * d_z].reshape(m_act, d_z, d)
            y_m = y_all[m_act * d_z :].reshape(m_act, d_z, d)
            J_all = (0.5 * (y_p - y_m)).double()
            eps_r = 1e-4
            t_stack = torch.stack(t_mats, dim=0)
            t_inv_stack = torch.stack(t_invs, dim=0)
            L_all = torch.matmul(J_all.transpose(1, 2), J_all)
            tr_term = (1.0 / (1.0 + eps_r)) * (t_inv_stack * L_all).sum(dim=(-2, -1))
            L_reg = 0.5 * (L_all + L_all.transpose(-1, -2)) + eps_r * t_stack
            chol_L = torch.linalg.cholesky(L_reg)
            logdet_L = 2.0 * torch.diagonal(chol_L, dim1=-2, dim2=-1).log().sum(dim=-1)
            with torch.no_grad():
                chol_T = torch.linalg.cholesky((1.0 + eps_r) * t_stack)
                logdet_T = 2.0 * torch.diagonal(chol_T, dim1=-2, dim2=-1).log().sum(dim=-1)
            stein_b = 0.5 * (tr_term - logdet_L + logdet_T - float(d) / (1.0 + eps_r)) / math.sqrt(float(d))
            sig_anc_t = torch.tensor(anc_sigs, dtype=torch.float64, device=x_fake.device)
            mean_sig = sig_anc_t.mean().clamp_min(1e-8)
            jac_loss = ((sig_anc_t.square() / mean_sig) * stein_b).mean()
            total_loss = total_loss + (float(w_jac) * float(cov_gate)) * jac_loss.to(dtype=x_fake.dtype)

    row_valid = in_basin_x & valid_c[asg_x]
    if bool(row_valid.any()) and (w_x > 0.0 or w_mu > 0.0):
        idx = asg_x[row_valid]
        xf_v = x_fake.float()[row_valid]
        rx = xf_v - bank.mu_x.float()[idx]
        sig = sig_y_vec[idx]
        r2_dimless = rx.square().sum(dim=-1, keepdim=True) / (float(d) * sig.square().unsqueeze(-1).clamp_min(1e-10))
        rx_s = rx * torch.rsqrt(1.0 + 0.05 * r2_dimless)
        counts = torch.bincount(idx, minlength=bank.k).clamp_min(1).float()
        n_active = float((counts[valid_c] > 0).sum().clamp_min(1).item())
        w_row = 1.0 / (n_active * counts[idx])
        if w_x > 0.0:
            quad = 0.5 * torch.einsum("ni,nij,nj->n", rx_s, op_x[idx], rx_s)
            phys_x = (sig / math.sqrt(float(d))) * quad
            total_loss = total_loss + (float(w_x) * float(cov_gate)) * (w_row * phys_x).sum().to(dtype=x_fake.dtype)
        if w_mu > 0.0:
            mu_inner = (xf_v * mu_force[idx]).sum(dim=-1)
            phys_mu = (sig / math.sqrt(float(d))) * mu_inner
            total_loss = total_loss + (float(w_mu) * float(cov_gate)) * (w_row * phys_mu).sum().to(dtype=x_fake.dtype)

    return total_loss, worst_def


@torch.no_grad()
def _det_sparse_unit_cands(seed_id: int, n_cand: int, dz: int, device: torch.device) -> torch.Tensor:
    """Zero-RNG Murmur32 Box-Muller unit directions on ``S^{dz-1}`` (identical on CPU and CUDA)."""
    idx = torch.arange(n_cand * dz, device=device, dtype=torch.int64)
    h = (idx ^ (int(seed_id) * 0x9E3779B1)) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    u1 = (h.float() + 0.5) / 4294967296.0
    h2 = (h * 0x27D4EB2D + 0x165667B1) & 0xFFFFFFFF
    u2 = (h2.float() + 0.5) / 4294967296.0
    g = torch.sqrt(-2.0 * torch.log(u1.clamp_min(1e-7))) * torch.cos(6.283185307179586 * u2)
    return F.normalize(g.view(n_cand, dz), dim=1)


def _safe_cholesky_solve_spd(mat: torch.Tensor, rhs: torch.Tensor, eye: torch.Tensor, base_scale: torch.Tensor) -> Optional[torch.Tensor]:
    """Solve ``mat @ X = rhs`` via guaranteed-SPD Cholesky with adaptive ridge escalation (0 eigh/svd/pinv)."""
    chol, info = torch.linalg.cholesky_ex(mat)
    if int(info.item()) == 0:
        return torch.cholesky_solve(rhs, chol)
    for mult in (1e-2, 1e-1, 1.0):
        chol, info = torch.linalg.cholesky_ex(mat + (mult * base_scale.clamp_min(1e-24)) * eye)
        if int(info.item()) == 0:
            return torch.cholesky_solve(rhs, chol)
    return None


def unified_v15_sparse_obs_wiener_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    *,
    record: Any = None,
    z_batch: Optional[torch.Tensor] = None,
    allow_update: bool = True,
    start_proto_step: int = 15,
    merge_cos: float = 0.55,
    k_nlm: int = 32,
    eta_b: float = 0.15,
) -> torch.Tensor:
    """Zero-`eigh`, zero-RNG, scale-homogeneous Wiener-NLM + Laguerre transport for coordinate-masked vectors.

    Recovers dense signal targets strictly from coordinate-masked observations ``y_obs = m * y`` via
    guaranteed-SPD Linear-MMSE Wiener filtering ``W^T = (1/p) * (I - (1-p) S^{-1} diag(S))`` + 2-pass
    Non-Local Means, and transports masked generator outputs via a deterministic semi-discrete Laguerre
    partition in latent space (zero calls to ``G.eval()``, zero oracle, bit-exact checkpointed state).
    """
    xf = x_fake.reshape(int(x_fake.shape[0]), -1).float()
    xr = y_real.detach().reshape(int(y_real.shape[0]), -1).float()
    b_f, dim = int(xf.shape[0]), int(xf.shape[1])
    b_r = int(xr.shape[0])
    zero = 0.0 * xf.sum()
    bank = getattr(record, "sparse_obs_bank", None) if record is not None else None
    if bank is None or b_f < 2 or b_r < 2 or dim < 16:
        return zero.to(dtype=x_fake.dtype)

    dev = xf.device
    mr = (xr != 0).float()
    mf_m = (xf.detach() != 0).float()
    z_2d: Optional[torch.Tensor] = None
    dz = 0
    if z_batch is not None and z_batch.ndim >= 2 and not _has_spatial_dims(z_batch) and int(z_batch.shape[0]) == b_f:
        z_2d = z_batch.detach().reshape(b_f, -1).float().to(device=dev)
        dz = int(z_2d.shape[1])

    if not allow_update:
        if bank.step == 0 or bank.S_y is None or bank.mu_y is None or bank.dim != dim:
            return zero.to(dtype=x_fake.dtype)
        if bank.S_y.device != dev:
            bank.restore_device(dev)
        p_y = max(1e-4, float(bank.p_y))
        p_x = max(1e-4, float(bank.p_x))
        diag_sy = bank.S_y.diagonal().clamp_min(1e-24)
        sc_y = (diag_sy.sum() / p_y).clamp_min(1e-24)
        if bank.proto_k > 0 and bank.proto_y is not None and z_2d is not None and dz == bank.dz:
            pk = int(bank.proto_k)
            pw_all = bank.proto_w[:pk]
            mat_mask = pw_all >= 2.0
            if not bool(mat_mask.any()):
                mat_mask = torch.ones_like(pw_all, dtype=torch.bool)
            act_idx = mat_mask.nonzero(as_tuple=False).squeeze(-1)
            zu = F.normalize(z_2d, dim=1)
            scores = zu @ bank.proto_z[act_idx].T + bank.proto_b[act_idx].unsqueeze(0)
            asg_loc = scores.argmax(dim=1)
            tgt_dense = bank.proto_y[act_idx][asg_loc]
            loss_nlm = ((xf - mf_m * tgt_dense).pow(2).sum(dim=1).mean()) / (p_x * sc_y)
            return loss_nlm.to(dtype=x_fake.dtype)
        zx = xf - mf_m * bank.mu_y.unsqueeze(0)
        sig_y = (bank.S_y - (1.0 - p_y) * torch.diag(diag_sy)) / (p_y * p_y)
        sig_x = (bank.S_x - (1.0 - p_x) * torch.diag(bank.S_x.diagonal())) / (p_x * p_x)
        diff_sig = (sig_x - sig_y).detach()
        diff_d = diff_sig.diagonal()
        diff_o = diff_sig - torch.diag(diff_d)
        norm_sig = sig_y.pow(2).sum().clamp_min(1e-24)
        s_cur = (zx.T @ zx) / float(b_f)
        loss_cov = ((diff_o * s_cur).sum() / (p_x * p_x) + (diff_d * s_cur.diagonal()).sum() / p_x) / norm_sig
        diff_mu = (bank.mu_x - bank.mu_y).detach()
        mu_cur = xf.mean(dim=0) / p_x
        loss_mu = (diff_mu * mu_cur).sum() / sc_y
        return (loss_cov + loss_mu).to(dtype=x_fake.dtype)

    with torch.no_grad():
        if bank.step > 0 and (bank.dim != dim or (bank.dz > 0 and dz > 0 and bank.dz != dz)):
            bank.reset()
        if bank.step > 0 and bank.S_y is not None and bank.S_y.device != dev:
            bank.restore_device(dev)
        bank.step += 1
        st_n = int(bank.step)
        bank.dim = int(dim)
        if dz > 0:
            bank.dz = int(dz)

        by = 1.0 / float(min(st_n, 400))
        bx = 1.0 / float(min(st_n, 10))
        pr_b = float(mr.mean().clamp_min(1e-4).item())
        px_b = float(mf_m.mean().clamp_min(1e-4).item())
        mur_b = xr.mean(dim=0) / pr_b
        mux_b = xf.detach().mean(dim=0) / px_b

        if bank.S_y is None:
            bank.p_y = pr_b
            bank.mu_y = mur_b
            bank.p_x = px_b
            bank.mu_x = mux_b
        else:
            bank.p_y = (1.0 - by) * bank.p_y + by * pr_b
            bank.mu_y = (1.0 - by) * bank.mu_y + by * mur_b
            bank.p_x = (1.0 - bx) * bank.p_x + bx * px_b
            bank.mu_x = (1.0 - bx) * bank.mu_x + bx * mux_b

        zr = xr - mr * bank.mu_y.unsqueeze(0)
        zx_det = xf.detach() - mf_m * bank.mu_y.unsqueeze(0)
        sy_b = (zr.T @ zr) / float(b_r)
        sx_b = (zx_det.T @ zx_det) / float(b_f)
        sx_prev = None if bank.S_x is None else bank.S_x.clone()
        if bank.S_y is None:
            bank.S_y = sy_b
            bank.S_x = sx_b
        else:
            bank.S_y = (1.0 - by) * bank.S_y + by * sy_b
            bank.S_x = (1.0 - bx) * bank.S_x + bx * sx_b

        p_y = max(1e-4, float(bank.p_y))
        p_x = max(1e-4, float(bank.p_x))
        diag_sy = bank.S_y.diagonal().clamp_min(1e-24)
        diag_mean = diag_sy.mean().clamp_min(1e-24)
        eye_d = torch.eye(dim, device=dev, dtype=torch.float32)
        n_eff = float(min(st_n, 400) * b_r)
        ridge_f = max(0.02, min(1.0, float(dim) / max(n_eff, 1.0)))
        s_spd = bank.S_y + (ridge_f * diag_mean) * eye_d
        inv_s_diag = _safe_cholesky_solve_spd(s_spd, torch.diag(diag_sy), eye_d, diag_mean)
        if inv_s_diag is None:
            w_t = eye_d
        else:
            w_t = (eye_d - (1.0 - p_y) * inv_s_diag) / p_y
        sc_y = (diag_sy.sum() / p_y).clamp_min(1e-24)

        if z_2d is not None and dz > 0:
            z_u = z_2d - z_2d.mean(dim=0, keepdim=True)
            szz_b = (z_u.T @ z_u) / float(b_f)
            szx_b = (z_u.T @ (xf.detach() - mf_m * bank.mu_x.unsqueeze(0))) / (float(b_f) * p_x)
            bz = 1.0 / float(min(st_n, 15))
            if bank.S_zz is None:
                bank.S_zz = szz_b
                bank.S_zx = szx_b
            else:
                bank.S_zz = (1.0 - bz) * bank.S_zz + bz * szz_b
                bank.S_zx = (1.0 - bz) * bank.S_zx + bz * szx_b

        max_b = int(bank.max_buf)
        if bank.buf_y is None or bank.buf_m is None:
            bank.buf_y = torch.zeros((max_b, dim), device=dev, dtype=torch.float32)
            bank.buf_m = torch.zeros((max_b, dim), device=dev, dtype=torch.bool)
        ptr = int(bank.buf_ptr)
        take = min(b_r, max_b)
        end = min(max_b, ptr + take)
        n1 = end - ptr
        bank.buf_y[ptr:end] = xr[:n1]
        bank.buf_m[ptr:end] = xr[:n1] != 0
        if n1 < take:
            rem = take - n1
            bank.buf_y[:rem] = xr[n1 : n1 + rem]
            bank.buf_m[:rem] = xr[n1 : n1 + rem] != 0
        bank.buf_ptr = (ptr + take) % max_b
        bank.buf_n = min(max_b, int(bank.buf_n) + take)

        n_b = int(bank.buf_n)
        yb = bank.buf_y[:n_b]
        mb = bank.buf_m[:n_b].float()
        zb_w = (yb - mb * bank.mu_y.unsqueeze(0)) @ w_t
        zr_w = zr @ w_t
        sim1 = F.normalize(zr_w, dim=1) @ F.normalize(zb_w, dim=1).T
        kk = min(int(k_nlm), n_b)
        top_s, top_i = torch.topk(sim1, k=kk, dim=1)
        w_full = torch.zeros((b_r, n_b), device=dev, dtype=torch.float32)
        w_full.scatter_(1, top_i, F.softmax(top_s * 18.0, dim=1))
        num = w_full @ yb
        den = w_full @ mb
        y_mmse = bank.mu_y.unsqueeze(0) + zr_w
        y_clean = torch.where(den > 1e-4, num / den.clamp_min(1e-4), y_mmse)

        if n_b >= 512:
            dot2 = y_clean @ yb.T
            n_c2 = (y_clean.pow(2) @ mb.T).clamp_min(1e-24).sqrt()
            n_b2 = yb.pow(2).sum(dim=1, keepdim=True).T.clamp_min(1e-24).sqrt()
            cos2 = dot2 / (n_c2 * n_b2)
            top_s2, top_i2 = torch.topk(cos2, k=kk, dim=1)
            w_full2 = torch.zeros((b_r, n_b), device=dev, dtype=torch.float32)
            w_full2.scatter_(1, top_i2, F.softmax(top_s2 * 14.0, dim=1))
            num2 = w_full2 @ yb
            den2 = w_full2 @ mb
            y_clean = torch.where(den2 > 1e-4, num2 / den2.clamp_min(1e-4), y_clean)

        tgt_dense: Optional[torch.Tensor] = None
        if st_n >= int(start_proto_step) and z_2d is not None and dz > 0:
            max_p = int(bank.max_protos)
            yc_c = y_clean - bank.mu_y.unsqueeze(0)
            yc_u = F.normalize(yc_c, dim=1)
            if bank.proto_y is None:
                bank.proto_y = torch.zeros((max_p, dim), device=dev, dtype=torch.float32)
                bank.proto_w = torch.zeros((max_p,), device=dev, dtype=torch.float32)
                bank.proto_z = torch.zeros((max_p, dz), device=dev, dtype=torch.float32)
                bank.proto_b = torch.zeros((max_p,), device=dev, dtype=torch.float32)
            pk = int(bank.proto_k)
            matched_mask = torch.zeros(b_r, dtype=torch.bool, device=dev)
            if pk > 0:
                py_u = F.normalize(bank.proto_y[:pk] - bank.mu_y.unsqueeze(0), dim=1)
                sims_all = yc_u @ py_u.T
                best_s, best_j = sims_all.max(dim=1)
                matched_mask = (best_s >= float(merge_cos)) | (pk >= max_p)
                if bool(matched_mask.any()):
                    mj = best_j[matched_mask]
                    my = y_clean[matched_mask]
                    cnt = torch.bincount(mj, minlength=pk).float()
                    sum_y = torch.zeros((pk, dim), device=dev, dtype=torch.float32)
                    sum_y.index_add_(0, mj, my)
                    upd = cnt > 0
                    w_old = bank.proto_w[:pk]
                    w_new = (w_old + cnt).clamp_max(300.0)
                    alpha = torch.where(upd, cnt / w_new.clamp_min(1.0), torch.zeros_like(cnt))
                    mean_y = sum_y / cnt.clamp_min(1.0).unsqueeze(1)
                    bank.proto_y[:pk] = torch.where(
                        upd.unsqueeze(1),
                        (1.0 - alpha.unsqueeze(1)) * bank.proto_y[:pk] + alpha.unsqueeze(1) * mean_y,
                        bank.proto_y[:pk],
                    )
                    bank.proto_w[:pk] = torch.where(upd, w_new, w_old)

            novel_idx = (~matched_mask).nonzero(as_tuple=False).squeeze(-1)
            if int(novel_idx.numel()) > 0 and pk < max_p:
                a_z: Optional[torch.Tensor] = None
                if bank.S_zz is not None and bank.S_zx is not None:
                    eye_z = torch.eye(dz, device=dev, dtype=torch.float32)
                    z_diag = bank.S_zz.diagonal().mean().clamp_min(1e-12)
                    a_z = _safe_cholesky_solve_spd(bank.S_zz + (0.05 * z_diag) * eye_z, bank.S_zx, eye_z, z_diag)
                for bi_t in novel_idx.tolist():
                    if pk == 0:
                        bank.proto_y[0] = y_clean[bi_t]
                        bank.proto_w[0] = 1.0
                        if a_z is not None:
                            z_dir = F.normalize(a_z @ yc_c[bi_t], dim=0)
                            if float(z_dir.norm().item()) < 0.5:
                                z_dir = _det_sparse_unit_cands(0, 1, dz, dev)[0]
                        else:
                            z_dir = _det_sparse_unit_cands(0, 1, dz, dev)[0]
                        bank.proto_z[0] = z_dir
                        bank.proto_b[0] = 0.0
                        pk = 1
                        continue
                    py_c = bank.proto_y[:pk] - bank.mu_y.unsqueeze(0)
                    sims = (yc_u[bi_t : bi_t + 1] @ F.normalize(py_c, dim=1).T).squeeze(0)
                    bs, bj = sims.max(dim=0)
                    if float(bs.item()) >= float(merge_cos) or pk >= max_p:
                        j = int(bj.item())
                        w_new_j = min(300.0, float(bank.proto_w[j].item()) + 1.0)
                        a_j = 1.0 / w_new_j
                        bank.proto_y[j] = (1.0 - a_j) * bank.proto_y[j] + a_j * y_clean[bi_t]
                        bank.proto_w[j] = w_new_j
                    else:
                        bank.proto_y[pk] = y_clean[bi_t]
                        bank.proto_w[pk] = 1.0
                        if a_z is not None:
                            z_dir = F.normalize(a_z @ yc_c[bi_t], dim=0)
                        else:
                            z_dir = _det_sparse_unit_cands(pk, 1, dz, dev)[0]
                        if float(z_dir.norm().item()) < 0.5 or float((bank.proto_z[:pk] @ z_dir).max().item()) > 0.85:
                            cand = _det_sparse_unit_cands(pk + 17, 32, dz, dev)
                            z_dir = cand[(cand @ bank.proto_z[:pk].T).max(dim=1).values.argmin()]
                        bank.proto_z[pk] = z_dir
                        bank.proto_b[pk] = 0.0
                        pk += 1
                bank.proto_k = pk

            pw_all = bank.proto_w[:pk]
            mat_mask = pw_all >= 2.0
            if not bool(mat_mask.any()):
                mat_mask = torch.ones_like(pw_all, dtype=torch.bool)
            act_idx = mat_mask.nonzero(as_tuple=False).squeeze(-1)
            k_act = int(act_idx.numel())
            py_act = bank.proto_y[act_idx]
            pz_act = bank.proto_z[act_idx]
            pw_act = pw_all[act_idx]
            pb_act = bank.proto_b[act_idx]

            zu = F.normalize(z_2d, dim=1)
            scores = zu @ pz_act.T + pb_act.unsqueeze(0)
            asg_loc = scores.argmax(dim=1)
            p_tgt = pw_act / pw_act.sum().clamp_min(1e-8)
            counts = torch.bincount(asg_loc, minlength=k_act).float() / float(b_f)
            pb_new = pb_act + float(eta_b) * (p_tgt - counts)
            pb_new = (pb_new - pb_new.mean()).clamp(-0.5, 0.5)
            bank.proto_b[act_idx] = pb_new
            tgt_dense = py_act[asg_loc]

    zx = xf - mf_m * bank.mu_y.unsqueeze(0)
    if tgt_dense is not None:
        loss_nlm = ((xf - mf_m * tgt_dense).pow(2).sum(dim=1).mean()) / (p_x * sc_y)
        if hasattr(record, "observe_sparse_push_res"):
            record.observe_sparse_push_res(float(loss_nlm.detach().sqrt().clamp(0.0, 2.0).item()))
        return loss_nlm.to(dtype=x_fake.dtype)

    if sx_prev is not None:
        sig_y = (bank.S_y - (1.0 - p_y) * torch.diag(diag_sy)) / (p_y * p_y)
        sig_x_prev = (sx_prev - (1.0 - p_x) * torch.diag(sx_prev.diagonal())) / (p_x * p_x)
        diff_sig = (sig_x_prev - sig_y).detach()
        diff_d = diff_sig.diagonal()
        diff_o = diff_sig - torch.diag(diff_d)
        norm_sig = sig_y.pow(2).sum().clamp_min(1e-24)
        s_cur = (zx.T @ zx) / float(b_f)
        loss_cov = ((diff_o * s_cur).sum() / (p_x * p_x) + (diff_d * s_cur.diagonal()).sum() / p_x) / norm_sig
        diff_mu = (bank.mu_x - bank.mu_y).detach()
        mu_cur = xf.mean(dim=0) / p_x
        loss_mu = (diff_mu * mu_cur).sum() / sc_y
        if hasattr(record, "observe_sparse_push_res"):
            cov_rel = float((diff_sig.pow(2).sum() / norm_sig).sqrt().clamp(0.0, 2.0).item())
            record.observe_sparse_push_res(cov_rel)
        return (loss_cov + loss_mu).to(dtype=x_fake.dtype)

    return zero.to(dtype=x_fake.dtype)


class UnifiedV15GANLoss(GANLoss):
    """Unified-v15 4-Law + Zero-`eigh` CondFiber GANLoss."""

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
        chart_weight: float = 1.4,
        halo_mult: float = 2.1,
        multiscale_horizon: bool = True,
        cond_drift_deconflict: bool = True,
        macro_slack_weight: float = 1.0,
        mean_op_weight: float = 0.36,
        cov_op_weight: float = 0.30,
        imq_active_scale: float = 0.40,
        bank_mean_blend: float = 0.85,
        record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        for name, val in (
            ("critic_weight", critic_weight),
            ("cond_ot_weight", cond_ot_weight),
            ("macro_slack_weight", macro_slack_weight),
            ("mean_op_weight", mean_op_weight),
            ("cov_op_weight", cov_op_weight),
            ("imq_active_scale", imq_active_scale),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, val in (("fibre_rel_blend", fibre_rel_blend), ("bank_mean_blend", bank_mean_blend)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0 or val > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if type(cond_ot_mode) is not str or cond_ot_mode not in ("imq_cmmd", "diag_imq_cmmd"):
            raise ValueError("cond_ot_mode must be 'imq_cmmd' or 'diag_imq_cmmd'")
        if type(conditional_fibre) is not str or conditional_fibre not in ("joint", "slack", "calibrated"):
            raise ValueError("conditional_fibre must be 'joint', 'slack', or 'calibrated'")

        self.critic_weight = float(critic_weight)
        self.conditional_fibre = conditional_fibre
        self.fibre_rel_blend = float(fibre_rel_blend)
        self.cond_ot_weight = float(cond_ot_weight)
        self.cond_ot_mode = cond_ot_mode
        self.macro_slack_weight = float(macro_slack_weight)
        self.mean_op_weight = float(mean_op_weight)
        self.cov_op_weight = float(cov_op_weight)
        self.imq_active_scale = float(imq_active_scale)
        self.bank_mean_blend = float(bank_mean_blend)
        self.record = record
        self.t1c = UnifiedV15CouplingLoss(
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

    @staticmethod
    def _scalar_pair_relativistic(
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        *,
        detach_neg: bool = False,
    ) -> Optional[torch.Tensor]:
        b_p, b_n = int(pos_logits.shape[0]), int(neg_logits.shape[0])
        if b_p != b_n or b_p < 1:
            return None
        p_flat = pos_logits.reshape(b_p, -1).mean(dim=1) if pos_logits.ndim > 1 else pos_logits.reshape(b_p)
        n_src = neg_logits.detach() if detach_neg else neg_logits
        n_flat = n_src.reshape(b_n, -1).mean(dim=1) if n_src.ndim > 1 else n_src.reshape(b_n)
        return F.softplus(n_flat - p_flat).mean()

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        base = super().d_loss(real_logits, fake_logits)
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, _, _, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if rec is not None and hasattr(rec, "observe_margin_and_scale") and not has_cont_cond:
            if captured:
                rec.observe_margin_and_scale(real_logits, fake_logits, x_r, x_f)
            else:
                rec.observe_margin_and_scale(real_logits, fake_logits)
        if self.critic_weight == 0.0 and self.fibre_rel_blend == 0.0:
            return base
        if not captured:
            if self.loss == "non_saturating" and self.t1c.weight > 0.0 and self.fibre_rel_blend > 0.0:
                pair_d = self._scalar_pair_relativistic(real_logits, fake_logits)
                if pair_d is not None:
                    return (1.0 - self.fibre_rel_blend) * base + self.fibre_rel_blend * pair_d
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
        dim_d = max(1, x_r[0].numel()) if x_r.shape[0] > 0 else 0
        max_t1c_dim_d = 4096 if _has_spatial_dims(x_r) else 1024
        if not has_cont_cond and x_r.ndim >= 2 and dim_d > max_t1c_dim_d:
            if self.loss == "non_saturating" and self.t1c.weight > 0.0 and self.fibre_rel_blend > 0.0:
                pair_d = self._scalar_pair_relativistic(real_logits, fake_logits)
                if pair_d is not None:
                    return (1.0 - self.fibre_rel_blend) * base + self.fibre_rel_blend * pair_d
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
            raise ValueError("UnifiedV15GANLoss requires real_logits in g_loss when t1c_weight > 0")
        if self.t1c.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)

        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if not captured:
            base_g = super().g_loss(fake_logits, real_logits)
            if self.loss == "non_saturating" and self.fibre_rel_blend > 0.0:
                pair_g = self._scalar_pair_relativistic(fake_logits, real_logits, detach_neg=True)
                if pair_g is not None:
                    return self.t1c.weight * ((1.0 - self.fibre_rel_blend) * base_g + self.fibre_rel_blend * pair_g)
            return self.t1c.weight * base_g

        b_f = fake_logits.shape[0]
        b_r = real_logits.shape[0]

        # 1. Continuous-conditioned fibers (DDGAN / AE-DDGAN / residual_student):
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
                            cond_ot = unified_v15_conditional_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
                                record=rec,
                                mean_op_weight=self.mean_op_weight,
                                cov_op_weight=self.cov_op_weight,
                                imq_active_scale=self.imq_active_scale,
                                bank_mean_blend=self.bank_mean_blend,
                            )
                            if rec is not None and hasattr(rec, "_simplicial_step"):
                                rec._simplicial_step = False
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

        # 2. High-dimensional domains (dim > 4096 on images e.g. 64x64..256x256, or dim > 1024 on flat vectors):
        dim = max(1, x_r[0].numel()) if b_r > 0 else 0
        has_spatial = _has_spatial_dims(x_r)
        max_t1c_dim = 4096 if has_spatial else 1024
        if x_r.ndim >= 2 and dim > max_t1c_dim:
            self.t1c._discrete_prior_step = False
            self.t1c._simplicial_active_step = False
            if rec is not None and hasattr(rec, "_t1c_disc_gate"):
                rec._t1c_disc_gate = 0.0
            if rec is not None and hasattr(rec, "_simplicial_step"):
                rec._simplicial_step = False
            base_g = super().g_loss(fake_logits, real_logits)
            if self.loss == "non_saturating" and self.fibre_rel_blend > 0.0:
                pair_g = self._scalar_pair_relativistic(fake_logits, real_logits, detach_neg=True)
                if pair_g is not None:
                    return self.t1c.weight * ((1.0 - self.fibre_rel_blend) * base_g + self.fibre_rel_blend * pair_g)
            return base_g

        # 2b. High-D coordinate-masked sparse-observation Wiener-NLM + semi-discrete Laguerre transport:
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        is_flat_vec = x_f.ndim >= 2 and not _has_spatial_dims(x_f) and not has_spatial
        if (
            rec is not None
            and hasattr(rec, "sparse_obs_bank")
            and cond_f is None
            and cond_r is None
            and is_flat_vec
            and 128 <= dim <= 1024
            and b_f >= 8
            and b_r >= 8
        ):
            zf_r = float((x_r.detach() == 0).float().mean().item())
            zf_f = float((x_f.detach() == 0).float().mean().item())
            if 0.50 < zf_r < 0.995 and 0.50 < zf_f < 0.995:
                self.t1c._discrete_prior_step = False
                self.t1c._simplicial_active_step = False
                if hasattr(rec, "_t1c_disc_gate"):
                    rec._t1c_disc_gate = 0.0
                if hasattr(rec, "_simplicial_step"):
                    rec._simplicial_step = False
                g_lat_sp = getattr(rec, "_g_last_latent", None)
                z_sp = g_lat_sp[1] if g_lat_sp is not None else None
                sp_loss = unified_v15_sparse_obs_wiener_loss(
                    x_f,
                    x_r,
                    record=rec,
                    z_batch=z_sp,
                    allow_update=allow_ell,
                )
                return self.t1c.weight * sp_loss.to(dtype=fake_logits.dtype)

        # 3. H_0-connectedness-gated chart coupling + zero-eigh orphan macro-transport:
        is_discrete = bool(self.t1c._discrete_prior_step)
        is_simplicial = bool(self.t1c._simplicial_active_step)
        if rec is not None and hasattr(rec, "_simplicial_step"):
            rec._simplicial_step = is_simplicial
        saved_cw = self.t1c.chart_weight

        g_conn_fwd = 0.0
        if (is_simplicial and saved_cw > 0.0) or (
            self.macro_slack_weight > 0.0
            and is_discrete
            and not is_simplicial
            and cond_f is None
            and cond_r is None
            and is_flat_vec
            and 2 <= dim <= 256
        ):
            if b_r >= 4:
                y_feat = project_transport_features(x_r.detach(), max_spatial=self.t1c.max_spatial)
                g_conn_fwd, _, _, _ = self.t1c._compute_manifold_gates_eff(y_feat, record=rec)

        if is_simplicial and saved_cw > 0.0:
            self.t1c.chart_weight = saved_cw * max(0.0, 1.0 - float(g_conn_fwd))

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

        if (
            self.macro_slack_weight > 0.0
            and is_discrete
            and not is_simplicial
            and cond_f is None
            and cond_r is None
            and is_flat_vec
            and 2 <= dim <= 256
            and g_conn_fwd <= 0.05
            and b_f >= 4
            and b_r >= 4
        ):
            x_f_m = x_f.reshape(b_f, -1) if x_f.ndim > 2 else x_f
            x_r_m = x_r.reshape(b_r, -1) if x_r.ndim > 2 else x_r
            with torch.no_grad():
                ell_m = self.t1c._update_ell(x_r_m, record=rec, allow_update=False)
                c_hat_m = born_infeld_c_hat_detached(x_f_m, x_r_m, ell_m)
                f_1d = fake_logits.detach().reshape(b_f, -1).mean(dim=1)
                r_1d = real_logits.detach().reshape(b_r, -1).mean(dim=1)
                c_mid = 0.5 * (f_1d.mean() + r_1d.mean())
                d_f_tilde = (f_1d - c_mid).clamp(-12.0, 12.0)
                d_r_tilde = (r_1d - c_mid).clamp(-12.0, 12.0)
                P_m = sinkhorn_slack(
                    c_hat_m,
                    d_f_tilde,
                    d_r_tilde,
                    rho=self.t1c.rho,
                    tau=self.t1c.tau,
                    eps=self.t1c.eps,
                    iters=self.t1c.iters,
                )
            macro_loss = v15_zero_eigh_bipartite_deficit_macro_loss(
                x_f_m,
                x_r_m,
                c_hat_m,
                P_m[:b_f, b_r],
                P_m[b_f, :b_r],
                rho=self.t1c.rho,
                g_conn=g_conn_fwd,
            )
            s_th = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
            if s_th < 1.0 and macro_loss.requires_grad:
                macro_loss = macro_loss * s_th + macro_loss.detach() * (1.0 - s_th)
            loss_val = loss_val + (self.t1c.weight * self.macro_slack_weight) * macro_loss

        if (
            rec is not None
            and cond_f is None
            and cond_r is None
            and is_flat_vec
            and 16 < dim <= 256
            and b_f >= 8
            and b_r >= 8
        ):
            g_lat = getattr(rec, "_g_last_latent", None)
            g_mod = g_lat[0] if g_lat is not None else None
            z_b = g_lat[1] if g_lat is not None else None
            stein_loss, worst_def = unified_v15_highd_basin_stein_loss(
                x_f,
                x_r,
                record=rec,
                generator=g_mod,
                z_batch=z_b,
                allow_update=allow_ell,
            )
            if allow_ell and hasattr(rec, "_thermal_scale"):
                u_def = max(0.0, min(1.0, (worst_def - 0.12) / 0.55))
                s_def = 0.12 + 0.88 * (u_def * u_def * (3.0 - 2.0 * u_def))
                rec._thermal_scale = max(float(rec._thermal_scale), s_def)
            loss_val = loss_val + self.t1c.weight * stein_loss.to(dtype=loss_val.dtype)

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
    "UnifiedV15CouplingLoss",
    "UnifiedV15GANLoss",
    "unified_v15_conditional_loss",
    "unified_v15_highd_basin_stein_loss",
    "unified_v15_sparse_obs_wiener_loss",
]
