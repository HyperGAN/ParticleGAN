"""Track 1B Deconflicted Piecewise Coupling & Screened Transport Primitive."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._bispf.thermal import _publish_thermal
from .charts import (
    compute_chart_valley_gate,
    deconflicted_chart_routing,
    deterministic_spatial_pool2d,
    project_chart_features,
    project_transport_features,
    update_streaming_charts,
)


class T1CouplingLoss(nn.Module):
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
        max_spatial: int = 8,
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
        ):
            raise ValueError("self_coupling must be 'hollow', 'feydy', 'feydy_unique', or 'feydy_discrete'")

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
        kappa_h0 = T1CouplingLoss._minimax_persistence_max(d_yy, steps=8) / q90_t
        u_conn = ((10.0 - kappa_h0) / 3.0).clamp(0.0, 1.0)
        u_span = ((span_y - 1.15) / 0.25).clamp(0.0, 1.0)
        u_macro = ((span_y - 3.80) / 1.00).clamp(0.0, 1.0)
        s_conn = u_conn * u_conn * (3.0 - 2.0 * u_conn)
        s_span = u_span * u_span * (3.0 - 2.0 * u_span)
        s_macro = u_macro * u_macro * (3.0 - 2.0 * u_macro)
        return float((s_conn * s_span).item()), float((s_conn * s_span * s_macro).item()), q90_y_u, y_scale

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
            _, g_macro_fwd, _, _ = self._compute_manifold_gates(y_proj.detach())
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

        g_conn_fwd = 0.0
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
            g_conn_fwd, _, _, _ = self._compute_manifold_gates(y_cf_det)
            w_valley_raw = compute_chart_valley_gate(y_cf_det, charts_c, ell_chart, self.rho)
            # Connected curved manifolds (e.g. circular arcs / saddle sheets) have g_conn -> 1
            # even when Euclidean chords cut across ambient space; gate strictly by (1 - g_conn).
            w_valley_chart = float(w_valley_raw * (1.0 - g_conn_fwd))
            if can_mutate and record is not None:
                record._t1c_disc_gate = float(w_valley_chart) if charts_c.shape[0] >= 3 else 0.0
            s_th_cur = float(getattr(record, "_thermal_scale", 1.0)) if record is not None else 1.0
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
        use_hollow = self.self_coupling == "hollow"
        dedup_xx = self.self_coupling == "feydy_unique" or (
            self.self_coupling == "feydy_discrete" and discrete_step
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
        else:
            local_critic_slack = -(p_slack_eff * (0.5 * self.tau * slack_scale * d_fake_tilde)).sum()
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
            g_conn_m, g_macro_m, q90_m, y_scale = self._compute_manifold_gates(y_flat)
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
