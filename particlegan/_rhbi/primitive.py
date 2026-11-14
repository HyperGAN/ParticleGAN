"""Relativistic Hellinger-Born-Infeld (RHBI) slack-screened transport operator."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._bispf.thermal import _publish_thermal


class RHBILoss(nn.Module):
    """Relativistic Hellinger-Born-Infeld (RHBI) slack-screened transport primitive.

    Unifies nonlocal Born-Infeld optimal transport, hollow self-coupling entropic
    debiasing, and local relativistic critic log-odds ascent through a single
    ``(B + 1) x (B + 1)`` Kantorovich-Hellinger slack coupling with zero RNG draws
    and zero schedule annealing.
    """

    def __init__(
        self,
        weight: float = 1.0,
        rho: float = 2.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        *,
        rapidity_screening: bool = True,
        self_coupling: str = "hollow",
        relativistic_weight: float = 0.0,
        scale_log_odds: bool = False,
        akt_horizon: bool = False,
        akt_mode: str = "sqrt",
        slack_gain: float = 0.0,
        thermal_tau: float = 0.0,
        thermal_shock_mu: float = 0.15,
        cvx_mu_gain: float = 0.40,
        tail_spill_gain: float = 0.0,
        simplex_macro_settle: bool = False,
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
        if type(rapidity_screening) is not bool:
            raise ValueError("rapidity_screening must be a boolean")
        if type(scale_log_odds) is not bool:
            raise ValueError("scale_log_odds must be a boolean")
        if type(akt_horizon) is not bool:
            raise ValueError("akt_horizon must be a boolean")
        if type(simplex_macro_settle) is not bool:
            raise ValueError("simplex_macro_settle must be a boolean")
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
        self.register_buffer("ell_ema", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ell_initialized", torch.tensor(False, dtype=torch.bool))
        self._ell_init_py: bool = False
        self.last_p_xy_slack: Optional[torch.Tensor] = None
        self._discrete_prior_step: bool = False
        self._simplicial_active_step: bool = False

    def bind_discrete_prior_step(self, active: bool, *, simplicial_active: bool = False) -> None:
        """Mark whether the current generator step samples from a zero-noise discrete ParticlePrior."""
        if type(simplicial_active) is not bool:
            raise ValueError("simplicial_active must be a boolean")
        self._discrete_prior_step = bool(active)
        self._simplicial_active_step = bool(active and simplicial_active)

    @staticmethod
    @torch.no_grad()
    def _compute_manifold_gates(y_flat: torch.Tensor) -> Tuple[float, float, float, torch.Tensor]:
        """Return ``(g_conn, g_macro, q90_y_u, y_scale)`` via 8-step minimax persistence closure."""
        b_r = y_flat.shape[0]
        y_scale = y_flat.float().abs().amax().clamp_min(1.0)
        if b_r < 4:
            return 0.0, 0.0, 0.1, y_scale
        y_u = y_flat.float() / y_scale
        d_yy = torch.cdist(y_u, y_u, p=2.0)
        off_y = ~torch.eye(b_r, dtype=torch.bool, device=y_u.device)
        d_yy_nn = d_yy.masked_fill(~off_y, float("inf"))
        nn_y = d_yy_nn.amin(dim=-1)
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
        return g_conn, g_macro, q90_y_u, y_scale

    @staticmethod
    @torch.no_grad()
    def _compute_valley_gate(y_real: torch.Tensor, ell: torch.Tensor, rho: float) -> float:
        """Horizon-normalized cyclic-midpoint inter-mode valley indicator ``w_valley`` in ``[0, 1]``."""
        b_r = y_real.shape[0]
        if b_r < 8:
            return 0.0
        yr = y_real.detach().reshape(b_r, -1).double()
        off = ~torch.eye(b_r, dtype=torch.bool, device=yr.device)
        d_yy = torch.cdist(yr, yr, p=2.0)
        yy_q75 = torch.quantile(d_yy.masked_fill(~off, float("inf")).amin(dim=1), 0.75)
        yr_roll = torch.roll(yr, shifts=1, dims=0)
        y_mid = 0.5 * (yr + yr_roll)
        d_my = torch.cdist(y_mid, yr, p=2.0)
        idx = torch.arange(b_r, device=yr.device)
        d_my[idx, idx] = float("inf")
        d_my[idx, (idx - 1) % b_r] = float("inf")
        mid_q75 = torch.quantile(d_my.amin(dim=1), 0.75)
        denom_h = torch.maximum(yy_q75, ell.detach().double()).clamp_min(1e-8)
        nu_val = mid_q75 / denom_h
        u_val = ((nu_val - float(rho)) / 2.0).clamp(0.0, 1.0)
        return float((u_val * u_val * (3.0 - 2.0 * u_val)).item())

    @staticmethod
    def _sq_euclidean_f64(x_flat: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
        """Exact-gradient, O(B_x * B_y)-memory squared Euclidean matrix in float64.

        In float64, every finite float32 input up to ``FLT_MAX`` (~3.4e38) squares
        without overflow (``(3.4e38)^2 ~ 1.15e77 << DBL_MAX ~ 1.8e308``). Adding the
        detached floor correction keeps forward values strictly ``>= 0`` while
        preserving exact analytical gradients ``2 * (x_i - y_j)`` everywhere,
        including coincident points ``x_i == y_j``.
        """
        x64 = x_flat.double()
        y64 = y_flat.double()
        sq_raw = (
            x64.square().sum(dim=-1, keepdim=True)
            + y64.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * (x64 @ y64.T)
        )
        return sq_raw + (sq_raw.detach().clamp_min(0.0) - sq_raw.detach())

    @staticmethod
    def _born_infeld_pair(
        x: torch.Tensor,
        y: torch.Tensor,
        ell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Overflow-safe, cancellation-free C^infinity Born-Infeld pairwise costs.

        Returns ``(c_phys, c_hat)`` of shape ``[B_x, B_y]`` where
        ``c_phys = sqrt(ell^2 + ||x - y||_2^2) - ell`` gives exact intra-mode W_2
        quadratic moments ``||x - y||_2^2 / (2 * ell)`` and unit Euclidean velocity
        cap ``<= 1`` in the far tail across all coordinate scales, while
        ``c_hat = c_phys / ell`` measures dimensionless local-horizon distance for
        Sinkhorn slack screening.
        """
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dim = max(1, x_flat.shape[-1])
        ell_safe = ell.clamp_min(1e-6)

        sq64 = RHBILoss._sq_euclidean_f64(x_flat, y_flat)
        ell64 = ell_safe.detach().double()
        rad_phys = torch.sqrt(ell64.square() + sq64)
        val_phys = sq64 / (rad_phys + ell64)
        # Cancellation-free forward value paired with single-sqrt backward derivative.
        c_phys64 = (rad_phys - ell64) + (val_phys - (rad_phys - ell64)).detach()

        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
        c_phys = c_phys64.clamp_max(fmax).to(x_flat.dtype)

        with torch.no_grad():
            val_hat = val_phys.detach() / ell64
            ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
            c_hat = val_hat.clamp_max(ratio_max).to(x_flat.dtype)
        return c_phys, c_hat

    @staticmethod
    @torch.no_grad()
    def _born_infeld_c_hat_detached(
        x: torch.Tensor,
        y: torch.Tensor,
        ell: torch.Tensor,
    ) -> torch.Tensor:
        """Fast pure-float32 overflow-safe detached dimensionless cost ``c_hat``."""
        x_flat = x.detach().reshape(x.shape[0], -1).float()
        y_flat = y.detach().reshape(y.shape[0], -1).float()
        dim = max(1, x_flat.shape[-1])
        ell_f = ell.detach().float().clamp_min(1e-6)
        scale = torch.maximum(
            torch.maximum(x_flat.abs().amax(), y_flat.abs().amax()),
            ell_f,
        ).clamp_min(1.0)
        dist_scaled = torch.cdist(x_flat / scale, y_flat / scale, p=2.0)
        inv_u = ell_f / scale
        # c_hat = sqrt(1 + (dist_scaled / inv_u)^2) - 1 = dist_scaled^2 / (inv_u * (sqrt(inv_u^2 + dist_scaled^2) + inv_u))
        rad = torch.sqrt(inv_u.square() + dist_scaled.square())
        val_hat = dist_scaled.square() / ((rad + inv_u) * inv_u).clamp_min(1e-38)
        ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
        return val_hat.clamp_max(ratio_max).to(x.dtype)

    @staticmethod
    def _born_infeld_cost(x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        """Dimensionless Born-Infeld cost ``sqrt(1 + ||(x - y)/ell||_2^2) - 1``."""
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dim = max(1, x_flat.shape[-1])
        ell64 = ell.clamp_min(1e-6).double()
        sq64 = RHBILoss._sq_euclidean_f64(x_flat, y_flat)
        u_sq = sq64 / ell64.square()
        rad = torch.sqrt(1.0 + u_sq)
        val = u_sq / (rad + 1.0)
        c64 = (rad - 1.0) + (val - (rad - 1.0)).detach()
        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
        return c64.clamp_max(fmax).to(x_flat.dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ell_init_py = bool(self.ell_initialized.item())

    @staticmethod
    @torch.no_grad()
    def _batch_ell_quantile_tensor(
        y_real: torch.Tensor,
        nn_quantile: float = 0.25,
        *,
        akt_horizon: bool = False,
        akt_mode: str = "sqrt",
    ) -> torch.Tensor:
        """Zero-sync scale-equivariant nearest-neighbor horizon quantile on device."""
        b_y = y_real.shape[0]
        if b_y <= 1:
            return y_real.new_tensor(1.0, dtype=torch.float32)
        y_flat = y_real.detach().reshape(b_y, -1).float()
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
        if record is not None and hasattr(record, "rhbi_ell_initialized"):
            init_cur = bool(record.rhbi_ell_initialized)
            ell_t = getattr(record, "_rhbi_ell_t", None)
            if ell_t is not None and ell_t.device == device:
                return ell_t, init_cur
            val = float(record._rhbi_ell_scalar) if hasattr(record, "_rhbi_ell_scalar") else float(record.rhbi_ell_ema)
            t = torch.tensor(val, dtype=torch.float32, device=device)
            if hasattr(record, "_rhbi_ell_t"):
                record._rhbi_ell_t = t
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
        if record is not None and hasattr(record, "rhbi_ell_initialized"):
            record.rhbi_ell_initialized = True
            if hasattr(record, "_rhbi_ell_t"):
                record._rhbi_ell_t = ell_f32.detach()
            else:
                record.rhbi_ell_ema = float(ell_f32.item())

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
        )
        if not init_cur:
            new_ell_f32 = q_f32
        else:
            new_ell_f32 = (1.0 - self.beta_ell) * ell_cur_f32 + self.beta_ell * q_f32
        self._set_ell_tensor(new_ell_f32, record)
        return new_ell_f32.clamp_min(1e-6).to(dtype=y_real.dtype)

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

        # Slack reservoir marginal S = 2.0 ensures P[b_r, b_c] = 1 + alpha in [1, 2]
        # stays strictly inside the Birkhoff polytope even when active mass alpha -> 0
        # (disjoint classes, 100% singleton batches, or B = 1 hollow self-coupling),
        # preserving exponential Sinkhorn convergence and pair-local cross-ratios.
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
    @torch.no_grad()
    def _split_two_sample_residuals(
        x: torch.Tensor,
        y: torch.Tensor,
        ell: Optional[torch.Tensor] = None,
        *,
        cvx_mu_gain: float = 0.40,
    ) -> Tuple[float, float, float, float, float]:
        """Zero-bias two-sample Energy Distance, barycenter, spread, precision, and coverage residuals."""
        xf = x.detach().reshape(x.shape[0], -1).float()
        yf = y.detach().reshape(y.shape[0], -1).float()
        bx, by = xf.shape[0], yf.shape[0]
        d_xy_mat = torch.cdist(xf, yf, p=2)
        d_xx_mat = torch.cdist(xf, xf, p=2)
        d_yy_mat = torch.cdist(yf, yf, p=2)

        d_xy = d_xy_mat.mean()
        d_xx = d_xx_mat.sum() / max(1, bx * (bx - 1))
        d_yy = d_yy_mat.sum() / max(1, by * (by - 1))
        denom_ed = (0.5 * (d_xy + 0.5 * (d_xx + d_yy))).clamp_min(1e-8)
        r_ed_t = (2.0 * d_xy - d_xx - d_yy) / denom_ed

        mu_diff_sq = (xf.mean(dim=0) - yf.mean(dim=0)).square().sum()
        var_x = xf.var(dim=0, unbiased=True).sum() if bx > 1 else mu_diff_sq.new_zeros(())
        var_y = yf.var(dim=0, unbiased=True).sum() if by > 1 else mu_diff_sq.new_zeros(())
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
                ell.detach().float().clamp_min(1e-8)
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
                    y_mid = 0.5 * (yf + yf.roll(1, dims=0))
                    d_my = torch.cdist(y_mid, yf, p=2)
                    idx = torch.arange(by, device=yf.device)
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

            # Fake-to-real 75th-quantile precision excess (tracks intra-mode halo convergence cleanly)
            prec_q75 = torch.quantile(xy_row_nn, 0.75) - torch.quantile(yy_loo_nn, 0.75)
            r_q90_t = 10.2 * prec_q75 / denom_ed

            # Real-to-fake 90th-quantile manifold/mode coverage excess
            cov90 = torch.quantile(xy_col_nn, 0.90) - torch.quantile(yy_loo_nn, 0.90)
            r_hmax_t = cov90 / denom_ed
        else:
            r_mu_sq_t = 0.16 * r_mu_sq_raw
            r_q90_t = r_ed_t
            r_hmax_t = r_ed_t

        vals = torch.stack((r_ed_t, r_mu_sq_t, r_spread_t, r_q90_t, r_hmax_t)).tolist()
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])

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
        # Compute centered difference in float64 so large constant shifts (+1000.0) cancel cleanly.
        d_bar = 0.5 * (f_flat.detach().double().mean() + r_flat.detach().double().mean())
        if self.scale_log_odds and ell is not None:
            inv_ell = 1.0 / ell.detach().double().clamp_min(1e-6)
            f_centered = ((f_flat.double() - d_bar) * inv_ell).to(f_flat.dtype)
            r_centered = ((r_flat.double() - d_bar) * inv_ell).to(r_flat.dtype)
        else:
            f_centered = (f_flat.double() - d_bar).to(f_flat.dtype)
            r_centered = (r_flat.double() - d_bar).to(r_flat.dtype)
        d_fake_tilde = self.c_max * torch.tanh(f_centered / self.c_max)
        d_real_tilde = self.c_max * torch.tanh(r_centered / self.c_max)
        return d_fake_tilde, d_real_tilde

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
        """Optional critic slack-focusing regularizer ``L_{D,slack}``."""
        if critic_weight == 0.0:
            return d_real.new_zeros(())
        b_f = x_fake.shape[0]
        b_r = y_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=False)
        c_hat_xy = self._born_infeld_c_hat_detached(x_fake, y_real, ell)
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

        # Protocol Invariant 1: Exact zero-strength bypass (no buffer mutation or RNG draw)
        if self.weight == 0.0:
            zero = torch.zeros((), device=x_fake.device, dtype=x_fake.dtype)
            if not return_metrics:
                return zero, {}
            ell_cur = (
                float(record.rhbi_ell_ema)
                if record is not None and hasattr(record, "rhbi_ell_initialized")
                else float(self.ell_ema.item())
            )
            return zero, {"rhbi_active_rate": 0.0, "rhbi_ell": ell_cur}

        b_f = x_fake.shape[0]
        b_r = y_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=allow_ell_update)

        defer_thermal = False
        if (
            simplicial_step
            and self.simplex_macro_settle
            and allow_ell_update
            and cond_fake is None
            and cond_real is None
            and b_r >= 4
        ):
            _, g_macro_fwd, _, _ = self._compute_manifold_gates(y_real.detach().reshape(b_r, -1))
            if g_macro_fwd > 1e-4:
                defer_thermal = True

        y_target = y_real if symmetric else y_real.detach()
        if feat_fake is not None and feat_real is not None:
            f_real = feat_real if symmetric else feat_real.detach()
            x_aug = torch.cat([x_fake.reshape(b_f, -1), feat_fake.reshape(b_f, -1) * ell], dim=-1)
            y_aug = torch.cat([y_target.reshape(b_r, -1), f_real.reshape(b_r, -1) * ell], dim=-1)
        else:
            x_aug = x_fake.reshape(b_f, -1)
            y_aug = y_target.reshape(b_r, -1)

        c_phys_xy, c_hat_xy = self._born_infeld_pair(x_aug, y_aug, ell)
        c_phys_xx, c_hat_xx = self._born_infeld_pair(x_aug, x_aug, ell)

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

        # Match self-coupling debiasing to each row's active OT transport
        # fraction so slack-routed singletons and empty-batch modes never suffer
        # uncompensated intra-mode repulsion.
        a_f = (float(b_f) * P_xy_act.sum(dim=1)).clamp(0.0, 1.0)
        P_xx_eff = P_xx_act * torch.sqrt(a_f.unsqueeze(1) * a_f.unsqueeze(0))

        slack_scale = ell.detach() if self.scale_log_odds else 1.0
        transport_xy = (P_xy_act * c_phys_xy).sum()
        s_f = (float(b_f) * P_xy_slack).clamp(0.0, 1.0)
        p_slack_eff = (
            P_xy_slack * (1.0 + self.slack_gain * s_f) if self.slack_gain > 0.0 else P_xy_slack
        )
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
        critic_total = local_critic_slack
        if symmetric:
            c_phys_yy, c_hat_yy = self._born_infeld_pair(y_aug, y_aug, ell)
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
                allow_ell_update
                and not defer_thermal
                and cond_fake is None
                and cond_real is None
                and b_r >= 4
                and b_f >= 4
                and hasattr(record, "observe_rhbi_thermal")
            ):
                r_ed, r_mu_sq, r_spread, r_q90, r_hmax = self._split_two_sample_residuals(
                    x_fake, y_real, ell=ell, cvx_mu_gain=self.cvx_mu_gain
                )
                r_tail = 0.0
                if (
                    not discrete_step
                    and not symmetric
                    and self.tail_spill_gain > 0.0
                    and b_f >= 8
                    and b_r >= 8
                ):
                    w_valley = self._compute_valley_gate(y_real, ell, self.rho)
                    if w_valley > 0.0:
                        with torch.no_grad():
                            xf = x_fake.detach().reshape(b_f, -1).double()
                            yr = y_real.detach().reshape(b_r, -1).double()
                            d_xy = torch.cdist(xf, yr, p=2.0)
                            d_yy = torch.cdist(yr, yr, p=2.0)
                            off_yy = ~torch.eye(b_r, dtype=torch.bool, device=yr.device)
                            denom_ed = d_yy[off_yy].mean().clamp_min(1e-6)
                            xy_row_nn = d_xy.amin(dim=1)
                            yy_loo_nn = d_yy.masked_fill(~off_yy, float("inf")).amin(dim=1)
                            prec_tail = torch.quantile(xy_row_nn, 0.96) - torch.quantile(yy_loo_nn, 0.96)
                            r_tail = float((w_valley * float(self.tail_spill_gain) * prec_tail / denom_ed).item())
                s_obs = record.observe_rhbi_thermal(
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
                if getattr(d_fake, "_rhbi_thermally_scaled", False):
                    # d_fake already carries first-order straight-through thermal scaling
                    # from _rhbi_critic_forward_hook; scale only coordinate-direct OT terms
                    # so transport and critic-slack preserve their exact 1:1 impedance ratio.
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

        metrics = {
            "rhbi_active_rate": active_rate,
            "rhbi_slack_rate": 1.0 - active_rate,
            "rhbi_tunnel_flux": float(m_vals[1]),
            "rhbi_ell": float(m_vals[2]),
        }
        return loss, metrics

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
                float(record.rhbi_ell_ema)
                if record is not None and hasattr(record, "rhbi_ell_initialized")
                else float(self.ell_ema.item())
            )
            return zero, {"rhbi_active_rate": 0.0, "rhbi_ell": ell_cur}

        b_f = d_fake.shape[0]
        b_r = d_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=allow_ell_update)

        with torch.no_grad():
            c_hat_xy = self._born_infeld_c_hat_detached(x_fake, y_real, ell)
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
        metrics = {
            "rhbi_active_rate": active_rate,
            "rhbi_slack_rate": 1.0 - active_rate,
            "rhbi_tunnel_flux": float(m_vals[1]),
            "rhbi_ell": float(m_vals[2]),
        }
        return loss, metrics

    def conditional_fiber_transport_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        c_cont: torch.Tensor,
        *,
        cond_disc: Optional[torch.Tensor] = None,
        mode: str = "kernel_ed",
    ) -> torch.Tensor:
        """Unbiased hollow context-screened conditional Born-Infeld discrepancy on ``(x | c)``.

        Because off-diagonal symmetric context weights ``W_{ik}(C) = W_{ki}(C)`` (``W_{ii} = 0``)
        and fiber-constant horizons ``ell_i = ell_k`` depend solely on detached contexts,
        ``E[grad_theta sum_{i != k} W_{ik}(C) (c_phys(X_i, Y_k) - 0.5 c_phys(X_i, X_k)) | C] == 0``
        identically at ``q_theta = p`` for every finite batch size ``B`` and kernel bandwidth.
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
            k_hsk_idx = torch.minimum(torch.full_like(n_i, 4), n_i.clamp_min(1)) - 1
            sigma_hsk_raw = d_sorted.gather(1, k_hsk_idx.unsqueeze(1)).squeeze(1)
            sigma_hsk_i = torch.where(
                valid_row,
                torch.where(torch.isfinite(sigma_hsk_raw), sigma_hsk_raw, torch.zeros_like(sigma_hsk_raw)).clamp_min(0.0)
                + sigma_floor,
                torch.ones_like(sigma_hsk_raw),
            )

            y_f32 = y_flat.float()
            y_scale = y_f32.abs().amax().clamp_min(1.0)
            y_unit = y_f32 / y_scale
            delta_nn = torch.linalg.vector_norm(y_unit - y_unit[nn_idx], ord=2, dim=-1) * (
                y_scale / math.sqrt(2.0)
            )
            same_valid = same_t & valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
            fiber_cnt = same_valid.float().sum(dim=1).clamp_min(1.0)
            ell_fiber = ((same_valid.float() @ delta_nn) / fiber_cnt).clamp_min(1e-5)
            ell_fiber = torch.where(valid_row, ell_fiber, torch.ones_like(ell_fiber))
            ell64 = ell_fiber.double().unsqueeze(1)

        sq_xy = self._sq_euclidean_f64(x_flat, y_flat)
        rad_xy = torch.sqrt(ell64.square() + sq_xy)
        val_xy = sq_xy / (rad_xy + ell64)
        c_xy = ((rad_xy - ell64) + (val_xy - (rad_xy - ell64)).detach()).clamp_max(fmax).to(x_flat.dtype)

        sq_xx = self._sq_euclidean_f64(x_flat, x_flat)
        rad_xx = torch.sqrt(ell64.square() + sq_xx)
        val_xx = sq_xx / (rad_xx + ell64)
        c_xx = ((rad_xx - ell64) + (val_xx - (rad_xx - ell64)).detach()).clamp_max(fmax).to(x_flat.dtype)

        if mode == "hollow_sinkhorn":
            with torch.no_grad():
                denom_c = torch.sqrt(sigma_hsk_i.unsqueeze(1) * sigma_hsk_i.unsqueeze(0)).clamp_min(1e-6)
                c_ctx_hat = (1.30 * (d_c / denom_c)).clamp_max(1e6)
                ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
                c_hat_xy = (val_xy.detach() / ell64).clamp_max(ratio_max).float() + c_ctx_hat
                c_hat_xx = (val_xx.detach() / ell64).clamp_max(ratio_max).float() + c_ctx_hat
                zeros = x_flat.new_zeros(b_n, dtype=torch.float32)
                p_xy_f = self._sinkhorn_slack(c_hat_xy, zeros, zeros, hollow=True, pair_mask=m_valid)[:b_n, :b_n]
                p_xy_r = self._sinkhorn_slack(c_hat_xy.T, zeros, zeros, hollow=True, pair_mask=m_valid.T)[:b_n, :b_n]
                p_xy = 0.5 * (p_xy_f + p_xy_r.T)
                p_xx_raw = self._sinkhorn_slack(c_hat_xx, zeros, zeros, hollow=True, pair_mask=m_valid)[:b_n, :b_n]
                p_xx = 0.5 * (p_xx_raw + p_xx_raw.T)
                p_xy.fill_diagonal_(0.0)
                p_xx.fill_diagonal_(0.0)
                a_xy = p_xy.sum(dim=1)
                a_xx = p_xx.sum(dim=1)
                r_self = (a_xy / a_xx.clamp_min(1e-12)).clamp(0.0, 1.0)
                p_xx_eff = (p_xx * torch.sqrt(r_self.unsqueeze(1) * r_self.unsqueeze(0))).to(x_flat.dtype)
                p_xy_dt = p_xy.to(x_flat.dtype)
            return (p_xy_dt * c_xy).sum() - 0.5 * (p_xx_eff * c_xx).sum()

        if mode == "imq_cmmd":
            psi_xy = torch.zeros_like(sq_xy)
            psi_xx = torch.zeros_like(sq_xx)
            for s_mult in (0.25, 1.0, 4.0):
                ell_s = ell64 * s_mult
                ell_s2 = ell_s.square()
                rad_s_xy = torch.sqrt(ell_s2 + sq_xy)
                psi_xy = psi_xy + (ell_s * sq_xy) / (rad_s_xy * (ell_s + rad_s_xy))
                rad_s_xx = torch.sqrt(ell_s2 + sq_xx)
                psi_xx = psi_xx + (ell_s * sq_xx) / (rad_s_xx * (ell_s + rad_s_xx))
            psi_xy_t = (psi_xy * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)
            psi_xx_t = (psi_xx * (1.0 / 3.0)).clamp_max(fmax).to(x_flat.dtype)
            with torch.no_grad():
                denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
                k_c = torch.where(m_valid, torch.exp(-d_c.square() / denom_2s), 0.0)
                deg_c = 1.0 + k_c.sum(dim=1)
                inv_sqrt_deg = torch.rsqrt(deg_c)
                w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
                w_imq_dt = (0.5 * (w_imq + w_imq.T)).to(x_flat.dtype)
            return (w_imq_dt * psi_xy_t).sum() - 0.5 * (w_imq_dt * psi_xx_t).sum()

        with torch.no_grad():
            denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
            log_a = torch.where(m_valid, -d_c.square() / denom_2s, float("-inf"))
            safe_diag = (~valid_row).diag()
            log_a_safe = torch.where(safe_diag, torch.zeros_like(log_a), log_a)
            u = log_a.new_zeros(b_n)
            for _ in range(6):
                u = torch.where(
                    valid_row,
                    0.5 * (u - torch.logsumexp(log_a_safe + u.unsqueeze(0), dim=1)),
                    u,
                )
            w_sym = torch.where(m_valid, torch.exp(log_a_safe + u.unsqueeze(1) + u.unsqueeze(0)), 0.0)
            w_sym = (0.5 * (w_sym + w_sym.T)) / float(b_n)
            w_dt = w_sym.to(x_flat.dtype)
        return (w_dt * c_xy).sum() - 0.5 * (w_dt * c_xx).sum()

    @staticmethod
    def _born_infeld_aligned(x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        """Row-wise aligned C^infinity physical Born-Infeld distance ``[N]``."""
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dim = max(1, x_flat.shape[-1])
        ell64 = ell.detach().clamp_min(1e-6).double()
        diff64 = x_flat.double() - y_flat.double()
        sq64 = diff64.square().sum(dim=-1)
        rad = torch.sqrt(ell64.square() + sq64)
        val = sq64 / (rad + ell64)
        c64 = (rad - ell64) + (val - (rad - ell64)).detach()
        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
        return c64.clamp_max(fmax).to(x_flat.dtype)

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
    ) -> torch.Tensor:
        """Local k-NN 1-simplex and 2-simplex manifold chart regularizer.

        Queries interior points along local latent k-NN chords and triangle faces
        using deterministic Weyl stratification (zero RNG draws, zero step clocks).
        When ``chart_weight > 0`` and ``supp(y_real)`` forms a low-dimensional
        connected manifold under ``H_0`` Vietoris-Rips persistence and diameter-to-step
        span, jointly anchors vertices via balanced Born-Infeld Sinkhorn transport,
        enforces triangle inradius interior margins, and aligns local latent geodesic
        distances with output manifold shortest-path chart geometry.
        """
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
        m_p = z_table.shape[0]
        b_r = y_real.shape[0]
        if self.weight == 0.0 or m_p < 2 or b_r < 1:
            return y_real[:0].sum()

        y_flat = y_real.detach().reshape(b_r, -1)
        g_macro = 0.0
        if use_macro and chart_weight > 0.0 and b_r >= 4:
            g_conn_m, g_macro, q90_m, y_scale_m = self._compute_manifold_gates(y_flat)

        with torch.no_grad():
            z_det = z_table.detach()
            ell = self._update_ell(y_real, record=record, allow_update=False)
            d_zz = torch.cdist(z_det.double(), z_det.double(), p=2.0)
            d_zz = d_zz + 1e-12 * torch.arange(m_p, device=z_det.device, dtype=torch.float64).unsqueeze(0)
            d_zz.fill_diagonal_(float("inf"))
            k_nn = min(4, m_p - 1)
            nbrs = d_zz.topk(k_nn, dim=-1, largest=False).indices

            if g_macro > 1e-4:
                g_conn = g_conn_m
                q90_y_u = q90_m
                y_scale = y_scale_m
                y_u = y_flat.float() / y_scale
            else:
                g_conn = 0.0
                q90_y_u = 0.1
                y_scale = y_flat.float().abs().amax().clamp_min(1.0)
                y_u = y_flat.float() / y_scale
                if chart_weight > 0.0 and b_r >= 4:
                    d_yy = torch.cdist(y_u, y_u, p=2.0)
                    off_y = ~torch.eye(b_r, dtype=torch.bool, device=y_u.device)
                    d_yy_nn = d_yy.masked_fill(~off_y, float("inf"))
                    nn_y = d_yy_nn.amin(dim=-1)
                    q90_t = torch.quantile(nn_y, 0.90).clamp_min(1e-8)
                    q90_y_u = float(q90_t.clamp_min(1e-4).item())
                    span_y = torch.quantile(d_yy[off_y], 0.90) / q90_t
                    m_path = d_yy.clone()
                    for _ in range(7):
                        m_path = torch.minimum(
                            m_path, torch.maximum(m_path.unsqueeze(2), m_path.unsqueeze(0)).amin(dim=1)
                        )
                    kappa_h0 = m_path.amax() / q90_t
                    u_conn = ((10.0 - kappa_h0) / 3.0).clamp(0.0, 1.0)
                    u_span = ((span_y - 1.15) / 0.25).clamp(0.0, 1.0)
                    s_conn = u_conn * u_conn * (3.0 - 2.0 * u_conn)
                    s_span = u_span * u_span * (3.0 - 2.0 * u_span)
                    g_conn = float((s_conn * s_span).item())

            phi = float(getattr(record, "rhbi_chord_phase", 0.0)) if record is not None else 0.0
            n_q = max(4, b_r)
            n_edge = (n_q // 2) if (m_p >= 3 and k_nn >= 2 and n_q >= 4) else n_q
            n_tri = n_q - n_edge

            k_edge = min(3, k_nn)
            r_e = torch.arange(n_edge, device=z_det.device)
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
                # Half exact barycenters (1/3, 1/3, 1/3), half interior barycentric ring
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
            x_verts_live = gen_fn(z_table).reshape(m_p, -1)
            x_verts = x_verts_live.detach()
        else:
            x_verts_live = None
            with torch.no_grad():
                x_verts = gen_fn(z_det).detach().reshape(m_p, -1)

        x_simp = gen_fn(z_simp).reshape(n_q, -1)

        if (
            g_macro > 1e-4
            and self.thermal_tau > 0.0
            and record is not None
            and torch.is_grad_enabled()
            and not getattr(record, "_in_protected_probe", False)
            and b_r >= 4
            and hasattr(record, "observe_rhbi_thermal")
        ):
            with torch.no_grad():
                x_man = torch.cat([x_verts, x_simp.detach()], dim=0)
                r_ed, r_mu_sq, r_spread, r_q90, r_hmax = self._split_two_sample_residuals(
                    x_man, y_flat, ell=ell, cvx_mu_gain=self.cvx_mu_gain
                )
                s_obs = record.observe_rhbi_thermal(
                    r_ed,
                    r_mu_sq,
                    r_spread,
                    r_q90,
                    r_hmax,
                    self.thermal_tau * (1.0 + 0.25 * g_macro),
                    shock_mu_ratio=self.thermal_shock_mu,
                    r_tail=0.0,
                )
                _publish_thermal(record, y_real.device, s_obs)

        c_phys_sr, c_hat_sr = self._born_infeld_pair(x_simp, y_flat, ell)
        w_sr = torch.softmax(-c_hat_sr.float() / self.eps, dim=1).to(c_phys_sr.dtype)
        proj_loss = (w_sr * c_phys_sr).sum(dim=1).mean()

        eff_iso = ((1.0 - g_macro) * float(iso_weight) + g_macro * 0.25) if g_macro > 1e-4 else float(iso_weight)
        iso_loss = proj_loss.new_zeros(())
        if eff_iso > 0.0:
            x_e = x_simp[:n_edge]
            d0_e = self._born_infeld_aligned(x_e, x_verts[v0_e], ell)
            d1_e = self._born_infeld_aligned(x_e, x_verts[v1_e], ell)
            iso_edge = ((1.0 - t_e.to(d0_e.dtype)) * d0_e - t_e.to(d0_e.dtype) * d1_e).abs().mean()
            if n_tri > 0:
                x_t = x_simp[n_edge:]
                d0_t = self._born_infeld_aligned(x_t, x_verts[v0_t], ell)
                d1_t = self._born_infeld_aligned(x_t, x_verts[v1_t], ell)
                d2_t = self._born_infeld_aligned(x_t, x_verts[v2_t], ell)
                with torch.no_grad():
                    if g_conn > 1e-4 and chart_weight > 0.0:
                        vx0 = x_verts[v0_t].double()
                        vx1 = x_verts[v1_t].double()
                        vx2 = x_verts[v2_t].double()
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
                        ell64 = ell.detach().clamp_min(1e-6).double()
                        margin_raw = 0.70 * (torch.sqrt(ell64.square() + r_in.square()) - ell64)
                        if g_macro > 1e-4:
                            zv0 = z_det[v0_t].double()
                            zv1 = z_det[v1_t].double()
                            zv2 = z_det[v2_t].double()
                            zt64 = z_tri.detach().double()
                            z_diam = torch.maximum(
                                torch.maximum(
                                    torch.linalg.vector_norm(zv0 - zv1, ord=2, dim=-1),
                                    torch.linalg.vector_norm(zv1 - zv2, ord=2, dim=-1),
                                ),
                                torch.linalg.vector_norm(zv2 - zv0, ord=2, dim=-1),
                            ).clamp_min(1e-8)
                            z_min_d = torch.minimum(
                                torch.minimum(
                                    torch.linalg.vector_norm(zt64 - zv0, ord=2, dim=-1),
                                    torch.linalg.vector_norm(zt64 - zv1, ord=2, dim=-1),
                                ),
                                torch.linalg.vector_norm(zt64 - zv2, ord=2, dim=-1),
                            )
                            u_fat = ((z_min_d / z_diam - 0.12) / 0.10).clamp(0.0, 1.0)
                            eta_fat = u_fat * u_fat * (3.0 - 2.0 * u_fat)
                            margin = ((1.0 - g_macro + g_macro * eta_fat) * margin_raw).to(d0_t.dtype)
                        else:
                            margin = margin_raw.to(d0_t.dtype)
                    else:
                        e01 = self._born_infeld_aligned(x_verts[v0_t], x_verts[v1_t], ell)
                        e12 = self._born_infeld_aligned(x_verts[v1_t], x_verts[v2_t], ell)
                        e20 = self._born_infeld_aligned(x_verts[v2_t], x_verts[v0_t], ell)
                        e_min = torch.minimum(torch.minimum(e01, e12), e20)
                        margin = 0.20 * e_min
                iso_tri = (
                    F.relu(margin - d0_t) + F.relu(margin - d1_t) + F.relu(margin - d2_t)
                ).mean() / 3.0
                iso_loss = 0.5 * (iso_edge + iso_tri)
            else:
                iso_loss = iso_edge

        total = proj_loss + eff_iso * iso_loss

        if g_conn > 1e-4 and chart_weight > 0.0:
            if x_verts_live is not None:
                c_phys_vy, _ = self._born_infeld_pair(x_verts_live, y_flat, ell)
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
                total = total + (1.10 * g_conn) * vert_ot_loss

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
                    w_chart = torch.where(off_mask, (0.20 + g_hat.square()).pow(-2.0), 0.0)
                    w_chart = w_chart / w_chart.sum().clamp_min(1e-12)
                dz_sq = self._sq_euclidean_f64(z_table, z_table)
                dz_rms = torch.sqrt(dz_sq[off_mask].mean().clamp_min(1e-12))
                dz_hat = torch.sqrt(dz_sq + 1e-12) / dz_rms
                chart_stress = (w_chart * (dz_hat - g_hat).square()).sum().to(total.dtype)
                total = total + (g_conn * float(chart_weight)) * ell.detach().to(total.dtype) * chart_stress

        s_therm = float(getattr(record, "_thermal_scale", 1.0)) if record is not None else 1.0
        if s_therm < 1.0 and total.requires_grad:
            total = total * s_therm + total.detach() * (1.0 - s_therm)
        return self.weight * total
