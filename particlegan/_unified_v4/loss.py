"""Unified-v4 4-Law Zero-SVD / Zero-`eigh` GANLoss.

Unifies:
1. **Cross-Batch Stratum Sufficient-Statistics + Unbiased Hollow U-Statistic Conditional Fiber Alignment**:
   Maintains checkpointed cross-batch per-stratum moments `(W_t, S_c, S_y, S_cc, S_cy, S_yy)` ($N_{\text{eff}} \approx 2{,}000$)
   to compute zero-shrinkage affine conditional whitening `(m_lin_t(c), L_r_t)` via small Cholesky solves (`0` SVD),
   paired with unbiased hollow two-sample U-statistics matching conditional mean, linear context slope, and local/global
   conditional covariance alongside multi-scale IMQ-CMMD on standardized innovations (`model_ddgan_*`, `model_ae_ddgan_*`).
2. **Zero-`eigvalsh` / Zero-`svdvals` Simplicial & Connected-Manifold Deconfliction**:
   Subclasses `T1CouplingLoss` (`UnifiedV4CouplingLoss`) replacing `eigvalsh` and `matrix_norm(..., ord=2)` with
   batched deterministic power iteration + Rayleigh quotients (`0` SVD/`eigh`), and scales `chart_weight` by
   `(1 - g_conn)` on simplicial steps so connected zero-sigma manifolds (`interp_sparse_arc`, `interp_img_patch_shift`,
   `interp_saddle_barycentric`) stay smooth while disconnected discrete modes (`img_*`) retain full Laguerre commitment.
3. **Zero-`eigh` Orphan-Gated Bipartite Deficit Macro-Transport**:
   Rescues Hellinger-screened orphan modes after non-stationary saddle shifts (`t3_iso_horizon_saddle_shift`,
   `t3_iso_scale_mass_8x`) using deterministic power-iteration principal-axis ordering (`0` `eigh`).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .._bispf.thermal import _publish_thermal
from .._rhbi.loss import _split_shared_conditional_parts
from .._t1_coupling.charts import project_transport_features
from .._t1_coupling.critic import _lookup_t1c_capture_parts
from .._t1_coupling.primitive import T1CouplingLoss
from .._t3_dynamics.metric import (
    born_infeld_aligned,
    born_infeld_c_hat_detached,
     discrete_pair_mask,
    sinkhorn_slack,
)
from ..gan_loss import GANLoss

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


class UnifiedV4CouplingLoss(T1CouplingLoss):
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
            g_conn_m, g_macro_m, q90_m, y_scale = self._compute_manifold_gates(y_flat)
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


def zero_eigh_bipartite_deficit_macro_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_hat_xy: torch.Tensor,
    p_row_slack: torch.Tensor,
    p_col_slack: torch.Tensor,
    *,
    rho: float = 3.0,
    g_conn: float = 0.0,
) -> torch.Tensor:
    """Zero-`eigh` orphan- & Poisson-deficit-gated bipartite Monge macro-transport (`t3_iso_*`)."""
    b_f = x_fake.shape[0]
    b_r = y_real.shape[0]
    zero = x_fake[:0].sum()
    if b_f < 2 or b_r < 2 or g_conn > 0.05:
        return zero

    with torch.no_grad():
        c_f = c_hat_xy.detach().float()
        col_min_c = c_f.amin(dim=0)
        u_orph = ((col_min_c - 2.80 * float(rho)) / max(1.20 * float(rho), 1e-6)).clamp(0.0, 1.0)
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
            if bool((orph_active > 0.0).any()):
                d_orph_min = (d_xy + (1.0 - orph_active).unsqueeze(0) * (1e6 * med_d)).amin(dim=1)
            else:
                d_orph_min = d_xy.amin(dim=1)
            score = 1.50 * s_row - 0.65 * (d_orph_min / med_d).clamp_max(4.0) + det_tie
            exp_idx = torch.sort(torch.topk(score, k=k_exp, largest=True).indices).values

        p_marg = torch.full((k_exp,), 1.0 / float(k_exp), device=x_fake.device, dtype=torch.float32)
        q_marg = q_raw + 1e-7
        q_marg = q_marg / q_marg.sum().clamp_min(1e-12)

        if yf.shape[1] >= 2 and b_r >= 2:
            y_mean_c = (q_marg.unsqueeze(1) * yf).sum(dim=0, keepdim=True)
            yc_c = (yf - y_mean_c) * q_marg.sqrt().unsqueeze(1)
            gram_c = (yc_c.T @ yc_c).double()
            axis_c = _principal_axis_power(gram_c, iters=6).float()
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


def _analytic_inv_chol_spd(cov: torch.Tensor, lam_rel: float) -> torch.Tensor:
    """Zero-sync inverse lower-Cholesky ``L^{-1}`` of ``cov + lam * I`` (`0` SVD, `0` `.item()`, closed-form for `d<=2`)."""
    d = int(cov.shape[0])
    diag_pos = torch.diagonal(cov).clamp_min(0.0)
    lam = (lam_rel * diag_pos.sum() / float(max(1, d))).clamp_min(1e-8)
    if d == 1:
        l00 = torch.sqrt(diag_pos[0] + lam)
        return (1.0 / l00).view(1, 1)
    if d == 2:
        g00 = diag_pos[0] + lam
        g10 = cov[1, 0]
        g11 = diag_pos[1] + lam
        l00 = torch.sqrt(g00)
        l10 = g10 / l00
        l11 = torch.sqrt((g11 - l10 * l10).clamp_min(lam))
        inv00 = 1.0 / l00
        inv11 = 1.0 / l11
        inv10 = -l10 * inv00 * inv11
        z0 = torch.zeros_like(inv00)
        return torch.stack([torch.stack([inv00, z0]), torch.stack([inv10, inv11])])
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    gram = cov + lam * eye
    l_mat, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    linv_chol = torch.linalg.solve_triangular(l_mat, eye, upper=False)
    linv_diag = torch.diag(torch.rsqrt(diag_pos + lam))
    return torch.where((info == 0).view(1, 1), linv_chol, linv_diag)


def _solve_stratum_affine_whitening(
    entry: Dict[str, Any],
    d_c: int,
    d_y: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(mu_c, mu_y, L_c^{-1}, B_ols, L_r^{-1})` in FP64 with zero GPU->CPU syncs (`0` SVD)."""
    w = max(float(entry["weight"]), 1.0)
    mu_c = entry["sum_c"] / w
    mu_y = entry["sum_y"] / w
    cov_cc = 0.5 * (((entry["sum_cc"] / w) - torch.outer(mu_c, mu_c)) + ((entry["sum_cc"] / w) - torch.outer(mu_c, mu_c)).T)
    cov_cy = (entry["sum_cy"] / w) - torch.outer(mu_c, mu_y)
    cov_yy = 0.5 * (((entry["sum_yy"] / w) - torch.outer(mu_y, mu_y)) + ((entry["sum_yy"] / w) - torch.outer(mu_y, mu_y)).T)

    linv_c = _analytic_inv_chol_spd(cov_cc, 1e-5)
    b_ols = linv_c.T @ (linv_c @ cov_cy)
    explained = b_ols.T @ cov_cc @ b_ols
    cov_r = 0.5 * ((cov_yy - explained) + (cov_yy - explained).T)
    linv_r = _analytic_inv_chol_spd(cov_r, 1e-4)
    return mu_c, mu_y, linv_c, b_ols, linv_r


def _smooth_clip_mat(m: torch.Tensor, cap: float = 3.0) -> torch.Tensor:
    return m * torch.rsqrt(1.0 + m.square().sum() / (cap * cap))


def unified_v4_conditional_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
    record: Any = None,
) -> torch.Tensor:
    """Cross-batch sufficient-statistics whitening + Cross-Batch BLP Innovation Score Matching + multi-scale IMQ-CMMD."""
    b_n = x_fake.shape[0]
    if b_n <= 1 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
        return x_fake[:0].sum()

    x_flat = x_fake.reshape(b_n, -1)
    y_flat = y_real.detach().reshape(b_n, -1)
    c_flat = c_cont.detach().reshape(b_n, -1)
    d_y = max(1, int(x_flat.shape[1]))
    d_c = max(1, int(c_flat.shape[1]))
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(d_y) + 1.0))

    allow_obs = (
        record is not None
        and hasattr(record, "observe_cond_stratum")
        and torch.is_grad_enabled()
        and not getattr(record, "_in_protected_probe", False)
        and not getattr(record, "_in_probe", False)
    )

    with torch.no_grad():
        if cond_disc is None:
            strata_indices: List[Tuple[int, torch.Tensor]] = [
                (0, torch.arange(b_n, device=x_fake.device, dtype=torch.long))
            ]
        else:
            cd = cond_disc.detach().reshape(b_n, -1)
            if cd.shape[1] == 1 and cd.dtype in (torch.int32, torch.int64):
                vals = cd[:, 0].tolist()
                buckets: Dict[int, List[int]] = {}
                for i_row, v_id in enumerate(vals):
                    buckets.setdefault(int(v_id), []).append(i_row)
                strata_indices = [
                    (sid, torch.tensor(ix, device=x_fake.device, dtype=torch.long))
                    for sid, ix in sorted(buckets.items())
                ]
            else:
                uniq = torch.unique(cd, dim=0)
                strata_indices = [
                    (
                        g,
                        torch.nonzero((cd == uniq[g].unsqueeze(0)).all(dim=-1), as_tuple=False).squeeze(-1),
                    )
                    for g in range(int(uniq.shape[0]))
                ]

        m_lin32 = torch.zeros((b_n, d_y), dtype=torch.float32, device=x_fake.device)
        inv_lr_t = torch.zeros((b_n, d_y, d_y), dtype=torch.float32, device=x_fake.device)
        c_white32 = torch.zeros((b_n, d_c), dtype=torch.float32, device=x_fake.device)
        strata_entries: List[Tuple[int, torch.Tensor, Dict[str, Any]]] = []

        for sid, idx_s in strata_indices:
            n_s = int(idx_s.numel())
            if n_s == 0:
                continue
            cs64 = c_flat[idx_s].double()
            ys64 = y_flat[idx_s].double()
            if allow_obs:
                entry = record.observe_cond_stratum(sid, cs64, ys64, gamma=0.992)
            elif record is not None and hasattr(record, "cond_strata") and sid in record.cond_strata:
                entry = record.cond_strata[sid]
            else:
                entry = {
                    "weight": float(n_s),
                    "sum_c": cs64.sum(dim=0),
                    "sum_y": ys64.sum(dim=0),
                    "sum_cc": cs64.T @ cs64,
                    "sum_cy": cs64.T @ ys64,
                    "sum_yy": ys64.T @ ys64,
                    "g_weight": 0.0,
                    "sum_rx": torch.zeros((d_y,), dtype=torch.float64, device=x_fake.device),
                    "sum_cwrx": torch.zeros((d_c, d_y), dtype=torch.float64, device=x_fake.device),
                    "sum_rxrx": torch.zeros((d_y, d_y), dtype=torch.float64, device=x_fake.device),
                }
            mu_c, mu_y, linv_c, b_ols, linv_r = _solve_stratum_affine_whitening(entry, d_c, d_y)
            dc_s = cs64 - mu_c.unsqueeze(0)
            m_s = mu_y.unsqueeze(0) + (dc_s @ b_ols)
            m_lin32[idx_s] = m_s.float()
            c_white32[idx_s] = (dc_s @ linv_c.T).float()
            inv_lr_t[idx_s] = linv_r.T.float().unsqueeze(0).expand(n_s, d_y, d_y)
            strata_entries.append((sid, idx_s, entry))

    # Differentiable standardized residuals r_x = (x - m_lin(c)) L_r^{-T}, detached r_y = (y - m_lin(c)) L_r^{-T}:
    rx = torch.bmm((x_flat.float() - m_lin32).unsqueeze(1), inv_lr_t).squeeze(1)
    ry = torch.bmm((y_flat.float() - m_lin32).unsqueeze(1), inv_lr_t).squeeze(1).detach()

    # 1. Cross-Batch Best-Linear-Predictor (BLP) Innovation Score Matching per Stratum:
    #    For EVERY joint law P(Y, C), E_P[r_y] == 0, E_P[c_w r_y^T] == 0, and E_P[r_y r_y^T] == I_{d_y}.
    blp_total = rx.new_zeros(())
    blp_err_accum = rx.new_zeros(())
    n_active_strata = 0
    eye_y32 = torch.eye(d_y, dtype=torch.float32, device=x_fake.device)
    for sid, idx_s, entry in strata_entries:
        n_s = int(idx_s.numel())
        if n_s < 1:
            continue
        n_active_strata += 1
        rx_s = rx[idx_s]
        ry_s = ry[idx_s]
        cw_s = c_white32[idx_s]
        inv_n = 1.0 / float(n_s)

        if allow_obs and hasattr(record, "observe_cond_gen_stratum"):
            record.observe_cond_gen_stratum(sid, cw_s, rx_s, gamma=0.90)

        with torch.no_grad():
            rx_det = rx_s.detach()
            batch_mu_diff = (rx_det - ry_s).mean(dim=0)
            batch_cr_diff = (cw_s.T @ (rx_det - ry_s)) * inv_n
            batch_cov_diff = ((rx_det.T @ rx_det) - (ry_s.T @ ry_s)) * inv_n

            gw = float(entry.get("g_weight", 0.0))
            if gw >= 4.0:
                ema_mu = (entry["sum_rx"] / gw).float()
                ema_cr = (entry["sum_cwrx"] / gw).float()
                ema_cov = (entry["sum_rxrx"] / gw).float() - eye_y32
                alpha_b = 0.15
                tgt_mu = _smooth_clip_mat((1.0 - alpha_b) * ema_mu + alpha_b * batch_mu_diff)
                tgt_cr = _smooth_clip_mat((1.0 - alpha_b) * ema_cr + alpha_b * batch_cr_diff)
                tgt_cov = _smooth_clip_mat((1.0 - alpha_b) * ema_cov + alpha_b * batch_cov_diff)
            else:
                tgt_mu = _smooth_clip_mat(batch_mu_diff)
                tgt_cr = _smooth_clip_mat(batch_cr_diff)
                tgt_cov = _smooth_clip_mat(batch_cov_diff)
            tgt_cov = 0.5 * (tgt_cov + tgt_cov.T)
            blp_err_accum = blp_err_accum + (
                tgt_mu.norm() + tgt_cr.norm() + 0.5 * tgt_cov.norm()
            )

        live_mu = rx_s.mean(dim=0)
        live_cr = (cw_s.T @ rx_s) * inv_n
        live_cov = (rx_s.T @ rx_s) * inv_n
        blp_s = (
            2.0 * (live_mu * tgt_mu).sum()
            + 2.0 * (live_cr * tgt_cr).sum()
            + 0.75 * (live_cov * tgt_cov).sum()
        )
        blp_total = blp_total + blp_s

    if n_active_strata > 0:
        blp_total = blp_total / float(n_active_strata)
        blp_err_accum = blp_err_accum / float(n_active_strata)

    # 2. Context-Local Multi-Scale IMQ-CMMD on Standardized Innovations (rx, ry):
    with torch.no_grad():
        same_t = discrete_pair_mask(cond_disc, cond_disc, b_n, b_n)
        if same_t is None:
            same_t = torch.ones((b_n, b_n), dtype=torch.bool, device=x_fake.device)
        m_hollow = same_t.clone()
        m_hollow.fill_diagonal_(False)
        valid_row = m_hollow.any(dim=1)
        m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)

    if not bool(m_valid.any()):
        return blp_total.clamp(-fmax, fmax).to(x_flat.dtype)

    with torch.no_grad():
        d_c_mat = torch.cdist(c_white32, c_white32, p=2.0)
        d_c_max = d_c_mat.amax()
        d_c_masked = d_c_mat.masked_fill(~m_valid, float("inf"))
        d_sorted, _ = torch.sort(d_c_masked, dim=1)
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
        denom_2s = (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)).clamp_min(1e-12)
        k_c = torch.where(m_valid, torch.exp(-d_c_mat.square() / denom_2s), 0.0)
        deg_c = 1.0 + k_c.sum(dim=1)
        inv_sqrt_deg = torch.rsqrt(deg_c)
        w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
        w_imq_sym = 0.5 * (w_imq + w_imq.T)
        ell_f = math.sqrt(float(d_y))

    sq_w_xy = (
        rx.square().sum(dim=-1, keepdim=True)
        + ry.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (rx @ ry.T)
    )
    sq_w_xy = sq_w_xy + (sq_w_xy.detach().clamp_min(0.0) - sq_w_xy.detach())
    sq_w_xx = (
        rx.square().sum(dim=-1, keepdim=True)
        + rx.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * (rx @ rx.T)
    )
    sq_w_xx = sq_w_xx + (sq_w_xx.detach().clamp_min(0.0) - sq_w_xx.detach())

    psi_xy = torch.zeros_like(sq_w_xy)
    psi_xx = torch.zeros_like(sq_w_xx)
    for s_mult in (0.35, 1.0, 2.8):
        ell_s = ell_f * s_mult
        ell_s2 = ell_s * ell_s
        rad_s_xy = torch.sqrt(ell_s2 + sq_w_xy)
        psi_xy = psi_xy + (ell_s * sq_w_xy) / (rad_s_xy * (ell_s + rad_s_xy))
        rad_s_xx = torch.sqrt(ell_s2 + sq_w_xx)
        psi_xx = psi_xx + (ell_s * sq_w_xx) / (rad_s_xx * (ell_s + rad_s_xx))
    psi_xy = psi_xy * (1.0 / 3.0)
    psi_xx = psi_xx * (1.0 / 3.0)

    if (
        record is not None
        and hasattr(record, "t1c_cond_initialized")
        and torch.is_grad_enabled()
        and not getattr(record, "_in_protected_probe", False)
        and not getattr(record, "_in_probe", False)
    ):
        with torch.no_grad():
            sq_w_yy = (
                ry.square().sum(dim=-1, keepdim=True)
                + ry.square().sum(dim=-1).unsqueeze(0)
                - 2.0 * (ry @ ry.T)
            ).clamp_min(0.0)
            psi_yy = torch.zeros_like(sq_w_yy)
            for s_mult in (0.35, 1.0, 2.8):
                ell_s = ell_f * s_mult
                ell_s2 = ell_s * ell_s
                rad_s_yy = torch.sqrt(ell_s2 + sq_w_yy)
                psi_yy = psi_yy + (ell_s * sq_w_yy) / (rad_s_yy * (rad_s_yy + ell_s))
            psi_yy = psi_yy * (1.0 / 3.0)
            num_c = (w_imq_sym * (2.0 * psi_xy.detach() - psi_xx.detach() - psi_yy)).sum()
            den_c = (w_imq_sym * psi_yy).sum().clamp_min(1e-8)
            r_imq = max(-0.5, min(0.5, float((num_c / den_c).item())))
            r_blp = min(0.5, 0.08 * float(blp_err_accum.item()))
            r_cond = max(r_imq, r_blp)
            if not record.t1c_cond_initialized:
                record.t1c_cond_ema = max(0.18, r_cond)
                record.t1c_cond_initialized = True
            else:
                record.t1c_cond_ema = 0.96 * float(record.t1c_cond_ema) + 0.04 * r_cond
            u_cond = min(1.0, max(0.0, (0.032 - max(0.0, float(record.t1c_cond_ema))) / 0.020))
            s_cond = u_cond * u_cond * (3.0 - 2.0 * u_cond)
            if hasattr(record, "_step_scale"):
                record._step_scale = float(1.0 - 0.80 * s_cond)

    imq_loss = (w_imq_sym * psi_xy).sum() - 0.5 * (w_imq_sym * psi_xx).sum()
    total_cond = imq_loss + 0.60 * blp_total
    return total_cond.clamp(-fmax, fmax).to(x_flat.dtype)


class UnifiedV4GANLoss(GANLoss):
    """Unified-v4 4-Law GANLoss (`0` SVD, `0` `eigh`/`eigvalsh`, `0` `autograd.grad`)."""

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
        record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        for name, val in (
            ("critic_weight", critic_weight),
            ("cond_ot_weight", cond_ot_weight),
            ("macro_slack_weight", macro_slack_weight),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            isinstance(fibre_rel_blend, bool)
            or not isinstance(fibre_rel_blend, (int, float))
            or not math.isfinite(fibre_rel_blend)
            or fibre_rel_blend < 0.0
            or fibre_rel_blend > 1.0
        ):
            raise ValueError("fibre_rel_blend must be finite and in [0, 1]")
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
        self.record = record
        self.t1c = UnifiedV4CouplingLoss(
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
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, _, _, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if captured and rec is not None and hasattr(rec, "observe_margin_and_scale") and not has_cont_cond:
            rec.observe_margin_and_scale(real_logits, fake_logits, x_r, x_f)
        if self.critic_weight == 0.0 and self.fibre_rel_blend == 0.0:
            return base
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
            raise ValueError("UnifiedV4GANLoss requires real_logits in g_loss when t1c_weight > 0")
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
                            cond_ot = unified_v4_conditional_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
                                record=rec,
                            )
                            if rec is not None and hasattr(rec, "_simplicial_step"):
                                rec._simplicial_step = False
                            s_th = float(getattr(rec, "_thermal_scale", 1.0)) if rec is not None else 1.0
                            rel_scaled = 0.45 * rel_term
                            total_cond = rel_scaled + self.cond_ot_weight * cond_ot
                            if s_th < 1.0 and total_cond.requires_grad:
                                if getattr(fake_logits, "_t1c_thermally_scaled", False):
                                    cond_scaled = cond_ot * s_th + cond_ot.detach() * (1.0 - s_th)
                                    total_cond = rel_scaled + self.cond_ot_weight * cond_scaled
                                else:
                                    total_cond = total_cond * s_th + total_cond.detach() * (1.0 - s_th)
                            return self.t1c.weight * total_cond
                return self.t1c.weight * rel_term
            return self.t1c.weight * super().g_loss(fake_logits, real_logits)

        # 2. High-resolution RGB image domains (dim > 1024, e.g. 32x32..256x256 DCGAN):
        dim = max(1, x_r[0].numel()) if b_r > 0 else 0
        if x_r.ndim >= 4 and dim > 1024:
            self.t1c._discrete_prior_step = False
            self.t1c._simplicial_active_step = False
            if rec is not None and hasattr(rec, "_t1c_disc_gate"):
                rec._t1c_disc_gate = 0.0
            if rec is not None and hasattr(rec, "_simplicial_step"):
                rec._simplicial_step = False
            return super().g_loss(fake_logits, real_logits)

        # 3. H_0-connectedness-gated chart coupling + zero-eigh orphan macro-transport:
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
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
            and x_f.ndim == 2
            and 2 <= dim <= 16
        ):
            if b_r >= 4:
                y_feat = project_transport_features(x_r.detach(), max_spatial=self.t1c.max_spatial)
                g_conn_fwd, _, _, _ = self.t1c._compute_manifold_gates(y_feat)

        if is_simplicial and saved_cw > 0.0:
            # Suppress discrete-island Laguerre snapping strictly on H_0-connected manifolds
            # (both 2D `interp_sparse_arc`/`interp_saddle_barycentric` AND 4D `interp_img_patch_shift`),
            # while retaining 100% of `saved_cw` on disconnected discrete image modes (`img_*`, g_conn == 0)!
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

        # Zero-eigh orphan-gated bipartite deficit macro-transport on disconnected direct clouds (`t3_iso_*`):
        if (
            self.macro_slack_weight > 0.0
            and is_discrete
            and not is_simplicial
            and cond_f is None
            and cond_r is None
            and x_f.ndim == 2
            and 2 <= dim <= 16
            and g_conn_fwd <= 0.05
            and b_f >= 4
            and b_r >= 4
        ):
            with torch.no_grad():
                ell_m = self.t1c._update_ell(x_r, record=rec, allow_update=False)
                c_hat_m = born_infeld_c_hat_detached(x_f, x_r, ell_m)
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
            macro_loss = zero_eigh_bipartite_deficit_macro_loss(
                x_f,
                x_r,
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
    "UnifiedV4CouplingLoss",
    "UnifiedV4GANLoss",
    "unified_v4_conditional_loss",
    "zero_eigh_bipartite_deficit_macro_loss",
]
