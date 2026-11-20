"""CondFiber GANLoss: Nonparametric Paired-Ring + Cross-Batch Sufficient-Statistics Conditional Fiber.

Composes cleanly on ``UnifiedGANLoss`` without modifying ``particlegan/_unified/``:
- On all unconditional domains (1D/2D/100-mode MoG, discrete images, 32^2..256^2 RGB DCGAN),
  delegates directly to ``UnifiedGANLoss`` with zero overhead.
- On continuous-conditioned fibers (DDGAN / AE-DDGAN reverse chains), uses ``CondFiberBank``
  (cross-batch sufficient statistics + strictly past-batch paired-context FIFO ring) providing:
  1. Swap-invariant cross-batch pooled drift removal ``m_pool(c, t)`` for multi-scale hollow IMQ-CMMD;
  2. Exact target-mean-canceling, nonparametric local conditional mean-operator matching
     ``Delta_m_loc(c_i) = sum_j alpha_ij (X_j^past - Y_j^past)`` (zero 2D-corner blind spots,
     zero boundary bias at equilibrium, zero variance collapse);
  3. Exact midpoint-canceling, split-sign bounded local conditional covariance-operator matching
     ``E_cov_loc(c_i) = sum_j alpha_ij (rx_j rx_j^T - ry_j ry_j^T)``;
  4. Operator-aware additive terminal settling controller.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .._rhbi.loss import _split_shared_conditional_parts
from .._t3_dynamics.metric import discrete_pair_mask
from .._unified.loss import UnifiedGANLoss, _zero_shrinkage_pooled_mean
from .bank import CondFiberBank


def _odd_radial_saturate(v: torch.Tensor, tau: float) -> torch.Tensor:
    """Smooth odd radial saturation ``(tau * tanh(||v|| / tau) / ||v||) * v``, linear near zero."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (tau * torch.tanh(norm / tau)) / norm
    return v * scale


def _odd_frob_saturate(mat: torch.Tensor, tau: float) -> torch.Tensor:
    """Smooth odd Frobenius-norm saturation on ``[n, d, d]`` symmetric matrices, linear near zero."""
    norm = mat.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    scale = (tau * torch.tanh(norm / tau)) / norm
    return mat * scale


def _group_strata_cpu(
    b_n: int,
    cond_disc: Optional[torch.Tensor],
) -> List[Tuple[Tuple[int, ...], torch.Tensor]]:
    """Deterministic sorted discrete stratum partition on CPU returning ``(key, idx_cpu)`` pairs."""
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


def cond_fiber_conditional_loss(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    *,
    cond_disc: Optional[torch.Tensor] = None,
    record: Any = None,
    mean_op_weight: float = 0.36,
    cov_op_weight: float = 0.30,
    bank_mean_blend: float = 0.85,
) -> torch.Tensor:
    """Cross-batch paired-ring + sufficient-statistics conditional fiber discrepancy on ``(x | c, t)``."""
    b_n = x_fake.shape[0]
    if b_n <= 1 or y_real.shape[0] != b_n or c_cont.shape[0] != b_n:
        return x_fake[:0].sum()

    x_flat = x_fake.reshape(b_n, -1)
    y_flat = y_real.detach().reshape(b_n, -1)
    orig_dev = x_flat.device
    dim = max(1, x_flat.shape[-1])
    fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))

    bank: Optional[CondFiberBank] = getattr(record, "cond_fiber_bank", None) if record is not None else None
    allow_state_update = (
        record is not None
        and torch.is_grad_enabled()
        and not getattr(record, "_in_protected_probe", False)
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
        # Single GPU->CPU transfer for (c, x, y) to avoid per-tensor synchronization stalls:
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

                    ecov_w = torch.einsum("ab,nbc,cd->nad", w_mat, q["delta_cov_loc"], w_mat)
                    ecov_w = _odd_frob_saturate(0.5 * (ecov_w + ecov_w.transpose(-1, -2)), tau=1.5).float()

                    # Leverage-equalizing context weight (strictly shared c => exact zero bias):
                    omega_g = q["omega"].float()
                    omega_norm = omega_g / omega_g.mean().clamp_min(1e-6)

                    has_op_row[idx_cpu] = True
                    op_w_mat[idx_cpu] = w_mat.float().unsqueeze(0)
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

        # Update bank + FIFO ring strictly AFTER querying all strata (zero same-batch leakage):
        if bank is not None and allow_state_update:
            for key, idx_cpu, z_innov in pending_updates:
                bank.update_stratum(
                    key,
                    c_cpu64[idx_cpu],
                    x_cpu64[idx_cpu],
                    y_cpu64[idx_cpu],
                    z_innov=z_innov,
                )

        # Residual conditional IMQ-CMMD geometry using cross-batch pooled drift m_hat32:
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
        k_c = torch.where(m_valid, torch.exp(-d_c.square() / denom_2s), 0.0)
        deg_c = 1.0 + k_c.sum(dim=1)
        inv_sqrt_deg = torch.rsqrt(deg_c)
        w_imq = (k_c * inv_sqrt_deg.unsqueeze(1) * inv_sqrt_deg.unsqueeze(0)) / float(b_n)
        w_imq_sym = 0.5 * (w_imq + w_imq.T)
        w_imq_dt = w_imq_sym.to(x_flat.dtype)

        has_any_op = bool(has_op_row.any())
        if has_any_op:
            if cov_op_weight > 0.0:
                # Eigensplit on CPU (15x faster than CUDA cuSOLVER batched eigh for small d_x):
                e_vals_cpu, e_vecs_cpu = torch.linalg.eigh(op_ecov_w)
                op_epos_cpu = torch.einsum("nij,nj,nkj->nik", e_vecs_cpu, e_vals_cpu.clamp_min(0.0), e_vecs_cpu)
                op_eneg_cpu = torch.einsum("nij,nj,nkj->nik", e_vecs_cpu, e_vals_cpu.clamp_max(0.0), e_vecs_cpu)
                op_epos_dev = op_epos_cpu.to(device=orig_dev)
                op_eneg_dev = op_eneg_cpu.to(device=orig_dev)
            op_w_mat_dev = op_w_mat.to(device=orig_dev)
            op_dm_w_dev = op_dm_w.to(device=orig_dev)
            op_mx_dev = op_mx.to(device=orig_dev)
            op_row_w_dev = op_row_w.to(device=orig_dev)

    # 1. Multi-scale residual conditional IMQ-CMMD (higher moments & multimodality):
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

    # 2. Nonparametric paired-ring local conditional mean + split-sign bounded covariance operators:
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

    # 3. Operator-aware additive terminal settling controller:
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

    # Preserve total conditional gradient budget relative to auxiliary losses (e.g. AE reconstruction)
    # once the cross-batch 1st/2nd-moment operator bank is active:
    imq_scale = 0.40 if (has_any_op and (mean_op_weight > 0.0 or cov_op_weight > 0.0)) else 1.0
    return imq_scale * loss_imq + loss_op


class CondFiberGANLoss(UnifiedGANLoss):
    """UnifiedGANLoss extended with cross-batch paired-ring + sufficient-statistics conditional fiber matching."""

    def __init__(
        self,
        *args: Any,
        mean_op_weight: float = 0.36,
        cov_op_weight: float = 0.30,
        bank_mean_blend: float = 0.85,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        for name, val in (
            ("mean_op_weight", mean_op_weight),
            ("cov_op_weight", cov_op_weight),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            isinstance(bank_mean_blend, bool)
            or not isinstance(bank_mean_blend, (int, float))
            or not math.isfinite(bank_mean_blend)
            or bank_mean_blend < 0.0
            or bank_mean_blend > 1.0
        ):
            raise ValueError("bank_mean_blend must be finite and in [0, 1]")
        self.mean_op_weight = float(mean_op_weight)
        self.cov_op_weight = float(cov_op_weight)
        self.bank_mean_blend = float(bank_mean_blend)

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        if real_logits is None or self.t1c.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)

        x_f, x_r, cond_f, cond_r, rec, has_cont_cond, parts_f, parts_r, captured = self._resolve_pair_parts(
            fake_logits, real_logits
        )
        if captured and self.conditional_fibre == "slack" and has_cont_cond:
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
                            cond_ot = cond_fiber_conditional_loss(
                                x_sf,
                                x_sr,
                                c_cont,
                                cond_disc=cond_f,
                                record=rec,
                                mean_op_weight=self.mean_op_weight,
                                cov_op_weight=self.cov_op_weight,
                                bank_mean_blend=self.bank_mean_blend,
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
        return super().g_loss(fake_logits, real_logits)


__all__ = [
    "CondFiberGANLoss",
    "cond_fiber_conditional_loss",
]
