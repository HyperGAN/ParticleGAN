"""P7 Adversarial Mathematical & Empirical Audit of Law 1 (Measure & Prior Topology).

Reproduces every topological proof, Gaussian isoperimetric bound, constructive
Wasserstein/bridge-mass approximation, ParticleGAN Law-1 invariant audit,
zero-sync pullback fastpath parity/sync benchmark, and non-adversarial
division-of-labor experiment reported in:
  reports/four-laws-audit/p7-law1-topology-audit.md

Run:
  PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python \
      scripts/four_laws_audit/p7_law1_topology.py --device cuda:1
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

import particlegan
from particlegan._empf.prototypes import update_emeans_prototypes
from particlegan._minimal.prior import MinimalPriorRegularizer, _coalesce_isolated
from particlegan._t1_coupling.charts import (
    deconflicted_chart_routing,
    laguerre_hard_commitment,
    update_streaming_charts,
)
from particlegan._t1_prior.pullback import T1PullbackPriorRegularizer
from particlegan._t4_arch.pullback import T4ArchPriorRegularizer
from particlegan._unified.prior import (
    UnifiedPriorRegularizer,
    _compute_pythagorean_smoothed_cells,
)
from particlegan._unified_v4.prior import UnifiedV4PriorRegularizer, unified_v4_pullback_step
from particlegan._unified_v11.prior import (
    UnifiedV11PriorRegularizer,
    _compute_pythagorean_smoothed_cells_fast,
    _sinkhorn_barycenter_fast,
    unified_v11_pullback_step,
)
from particlegan.init import deterministic_orthogonal_
from particlegan.particle_prior import MoGParticlePrior
from particlegan.recipes import get_recipe
from particlegan.training import GANTrainer
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians


# ---------------------------------------------------------------------------
# Exact FP64 Normal CDF / PDF / Quantile & Isoperimetric Bounds
# ---------------------------------------------------------------------------

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def normal_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def normal_ppf(p: float) -> float:
    """High-precision FP64 standard normal quantile via torch.special.ndtri + Newton polish."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must lie strictly in (0, 1), got {p}")
    x = float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64)).item())
    for _ in range(4):
        err = normal_cdf(x) - p
        pdf = normal_pdf(x)
        if pdf <= 0.0:
            break
        x -= err / pdf
    return x


def bimodal_bridge_lower_bound(delta: float, lip: float) -> float:
    """Exact Borell-TIS Gaussian isoperimetric lower bound on equal-mass bridge mass.

    For any L-Lipschitz G: R^{d_z} -> R^d pushing N(0, I_{d_z}) onto two equal-mass
    components separated by Delta > 0:
        p_bridge >= 2 * Phi(Delta / (2 * L)) - 1 = erf(Delta / (2 * sqrt(2) * L)).
    """
    if delta <= 0.0 or lip <= 0.0:
        raise ValueError("delta and lip must be positive")
    return 2.0 * normal_cdf(delta / (2.0 * lip)) - 1.0


def bimodal_lip_lower_bound_for_bridge(delta: float, alpha: float) -> float:
    """Exact sharp lower bound on ||J_G||_inf for equal-mass bimodal bridge mass <= alpha.

    Inverts 2 * Phi(Delta / (2 * L)) - 1 <= alpha:
        L >= Delta / (2 * Phi^{-1}((1 + alpha) / 2)) ~ Delta / (sqrt(2*pi) * alpha).
    """
    if delta <= 0.0 or not (0.0 < alpha < 1.0):
        raise ValueError("delta must be positive and alpha in (0, 1)")
    return delta / (2.0 * normal_ppf(0.5 + 0.5 * alpha))


def bimodal_w1_lower_bound(delta: float, lip: float) -> float:
    """Exact sharp dimension-free lower bound on W_1(G_# N(0, I), P_Y) for equal-mass bimodal P_Y.

    Derived via 1-Lipschitz signed distance h(x) = 0.5*(dist(x, K_1) - dist(x, K_2))
    and Borell quantile concentration:
        W_1 >= Delta * (Phi(d0) - 0.5) - 2 * L * (phi(0) - phi(d0)),  d0 = Delta / (2 * L).
    Attained with exact equality by the 1D ramp map G_{d0}(z_1).
    """
    if delta <= 0.0 or lip <= 0.0:
        raise ValueError("delta and lip must be positive")
    d0 = delta / (2.0 * lip)
    return max(0.0, delta * (normal_cdf(d0) - 0.5) - 2.0 * lip * (normal_pdf(0.0) - normal_pdf(d0)))


def bimodal_w2_lower_bound(delta: float, lip: float) -> float:
    """Exact sharp dimension-free lower bound on W_2(G_# N(0, I), P_Y) for equal-mass bimodal P_Y.

    W_2^2 >= (0.5 * Delta^2 + 2 * L^2) * (Phi(d0) - 0.5) - 2 * Delta * L * phi(0) + Delta * L * phi(d0),
    with d0 = Delta / (2 * L). Attained with exact equality by the 1D ramp map G_{d0}(z_1).
    """
    if delta <= 0.0 or lip <= 0.0:
        raise ValueError("delta and lip must be positive")
    d0 = delta / (2.0 * lip)
    w2_sq = (
        (0.5 * delta * delta + 2.0 * lip * lip) * (normal_cdf(d0) - 0.5)
        - 2.0 * delta * lip * normal_pdf(0.0)
        + delta * lip * normal_pdf(d0)
    )
    return math.sqrt(max(0.0, w2_sq))


def multimode_lip_lower_bound_isoperimetric(delta: float, m_modes: int, alpha: float) -> float:
    """Dimension-free M-mode Gaussian isoperimetric lower bound on ||J_G||_inf (all d_z >= 1).

    Since half-radius enlargements A_m^{r/2} (r = Delta / L) are pairwise disjoint,
        alpha >= sum_{m=1}^M (Phi(Phi^{-1}(p_m) + Delta / (2*L)) - p_m).
    For equal mode masses p_m = (1 - alpha) / M, exact inversion gives:
        L >= Delta / (2 * (Phi^{-1}(1/M) - Phi^{-1}((1 - alpha) / M))).
    Reduces identically to bimodal_lip_lower_bound_for_bridge when M = 2.
    """
    if delta <= 0.0 or m_modes < 2 or not (0.0 < alpha < 1.0):
        raise ValueError("invalid delta, m_modes, or alpha")
    p_m = (1.0 - alpha) / float(m_modes)
    q_lo = normal_ppf(p_m)
    q_hi = normal_ppf(1.0 / float(m_modes))
    half_r = q_hi - q_lo
    return delta / (2.0 * half_r)


def multimode_1d_packing_lip_lower_bound(delta: float, m_modes: int, alpha: float) -> float:
    """Exact 1D symmetric interval-packing lower bound on ||J_G||_inf for d_z = 1.

    For M equal-mass modes p_m = (1 - alpha) / M packed symmetrically around 0 with
    M - 1 gaps of width r = Delta / L, finds the exact r* whose total gap Gaussian mass
    equals alpha via bisection, returning L = Delta / r*.
    """
    if delta <= 0.0 or m_modes < 2 or not (0.0 < alpha < 1.0):
        raise ValueError("invalid delta, m_modes, or alpha")
    p_m = (1.0 - alpha) / float(m_modes)

    def _total_gap_mass(r: float) -> float:
        # Place M intervals of Gaussian mass p_m symmetrically around z = 0 with gap r.
        if m_modes % 2 == 0:
            # Central gap [-r/2, +r/2], then M/2 intervals on each side.
            gap_mass = 2.0 * normal_cdf(0.5 * r) - 1.0
            cur_z = 0.5 * r
            n_side = m_modes // 2
            for j in range(n_side):
                target_cdf = normal_cdf(cur_z) + p_m
                if target_cdf >= 1.0 - 1e-15:
                    return 1.0
                end_z = normal_ppf(target_cdf)
                if j < n_side - 1:
                    next_z = end_z + r
                    gap_mass += 2.0 * (normal_cdf(next_z) - normal_cdf(end_z))
                    cur_z = next_z
            return gap_mass
        # Odd M: central interval [-z0, +z0] of mass p_m, then (M-1)/2 gaps + intervals per side.
        cur_z = normal_ppf(0.5 + 0.5 * p_m)
        gap_mass = 0.0
        n_side = (m_modes - 1) // 2
        for _ in range(n_side):
            next_z = cur_z + r
            gap_mass += 2.0 * (normal_cdf(next_z) - normal_cdf(cur_z))
            target_cdf = normal_cdf(next_z) + p_m
            if target_cdf >= 1.0 - 1e-15:
                return 1.0
            cur_z = normal_ppf(target_cdf)
        return gap_mass

    lo, hi = 0.0, 20.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if _total_gap_mass(mid) < alpha:
            lo = mid
        else:
            hi = mid
    r_star = 0.5 * (lo + hi)
    return delta / r_star


# ---------------------------------------------------------------------------
# Part 1: Topological vs Approximation Experiments
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_exp1_topology_and_isoperimetry() -> Dict[str, Any]:
    """Verify exact topological obstruction vs constructive W1/W2/TV/MMD approximation & sharp bounds."""
    delta = 2.0
    # Antithetic Gaussian sample (exact 50/50 sign symmetry eliminates binomial mass-split noise)
    # plus exact midpoint quantile grid to verify theoretical sharpness to high precision.
    rng = torch.Generator(device="cpu").manual_seed(0)
    z_half = torch.randn(200_000, generator=rng, dtype=torch.float64)
    z_mc = torch.cat([z_half, -z_half], dim=0)
    z_sorted, _ = torch.sort(z_mc)

    n_quad = 200_000
    u_quad = (torch.arange(n_quad, dtype=torch.float64) + 0.5) / float(n_quad)
    z_quad = torch.special.ndtri(u_quad)
    y_quad = torch.cat([
        torch.full((n_quad // 2,), -0.5 * delta, dtype=torch.float64),
        torch.full((n_quad // 2,), 0.5 * delta, dtype=torch.float64),
    ])

    # Exact target quantiles for 0.5*delta_{-1} + 0.5*delta_{+1}
    n_mc = len(z_sorted)
    y_sorted = torch.cat([
        torch.full((n_mc // 2,), -0.5 * delta, dtype=torch.float64),
        torch.full((n_mc - n_mc // 2,), 0.5 * delta, dtype=torch.float64),
    ])

    # 1A: Sweep bridge mass alpha for constructive piecewise-linear ramp G_delta vs smooth tanh
    alphas = [0.20, 0.10, 0.05, 0.01, 0.001, 1e-4]
    bimodal_rows = []
    for alpha in alphas:
        lip_bound = bimodal_lip_lower_bound_for_bridge(delta, alpha)
        w1_bound = bimodal_w1_lower_bound(delta, lip_bound)
        w2_bound = bimodal_w2_lower_bound(delta, lip_bound)

        # Constructive continuous ramp map G_d0(z) = clamp(L * z, -Delta/2, +Delta/2)
        gx_ramp = torch.clamp(lip_bound * z_sorted, -0.5 * delta, 0.5 * delta)
        gx_quad = torch.clamp(lip_bound * z_quad, -0.5 * delta, 0.5 * delta)
        emp_bridge_ramp = float((gx_ramp.abs() < (0.5 * delta - 1e-12)).to(torch.float64).mean().item())
        emp_w1_ramp = float((gx_ramp - y_sorted).abs().mean().item())
        emp_w2_ramp = float((gx_ramp - y_sorted).square().mean().sqrt().item())
        quad_w1_ramp = float((gx_quad - y_quad).abs().mean().item())
        quad_w2_ramp = float((gx_quad - y_quad).square().mean().sqrt().item())
        # For pure point masses, TV((G_d0)_# N(0,1), P_Y) == p_bridge == alpha exactly.
        emp_tv_ramp = emp_bridge_ramp

        # Smooth C^infty map G_beta(z) = (Delta/2) * tanh(beta * z) with same Lipschitz L = (Delta/2)*beta
        beta = 2.0 * lip_bound / delta
        gx_tanh = 0.5 * delta * torch.tanh(beta * z_sorted)
        emp_bridge_tanh = float((gx_tanh.abs() < 0.95 * 0.5 * delta).to(torch.float64).mean().item())
        emp_w1_tanh = float((gx_tanh - y_sorted).abs().mean().item())
        emp_w2_tanh = float((gx_tanh - y_sorted).square().mean().sqrt().item())

        # RBF MMD^2 on 4096 subsample for ramp map (quantiles to avoid binomial noise)
        sub_x = gx_quad[:: n_quad // 4096][:4096]
        sub_y = y_quad[:: n_quad // 4096][:4096]
        k_xx = torch.exp(-0.5 * (sub_x.unsqueeze(1) - sub_x.unsqueeze(0)).square()).mean()
        k_yy = torch.exp(-0.5 * (sub_y.unsqueeze(1) - sub_y.unsqueeze(0)).square()).mean()
        k_xy = torch.exp(-0.5 * (sub_x.unsqueeze(1) - sub_y.unsqueeze(0)).square()).mean()
        mmd_ramp = math.sqrt(max(0.0, float((k_xx + k_yy - 2.0 * k_xy).item())))

        bimodal_rows.append({
            "alpha_target": alpha,
            "lip_lower_bound": lip_bound,
            "lip_asymptotic": delta / (math.sqrt(2.0 * math.pi) * alpha),
            "w1_exact_bound": w1_bound,
            "w1_quad_ramp": quad_w1_ramp,
            "w1_emp_ramp": emp_w1_ramp,
            "w2_exact_bound": w2_bound,
            "w2_quad_ramp": quad_w2_ramp,
            "w2_emp_ramp": emp_w2_ramp,
            "bridge_emp_ramp": emp_bridge_ramp,
            "tv_emp_ramp": emp_tv_ramp,
            "mmd_emp_ramp": mmd_ramp,
            "w1_emp_tanh": emp_w1_tanh,
            "w2_emp_tanh": emp_w2_tanh,
            "bridge_95_tanh": emp_bridge_tanh,
        })

    # 1B: Exact Gaussian-mixture quantile map slope explosion (connected support, quantitative cliff!)
    # For P_Y = 0.5 * N(-c, sigma^2) + 0.5 * N(+c, sigma^2), midpoint density p_Y(0) = phi(c/sigma)/sigma,
    # so exact C^infty Brenier map G = F_Y^{-1} o Phi has midpoint derivative G'(0) = phi(0) / p_Y(0)
    # = sigma * exp(0.5 * (c / sigma)^2).
    mog_cliff_rows = []
    for label, c_half, sigma in (
        ("4sigma_sep", 0.5, 0.25),
        ("8sigma_sep", 1.0, 0.25),
        ("12sigma_sep", 1.5, 0.25),
        ("20sigma_sep", 1.0, 0.10),
        ("grid100_adj_33.3sigma", 0.5, 0.03),
    ):
        sep_sigma = (2.0 * c_half) / sigma
        log10_slope = math.log10(sigma) + (0.5 * (c_half / sigma) ** 2) / math.log(10.0)
        slope = sigma * math.exp(0.5 * (c_half / sigma) ** 2) if log10_slope < 300.0 else float("inf")
        mog_cliff_rows.append({
            "case": label,
            "separation_sigma": sep_sigma,
            "midpoint_valley_density_log10": (
                math.log10(_INV_SQRT_2PI / sigma) - (0.5 * (c_half / sigma) ** 2) / math.log(10.0)
            ),
            "exact_brenier_midpoint_slope_log10": log10_slope,
            "exact_brenier_midpoint_slope": slope,
        })

    # 1C: Multi-mode scaling (M = 2, 4, 16, 25, 100) and exact grid100 bounds
    grid100_sigma = 0.03
    grid100_core_r = 3.5 * grid100_sigma  # contains 1 - exp(-3.5^2/2) = 99.78% of each 2D Gaussian
    grid100_delta = 1.0 - 2.0 * grid100_core_r  # = 0.79
    grid100_macro_std = math.sqrt((10.0 ** 2 - 1.0) / 12.0 + grid100_sigma ** 2)  # = 2.872438...
    multimode_rows = []
    for m_modes in (2, 4, 16, 25, 100):
        for alpha in (0.10, 0.05, 0.01, 0.004):
            iso_unit = multimode_lip_lower_bound_isoperimetric(1.0, m_modes, alpha)
            pack1d_unit = multimode_1d_packing_lip_lower_bound(1.0, m_modes, alpha)
            multimode_rows.append({
                "m_modes": m_modes,
                "alpha": alpha,
                "lip_iso_per_delta": iso_unit,
                "lip_1d_pack_per_delta": pack1d_unit,
                "grid100_lip_iso": iso_unit * grid100_delta if m_modes == 100 else None,
                "grid100_lip_1d_pack": pack1d_unit * grid100_delta if m_modes == 100 else None,
            })

    # 1D: Direct W1 / W2 vs L trade-off table (Delta = 2.0)
    lip_tradeoff = []
    for lip in (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0):
        lip_tradeoff.append({
            "lip": lip,
            "bridge_min": bimodal_bridge_lower_bound(delta, lip),
            "w1_min": bimodal_w1_lower_bound(delta, lip),
            "w2_min": bimodal_w2_lower_bound(delta, lip),
        })

    return {
        "bimodal_constructive_vs_bounds": bimodal_rows,
        "mog_connected_support_brenier_cliff": mog_cliff_rows,
        "multimode_scaling": multimode_rows,
        "grid100_geometry": {
            "sigma_mode": grid100_sigma,
            "core_radius_3p5sigma": grid100_core_r,
            "core_mass_fraction": 1.0 - math.exp(-0.5 * 3.5 * 3.5),
            "valley_separation_delta": grid100_delta,
            "poincare_macro_lip_lower_bound": grid100_macro_std,
            "t1_prior_doc_heuristic": 2.25,
            "adaptive_mog_sigma_z_0025_affine_lip": grid100_sigma / 0.025,
        },
        "lip_tradeoff_delta2": lip_tradeoff,
    }


# ---------------------------------------------------------------------------
# Part 2: Code & Invariant Audit of ParticleGAN's Law-1 Machinery
# ---------------------------------------------------------------------------

@torch.no_grad()
def _audit_mass_conservation() -> Dict[str, Any]:
    """Audit Invariant 1: mass conservation of proto_w AND w_fake_ema across all Law-1 implementations."""
    rng = torch.Generator(device="cpu").manual_seed(0)

    # Construct a 4-mode streaming batch sequence + sudden far outlier mode at step 15
    # + extreme imbalance/variance mode to trigger dead-cell split, Gersho merge-split, and teleportation!
    def _sample_phase(step: int, b: int = 128) -> torch.Tensor:
        if step < 15:
            # 3 tight modes + 1 very wide high-variance mode -> triggers Gersho split
            n0, n1, n2 = b // 2, b // 4, b // 8
            n3 = b - n0 - n1 - n2
            p0 = torch.tensor([-5.0, 0.0]) + 1.2 * torch.randn(n0, 2, generator=rng)
            p1 = torch.tensor([5.0, 0.0]) + 0.05 * torch.randn(n1, 2, generator=rng)
            p2 = torch.tensor([0.0, 5.0]) + 0.05 * torch.randn(n2, 2, generator=rng)
            p3 = torch.tensor([0.0, -5.0]) + 0.05 * torch.randn(n3, 2, generator=rng)
            return torch.cat([p0, p1, p2, p3], dim=0)
        # Step >= 15: inject a brand-new far-away mode at (+80, +80) -> triggers uncovered teleportation
        # across ALL quantizer variants (including MinimalPriorRegularizer without Gersho splits)!
        n_far = 8
        p_far = torch.tensor([80.0, 80.0]) + 0.05 * torch.randn(n_far, 2, generator=rng)
        p_old = torch.tensor([-5.0, 0.0]) + 0.10 * torch.randn(b - n_far, 2, generator=rng)
        return torch.cat([p_old, p_far], dim=0)

    batches = [_sample_phase(t) for t in range(30)]
    results: Dict[str, Any] = {}

    # 1) Pullback quantizers: T1, Unified, UnifiedV11, Minimal
    for name, cls in (
        ("t1_prior", T1PullbackPriorRegularizer),
        ("unified", UnifiedPriorRegularizer),
        ("unified_v11", UnifiedV11PriorRegularizer),
        ("minimal", MinimalPriorRegularizer),
    ):
        reg = cls(max_prototypes=64)
        max_w_err = 0.0
        min_w = float("inf")
        max_wfake_err = 0.0
        wfake_sums = []
        for yr in batches:
            reg.update_quantizer(yr)
            w_sum = float(reg.proto_w.sum().item())
            max_w_err = max(max_w_err, abs(w_sum - 1.0))
            min_w = min(min_w, float(reg.proto_w.min().item()))
            wf_sum = float(reg.w_fake_ema.sum().item())
            wfake_sums.append(wf_sum)
            max_wfake_err = max(max_wfake_err, abs(wf_sum - 1.0))
        results[name] = {
            "max_proto_w_sum_err": max_w_err,
            "min_proto_w": min_w,
            "max_w_fake_ema_sum_err": max_wfake_err,
            "final_w_fake_ema_sum": wfake_sums[-1],
            "k_final": int(reg.proto_c.shape[0]),
        }

    # 2) T4ArchPriorRegularizer (dynamic atlas with birth, merge, prune; force cap hit with max_prototypes=4)
    reg_t4 = T4ArchPriorRegularizer(max_prototypes=4)
    max_w_err_t4, min_w_t4, max_wf_err_t4 = 0.0, float("inf"), 0.0
    for yr in batches:
        reg_t4.update_atlas(yr)
        max_w_err_t4 = max(max_w_err_t4, abs(float(reg_t4.proto_w.sum().item()) - 1.0))
        min_w_t4 = min(min_w_t4, float(reg_t4.proto_w.min().item()))
        max_wf_err_t4 = max(max_wf_err_t4, abs(float(reg_t4.w_fake.sum().item()) - 1.0))
    results["t4_arch_cap4"] = {
        "max_proto_w_sum_err": max_w_err_t4,
        "min_proto_w": min_w_t4,
        "max_w_fake_sum_err": max_wf_err_t4,
        "final_w_fake_sum": float(reg_t4.w_fake.sum().item()),
        "k_final": int(reg_t4.proto_c.shape[0]),
    }

    # 3) _t1_coupling streaming charts + chart_fake_mass when fakes lie in valley
    csum = ccnt = csq = cfm = None
    phase, revs = 0.0, 0
    max_w_err_c, min_w_c = 0.0, float("inf")
    for yr in batches:
        c_pts, w_real, radii, b_cnt, csum, ccnt, csq, cfm, phase, revs = update_streaming_charts(
            yr, csum, ccnt, csq, cfm, phase, revs, max_charts=64
        )
        max_w_err_c = max(max_w_err_c, abs(float(w_real.sum().item()) - 1.0))
        min_w_c = min(min_w_c, float(w_real.min().item()))
    # Test deconflicted_chart_routing with valley fakes at origin (outside all basins)
    valley_fakes = torch.zeros((64, 2), dtype=torch.float32)
    _, _, _, _, next_fm_valley = deconflicted_chart_routing(
        valley_fakes, c_pts, w_real, radii, b_cnt, cfm, torch.tensor(0.5),
        w_valley=1.0, allow_update=True,
    )
    results["t1_coupling_charts"] = {
        "max_w_real_sum_err": max_w_err_c,
        "min_w_real": min_w_c,
        "fake_mass_sum_before_valley": float(cfm.sum().item()),
        "fake_mass_sum_after_valley": float(next_fm_valley.sum().item()),
        "k_final": int(c_pts.shape[0]),
        "revivals": int(revs),
    }

    # 4) _empf online e-means prototypes
    psum = pcnt = None
    ephase, erevs = 0.0, 0
    max_w_err_e, min_w_e = 0.0, float("inf")
    for yr in batches:
        _, ew, psum, pcnt, ephase, erevs = update_emeans_prototypes(
            yr, psum, pcnt, ephase, erevs, max_prototypes=64
        )
        max_w_err_e = max(max_w_err_e, abs(float(ew.sum().item()) - 1.0))
        min_w_e = min(min_w_e, float(ew.min().item()))
    results["empf_emeans"] = {
        "max_proto_w_sum_err": max_w_err_e,
        "min_proto_w": min_w_e,
        "revivals": int(erevs),
    }
    return results


@torch.no_grad()
def _audit_pythagorean_ltv() -> Dict[str, Any]:
    """Audit Invariant 2: Pythagorean LTV coalescence exactness, anisotropy loss, and code bugs."""
    # 2A: Exact algebraic verification on 2-, 3-, and 4-cell isolated sibling clusters in 2D and 4D
    # Place 2 isolated modes far apart so external isolation (>= 4.2*sqrt(D)*sigma) holds strictly.
    # Mode 1: 3 sub-cells around (0, 0) with unequal weights & variances
    c1 = torch.tensor([[0.0, 0.0], [0.10, 0.0], [0.0, 0.12]], dtype=torch.float32)
    v1 = torch.tensor([0.04, 0.05, 0.045], dtype=torch.float32)
    w1 = torch.tensor([0.20, 0.15, 0.15], dtype=torch.float32)
    # Mode 2: 2 sub-cells around (20, 0)
    c2 = torch.tensor([[20.0, 0.0], [20.14, 0.0]], dtype=torch.float32)
    v2 = torch.tensor([0.03, 0.035], dtype=torch.float32)
    w2 = torch.tensor([0.30, 0.20], dtype=torch.float32)

    pc = torch.cat([c1, c2], dim=0)
    pv = torch.cat([v1, v2], dim=0)
    pw = torch.cat([w1, w2], dim=0)

    c_bar_ref, sig_bar_ref, w_mat_ref, _ = _compute_pythagorean_smoothed_cells(pc, pv, pw)
    c_bar_fast, sig_bar_fast, w_mat_fast, _ = _compute_pythagorean_smoothed_cells_fast(pc, pv, pw)

    # Compute exact FP64 analytical centroid & trace LTV for Mode 1 and Mode 2
    def _exact_ltv(c_sub: torch.Tensor, v_sub: torch.Tensor, w_sub: torch.Tensor) -> Tuple[torch.Tensor, float]:
        wn = w_sub.double() / w_sub.double().sum()
        cbar = (wn.unsqueeze(1) * c_sub.double()).sum(dim=0)
        between = (wn * (c_sub.double() - cbar.unsqueeze(0)).square().sum(dim=-1)).sum() / float(c_sub.shape[1])
        within = (wn * v_sub.double()).sum()
        return cbar, float((within + between).item())

    c1_exact, v1_exact = _exact_ltv(c1, v1, w1)
    c2_exact, v2_exact = _exact_ltv(c2, v2, w2)
    ltv_m1_err = abs(float(sig_bar_fast[0].item()) ** 2 - v1_exact)
    ltv_m2_err = abs(float(sig_bar_fast[3].item()) ** 2 - v2_exact)
    c1_err = float((c_bar_fast[0].double() - c1_exact).norm().item())
    c2_err = float((c_bar_fast[3].double() - c2_exact).norm().item())
    fast_vs_ref_max = max(
        float((c_bar_fast - c_bar_ref).abs().max().item()),
        float((sig_bar_fast - sig_bar_ref).abs().max().item()),
        float((w_mat_fast - w_mat_ref).abs().max().item()),
    )

    # 2B: Anisotropic covariance destruction when an isolated anisotropic mode (|C|=2) is coalesced
    # Two half-Gaussians of N(0, diag(sigma1^2, sigma2^2)) split along x_1:
    # Conditional means (+-sqrt(2/pi)*sigma1, 0), within-cell cov diag((1 - 2/pi)*sigma1^2, sigma2^2).
    aniso_rows = []
    for kappa in (1.0, 2.0, 3.0, 5.0, 10.0):
        s2 = 0.05
        s1 = kappa * s2
        true_cov = torch.diag(torch.tensor([s1 * s1, s2 * s2], dtype=torch.float64))
        # Exact sub-cell means and trace variances of the two half-Gaussians:
        mu_half = math.sqrt(2.0 / math.pi) * s1
        v_within_trace = 0.5 * ((1.0 - 2.0 / math.pi) * s1 * s1 + s2 * s2)
        c_iso = torch.tensor([[-mu_half, 0.0], [mu_half, 0.0], [50.0, 0.0]], dtype=torch.float32)
        v_iso = torch.tensor([v_within_trace, v_within_trace, v_within_trace], dtype=torch.float32)
        w_iso = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
        _, sbar_a, _, coal_a = _compute_pythagorean_smoothed_cells_fast(c_iso, v_iso, w_iso)
        v_coal = float(sbar_a[0].item()) ** 2
        true_trace_per_d = 0.5 * (s1 * s1 + s2 * s2)
        iso_cov = v_coal * torch.eye(2, dtype=torch.float64)
        rel_frob_err = float((iso_cov - true_cov).norm().item() / true_cov.norm().item())
        analytic_rel_err = (kappa * kappa - 1.0) / math.sqrt(2.0 * (kappa ** 4 + 1.0))
        aniso_rows.append({
            "aspect_ratio_kappa": kappa,
            "coalesced": bool(coal_a[0].item()),
            "true_trace_per_d": true_trace_per_d,
            "coalesced_v_bar": v_coal,
            "trace_err": abs(v_coal - true_trace_per_d),
            "cov_rel_frob_err": rel_frob_err,
            "analytic_iso_frob_err": analytic_rel_err,
        })

    # 2C: Bug 2A (Gersho merge-before-split in UnifiedV11PriorRegularizer omits proto_var LTV update!)
    # Construct k=16 prototypes with no dead cells, one high-distortion cell pi=0 (E_0 > 10 * med_E),
    # and one close merge pair (a=1, b=2) with distinct variances & non-zero separation.
    reg_m = UnifiedV11PriorRegularizer(max_prototypes=16)
    k_m = 16
    centers_m = torch.zeros((k_m, 2), dtype=torch.float32)
    for i in range(k_m):
        centers_m[i, 0] = float(i) * 10.0
    # Make (1, 2) very close (within 8*D*med_v) with distinct positions & different variances
    centers_m[1] = torch.tensor([10.0, 0.0])
    centers_m[2] = torch.tensor([11.0, 0.0])  # d^2 = 1.0 <= 8 * 2 * 0.25 = 4.0
    reg_m.quantizer_initialized = True
    reg_m.proto_c = centers_m.clone()
    reg_m.proto_var = torch.full((k_m,), 0.25, dtype=torch.float32)
    reg_m.proto_var[0] = 20.0  # huge distortion -> triggers need_gersho_split!
    reg_m.proto_var[1] = 0.10
    reg_m.proto_var[2] = 0.40
    reg_m.proto_w = torch.full((k_m,), 1.0 / float(k_m), dtype=torch.float32)
    reg_m.proto_hits = torch.full((k_m,), 50.0, dtype=torch.float32)
    reg_m.w_fake_ema = reg_m.proto_w.clone()
    reg_m.psi = torch.zeros(k_m, dtype=torch.float32)
    reg_m.v_push = reg_m.proto_var.unsqueeze(1).expand_as(reg_m.proto_c).clone()

    # Pass a batch matching the centers so hit updates don't erase the setup, and check merge of (1, 2)!
    # Even cleaner: place batch points at exact centers (each hit once -> resid_sq_prev = 0 -> v_est clamped,
    # wait: proto_var decays by 0.88 if hit!). To inspect exact pre-merge vs post-merge LTV cleanly,
    # put all batch points on centers 3..15 (so cells 0, 1, 2 have b_cnt == 0 and are untouched before merge!).
    yr_m = centers_m[3:].repeat(4, 1)  # 52 points on cells 3..15 (cells 0, 1, 2 untouched by EMA!)
    # Ensure cell 0, 1, 2 weights remain well above 0.04/k (alpha_w = 0.02 -> 0.98/16 = 0.06125 >> 0.0025)
    v1_before = float(reg_m.proto_var[1].item())
    v2_before = float(reg_m.proto_var[2].item())
    c1_before = reg_m.proto_c[1].clone()
    c2_before = reg_m.proto_c[2].clone()
    w1_before = float(reg_m.proto_w[1].item())
    w2_before = float(reg_m.proto_w[2].item())
    # Exact expected LTV variance of merged cell 1:
    wt = w1_before + w2_before
    expected_v_merged = (
        (w1_before * v1_before + w2_before * v2_before) / wt
        + (w1_before * w2_before / (wt * wt)) * float((c1_before - c2_before).square().sum().item()) / 2.0
    )
    v0_before = float(reg_m.proto_var[0].item())
    reg_m.update_quantizer(yr_m)
    v1_after_merge = float(reg_m.proto_var[1].item())
    # Also check split of cell 0 into (0, 2): exact LTV requires v_child + ||delta||^2/D == v0_before
    # Actual code sets v_child = 0.65 * v0_before and ||delta|| = 0.40 * sqrt(D * v0_before) -> between = 0.16 * v0_before
    v0_after_split = float(reg_m.proto_var[0].item())
    v2_after_split = float(reg_m.proto_var[2].item())
    c0_after = reg_m.proto_c[0]
    c2_after = reg_m.proto_c[2]
    split_total_var_after = 0.5 * (v0_after_split + v2_after_split) + 0.25 * float((c0_after - c2_after).square().sum().item()) / 2.0
    split_var_retention = split_total_var_after / v0_before

    # 2D: Bug 2C (proto_var estimator bias on stationary unit Gaussian N(0, I_2), single prototype)
    rng_b = torch.Generator(device="cpu").manual_seed(0)
    reg_t1_b = T1PullbackPriorRegularizer(max_prototypes=2)
    reg_v11_b = UnifiedV11PriorRegularizer(max_prototypes=2)
    # Initialize with 1 prototype (force single cell at origin via small batch b=1 -> k=1)
    for reg_b in (reg_t1_b, reg_v11_b):
        reg_b.quantizer_initialized = True
        reg_b.proto_c = torch.zeros((1, 2), dtype=torch.float32)
        reg_b.proto_var = torch.ones(1, dtype=torch.float32)
        reg_b.proto_w = torch.ones(1, dtype=torch.float32)
        reg_b.proto_hits = torch.ones(1, dtype=torch.float32)
        reg_b.w_fake_ema = torch.ones(1, dtype=torch.float32)
        reg_b.psi = torch.zeros(1, dtype=torch.float32)
    # Small per-cell batch size b=4 (typical of grid100 with B=256, M=100 -> ~2.56 pts/mode!)
    t1_vars, v11_vars = [], []
    for step in range(400):
        yb = torch.randn(4, 2, generator=rng_b)
        reg_t1_b.update_quantizer(yb)
        reg_v11_b.update_quantizer(yb)
        if step >= 100:
            t1_vars.append(float(reg_t1_b.proto_var[0].item()))
            v11_vars.append(float(reg_v11_b.proto_var[0].item()))

    # 2E: Bug 2D (_minimal._coalesce_isolated missing sig_max <= 2.2*sig_min scale-ratio guard!)
    pc_min = torch.tensor([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]], dtype=torch.float32)
    pv_min = torch.tensor([0.01, 1.0, 0.01], dtype=torch.float32)  # sigma = 0.1 vs 1.0 (10x width ratio!)
    pw_min = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    _, _, _, coal_v11_scale = _compute_pythagorean_smoothed_cells_fast(pc_min, pv_min, pw_min)
    _, _, _, coal_min_scale = _coalesce_isolated(pc_min, pv_min, pw_min)

    return {
        "exact_scalar_ltv_3cell_err": ltv_m1_err,
        "exact_scalar_ltv_2cell_err": ltv_m2_err,
        "centroid_err_max": max(c1_err, c2_err),
        "fast_vs_ref_max_diff": fast_vs_ref_max,
        "anisotropic_destruction": aniso_rows,
        "gersho_merge_ltv_bug": {
            "v1_before": v1_before,
            "v2_before": v2_before,
            "expected_ltv_v_merged": expected_v_merged,
            "actual_v1_after_merge": v1_after_merge,
            "relative_ltv_deficit": (expected_v_merged - v1_after_merge) / expected_v_merged,
        },
        "split_ltv_bug": {
            "v0_before_split": v0_before,
            "split_total_var_after": split_total_var_after,
            "variance_retention_ratio": split_var_retention,
            "variance_loss_fraction": 1.0 - split_var_retention,
        },
        "proto_var_estimator_bias_n4": {
            "true_var": 1.0,
            "t1_prior_mean_var": float(np.mean(t1_vars)),
            "unified_v11_mean_var": float(np.mean(v11_vars)),
            "t1_prior_rel_bias": float(np.mean(t1_vars)) - 1.0,
            "unified_v11_rel_bias": float(np.mean(v11_vars)) - 1.0,
        },
        "minimal_cross_scale_false_merge": {
            "unified_v11_coalesced_10x_pair": bool(coal_v11_scale[0].item()),
            "minimal_coalesced_10x_pair": bool(coal_min_scale[0].item()),
        },
    }


@torch.no_grad()
def _run_moreau_yosida_laguerre_sim(
    x_particles: torch.Tensor,
    proto_c: torch.Tensor,
    proto_w: torch.Tensor,
    *,
    steps: int = 300,
    batch_size: Optional[int] = None,
    laguerre_lr: float = 0.015,
    w_mat: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Exact reproduction of unified_v11 Part-B Moreau-Yosida damped Laguerre dual iteration."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    n_tot, dim = x_particles.shape
    k = proto_c.shape[0]
    if w_mat is None:
        w_mat = torch.eye(k, dtype=torch.float32)
    c_bar = w_mat @ proto_c
    psi = torch.zeros(k, dtype=torch.float32)
    w_fake_ema = torch.full((k,), 1.0 / float(k), dtype=torch.float32)
    r_macro_safe = x_particles.std(dim=0, unbiased=False).mean().clamp_min(1e-4)
    r_mac_sq = float(r_macro_safe.square().item())
    eta_scale = min(3.0, max(0.35, float(laguerre_lr) * float(k)))
    psi_cap = 2.5 * float(dim) * r_mac_sq + 0.5

    empty_counts = []
    psi_means = []
    clamp_hits = 0
    for _ in range(steps):
        if batch_size is None or batch_size >= n_tot:
            xu = x_particles
        else:
            idx = torch.randint(0, n_tot, (batch_size,), generator=rng)
            xu = x_particles[idx]
        n_eff = float(xu.shape[0])
        psi_eff = w_mat @ psi
        d2 = torch.cdist(xu, c_bar, p=2.0).square()
        k_star = (d2 - psi_eff.unsqueeze(0)).argmin(dim=1)
        counts = torch.bincount(k_star, minlength=k)
        empty_counts.append(int((counts == 0).sum().item()))

        w_hard_u = w_mat[k_star].mean(dim=0)
        w_fake_ema = 0.88 * w_fake_ema + 0.12 * w_hard_u
        w_feedback = 0.35 * w_hard_u + 0.65 * w_fake_ema

        tv_now = 0.5 * float((proto_w - w_fake_ema).abs().sum().item())
        tv_floor = 0.35 * math.sqrt(float(k) / max(n_eff * 8.0, 1.0))
        psi_gate = max(0.0, min(1.0, (tv_now - tv_floor) / max(0.08, tv_floor)))
        eta_psi = psi_gate * eta_scale * float(dim) * r_mac_sq

        psi_next = psi * (1.0 - 0.008 * psi_gate)
        psi_next = psi_next + eta_psi * (proto_w - w_feedback)
        psi_next = w_mat @ psi_next
        psi_next = psi_next - psi_next.mean()
        if bool((psi_next.abs() > psi_cap).any()):
            clamp_hits += 1
        psi = psi_next.clamp(-psi_cap, psi_cap)
        psi_means.append(abs(float(psi.mean().item())))

    # Evaluate final full-cloud hard assignment mass vs target proto_w
    d2_full = torch.cdist(x_particles, c_bar, p=2.0).square()
    k_full = (d2_full - (w_mat @ psi).unsqueeze(0)).argmin(dim=1)
    w_final = w_mat[k_full].mean(dim=0)
    tv_final = 0.5 * float((proto_w - w_final).abs().sum().item())
    max_quota_err = float((proto_w - w_final).abs().max().item())
    n_eff_eval = float(batch_size if (batch_size is not None and batch_size < n_tot) else n_tot)
    tv_floor_val = 0.35 * math.sqrt(float(k) / max(n_eff_eval * 8.0, 1.0))
    return {
        "tv_final": tv_final,
        "tv_floor_deadband": tv_floor_val,
        "max_quota_err": max_quota_err,
        "final_weights": [round(float(v), 5) for v in w_final.tolist()[:8]],
        "target_weights": [round(float(v), 5) for v in proto_w.tolist()[:8]],
        "mean_empty_cells_per_step": float(np.mean(empty_counts)),
        "final_empty_cells_full": int((w_final <= 1e-12).sum().item()),
        "max_abs_psi_mean": float(max(psi_means)),
        "clamp_hits": int(clamp_hits),
    }


@torch.no_grad()
def _audit_laguerre_assignment() -> Dict[str, Any]:
    """Audit Invariant 3: Moreau-Yosida damped Laguerre dual AND _t1_coupling greedy Monge assignment."""
    rng = torch.Generator(device="cpu").manual_seed(0)

    # --- 3A: Moreau-Yosida damped Laguerre dual edge cases ---
    x_cloud = torch.randn(2048, 2, generator=rng) * 2.0

    # Case 1: Unequal weights [0.55, 0.30, 0.13, 0.02] (vector_unequal_mass)
    c_un = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    w_un = torch.tensor([0.55, 0.30, 0.13, 0.02], dtype=torch.float32)
    res_unequal = _run_moreau_yosida_laguerre_sim(x_cloud, c_un, w_un, steps=400, batch_size=256)

    # Case 2: Rare 2% mode completely starved at init (initial TV = 0.02 < tv_floor = 0.0155? Check small batch B=64 -> tv_floor=0.031!)
    x_starved = torch.cat([
        c_un[0] + 0.1 * torch.randn(1126, 2, generator=rng),
        c_un[1] + 0.1 * torch.randn(614, 2, generator=rng),
        c_un[2] + 0.1 * torch.randn(308, 2, generator=rng),  # 0 particles on mode 3 (w=0.02)!
    ], dim=0)
    res_starved_b64 = _run_moreau_yosida_laguerre_sim(x_starved, c_un, w_un, steps=300, batch_size=64)

    # Case 3: Far outlier mode (empty cell + psi_cap saturation + broken zero-mean gauge!)
    c_out = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [30.0, 0.0]], dtype=torch.float32)
    w_out = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    res_outlier = _run_moreau_yosida_laguerre_sim(x_cloud, c_out, w_out, steps=400, batch_size=256)

    # Case 4: B < M regime (grid100: M=100, B=64)
    gx, gy = torch.meshgrid(torch.linspace(-4.5, 4.5, 10), torch.linspace(-4.5, 4.5, 10), indexing="ij")
    c_grid100 = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    w_grid100 = torch.full((100,), 0.01, dtype=torch.float32)
    x_grid = torch.randn(2048, 2, generator=rng) * 3.0
    res_b_lt_m = _run_moreau_yosida_laguerre_sim(x_grid, c_grid100, w_grid100, steps=400, batch_size=64)

    # Case 5: Collinear & Coincident prototypes
    c_col = torch.tensor([[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    w_col = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    res_collinear = _run_moreau_yosida_laguerre_sim(x_cloud, c_col, w_col, steps=400, batch_size=256)

    c_coinc = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-3.0, 0.0]], dtype=torch.float32)
    w_coinc = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    res_coincident = _run_moreau_yosida_laguerre_sim(x_cloud, c_coinc, w_coinc, steps=400, batch_size=256)

    # Case 6: Verify _minimal/prior.py:190 pre-coalesced mean-subtraction bug!
    # Suppose 3 prototypes: {0, 1} coalesced with unequal weights (0.8, 0.2) and {2} singleton.
    w_mat_bug = torch.tensor([
        [0.8, 0.2, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    psi_in = torch.tensor([2.0, -2.0, 0.0], dtype=torch.float32)  # mean == 0.0
    # _minimal/prior.py:190 executes: (w_mat @ psi).sub(psi.mean()) -> mean != 0!
    psi_minimal_out = (w_mat_bug @ psi_in).sub(psi_in.mean())
    minimal_psi_mean_err = abs(float(psi_minimal_out.mean().item()))

    # --- 3B: _t1_coupling/charts.py laguerre_hard_commitment ---
    # Test 1: Exact integer quota satisfaction vs exact discrete OT optimality gap (Hungarian)!
    def _exact_ot_cost_for_quotas(x_pts: torch.Tensor, c_pts: torch.Tensor, w_pts: torch.Tensor, ell: float) -> Tuple[float, float, bool]:
        b_f, k = x_pts.shape[0], c_pts.shape[0]
        _, _, win_idx = laguerre_hard_commitment(
            x_pts, c_pts, w_pts, torch.ones(k), torch.tensor(ell), eps_chart=0.08, iters=8
        )
        # Reconstruct exact quotas used by laguerre_hard_commitment
        w_n = w_pts.float().clamp_min(1e-12)
        w_n = w_n / w_n.sum()
        raw_q = w_n * float(b_f)
        quotas = torch.floor(raw_q).to(torch.long)
        rem = int(b_f - int(quotas.sum().item()))
        if rem > 0:
            frac = raw_q - quotas.float() - 1e-7 * torch.arange(k, dtype=torch.float32)
            quotas[torch.topk(frac, k=min(rem, k), largest=True).indices] += 1
        actual_counts = torch.bincount(win_idx, minlength=k)
        exact_quota_ok = bool(torch.equal(actual_counts, quotas))

        # Exact Born-Infeld cost matrix used inside laguerre_hard_commitment
        scale = max(float(x_pts.abs().amax().item()), float(c_pts.abs().amax().item()), ell, 1.0)
        d2_u = torch.cdist(x_pts / scale, c_pts / scale).square()
        ell_u = max(ell / scale, 1e-6)
        bi_u = torch.sqrt(ell_u * ell_u + d2_u) - ell_u
        med_bi = float(bi_u.median().clamp_min(1e-8).item())
        cost_mat = (bi_u / med_bi).clamp_max(200.0).double().numpy()

        greedy_cost = float(sum(cost_mat[i, int(win_idx[i].item())] for i in range(b_f)))
        # Replicate each chart column q_k times and solve exact Hungarian assignment!
        cols = []
        for j in range(k):
            for _ in range(int(quotas[j].item())):
                cols.append(j)
        expanded_cost = cost_mat[:, cols]
        r_ind, c_ind = linear_sum_assignment(expanded_cost)
        opt_cost = float(expanded_cost[r_ind, c_ind].sum())
        return greedy_cost, opt_cost, exact_quota_ok

    # Explicit minimal counterexample (B=4, K=3) where 8-iter Sinkhorn + greedy margin is strictly suboptimal
    x_ce = torch.tensor([[0.0, 0.0], [0.8, 0.0], [1.6, 0.0], [2.4, 0.0]], dtype=torch.float32)
    c_ce = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    w_ce = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    # Also sweep 40 random trials of (B=32, K=8) with unequal weights to quantify average & worst suboptimality
    subopt_gaps = []
    all_quotas_exact = True
    rng_ot = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(40):
        x_t = torch.randn(32, 2, generator=rng_ot) * 2.5
        c_t = torch.randn(8, 2, generator=rng_ot) * 2.5
        w_t = torch.tensor([0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03], dtype=torch.float32)
        g_cost, o_cost, q_ok = _exact_ot_cost_for_quotas(x_t, c_t, w_t, ell=0.25)
        all_quotas_exact = all_quotas_exact and q_ok
        subopt_gaps.append((g_cost - o_cost) / max(o_cost, 1e-12))

    # Explicit 2D counterexample (B=6, K=3) where greedy margin-ordered assignment is strictly suboptimal
    x_adv = torch.tensor(
        [[0.0, 0.0], [0.5, 0.1], [0.9, -0.1], [1.2, 0.4], [2.2, 0.0], [2.5, 0.2]],
        dtype=torch.float32,
    )
    c_adv = torch.tensor([[-0.2, 0.0], [1.0, 0.0], [2.6, 0.0]], dtype=torch.float32)
    w_adv = torch.tensor([0.50, 0.34, 0.16], dtype=torch.float32)
    g_adv, o_adv, q_adv = _exact_ot_cost_for_quotas(x_adv, c_adv, w_adv, ell=0.1)
    g_ce, o_ce, q_ce = _exact_ot_cost_for_quotas(x_ce, c_ce, w_ce, ell=0.1)
    all_quotas_exact = all_quotas_exact and q_adv and q_ce

    # Test 2: Severe B_f < K index-bias bug in laguerre_hard_commitment!
    # Place 64 particles DIRECTLY ON charts 36..99 of grid100 (zero distance to charts 36..99!).
    # Under uniform weights w_k = 1/100 and B_f = 64 < K = 100, charts 64..99 get quota == 0!
    x_on_high = c_grid100[36:100].clone()  # 64 particles sitting directly on charts 36..99!
    _, _, win_high = laguerre_hard_commitment(
        x_on_high, c_grid100, w_grid100, torch.full((100,), 0.03), torch.tensor(0.25), eps_chart=0.08, iters=8
    )
    assigned_to_ge64 = int((win_high >= 64).sum().item())
    evicted_zero_dist_particles = int((win_high[28:] != torch.arange(64, 100)).sum().item())
    eviction_mean_dist = float((x_on_high[28:] - c_grid100[win_high[28:]]).norm(dim=-1).mean().item())

    return {
        "moreau_yosida_dual": {
            "unequal_mass_b256": res_unequal,
            "starved_2pct_mode_b64": res_starved_b64,
            "far_outlier_cap_saturation": res_outlier,
            "b_lt_m_grid100_b64": res_b_lt_m,
            "collinear_4modes": res_collinear,
            "coincident_prototypes": res_coincident,
            "minimal_pre_coalesced_mean_bug": minimal_psi_mean_err,
        },
        "t1_coupling_laguerre_hard_commitment": {
            "all_40_random_trials_exact_integer_quotas": all_quotas_exact,
            "random_40_trials_mean_rel_ot_gap": float(np.mean(subopt_gaps)),
            "random_40_trials_max_rel_ot_gap": float(np.max(subopt_gaps)),
            "random_40_trials_frac_strictly_suboptimal": float(np.mean([g > 1e-6 for g in subopt_gaps])),
            "line_4x3_gap": (g_ce - o_ce) / max(o_ce, 1e-12),
            "adv_6x3_gap": (g_adv - o_adv) / max(o_adv, 1e-12),
            "b_lt_m_index_bias_bug": {
                "b_f": 64,
                "k_charts": 100,
                "particles_placed_on_charts_64_to_99": 36,
                "particles_assigned_to_charts_ge_64": assigned_to_ge64,
                "zero_dist_particles_evicted": evicted_zero_dist_particles,
                "eviction_mean_euclidean_dist": eviction_mean_dist,
            },
        },
    }


@torch.no_grad()
def _audit_pullback_isometry_and_fastpath(device: torch.device) -> Dict[str, Any]:
    """Audit Invariant 4: orthogonal pullback covariance isometry accuracy + v12 fastpath parity & syncs."""
    rng = torch.Generator(device="cpu").manual_seed(0)

    # --- 4A: Linear G(z) = A z accuracy across 2D normalized, 4D Radon-Hurwitz, and minimal 2-probe ---
    a2_iso = torch.tensor([[1.2, 0.0], [0.0, 1.2]], dtype=torch.float64)
    a2_aniso = torch.tensor([[2.5, 0.8], [-0.4, 0.3]], dtype=torch.float64)  # kappa ~ 5+
    a4_aniso = torch.tensor([[2.2, -0.7, 0.5, 0.1], [0.1, 0.3, -0.2, 0.4]], dtype=torch.float64)
    sigma_z = 0.025

    eps2 = torch.randn(512, 2, generator=rng, dtype=torch.float64)
    e1_u = eps2 / eps2.norm(dim=-1, keepdim=True)
    e2_u = torch.stack((-e1_u[:, 1], e1_u[:, 0]), dim=1)

    def _probe_2d_linear(a_mat: torch.Tensor) -> float:
        true_cov = (sigma_z ** 2) * (a_mat @ a_mat.T)
        dx1 = (sigma_z * e1_u) @ a_mat.T
        dx2 = (sigma_z * e2_u) @ a_mat.T
        m_dx = torch.stack((dx1, dx2), dim=-1)
        sig_push = m_dx @ m_dx.transpose(1, 2)
        return float((sig_push - true_cov.unsqueeze(0)).norm(dim=(1, 2)).max().item() / true_cov.norm().item())

    err_2d_iso = _probe_2d_linear(a2_iso)
    err_2d_aniso = _probe_2d_linear(a2_aniso)

    # 4D Radon-Hurwitz exact vs clamp_min(0.15) bias when ||eps||^2 < 0.15
    eps4 = torch.randn(2000, 4, generator=rng, dtype=torch.float64)
    # Also test an explicit small-norm draw ||eps||^2 = 0.05 < 0.15
    eps4_small = torch.tensor([[0.1, 0.1, 0.1, math.sqrt(0.02)]], dtype=torch.float64)
    true_cov4 = (sigma_z ** 2) * (a4_aniso @ a4_aniso.T)

    def _probe_4d_rh(es: torch.Tensor, clamp_min_val: float) -> Tuple[float, float]:
        a_c, b_c, c_c, d_c = es[:, 0], es[:, 1], es[:, 2], es[:, 3]
        e2 = torch.stack((-b_c, a_c, -d_c, c_c), dim=1)
        e3 = torch.stack((-c_c, d_c, a_c, -b_c), dim=1)
        e4 = torch.stack((-d_c, -c_c, b_c, a_c), dim=1)
        dxs = [(sigma_z * ek) @ a4_aniso.T for ek in (es, e2, e3, e4)]
        m_dx = torch.stack(dxs, dim=-1)
        eps_sq = es.square().sum(dim=-1).clamp_min(clamp_min_val).view(-1, 1, 1)
        sig_push = (m_dx @ m_dx.transpose(1, 2)) / eps_sq
        rel = (sig_push - true_cov4.unsqueeze(0)).norm(dim=(1, 2)) / true_cov4.norm()
        return float(rel.mean().item()), float(rel.max().item())

    rh_mean_unclamped, rh_max_unclamped = _probe_4d_rh(eps4, 1e-12)
    rh_mean_clamped, rh_max_clamped = _probe_4d_rh(eps4, 0.15)
    rh_small_clamped_mean, _ = _probe_4d_rh(eps4_small, 0.15)

    # Minimal 2-probe in d_z=4 (_minimal/prior.py:162-165)
    ep_min = torch.stack((-eps4[:, 1::2], eps4[:, 0::2]), dim=-1).reshape(-1, 4)
    dx0_m = (sigma_z * eps4) @ a4_aniso.T
    dxp_m = (sigma_z * ep_min) @ a4_aniso.T
    m2_m = torch.stack((dx0_m, dxp_m), dim=-1)
    sp2_m = (m2_m @ m2_m.transpose(1, 2)) / (eps4.square().sum(dim=-1) * 0.5).clamp_min(0.15).view(-1, 1, 1)
    min_rel_err_per_sample = float(((sp2_m - true_cov4.unsqueeze(0)).norm(dim=(1, 2)) / true_cov4.norm()).mean().item())
    min_rel_err_mean_cov = float((sp2_m.mean(dim=0) - true_cov4).norm().item() / true_cov4.norm().item())

    # --- 4B: Nonlinear curvature bias (O(sigma_z) forward probe vs O(sigma_z^2) central probe) ---
    # G(z) = A z + 0.5 * c_quad * (z_1^2 + z_2^2) * b_vec + (1/6) * cubic
    b_quad = torch.tensor([1.5, -1.0], dtype=torch.float64)
    mu0 = torch.tensor([[0.3, -0.2]], dtype=torch.float64)

    def _g_nonlin(z: torch.Tensor) -> torch.Tensor:
        lin = z @ a2_aniso.T
        q = 0.5 * z.square().sum(dim=-1, keepdim=True) * b_quad.unsqueeze(0)
        cub = 0.25 * torch.tanh(1.5 * z)
        return lin + q + cub

    # Exact Jacobian of _g_nonlin at mu0: J = A + b_quad @ mu0 + 0.25 * 1.5 * diag(sech^2(1.5*mu0))
    sech2 = 1.0 / torch.cosh(1.5 * mu0[0]).square()
    j_exact = a2_aniso + torch.outer(b_quad, mu0[0]) + torch.diag(0.375 * sech2)
    e1_fix = torch.tensor([[0.8, 0.6]], dtype=torch.float64)
    e2_fix = torch.tensor([[-0.6, 0.8]], dtype=torch.float64)
    nonlin_rows = []
    for sz in (1e-3, 3e-3, 1e-2, 0.025, 0.05, 0.10, 0.25):
        true_cov_nl = (sz ** 2) * (j_exact @ j_exact.T)
        g0 = _g_nonlin(mu0)
        # Forward one-sided probe (as implemented in _unified_v11/prior.py:1487-1494)
        dx1_f = _g_nonlin(mu0 + sz * e1_fix) - g0
        dx2_f = _g_nonlin(mu0 + sz * e2_fix) - g0
        mf = torch.stack((dx1_f, dx2_f), dim=-1)
        cov_f = (mf @ mf.transpose(1, 2))[0]
        rel_fwd = float((cov_f - true_cov_nl).norm().item() / true_cov_nl.norm().item())
        # Symmetric central probe (cancels O(sigma_z^3) quadratic cross term!)
        dx1_c = 0.5 * (_g_nonlin(mu0 + sz * e1_fix) - _g_nonlin(mu0 - sz * e1_fix))
        dx2_c = 0.5 * (_g_nonlin(mu0 + sz * e2_fix) - _g_nonlin(mu0 - sz * e2_fix))
        mc = torch.stack((dx1_c, dx2_c), dim=-1)
        cov_c = (mc @ mc.transpose(1, 2))[0]
        rel_ctr = float((cov_c - true_cov_nl).norm().item() / true_cov_nl.norm().item())
        nonlin_rows.append({
            "sigma_z": sz,
            "forward_probe_rel_err": rel_fwd,
            "central_probe_rel_err": rel_ctr,
        })
    # Fit log-log slopes on small sigma_z (1e-3..0.05)
    xs = np.log([r["sigma_z"] for r in nonlin_rows[:5]])
    slope_fwd = float(np.polyfit(xs, np.log([r["forward_probe_rel_err"] for r in nonlin_rows[:5]]), 1)[0])
    slope_ctr = float(np.polyfit(xs, np.log([r["central_probe_rel_err"] for r in nonlin_rows[:5]]), 1)[0])

    # --- 4C: v12 Zero-Sync Pullback Fastpath (_unified_v11/prior.py) vs Original (_unified_v4/prior.py) ---
    # 1) Direct primitive parity (Sinkhorn fastpath + update_quantizer over 20 steps of grid100)
    rng_fp = torch.Generator(device=device).manual_seed(0)
    b_fp, k_fp = 256, 192
    l_xp = -torch.rand(b_fp, k_fp, generator=rng_fp, device=device) * 20.0
    w_fp = torch.rand(k_fp, generator=rng_fp, device=device) + 0.1
    w_fp = w_fp / w_fp.sum()
    c_fp = torch.randn(k_fp, 2, generator=rng_fp, device=device)
    log_a = -math.log(float(b_fp))
    log_b = torch.log(w_fp.clamp_min(1e-12))

    # Reference v4 loop Sinkhorn
    u_ref = torch.zeros(b_fp, device=device, dtype=torch.float32)
    v_ref = torch.zeros(k_fp, device=device, dtype=torch.float32)
    for _ in range(10):
        v_ref = log_b - torch.logsumexp(l_xp + u_ref.unsqueeze(1), dim=0)
        u_ref = log_a - torch.logsumexp(l_xp + v_ref.unsqueeze(0), dim=1)
    p_ref = torch.exp(l_xp + u_ref.unsqueeze(1) + v_ref.unsqueeze(0))
    row_p = (float(b_fp) * p_ref).clamp_min(0.0)
    row_p = row_p / row_p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    y_ot_ref = row_p @ c_fp
    y_ot_fast = _sinkhorn_barycenter_fast(l_xp, log_a, log_b, c_fp, b_fp, torch.float32)
    sinkhorn_max_diff = float((y_ot_fast - y_ot_ref).abs().max().item())

    # 20-step update_quantizer parity (UnifiedV4PriorRegularizer vs UnifiedV11PriorRegularizer) on grid100
    reg_v4_q = UnifiedV4PriorRegularizer(max_prototypes=256)
    reg_v11_q = UnifiedV11PriorRegularizer(max_prototypes=256)
    g_data = torch.Generator(device=device).manual_seed(0)
    for _ in range(20):
        yr_g = sample_100gaussians(256, device, generator=g_data)
        reg_v4_q.update_quantizer(yr_g)
        reg_v11_q.update_quantizer(yr_g)
    quant_c_diff = float((reg_v11_q.proto_c - reg_v4_q.proto_c).abs().max().item())
    quant_v_diff = float((reg_v11_q.proto_var - reg_v4_q.proto_var).abs().max().item())
    quant_w_diff = float((reg_v11_q.proto_w - reg_v4_q.proto_w).abs().max().item())

    # 2) Full GANTrainer pullback step parity & EXACT host-sync count (unified_v4 vs unified_v11/v12) on grid100
    sync_bench: Dict[str, Any] = {}
    with torch.enable_grad():
        sync_bench = _benchmark_v4_vs_v11_pullback_step(device)

    return {
        "linear_probe_accuracy": {
            "2d_normalized_iso_max_rel_err": err_2d_iso,
            "2d_normalized_aniso_max_rel_err": err_2d_aniso,
            "4d_rh_unclamped_mean_rel_err": rh_mean_unclamped,
            "4d_rh_unclamped_max_rel_err": rh_max_unclamped,
            "4d_rh_clamped_015_mean_rel_err": rh_mean_clamped,
            "4d_rh_clamped_015_max_rel_err": rh_max_clamped,
            "4d_rh_small_norm_005_clamped_rel_err": rh_small_clamped_mean,
            "4d_minimal_2probe_single_sample_rel_err": min_rel_err_per_sample,
            "4d_minimal_2probe_ensemble_mean_rel_err": min_rel_err_mean_cov,
        },
        "nonlinear_curvature_bias": {
            "sweep": nonlin_rows,
            "forward_probe_log_slope": slope_fwd,
            "central_probe_log_slope": slope_ctr,
        },
        "v12_fastpath_vs_v4_original": {
            "device": str(device),
            "sinkhorn_barycenter_max_abs_diff": sinkhorn_max_diff,
            "update_quantizer_20step_proto_c_max_diff": quant_c_diff,
            "update_quantizer_20step_proto_var_max_diff": quant_v_diff,
            "update_quantizer_20step_proto_w_max_diff": quant_w_diff,
            "full_pullback_step_bench": sync_bench,
        },
    }


@contextmanager
def _count_host_syncs(device: torch.device):
    """Count explicit Tensor.item(), Tensor.tolist(), Tensor.__bool__(), and torch.unique calls on GPU."""
    counts = {"item": 0, "tolist": 0, "bool": 0, "unique": 0}
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_bool = torch.Tensor.__bool__
    orig_unique = torch.unique

    def _wrap_item(self, *args, **kwargs):
        if self.device.type == device.type:
            counts["item"] += 1
        return orig_item(self, *args, **kwargs)

    def _wrap_tolist(self, *args, **kwargs):
        if self.device.type == device.type:
            counts["tolist"] += 1
        return orig_tolist(self, *args, **kwargs)

    def _wrap_bool(self, *args, **kwargs):
        if self.device.type == device.type:
            counts["bool"] += 1
        return orig_bool(self, *args, **kwargs)

    def _wrap_unique(inp, *args, **kwargs):
        if isinstance(inp, torch.Tensor) and inp.device.type == device.type:
            counts["unique"] += 1
        return orig_unique(inp, *args, **kwargs)

    torch.Tensor.item = _wrap_item
    torch.Tensor.tolist = _wrap_tolist
    torch.Tensor.__bool__ = _wrap_bool
    torch.unique = _wrap_unique
    try:
        yield counts
    finally:
        torch.Tensor.item = orig_item
        torch.Tensor.tolist = orig_tolist
        torch.Tensor.__bool__ = orig_bool
        torch.unique = orig_unique


def _benchmark_v4_vs_v11_pullback_step(device: torch.device) -> Dict[str, Any]:
    """Run matched 10-step GANTrainer pullback steps (unified_v4 vs unified_v11 vs unified_v12) on grid100."""
    def _build_trainer(preset: str) -> GANTrainer:
        g_mod = SimpleMLPGenerator(2, 128, n_hidden=3, out_dim=2).to(device)
        d_mod = SimpleMLPDiscriminator(2, 128, n_hidden=3).to(device)
        prior = MoGParticlePrior(num_particles=1024, z_dim=2, sigma=0.025, device=device)
        deterministic_orthogonal_(g_mod, seed=0)
        deterministic_orthogonal_(d_mod, seed=0)
        deterministic_orthogonal_(prior, seed=0)
        recipe = get_recipe(preset, num_particles=1024, z_dim=2, prior_kind="mog")
        return GANTrainer(
            recipe,
            g_mod,
            d_mod,
            prior=prior,
            seed=0,
        )

    # Pre-generate identical grid100 batch sequence
    g_data = torch.Generator(device=device).manual_seed(0)
    batches = [sample_100gaussians(256, device, generator=g_data) for _ in range(15)]

    out: Dict[str, Any] = {}
    for preset in ("unified_v4", "unified_v11", "unified_v12"):
        trainer = _build_trainer(preset)
        # Warmup 5 steps (builds quantizer + triggers steady-state coalesced pullback path)
        for t in range(5):
            trainer.step(batches[t])
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Measure 10 steady-state steps: wall time + explicit tensor->host sync counts
        t0 = time.perf_counter()
        with _count_host_syncs(device) as syncs:
            for t in range(5, 15):
                trainer.step(batches[t])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / 10.0

        total_syncs_per_step = sum(syncs.values()) / 10.0
        out[preset] = {
            "ms_per_step": dt_ms,
            "host_syncs_per_step_total": total_syncs_per_step,
            "host_syncs_per_step_breakdown": {k: v / 10.0 for k, v in syncs.items()},
        }

    # Measure isolated pullback-step-only parity & syncs: identical G, D, critic, and shared Adam opt_g
    # so BOTH unified_v4_pullback_step and unified_v11_pullback_step execute the exact same optimizer branch!
    def _build_isolated_pullback_pair() -> Dict[str, Any]:
        tr_v11 = _build_trainer("unified_v11")
        tr_v4 = _build_trainer("unified_v11")
        tr_v4.prior_regularizer = UnifiedV4PriorRegularizer(
            max_prototypes=256,
            bound_device=device,
            bound_shape=tuple(tr_v4.prior.z.shape),
        )
        # Run 5 shared warmup steps on identical batches with v11, then clone exact state to both
        for t in range(5):
            tr_v11.step(batches[t])
        tr_v4.G.load_state_dict(tr_v11.G.state_dict())
        tr_v4.D.load_state_dict(tr_v11.D.state_dict())
        tr_v4.prior.load_state_dict(tr_v11.prior.state_dict())
        tr_v4.prior_regularizer.load_state_dict(tr_v11.prior_regularizer.state_dict())
        # Use identical plain Adam opt_g on both trainers to eliminate UnifiedV4 vs UnifiedV11 optimizer branch diffs
        tr_v11.opt_g = torch.optim.Adam(list(tr_v11.G.parameters()) + [tr_v11.prior.z], lr=1e-3)
        tr_v4.opt_g = torch.optim.Adam(list(tr_v4.G.parameters()) + [tr_v4.prior.z], lr=1e-3)
        tr_v11.recipe = tr_v11.recipe.replace(constraint_geometry_mode="none")
        tr_v4.recipe = tr_v4.recipe.replace(constraint_geometry_mode="none")
        tr_v11.policy._phase = "generator"
        tr_v4.policy._phase = "generator"

        # Execute 1 matched pullback step directly with identical inputs, measuring pure pullback syncs & diffs!
        real_b = batches[5]
        z_rng = torch.Generator(device=device).manual_seed(777)
        latent_v11, idx_v11 = tr_v11.prior.sample(len(real_b), generator=z_rng)
        z_rng2 = torch.Generator(device=device).manual_seed(777)
        latent_v4, idx_v4 = tr_v4.prior.sample(len(real_b), generator=z_rng2)
        n_gen_1 = torch.Generator(device=device).manual_seed(888)
        n_gen_2 = torch.Generator(device=device).manual_seed(888)
        zero_t = real_b.new_zeros(())

        with _count_host_syncs(device) as sync_v11_pb:
            lg11, _, _, _, _, _ = unified_v11_pullback_step(
                tr_v11, real_b, None, latent_v11, idx_v11,
                sigma_out=0.0, noise=n_gen_1, critic=tr_v11.D, loss_d=zero_t, penalty=zero_t,
            )
        with _count_host_syncs(device) as sync_v4_pb:
            lg4, _, _, _, _, _ = unified_v4_pullback_step(
                tr_v4, real_b, None, latent_v4, idx_v4,
                sigma_out=0.0, noise=n_gen_2, critic=tr_v4.D, loss_d=zero_t, penalty=zero_t,
            )
        g_param_diff = max(
            float((p11 - p4).abs().max().item())
            for p11, p4 in zip(tr_v11.G.parameters(), tr_v4.G.parameters())
        )
        return {
            "v4_pullback_only_syncs": sum(sync_v4_pb.values()),
            "v4_pullback_only_breakdown": dict(sync_v4_pb),
            "v11_pullback_only_syncs": sum(sync_v11_pb.values()),
            "v11_pullback_only_breakdown": dict(sync_v11_pb),
            "single_step_loss_g_abs_diff": abs(float(lg11.detach().item()) - float(lg4.detach().item())),
            "single_step_proto_c_max_diff": float((tr_v11.prior_regularizer.proto_c - tr_v4.prior_regularizer.proto_c).abs().max().item()),
            "single_step_proto_var_max_diff": float((tr_v11.prior_regularizer.proto_var - tr_v4.prior_regularizer.proto_var).abs().max().item()),
            "single_step_psi_max_diff": float((tr_v11.prior_regularizer.psi - tr_v4.prior_regularizer.psi).abs().max().item()),
            "single_step_prior_z_max_diff": float((tr_v11.prior.z - tr_v4.prior.z).abs().max().item()),
            "single_step_g_param_max_diff": g_param_diff,
        }

    out["isolated_pullback_step_v11_vs_v4"] = _build_isolated_pullback_pair()
    return out


def run_exp2_invariant_audit(device: torch.device) -> Dict[str, Any]:
    return {
        "invariant_1_mass_conservation": _audit_mass_conservation(),
        "invariant_2_pythagorean_ltv": _audit_pythagorean_ltv(),
        "invariant_3_laguerre_dual": _audit_laguerre_assignment(),
        "invariant_4_pullback_isometry_and_fastpath": _audit_pullback_isometry_and_fastpath(device),
    }


# ---------------------------------------------------------------------------
# Part 3: Non-Adversarial Division-of-Labor Experiment (Zero Discriminator!)
# ---------------------------------------------------------------------------

def make_disconnected_16mode_target(device: torch.device) -> Dict[str, torch.Tensor]:
    """16-mode disconnected anisotropic, unequal-width (3x), unequal-mass (4x) 2D Gaussian mixture."""
    coords = torch.tensor([-4.5, -1.5, 1.5, 4.5], device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # [16, 2], spacing = 3.0

    # 4x mass ratio across 16 modes
    raw_w = torch.tensor(
        [4.0, 3.0, 2.0, 1.0,
         2.5, 4.0, 1.2, 3.0,
         1.5, 2.0, 3.5, 1.0,
         3.0, 1.0, 2.5, 4.0],
        device=device,
        dtype=torch.float32,
    )
    weights = raw_w / raw_w.sum()

    # Unequal widths sigma_1 in [0.06, 0.18] (3x) and aspect ratios in [1.0, 2.5] at varied angles
    sig1 = torch.tensor(
        [0.06, 0.12, 0.18, 0.08,
         0.15, 0.06, 0.18, 0.10,
         0.09, 0.18, 0.07, 0.14,
         0.16, 0.08, 0.11, 0.06],
        device=device,
        dtype=torch.float32,
    )
    ratios = torch.tensor(
        [1.0, 2.0, 2.5, 1.5,
         2.2, 1.0, 2.5, 1.8,
         1.4, 2.4, 1.2, 2.0,
         2.5, 1.6, 1.8, 1.0],
        device=device,
        dtype=torch.float32,
    )
    sig2 = sig1 / ratios
    angles = torch.linspace(0.0, math.pi * (15.0 / 16.0), 16, device=device, dtype=torch.float32)

    chol_list = []
    cov_list = []
    inv_cov_list = []
    for m in range(16):
        c_a, s_a = torch.cos(angles[m]), torch.sin(angles[m])
        rot = torch.stack([torch.stack([c_a, -s_a]), torch.stack([s_a, c_a])])
        scale_mat = torch.diag(torch.stack([sig1[m], sig2[m]]))
        cov = rot @ (scale_mat @ scale_mat) @ rot.T
        cov_list.append(cov)
        chol_list.append(torch.linalg.cholesky(cov))
        inv_cov_list.append(torch.linalg.inv(cov))

    return {
        "centers": centers,
        "weights": weights,
        "covs": torch.stack(cov_list, dim=0),
        "chols": torch.stack(chol_list, dim=0),
        "inv_covs": torch.stack(inv_cov_list, dim=0),
    }


@torch.no_grad()
def sample_16mode_target(
    spec: Dict[str, torch.Tensor],
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    device = spec["centers"].device
    cdf = torch.cumsum(spec["weights"], dim=0)
    cdf[-1] = 1.0
    u = torch.rand(n, device=device, generator=generator)
    idx = torch.searchsorted(cdf, u).clamp(0, 15)
    eps = torch.randn(n, 2, device=device, generator=generator)
    return spec["centers"][idx] + torch.einsum("nij,nj->ni", spec["chols"][idx], eps)


def debiased_sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 0.08,
    iters: int = 20,
) -> torch.Tensor:
    """Exact differentiable Feydy-debiased Sinkhorn divergence S_eps(X, Y) (squared Euclidean cost)."""
    def _ot_eps(a: torch.Tensor, b: torch.Tensor, symmetric: bool) -> torch.Tensor:
        n, m = a.shape[0], b.shape[0]
        c_ab = 0.5 * torch.cdist(a, b, p=2.0).square()
        log_k = -c_ab / eps
        log_p = -math.log(float(n))
        log_q = -math.log(float(m))
        if symmetric:
            f = torch.zeros(n, device=a.device, dtype=a.dtype)
            for _ in range(iters):
                f = 0.5 * (f + log_p - torch.logsumexp(log_k + f.unsqueeze(0), dim=1))
            return eps * (2.0 * f.mean())
        f = torch.zeros(n, device=a.device, dtype=a.dtype)
        g = torch.zeros(m, device=a.device, dtype=a.dtype)
        for _ in range(iters):
            g = log_q - torch.logsumexp(log_k + f.unsqueeze(1), dim=0)
            f = log_p - torch.logsumexp(log_k + g.unsqueeze(0), dim=1)
        return eps * (f.mean() + g.mean())

    ot_xy = _ot_eps(x, y, symmetric=False)
    ot_xx = _ot_eps(x, x, symmetric=True)
    with torch.no_grad():
        ot_yy = _ot_eps(y, y, symmetric=True)
    return (ot_xy - 0.5 * ot_xx - 0.5 * ot_yy).clamp_min(0.0)


@torch.no_grad()
def sliced_w2_distance(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256) -> float:
    """Exact deterministic 2D Sliced Wasserstein-2 distance over equally spaced angles on [0, pi)."""
    angles = torch.linspace(0.0, math.pi * (1.0 - 1.0 / n_proj), n_proj, device=x.device, dtype=torch.float32)
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0)  # [2, n_proj]
    px = torch.sort(x.float() @ dirs, dim=0).values
    py = torch.sort(y.float() @ dirs, dim=0).values
    return float((px - py).square().mean().sqrt().item())


def compute_generator_jacobian_stats(
    g_mod: nn.Module,
    z_eval: torch.Tensor,
) -> Dict[str, float]:
    """Compute exact batched Jacobian spectral norm ||J_G(z)||_2 and condition number via jacrev."""
    g_mod.eval()

    def _single_g(z_one: torch.Tensor) -> torch.Tensor:
        return g_mod(z_one.unsqueeze(0)).squeeze(0)

    jacs = torch.func.vmap(torch.func.jacrev(_single_g))(z_eval.detach())  # [N, 2, 2]
    sv = torch.linalg.svdvals(jacs.float())  # [N, 2], descending
    s1 = sv[:, 0]
    s2 = sv[:, 1].clamp_min(1e-8)
    cond = (s1 / s2).clamp_max(1e6)
    return {
        "jac_max": float(s1.max().item()),
        "jac_p95": float(torch.quantile(s1, 0.95).item()),
        "jac_med": float(s1.median().item()),
        "jac_mean": float(s1.mean().item()),
        "jac_cond_med": float(cond.median().item()),
        "jac_cond_p95": float(torch.quantile(cond, 0.95).item()),
    }


@torch.no_grad()
def evaluate_nonadversarial_arm(
    g_mod: nn.Module,
    sample_z_fn: Any,
    spec: Dict[str, torch.Tensor],
    y_eval: torch.Tensor,
    *,
    n_eval: int = 8192,
    n_jac: int = 4096,
) -> Dict[str, Any]:
    g_mod.eval()
    z_eval = sample_z_fn(n_eval)
    x_eval = g_mod(z_eval).detach().float()
    y_sub = y_eval[:n_eval]

    sw2 = sliced_w2_distance(x_eval, y_sub)
    s_div = float(debiased_sinkhorn_divergence(x_eval[:2048], y_sub[:2048], eps=0.05, iters=25).item())
    sinkhorn_w2 = math.sqrt(max(0.0, 2.0 * s_div))

    # Mahalanobis distance to each of the 16 anisotropic modes:
    diff = x_eval.unsqueeze(1) - spec["centers"].unsqueeze(0)  # [N, 16, 2]
    maha_sq = torch.einsum("nmi,mij,nmj->nm", diff, spec["inv_covs"], diff)
    min_maha, assign = maha_sq.clamp_min(0.0).sqrt().min(dim=1)
    p_bridge = float((min_maha > 3.5).float().mean().item())

    # Mode coverage & mass TV (counting in-mode samples within 3.5 sigma)
    in_mode = min_maha <= 3.5
    counts_all = torch.bincount(assign, minlength=16).float()
    w_hat = counts_all / float(n_eval)
    mass_tv = 0.5 * float((w_hat - spec["weights"]).abs().sum().item())

    counts_in = torch.bincount(assign[in_mode], minlength=16).float() if bool(in_mode.any()) else torch.zeros(16, device=x_eval.device)
    w_in = counts_in / float(n_eval)
    covered_mask = (w_in >= 0.25 * spec["weights"]) & (counts_in >= 5.0)
    modes_covered = int(covered_mask.sum().item())

    # Mean relative Frobenius covariance error across covered modes (testing anisotropy preservation!)
    cov_errs = []
    for m in range(16):
        pts_m = x_eval[in_mode & (assign == m)]
        if pts_m.shape[0] >= 15:
            pts_c = pts_m - pts_m.mean(dim=0, keepdim=True)
            cov_hat = (pts_c.T @ pts_c) / float(pts_m.shape[0] - 1)
            cov_true = spec["covs"][m]
            cov_errs.append(float((cov_hat - cov_true).norm().item() / cov_true.norm().item()))
    mean_cov_err = float(np.mean(cov_errs)) if cov_errs else float("inf")

    jac_stats = compute_generator_jacobian_stats(g_mod, z_eval[:n_jac])
    return {
        "sw2": sw2,
        "sinkhorn_w2": sinkhorn_w2,
        "p_bridge": p_bridge,
        "modes_covered": modes_covered,
        "mass_tv": mass_tv,
        "mean_cov_rel_frob_err": mean_cov_err,
        **jac_stats,
    }


def _build_fixed_4mode_mismatched_prior(device: torch.device) -> Tuple[torch.Tensor, float]:
    """Fixed mismatched 4-mode prior (wrong count: 4 vs 16; wrong locations)."""
    c4 = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]], device=device, dtype=torch.float32)
    return c4, 0.08


def run_exp3_nonadversarial_division_of_labor(
    device: torch.device,
    *,
    steps: int = 1200,
    batch_size: int = 256,
    eval_every: int = 100,
) -> Dict[str, Any]:
    """Test Law 1 Division of Labor under pure non-adversarial debiased Sinkhorn loss (ZERO critic!).

    Arms (all protocol seed 0, identical real batch stream, deterministic_orthogonal_ init):
      - Arm A: Fixed unimodal N(0, I_2) + 3-hidden-layer MLP (33,666 params)
      - Arm B: Fixed mismatched 4-mode MoG + identical 3-hidden-layer MLP (33,666 params)
      - Arm C0: Learnable MoGParticlePrior (N=1024, sigma=0.05) via pure Sinkhorn grad (no Laguerre)
                + identical 3-hidden-layer MLP (minimal machinery test!)
      - Arm C1: Learnable MoGParticlePrior (N=1024, sigma=0.05) + Non-Adversarial Streaming Atlas
                & Laguerre Dual Condensation + identical 3-hidden-layer MLP
      - Arm C2: Learnable MoGParticlePrior (N=1024, sigma=0.05) + Non-Adversarial Streaming Atlas
                & Laguerre Dual Condensation + TINY 1-hidden-layer width-16 MLP (82 params, 410x smaller!)
    """
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        return _run_exp3_body(device, steps=steps, batch_size=batch_size, eval_every=eval_every)
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def _run_exp3_body(
    device: torch.device,
    *,
    steps: int = 1200,
    batch_size: int = 256,
    eval_every: int = 100,
) -> Dict[str, Any]:
    spec = make_disconnected_16mode_target(device)

    # Pre-generate identical real minibatch stream and fixed held-out eval set (isolated RNGs!)
    rng_data = torch.Generator(device=device).manual_seed(0)
    real_batches = [sample_16mode_target(spec, batch_size, rng_data) for _ in range(steps)]
    rng_eval = torch.Generator(device=device).manual_seed(12345)
    y_eval = sample_16mode_target(spec, 8192, rng_eval)

    # True target reference metrics (oracle floor)
    y_ref2 = sample_16mode_target(spec, 8192, torch.Generator(device=device).manual_seed(54321))
    diff_ref = y_ref2.unsqueeze(1) - spec["centers"].unsqueeze(0)
    maha_ref = torch.einsum("nmi,mij,nmj->nm", diff_ref, spec["inv_covs"], diff_ref).clamp_min(0.0).sqrt().min(dim=1).values
    oracle_sdiv = float(debiased_sinkhorn_divergence(y_ref2[:2048], y_eval[:2048], eps=0.05, iters=25).item())
    oracle_ref = {
        "sw2": sliced_w2_distance(y_ref2, y_eval),
        "sinkhorn_w2": math.sqrt(max(0.0, 2.0 * oracle_sdiv)),
        "p_bridge": float((maha_ref > 3.5).float().mean().item()),
        "analytic_3p5sigma_tail": math.exp(-0.5 * 3.5 * 3.5),
    }

    arms_config = [
        ("arm_a_fixed_unimodal", "unimodal", 128, 3),
        ("arm_b_fixed_mismatched_4mode", "mismatched_4mode", 128, 3),
        ("arm_c0_learnable_mog_pure_sinkhorn", "adaptive_pure_sinkhorn", 128, 3),
        ("arm_c1_adaptive_laguerre_atlas", "adaptive_laguerre", 128, 3),
        ("arm_c2_adaptive_laguerre_tiny_g", "adaptive_laguerre", 16, 1),
    ]

    arm_results: Dict[str, Any] = {}
    c4_mismatch, sig4_mismatch = _build_fixed_4mode_mismatched_prior(device)

    for arm_name, mode, hidden_dim, n_hidden in arms_config:
        g_mod = SimpleMLPGenerator(2, hidden_dim, n_hidden=n_hidden, out_dim=2).to(device)
        deterministic_orthogonal_(g_mod, seed=0)
        n_params_g = sum(p.numel() for p in g_mod.parameters())

        prior: Optional[MoGParticlePrior] = None
        if mode in ("adaptive_pure_sinkhorn", "adaptive_laguerre"):
            prior = MoGParticlePrior(
                num_particles=1024,
                z_dim=2,
                sigma=0.05,
                standardize=False,
                device=device,
            )
            deterministic_orthogonal_(prior, seed=0)

        rng_z = torch.Generator(device=device).manual_seed(100)
        opt_g = torch.optim.Adam(g_mod.parameters(), lr=3e-3)
        opt_prior = (
            torch.optim.Adam([prior.z], lr=4e-2)
            if (prior is not None and mode == "adaptive_pure_sinkhorn")
            else None
        )

        # Non-adversarial streaming atlas + Laguerre dual state for Arm C1 / C2
        atlas_reg = UnifiedV11PriorRegularizer(max_prototypes=64) if mode == "adaptive_laguerre" else None
        row_m = torch.zeros_like(prior.z) if (prior is not None and mode == "adaptive_laguerre") else None
        row_v = torch.zeros((1024, 1), device=device) if (prior is not None and mode == "adaptive_laguerre") else None
        row_n = torch.zeros((1024, 1), device=device) if (prior is not None and mode == "adaptive_laguerre") else None

        def _make_eval_sampler(arm_mode: str, arm_prior: Optional[MoGParticlePrior]):
            def _sample(n_s: int) -> torch.Tensor:
                ev_rng = torch.Generator(device=device).manual_seed(9999)
                if arm_mode == "unimodal":
                    return torch.randn(n_s, 2, device=device, generator=ev_rng)
                if arm_mode == "mismatched_4mode":
                    idx = torch.randint(0, 4, (n_s,), device=device, generator=ev_rng)
                    return c4_mismatch[idx] + sig4_mismatch * torch.randn(n_s, 2, device=device, generator=ev_rng)
                assert arm_prior is not None
                z_s, _ = arm_prior.sample(n_s, generator=ev_rng)
                return z_s
            return _sample

        eval_sampler = _make_eval_sampler(mode, prior)
        steps_to_sw2_035: Optional[int] = None
        steps_to_sinkhorn_035: Optional[int] = None
        steps_to_bridge_005: Optional[int] = None
        trajectory: List[Dict[str, Any]] = []

        t_start = time.perf_counter()
        for step in range(1, steps + 1):
            g_mod.train()
            yr = real_batches[step - 1]

            if mode == "unimodal":
                z = torch.randn(batch_size, 2, device=device, generator=rng_z)
                opt_g.zero_grad()
                xf = g_mod(z)
                loss = debiased_sinkhorn_divergence(xf, yr, eps=0.08, iters=18)
                loss.backward()
                opt_g.step()

            elif mode == "mismatched_4mode":
                idx = torch.randint(0, 4, (batch_size,), device=device, generator=rng_z)
                z = c4_mismatch[idx] + sig4_mismatch * torch.randn(batch_size, 2, device=device, generator=rng_z)
                opt_g.zero_grad()
                xf = g_mod(z)
                loss = debiased_sinkhorn_divergence(xf, yr, eps=0.08, iters=18)
                loss.backward()
                opt_g.step()

            elif mode == "adaptive_pure_sinkhorn":
                assert prior is not None and opt_prior is not None
                z, _ = prior.sample(batch_size, generator=rng_z)
                opt_g.zero_grad()
                opt_prior.zero_grad()
                xf = g_mod(z)
                loss = debiased_sinkhorn_divergence(xf, yr, eps=0.08, iters=18)
                loss.backward()
                opt_g.step()
                opt_prior.step()

            else:
                # Arm C1 / C2: Non-adversarial Division of Labor!
                # Part A: Update G_theta via pure Sinkhorn S_eps(G(z), Y) with prior detached!
                assert prior is not None and atlas_reg is not None
                assert row_m is not None and row_v is not None and row_n is not None
                z_raw, idx = prior.sample(batch_size, generator=rng_z)
                opt_g.zero_grad()
                xf = g_mod(z_raw.detach())
                loss = debiased_sinkhorn_divergence(xf, yr, eps=0.08, iters=18)
                loss.backward()
                opt_g.step()

                # Part B: Freeze G_theta, update streaming atlas (C_k, w_k, v_k) and Laguerre-balanced prior.z!
                with torch.no_grad():
                    atlas_reg.update_quantizer(yr)
                    c_bar, s_bar, w_mat, _ = _compute_pythagorean_smoothed_cells_fast(
                        atlas_reg.proto_c, atlas_reg.proto_var, atlas_reg.proto_w
                    )
                    r_mac = yr.std(dim=0, unbiased=False).mean().clamp_min(1e-4)
                    r_mac_sq = float(r_mac.square().item())
                    s_med = torch.quantile(s_bar, 0.35).clamp_min(1e-4)
                    iso_target = max(1.0, min(4.5, float((0.05 * r_mac / s_med).item())))
                    atlas_reg.iso_ema = 0.95 * atlas_reg.iso_ema + 0.05 * iso_target

                u_rows = torch.unique(idx)
                for p in g_mod.parameters():
                    p.requires_grad_(False)
                try:
                    c_u = prior.z[u_rows]
                    x_u = g_mod(c_u)
                    with torch.no_grad():
                        xu_det = x_u.detach().float()
                        psi_eff = w_mat @ atlas_reg.psi
                        d2_up = torch.cdist(xu_det, c_bar, p=2.0).square()
                        k_star = (d2_up - psi_eff.unsqueeze(0)).argmin(dim=1)

                        w_hard_u = w_mat[k_star].mean(dim=0)
                        atlas_reg.w_fake_ema = 0.88 * atlas_reg.w_fake_ema + 0.12 * w_hard_u
                        w_fb = 0.35 * w_hard_u + 0.65 * atlas_reg.w_fake_ema

                        k_f = float(len(atlas_reg.proto_c))
                        tv_now = 0.5 * float((atlas_reg.proto_w - atlas_reg.w_fake_ema).abs().sum().item())
                        # Lower deadband floor so rare 2.6% modes balance cleanly without stalling!
                        tv_fl = 0.15 * math.sqrt(k_f / max(float(len(u_rows)) * 8.0, 1.0))
                        pg = max(0.0, min(1.0, (tv_now - tv_fl) / max(0.04, tv_fl)))
                        eta_psi = pg * min(3.0, max(0.35, 0.02 * k_f)) * 2.0 * r_mac_sq
                        psi_next = atlas_reg.psi * (1.0 - 0.008 * pg) + eta_psi * (atlas_reg.proto_w - w_fb)
                        psi_next = w_mat @ psi_next
                        psi_next = psi_next - psi_next.mean()
                        pcap = 2.5 * 2.0 * r_mac_sq + 0.5
                        atlas_reg.psi = psi_next.clamp(-pcap, pcap)

                        c_targ = c_bar[k_star]
                        sc_u = s_bar[k_star]

                    # Condense prior rows toward their Laguerre-assigned mode centers + isometric cloud envelope
                    sq_u = (x_u - c_targ).square().sum(dim=-1)
                    loss_cond = (torch.sqrt((0.4 * sc_u).square() + sq_u) - 0.4 * sc_u).mean()
                    mu_c = c_u - c_u.mean(dim=0, keepdim=True)
                    mu_std = (mu_c.square().mean(dim=0) + 1e-6).sqrt()
                    loss_iso_mu = F.relu(atlas_reg.iso_ema - mu_std).square().mean()
                    loss_mu = loss_cond + 0.15 * loss_iso_mu

                    if prior.z.grad is not None:
                        prior.z.grad.zero_()
                    loss_mu.backward()
                    with torch.no_grad():
                        g_rows = (prior.z.grad[u_rows] * float(len(u_rows))).clone()
                        prior.z.grad.zero_()
                        m = row_m[u_rows] * 0.80 + 0.20 * g_rows
                        v = row_v[u_rows] * 0.80 + 0.20 * g_rows.square().sum(dim=-1, keepdim=True)
                        n_r = row_n[u_rows] + 1.0
                        row_m[u_rows], row_v[u_rows], row_n[u_rows] = m, v, n_r
                        bc = 1.0 - (0.80 ** n_r)
                        step_vec = (m / bc) / ((v / bc).sqrt() + 1e-5)
                        dist_i = sq_u.detach().sqrt().unsqueeze(1)
                        part_scale = (dist_i / (1.5 * sc_u.unsqueeze(1))).clamp(0.02, 1.2)
                        prior.z.index_add_(0, u_rows, step_vec * part_scale, alpha=-0.06)
                finally:
                    for p in g_mod.parameters():
                        p.requires_grad_(True)

            if step % eval_every == 0 or step == steps:
                metrics = evaluate_nonadversarial_arm(
                    g_mod, eval_sampler, spec, y_eval,
                    n_eval=4096 if step < steps else 8192,
                    n_jac=2048 if step < steps else 4096,
                )
                if steps_to_sw2_035 is None and metrics["sw2"] <= 0.35:
                    steps_to_sw2_035 = step
                if steps_to_sinkhorn_035 is None and metrics["sinkhorn_w2"] <= 0.35:
                    steps_to_sinkhorn_035 = step
                if steps_to_bridge_005 is None and metrics["p_bridge"] <= 0.05:
                    steps_to_bridge_005 = step
                if step in (100, 300, 600, 900, steps):
                    trajectory.append({
                        "step": step,
                        "sw2": round(metrics["sw2"], 4),
                        "sinkhorn_w2": round(metrics["sinkhorn_w2"], 4),
                        "p_bridge": round(metrics["p_bridge"], 4),
                        "modes": metrics["modes_covered"],
                        "mass_tv": round(metrics["mass_tv"], 4),
                        "jac_max": round(metrics["jac_max"], 2),
                        "jac_med": round(metrics["jac_med"], 2),
                    })

        wall_s = time.perf_counter() - t_start
        arm_results[arm_name] = {
            "g_params": n_params_g,
            "wall_seconds": round(wall_s, 2),
            "steps_to_sw2_035": steps_to_sw2_035,
            "steps_to_sinkhorn_035": steps_to_sinkhorn_035,
            "steps_to_bridge_005": steps_to_bridge_005,
            "final": metrics,
            "trajectory": trajectory,
        }

    return {
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "oracle_target_reference": oracle_ref,
        "arms": arm_results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    print(f"particlegan.__file__ = {particlegan.__file__}")
    print(f"device = {device}")

    exp1 = run_exp1_topology_and_isoperimetry()
    exp2 = run_exp2_invariant_audit(device)
    exp3 = run_exp3_nonadversarial_division_of_labor(device, steps=args.steps)

    report = {
        "particlegan_file": str(particlegan.__file__),
        "device": str(device),
        "exp1_topology_isoperimetry": exp1,
        "exp2_invariant_audit": exp2,
        "exp3_nonadversarial_division_of_labor": exp3,
    }

    print("\n=== EXP 1A: BIMODAL (Delta=2.0) CONSTRUCTIVE RAMP vs EXACT ISOPERIMETRIC BOUNDS ===")
    for r in exp1["bimodal_constructive_vs_bounds"]:
        print(
            f"alpha={r['alpha_target']:7.4f} | L_min={r['lip_lower_bound']:8.2f} "
            f"| W1_bnd={r['w1_exact_bound']:.5f} (quad={r['w1_quad_ramp']:.5f}, mc={r['w1_emp_ramp']:.5f}) "
            f"| W2_bnd={r['w2_exact_bound']:.5f} (quad={r['w2_quad_ramp']:.5f}, mc={r['w2_emp_ramp']:.5f}) "
            f"| bridge_mc={r['bridge_emp_ramp']:.5f} | MMD={r['mmd_emp_ramp']:.5f}"
        )

    print("\n=== EXP 1C: GRID100 (M=100, Delta=0.79, sigma=0.03) BOUNDS ===")
    g100 = exp1["grid100_geometry"]
    print(
        f"Poincare macro L >= {g100['poincare_macro_lip_lower_bound']:.4f} "
        f"(vs doc heuristic {g100['t1_prior_doc_heuristic']}), "
        f"adaptive MoG affine L = {g100['adaptive_mog_sigma_z_0025_affine_lip']:.2f}"
    )
    for r in exp1["multimode_scaling"]:
        if r["m_modes"] == 100:
            print(
                f"M=100, alpha={r['alpha']:.3f}: L_iso >= {r['grid100_lip_iso']:.2f} "
                f"(any d_z), L_1d_pack >= {r['grid100_lip_1d_pack']:.2f} (d_z=1)"
            )

    print("\n=== EXP 2: INVARIANT AUDIT SUMMARY ===")
    print("Mass conservation:", json.dumps(exp2["invariant_1_mass_conservation"], indent=2))
    print("Pythagorean LTV:", json.dumps(exp2["invariant_2_pythagorean_ltv"], indent=2))
    print("Laguerre dual:", json.dumps(exp2["invariant_3_laguerre_dual"], indent=2))
    print("Pullback isometry & fastpath:", json.dumps(exp2["invariant_4_pullback_isometry_and_fastpath"], indent=2))

    print("\n=== EXP 3: NON-ADVERSARIAL DIVISION OF LABOR (ZERO DISCRIMINATOR!) ===")
    print("Oracle ref:", json.dumps(exp3["oracle_target_reference"]))
    for arm, info in exp3["arms"].items():
        fin = info["final"]
        print(
            f"{arm:36s} | params={info['g_params']:5d} | SW2={fin['sw2']:.4f} "
            f"| S_W2={fin['sinkhorn_w2']:.4f} | p_bridge={fin['p_bridge']:.4f} "
            f"| modes={fin['modes_covered']:2d}/16 | tv={fin['mass_tv']:.4f} "
            f"| cov_err={fin['mean_cov_rel_frob_err']:.3f} | J_max={fin['jac_max']:6.2f} "
            f"| J_med={fin['jac_med']:5.2f} | cond_med={fin['jac_cond_med']:5.2f} "
            f"| t(S_W2<=.35)={info['steps_to_sinkhorn_035']} | t(br<=.05)={info['steps_to_bridge_005']}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved full JSON results to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
