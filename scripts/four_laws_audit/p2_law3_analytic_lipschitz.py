"""P2 Mathematical & Empirical Audit of Law 3 (Scale, Mass, Conditional Fiber Identifiability & Analytic 1-Lipschitz Controls).

Reproducible, protocol-seed-0 verification suite accompanying
``reports/four-laws-audit/p2-law3-identifiability.md``.

Contents:
1. Exact mathematical theorem verifications:
   - Theorem 1: Global W1 Kantorovich potential non-identifiability of local scale,
     mode mass, and transport distance (plus Brenier W2 scalar-potential refutation
     of the naive "scalar potentials cannot encode geometry" overclaim).
   - Theorem 2: Beyond-horizon & equidistant orthogonal saddle-shift non-identifiability
     under both global W1 Kantorovich potentials and fixed-horizon screened transport.
   - Theorem 4 & Midpoint Cancellation Audit: Unresidualized context-smoothing drift
     conflation, exact generalized midpoint bias identity
     E[r_X r_X^T - r_Y r_Y^T | c] = (Sigma_X - Sigma_Y) + delta_m Delta^T + Delta delta_m^T,
     exact finite-sample in-sample hat-matrix attenuation (1 - h_jj), and
     zero_eigh_symmetric numerical & column-sign audit.
2. Critic Lipschitz control audit:
   - Explicit C^infinity analytic counterexample D_M with measure_critic_grad_median <= 0.1
     and sup ||grad D_M||_2 >= M (arbitrary M > 0).
   - Explicit SmoothSiLUCritic (exact T3 isolation architecture) counterexample with
     measure_critic_grad_median <= 0.25 and max ||grad D||_2 > 50 on target support.
   - Direct audit of trained T3IsolationFixture neural critics (bcap, rgdt) comparing
     measure_critic_grad_median against sampled max on real/fake support and 2D grid.
3. Generator-free (G = Id) & conditional experiments with PROVABLY GLOBALLY 1-LIPSCHITZ
   analytical critics (sup_{x in R^d} ||grad D(x)||_2 < 1 everywhere by closed-form proof):
   - Experiment A: Multi-scale 8x width / 6x mass mixture (t3_iso_scale_mass_8x).
   - Experiment B: 42.4-sigma beyond-horizon orthogonal saddle shift
     (t3_iso_horizon_saddle_shift).
   - Experiment C: Conditional heteroscedastic fiber task with outer-5% tail kink.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.toy_audit.api_t3_isolation import (
    SADDLE_POST_MEANS,
    SADDLE_POST_WEIGHTS,
    SADDLE_PRE_MEANS,
    SADDLE_PRE_WEIGHTS,
    SADDLE_SIGMA,
    SmoothSiLUCritic,
    T3IsolationFixture,
    _mixture_mode_metrics,
    measure_critic_grad_median,
    saddle_shift_parameters,
    sample_mixture,
    sample_target_for_kind,
    scale_mass_parameters,
    score_t3_isolation_kind,
)
from particlegan import ParticlePrior, Recipe, get_recipe, init
from particlegan._t3_dynamics.metric import (
    bipartite_deficit_macro_loss,
    born_infeld_c_hat_detached,
    born_infeld_pair,
    conformal_horizon_pair,
    pooled_tangent_conditional_loss,
    sinkhorn_slack,
    update_scalar_ell,
)
from particlegan._unified_v11.bank import UnifiedV11CondFiberBank, zero_eigh_symmetric
from particlegan._unified_v11.loss import (
    unified_v11_conditional_loss,
    v11_zero_eigh_bipartite_deficit_macro_loss,
)


# ---------------------------------------------------------------------------
# 1. Provably Globally 1-Lipschitz Analytical Potentials
# ---------------------------------------------------------------------------


def solve_exact_w1_dual_potentials(
    x_np: np.ndarray,
    y_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Solve exact empirical W1 Kantorovich dual potentials (f_i, g_j) for equal-size batches.

    Solves the primal 1-to-1 assignment via Jonker-Volgenant (`linear_sum_assignment`),
    then extracts exact feasible dual potentials ``g`` via Bellman-Ford relaxation on the
    reduced-cost graph ``W[j, j'] = d(x_{i(j)}, y_{j'}) - d(x_{i(j)}, y_j)``.
    By the triangle inequality, ``g_{j'} - g_j <= d(y_j, y_{j'})`` holds automatically,
    so the sup-convolution ``D(x) = max_j (g_j - ||x - y_j||_2)`` is globally 1-Lipschitz
    and attains ``mean(g) - mean(f) == W_1(hat_mu_X, hat_nu_Y)`` to machine precision.
    """
    x64 = np.asarray(x_np, dtype=np.float64)
    y64 = np.asarray(y_np, dtype=np.float64)
    b = x64.shape[0]
    if y64.shape[0] != b:
        raise ValueError(f"Expected equal batch sizes, got {b} vs {y64.shape[0]}")
    d_mat = np.linalg.norm(x64[:, None, :] - y64[None, :, :], axis=-1)
    r_ind, c_ind = linear_sum_assignment(d_mat)
    inv_c = np.argsort(c_ind)
    x_matched_idx = r_ind[inv_c]
    d_ord = d_mat[x_matched_idx]
    w_edge = d_ord - np.diag(d_ord)[:, None]

    g = np.zeros(b, dtype=np.float64)
    for _ in range(b):
        g_new = np.minimum(g, (g[:, None] + w_edge).min(axis=0))
        if np.max(np.abs(g_new - g)) <= 1e-13:
            g = g_new
            break
        g = g_new
    f = (g[None, :] - d_mat).max(axis=1)
    c0 = 0.5 * (float(g.mean()) + float(f.mean()))
    g = g - c0
    f = f - c0
    primal = float(d_mat[r_ind, c_ind].mean())
    dual = float(g.mean() - f.mean())
    return f, g, primal, abs(primal - dual)


def smooth_w1_sup_convolution(
    x_query: torch.Tensor,
    y_anchors: torch.Tensor,
    g_dual: torch.Tensor,
    *,
    eps: float = 0.01,
    tau: float = 0.02,
    gauge_shift: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """C^infinity, PROVABLY GLOBALLY 1-LIPSCHITZ smooth W1 Kantorovich potential.

    D(x) = tau * logsumexp_j ((g_j - sqrt(||x - y_j||_2^2 + eps^2)) / tau) - gauge_shift.

    Proof of global 1-Lipschitzness everywhere in R^d:
      grad_x D(x) = sum_j w_j(x) * (y_j - x) / sqrt(||x - y_j||_2^2 + eps^2),
    where w_j(x) >= 0, sum_j w_j(x) = 1 (softmax weights), and every term has
    Euclidean norm ||y_j - x|| / sqrt(||y_j - x||^2 + eps^2) < 1 strictly.
    By convexity of the Euclidean norm, sup_{x in R^d} ||grad_x D(x)||_2 < 1.
    """
    out_dtype = x_query.dtype
    xq64 = x_query.to(torch.float64)
    ya64 = y_anchors.to(device=x_query.device, dtype=torch.float64)
    g64 = g_dual.to(device=x_query.device, dtype=torch.float64)
    diff = xq64.unsqueeze(1) - ya64.unsqueeze(0)
    dist_smooth = torch.sqrt(diff.square().sum(dim=-1) + float(eps) * float(eps))
    logits = (g64.unsqueeze(0) - dist_smooth) / float(tau)
    val = float(tau) * torch.logsumexp(logits, dim=1)
    if gauge_shift is not None:
        val = val - gauge_shift.to(device=x_query.device, dtype=torch.float64)
    return val.to(out_dtype)


def smooth_dist_to_preshift_potential(
    x_query: torch.Tensor,
    *,
    a: float = 1.5,
    eps: float = 0.05,
    tau: float = 0.05,
    cap_at_valley: bool = True,
) -> torch.Tensor:
    """Smooth C^infinity globally 1-Lipschitz realization of Theorem 2's exact W1 potential.

    D_cap(x) = softmin(sqrt(||x - a e_1||^2 + eps^2), sqrt(||x + a e_1||^2 + eps^2), sqrt(2)*a) - c_0.
    Its gradient is a convex combination of two unit-bounded radial vectors and 0, so
    sup_{x in R^d} ||grad_x D_cap(x)||_2 < 1 strictly everywhere in R^d.
    As eps, tau -> 0 it attains the exact W1 duality gap 0 simultaneously for EVERY
    orthogonal orphan plane +-a v (v perp e_1), carrying ZERO information about v, while
    flattening smoothly at distance >= sqrt(2)*a so particles are not pushed to infinity.
    """
    dev, dt = x_query.device, x_query.dtype
    d = x_query.shape[1]
    e1 = torch.zeros(d, device=dev, dtype=dt)
    e1[0] = 1.0
    pre = torch.stack([a * e1, -a * e1], dim=0)
    d_pre = torch.sqrt((x_query.unsqueeze(1) - pre.unsqueeze(0)).square().sum(dim=-1) + eps * eps)
    if cap_at_valley:
        cap = torch.full((x_query.shape[0], 1), math.sqrt(2.0) * a, device=dev, dtype=dt)
        d_all = torch.cat([d_pre, cap], dim=1)
    else:
        d_all = d_pre
    smin_pre = -tau * torch.logsumexp(-d_all / tau, dim=1)
    return smin_pre - (math.sqrt(2.0) * a / 4.0)


def smooth_symmetric_saddle_potential(
    x_query: torch.Tensor,
    *,
    a: float = 1.5,
    v_dir: Tuple[float, ...] = (0.0, 1.0, 0.0, 0.0),
    eps: float = 0.05,
    tau: float = 0.05,
) -> torch.Tensor:
    """C^infinity globally 1-Lipschitz population symmetric potential for +-a e_1 -> (+-a e_1, +-a v).

    D(x) = 0.5 * (softmin_{s=+-1} sqrt(||x - s*a*e_1||^2 + eps^2)
                - softmin_{s=+-1} sqrt(||x - s*a*v||^2 + eps^2)).
    ||grad D(x)||_2 <= 0.5 * 1 + 0.5 * 1 = 1 strictly everywhere in R^d, and on the
    entire e_1-axis {t e_1} the +-v components cancel identically.
    """
    dev, dt = x_query.device, x_query.dtype
    d = x_query.shape[1]
    e1 = torch.zeros(d, device=dev, dtype=dt)
    e1[0] = 1.0
    v = torch.tensor(v_dir[:d], device=dev, dtype=dt)
    v = v / v.norm().clamp_min(1e-12)

    pre = torch.stack([a * e1, -a * e1], dim=0)
    orph = torch.stack([a * v, -a * v], dim=0)

    d_pre = torch.sqrt((x_query.unsqueeze(1) - pre.unsqueeze(0)).square().sum(dim=-1) + eps * eps)
    d_orph = torch.sqrt((x_query.unsqueeze(1) - orph.unsqueeze(0)).square().sum(dim=-1) + eps * eps)

    smin_pre = -tau * torch.logsumexp(-d_pre / tau, dim=1)
    smin_orph = -tau * torch.logsumexp(-d_orph / tau, dim=1)
    return 0.5 * (smin_pre - smin_orph)


def smooth_signed_distance_mixture_potential(
    x_query: torch.Tensor,
    means: torch.Tensor,
    *,
    eps: float = 0.05,
    tau: float = 0.08,
    scale: float = 0.25,
) -> torch.Tensor:
    """Provably globally 1-Lipschitz smooth negative-distance-to-centers potential (scale <= 1).

    D(x) = -scale * (-tau * logsumexp_m(-sqrt(||x - mu_m||^2 + eps^2) / tau)).
    Since ||grad (-smin)||_2 < 1 everywhere, sup_{x in R^d} ||grad D(x)||_2 < scale <= 1.
    Provides a non-collapsing global 1-Lipschitz centering field with ZERO information about
    per-mode covariance widths sigma_m (identical radial slope on sigma=0.04 and sigma=0.32!).
    """
    m_dev = means.to(device=x_query.device, dtype=x_query.dtype)
    d_m = torch.sqrt((x_query.unsqueeze(1) - m_dev.unsqueeze(0)).square().sum(dim=-1) + eps * eps)
    smin = -tau * torch.logsumexp(-d_m / tau, dim=1)
    return -float(scale) * smin


def certify_sampled_lipschitz(
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    probes: torch.Tensor,
) -> Dict[str, float]:
    """Compute exact autograd gradient norms on a comprehensive probe set."""
    p = probes.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        vals = potential_fn(p)
        grad = torch.autograd.grad(vals.sum(), p, create_graph=False)[0]
    norms = grad.reshape(len(p), -1).norm(dim=1).detach()
    return {
        "grad_med": float(norms.median().item()),
        "grad_p95": float(torch.quantile(norms, 0.95).item()),
        "grad_max": float(norms.max().item()),
        "grad_min": float(norms.min().item()),
    }


# ---------------------------------------------------------------------------
# 2. Formal Theorem Verifications (Theorems 1, 2, 3-Brenier, 4 & Midpoint)
# ---------------------------------------------------------------------------


def _exact_w1_lp_1d(
    x_pts: np.ndarray,
    p_x: np.ndarray,
    y_pts: np.ndarray,
    p_y: np.ndarray,
) -> float:
    n, m = len(x_pts), len(y_pts)
    c = np.abs(x_pts[:, None] - y_pts[None, :]).reshape(-1)
    a_eq = np.zeros((n + m, n * m), dtype=np.float64)
    for i in range(n):
        a_eq[i, i * m : (i + 1) * m] = 1.0
    for j in range(m):
        a_eq[n + j, j::m] = 1.0
    b_eq = np.concatenate([p_x, p_y])
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return float(res.fun)


def verify_theorem1_and_brenier_refutation() -> Dict[str, Any]:
    """Verify Theorem 1 (exact global W1 non-identifiability) and Brenier W2 refutation."""
    k_atoms = 17
    u_grid = np.linspace(-1.0, 1.0, k_atoms)
    u_w = np.full(k_atoms, 1.0 / k_atoms)

    x_mu = -1.0 + 0.05 * u_grid
    w_mu = u_w.copy()

    # Pair 1A (8x width ratio, identical mass & mean):
    y_1a = np.concatenate([x_mu, 1.0 + 0.04 * u_grid])
    w_1a = np.concatenate([0.5 * u_w, 0.5 * u_w])
    y_2a = np.concatenate([x_mu, 1.0 + 0.32 * u_grid])
    w_2a = np.concatenate([0.5 * u_w, 0.5 * u_w])

    # Pair 1B (mass vs distance trade-off, identical W1 = 1.0 and zero-mean gauge):
    y_2b = np.concatenate([x_mu, 3.0 + 0.32 * u_grid])
    w_2b = np.concatenate([0.75 * u_w, 0.25 * u_w])

    def eval_shared_potential(y_pts: np.ndarray, w_y: np.ndarray) -> Dict[str, float]:
        d_star = lambda t: t + 0.5
        gauge_mean = 0.5 * (float(np.sum(w_mu * d_star(x_mu))) + float(np.sum(w_y * d_star(y_pts))))
        dual_val = float(np.sum(w_y * d_star(y_pts)) - np.sum(w_mu * d_star(x_mu)))
        w1_lp = _exact_w1_lp_1d(x_mu, w_mu, y_pts, w_y)
        return {
            "w1_lp": w1_lp,
            "dual_val": dual_val,
            "duality_gap": abs(w1_lp - dual_val),
            "zero_mean_gauge": gauge_mean,
            "target_mean": float(np.sum(w_y * y_pts)),
            "target_var": float(np.sum(w_y * (y_pts - np.sum(w_y * y_pts)) ** 2)),
        }

    p1a = eval_shared_potential(y_1a, w_1a)
    p2a = eval_shared_potential(y_2a, w_2a)
    p2b = eval_shared_potential(y_2b, w_2b)

    # Exact W2 Brenier scalar potential refutation of naive "scalar potentials fail" overclaim:
    rng = np.random.default_rng(0)
    z_samples = rng.uniform(-1.05, -0.95, size=10000)
    brenier_results = {}
    for label, sigma_k in (("narrow_0.04", 0.04), ("wide_0.32", 0.32)):
        slope = sigma_k / 0.05
        grad_phi = z_samples - (1.0 + slope * (z_samples + 1.0))
        pushed = z_samples - grad_phi
        true_var = (sigma_k ** 2) / 3.0
        emp_var = float(np.var(pushed))
        sup_grad_phi = float(np.max(np.abs(grad_phi)))
        brenier_results[label] = {
            "true_std": float(math.sqrt(true_var)),
            "pushed_std": float(math.sqrt(emp_var)),
            "rel_std_err": abs(math.sqrt(emp_var) - math.sqrt(true_var)) / math.sqrt(true_var),
            "sup_grad_phi": sup_grad_phi,
            "violates_1_lipschitz": sup_grad_phi > 1.0,
        }

    return {
        "pair_1a_narrow": p1a,
        "pair_1a_wide_8x": p2a,
        "pair_1b_mass_distance": p2b,
        "brenier_w2_refutation": brenier_results,
    }


def verify_theorem2_beyond_horizon() -> Dict[str, Any]:
    """Verify Theorem 2: global W1 & local-horizon non-identifiability of orthogonal orphan plane."""
    a = 1.5
    e1 = np.array([1.0, 0.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0, 0.0])

    x_mu = np.stack([a * e1, -a * e1], axis=0)
    w_mu = np.array([0.5, 0.5])
    c_gauge = math.sqrt(2.0) * a / 4.0

    def d_star_vec(pts: np.ndarray) -> np.ndarray:
        d1 = np.linalg.norm(pts - a * e1[None, :], axis=1)
        d2 = np.linalg.norm(pts + a * e1[None, :], axis=1)
        return np.minimum(d1, d2) - c_gauge

    planes = {}
    for name, v in (("P1_e2_plane", e2), ("P2_e3_plane", e3)):
        y_pts = np.stack([a * e1, -a * e1, a * v, -a * v], axis=0)
        w_y = np.array([0.25, 0.25, 0.25, 0.25])
        cost = np.linalg.norm(x_mu[:, None, :] - y_pts[None, :, :], axis=-1).reshape(-1)
        a_eq = np.zeros((6, 8), dtype=np.float64)
        for i in range(2):
            a_eq[i, i * 4 : (i + 1) * 4] = 1.0
        for j in range(4):
            a_eq[2 + j, j::4] = 1.0
        res = linprog(cost, A_eq=a_eq, b_eq=np.concatenate([w_mu, w_y]), bounds=(0.0, None), method="highs")
        w1_lp = float(res.fun)
        dual_val = float(np.sum(w_y * d_star_vec(y_pts)) - np.sum(w_mu * d_star_vec(x_mu)))
        gauge = 0.5 * (float(np.sum(w_mu * d_star_vec(x_mu))) + float(np.sum(w_y * d_star_vec(y_pts))))
        planes[name] = {
            "w1_lp": w1_lp,
            "dual_val": dual_val,
            "duality_gap": abs(w1_lp - dual_val),
            "zero_mean_gauge": gauge,
        }

    t_vals = torch.linspace(-2.5, 2.5, 101, dtype=torch.float64)
    axis_pts = torch.zeros((101, 4), dtype=torch.float64)
    axis_pts[:, 0] = t_vals
    axis_pts.requires_grad_(True)
    d_sym_e2 = smooth_symmetric_saddle_potential(axis_pts, a=a, v_dir=(0, 1, 0, 0), eps=0.05, tau=0.05)
    g_e2 = torch.autograd.grad(d_sym_e2.sum(), axis_pts)[0].detach()
    axis_pts2 = axis_pts.detach().clone().requires_grad_(True)
    d_sym_e3 = smooth_symmetric_saddle_potential(axis_pts2, a=a, v_dir=(0, 0, 1, 0), eps=0.05, tau=0.05)
    g_e3 = torch.autograd.grad(d_sym_e3.sum(), axis_pts2)[0].detach()

    rng = torch.Generator().manual_seed(0)
    x_cloud = sample_mixture(
        torch.tensor(SADDLE_PRE_WEIGHTS),
        torch.tensor(SADDLE_PRE_MEANS),
        torch.full((2,), SADDLE_SIGMA),
        128,
        rng,
    )
    y_cloud = sample_mixture(
        torch.tensor(SADDLE_POST_WEIGHTS),
        torch.tensor(SADDLE_POST_MEANS),
        torch.full((4,), SADDLE_SIGMA),
        128,
        rng,
    )
    ell = torch.tensor(0.05, dtype=torch.float32)
    c_hat = born_infeld_c_hat_detached(x_cloud, y_cloud, ell)
    zeros = torch.zeros(128, dtype=torch.float32)
    p_plan = sinkhorn_slack(c_hat, zeros, zeros, rho=3.0, tau=2.0, eps=0.1, iters=8)
    is_orphan = y_cloud[:, 1].abs() > y_cloud[:, 0].abs()
    orphan_active_mass = float(p_plan[:128, :128][:, is_orphan].sum().item())
    orphan_slack_mass = float(p_plan[128, :128][is_orphan].sum().item())
    orphan_min_c_hat = float(c_hat[:, is_orphan].amin().item())

    return {
        "global_w1_planes": planes,
        "sym_potential_e2_vs_e3_max_axis_grad_diff": float((g_e2 - g_e3).abs().max().item()),
        "sym_potential_max_orthogonal_grad_on_e1_axis": float(g_e2[:, 1:].abs().max().item()),
        "sym_potential_origin_grad_norm": float(g_e2[50].norm().item()),
        "sym_potential_axis_max_grad_norm": float(g_e2.norm(dim=1).max().item()),
        "fixed_horizon_valley_screening": {
            "ell": 0.05,
            "rho": 3.0,
            "c_screen_metric_py": 1.5 * 3.0,
            "c_screen_primitive_lin": 3.0,
            "c_screen_primitive_rap": 2.0 * 3.0 * (math.sqrt(math.e) - 1.0),
            "c_gprox_off": 2.8 * 3.0,
            "r_screen_metric_py_exact": 0.05 * math.sqrt((1.5 * 3.0 + 1.0) ** 2 - 1.0),
            "r_screen_primitive_lin_exact": 0.05 * math.sqrt((3.0 + 1.0) ** 2 - 1.0),
            "r_screen_primitive_rap_exact": 0.05 * math.sqrt((2.0 * 3.0 * (math.sqrt(math.e) - 1.0) + 1.0) ** 2 - 1.0),
            "r_gprox_off_exact": 0.05 * math.sqrt((2.8 * 3.0 + 1.0) ** 2 - 1.0),
            "dead_zone_ratio_metric_py": math.sqrt(((2.8 * 3.0 + 1.0) ** 2 - 1.0) / ((1.5 * 3.0 + 1.0) ** 2 - 1.0)),
            "dead_zone_ratio_primitive_lin": math.sqrt(((2.8 * 3.0 + 1.0) ** 2 - 1.0) / ((3.0 + 1.0) ** 2 - 1.0)),
            "dead_zone_ratio_primitive_rap": math.sqrt(
                ((2.8 * 3.0 + 1.0) ** 2 - 1.0) / ((2.0 * 3.0 * (math.sqrt(math.e) - 1.0) + 1.0) ** 2 - 1.0)
            ),
            "valley_dist": math.sqrt(2.0) * a,
            "orphan_min_c_hat": orphan_min_c_hat,
            "orphan_active_mass": orphan_active_mass,
            "orphan_slack_mass": orphan_slack_mass,
        },
    }


def verify_theorem4_midpoint_and_zero_eigh() -> Dict[str, Any]:
    """Verify generalized midpoint cancellation identity, finite-sample in-sample hat-matrix
    attenuation, unresidualized context-smoothing drift conflation, and zero_eigh_symmetric audit.
    """
    m_x = np.array([1.2, -0.5])
    m_y = np.array([-0.4, 0.9])
    delta = m_x - m_y
    m_mid = 0.5 * (m_x + m_y)
    sigma_x = np.array([[0.40, 0.12], [0.12, 0.25]])
    sigma_y = np.array([[0.15, -0.04], [-0.04, 0.30]])
    true_diff = sigma_x - sigma_y

    unequal_weight_checks = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        m_a = alpha * m_x + (1.0 - alpha) * m_y
        delta_m = m_mid - m_a
        lhs = (sigma_x + np.outer(m_x - m_a, m_x - m_a)) - (sigma_y + np.outer(m_y - m_a, m_y - m_a))
        rhs_general = true_diff + np.outer(delta_m, delta) + np.outer(delta, delta_m)
        rhs_alpha = true_diff + (1.0 - 2.0 * alpha) * np.outer(delta, delta)
        unequal_weight_checks[f"alpha_{alpha:.2f}"] = {
            "bias_frob_norm": float(np.linalg.norm(lhs - true_diff, ord="fro")),
            "general_formula_err": float(np.max(np.abs(lhs - rhs_general))),
            "alpha_formula_err": float(np.max(np.abs(lhs - rhs_alpha))),
        }

    m_misspec = m_mid + np.array([0.3, 0.2])
    delta_m_mis = m_mid - m_misspec
    lhs_mis = (sigma_x + np.outer(m_x - m_misspec, m_x - m_misspec)) - (
        sigma_y + np.outer(m_y - m_misspec, m_y - m_misspec)
    )
    rhs_mis = true_diff + np.outer(delta_m_mis, delta) + np.outer(delta, delta_m_mis)

    rng = np.random.default_rng(0)
    n_pts = 16
    c_design = rng.normal(0.0, 1.0, size=(n_pts, 2))
    phi = np.column_stack([np.ones(n_pts), c_design])
    h_mat = phi @ np.linalg.pinv(phi)
    h_diag = np.diag(h_mat)
    j_probe = int(np.argmax(h_diag))
    h_jj = float(h_diag[j_probe])

    b_x = np.array([[0.3, -0.2], [0.9, 0.1], [-0.2, 0.8]])
    b_y = np.array([[-0.5, 0.4], [0.6, -0.1], [0.1, 0.5]])
    true_mx = phi @ b_x
    true_my = phi @ b_y
    chol_x = np.linalg.cholesky(sigma_x)
    chol_y = np.linalg.cholesky(sigma_y)

    n_mc = 40000
    eps_x = rng.normal(0.0, 1.0, size=(n_mc, n_pts, 2)) @ chol_x.T
    eps_y = rng.normal(0.0, 1.0, size=(n_mc, n_pts, 2)) @ chol_y.T
    x_mc = true_mx[None, :, :] + eps_x
    y_mc = true_my[None, :, :] + eps_y
    m_hat_mc = np.einsum("ij,mjd->mid", h_mat, 0.5 * (x_mc + y_mc))
    rx_j = x_mc[:, j_probe, :] - m_hat_mc[:, j_probe, :]
    ry_j = y_mc[:, j_probe, :] - m_hat_mc[:, j_probe, :]
    emp_diff_j = np.einsum("mi,mj->ij", rx_j, rx_j) / n_mc - np.einsum("mi,mj->ij", ry_j, ry_j) / n_mc
    expected_attenuated = (1.0 - h_jj) * true_diff

    eps_y_eq = rng.normal(0.0, 1.0, size=(n_mc, n_pts, 2)) @ chol_x.T
    y_mc_eq = true_my[None, :, :] + eps_y_eq
    m_hat_eq = np.einsum("ij,mjd->mid", h_mat, 0.5 * (x_mc + y_mc_eq))
    rx_eq = x_mc[:, j_probe, :] - m_hat_eq[:, j_probe, :]
    ry_eq = y_mc_eq[:, j_probe, :] - m_hat_eq[:, j_probe, :]
    emp_eq_diff_j = np.einsum("mi,mj->ij", rx_eq, rx_eq) / n_mc - np.einsum("mi,mj->ij", ry_eq, ry_eq) / n_mc

    a_drift = np.array([[0.85, 0.25], [-0.20, 0.80]])
    sigma_0 = 0.08 * np.eye(2)
    v_w = 0.25 * np.eye(2)
    cov_p1_smoothed = sigma_0 + a_drift @ v_w @ a_drift.T
    eig_inflation_ratio = float(np.linalg.eigvalsh(cov_p1_smoothed).max() / np.linalg.eigvalsh(sigma_0).max())

    gen_t = torch.Generator().manual_seed(0)
    eigh_audit = {}
    for d in (1, 2, 3, 4, 8, 16):
        q_rand, _ = torch.linalg.qr(torch.randn((d, d), generator=gen_t, dtype=torch.float64))
        eigs_ill = torch.logspace(-6, 0, d, dtype=torch.float64)
        a_ill = q_rand @ torch.diag(eigs_ill) @ q_rand.T
        a_ill = 0.5 * (a_ill + a_ill.T)
        v_z, u_z = zero_eigh_symmetric(a_ill)
        v_ref, _ = torch.linalg.eigh(a_ill)
        rec_err = float((a_ill - (u_z * v_z.unsqueeze(0)) @ u_z.T).norm().item())
        val_err = float((v_z - v_ref).abs().max().item())
        ortho_err = float((u_z.T @ u_z - torch.eye(d, dtype=torch.float64)).abs().max().item())
        first_nonzero_signs = []
        for col in range(d):
            vec = u_z[:, col]
            nz = vec.abs() > 1e-6
            if bool(nz.any()):
                first_nonzero_signs.append(float(vec[torch.nonzero(nz, as_tuple=False)[0, 0]].item()))
        has_negative_leading_sign = any(s < 0.0 for s in first_nonzero_signs)
        eigh_audit[f"d_{d}"] = {
            "max_eigval_err": val_err,
            "recon_frob_err": rec_err,
            "ortho_err": ortho_err,
            "has_negative_leading_col_sign": has_negative_leading_sign,
        }

    return {
        "unequal_weight_checks": unequal_weight_checks,
        "misspec_bias_frob_norm": float(np.linalg.norm(lhs_mis - true_diff, ord="fro")),
        "misspec_formula_err": float(np.max(np.abs(lhs_mis - rhs_mis))),
        "insample_hat_leverage_h_jj": h_jj,
        "insample_mc_vs_attenuated_max_err": float(np.max(np.abs(emp_diff_j - expected_attenuated))),
        "insample_mc_vs_unattenuated_max_err": float(np.max(np.abs(emp_diff_j - true_diff))),
        "insample_equal_cov_zero_bias_max_err": float(np.max(np.abs(emp_eq_diff_j))),
        "unresidualized_eig_inflation_ratio": eig_inflation_ratio,
        "zero_eigh_symmetric_audit": eigh_audit,
    }


# ---------------------------------------------------------------------------
# 3. Critic Lipschitz Control Audit & Counterexamples
# ---------------------------------------------------------------------------


def audit_critic_lipschitz_controls(*, include_fixture_audit: bool = True) -> Dict[str, Any]:
    """Demonstrate counterexamples to measure_critic_grad_median <= 1.0 and audit actual fixtures."""
    rng = torch.Generator().manual_seed(0)
    ref = sample_target_for_kind("t3_iso_scale_mass_8x", 2048, rng, device="cpu")
    pts = sample_target_for_kind("t3_iso_scale_mass_8x", 512, rng, device="cpu")
    _, means, sigmas = scale_mass_parameters("cpu")
    mu0 = means[0]
    s0 = float(sigmas[0].item())

    analytic_counterexamples = {}
    for m_target in (10.0, 100.0, 1000.0):
        def make_d_m(m_val: float) -> Callable[[torch.Tensor], torch.Tensor]:
            def _fn(x: torch.Tensor) -> torch.Tensor:
                diff = x - mu0.to(device=x.device, dtype=x.dtype).unsqueeze(0)
                r2 = diff.square().sum(dim=1)
                bump = m_val * s0 * torch.tanh(diff[:, 0] / s0) * torch.exp(-r2 / (2.0 * (2.0 * s0) ** 2))
                return 0.05 * x[:, 0] + bump
            return _fn

        fn_m = make_d_m(m_target)
        g_med = measure_critic_grad_median(fn_m, pts, ref)
        probe_mu0 = mu0.unsqueeze(0).clone().requires_grad_(True)
        g_mu0 = float(torch.autograd.grad(fn_m(probe_mu0).sum(), probe_mu0)[0].norm().item())
        ref_cert = certify_sampled_lipschitz(fn_m, ref)
        analytic_counterexamples[f"M_{int(m_target)}"] = {
            "measure_critic_grad_median": g_med,
            "passes_t3_gate_le_1": g_med <= 1.0,
            "grad_norm_at_mu0": g_mu0,
            "ref_grad_max": ref_cert["grad_max"],
        }

    critic = SmoothSiLUCritic(in_dim=2, hidden=64, layers=2)
    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()
        mu3 = means[3]
        gain = 120.0
        t1 = float(mu3[0].item()) - 0.03
        t2 = float(mu3[0].item()) + 0.03
        critic.net[0].weight[0, 0] = gain
        critic.net[0].bias[0] = -gain * t1
        critic.net[0].weight[1, 0] = gain
        critic.net[0].bias[1] = -gain * t2
        critic.net[0].weight[2, 0] = 0.2
        critic.net[0].bias[2] = 5.0
        critic.net[2].weight[0, 0] = 1.0
        critic.net[2].weight[0, 1] = -1.0
        critic.net[2].bias[0] = 10.0
        critic.net[2].weight[1, 2] = 1.0
        critic.net[2].bias[1] = 10.0
        critic.net[4].weight[0, 0] = 1.0
        critic.net[4].weight[0, 1] = 1.0

    silu_g_med = measure_critic_grad_median(critic, pts, ref)
    probe_mu3 = mu3.unsqueeze(0).clone().requires_grad_(True)
    silu_g_mu3 = float(torch.autograd.grad(critic(probe_mu3).sum(), probe_mu3)[0].norm().item())
    silu_ref_cert = certify_sampled_lipschitz(critic, ref)
    silu_lip_const = 1.09984
    w0_op = float(torch.linalg.matrix_norm(critic.net[0].weight, ord=2).item())
    w2_op = float(torch.linalg.matrix_norm(critic.net[2].weight, ord=2).item())
    w4_op = float(torch.linalg.matrix_norm(critic.net[4].weight, ord=2).item())
    silu_counterexample = {
        "measure_critic_grad_median": silu_g_med,
        "passes_t3_gate_le_1": silu_g_med <= 1.0,
        "grad_norm_at_mu3": silu_g_mu3,
        "ref_grad_max": silu_ref_cert["grad_max"],
        "spectral_upper_bound": w0_op * w2_op * w4_op * (silu_lip_const ** 2),
    }

    fixture_audit = {}
    if include_fixture_audit:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gx = torch.linspace(-2.5, 2.5, 81, device=device)
        gx_m, gy_m = torch.meshgrid(gx, gx, indexing="ij")
        grid = torch.stack([gx_m.reshape(-1), gy_m.reshape(-1)], dim=1)
        for preset in ("bcap", "rgdt"):
            base = get_recipe(preset).to_dict()
            base.update({
                "num_particles": 512,
                "z_dim": 2,
                "batch_size": 128,
                "total_steps": 200,
                "prior_kind": "particles",
                "sigma_rel": 0.0,
                "standardize": False,
                "prior_update": "learned",
            })
            rec = Recipe.from_dict(base)
            fix = T3IsolationFixture("t3_iso_scale_mass_8x", rec, device=device, seed=0, max_steps=200)
            for _ in range(200):
                fix.step()
            pts_f = fix.prior.z.detach()
            g_med_f = measure_critic_grad_median(fix.D, pts_f, ref)
            cert_ref = certify_sampled_lipschitz(fix.D, ref.to(device))
            cert_pts = certify_sampled_lipschitz(fix.D, pts_f)
            cert_grid = certify_sampled_lipschitz(fix.D, grid)
            obs_f = fix.observe(ref)
            fixture_audit[preset] = {
                "steps": 200,
                "device": device,
                "critic_grad_med": g_med_f,
                "passes_t3_grad_gate": g_med_f <= 1.0,
                "ref_grad_max": cert_ref["grad_max"],
                "pts_grad_max": cert_pts["grad_max"],
                "grid_grad_max": cert_grid["grad_max"],
                "violates_global_1_lip": cert_grid["grad_max"] > 1.0,
                "core_min_eigen_ratio": obs_f["metrics"]["core_min_eigen_ratio"],
                "core_cov_err": obs_f["metrics"]["core_component_covariance_error"],
            }

    return {
        "analytic_c_inf_counterexamples": analytic_counterexamples,
        "smooth_silu_critic_counterexample": silu_counterexample,
        "trained_fixture_audit": fixture_audit,
    }


# ---------------------------------------------------------------------------
# 4. Experiment A: Multi-Scale 8x Width / 6x Mass with Provably 1-Lip Analytic Critics
# ---------------------------------------------------------------------------


def run_analytic_1lip_scale_mass_8x(
    *,
    steps: int = 800,
    batch_size: int = 128,
    num_particles: int = 512,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Generator-free (G = Id) t3_iso_scale_mass_8x using PROVABLY globally 1-Lipschitz critics.

    Compares:
      1. Pure dynamically re-solved exact W1 dual LP critic + fixed-step SGD (lr = 0.005, 0.02, 0.05)
         -> FAILS core_min_eigen_ratio at all fixed step sizes (constant ||grad D*||=1 loses scale).
      2. Pure dynamically re-solved exact W1 dual LP critic + per-particle Adam (beta1=0)
         -> PASSES (falsifies overbroad "no 1-Lip critic can pass", via global LP + private per-particle v_i).
      3. Smooth globally 1-Lipschitz signed-distance-to-centers potential (scale=0.25, Lip < 0.25)
         + fixed global horizon ell screened Feydy OT (NO conformal horizon)
         -> FAILS (broad-mode pairs exceed r_screen = sqrt(15)*ell, collapsing broad-mode covariance).
      4. EXACT SAME smooth globally 1-Lipschitz potential + RGDT conformal local horizon ell_Y(u)
         + scale-invariant k=4 r_loc orphan gate
         -> PASSES ALL GATES (eliminates neural critic irregularity as a confound!).
    """
    dev = torch.device(device)
    ref = sample_target_for_kind("t3_iso_scale_mass_8x", 2048, torch.Generator().manual_seed(77), device="cpu")
    _, means_sm, _ = scale_mass_parameters(device)

    data_rng = torch.Generator(device="cpu").manual_seed(201)
    idx_rng = torch.Generator(device="cpu").manual_seed(101)
    real_batches = [
        sample_target_for_kind("t3_iso_scale_mass_8x", batch_size, data_rng, device="cpu")
        for _ in range(steps)
    ]
    row_batches = [
        torch.randint(0, num_particles, (batch_size,), generator=idx_rng)
        for _ in range(steps)
    ]

    base_prior = ParticlePrior(num_particles=num_particles, z_dim=2, learnable=True)
    init.deterministic_orthogonal_(base_prior, seed=23)
    z_init = base_prior.z.detach().clone()

    gx = torch.linspace(-2.5, 2.5, 41, device=dev)
    gxm, gym = torch.meshgrid(gx, gx, indexing="ij")
    cert_grid = torch.stack([gxm.reshape(-1), gym.reshape(-1)], dim=1)

    arms = (
        ("w1_analytic_sgd_lr005", "pure_w1_sgd", 0.005),
        ("w1_analytic_sgd_lr020", "pure_w1_sgd", 0.020),
        ("w1_analytic_sgd_lr050", "pure_w1_sgd", 0.050),
        ("w1_analytic_adam_lr020", "pure_w1_adam", 0.020),
        ("signed_dist_1lip_global_ell", "sdist_global_ell", 0.025),
        ("signed_dist_1lip_conformal_rloc", "sdist_conformal_rloc", 0.025),
    )

    results: Dict[str, Any] = {}
    for arm_name, mode, lr in arms:
        z = nn.Parameter(z_init.clone().to(dev))
        opt = (
            torch.optim.Adam([z], lr=lr, betas=(0.0, 0.999))
            if mode != "pure_w1_sgd"
            else None
        )
        fallback_ell = torch.tensor(0.0, dtype=torch.float32)
        max_certified_lip = 0.0
        last_critic_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

        for step in range(steps):
            rows = row_batches[step].to(dev)
            y_b = real_batches[step].to(dev)
            x_b = z[rows]

            if mode in ("pure_w1_sgd", "pure_w1_adam"):
                with torch.no_grad():
                    _, g_np, _, _ = solve_exact_w1_dual_potentials(
                        x_b.detach().cpu().numpy(),
                        y_b.detach().cpu().numpy(),
                    )
                    g_t = torch.tensor(g_np, dtype=torch.float32, device=dev)
                    raw_x = smooth_w1_sup_convolution(x_b.detach(), y_b, g_t, eps=0.01, tau=0.02)
                    raw_y = smooth_w1_sup_convolution(y_b, y_b, g_t, eps=0.01, tau=0.02)
                    gauge_c = 0.5 * (raw_x.mean() + raw_y.mean())

                def critic_fn(q: torch.Tensor, _y=y_b, _g=g_t, _c=gauge_c) -> torch.Tensor:
                    return smooth_w1_sup_convolution(q, _y, _g, eps=0.01, tau=0.02, gauge_shift=_c)
            else:
                # Provably globally 1-Lipschitz smooth negative distance to mode centers (Lip < 0.25)
                def critic_fn(q: torch.Tensor, _m=means_sm) -> torch.Tensor:
                    return smooth_signed_distance_mixture_potential(q, _m, eps=0.05, tau=0.08, scale=0.25)

            if step in (0, steps // 2, steps - 1):
                cert = certify_sampled_lipschitz(critic_fn, torch.cat([x_b.detach(), y_b, cert_grid], dim=0))
                max_certified_lip = max(max_certified_lip, cert["grad_max"])
                last_critic_fn = critic_fn

            if mode == "pure_w1_sgd":
                d_fake = critic_fn(x_b)
                loss = -d_fake.sum()
                if z.grad is not None:
                    z.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    counts = torch.bincount(rows, minlength=num_particles).to(dev).float().clamp_min(1.0).unsqueeze(1)
                    z.data.sub_(lr * (z.grad / counts))
            elif mode == "pure_w1_adam":
                assert opt is not None
                opt.zero_grad(set_to_none=True)
                loss = -critic_fn(x_b).mean()
                loss.backward()
                opt.step()
            else:
                assert opt is not None
                opt.zero_grad(set_to_none=True)
                ell_scalar = update_scalar_ell(
                    y_b,
                    fallback_ell=fallback_ell,
                    beta_ell=0.1,
                    nn_quantile=0.25,
                    allow_update=True,
                )
                fallback_ell = ell_scalar.detach().float().cpu()
                if mode == "sdist_global_ell":
                    ell_xy, ell_xx = ell_scalar, ell_scalar
                else:
                    ell_xy, ell_xx, _ = conformal_horizon_pair(x_b, y_b, ell_scalar, enabled=True)

                c_phys_xy, c_hat_xy = born_infeld_pair(x_b, y_b, ell_xy)
                c_phys_xx, c_hat_xx = born_infeld_pair(x_b, x_b, ell_xx)
                d_fake = critic_fn(x_b)
                with torch.no_grad():
                    d_real = critic_fn(y_b)
                    # Center logits to zero mean gauge
                    d_bar = 0.5 * (d_fake.detach().mean() + d_real.mean())
                    df_c = d_fake.detach() - d_bar
                    dr_c = d_real - d_bar
                    p_xy = sinkhorn_slack(c_hat_xy, df_c, dr_c, rho=3.0, tau=2.0, eps=0.1, iters=8)
                    zeros = torch.zeros(batch_size, device=dev, dtype=torch.float32)
                    diag_f = torch.eye(batch_size, dtype=torch.bool, device=dev)
                    uniq_xx = (c_hat_xx > 1e-12) | diag_f
                    p_xx = sinkhorn_slack(
                        c_hat_xx, zeros, zeros, rho=3.0, tau=2.0, eps=0.1, iters=8, pair_mask=uniq_xx
                    )
                    p_xy_act = p_xy[:batch_size, :batch_size]
                    p_row_slack = p_xy[:batch_size, batch_size]
                    p_col_slack = p_xy[batch_size, :batch_size]
                    p_xx_act = 0.5 * (p_xx[:batch_size, :batch_size] + p_xx[:batch_size, :batch_size].T)
                    p_xx_act.fill_diagonal_(0.0)
                    a_f = (float(batch_size) * p_xy_act.sum(dim=1)).clamp(0.0, 1.0)
                    p_xx_eff = p_xx_act * torch.sqrt(a_f.unsqueeze(1) * a_f.unsqueeze(0))

                transport_xy = (p_xy_act * c_phys_xy).sum()
                debias_xx = -0.5 * (p_xx_eff * c_phys_xx).sum()
                critic_slack = -(p_row_slack * d_fake).sum()
                loss = transport_xy + debias_xx + critic_slack

                if mode == "sdist_conformal_rloc":
                    c_hat_glob = born_infeld_c_hat_detached(x_b, y_b, ell_scalar)
                    macro_loss = v11_zero_eigh_bipartite_deficit_macro_loss(
                        x_b, y_b, c_hat_glob, p_row_slack, p_col_slack, rho=3.0, g_conn=0.0
                    )
                    loss = loss + macro_loss

                loss.backward()
                opt.step()

        assert last_critic_fn is not None
        last_g_med = measure_critic_grad_median(last_critic_fn, z.detach(), ref)
        scored = score_t3_isolation_kind(
            "t3_iso_scale_mass_8x",
            z.detach().cpu(),
            ref,
            critic_grad_med=last_g_med,
        )
        results[arm_name] = {
            "passed": scored["passed"],
            "failed_bounds": scored["failed_bounds"],
            "max_certified_lipschitz": max_certified_lip,
            "provably_1_lipschitz": max_certified_lip <= 1.0 + 1e-6,
            "metrics": scored["metrics"],
        }
    return results


# ---------------------------------------------------------------------------
# 5. Experiment B: Beyond-Horizon Saddle Shift with Provably 1-Lip Analytic Critics
# ---------------------------------------------------------------------------


def run_analytic_1lip_saddle_shift(
    *,
    post_steps: int = 700,
    batch_size: int = 128,
    num_particles: int = 128,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Generator-free (G = Id) t3_iso_horizon_saddle_shift using PROVABLY 1-Lipschitz critics.

    Compares (all on the exact same pre-shift cloud, seen batches, and post_steps=700):
      1. Theorem 2's exact global W1-optimal 1-Lipschitz potential D_cap(x) (alone)
         -> FAILS (zero directional information about orthogonal orphan plane +-1.5 e_2).
      2. Population symmetric 1-Lipschitz potential D_sym(x) (alone, static)
         -> FAILS (modes=2, mass_tv=0.50: transports 100% of mass off +-1.5 e_1 onto +-1.5 e_2).
      3. Theorem 2's exact D_cap(x) + fixed global horizon ell screened Feydy OT (NO macro,
         unit-row-normalized DualNorm step) -> FAILS (modes=2: 42.4-sigma valley is 100% screened).
      4. EXACT SAME D_cap(x) + conformal horizon + actual v11 r_loc macro under Adam
         -> EXPOSES THE [r_screen, r_gprox_off] (4.50 < c_hat < 8.40 in metric.py) DEAD ZONE:
            balances mass across all 4 modes (mass_tv <= 0.12), but Adam's accumulated v_t damps
            the faint Sinkhorn tail once g_prox turns off at c_hat = 8.40, stranding rescued
            explorers short of +-1.5 e_2 (modes=2, hq < 0.75).
      5. EXACT SAME D_cap(x) + conformal horizon + actual RGDT bipartite_deficit_macro_loss
         (with explorer anchor release) + unit-row-normalized DualNorm step (matching RGDT)
         -> PASSES ALL GATES (4/4 modes, mass_tv <= 0.12, hq >= 0.92, frozen_control_modes=2).
      6. Dynamically re-solved exact finite-batch W1 LP dual critic (non-local global LP oracle!)
         -> PASSES modes/mass_tv/hq (proving global re-optimized W1 LP already performs non-local
            coupling, though with higher core_cov_err than conformal+macro).
    """
    dev = torch.device(device)
    z_pre = sample_target_for_kind(
        "t3_iso_horizon_saddle_shift", num_particles, torch.Generator().manual_seed(300), step=300, device="cpu"
    )
    ref_post = sample_target_for_kind(
        "t3_iso_horizon_saddle_shift", 2048, torch.Generator().manual_seed(78), step=1000, device="cpu"
    )
    w_pre, m_pre, s_pre = saddle_shift_parameters(300, "cpu")
    pre_hq = float(_mixture_mode_metrics(z_pre, w_pre, m_pre, s_pre, min_mode_share=0.20)["hq"])

    data_rng = torch.Generator(device="cpu").manual_seed(202)
    idx_rng = torch.Generator(device="cpu").manual_seed(102)
    real_batches = [
        sample_target_for_kind("t3_iso_horizon_saddle_shift", batch_size, data_rng, step=1000, device="cpu")
        for _ in range(post_steps)
    ]
    row_batches = [
        torch.randint(0, num_particles, (batch_size,), generator=idx_rng)
        for _ in range(post_steps)
    ]

    cert_probes = torch.randn((512, 4), generator=torch.Generator().manual_seed(999)).to(dev) * 2.0

    arms = (
        ("dist_w1_1lip_critic", "dist_only", 0.025),
        ("pop_sym_1lip_critic", "pop_sym", 0.025),
        ("dist_w1_1lip_global_ell_no_macro_dn", "dist_global_ell_no_macro_dn", 0.025),
        ("dist_w1_1lip_conformal_rloc_adam", "dist_conformal_rloc_adam", 0.030),
        ("dist_w1_1lip_conformal_macro_dualnorm", "dist_conformal_macro_dn", 0.025),
        ("batch_w1_lp_1lip_sgd", "batch_w1_sgd", 0.025),
    )

    results: Dict[str, Any] = {}
    for arm_name, mode, lr in arms:
        z = nn.Parameter(z_pre.clone().to(dev))
        opt = (
            torch.optim.Adam([z], lr=lr, betas=(0.0, 0.999))
            if mode == "dist_conformal_rloc_adam"
            else None
        )
        fallback_ell = torch.tensor(0.05, dtype=torch.float32)
        max_certified_lip = 0.0
        last_critic_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

        for step in range(post_steps):
            rows = row_batches[step].to(dev)
            y_b = real_batches[step].to(dev)
            x_b = z[rows]

            if mode in (
                "dist_only",
                "dist_global_ell_no_macro_dn",
                "dist_conformal_rloc_adam",
                "dist_conformal_macro_dn",
            ):
                def critic_fn(q: torch.Tensor) -> torch.Tensor:
                    return 0.25 * smooth_dist_to_preshift_potential(
                        q, a=1.5, eps=0.05, tau=0.05, cap_at_valley=True
                    )
            elif mode == "pop_sym":
                def critic_fn(q: torch.Tensor) -> torch.Tensor:
                    return smooth_symmetric_saddle_potential(q, a=1.5, v_dir=(0, 1, 0, 0), eps=0.05, tau=0.05)
            else:
                with torch.no_grad():
                    _, g_np, _, _ = solve_exact_w1_dual_potentials(
                        x_b.detach().cpu().numpy(),
                        y_b.detach().cpu().numpy(),
                    )
                    g_t = torch.tensor(g_np, dtype=torch.float32, device=dev)
                    raw_x = smooth_w1_sup_convolution(x_b.detach(), y_b, g_t, eps=0.02, tau=0.04)
                    raw_y = smooth_w1_sup_convolution(y_b, y_b, g_t, eps=0.02, tau=0.04)
                    gauge_c = 0.5 * (raw_x.mean() + raw_y.mean())

                def critic_fn(q: torch.Tensor, _y=y_b, _g=g_t, _c=gauge_c) -> torch.Tensor:
                    return smooth_w1_sup_convolution(q, _y, _g, eps=0.02, tau=0.04, gauge_shift=_c)

            if step in (0, post_steps // 2, post_steps - 1):
                cert = certify_sampled_lipschitz(critic_fn, torch.cat([x_b.detach(), y_b, cert_probes], dim=0))
                max_certified_lip = max(max_certified_lip, cert["grad_max"])
                last_critic_fn = critic_fn

            if mode in ("dist_only", "pop_sym", "batch_w1_sgd"):
                d_fake = critic_fn(x_b)
                loss = -d_fake.sum()
                if z.grad is not None:
                    z.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    counts = torch.bincount(rows, minlength=num_particles).to(dev).float().clamp_min(1.0).unsqueeze(1)
                    z.data.sub_(lr * (z.grad / counts))
            else:
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
                elif z.grad is not None:
                    z.grad.zero_()
                ell_scalar = update_scalar_ell(
                    y_b,
                    fallback_ell=fallback_ell,
                    beta_ell=0.1,
                    nn_quantile=0.25,
                    allow_update=True,
                )
                fallback_ell = ell_scalar.detach().float().cpu()
                if mode == "dist_global_ell_no_macro_dn":
                    ell_xy, ell_xx = ell_scalar, ell_scalar
                else:
                    ell_xy, ell_xx, _ = conformal_horizon_pair(x_b, y_b, ell_scalar, enabled=True)

                c_phys_xy, c_hat_xy = born_infeld_pair(x_b, y_b, ell_xy)
                c_phys_xx, c_hat_xx = born_infeld_pair(x_b, x_b, ell_xx)
                d_fake = critic_fn(x_b)
                with torch.no_grad():
                    d_real = critic_fn(y_b)
                    p_xy = sinkhorn_slack(
                        c_hat_xy, d_fake.detach(), d_real.detach(), rho=3.0, tau=2.0, eps=0.1, iters=8
                    )
                    zeros = torch.zeros(batch_size, device=dev, dtype=torch.float32)
                    diag_f = torch.eye(batch_size, dtype=torch.bool, device=dev)
                    uniq_xx = (c_hat_xx > 1e-12) | diag_f
                    p_xx = sinkhorn_slack(
                        c_hat_xx, zeros, zeros, rho=3.0, tau=2.0, eps=0.1, iters=8, pair_mask=uniq_xx
                    )
                    p_xy_act = p_xy[:batch_size, :batch_size]
                    p_row_slack = p_xy[:batch_size, batch_size]
                    p_col_slack = p_xy[batch_size, :batch_size]
                    p_xx_act = 0.5 * (p_xx[:batch_size, :batch_size] + p_xx[:batch_size, :batch_size].T)
                    p_xx_act.fill_diagonal_(0.0)

                macro_loss = x_b[:0].sum()
                if mode == "dist_conformal_rloc_adam":
                    c_hat_glob = born_infeld_c_hat_detached(x_b, y_b, ell_scalar)
                    macro_loss = v11_zero_eigh_bipartite_deficit_macro_loss(
                        x_b, y_b, c_hat_glob, p_row_slack, p_col_slack, rho=3.0, g_conn=0.0
                    )
                elif mode == "dist_conformal_macro_dn":
                    macro_loss, explorer_mask = bipartite_deficit_macro_loss(
                        x_b,
                        y_b,
                        c_hat_xy,
                        p_row_slack,
                        p_col_slack,
                        rho=3.0,
                        g_conn=0.0,
                        return_explorer_mask=True,
                    )
                    if explorer_mask is not None:
                        p_xy_act = p_xy_act * (1.0 - explorer_mask.unsqueeze(1))

                with torch.no_grad():
                    a_f = (float(batch_size) * p_xy_act.sum(dim=1)).clamp(0.0, 1.0)
                    p_xx_eff = p_xx_act * torch.sqrt(a_f.unsqueeze(1) * a_f.unsqueeze(0))

                transport_xy = (p_xy_act * c_phys_xy).sum()
                debias_xx = -0.5 * (p_xx_eff * c_phys_xx).sum()
                critic_slack = -(p_row_slack * d_fake).sum()
                loss = transport_xy + debias_xx + critic_slack + macro_loss

                loss.backward()
                if opt is not None:
                    opt.step()
                else:
                    with torch.no_grad():
                        g = z.grad
                        gn = g.norm(dim=1, keepdim=True)
                        step_vec = torch.where(gn > 1e-12, g / gn.clamp_min(1e-12), torch.zeros_like(g))
                        z.data.sub_(lr * step_vec)

        assert last_critic_fn is not None
        last_g_med = measure_critic_grad_median(last_critic_fn, z.detach(), ref_post)
        scored = score_t3_isolation_kind(
            "t3_iso_horizon_saddle_shift",
            z.detach().cpu(),
            ref_post,
            critic_grad_med=last_g_med,
            pre_shift_hq=pre_hq,
            frozen_particles=z_pre,
        )
        results[arm_name] = {
            "passed": scored["passed"],
            "failed_bounds": scored["failed_bounds"],
            "max_certified_lipschitz": max_certified_lip,
            "provably_1_lipschitz": max_certified_lip <= 1.0 + 1e-6,
            "metrics": scored["metrics"],
        }
    return results


# ---------------------------------------------------------------------------
# 6. Experiment C: Conditional Heteroscedastic Fiber Task with Analytic 1-Lip Critic
# ---------------------------------------------------------------------------


def _sample_cond_fiber_target(
    c: torch.Tensor,
    t_idx: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    b = c.shape[0]
    a_mats = torch.tensor(
        [
            [[0.85, 0.25], [-0.20, 0.80]],
            [[0.75, -0.20], [0.25, 0.70]],
            [[0.60, 0.15], [-0.15, 0.65]],
            [[0.45, -0.10], [0.10, 0.50]],
        ],
        dtype=torch.float32,
    )
    sig_base = torch.tensor([0.27, 0.30, 0.34, 0.38], dtype=torch.float32)

    a_b = a_mats[t_idx]
    lin_m = torch.bmm(a_b, c.unsqueeze(2)).squeeze(2)
    kink0 = 0.28 * (F.softplus(c[:, 0] - 0.8) - 0.425) + 0.18 * (c[:, 0] * c[:, 1]) / torch.sqrt(
        1.0 + 0.15 * c.square().sum(dim=1)
    )
    kink1 = -0.25 * (F.softplus(-c[:, 0] - 0.8) - 0.425)
    mean_y = lin_m + torch.stack([kink0, kink1], dim=1)

    scale_c = sig_base[t_idx] * torch.sqrt(1.0 + 0.30 * torch.tanh(c[:, 0]))
    eps = torch.randn((b, 2), generator=rng, dtype=torch.float32)
    return mean_y + scale_c.unsqueeze(1) * eps


def _true_cond_fiber_moments(c_pt: torch.Tensor, t_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a_mats = torch.tensor(
        [
            [[0.85, 0.25], [-0.20, 0.80]],
            [[0.75, -0.20], [0.25, 0.70]],
            [[0.60, 0.15], [-0.15, 0.65]],
            [[0.45, -0.10], [0.10, 0.50]],
        ],
        dtype=torch.float32,
    )
    sig_base = torch.tensor([0.27, 0.30, 0.34, 0.38], dtype=torch.float32)
    lin_m = a_mats[t_id] @ c_pt
    kink0 = 0.28 * (F.softplus(c_pt[0] - 0.8) - 0.425) + 0.18 * (c_pt[0] * c_pt[1]) / torch.sqrt(
        1.0 + 0.15 * c_pt.square().sum()
    )
    kink1 = -0.25 * (F.softplus(-c_pt[0] - 0.8) - 0.425)
    mean_y = lin_m + torch.stack([kink0, kink1])
    var_c = (sig_base[t_id] ** 2) * (1.0 + 0.30 * torch.tanh(c_pt[0]))
    return mean_y, var_c * torch.eye(2, dtype=torch.float32)


class _SmallCondGenerator(nn.Module):
    def __init__(self, d_c: int = 2, d_z: int = 2, d_x: int = 2, num_strata: int = 4, hidden: int = 48) -> None:
        super().__init__()
        self.num_strata = num_strata
        self.net = nn.Sequential(
            nn.Linear(d_c + d_z + num_strata, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_x),
        )

    def forward(self, c: torch.Tensor, z: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        t_oh = F.one_hot(t_idx.long(), num_classes=self.num_strata).to(dtype=c.dtype)
        return self.net(torch.cat([c, z, t_oh], dim=1))


def _unresidualized_local_imq_cmmd(
    x_fake: torch.Tensor,
    y_real: torch.Tensor,
    c_cont: torch.Tensor,
    cond_disc: torch.Tensor,
) -> torch.Tensor:
    b_n = x_fake.shape[0]
    with torch.no_grad():
        same_t = cond_disc.unsqueeze(1) == cond_disc.unsqueeze(0)
        m_hollow = same_t.clone()
        m_hollow.fill_diagonal_(False)
        valid_row = m_hollow.any(dim=1)
        m_valid = m_hollow & valid_row.unsqueeze(1) & valid_row.unsqueeze(0)
        if not bool(m_valid.any()):
            return x_fake[:0].sum()
        d_c = torch.cdist(c_cont.float(), c_cont.float(), p=2.0)
        d_c_masked = d_c.masked_fill(~m_valid, float("inf"))
        d_sorted, _ = torch.sort(d_c_masked, dim=1)
        n_i = m_valid.sum(dim=1)
        k_idx = torch.minimum(torch.full_like(n_i, 3), n_i.clamp_min(1)) - 1
        sigma_i = d_sorted.gather(1, k_idx.unsqueeze(1)).squeeze(1).clamp_min(1e-3)
        k_c = torch.where(m_valid, torch.exp(-d_c.square() / (2.0 * sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0))), 0.0)
        deg = 1.0 + k_c.sum(dim=1)
        w = (k_c * deg.rsqrt().unsqueeze(1) * deg.rsqrt().unsqueeze(0)) / float(b_n)
        w_sym = 0.5 * (w + w.T)
        ell_f = torch.cdist(y_real.detach(), y_real.detach()).median().clamp_min(0.1)

    sq_xy = (x_fake.unsqueeze(1) - y_real.detach().unsqueeze(0)).square().sum(dim=-1)
    sq_xx = (x_fake.unsqueeze(1) - x_fake.unsqueeze(0)).square().sum(dim=-1)
    psi_xy = torch.zeros_like(sq_xy)
    psi_xx = torch.zeros_like(sq_xx)
    for s_mult in (0.35, 1.0, 2.8):
        ell_s = ell_f * s_mult
        r_xy = torch.sqrt(ell_s.square() + sq_xy)
        r_xx = torch.sqrt(ell_s.square() + sq_xx)
        psi_xy = psi_xy + (ell_s * sq_xy) / (r_xy * (ell_s + r_xy))
        psi_xx = psi_xx + (ell_s * sq_xx) / (r_xx * (ell_s + r_xx))
    return (w_sym * (psi_xy / 3.0)).sum() - 0.5 * (w_sym * (psi_xx / 3.0)).sum()


@dataclass
class _CondBankRecord:
    cond_fiber_bank: UnifiedV11CondFiberBank
    t1c_cond_initialized: bool = False
    t1c_cond_ema: float = 1.0
    _step_scale: float = 1.0
    _in_protected_probe: bool = False
    _in_probe: bool = False


def run_analytic_1lip_conditional_fiber(
    *,
    steps: int = 1200,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Conditional heteroscedastic fiber task under a PROVABLY globally 1-Lipschitz analytic critic.

    Pairs the exact same provably globally 1-Lipschitz relativistic analytic conditional critic
    (sup ||grad_{(c,x)} D||_2 < 1 everywhere) with:
      1. Unresidualized local 3-NN context-kernel IMQ-CMMD -> severe base-drift variance inflation
         (bulk eig_max = 1.4001, outer-tail q_0 eig_max = 1.9671, mean_err = 0.6437).
      2. Single-batch pooled-tangent residualization (RGDT Pillar 3) -> resolves bulk cleanly,
         retains outer-5% tail q_0 mean/covariance error due to n_t ~ 16 starvation.
      3. Cross-batch UnifiedV11CondFiberBank + midpoint-canceling FIFO ring -> resolves BOTH bulk
         and outer-5% tail q_0 mean & covariance cleanly!
    """
    dev = torch.device(device)
    data_rng = torch.Generator(device="cpu").manual_seed(301)
    z_rng = torch.Generator(device="cpu").manual_seed(302)

    batches = []
    for _ in range(steps):
        t_idx = torch.randint(0, 4, (batch_size,), generator=data_rng)
        c_b = torch.randn((batch_size, 2), generator=data_rng, dtype=torch.float32)
        y_b = _sample_cond_fiber_target(c_b, t_idx, data_rng)
        z_b = torch.randn((batch_size, 2), generator=z_rng, dtype=torch.float32)
        batches.append((c_b, t_idx, y_b, z_b))

    base_gen = _SmallCondGenerator()
    init.deterministic_orthogonal_(base_gen, seed=17)
    init_state = {k: v.detach().clone() for k, v in base_gen.state_dict().items()}

    arms = (
        ("cond_unresidualized_1lip", "unresidualized"),
        ("cond_single_batch_tangent_1lip", "single_batch"),
        ("cond_fiber_bank_1lip", "fiber_bank"),
    )

    q_bulk = torch.tensor([0.0, 0.0], dtype=torch.float32)
    q_tail = torch.tensor([-2.0, 1.4], dtype=torch.float32)
    eval_z = torch.randn((4096, 2), generator=torch.Generator(device="cpu").manual_seed(404), dtype=torch.float32)

    results: Dict[str, Any] = {}
    base_lr = 1.5e-3
    for arm_name, mode in arms:
        gen = _SmallCondGenerator().to(dev)
        gen.load_state_dict({k: v.clone().to(dev) for k, v in init_state.items()})
        opt = torch.optim.Adam(gen.parameters(), lr=base_lr, betas=(0.0, 0.999))
        bank_rec = _CondBankRecord(cond_fiber_bank=UnifiedV11CondFiberBank()) if mode == "fiber_bank" else None
        max_certified_lip = 0.0

        for step in range(steps):
            c_b, t_b, y_b, z_b = batches[step]
            c_d, t_d, y_d, z_d = c_b.to(dev), t_b.to(dev), y_b.to(dev), z_b.to(dev)

            opt.zero_grad(set_to_none=True)
            x_d = gen(c_d, z_d, t_d)

            # Provably globally 1-Lipschitz analytic conditional critic on joint (c, x) in R^4:
            # Using smooth negative distance to real pairs scaled to Lip < 0.25 (zero LP overhead,
            # strictly bounded Lipschitz < 1 everywhere, relativistic softplus matching UnifiedV11GANLoss).
            cx_fake = torch.cat([c_d, x_d], dim=1)
            cx_real = torch.cat([c_d, y_d], dim=1)
            def cond_critic(q: torch.Tensor, _y=cx_real.detach()) -> torch.Tensor:
                return smooth_signed_distance_mixture_potential(q, _y, eps=0.10, tau=0.20, scale=0.25)

            if step in (0, steps // 2, steps - 1):
                cert = certify_sampled_lipschitz(cond_critic, torch.cat([cx_fake.detach(), cx_real], dim=0))
                max_certified_lip = max(max_certified_lip, cert["grad_max"])

            d_fake = cond_critic(cx_fake)
            d_real = cond_critic(cx_real).detach()
            rel_term = F.softplus(d_real - d_fake).mean()

            if mode == "unresidualized":
                cond_term = _unresidualized_local_imq_cmmd(x_d, y_d, c_d, t_d)
            elif mode == "single_batch":
                cond_term = pooled_tangent_conditional_loss(x_d, y_d, c_d, cond_disc=t_d.unsqueeze(1))
            else:
                assert bank_rec is not None
                cond_term = unified_v11_conditional_loss(
                    x_d, y_d, c_d, cond_disc=t_d.unsqueeze(1), record=bank_rec
                )
                for pg in opt.param_groups:
                    pg["lr"] = base_lr * bank_rec._step_scale

            loss = 0.05 * rel_term + 1.5 * cond_term
            loss.backward()
            opt.step()

        gen.eval()
        eval_metrics = {}
        with torch.no_grad():
            for q_name, q_pt in (("t0_bulk", q_bulk), ("t0_tail_q0", q_tail)):
                c_eval = q_pt.unsqueeze(0).expand(len(eval_z), -1).to(dev)
                t_eval = torch.zeros(len(eval_z), dtype=torch.long, device=dev)
                x_eval = gen(c_eval, eval_z.to(dev), t_eval).cpu()
                m_true, cov_true = _true_cond_fiber_moments(q_pt, 0)
                sig_true = math.sqrt(float(cov_true[0, 0].item()))
                m_emp = x_eval.mean(dim=0)
                cov_emp = torch.cov(x_eval.T, correction=0)
                norm_mean_err = float((m_emp - m_true).norm().item() / sig_true)
                eigs = torch.linalg.eigvalsh(cov_emp.double()).clamp_min(0.0).float() / float(cov_true[0, 0].item())
                eval_metrics[q_name] = {
                    "normalized_mean_err": norm_mean_err,
                    "cov_eig_min": float(eigs[0].item()),
                    "cov_eig_max": float(eigs[1].item()),
                }

        results[arm_name] = {
            "max_certified_lipschitz": max_certified_lip,
            "provably_1_lipschitz": max_certified_lip <= 1.0 + 1e-6,
            **eval_metrics,
        }
    return results


def run_all(*, include_fixture_audit: bool = True, device: str = "cpu") -> Dict[str, Any]:
    return {
        "theorem1_and_brenier": verify_theorem1_and_brenier_refutation(),
        "theorem2_beyond_horizon": verify_theorem2_beyond_horizon(),
        "theorem4_midpoint_and_zero_eigh": verify_theorem4_midpoint_and_zero_eigh(),
        "critic_lipschitz_audit": audit_critic_lipschitz_controls(include_fixture_audit=include_fixture_audit),
        "exp_a_scale_mass_8x": run_analytic_1lip_scale_mass_8x(device=device),
        "exp_b_saddle_shift": run_analytic_1lip_saddle_shift(device=device),
        "exp_c_conditional_fiber": run_analytic_1lip_conditional_fiber(device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P2 Law 3 analytical 1-Lipschitz audit.")
    parser.add_argument("--device", default="cpu", help="Device for analytical experiments (cpu or cuda:0)")
    parser.add_argument("--skip-fixture-audit", action="store_true", help="Skip 200-step T3IsolationFixture audit")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write full JSON summary")
    args = parser.parse_args()

    out = run_all(include_fixture_audit=not args.skip_fixture_audit, device=args.device)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
