"""Reproducible mathematical proofs, counterexamples, and analytical 1-Lipschitz tests for P2 (Law 3).

Tests every formal theorem, algebraic identity, critic Lipschitz counterexample,
and generator-free (G = Id) provably globally 1-Lipschitz experiment reported in
``reports/four-laws-audit/p2-law3-identifiability.md``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from scripts.four_laws_audit.p2_law3_analytic_lipschitz import (
    audit_critic_lipschitz_controls,
    certify_sampled_lipschitz,
    run_analytic_1lip_conditional_fiber,
    run_analytic_1lip_saddle_shift,
    run_analytic_1lip_scale_mass_8x,
    smooth_dist_to_preshift_potential,
    smooth_signed_distance_mixture_potential,
    smooth_symmetric_saddle_potential,
    smooth_w1_sup_convolution,
    solve_exact_w1_dual_potentials,
    verify_theorem1_and_brenier_refutation,
    verify_theorem2_beyond_horizon,
    verify_theorem4_midpoint_and_zero_eigh,
)


def test_exact_w1_dual_solver_and_global_1_lipschitz_certificate():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(64, 3))
    y = rng.normal(1.2, 0.4, size=(64, 3))
    f, g, primal, gap = solve_exact_w1_dual_potentials(x, y)
    assert primal > 0.5
    assert gap < 1e-12
    assert abs(0.5 * (float(f.mean()) + float(g.mean()))) < 1e-12

    y_t = torch.tensor(y, dtype=torch.float32)
    g_t = torch.tensor(g, dtype=torch.float32)
    probes = torch.randn((512, 3), generator=torch.Generator().manual_seed(1)) * 5.0
    cert = certify_sampled_lipschitz(
        lambda q: smooth_w1_sup_convolution(q, y_t, g_t, eps=0.01, tau=0.02),
        probes,
    )
    assert cert["grad_max"] <= 1.0 + 1e-6


def test_theorem1_w1_nonidentifiability_and_brenier_refutation():
    res = verify_theorem1_and_brenier_refutation()
    # Both 8x-width targets and mass-vs-distance trade-off target share the exact same
    # global zero-mean-gauged 1-Lipschitz Kantorovich potential D*(x) = x + 0.5
    for key in ("pair_1a_narrow", "pair_1a_wide_8x", "pair_1b_mass_distance"):
        entry = res[key]
        assert abs(entry["w1_lp"] - 1.0) < 1e-12
        assert entry["duality_gap"] < 1e-12
        assert abs(entry["zero_mean_gauge"]) < 1e-12
    # Distinct target variances despite identical W1 potential everywhere
    assert res["pair_1a_wide_8x"]["target_var"] > res["pair_1a_narrow"]["target_var"] + 0.015
    assert res["pair_1b_mass_distance"]["target_var"] > 2.5 * res["pair_1a_narrow"]["target_var"]

    # Brenier W2 scalar potential refutes the naive overclaim ("scalar potentials cannot encode geometry")
    for label in ("narrow_0.04", "wide_0.32"):
        b = res["brenier_w2_refutation"][label]
        assert b["rel_std_err"] < 0.01
        assert b["violates_1_lipschitz"] is True
        assert b["sup_grad_phi"] > 2.0


def test_theorem2_beyond_horizon_and_saddle_nonidentifiability():
    res = verify_theorem2_beyond_horizon()
    # Orthogonal planes +-1.5 e_2 and +-1.5 e_3 yield exact zero duality gap under the SAME
    # global 1-Lipschitz potential D*(x) = min(||x - a e_1||, ||x + a e_1||) - sqrt(2)*a/4
    for plane in ("P1_e2_plane", "P2_e3_plane"):
        entry = res["global_w1_planes"][plane]
        assert abs(entry["w1_lp"] - math.sqrt(2.0) * 1.5 / 2.0) < 1e-12
        assert entry["duality_gap"] < 1e-12
        assert abs(entry["zero_mean_gauge"]) < 1e-12

    # Smooth symmetrized potential has exact zero orthogonal gradient on the entire e_1 axis
    assert res["sym_potential_e2_vs_e3_max_axis_grad_diff"] < 1e-14
    assert res["sym_potential_max_orthogonal_grad_on_e1_axis"] < 1e-14
    assert res["sym_potential_origin_grad_norm"] < 1e-12
    assert res["sym_potential_axis_max_grad_norm"] < 1.0

    # Fixed-horizon Sinkhorn screens 100% of 42.4-sigma orphan pairs into slack
    # and exhibits exact [r_screen, r_gprox_off] dead zones:
    #   metric.py (1.5*rho=4.50 -> 2.8*rho=8.40): 1.728x
    #   primitive.py rapidity_screening=True (3.8923 -> 8.40): 1.952x
    #   primitive.py rapidity_screening=False (rho=3.00 -> 8.40): 2.413x
    scr = res["fixed_horizon_valley_screening"]
    assert scr["orphan_min_c_hat"] > 30.0
    assert scr["orphan_active_mass"] < 1e-12
    assert scr["orphan_slack_mass"] > 0.40
    assert abs(scr["dead_zone_ratio_metric_py"] - 1.728198) < 1e-4
    assert abs(scr["dead_zone_ratio_primitive_rap"] - 1.951678) < 1e-4
    assert abs(scr["dead_zone_ratio_primitive_lin"] - 2.413297) < 1e-4


def test_theorem4_generalized_midpoint_cancellation_and_zero_eigh():
    res = verify_theorem4_midpoint_and_zero_eigh()
    # Exact midpoint cancellation holds ONLY at alpha = 0.5; unequal weight leaves (1 - 2*alpha) Delta Delta^T
    assert res["unequal_weight_checks"]["alpha_0.50"]["bias_frob_norm"] < 1e-14
    for key in ("alpha_0.00", "alpha_0.25", "alpha_0.75", "alpha_1.00"):
        entry = res["unequal_weight_checks"][key]
        assert entry["bias_frob_norm"] > 2.0
        assert entry["general_formula_err"] < 1e-14
        assert entry["alpha_formula_err"] < 1e-14

    # Basis misspecification / ridge shrinkage leaves exact delta_m Delta^T + Delta delta_m^T bias
    assert res["misspec_bias_frob_norm"] > 1.0
    assert res["misspec_formula_err"] < 1e-14

    # Finite-sample in-sample hat-matrix coupling attenuates (Sigma_X - Sigma_Y) by (1 - h_jj)
    # while preserving exact zero bias at Sigma_X == Sigma_Y!
    assert res["insample_hat_leverage_h_jj"] > 0.3
    assert res["insample_mc_vs_attenuated_max_err"] < 0.01
    assert res["insample_mc_vs_unattenuated_max_err"] > 0.05
    assert res["insample_equal_cov_zero_bias_max_err"] < 0.015

    # Unresidualized context smoothing inflates conditional covariance eigenvalues by > 3x
    assert res["unresidualized_eig_inflation_ratio"] > 3.0

    # zero_eigh_symmetric achieves < 1e-14 accuracy across d <= 16, but lacks column-sign canonicalization
    for d_key, audit in res["zero_eigh_symmetric_audit"].items():
        assert audit["max_eigval_err"] < 1e-14, d_key
        assert audit["recon_frob_err"] < 1e-14, d_key
        assert audit["ortho_err"] < 1e-14, d_key
    assert res["zero_eigh_symmetric_audit"]["d_2"]["has_negative_leading_col_sign"] is True


def test_critic_lipschitz_median_gate_counterexamples():
    # Fast CPU-only check of analytic C^inf and SmoothSiLUCritic counterexamples
    res = audit_critic_lipschitz_controls(include_fixture_audit=False)
    for m_key, m_val in (("M_10", 10.0), ("M_100", 100.0), ("M_1000", 1000.0)):
        c = res["analytic_c_inf_counterexamples"][m_key]
        assert c["passes_t3_gate_le_1"] is True
        assert c["measure_critic_grad_median"] <= 0.10
        assert c["grad_norm_at_mu0"] >= m_val

    silu = res["smooth_silu_critic_counterexample"]
    assert silu["passes_t3_gate_le_1"] is True
    assert silu["measure_critic_grad_median"] <= 0.30
    assert silu["grad_norm_at_mu3"] >= 100.0
    assert silu["ref_grad_max"] >= 100.0


def test_all_analytical_critics_are_provably_globally_1_lipschitz():
    probes_2d = torch.randn((1024, 2), generator=torch.Generator().manual_seed(11)) * 4.0
    probes_4d = torch.randn((1024, 4), generator=torch.Generator().manual_seed(12)) * 4.0
    means = torch.tensor([[-1.2, -1.2], [-1.2, 1.2], [1.2, -1.2], [1.2, 1.2]])

    c1 = certify_sampled_lipschitz(
        lambda q: smooth_signed_distance_mixture_potential(q, means, eps=0.05, tau=0.08, scale=0.25),
        probes_2d,
    )
    c2 = certify_sampled_lipschitz(
        lambda q: smooth_dist_to_preshift_potential(q, a=1.5, eps=0.05, tau=0.05, cap_at_valley=True),
        probes_4d,
    )
    c3 = certify_sampled_lipschitz(
        lambda q: smooth_symmetric_saddle_potential(q, a=1.5, v_dir=(0, 1, 0, 0), eps=0.05, tau=0.05),
        probes_4d,
    )
    assert c1["grad_max"] <= 0.25 + 1e-6
    assert c2["grad_max"] <= 1.0 + 1e-6
    assert c3["grad_max"] <= 1.0 + 1e-6


def test_analytic_1lip_experiments_a_b_c():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Experiment A: 8x scale-mass with provably globally 1-Lipschitz analytic critics
    exp_a = run_analytic_1lip_scale_mass_8x(steps=800, device=device)
    for arm, data in exp_a.items():
        assert data["provably_1_lipschitz"] is True, arm
    # Fixed-step pure W1 SGD fails core_min_eigen_ratio at all three step sizes
    for arm in ("w1_analytic_sgd_lr005", "w1_analytic_sgd_lr020", "w1_analytic_sgd_lr050"):
        assert exp_a[arm]["passed"] is False, arm
    # Per-particle Adam on dynamically re-solved global W1 LP critic PASSES (falsifies naive overclaim!)
    assert exp_a["w1_analytic_adam_lr020"]["passed"] is True
    # Under identical smooth globally 1-Lipschitz critic: global_ell FAILS badly while conformal+rloc PASSES!
    assert exp_a["signed_dist_1lip_global_ell"]["passed"] is False
    assert exp_a["signed_dist_1lip_global_ell"]["metrics"]["core_min_eigen_ratio"] < 0.05
    assert exp_a["signed_dist_1lip_conformal_rloc"]["passed"] is True
    assert exp_a["signed_dist_1lip_conformal_rloc"]["metrics"]["core_min_eigen_ratio"] >= 0.45

    # Experiment B: 42.4-sigma orthogonal saddle shift with provably globally 1-Lipschitz analytic critics
    exp_b = run_analytic_1lip_saddle_shift(post_steps=700, device=device)
    for arm, data in exp_b.items():
        assert data["provably_1_lipschitz"] is True, arm
    assert exp_b["dist_w1_1lip_critic"]["passed"] is False
    assert exp_b["pop_sym_1lip_critic"]["passed"] is False
    assert exp_b["pop_sym_1lip_critic"]["metrics"]["modes"] == 2
    assert exp_b["dist_w1_1lip_global_ell_no_macro_dn"]["passed"] is False
    assert exp_b["dist_w1_1lip_global_ell_no_macro_dn"]["metrics"]["modes"] == 2
    # Exposes the [4.50, 8.40] metric.py c_hat dead zone under Adam (balanced mass, stranded short of +-1.5 e_2)
    assert exp_b["dist_w1_1lip_conformal_rloc_adam"]["passed"] is False
    assert exp_b["dist_w1_1lip_conformal_rloc_adam"]["metrics"]["mass_tv"] <= 0.12
    # Exact same D_cap + conformal + RGDT macro + unit-row DualNorm PASSES ALL GATES!
    assert exp_b["dist_w1_1lip_conformal_macro_dualnorm"]["passed"] is True
    assert exp_b["dist_w1_1lip_conformal_macro_dualnorm"]["metrics"]["modes"] == 4
    assert exp_b["batch_w1_lp_1lip_sgd"]["passed"] is True

    # Experiment C: Conditional heteroscedastic fiber task under provably globally 1-Lipschitz analytic critic
    exp_c = run_analytic_1lip_conditional_fiber(steps=1200, device=device)
    for arm, data in exp_c.items():
        assert data["provably_1_lipschitz"] is True, arm
    # Unresidualized local context kernel suffers severe outer-tail covariance inflation (> 1.7)
    assert exp_c["cond_unresidualized_1lip"]["t0_tail_q0"]["cov_eig_max"] > 1.70
    # Single-batch tangent residualization fixes bulk, still fails outer-5% tail q0
    assert exp_c["cond_single_batch_tangent_1lip"]["t0_bulk"]["cov_eig_max"] < 1.25
    assert exp_c["cond_single_batch_tangent_1lip"]["t0_tail_q0"]["cov_eig_max"] > 1.40
    # Cross-batch UnifiedV11CondFiberBank resolves BOTH bulk and outer-5% tail q0!
    assert exp_c["cond_fiber_bank_1lip"]["t0_tail_q0"]["cov_eig_max"] < 1.30
    assert exp_c["cond_fiber_bank_1lip"]["t0_tail_q0"]["cov_eig_min"] > 0.65
    assert (
        exp_c["cond_fiber_bank_1lip"]["t0_tail_q0"]["normalized_mean_err"]
        < exp_c["cond_single_batch_tangent_1lip"]["t0_tail_q0"]["normalized_mean_err"] - 0.15
    )
