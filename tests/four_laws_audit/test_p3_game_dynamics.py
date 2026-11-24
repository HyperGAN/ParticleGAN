"""P3 Adversarial Audit Regression & Counterexample Tests.

Verifies every mathematical theorem and empirical counterexample from
`reports/four-laws-audit/p3-game-dynamics-counterexample.md` and
`scripts/four_laws_audit/p3_game_dynamics.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.four_laws_audit.p3_game_dynamics import (
    alt_gd_matrix,
    alt_momentum_companion,
    audit_particlegan_penalties,
    build_damped_bilinear_jacobian,
    estimate_kreiss_and_transient,
    ogda_companion,
    run_analytic_game,
    run_nonlinear_mlp_game,
    schur_cohn_eta_max,
    sim_gd_matrix,
    sim_momentum_companion,
    spectral_radius,
)


def test_symmetric_antisymmetric_and_sim_vs_alt_exact() -> None:
    """Verify purely imaginary spectrum, simultaneous expansion, and exact alternating det = 1 - eta_d * mu_d."""
    # Undamped bilinear game: J_F is purely antisymmetric, eigenvalues +/- i
    J0 = build_damped_bilinear_jacobian(mu_d=0.0, omega=1.0, rho=1.0)
    S0 = 0.5 * (J0 + J0.T)
    A0 = 0.5 * (J0 - J0.T)
    assert np.allclose(S0, 0.0)
    assert np.allclose(A0 + A0.T, 0.0)
    eigs0 = np.linalg.eigvals(J0)
    assert np.allclose(np.real(eigs0), 0.0, atol=1e-12)
    assert np.allclose(np.abs(np.imag(eigs0)), 1.0, atol=1e-12)

    # Simultaneous Euler GD expands strictly for EVERY eta > 0
    for eta in (1e-4, 0.014, 0.05, 0.2):
        rho_sim = spectral_radius(sim_gd_matrix(J0, eta))
        expected = math.sqrt(1.0 + eta * eta)
        assert math.isclose(rho_sim, expected, rel_tol=1e-12)
        assert rho_sim > 1.0

    # Alternating 1:1 D-then-G GD is exact symplectic (det = 1, rho = 1) at mu_d = 0
    M_alt0 = alt_gd_matrix(mu_d=0.0, eta_g=0.014, eta_d=0.021)
    assert math.isclose(float(np.linalg.det(M_alt0)), 1.0, abs_tol=1e-12)
    assert math.isclose(spectral_radius(M_alt0), 1.0, abs_tol=1e-12)

    # Weak damping mu_d = 0.01 (unified_v11/v12 lambda_floor = 0.005):
    # Simultaneous GD diverges at eta = 0.014 (> mu_d = 0.01), while alternating GD contracts!
    J_weak = build_damped_bilinear_jacobian(mu_d=0.01, omega=1.0, rho=1.0)
    M_sim_weak = np.eye(2) - np.diag([0.014, 0.021]) @ J_weak
    M_alt_weak = alt_gd_matrix(mu_d=0.01, eta_g=0.014, eta_d=0.021)
    assert spectral_radius(M_sim_weak) > 1.0
    assert math.isclose(float(np.linalg.det(M_alt_weak)), 1.0 - 0.021 * 0.01, abs_tol=1e-12)
    assert math.isclose(spectral_radius(M_alt_weak), math.sqrt(1.0 - 0.021 * 0.01), abs_tol=1e-12)
    assert spectral_radius(M_alt_weak) < 1.0


def test_non_normality_and_kreiss_transient_amplification() -> None:
    """Verify capacity mismatch rho inflates Kreiss transient amplification with identical contracting spectrum."""
    spec_rads = []
    peaks = []
    for rho_cap in (1.0, 10.0, 100.0, 500.0):
        J = build_damped_bilinear_jacobian(mu_d=0.5, omega=1.0, rho=rho_cap)
        M = sim_gd_matrix(J, eta=0.1)
        kt = estimate_kreiss_and_transient(M, max_power=150)
        spec_rads.append(kt["spectral_radius"])
        peaks.append(kt["peak_transient_norm"])
        # Kreiss Matrix Theorem lower & upper bounds: K(M) <= sup_k ||M^k||_2 <= e * n * K(M)
        assert kt["kreiss_constant"] <= kt["peak_transient_norm"] + 1e-6
        assert kt["peak_transient_norm"] <= math.e * 2.0 * kt["kreiss_constant"] + 1e-6

    # Identical contracting eigenvalues across all rho_cap!
    assert all(r < 0.985 for r in spec_rads)
    assert np.allclose(spec_rads, spec_rads[0], atol=1e-10)
    # Yet transient amplification explodes linearly with rho_cap (> 300x at rho=500)!
    assert peaks[0] < 1.1
    assert peaks[-1] > 300.0

    # Shifted Jordan block with spectral radius 0.98 < 1 exhibits > 90x transient blowup
    J_shift = np.array([[0.2, 50.0], [0.0, 0.2]], dtype=np.float64)
    kt_shift = estimate_kreiss_and_transient(sim_gd_matrix(J_shift, eta=0.1), max_power=150)
    assert kt_shift["spectral_radius"] < 0.981
    assert kt_shift["peak_transient_norm"] > 90.0
    assert kt_shift["kreiss_constant"] > 50.0


def test_schur_cohn_momentum_stability_and_alternating_destabilization() -> None:
    """Verify exact closed-form Schur-Cohn boundary and positive momentum destabilization of rotation."""
    # Exact boundary verification on damped game (mu_d = 0.20)
    J = build_damped_bilinear_jacobian(mu_d=0.20, omega=1.0, rho=1.0)
    eig = np.linalg.eigvals(J)[0]
    gamma, omega = float(np.real(eig)), float(abs(np.imag(eig)))
    eta_max_by_b1 = {}
    for b1 in (-0.5, -0.3, 0.0, 0.5, 0.85, 0.9):
        emax = schur_cohn_eta_max(gamma, omega, b1)
        eta_max_by_b1[b1] = emax
        assert spectral_radius(sim_momentum_companion(J, 0.98 * emax, b1)) < 1.0
        assert spectral_radius(sim_momentum_companion(J, 1.02 * emax, b1)) > 1.0

    # Positive momentum strictly SHRINKS stable step size on underdamped rotation; negative momentum EXPANDS it
    assert eta_max_by_b1[-0.5] > eta_max_by_b1[0.0] > eta_max_by_b1[0.5] > eta_max_by_b1[0.85] > eta_max_by_b1[0.9]

    # Undamped alternating game (mu_d = 0): beta1 > 0 ALWAYS destabilizes, beta1 = 0 is neutral, beta1 < 0 contracts!
    assert math.isclose(spectral_radius(alt_momentum_companion(0.0, 0.014, 0.021, 0.0)), 1.0, abs_tol=1e-12)
    for b1_pos in (0.1, 0.5, 0.85, 0.9):
        assert spectral_radius(alt_momentum_companion(0.0, 0.014, 0.021, b1_pos)) > 1.0
    for b1_neg in (-0.5, -0.3):
        assert spectral_radius(alt_momentum_companion(0.0, 0.014, 0.021, b1_neg)) < 1.0


def test_ogda_neutral_boundary_and_noise_tradeoff() -> None:
    """Verify OGDA lam=0.5 sits on the O(eta^2) neutral boundary at mu_d=0 while lam=1.0 contracts."""
    J0 = build_damped_bilinear_jacobian(mu_d=0.0, omega=1.0, rho=1.0)
    rho_0 = spectral_radius(ogda_companion(J0, eta=0.05, lam=0.0))
    rho_05 = spectral_radius(ogda_companion(J0, eta=0.05, lam=0.5))
    rho_10 = spectral_radius(ogda_companion(J0, eta=0.05, lam=1.0))
    # lam=0.0 expands at O(eta^2), lam=0.5 cancels O(eta^2) leaving O(eta^4) > 1, lam=1.0 contracts strictly!
    assert rho_0 > 1.001
    assert rho_05 > 1.0
    assert rho_10 < 0.999


def test_particlegan_law2_penalty_local_curvature_gap() -> None:
    """Verify exact local Hessian curvature d^2 R / dw^2 at w=0 across actual ParticleGAN critic penalties."""
    curv = audit_particlegan_penalties()
    # 1. Pure BCap hinge cap has EXACTLY zero curvature inside ||grad D|| < 1
    assert curv["bcap_hinge"]["d2R_dw2_at_0"] == 0.0
    assert curv["bcap_hinge"]["d2R_db2_at_0"] == 0.0
    # 2. UnifiedV11CriticPenalty without lambda_floor has negligible slope curvature (< 1e-5)
    #    while gamma_gauge provides curvature ONLY on the constant bias b (= 0.03), NOT slope w!
    assert curv["unified_v11_no_floor"]["d2R_dw2_at_0"] < 1e-5
    assert math.isclose(curv["unified_v11_no_floor"]["d2R_db2_at_0"], 0.03, rel_tol=1e-2)
    # 3. UnifiedV11CriticPenalty default supplies exact mu_D = 0.010 from undocumented lambda_floor = 0.005
    assert math.isclose(curv["unified_v11_default"]["d2R_dw2_at_0"], 0.010, rel_tol=2e-2)
    # 4. MinimalCriticPenalty Born-Infeld supplies full unit R1/R2 quadratic curvature (> 1.0, > 100x v11!)
    assert curv["minimal_born_infeld"]["d2R_dw2_at_0"] > 1.0
    assert curv["minimal_born_infeld"]["d2R_dw2_at_0"] > 100.0 * curv["unified_v11_default"]["d2R_dw2_at_0"]
    # 5. Pure R1+R2 reference = 2.0
    assert math.isclose(curv["r1r2_reference"]["d2R_dw2_at_0"], 2.0, rel_tol=1e-2)


def test_counterexample_a1_dirac_2d_all_four_laws_fails_without_dissipation() -> None:
    """Counterexample A1: 2D bounded 1-Lipschitz Dirac-GAN satisfies all 4 stated laws yet fails to converge."""
    # Actively-clamping Law 4.1 pushforward trust band bounds divergence into a persistent limit cycle!
    res_trust = run_analytic_game(
        mode="sim_gd_trust",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2000,
        eta_g=0.05,
        eta_d=0.05,
        delta_g_max=0.004,
        mu_d_penalty="none",
    )
    assert res_trust["max_lip_observed"] < 1.0
    assert res_trust["clamp_count"] > 1000
    assert res_trust["r_tail_mean"] > 0.20  # Fails to converge despite continuous trust clamping!

    # Actual ParticleGAN UnifiedV12Optimizer with pure Law-2 1-Lipschitz critic fails to converge!
    res_v12_none = run_analytic_game(
        mode="unified_v12",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2500,
        mu_d_penalty="none",
    )
    assert res_v12_none["max_lip_observed"] < 1.0
    assert res_v12_none["final_ema_func_gain"] == pytest.approx(1.0)
    assert res_v12_none["final_stiff_gate"] == pytest.approx(0.0)
    assert res_v12_none["r_tail_mean"] > 0.15  # Persistent non-convergent limit cycle!

    # Adding softplus cap WITHOUT lambda_floor STILL fails to converge!
    res_v12_sp = run_analytic_game(
        mode="unified_v12",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2500,
        mu_d_penalty="unified_softplus_no_floor",
    )
    assert res_v12_sp["r_tail_mean"] > 0.15

    # Adding hidden lambda_floor=0.005 OR Born-Infeld restores convergence (> 10x error reduction)!
    res_v12_fl = run_analytic_game(
        mode="unified_v12",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2500,
        mu_d_penalty="unified_lambda_floor",
    )
    res_v12_bi = run_analytic_game(
        mode="unified_v12",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2500,
        mu_d_penalty="born_infeld",
    )
    assert res_v12_fl["r_tail_mean"] < 0.025
    assert res_v12_bi["r_tail_mean"] < 0.015

    # OGDA lam=0.5 stalls on neutral boundary, whereas classical OGDA lam=1.0 converges at mu_d=0!
    res_ogda05 = run_analytic_game(
        mode="sim_ogda",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2500,
        eta_g=0.05,
        eta_d=0.05,
        ogda_lam=0.5,
        mu_d_penalty="none",
    )
    res_ogda10 = run_analytic_game(
        mode="sim_ogda",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2500,
        eta_g=0.05,
        eta_d=0.05,
        ogda_lam=1.0,
        mu_d_penalty="none",
    )
    assert res_ogda05["r_tail_mean"] > 0.15
    assert res_ogda10["r_tail_mean"] < 0.02

    # Momentum beta_1 = 0.9 destabilizes the undamped game catastrophically!
    res_adam09 = run_analytic_game(
        mode="alt_adam_b09",
        loss_name="wasserstein",
        z_mask=(0.0, 0.0),
        real_mask=(0.0, 0.0),
        init_theta=(0.4, 0.0),
        init_psi=(0.0, 0.0),
        steps=2000,
        mu_d_penalty="none",
    )
    assert res_adam09["r_tail_mean"] > 1.0


def test_counterexample_a2_manifold_nonsat_logistic_normal_limit_cycle() -> None:
    """Counterexample A2: Under default non_saturating logistic loss, tangent axis converges while normal axis orbits."""
    res_alt = run_analytic_game(
        mode="alt_gd",
        loss_name="non_saturating",
        z_mask=(1.0, 0.0),
        real_mask=(1.0, 0.0),
        init_theta=(0.4, 0.4),
        init_psi=(0.0, 0.0),
        steps=2000,
        mu_d_penalty="none",
    )
    assert res_alt["max_lip_observed"] < 1.0
    # Tangent axis (variance=1) is damped by logistic curvature f''(0)=1/4 -> converges!
    assert res_alt["tang_coord_tail_rms"] < 0.02
    # Normal axis (codimension-1, variance=0) has J_DD=0 -> persistent limit cycle!
    assert res_alt["norm_coord_tail_rms"] > 0.15

    # UnifiedV12 without dissipation keeps normal error > 0.05; Born-Infeld crushes it to < 0.005!
    res_v12_none = run_analytic_game(
        mode="unified_v12",
        loss_name="non_saturating",
        z_mask=(1.0, 0.0),
        real_mask=(1.0, 0.0),
        init_theta=(0.4, 0.4),
        init_psi=(0.0, 0.0),
        steps=2500,
        mu_d_penalty="none",
    )
    res_v12_bi = run_analytic_game(
        mode="unified_v12",
        loss_name="non_saturating",
        z_mask=(1.0, 0.0),
        real_mask=(1.0, 0.0),
        init_theta=(0.4, 0.4),
        init_psi=(0.0, 0.0),
        steps=2500,
        mu_d_penalty="born_infeld",
    )
    assert res_v12_none["norm_coord_tail_rms"] > 0.05
    assert res_v12_bi["norm_coord_tail_rms"] < 0.005


def test_counterexample_b_nonlinear_mlp_actual_particlegan_unified_v12() -> None:
    """Counterexample B: Balanced 2D MLP minimax game with actual UnifiedV12Optimizer (NS + delta_g_max).

    Demonstrates that a Law-2 hinge cap (||grad_x D|| <= 1 + zero-mean gauge) exhibits large
    non-convergent rotational oscillation even under UnifiedV12Optimizer, while MinimalCriticPenalty's
    true Born-Infeld quadratic curvature suppresses the limit cycle by > 5x.
    """
    res_bcap = run_nonlinear_mlp_game(
        penalty_kind="bcap_hinge_gauge",
        loss_name="wasserstein",
        betas=(0.0, 0.999),
        simultaneous=False,
        steps=750,
        tail_window=200,
        seed=0,
    )
    res_bi = run_nonlinear_mlp_game(
        penalty_kind="minimal_born_infeld",
        loss_name="wasserstein",
        betas=(0.0, 0.999),
        simultaneous=False,
        steps=750,
        tail_window=200,
        seed=0,
    )
    # Verify actual ParticleGAN Law-4 mechanisms ran (2D Quintic NS + delta_g_max probes)
    assert res_bcap["ns_2d_activations"] > 0
    assert res_bcap["probe_consumptions"] == 750
    assert res_bcap["tail_lip_med_mean"] < 1.0
    assert res_bcap["tail_lip_max_max"] <= 1.15
    assert res_bcap["tail_abs_mean_d_mean"] < 0.1
    # Pure Law-2 hinge cap oscillates persistently with large peak-to-peak amplitude and critic slope
    assert res_bcap["tail_normal_mean_ptp"] > 0.25
    assert res_bcap["tail_critic_normal_slope_mean"] > 0.03

    # True Born-Infeld quadratic bottom suppresses oscillation and normal critic slope dramatically!
    assert res_bi["tail_critic_normal_slope_mean"] < 0.005
    assert res_bi["tail_critic_normal_slope_mean"] < 0.10 * res_bcap["tail_critic_normal_slope_mean"]
    assert res_bi["tail_normal_mean_ptp"] < 0.30 * res_bcap["tail_normal_mean_ptp"]
