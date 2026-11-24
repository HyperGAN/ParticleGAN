"""Reproducible mathematical proofs, counterexamples, and invariant controls for the P7 Law 1 audit.

Covers:
1. Exact bimodal & multi-mode Gaussian isoperimetric bounds (Borell-TIS), constructive continuous
   ramp/tanh approximation in W_1, W_2, TV, MMD, and connected-support MoG Brenier slope explosion.
2. Grid100 Poincaré macroscopic lower bound vs multi-mode isoperimetric cliff vs adaptive MoG isometry.
3. Invariant 1: exact proto_w mass conservation vs w_fake_ema / chart_fake_mass violations.
4. Invariant 2: exact scalar Pythagorean LTV vs anisotropic covariance destruction, kappa* disconnection
   threshold, Gersho merge-before-split missing LTV update, 19% split variance loss, proto_var bias,
   and _minimal cross-scale false merge.
5. Invariant 3: Moreau-Yosida damped Laguerre dual shrinkage/deadband/clamp/coincident failures,
   _minimal pre-coalesced psi mean bug, AND _t1_coupling laguerre_hard_commitment exact quotas vs
   discrete OT suboptimality and B_f < K deterministic index-bias eviction bug.
6. Invariant 4: orthogonal pullback covariance isometry linear exactness, 4D clamp_min(0.15) bias,
   _minimal 2-probe variance, O(sigma_z) forward vs O(sigma_z^2) central curvature bias, AND isolated
   unified_v11 fastpath vs unified_v4 numerical parity + strictly nonzero host-sync count.
7. Non-adversarial Division-of-Labor control (ZERO discriminator): tiny 82-param generator with
   adaptive Laguerre MoG prior outperforms 33,666-param generator on fixed unimodal prior.
"""
from __future__ import annotations

import math

import pytest
import torch

from particlegan._minimal.prior import _coalesce_isolated
from particlegan._t1_coupling.charts import laguerre_hard_commitment
from particlegan._unified.prior import _compute_pythagorean_smoothed_cells
from particlegan._unified_v11.prior import _compute_pythagorean_smoothed_cells_fast

from scripts.four_laws_audit.p7_law1_topology import (
    _audit_laguerre_assignment,
    _audit_mass_conservation,
    _audit_pullback_isometry_and_fastpath,
    _audit_pythagorean_ltv,
    bimodal_bridge_lower_bound,
    bimodal_lip_lower_bound_for_bridge,
    bimodal_w1_lower_bound,
    bimodal_w2_lower_bound,
    multimode_1d_packing_lip_lower_bound,
    multimode_lip_lower_bound_isoperimetric,
    normal_cdf,
    normal_ppf,
    run_exp1_topology_and_isoperimetry,
    run_exp3_nonadversarial_division_of_labor,
)


# ---------------------------------------------------------------------------
# 1. Exact Topological vs Quantitative Approximation Bounds (Borell-TIS)
# ---------------------------------------------------------------------------

def test_exact_bimodal_isoperimetric_bounds_and_constructive_sharpness() -> None:
    delta = 2.0
    for alpha in (0.20, 0.05, 0.01, 1e-3, 1e-4):
        lip_min = bimodal_lip_lower_bound_for_bridge(delta, alpha)
        # Verify exact inversion: 2 * Phi(Delta / (2 * L_min)) - 1 == alpha
        assert math.isclose(bimodal_bridge_lower_bound(delta, lip_min), alpha, rel_tol=1e-12, abs_tol=1e-14)
        # Verify corrected quantile argument (1 + alpha)/2 blows up as alpha -> 0 (NOT 1 - alpha/2!)
        wrong_arg = delta / (2.0 * normal_ppf(1.0 - 0.5 * alpha))
        if alpha <= 0.05:
            assert lip_min > 10.0 * wrong_arg
        # Asymptotic Delta / (sqrt(2*pi) * alpha) matches closely for small alpha
        asymp = delta / (math.sqrt(2.0 * math.pi) * alpha)
        if alpha <= 0.01:
            assert math.isclose(lip_min, asymp, rel_tol=1e-3)
            # Direct W1 and W2 exact closed-form bounds match small-delta asymptotics
            w1_lb = bimodal_w1_lower_bound(delta, lip_min)
            w2_lb = bimodal_w2_lower_bound(delta, lip_min)
            assert math.isclose(w1_lb, (delta ** 2) / (4.0 * math.sqrt(2.0 * math.pi) * lip_min), rel_tol=1e-3)
            assert math.isclose(
                w2_lb, math.sqrt((delta ** 3) / (12.0 * math.sqrt(2.0 * math.pi) * lip_min)), rel_tol=1e-3
            )

    exp1 = run_exp1_topology_and_isoperimetry()
    rows = exp1["bimodal_constructive_vs_bounds"]
    # Exact quadrature of constructive continuous ramp G_{Delta/(2L)} attains W1 & W2 lower bounds sharply!
    for r in rows:
        assert math.isclose(r["w1_quad_ramp"], r["w1_exact_bound"], rel_tol=5e-3, abs_tol=2e-5)
        assert math.isclose(r["w2_quad_ramp"], r["w2_exact_bound"], rel_tol=5e-3, abs_tol=2e-4)
        # Smooth tanh(beta*z) strictly respects W1 and W2 lower bounds
        assert r["w1_emp_tanh"] >= r["w1_exact_bound"] * 0.95
        assert r["w2_emp_tanh"] >= r["w2_exact_bound"] * 0.80
    # Monotone convergence to 0 in W1, W2, TV, MMD, and bridge mass as alpha -> 1e-4 (while L blows up!)
    assert rows[-1]["w1_quad_ramp"] < 1e-4
    assert rows[-1]["w2_quad_ramp"] < 1e-2
    assert rows[-1]["bridge_emp_ramp"] < 5e-4
    assert rows[-1]["mmd_emp_ramp"] < 5e-4
    assert rows[-1]["lip_lower_bound"] > 7900.0

    # Connected-support Gaussian mixture Brenier slope explosion (grid100 adjacent 33.3sigma -> > 10^58!)
    mog_rows = {r["case"]: r for r in exp1["mog_connected_support_brenier_cliff"]}
    assert mog_rows["grid100_adj_33.3sigma"]["exact_brenier_midpoint_slope_log10"] > 58.0


def test_multimode_isoperimetric_and_grid100_poincare_bounds() -> None:
    # M=2 multi-mode bound reduces identically to bimodal bound
    for alpha in (0.10, 0.05, 0.01, 0.004):
        l_bi = bimodal_lip_lower_bound_for_bridge(1.0, alpha)
        l_mm = multimode_lip_lower_bound_isoperimetric(1.0, 2, alpha)
        l_1d = multimode_1d_packing_lip_lower_bound(1.0, 2, alpha)
        assert math.isclose(l_mm, l_bi, rel_tol=1e-12)
        assert math.isclose(l_1d, l_bi, rel_tol=1e-9)

    # Strictly increasing with M, and 1D packing >= dimension-free isoperimetric bound for all M >= 2
    prev_iso = 0.0
    for m_modes in (2, 4, 16, 25, 100):
        l_iso = multimode_lip_lower_bound_isoperimetric(1.0, m_modes, 0.01)
        l_1d = multimode_1d_packing_lip_lower_bound(1.0, m_modes, 0.01)
        assert l_iso > prev_iso
        assert l_1d >= l_iso - 1e-9
        prev_iso = l_iso

    # Grid100 exact geometry: Delta = 0.79, Poincare macro std = sqrt(8.2509) = 2.8724 > 2.25 doc heuristic
    grid100_delta = 1.0 - 2.0 * (3.5 * 0.03)
    assert math.isclose(grid100_delta, 0.79, rel_tol=1e-12)
    macro_std = math.sqrt((10.0 ** 2 - 1.0) / 12.0 + 0.03 ** 2)
    assert macro_std > 2.87 and macro_std > 2.25
    l_grid100_p99 = multimode_lip_lower_bound_isoperimetric(grid100_delta, 100, 0.01)
    assert l_grid100_p99 > 104.0  # > 86x larger than adaptive MoG affine L = 0.03 / 0.025 = 1.20!
    assert math.isclose(normal_cdf(0.0), 0.5, abs_tol=1e-15)

    # Verify exact Mills-ratio multi-mode asymptotic: L ~ sqrt(2 ln M) / (2 ln(1/(1 - alpha))) * Delta
    m_big = 10 ** 8
    for alpha in (0.5, 0.1, 0.01):
        exact = multimode_lip_lower_bound_isoperimetric(1.0, m_big, alpha)
        mills_asymp = math.sqrt(2.0 * math.log(m_big)) / (2.0 * math.log(1.0 / (1.0 - alpha)))
        assert math.isclose(exact, mills_asymp, rel_tol=0.05)


# ---------------------------------------------------------------------------
# 2. Invariant 1: Mass Conservation (proto_w EXACT vs w_fake_ema VIOLATED)
# ---------------------------------------------------------------------------

def test_invariant1_mass_conservation_and_w_fake_ema_violation() -> None:
    res = _audit_mass_conservation()
    # Real mass proto_w / w_real is EXACT (up to FP32 ~1e-6) across all 7 implementations
    for key in ("t1_prior", "unified", "unified_v11", "minimal", "minimal_single_outlier_k8", "t4_arch_cap4", "empf_emeans"):
        assert res[key]["max_proto_w_sum_err"] < 1e-5
        assert res[key]["min_proto_w"] >= 0.0
    assert res["t1_coupling_charts"]["max_w_real_sum_err"] < 1e-5
    assert res["t1_coupling_charts"]["min_w_real"] >= 0.0

    # VIOLATION: w_fake_ema / w_fake mass conservation is broken on uncovered-mode teleportation!
    for key in ("t1_prior", "unified", "unified_v11"):
        assert res[key]["max_w_fake_ema_sum_err"] > 0.01
    assert res["minimal_single_outlier_k8"]["max_w_fake_ema_sum_err"] > 0.05
    assert res["t4_arch_cap4"]["max_w_fake_sum_err"] > 0.05
    # VIOLATION (intentional in-basin design): chart_fake_mass drops below 1.0 when fakes sit in valleys
    assert res["t1_coupling_charts"]["fake_mass_sum_after_valley"] < 0.95


# ---------------------------------------------------------------------------
# 3. Invariant 2: Pythagorean LTV Exactness, Anisotropy Loss & Code Defects
# ---------------------------------------------------------------------------

def test_invariant2_pythagorean_ltv_anisotropy_and_code_defects() -> None:
    res = _audit_pythagorean_ltv()
    # Exact scalar trace LTV holds to FP32 precision, and fastpath == ref bit-for-bit
    assert res["exact_scalar_ltv_3cell_err"] < 1e-6
    assert res["exact_scalar_ltv_2cell_err"] < 1e-6
    assert res["centroid_err_max"] < 1e-6
    assert res["fast_vs_ref_max_diff"] == 0.0

    # Anisotropic covariance destruction follows exact formula (kappa^2 - 1)/sqrt(2*(kappa^4 + 1))
    aniso = {r["aspect_ratio_kappa"]: r for r in res["anisotropic_destruction"]}
    for kappa in (1.0, 2.0, 3.0, 5.0):
        assert aniso[kappa]["coalesced"] is True
        assert aniso[kappa]["trace_err"] < 1e-6
        assert math.isclose(aniso[kappa]["cov_rel_frob_err"], aniso[kappa]["analytic_iso_frob_err"], abs_tol=1e-5)
    assert aniso[3.0]["cov_rel_frob_err"] > 0.62
    # Above critical kappa* = sqrt(6.76 / (8/pi - (1 - 2/pi)*6.76)) ~= 8.665, sibling gate disconnects!
    assert aniso[10.0]["coalesced"] is False
    assert aniso[10.0]["trace_err"] > 0.05

    # Bug 2A: Gersho merge-before-split omits proto_var[a] LTV update (73% variance deficit on test case)
    assert res["gersho_merge_ltv_bug"]["actual_v1_after_merge"] == pytest.approx(
        res["gersho_merge_ltv_bug"]["v1_before"], abs=1e-7
    )
    assert res["gersho_merge_ltv_bug"]["relative_ltv_deficit"] > 0.70

    # Bug 2B: Split heuristic 0.65 * v + 0.40^2 * v = 0.81 * v loses 19% of total variance
    assert math.isclose(res["split_ltv_bug"]["variance_retention_ratio"], 0.81, abs_tol=1e-4)

    # Bug 2C: Pre-update c_prev / b_cnt (v11) and post-update b_cnt-0.75 (t1_prior) are positively biased at b=4
    assert res["proto_var_estimator_bias_n4"]["unified_v11_rel_bias"] > 0.005
    assert res["proto_var_estimator_bias_n4"]["t1_prior_rel_bias"] > 0.02

    # Bug 2D: _minimal._coalesce_isolated falsely merges 10x width-ratio pair (missing sig_max <= 2.2*sig_min)
    assert res["minimal_cross_scale_false_merge"]["unified_v11_coalesced_10x_pair"] is False
    assert res["minimal_cross_scale_false_merge"]["minimal_coalesced_10x_pair"] is True

    # Semantic control: verify _compute_pythagorean_smoothed_cells_fast matches ref on a 10x-ratio case too
    pc = torch.tensor([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]], dtype=torch.float32)
    pv = torch.tensor([0.01, 1.0, 0.01], dtype=torch.float32)
    pw = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    c_r, s_r, _, _ = _compute_pythagorean_smoothed_cells(pc, pv, pw)
    c_f, s_f, _, _ = _compute_pythagorean_smoothed_cells_fast(pc, pv, pw)
    assert torch.equal(c_r, c_f) and torch.equal(s_r, s_f)
    _, _, _, coal_m = _coalesce_isolated(pc, pv, pw)
    assert bool(coal_m[0].item()) is True


# ---------------------------------------------------------------------------
# 4. Invariant 3: Laguerre Dual Assignment (Moreau-Yosida & Hard Commitment)
# ---------------------------------------------------------------------------

def test_invariant3_laguerre_dual_and_hard_commitment_defects() -> None:
    res = _audit_laguerre_assignment()
    my = res["moreau_yosida_dual"]

    # 3A: Moreau-Yosida damped dual ascent does NOT converge to exact quotas
    # Unequal weights leave residual TV > 0.02 (Tikhonov ridge + deadband bias)
    assert my["unequal_mass_b256"]["tv_final"] > 0.02
    # Rare 2% mode completely starved at B=64 stays permanently empty because TV=0.02 < tv_floor=0.031!
    assert my["starved_2pct_mode_b64"]["final_empty_cells_full"] == 1
    assert my["starved_2pct_mode_b64"]["tv_final"] < my["starved_2pct_mode_b64"]["tv_floor_deadband"]
    # Far outlier saturates psi_cap, leaves empty cell, and breaks zero-mean gauge!
    assert my["far_outlier_cap_saturation"]["clamp_hits"] > 100
    assert my["far_outlier_cap_saturation"]["final_empty_cells_full"] == 1
    assert my["far_outlier_cap_saturation"]["max_abs_psi_mean"] > 0.1
    # B < M (64 < 100): every batch has >= 36 empty cells (mean > 50)
    assert my["b_lt_m_grid100_b64"]["mean_empty_cells_per_step"] > 45.0
    # Coincident prototypes cause winner-take-all starvation of one slot
    assert my["coincident_prototypes"]["final_empty_cells_full"] == 1
    assert my["coincident_prototypes"]["tv_final"] > 0.20
    # _minimal/prior.py:190 pre-coalesced mean subtraction bug breaks zero-mean gauge by 0.8
    assert math.isclose(my["minimal_pre_coalesced_mean_bug"], 0.8, abs_tol=1e-5)

    # 3B: _t1_coupling laguerre_hard_commitment: exact integer quotas, yet suboptimal OT & B_f < K index bias!
    hc = res["t1_coupling_laguerre_hard_commitment"]
    assert hc["all_40_random_trials_exact_integer_quotas"] is True
    assert hc["random_40_trials_frac_strictly_suboptimal"] >= 0.70
    assert hc["random_40_trials_max_rel_ot_gap"] > 0.05
    assert hc["adv_6x3_gap"] > 0.05  # explicit 6x3 counterexample strictly suboptimal (> 7% cost gap)

    # B_f < K deterministic index-bias eviction bug: 36/36 zero-distance particles on charts 64..99 evicted!
    bug = hc["b_lt_m_index_bias_bug"]
    assert bug["particles_assigned_to_charts_ge_64"] == 0
    assert bug["zero_dist_particles_evicted"] == 36
    assert bug["eviction_mean_euclidean_dist"] > 2.0

    # Direct semantic control on laguerre_hard_commitment with B=2, K=4 uniform -> strictly picks charts 0, 1
    charts4 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    x_on_high = torch.tensor([[10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    _, _, win2 = laguerre_hard_commitment(
        x_on_high, charts4, torch.full((4,), 0.25), torch.ones(4), torch.tensor(0.5)
    )
    assert set(win2.tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# 5. Invariant 4: Orthogonal Pullback Isometry & Fastpath Parity / Syncs
# ---------------------------------------------------------------------------

def test_invariant4_pullback_isometry_and_fastpath_parity_and_syncs() -> None:
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    res = _audit_pullback_isometry_and_fastpath(device)

    lin = res["linear_probe_accuracy"]
    # 2D normalized orthogonal probe and unclamped 4D Radon-Hurwitz are EXACT for linear G (even anisotropic!)
    assert lin["2d_normalized_iso_max_rel_err"] < 1e-13
    assert lin["2d_normalized_aniso_max_rel_err"] < 1e-13
    assert lin["4d_rh_unclamped_max_rel_err"] < 1e-13
    # Clamping ||eps||^2 at 0.15 introduces exact 1 - 0.05/0.15 = 2/3 relative bias at ||eps||^2 = 0.05!
    assert math.isclose(lin["4d_rh_small_norm_005_clamped_rel_err"], 2.0 / 3.0, abs_tol=1e-12)
    # _minimal 2-probe in d_z=4 has high single-sample relative error (> 40%) on anisotropic J_G
    assert lin["4d_minimal_2probe_single_sample_rel_err"] > 0.40

    # Nonlinear curvature bias: forward probe is O(sigma_z^1), central probe is O(sigma_z^2)!
    nl = res["nonlinear_curvature_bias"]
    assert 0.95 <= nl["forward_probe_log_slope"] <= 1.05
    assert 1.95 <= nl["central_probe_log_slope"] <= 2.05

    # v12 fastpath (_unified_v11/prior.py) vs v4 original (_unified_v4/prior.py)
    fp = res["v12_fastpath_vs_v4_original"]
    assert fp["sinkhorn_barycenter_max_abs_diff"] < 1e-6
    assert fp["update_quantizer_20step_proto_c_max_diff"] < 1e-5
    assert fp["update_quantizer_20step_proto_var_max_diff"] < 1e-6
    assert fp["update_quantizer_20step_proto_w_max_diff"] < 1e-6

    iso = fp["full_pullback_step_bench"]["isolated_pullback_step_v11_vs_v4"]
    # Tight single-step numerical parity (< 1e-5 across all state tensors)
    assert iso["single_step_loss_g_abs_diff"] < 1e-5
    assert iso["single_step_proto_c_max_diff"] < 1e-5
    assert iso["single_step_psi_max_diff"] < 1e-5
    assert iso["single_step_prior_z_max_diff"] < 1e-5
    assert iso["single_step_g_param_max_diff"] < 1e-5
    # Large sync reduction (~5x), YET strictly nonzero syncs (falsifying literal "zero-sync pullback"!).
    # On CPU, _count_host_syncs(cpu) also intercepts 11 identical non-pullback CPU optimizer calls
    # on both v4 and v11 (9 Adam state["step"].item() calls + 2 set_sampled_rows bounds-check bools),
    # yielding 51 vs 19 on CPU vs 40 vs 8 on CUDA (exact same 32-sync pullback reduction).
    cpu_opt_offset = 11 if device.type == "cpu" else 0
    v11_pb_syncs = iso["v11_pullback_only_syncs"] - cpu_opt_offset
    v4_pb_syncs = iso["v4_pullback_only_syncs"] - cpu_opt_offset
    assert v11_pb_syncs < 0.35 * v4_pb_syncs
    assert v11_pb_syncs >= 4
    assert (iso["v4_pullback_only_syncs"] - iso["v11_pullback_only_syncs"]) >= 25
    bench = fp["full_pullback_step_bench"]
    assert bench["unified_v11"]["host_syncs_per_step_total"] < 0.35 * bench["unified_v4"]["host_syncs_per_step_total"]
    assert bench["unified_v12"]["host_syncs_per_step_total"] < 0.35 * bench["unified_v4"]["host_syncs_per_step_total"]


# ---------------------------------------------------------------------------
# 6. Non-Adversarial Division of Labor (ZERO Discriminator!)
# ---------------------------------------------------------------------------

def test_nonadversarial_division_of_labor_breaks_jacobian_cliff_and_ghost_mass() -> None:
    """Verify Law 1 holds WITHOUT adversarial training: 82-param tiny G + adaptive prior beats 33k-param G + N(0,I)."""
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    # Evaluate only at step 500 (unused intermediate step-250 2048x2048 Sinkhorn evals are skipped for CPU CI speed;
    # training trajectory and final metrics are bit-identical because eval uses an isolated local RNG).
    res = run_exp3_nonadversarial_division_of_labor(device, steps=500, batch_size=256, eval_every=500)

    arm_a = res["arms"]["arm_a_fixed_unimodal"]
    arm_b = res["arms"]["arm_b_fixed_mismatched_4mode"]
    arm_c0 = res["arms"]["arm_c0_learnable_mog_pure_sinkhorn"]
    arm_c1 = res["arms"]["arm_c1_adaptive_laguerre_atlas"]
    arm_c2 = res["arms"]["arm_c2_adaptive_laguerre_tiny_g"]

    # Confirm 410x parameter asymmetry (82 vs 33,666)
    assert arm_c2["g_params"] == 82 and arm_a["g_params"] == 33666

    # Fixed unimodal and fixed mismatched priors suffer massive valley ghost mass & steep Jacobians
    assert arm_a["final"]["p_bridge"] > 0.70
    assert arm_b["final"]["p_bridge"] > 0.85
    assert arm_a["final"]["jac_max"] > 9.0
    assert arm_b["final"]["jac_max"] > 15.0

    # Pure Sinkhorn learnable MoG (Arm C0) already recovers modes & slashes Jacobian cliff without Laguerre
    assert arm_c0["final"]["modes_covered"] >= 14
    assert arm_c0["final"]["jac_max"] < 0.6 * arm_a["final"]["jac_max"]
    assert arm_c0["final"]["sinkhorn_w2"] < 0.6 * arm_a["final"]["sinkhorn_w2"]

    # Adaptive Laguerre Atlas + TINY 82-param generator (Arm C2) crushes 33k-param Arm A on every metric!
    assert arm_c2["final"]["modes_covered"] == 16
    assert arm_c2["final"]["p_bridge"] < 0.08
    assert arm_c2["final"]["p_bridge"] < 0.15 * arm_a["final"]["p_bridge"]
    assert arm_c2["final"]["sinkhorn_w2"] < 0.45 * arm_a["final"]["sinkhorn_w2"]
    assert arm_c2["final"]["jac_max"] < 0.30 * arm_a["final"]["jac_max"]
    assert arm_c1["final"]["modes_covered"] >= 15
    assert arm_c1["final"]["jac_max"] < 0.45 * arm_a["final"]["jac_max"]
