"""Automated verification, falsification & semantic control tests for P4 Function-Space Generality Audit."""
from __future__ import annotations

import math

import torch

from scripts.four_laws_audit import p4_funcspace_experiments as p4

CPU = torch.device("cpu")


def test_exp0_particlegan_band_scale_non_c1_and_oneshot_failures() -> None:
    """Verify exact ParticleGAN implementation properties: non-C^1 kinks, one-shot failures, floor & 1e-38."""
    res = p4.run_exp0_code_audit(CPU)

    # 1. _band_scale is NOT C^1 at delta_max or at its shifted amplification kink delta_min*(1 - 1/alpha_max)
    bd = res["band_scale_derivatives"]
    assert not bd["is_c1_at_delta_max"]
    assert abs(bd["at_delta_max_left"]) < 1e-6
    assert abs(bd["at_delta_max_right"] - (-5.0)) < 1e-2
    assert not bd["is_c1_at_amp_kink"]
    assert bd["at_amp_kink_left"] < -10.0
    assert abs(bd["at_amp_kink_right"]) < 1e-6

    # 2. One-shot linear rescaling violates delta_max by ~5x on saturating tanh; backtracking succeeds
    sat = res["oneshot_vs_backtrack_saturating_tanh"]
    assert sat["oneshot_violation_ratio"] > 4.9
    assert sat["backtrack_actual_disp"] <= sat["delta_max"]

    # 3. One-shot amplification via delta_g_min overshoots delta_max by >10x on superlinear cubic
    sup = res["oneshot_amplification_superlinear_cubic"]
    assert sup["oneshot_alpha"] > 5.0
    assert sup["overshoot_vs_delta_max"] > 10.0

    # 4. Probe budgets and real observe_scale() + coord_scale 0.05 floor:
    # Unit-RMS RGB images have ema_data_scale ~ sqrt(d) -> 1.0x (no floor distortion!),
    # while tiny per-coord RMS (1e-3) hits the 0.05 floor with exact 50x distortion.
    assert res["probe_budgets"]["vec_2d_rms1"] == 16
    assert res["probe_budgets"]["img_64x64x3_rms1"] == 2
    assert res["probe_budgets"]["img_256x256x3_rms1"] == 2
    assert math.isclose(res["coord_scale_floor"]["img_256x256x3_rms1"]["distortion_factor"], 1.0, rel_tol=1e-4)
    assert math.isclose(res["coord_scale_floor"]["img_64x64x3_rms1"]["distortion_factor"], 1.0, rel_tol=1e-4)
    assert math.isclose(res["coord_scale_floor"]["vec_2d_rms1e-3"]["distortion_factor"], 50.0, rel_tol=1e-4)
    assert math.isclose(res["coord_scale_floor"]["img_256x256x3_rms1e-3"]["distortion_factor"], 50.0, rel_tol=1e-4)

    # 5. IEEE-754 1e-38 dynamic scaling rescues 1e-24 gradient where FP32 Adam underflows to 0
    u38 = res["dynamic_1e38_underflow_check"]
    assert u38["unified_v12_step_norm"] > 1e-3
    assert u38["standard_adam_step_norm"] == 0.0

    # 6. MP-scaling formula divergence across t4_nash / minimal / unified_v11 / unified_v12
    mp = res["mp_scaling_128x128"]
    assert math.isclose(mp["t4_nash_elem_factor"], mp["t4_nash_expected_1_over_2sqrt128"], rel_tol=1e-5)
    assert math.isclose(mp["unified_v11_polar_aspect"], 1.0, rel_tol=1e-6)
    assert math.isclose(mp["unified_v12_critic_frob_factor"], math.sqrt(128.0), rel_tol=1e-6)


def test_exp1_reparameterization_scaling_laws_and_non_equivalence_to_natural_gradient() -> None:
    """Verify alpha^2 (SGD), alpha^1 (Adam/Muon), alpha^0 (LARS/undamped GN), layer freeze under scalar trust, and gamma duality."""
    res = p4.run_exp1_reparameterization(CPU)

    # Exact initial functional equivalence across 10^6x parameter scale distortion
    for err in res["max_initial_equiv_error"].values():
        assert err < 1e-5

    hr = res["homogeneous_reparam"]
    # Raw SGD scales as alpha^2 (~10^6x) and diverges at alpha=1000
    assert hr["sgd"]["1000"]["step1_delta_f"] / hr["sgd"]["1"]["step1_delta_f"] > 5e5
    assert hr["sgd"]["1000"]["diverged"]

    # Raw Adam and Muon-NS scale as alpha^1 (~10^2-10^3x) and degrade badly at alpha=1000
    assert hr["adam"]["1000"]["step1_delta_f"] / hr["adam"]["1"]["step1_delta_f"] > 5e2
    assert hr["adam"]["1000"]["final_mse_100"] > 100.0 * hr["adam"]["1"]["final_mse_100"]
    assert hr["muon_ns"]["1000"]["step1_delta_f"] / hr["muon_ns"]["1"]["step1_delta_f"] > 4e2
    assert hr["muon_ns"]["1000"]["final_mse_100"] > 1000.0 * hr["muon_ns"]["1"]["final_mse_100"]

    # LARS is strictly alpha^0-invariant in BOTH ||Delta f|| AND W1/W2 layer balance
    assert math.isclose(hr["lars"]["1"]["step1_delta_f"], hr["lars"]["1000"]["step1_delta_f"], rel_tol=1e-4)
    assert math.isclose(hr["lars"]["1"]["step1_w1_over_w2_ratio"], hr["lars"]["1000"]["step1_w1_over_w2_ratio"], rel_tol=1e-4)
    assert math.isclose(hr["lars"]["1"]["final_mse_100"], hr["lars"]["1000"]["final_mse_100"], rel_tol=1e-3)
    # Undamped pinv(J) is NOT strictly alpha^0-invariant on homogeneous ReLU nets because rank(J) <= P - H:
    # at B=128 (overdetermined 512x384), H=32 symmetries leave rank <= 352; at B=16, pinv(J) freezes W1' completely!
    gn_rd = res["gn_rank_deficiency"]
    assert gn_rd["numerical_rank_1e-6"] <= gn_rd["n_params"] - gn_rd["n_hidden_symmetries"]
    assert gn_rd["sv_353_first_symmetry_zero"] < 1e-6
    assert gn_rd["b16_pinv_w1_over_w2"]["1"] > 0.1
    assert gn_rd["b16_pinv_w1_over_w2"]["1000"] < 1e-5
    assert not math.isclose(
        hr["exact_gn_undamped"]["1"]["step1_w1_over_w2_ratio"],
        hr["exact_gn_undamped"]["1000"]["step1_w1_over_w2_ratio"],
        rel_tol=1e-2,
    )
    assert hr["exact_gn_undamped"]["1000"]["final_mse_100"] < 1e-5
    # Isotropic damped GN breaks layer balance severely even while keeping ||Delta f|| bounded
    assert hr["exact_gn_damped"]["1000"]["step1_w1_over_w2_ratio"] < 1e-5

    # Scalar realized trust bounds ||Delta f|| <= 0.10 and prevents divergence,
    # YET does NOT restore W1/W2 layer balance (proving non-equivalence to Natural Gradient / LARS!)
    for arm in ("sgd+realized_trust", "adam+realized_trust", "muon_ns+realized_trust"):
        assert hr[arm]["1000"]["step1_delta_f"] <= 0.101
        assert not hr[arm]["1000"]["diverged"]
        assert hr[arm]["1000"]["final_mse_100"] < 0.02
        assert hr[arm]["1000"]["step1_w1_over_w2_ratio"] < 1e-5

    # External output-gain scaling (gamma=1000): SGD realized scales as gamma^5 (~10^5 per 10x),
    # LARS & Muon-NS scale as gamma^1 (>500x blowup), while +realized_trust stays <= 0.10
    eg = res["external_output_gain"]
    assert eg["sgd"]["1000"]["step1_delta_f"] / eg["sgd"]["100"]["step1_delta_f"] > 5e4
    assert eg["lars"]["1000"]["step1_delta_f"] / eg["lars"]["1"]["step1_delta_f"] > 500.0
    assert eg["muon_ns"]["1000"]["step1_delta_f"] / eg["muon_ns"]["1"]["step1_delta_f"] > 500.0
    assert eg["lars+realized_trust"]["1000"]["step1_delta_f"] <= 0.101
    assert eg["adam+realized_trust"]["1000"]["step1_delta_f"] <= 0.101


def test_exp2_nonlinearity_gap_jvp_vs_oneshot_vs_backtracking() -> None:
    """Verify both superlinear (Gamma >> 1) and saturating concave (Gamma << 1) regimes plus JVP blindness."""
    res = p4.run_exp2_nonlinearity_gap(CPU)

    # Deep 16-layer MLP exhibits both concave kink (lr=0.01 -> one-shot violation) and superlinear blowup (lr=0.2 -> Gamma > 1000)
    by_lr = {row["lr"]: row for row in res["deep_mlp16_sweep"]}
    assert by_lr[0.01]["gamma_real_over_lin"] < 0.5
    assert by_lr[0.01]["oneshot_actual_disp"] > 0.50  # >2.5x violation of delta_max=0.20
    assert by_lr[0.01]["backtrack_actual_disp"] <= 0.201
    assert by_lr[0.2]["gamma_real_over_lin"] > 1000.0
    assert by_lr[0.2]["backtrack_actual_disp"] <= 0.201

    # Deep MLP training: raw Adam diverges, one-shot suffers repeated trust violations, backtracking has 0 violations
    tr = res["deep_mlp16_train"]
    assert tr["adam"]["diverged"]
    assert tr["adam+oneshot_secant"]["trust_violations"] > 10
    assert tr["adam+backtrack_trust"]["trust_violations"] == 0
    assert tr["adam+backtrack_trust"]["final_mse"] < 0.02

    # Saturating tanh ResNet: one-shot secant violates delta_max by 2.5x-6x across all LRs; backtracking strictly satisfies <= 0.20
    for row in res["saturating_tanh_resnet16_sweep"]:
        assert row["gamma_real_over_lin"] < 0.10
        assert row["oneshot_violation_ratio"] > 2.5
        assert row["backtrack_actual_disp"] <= 0.201

    # Tanh rail jump & bilinear small-weight saddle: JVP is blind (delta_lin < delta_max -> alpha=1),
    # realized jumps >> delta_max, while both one-shot secant and backtracking enforce <= delta_max (with over-damping!)
    rj = res["tanh_rail_jump"]
    assert rj["delta_lin_jvp"] < 1e-4
    assert rj["jvp_trust_alpha"] == 1.0
    assert rj["jvp_trust_actual_disp"] > 0.4
    assert rj["gamma_real_over_lin"] > 1e4
    assert rj["oneshot_actual_disp"] < 0.05  # Superlinear over-damping on rail jump
    assert rj["backtrack_actual_disp"] <= rj["delta_max"] + 1e-5

    bs = res["bilinear_saddle_jvp_blindness"]
    assert bs["delta_lin_jvp"] < bs["delta_max"]
    assert bs["jvp_trust_alpha"] == 1.0
    assert bs["jvp_trust_actual_disp"] > 3.0 * bs["delta_max"]
    assert bs["oneshot_actual_disp"] <= bs["delta_max"] + 1e-5
    assert bs["backtrack_actual_disp"] <= bs["delta_max"] + 1e-5


def test_exp3_transformer_softmax_flip_and_training_stability() -> None:
    """Verify exact softmax-flip JVP blindness and Transformer training stability vs Adam/Clip/Armijo."""
    res = p4.run_exp3_transformer_attention(CPU)

    sw = res["softmax_flip_exact_witness"]
    assert sw["delta_lin_jvp"] < 1e-2
    assert sw["delta_real"] > 3.5
    assert sw["gamma_real_over_lin"] > 1000.0

    # Superlinear QK/residual gap at lr=0.3
    by_lr = {row["lr"]: row for row in res["transformer_gap_sweep"]}
    assert by_lr[0.3]["gamma_real_over_lin"] > 10.0
    assert by_lr[0.3]["backtrack_actual_disp"] <= 0.251

    # Training at hostile lr=0.05: plain Adam explodes while trust methods, grad-clip, and Armijo all stabilize
    t05 = res["transformer_training"]["lr=0.05"]
    assert t05["adam"]["diverged"]
    for arm in ("adam+grad_clip", "adam+jvp_trust", "adam+oneshot_secant", "adam+backtrack_trust", "adam+armijo"):
        assert not t05[arm]["diverged"]
        assert t05[arm]["train_mse"] < 0.05


def test_exp4_supervised_scope_and_dynamical_rollout_boundary() -> None:
    """Verify benign supervised overhead, hostile-gain rescue, Armijo parity, and Lorenz Lyapunov amplification."""
    res = p4.run_exp4_supervised_and_dynamics(CPU)

    # Benign spiral: function-space trust gives zero accuracy gain over Adam
    sp = res["supervised_benign_spiral"]
    assert sp["adam"]["val_acc"] > 0.95
    assert math.isclose(sp["adam+backtrack_trust"]["val_acc"], sp["adam"]["val_acc"], abs_tol=1e-6)

    # Hostile gain=50 regression: SGD diverges; realized trust & Armijo beat raw Adam cleanly
    hr = res["supervised_hostile_gain50_reg"]
    assert hr["sgd"]["diverged"]
    assert hr["adam+backtrack_trust"]["final_mse"] < 0.5 * hr["adam"]["final_mse"]
    assert hr["adam+armijo"]["final_mse"] < 0.5 * hr["adam"]["final_mse"]

    # Pendulum rollout: 1-step & rollout trust dramatically beat Adam/GradClip, while Armijo matches/beats on scalar loss
    pd = res["pendulum_rollout_training"]
    assert pd["adam+1step_func_trust"]["val_40step_rollout_rmse"] < 0.5 * pd["adam"]["val_40step_rollout_rmse"]
    assert pd["adam+rollout20_func_trust"]["val_40step_rollout_rmse"] < 0.5 * pd["adam"]["val_40step_rollout_rmse"]
    assert pd["adam+armijo_1step"]["val_40step_rollout_rmse"] <= pd["adam+1step_func_trust"]["val_40step_rollout_rmse"]

    # Lorenz-63 Lyapunov amplification: tiny 1-step perturbation <= 0.025 blows up >10x over 80 steps
    la = res["lorenz_lyapunov_amplification"]
    assert la["one_step_delta_f"] <= 0.025
    assert la["amplification_ratio_terminal_over_1step"] > 10.0


def test_exp5_adversarial_probe_blindspots_and_nullspace_obstruction() -> None:
    """Verify i.i.d. small-probe underestimation, 1% rare-mode failure (+ delta_g_min weaponization), and exact nullspace OOD."""
    res = p4.run_exp5_probe_blindspots(CPU)

    # 5A: Small probes (B=2 image regime, B=16 vector regime) severely underestimate heavy-tailed displacement
    by_b = res["iid_probe_size_scaling"]["by_probe_size"]
    assert by_b["B=2"]["heavy_underestimate_gt_1.5x_frac"] > 0.80
    assert by_b["B=16"]["heavy_underestimate_gt_1.5x_frac"] > 0.35
    assert by_b["B=256"]["heavy_underestimate_gt_1.5x_frac"] < 0.05

    # 5B: 1% rare tail mode missed with 85.1% (B=16) / 98.0% (B=2); delta_g_min amplifies blind-spot destruction!
    rm = res["rare_mode_1pct_blindspot"]
    assert math.isclose(rm["miss_prob_b16"], 0.99 ** 16, rel_tol=1e-6)
    assert math.isclose(rm["miss_prob_b2"], 0.99 ** 2, rel_tol=1e-6)
    assert rm["rare_unit_act_on_maj_probe"] == 0.0
    assert rm["ceiling_only_probe_disp"] <= 0.01
    assert rm["ceiling_only_rare_disp"] > 10.0
    assert rm["band_with_dmin_alpha"] > 2.0
    assert rm["band_with_dmin_rare_disp"] > 2.0 * rm["ceiling_only_rare_disp"]
    # Stratified probe reduces rare displacement dramatically (>40x), yet unweighted RMS still dilutes by ~sqrt(B_p)
    assert rm["stratified_probe_alpha"] < 0.05
    assert rm["stratified_rare_disp"] < rm["ceiling_only_rare_disp"] / 40.0
    assert rm["stratified_rare_disp"] > rm["delta_max"]

    # 5C: Exact probe-Jacobian nullspace (dim >= P - B*d_out) and natural OOD ratio
    ns = res["probe_nullspace_and_ood_blindspot"]
    assert ns["nullspace_dim_lower_bound"] == ns["n_params"] - ns["probe_constraints_B_times_dout"]
    assert ns["nullspace_dim_lower_bound"] > 700
    assert ns["nullspace_jvp_id_probe"] < 1e-6
    assert ns["nullspace_realized_id_probe"] < 0.02
    assert ns["nullspace_ood_over_id_ratio"] > 50.0
    assert ns["natural_ood_over_id_ratio"] > 3.0


def test_semantic_controls_jvp_taylor_limit_damage_and_determinism() -> None:
    """Semantic controls: exact Taylor limit for JVP vs realized, deliberate damage detection, and seed-0 determinism."""
    # 1. Exact first-order Taylor limit: lim_{h->0} ||f(theta + h*d) - f(theta)|| / (h * ||J d||) == 1.0
    net = p4.SupervisedMLP(4, 16, 2, out_gain=1.0).to(CPU)
    p4._init_orth_module(net, seed=0, gain=1.0)
    gen = torch.Generator(device=CPU).manual_seed(99)
    x = torch.randn(16, 4, generator=gen)
    base = p4._clone_params(net)
    d = [torch.randn(p.shape, generator=gen) for p in net.parameters()]
    d_lin = p4._jvp_rms_displacement(net, x, d)
    # Linearity of JVP: scaling tangent by 3.7 scales JVP by exactly 3.7
    d_lin_scaled = p4._jvp_rms_displacement(net, x, [3.7 * t for t in d])
    assert math.isclose(d_lin_scaled, 3.7 * d_lin, rel_tol=1e-5)
    # Small-step Taylor convergence
    h = 1e-4
    d_real_h = p4._realized_rms_displacement(net, x, base, d, h)
    assert math.isclose(d_real_h / (h * d_lin), 1.0, rel_tol=1e-2)

    # 2. Deliberate damage detection: perturbing final layer by +10.0 triggers huge realized displacement & clamping
    damage = [torch.zeros_like(p) for p in net.parameters()]
    damage[-1] = torch.full_like(damage[-1], 10.0)
    d_dmg = p4._realized_rms_displacement(net, x, base, damage, 1.0)
    assert math.isclose(d_dmg, 10.0, rel_tol=1e-5)
    a_bt, post_bt, _ = p4._backtrack_realized_trust(net, x, base, damage, 0.10)
    assert a_bt < 0.02
    assert post_bt <= 0.101

    # 3. Determinism control across repeated calls under protocol seed 0
    r1 = p4.run_exp0_code_audit(CPU)
    r2 = p4.run_exp0_code_audit(CPU)
    assert r1 == r2
    b1 = p4.run_exp5_probe_blindspots(CPU)
    b2 = p4.run_exp5_probe_blindspots(CPU)
    assert b1 == b2
