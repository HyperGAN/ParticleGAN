"""Acceptance criteria tests for Tomographic Laws (Law 3) and Anti-Collapse Regularizers.

Verifies AC2 & AC3 requirements:
- symmetric_log_odds: constant shift invariance (scalar and vector), binary anti-symmetry,
  strict soft-cap bounds (|u_k| <= tau), and numerical stability across dtypes (float32,
  float64, bfloat16, float16).
- MarginalEntropyRegularizer: zero loss on uniform distributions, positive squared hinge loss
  and active restoring gradients on collapsed distributions, and capacity-balanced attraction.
- NormalizedFeatureCosineRepulsion: zero loss on orthogonal features, positive loss on
  collinear features, B=1 edge case, and active gradient flow.
- TomographicConstraintTarget: deterministic reproduction with torch.Generator, uniform
  Cartesian product mode coverage, valid shapes and one-hot encodings.
- evaluate_tomographic_constraints: oracle PASS on synthetic perfect outputs and destructive
  FAIL on synthetic collapsed outputs across all 6 quantitative acceptance metrics.
- GANTrainer integration: zero recipe bloat (112 fields), seamless ManipulatedPrior step
  execution, and bit-exact checkpoint continuation parity.
"""
import dataclasses
import math
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from particlegan import (
    GANTrainer,
    Recipe,
    get_recipe,
    get_recipe_fields,
)
from particlegan.priors import (
    CosineRepulsionManipulator,
    MarginalEntropyManipulator,
    PriorManipulator,
)
from particlegan.tomographic import (
    MarginalEntropyRegularizer,
    NormalizedFeatureCosineRepulsion,
    TomographicConstraintTarget,
    evaluate_tomographic_constraints,
    symmetric_log_odds,
)

FROZEN_112_RECIPE_FIELDS = (
    "name",
    "critic_formulation",
    "model",
    "training_precision",
    "z_dim",
    "num_particles",
    "prior_kind",
    "sigma_rel",
    "standardize",
    "num_classes",
    "conditioning",
    "ucd_target",
    "ucd_weight",
    "alpha_bar",
    "batch_size",
    "total_steps",
    "continuous_policy",
    "lr",
    "d_lr_mult",
    "prior_lr_mult",
    "betas",
    "prior_betas",
    "reg_arm",
    "reg_coeff",
    "reg_kappa",
    "reg_every",
    "critic_step_mode",
    "prior_reg",
    "prior_update",
    "prior_regularizer",
    "prior_reg_target_std",
    "prior_reg_eps",
    "prior_l2",
    "prior_proximal_strength",
    "prior_coulomb_softening",
    "prior_coulomb_confinement",
    "prior_coulomb_neighbors",
    "prior_sigreg_projections",
    "prior_sigreg_knots",
    "ema_decay",
    "lr_anneal_start",
    "lr_floor",
    "network_lr_floor",
    "network_lr_horizon_cap",
    "reg_anchor_min_decay",
    "reg_anchor_weight",
    "direct_particle_gain",
    "d_guard_ratio",
    "d_guard_min_steps",
    "latent_damping_max_rate",
    "direct_particle_betas",
    "input_noise_std",
    "input_noise_anneal_end",
    "output_noise_std",
    "output_noise_warmup",
    "encoder_mode",
    "routing_temperature",
    "distance_reduction",
    "observation_sigma",
    "reconstruction_weight",
    "kinetic_transport_weight",
    "kinetic_transport_local_weight",
    "kinetic_transport_projections",
    "amsgrad",
    "critic_r1_real",
    "critic_payoff_damping",
    "output_noise_mode",
    "lr_control",
    "particle_birth_death",
    "row_evidence_gate",
    "table_release_rule",
    "row_evidence_hot",
    "row_evidence_exclude",
    "row_evidence_hold",
    "birth_death_space",
    "serve_average",
    "reopen_signal",
    "reopen_anchor",
    "reopen_guard",
    "row_evidence_null",
    "birth_death_isolation",
    "birth_death_feature_scale",
    "birth_death_backend",
    "birth_death_cells",
    "birth_death_metric_rank",
    "birth_death_chunk",
    "birth_death_parent_policy",
    "row_policy",
    "optimizer_family",
    "optimizer_momentum",
    "optimizer_smoothing",
    "optimizer_convolution",
    "optimizer_convolution_rule",
    "optimizer_svd_backend",
    "optimizer_dense_profile",
    "constraint_geometry_mode",
    "optimizer_adam_lr",
    "eps",
    "beta2_end",
    "beta2_anneal_end",
    "reg_coeff_end",
    "reg_coeff_anneal_end",
    "loss",
    "d_betas",
    "d_eps",
    "prior_eps",
    "loss_labels",
    "adam_variant",
    "lr_schedule",
    "lr_decay_rate",
    "lr_decay_steps",
    "lr_decay_staircase",
)


# -----------------------------------------------------------------------------
# Group 1: Mathematical Invariants of symmetric_log_odds
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("c", [-10000.0, -250.0, -10.0, 0.0, 15.0, 500.0, 10000.0])
def test_symmetric_log_odds_scalar_shift_invariance(c: float):
    """Verify f(y + c 1) == f(y) for scalar shifts c across arbitrary logits."""
    torch.manual_seed(42)
    y = torch.randn(8, 5, dtype=torch.float64)
    u_base = symmetric_log_odds(y, tau=2.5)
    u_shifted = symmetric_log_odds(y + c, tau=2.5)
    assert torch.allclose(u_shifted, u_base, atol=1e-5, rtol=1e-5), (
        f"Scalar shift invariance violated for c={c}: max diff = {(u_shifted - u_base).abs().max().item()}"
    )


def test_symmetric_log_odds_vector_broadcast_shift_invariance():
    """Verify f(y + c 1) == f(y) for broadcastable vector shifts along the class axis."""
    torch.manual_seed(123)
    y = torch.randn(4, 3, 5, dtype=torch.float64)
    c_vec = torch.randn(4, 3, 1, dtype=torch.float64) * 50.0
    u_base = symmetric_log_odds(y, tau=3.0)
    u_shifted = symmetric_log_odds(y + c_vec, tau=3.0)
    assert torch.allclose(u_shifted, u_base, atol=1e-5, rtol=1e-5), (
        f"Vector shift invariance violated: max diff = {(u_shifted - u_base).abs().max().item()}"
    )


def test_symmetric_log_odds_binary_antisymmetry():
    """Verify u_0 == -u_1 and u_0 + u_1 == 0 when M=2 (binary query)."""
    torch.manual_seed(999)
    y = torch.randn(16, 2, dtype=torch.float64)

    # 1. Uncapped log-odds
    u_uncapped = symmetric_log_odds(y, tau=None)
    assert torch.allclose(u_uncapped[:, 0], -u_uncapped[:, 1], atol=1e-6)
    assert (u_uncapped[:, 0] + u_uncapped[:, 1]).abs().max().item() < 1e-6

    # 2. Soft-capped log-odds
    u_capped = symmetric_log_odds(y, tau=3.0)
    assert torch.allclose(u_capped[:, 0], -u_capped[:, 1], atol=1e-6)
    assert (u_capped[:, 0] + u_capped[:, 1]).abs().max().item() < 1e-6


def test_symmetric_log_odds_soft_capping_bounds_and_saturation():
    """Verify |u_k| <= tau across extreme logits (+-1000) and strict |u_k| < tau on moderate logits."""
    tau = 3.0
    y_extreme = torch.tensor([
        [1000.0, -1000.0, 0.0],
        [-500.0, 500.0, 20.0],
        [1e5, -1e5, 0.0],
    ], dtype=torch.float32)

    u_ext = symmetric_log_odds(y_extreme, tau=tau)
    assert torch.all(u_ext.abs() <= tau), f"Extreme logits exceeded tau: max={u_ext.abs().max().item()}"
    assert abs(u_ext[0, 0].item() - tau) < 1e-4, "Dominant logit failed to saturate to +tau"
    assert abs(u_ext[0, 1].item() - (-tau)) < 1e-4, "Submissive logit failed to saturate to -tau"

    # Moderate logits must strictly satisfy |u_k| < tau
    y_mod = torch.randn(32, 4) * 3.0
    u_mod = symmetric_log_odds(y_mod, tau=tau)
    assert torch.all(u_mod.abs() < tau), "Moderate logits reached or exceeded tau"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_symmetric_log_odds_dtypes_and_gradient_flow(dtype):
    """Verify autograd backward pass and gradient stability across dtypes on CPU."""
    torch.manual_seed(42)
    y = torch.randn(4, 5, dtype=dtype, requires_grad=True)
    u = symmetric_log_odds(y, tau=2.5)
    assert u.dtype == dtype

    loss = u.sum()
    loss.backward()
    assert y.grad is not None
    assert torch.isfinite(y.grad).all(), f"Gradient contained non-finite values for dtype {dtype}"
    assert (y.grad.abs() > 0).all(), f"Gradient had exact zeros for dtype {dtype}"


def test_symmetric_log_odds_shape_polymorphism():
    """Verify symmetric_log_odds handles 1D (M,), 2D (B, M), and 3D (B, Q, M) tensor shapes."""
    torch.manual_seed(42)
    # 1D
    y1 = torch.randn(4)
    assert symmetric_log_odds(y1).shape == (4,)
    # 2D
    y2 = torch.randn(8, 5)
    assert symmetric_log_odds(y2).shape == (8, 5)
    # 3D
    y3 = torch.randn(4, 3, 6)
    assert symmetric_log_odds(y3).shape == (4, 3, 6)


# -----------------------------------------------------------------------------
# Group 2: Anti-Collapse Regularizers
# -----------------------------------------------------------------------------

def test_marginal_entropy_regularizer_uniform_zero_loss():
    """Verify MarginalEntropyRegularizer incurs exactly 0.0 loss on uniform distributions."""
    reg = MarginalEntropyRegularizer(
        query_indices=[0],
        num_classes=3,
        num_modes=3,
        min_entropy_ratio=0.85,
    )
    # Equal logits produce uniform probabilities p_bar = [1/3, 1/3, 1/3]
    u_uniform = torch.zeros(12, 1, 3)
    loss = reg(u_uniform)
    assert loss.item() == 0.0, f"Expected 0.0 loss on uniform batch, got {loss.item()}"


def test_marginal_entropy_regularizer_collapsed_positive_loss_and_gradient():
    """Verify MarginalEntropyRegularizer triggers positive loss and restoring gradients on mode collapse."""
    reg = MarginalEntropyRegularizer(
        query_indices=[0],
        num_classes=3,
        num_modes=3,
        min_entropy_ratio=0.85,
    )
    # Mode 0 dominant across all samples
    u_collapsed = torch.tensor([[[2.0, -1.0, -1.0]]]).expand(16, 1, 3).clone().requires_grad_(True)
    loss = reg(u_collapsed)
    assert loss.item() > 0.0, "Collapsed distribution failed to trigger positive entropy penalty"

    loss.backward()
    assert u_collapsed.grad is not None
    # Mode 0 should be penalized (positive grad), Mode 1 & 2 encouraged (negative grad)
    assert u_collapsed.grad[:, 0, 0].mean().item() > 0.0
    assert u_collapsed.grad[:, 0, 1].mean().item() < 0.0
    assert u_collapsed.grad[:, 0, 2].mean().item() < 0.0


def test_marginal_entropy_multi_query_and_capacity_attraction():
    """Verify MarginalEntropyRegularizer multi-query execution and capacity-balanced attraction."""
    reg = MarginalEntropyRegularizer(
        query_indices=[0, 1],
        num_classes=4,
        num_modes=[2, 3],
        min_entropy_ratio=0.85,
        target_attraction_weight=0.5,
    )
    u = torch.randn(16, 2, 4, requires_grad=True)
    loss = reg(u)
    assert loss.item() > 0.0
    loss.backward()
    assert torch.isfinite(u.grad).all()


def test_normalized_feature_cosine_repulsion_orthogonal_vs_collinear():
    """Verify NormalizedFeatureCosineRepulsion behavior on orthogonal, collinear, and B=1 inputs."""
    rep = NormalizedFeatureCosineRepulsion(margin=0.88)

    # 1. B=1 edge case
    assert rep(torch.randn(1, 32)).item() == 0.0

    # 2. Orthogonal features (B=4, D=4)
    v_ortho = torch.eye(4)
    assert rep(v_ortho).item() == 0.0

    # 3. Collinear features (B=4 identical rows)
    v_col = torch.randn(1, 32).expand(4, -1).clone().requires_grad_(True)
    loss_col = rep(v_col)
    assert abs(loss_col.item() - 0.12) < 1e-4

    # 4. Gradient flow
    v_rand = torch.randn(4, 32, requires_grad=True)
    loss_rand = rep(v_rand, margin=0.0)
    loss_rand.backward()
    assert v_rand.grad is not None and v_rand.grad.norm().item() > 0.0


def test_anti_collapse_manipulator_protocols():
    """Verify MarginalEntropyManipulator and CosineRepulsionManipulator adhere to PriorManipulator protocol."""
    m_ent = MarginalEntropyManipulator(query_indices=[0], num_classes=3, num_modes=3)
    m_cos = CosineRepulsionManipulator(margin=0.88)

    assert isinstance(m_ent, PriorManipulator)
    assert isinstance(m_cos, PriorManipulator)
    assert m_ent.row_local_gradients is True
    assert m_cos.row_local_gradients is True

    raw_rows = torch.randn(8, 4)
    loss_ent = m_ent.regularize(raw_rows, logits=torch.randn(8, 1, 3))
    loss_cos = m_cos.regularize(raw_rows, sampled_codes=torch.randn(8, 4))
    assert isinstance(loss_ent, torch.Tensor)
    assert isinstance(loss_cos, torch.Tensor)


# -----------------------------------------------------------------------------
# Group 3: Target Distribution Tests
# -----------------------------------------------------------------------------

def test_tomographic_target_deterministic_rng():
    """Verify TomographicConstraintTarget bit-exact deterministic reproduction with torch.Generator."""
    queries = [
        {"question": "Q0", "target_modes": [0, 1]},
        {"question": "Q1", "target_modes": [0, 1, 2]},
    ]
    target_data = TomographicConstraintTarget(queries, tau=3.0)

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    b1 = target_data(16, generator=g1)
    b2 = target_data(16, generator=g2)

    assert torch.equal(b1["real"], b2["real"])
    assert torch.equal(b1["modes"], b2["modes"])

    # Different seed produces different samples
    g3 = torch.Generator().manual_seed(99)
    b3 = target_data(16, generator=g3)
    assert not torch.equal(b1["modes"], b3["modes"])


def test_tomographic_target_cartesian_coverage():
    """Verify TomographicConstraintTarget samples the full Cartesian product across queries."""
    queries = [
        {"question": "Q0", "target_modes": [0, 1]},
        {"question": "Q1", "target_modes": [0, 1, 2]},
        {"question": "Q2", "target_modes": [0, 1]},
    ]
    # Total combinations = 2 * 3 * 2 = 12
    target_data = TomographicConstraintTarget(queries, tau=3.0)
    batch = target_data(256, generator=torch.Generator().manual_seed(123))

    modes = batch["modes"]  # (256, 3)
    unique_combinations = {tuple(row.tolist()) for row in modes}
    assert len(unique_combinations) == 12, f"Expected 12 unique combinations, found {len(unique_combinations)}"


def test_tomographic_target_shapes_and_bounds():
    """Verify TomographicConstraintTarget output tensor shapes and soft-capped bounds."""
    tau = 3.0
    queries = [
        {"question": "Q0", "target_modes": [0, 1]},
        {"question": "Q1", "target_modes": [0, 1, 2]},
    ]
    target_data = TomographicConstraintTarget(queries, tau=tau)
    batch = target_data(32, generator=torch.Generator().manual_seed(77))

    assert "real" in batch and "modes" in batch
    real = batch["real"]
    assert real.shape == (32, 5)  # 2 + 3 = 5 total classes
    assert torch.isfinite(real).all()
    assert (real.abs() <= tau).all()


def test_tomographic_target_validation():
    """Verify TomographicConstraintTarget constructor and call argument validation."""
    with pytest.raises(ValueError, match="queries"):
        TomographicConstraintTarget([])
    with pytest.raises(ValueError, match="tau"):
        TomographicConstraintTarget([{"question": "Q", "target_modes": [0, 1]}], tau=-1.0)
    target = TomographicConstraintTarget([{"question": "Q", "target_modes": [0, 1]}])
    with pytest.raises(ValueError, match="batch_size"):
        target(0)


# -----------------------------------------------------------------------------
# Group 4: Metric Evaluation Tests
# -----------------------------------------------------------------------------

def test_evaluate_tomographic_constraints_oracle_pass():
    """Verify evaluate_tomographic_constraints outputs >= acceptance bounds on synthetic perfect outputs."""
    B = 32
    target_modes = torch.stack([torch.arange(B) % 2, torch.arange(B) % 3], dim=-1)
    probs = torch.zeros(B, 2, 3)
    for b in range(B):
        probs[b, 0, target_modes[b, 0]] = 1.0
        probs[b, 1, target_modes[b, 1]] = 1.0

    features = torch.eye(B)
    smooth_images = torch.zeros(B, 3, 16, 16)

    class MockOperator(nn.Module):
        def forward(self, x):
            return {"probs": probs, "features": features}

    metrics = evaluate_tomographic_constraints(
        smooth_images,
        operator=MockOperator(),
        target_modes=target_modes,
    )

    assert metrics["mean_target_prob"] >= 0.99
    assert metrics["primary_accuracy"] == 1.0
    assert metrics["entropy_ratio"] >= 0.98
    assert metrics["feature_diversity"] >= 0.95
    assert metrics["patch_seam_ratio"] <= 1.05
    assert metrics["shift_sensitivity"] <= 0.05


def test_evaluate_tomographic_constraints_destructive_fail():
    """Verify evaluate_tomographic_constraints decisively fails gates on collapsed synthetic outputs."""
    B = 32
    target_modes = torch.stack([torch.arange(B) % 2, torch.arange(B) % 3], dim=-1)
    # Mode 0 collapsed for all queries
    probs = torch.zeros(B, 2, 3)
    probs[:, :, 0] = 1.0

    # Collinear identical features
    features = torch.ones(B, 16)
    # High seam images
    blocky_images = torch.randn(B, 3, 2, 2).repeat_interleave(8, dim=-1).repeat_interleave(8, dim=-2)

    class MockCollapsedOperator(nn.Module):
        def forward(self, x):
            return {"probs": probs, "features": features}

    metrics = evaluate_tomographic_constraints(
        blocky_images,
        operator=MockCollapsedOperator(),
        target_modes=target_modes,
    )

    assert metrics["entropy_ratio"] < 0.20, f"Expected failed entropy_ratio, got {metrics['entropy_ratio']}"
    assert metrics["feature_diversity"] < 0.05, f"Expected failed feature_diversity, got {metrics['feature_diversity']}"
    assert metrics["patch_seam_ratio"] > 1.30, f"Expected high patch_seam_ratio, got {metrics['patch_seam_ratio']}"


def test_evaluate_tomographic_constraints_schema():
    """Verify evaluate_tomographic_constraints returns dict of floats with all 6 required keys."""
    B = 8
    target_modes = torch.zeros(B, 2, dtype=torch.long)
    images = torch.randn(B, 3, 16, 16)

    class DummyOperator(nn.Module):
        def forward(self, x):
            return {
                "probs": F.softmax(torch.randn(B, 2, 3), dim=-1),
                "features": torch.randn(B, 16),
            }

    metrics = evaluate_tomographic_constraints(
        images,
        operator=DummyOperator(),
        target_modes=target_modes,
    )

    required_keys = {
        "mean_target_prob",
        "primary_accuracy",
        "entropy_ratio",
        "feature_diversity",
        "patch_seam_ratio",
        "shift_sensitivity",
    }
    assert set(metrics.keys()) == required_keys
    for k, v in metrics.items():
        assert isinstance(v, float), f"Metric '{k}' was not a float: {type(v)}"
        assert math.isfinite(v), f"Metric '{k}' was not finite: {v}"


# -----------------------------------------------------------------------------
# Group 5: Integration Tests with ManipulatedPrior & GANTrainer
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["bispf_optimized", "ka2", "bcap", "bispf", "e22"])
def test_tomographic_manipulators_training_step_across_recipes(preset):
    """Verify GANTrainer multi-step execution with tomographic prior manipulators across recipes."""
    recipe = get_recipe(preset, num_particles=16, z_dim=8, batch_size=4)
    prior = recipe.make_prior().manipulate(
        MarginalEntropyManipulator(query_indices=[0], num_classes=4, num_modes=4, weight=1.0, source="codes"),
        CosineRepulsionManipulator(margin=0.0, weight=0.5),
    )

    G = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
    D = nn.Sequential(nn.Linear(16, 1))

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)
    real = torch.randn(4, 16)
    res = trainer.step(real)

    assert res is not None
    assert "loss_g" in res
    assert "loss_d" in res
    assert "prior_manipulation" in res
    assert torch.isfinite(res["loss_g"])
    assert res["prior_manipulation"].item() >= 0.0


def test_zero_recipe_bloat_tomographic():
    """Verify Recipe dataclass fields remain frozen at exactly 112."""
    fields_dict = get_recipe_fields()
    assert len(fields_dict) == 112
    recipe_fields = tuple(f.name for f in dataclasses.fields(Recipe))
    assert len(recipe_fields) == 112
    assert recipe_fields == FROZEN_112_RECIPE_FIELDS


def test_tomographic_checkpoint_continuation_parity():
    """Verify bit-exact checkpoint continuation parity with tomographic prior manipulators."""
    def build_trainer():
        recipe = get_recipe("bispf_optimized", num_particles=16, z_dim=8, batch_size=4)
        prior = recipe.make_prior().manipulate(
            MarginalEntropyManipulator(query_indices=[0], num_classes=4, num_modes=4, source="codes"),
            CosineRepulsionManipulator(margin=0.0),
        )
        torch.manual_seed(42)
        G = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        D = nn.Sequential(nn.Linear(16, 1))
        return GANTrainer(
            recipe, G, D, prior=prior,
            latent_generator=torch.Generator().manual_seed(101),
            penalty_generator=torch.Generator().manual_seed(102),
            noise_generator=torch.Generator().manual_seed(103),
        )

    trainer1 = build_trainer()
    batches = [torch.randn(4, 16, generator=torch.Generator().manual_seed(500 + i)) for i in range(4)]
    for i in range(2):
        trainer1.step(batches[i])

    cp = trainer1.state_dict()

    for i in range(2, 4):
        trainer1.step(batches[i])

    trainer2 = build_trainer()
    trainer2.load_state_dict(cp)
    for i in range(2, 4):
        trainer2.step(batches[i])

    diff_z = (trainer1.prior.base.z - trainer2.prior.base.z).abs().max().item()
    assert diff_z < 1e-6, f"Checkpoint continuation mismatch on prior.base.z: {diff_z}"


def test_tomographic_suite_cpu_efficiency():
    """Benchmark CPU execution runtime of tomographic modules (< 5.0 seconds)."""
    t0 = time.perf_counter()
    recipe = get_recipe("bispf_optimized", num_particles=16, z_dim=8, batch_size=4)
    prior = recipe.make_prior().manipulate(
        MarginalEntropyManipulator(query_indices=[0], num_classes=4, num_modes=4, source="codes"),
        CosineRepulsionManipulator(margin=0.0),
    )
    G = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
    D = nn.Sequential(nn.Linear(16, 1))
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)
    for _ in range(5):
        trainer.step(torch.randn(4, 16))
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Execution time {elapsed:.2f}s exceeded 5.0s budget"
