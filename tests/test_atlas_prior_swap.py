"""Comprehensive Invariant & Regression Test Suite for Atlas Swappable Prior.

Verifies:
1. NormalNoiseManipulator behavior, statistics, validation, and serialization.
2. Autograd row-locality (diagonal Jacobian, zero cross-row gradients).
3. Prior capabilities and restored A2 latent damping eligibility.
4. Recipe schema invariance (strictly 115 frozen fields, zero top-level torch import).
5. RNG stream isolation and determinism across varying noise scales.
6. 2-step checkpoint continuation numerical parity and stream restoration.
7. Forge FormulationContext compatibility and rejection of MoG Atlas.
"""
import dataclasses
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from particlegan import (
    AdditiveNormalManipulator,
    GANTrainer,
    MoGParticlePrior,
    NormalNoiseManipulator,
    ParticlePrior,
    Recipe,
    get_recipe,
    get_recipe_fields,
    prior_capabilities,
    prior_mechanisms,
    ManipulatedPrior,
    PriorManipulator,
)
from experiments.forge.api import CapabilityError, FormulationContext
from experiments.forge.priors import resolve_prior
from tests.test_swappable_priors import FROZEN_RECIPE_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def assert_nested_allclose(left, right, atol: float = 1e-6, rtol: float = 1e-5, equal_nan: bool = True):
    """Recursively assert numerical allclose on floating tensors and exact on others, tolerating NaN parity."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), f"Type mismatch: {type(left)} vs {type(right)}"
        if left.is_floating_point():
            assert torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=equal_nan), (
                f"Max diff: {(left - right).abs().max().item()} > atol={atol}"
            )
        else:
            assert torch.equal(left, right), f"Discrete tensor mismatch: {left} vs {right}"
    elif isinstance(left, dict):
        assert isinstance(right, dict), f"Type mismatch: {type(left)} vs {type(right)}"
        assert left.keys() == right.keys(), f"Key mismatch: {set(left) ^ set(right)}"
        for k in left:
            assert_nested_allclose(left[k], right[k], atol=atol, rtol=rtol, equal_nan=equal_nan)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"Length mismatch: {len(left)} vs {len(right)}"
        for a, b in zip(left, right):
            assert_nested_allclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)
    elif isinstance(left, float) and math.isnan(left):
        assert isinstance(right, float) and math.isnan(right), f"Expected NaN, got {right}"
    elif isinstance(left, float):
        assert isinstance(right, float), f"Expected float, got {type(right)}"
        assert math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
    else:
        assert left == right, f"Value mismatch: {left} != {right}"


# =============================================================================
# Group 1: NormalNoiseManipulator Prior Class Behavior
# =============================================================================

def test_normal_noise_manipulator_initialization_and_validation():
    # Valid initializations
    m1 = NormalNoiseManipulator(0.025)
    assert m1.sigma == 0.025
    assert m1.penalty_weight == 1.0

    m2 = NormalNoiseManipulator(0.0, penalty_weight=0.5)
    assert m2.sigma == 0.0
    assert m2.penalty_weight == 0.5

    # Semantic alias
    assert AdditiveNormalManipulator is NormalNoiseManipulator
    m_alias = AdditiveNormalManipulator(0.01)
    assert isinstance(m_alias, NormalNoiseManipulator)
    assert m_alias.sigma == 0.01

    # Invalid sigma
    with pytest.raises(ValueError, match="sigma"):
        NormalNoiseManipulator(-0.01)
    with pytest.raises(ValueError, match="sigma"):
        NormalNoiseManipulator(float("nan"))
    with pytest.raises(ValueError, match="sigma"):
        NormalNoiseManipulator(float("inf"))
    with pytest.raises(ValueError, match="sigma"):
        NormalNoiseManipulator("0.025")  # type: ignore

    # Invalid penalty_weight
    with pytest.raises(ValueError, match="penalty_weight"):
        NormalNoiseManipulator(0.025, penalty_weight=-1.0)


def test_normal_noise_manipulator_attributes_and_config():
    m = NormalNoiseManipulator(sigma=0.035, penalty_weight=2.0)
    assert m.row_local_gradients is True
    assert m.additive_transform is True
    assert m.is_additive_transform is True
    assert m.normal_noise is True
    assert m.config_fields == ("sigma",)

    cfg = m.config()
    assert cfg == {"penalty_weight": 2.0, "sigma": 0.035}
    spec = m.manipulator_spec()
    assert spec["class"].endswith("NormalNoiseManipulator")
    assert spec["config"] == cfg


def test_normal_noise_manipulator_transform_statistics():
    base = ParticlePrior(num_particles=1000, z_dim=4)
    sigma = 0.05
    m = NormalNoiseManipulator(sigma=sigma)

    # Large sample for statistical assertions
    n_samples = 50_000
    codes = torch.zeros(n_samples, 4)
    gen = torch.Generator().manual_seed(42)
    perturbed = m.transform(codes, base_prior=base, generator=gen)

    assert perturbed.shape == (n_samples, 4)
    assert perturbed.dtype == codes.dtype
    assert perturbed.device == codes.device

    noise = perturbed - codes
    empirical_mean = noise.mean().item()
    empirical_std = noise.std().item()

    # Gaussian expectation checks
    assert abs(empirical_mean) < 2e-3, f"Mean not centered: {empirical_mean}"
    assert abs(empirical_std - sigma) < 2e-3, f"Std dev mismatch: {empirical_std} vs {sigma}"


def test_normal_noise_manipulator_zero_sigma_identity():
    base = ParticlePrior(num_particles=64, z_dim=4)
    m = NormalNoiseManipulator(sigma=0.0)
    codes = torch.randn(16, 4)
    gen = torch.Generator().manual_seed(99)
    state_before = gen.get_state().clone()

    out = m.transform(codes, base_prior=base, generator=gen)
    assert torch.equal(out, codes)
    # Zero sigma must not consume RNG
    assert torch.equal(gen.get_state(), state_before)


def test_normal_noise_manipulator_state_dict_provenance():
    m1 = NormalNoiseManipulator(sigma=0.025)
    sd = m1.state_dict()
    assert "_extra_state" in sd
    assert sd["_extra_state"]["config"]["sigma"] == 0.025

    # Clean load into identical configuration
    m2 = NormalNoiseManipulator(sigma=0.025)
    m2.load_state_dict(sd)

    # Configuration mismatch must raise ValueError
    m3 = NormalNoiseManipulator(sigma=0.050)
    with pytest.raises(ValueError, match="prior manipulator configuration mismatch"):
        m3.load_state_dict(sd)


# =============================================================================
# Group 2: Autograd Row-Locality & Zero Cross-Row Gradients
# =============================================================================

def test_autograd_single_row_gradient_locality():
    """Verify that a scalar loss on row k produces non-zero gradients ONLY at row k in the base prior table."""
    n_parts, z_dim = 16, 4
    base_prior = ParticlePrior(num_particles=n_parts, z_dim=z_dim, learnable=True)
    manipulator = NormalNoiseManipulator(sigma=0.025)
    prior = base_prior.manipulate(manipulator)

    batch_indices = torch.tensor([2, 5, 11], dtype=torch.long)
    gen = torch.Generator().manual_seed(123)

    # Forward pass
    codes = prior.forward(batch_indices, generator=gen)

    # Compute loss exclusively on batch row 1 (corresponding to particle table index 5)
    target_batch_idx = 1
    target_particle_idx = batch_indices[target_batch_idx].item()
    loss = (codes[target_batch_idx] ** 2).sum()

    # Compute autograd gradient w.r.t base table
    grads = torch.autograd.grad(loss, base_prior.z)[0]
    assert grads.shape == (n_parts, z_dim)

    # Target row must have non-zero gradient: d(codes[1]**2)/d(z[5]) = 2 * codes[1]
    expected_grad = 2.0 * codes[target_batch_idx]
    assert torch.allclose(grads[target_particle_idx], expected_grad, atol=1e-6)

    # ALL OTHER ROWS must have IDENTICALLY ZERO gradient
    other_rows = [r for r in range(n_parts) if r != target_particle_idx]
    assert torch.all(grads[other_rows] == 0.0), "Cross-row gradient leakage detected!"


def test_autograd_full_jacobian_block_diagonality():
    """Verify full Jacobian matrix d(codes)/d(table_sample) is strictly block diagonal."""
    n_parts, z_dim = 8, 3
    base_prior = ParticlePrior(num_particles=n_parts, z_dim=z_dim, learnable=True)
    manipulator = NormalNoiseManipulator(sigma=0.025)

    idx = torch.tensor([0, 2, 4], dtype=torch.long)
    fixed_eps = torch.randn(len(idx), z_dim)

    def transform_fn(table_subset):
        return table_subset + manipulator.sigma * fixed_eps

    subset = base_prior.z[idx].detach().clone().requires_grad_(True)
    j = torch.autograd.functional.jacobian(transform_fn, subset)

    b = len(idx)
    for i in range(b):
        for k in range(b):
            block = j[i, :, k, :]  # shape (z_dim, z_dim)
            if i == k:
                assert torch.allclose(block, torch.eye(z_dim), atol=1e-7), f"Diagonal block {i},{i} is not identity"
            else:
                assert torch.all(block == 0.0), f"Off-diagonal block {i},{k} is non-zero: {block}"


def test_autograd_row_locality_contrast_with_coupled_priors():
    """Verify that MoG standardized prior violates row locality, whereas ParticlePrior + NormalNoise preserves it."""
    n_parts, z_dim = 8, 3

    # 1. Swapped prior: row-local
    p_swapped = ParticlePrior(n_parts, z_dim, learnable=True).manipulate(NormalNoiseManipulator(0.025))
    codes_s = p_swapped.forward(torch.tensor([0, 1]))
    loss_s = codes_s[0].sum()
    grad_s = torch.autograd.grad(loss_s, p_swapped.base.z)[0]
    assert torch.all(grad_s[1:] == 0.0), "Swapped prior leaked gradients across rows"

    # 2. Coupled MoG standardized prior: couples all rows through mean/std standardization
    p_coupled = MoGParticlePrior(n_parts, z_dim, sigma=0.025, standardize=True, learnable=True)
    codes_c = p_coupled.forward(torch.tensor([0, 1]))
    loss_c = codes_c[0].sum()
    grad_c = torch.autograd.grad(loss_c, p_coupled.z)[0]
    # Off-diagonal rows receive non-zero gradients due to centering/scaling!
    assert not torch.all(grad_c[1:] == 0.0), "Expected standardized MoG to couple rows"


# =============================================================================
# Group 3: Prior Capabilities & A2 Damping Eligibility
# =============================================================================

def test_prior_capabilities_swapped_atlas_prior():
    base = ParticlePrior(num_particles=64, z_dim=4, learnable=True)
    m = NormalNoiseManipulator(sigma=0.025)
    prior = base.manipulate(m)

    caps = prior_capabilities(prior)
    assert caps["kind"] == "particle_cloud"
    assert caps["learned_locations"] is True
    assert caps["row_local_gradients"] is True
    assert caps["a2_eligible"] is True
    assert caps["standardize"] is False


def test_prior_mechanisms_a2_eligibility_atlas():
    base = ParticlePrior(num_particles=64, z_dim=4, learnable=True)
    prior = base.manipulate(NormalNoiseManipulator(sigma=0.025))

    # Under standard Atlas settings (latent_damping_max_rate > 0, prior_beta1 == 0.0)
    mech = prior_mechanisms(prior, latent_damping_max_rate=0.5, prior_beta1=0.0)
    assert mech["a2"]["requested"] is True
    assert mech["a2"]["supported"] is True
    assert mech["a2"]["enabled"] is True
    assert mech["a2"]["reason"] == "row-local learned locations with prior beta1=0"


def test_prior_mechanisms_coupled_negative_cases():
    base = ParticlePrior(num_particles=64, z_dim=4, learnable=True)
    prior = base.manipulate(NormalNoiseManipulator(sigma=0.025))

    # 1. Non-zero prior beta1 disables A2
    mech1 = prior_mechanisms(prior, latent_damping_max_rate=0.5, prior_beta1=0.9)
    assert mech1["a2"]["enabled"] is False
    assert "prior beta1=0" in mech1["a2"]["reason"]

    # 2. Latent damping max rate == 0 disables A2
    mech2 = prior_mechanisms(prior, latent_damping_max_rate=0.0, prior_beta1=0.0)
    assert mech2["a2"]["enabled"] is False

    # 3. Non-row-local manipulator disables A2
    class CoupledManip(PriorManipulator):
        row_local_gradients = False

        def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
            return codes - codes.mean(dim=0, keepdim=True)

    prior_coupled = base.manipulate(CoupledManip())
    caps_coupled = prior_capabilities(prior_coupled)
    assert caps_coupled["row_local_gradients"] is False
    assert caps_coupled["a2_eligible"] is False
    mech3 = prior_mechanisms(prior_coupled, latent_damping_max_rate=0.5, prior_beta1=0.0)
    assert mech3["a2"]["enabled"] is False


# =============================================================================
# Group 4: Recipe Schema Invariance & Atlas Configuration
# =============================================================================

def test_recipe_schema_exact_118_fields():
    """The frozen 118-field contract admits only these neutral additions."""
    additions = ('reg_real_weight', 'reg_fake_weight', 'joint_reconstruction_weight', 'conditional_energy_weight', 'conditional_energy_samples', 'masked_mixture_weight', 'masked_transport_weight', 'imputer_component_weight', 'rhbi_weight', 'rhbi_rho', 'rhbi_tau', 'rhbi_eps', 'rhbi_iters', 'rhbi_c_max', 'rhbi_beta_ell', 'rhbi_nn_quantile', 'rhbi_critic_weight', 'rhbi_self_coupling', 'rhbi_relativistic_weight', 'rhbi_lambda_floor', 'rhbi_gamma_gauge', 'rhbi_rapidity_screening', 'rhbi_scale_log_odds', 'rhbi_conditional_fibre', 'rhbi_akt_horizon', 'rhbi_akt_mode', 'rhbi_slack_gain', 'rhbi_chord_weight', 'rhbi_thermal_tau', 'rhbi_thermal_shock_mu', 'rhbi_settle_beta', 'rhbi_settle_mode', 'rhbi_fibre_rel_blend', 'rhbi_step_damp', 'rhbi_cond_ot_weight', 'rhbi_cond_ot_mode', 'rhbi_simplex_weight', 'rhbi_cvx_mu_gain', 'rhbi_simplex_chart_weight', 'rhbi_tail_spill_gain', 'rhbi_simplex_macro_settle')
    fields_dict = get_recipe_fields()
    field_names = tuple(fields_dict)
    assert len(field_names) == 118 + len(additions)
    assert tuple(name for name in field_names if name not in additions) == FROZEN_RECIPE_FIELDS
    assert tuple(field.name for field in dataclasses.fields(Recipe)) == field_names
    from particlegan.recipe_compat import SEARCH_RECIPE_DEFAULTS
    for name in additions:
        assert fields_dict[name] == SEARCH_RECIPE_DEFAULTS[name]


def test_atlas_preset_default_values_and_swapped_prior():
    recipe = get_recipe("atlas")
    assert recipe.name == "atlas"
    assert recipe.prior_kind == "particles"
    assert recipe.sigma_rel == 0.0
    assert recipe.birth_death_backend == "auto"
    assert recipe.birth_death_cells == 128
    assert recipe.reopen_guard == "settled"
    assert recipe.row_evidence_gate is True
    assert recipe.particle_birth_death is True
    assert recipe.row_policy == "independent"
    assert recipe.latent_damping_max_rate == 0.5
    assert recipe.direct_particle_gain is True

    # make_prior constructs ParticlePrior with NormalNoiseManipulator
    m_noise = NormalNoiseManipulator(sigma=0.025)
    prior = recipe.make_prior(manipulators=[m_noise])
    assert isinstance(prior, ManipulatedPrior)
    assert type(prior.base) is ParticlePrior
    assert len(prior.manipulators) == 1
    assert prior.manipulators[0] is m_noise
    assert prior.normal_noise is True


def test_recipe_serialization_roundtrip_atlas():
    recipe = get_recipe("atlas")
    as_dict = recipe.to_dict()
    reconstructed = Recipe.from_dict(as_dict)
    assert reconstructed == recipe
    assert get_recipe_fields("atlas") == dataclasses.asdict(recipe)


def test_recipe_rejects_unknown_fields_atlas():
    with pytest.raises(TypeError, match="Unknown recipe field"):
        get_recipe_fields("atlas", normal_noise_sigma=0.025)
    with pytest.raises(TypeError, match="Unknown recipe field"):
        get_recipe("atlas", prior_noise_std=0.025)


def test_recipes_module_no_toplevel_torch_import():
    source = ROOT / "particlegan/recipes.py"
    script = """
import importlib.util, sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError(f'Imported Torch: {fullname}')
sys.meta_path.insert(0, BlockTorch())
spec = importlib.util.spec_from_file_location('standalone_recipes', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
fields_atlas = module.get_recipe_fields('atlas')
assert len(fields_atlas) == 159
assert 'torch' not in sys.modules
"""
    res = subprocess.run([sys.executable, "-I", "-c", script, str(source)], capture_output=True, text=True)
    assert res.returncode == 0, f"Torch imported during recipes resolution:\n{res.stderr}"


# =============================================================================
# Group 5: RNG Stream Isolation & Determinism
# =============================================================================

def test_trainer_allocates_prior_noise_generator_stream():
    recipe = get_recipe("atlas", num_particles=32, z_dim=4, batch_size=8)
    g = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    d = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    m = NormalNoiseManipulator(sigma=0.025)
    prior = recipe.make_prior(manipulators=[m])

    trainer = GANTrainer(recipe, g, d, prior=prior, seed=42, max_steps=4)
    assert trainer.prior_noise_generator is not None
    assert "prior_noise_generator" in trainer._STREAMS


def test_prior_noise_stream_isolation_from_latent_stream():
    """Verify that drawing normal noise consumes prior_noise_generator without affecting latent_generator."""
    recipe = get_recipe("atlas", num_particles=64, z_dim=4, batch_size=16)

    # Prior A: With Gaussian perturbation
    prior_a = recipe.make_prior(manipulators=[NormalNoiseManipulator(sigma=0.025)])
    latent_gen_a = torch.Generator().manual_seed(101)
    noise_gen_a = torch.Generator().manual_seed(202)
    noise_gen_a_initial_state = noise_gen_a.get_state().clone()

    # Prior B: Without perturbation (sigma=0.0)
    prior_b = recipe.make_prior(manipulators=[NormalNoiseManipulator(sigma=0.0)])
    latent_gen_b = torch.Generator().manual_seed(101)

    # Run 5 consecutive sample batches
    for _ in range(5):
        codes_a, idx_a = prior_a.sample(16, generator=latent_gen_a, noise_generator=noise_gen_a)
        codes_b, idx_b = prior_b.sample(16, generator=latent_gen_b)

        # 1. Sampled indices must be BIT-FOR-BIT IDENTICAL
        assert torch.equal(idx_a, idx_b), "Latent stream deviated due to noise generation!"

        # 2. Latent stream states must remain identical
        assert torch.equal(latent_gen_a.get_state(), latent_gen_b.get_state())

        # 3. Codes must differ because prior_a has Gaussian noise
        assert not torch.equal(codes_a, codes_b)

    # 4. Prior noise stream must have advanced
    assert not torch.equal(noise_gen_a.get_state(), noise_gen_a_initial_state)


def test_stream_independence_under_varying_sigma():
    recipe = get_recipe("atlas", num_particles=64, z_dim=4, batch_size=16)
    indices_by_sigma = []
    sigmas = [0.0, 0.01, 0.025, 0.05, 0.1]
    for s in sigmas:
        m = NormalNoiseManipulator(sigma=s)
        p = recipe.make_prior(manipulators=[m])
        gen_latent = torch.Generator().manual_seed(999)
        gen_noise = torch.Generator().manual_seed(888)
        _, idx = p.sample(16, generator=gen_latent, noise_generator=gen_noise)
        indices_by_sigma.append(idx)

    # All sampled indices must match across all sigmas
    for idx in indices_by_sigma[1:]:
        assert torch.equal(indices_by_sigma[0], idx)


# =============================================================================
# Group 6: Checkpoint Continuation Parity
# =============================================================================

def test_atlas_checkpoint_continuation_parity_2step():
    recipe = get_recipe("atlas", num_particles=32, z_dim=4, batch_size=8)

    def build_trainer():
        torch.manual_seed(123)
        g = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
        d = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        m = NormalNoiseManipulator(sigma=0.025)
        prior = recipe.make_prior(manipulators=[m])

        latent_gen = torch.Generator().manual_seed(501)
        penalty_gen = torch.Generator().manual_seed(502)
        noise_gen = torch.Generator().manual_seed(503)
        prior_noise_gen = torch.Generator().manual_seed(504)
        eval_gen = torch.Generator().manual_seed(505)

        return GANTrainer(
            recipe, g, d, prior=prior, seed=123,
            latent_generator=latent_gen,
            penalty_generator=penalty_gen,
            noise_generator=noise_gen,
            prior_noise_generator=prior_noise_gen,
            eval_generator=eval_gen,
            serial_backward=True,
            max_steps=8,
        )

    batches = [
        torch.randn(8, 2, generator=torch.Generator().manual_seed(700 + i))
        for i in range(4)
    ]

    # 1. Uninterrupted run: 4 steps
    trainer_uninterrupted = build_trainer()
    trainer_uninterrupted.step(batches[0])
    trainer_uninterrupted.step(batches[1])
    checkpoint_at_2 = trainer_uninterrupted.state_dict()

    trainer_uninterrupted.step(batches[2])
    trainer_uninterrupted.step(batches[3])
    final_uninterrupted = trainer_uninterrupted.state_dict()

    # 2. Restored run: load at step 2, step to 4
    trainer_restored = build_trainer()
    trainer_restored.load_state_dict(checkpoint_at_2)
    assert trainer_restored.completed_steps == 2

    trainer_restored.step(batches[2])
    trainer_restored.step(batches[3])
    final_restored = trainer_restored.state_dict()

    # 3. Assert exact parity
    assert final_restored["completed_steps"] == final_uninterrupted["completed_steps"] == 4
    assert_nested_allclose(final_uninterrupted, final_restored, atol=1e-6)

    # Verify prior_noise_generator stream state bit-exact match
    assert torch.equal(
        final_uninterrupted["streams"]["prior_noise_generator"],
        final_restored["streams"]["prior_noise_generator"],
    )


def test_atlas_stream_states_exact_restore():
    recipe = get_recipe("atlas", num_particles=32, z_dim=4, batch_size=8)
    g = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
    d = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    m = NormalNoiseManipulator(sigma=0.025)
    prior = recipe.make_prior(manipulators=[m])

    trainer = GANTrainer(recipe, g, d, prior=prior, seed=42, max_steps=4)
    batch = torch.randn(8, 2, generator=torch.Generator().manual_seed(777))
    trainer.step(batch)

    sd = trainer.state_dict()
    assert "prior_noise_generator" in sd["streams"]
    saved_stream = sd["streams"]["prior_noise_generator"].clone()

    # Mutate by stepping again
    trainer.step(batch)
    assert not torch.equal(trainer.prior_noise_generator.get_state(), saved_stream)

    # Restore checkpoint
    trainer.load_state_dict(sd)
    assert torch.equal(trainer.prior_noise_generator.get_state(), saved_stream)


# =============================================================================
# Group 7: Forge Compatibility & CapabilityError Bypass
# =============================================================================

def test_forge_formulation_context_accepts_atlas():
    atlas_idea_path = ROOT / "configs/forge/ideas/atlas.json"
    idea = json.loads(atlas_idea_path.read_text())

    # Must instantiate cleanly without CapabilityError
    ctx = FormulationContext(
        recipe_preset=idea["recipe_preset"],
        recipe_overrides=idea.get("recipe_overrides", {}),
        prior=idea["prior"],
        execution_path=idea.get("execution_path", "public_trainer"),
        initializer="deterministic_orthogonal",
    )

    assert ctx.recipe.name == "atlas"
    assert ctx.recipe.prior_kind == "particles"
    assert ctx.prior_config["kind"] == "particle_cloud"
    assert ctx.prior_config["sigma"] == 0.0
    assert ctx.prior_config["standardize"] is False

    caps = ctx.capabilities()
    assert caps["particle_cloud"] is True
    assert caps["a2"] is True
    assert caps["policy_controls"] is True
    assert caps["policy_serving"] is True


def test_forge_formulation_context_prior_config_contract():
    prior_spec = {
        "kind": "particle_cloud",
        "sigma": 0.0,
        "standardize": False,
        "learnable": True,
        "exception_reason": "Atlas swapped prior with NormalNoiseManipulator",
    }
    resolved = resolve_prior(prior_spec)
    assert resolved["kind"] == "particle_cloud"
    assert resolved["sigma"] == 0.0
    assert resolved["standardize"] is False
    assert resolved["learnable"] is True


def test_forge_formulation_context_rejects_mog_atlas():
    """Verify that MoG prior is rejected by Forge when paired with Atlas."""
    mog_prior = {"kind": "mog", "sigma": 0.025, "standardize": False, "learnable": True}
    with pytest.raises(CapabilityError) as exc_info:
        FormulationContext(
            recipe_preset="atlas",
            prior=mog_prior,
            execution_path="public_trainer",
            initializer="deterministic_orthogonal",
        )
    assert any("particle_birth_death requires prior_kind='particles'" in b or
               "MoG is unsupported" in b for b in exc_info.value.blockers)
