import dataclasses
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import (
    GANTrainer,
    MoGParticlePrior,
    ParticlePrior,
    Recipe,
    UpdatePolicy,
    get_recipe,
    get_recipe_fields,
    learning_rate_scales,
    scale_learning_rates,
)
from particlegan.capabilities import prior_capabilities, prior_mechanisms
from particlegan.priors import (
    ManipulatedPrior,
    PriorManipulator,
    SimplexRoutingManipulator,
    StratifiedSamplingManipulator,
)

ROOT = Path(__file__).resolve().parents[1]

FROZEN_RECIPE_FIELDS = (
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
    "ebi_beta",
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
    "generator_gradient_response",
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
    "bispf_shape_gates",
    "bispf_gate_overrides",
    "prior_code_law",
    "kappa_radial",
)


class MockTrainableManipulator(PriorManipulator):
    """A minimal manipulator owning trainable 2D parameters for role routing tests."""
    def __init__(self, in_features: int = 4, out_features: int = 4):
        super().__init__()
        # Must be 2D matrix (ndim == 2) for DualNorm compatibility
        self.matrix = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.orthogonal_(self.matrix)

    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        return codes @ self.matrix

    def regularize(self, raw_rows, *, sampled_codes=None, indices=None, generated=None, logits=None):
        return 0.01 * (self.matrix ** 2).sum()


class MockNonRowLocalManipulator(PriorManipulator):
    """A manipulator with cross-batch coupling declaring row_local_gradients = False."""
    row_local_gradients: bool = False

    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        return codes - codes.mean(dim=0, keepdim=True)


class MockPostprocessManipulator(PriorManipulator):
    """A manipulator attaching postprocessing to generated samples."""
    def postprocess_generated(self, generated):
        return generated * 1.0


def assert_nested_allclose(left, right, atol: float = 1e-6, rtol: float = 1e-5):
    """Recursively assert numerical allclose on floating tensors and exact on others."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), f"Type mismatch: {type(left)} vs {type(right)}"
        if left.is_floating_point():
            assert torch.allclose(left, right, atol=atol, rtol=rtol), (
                f"Max diff: {(left - right).abs().max().item()} > atol={atol}"
            )
        else:
            assert torch.equal(left, right), f"Discrete tensor mismatch: {left} vs {right}"
    elif isinstance(left, dict):
        assert isinstance(right, dict), f"Type mismatch: {type(left)} vs {type(right)}"
        assert left.keys() == right.keys(), f"Key mismatch: {set(left) ^ set(right)}"
        for k in left:
            assert_nested_allclose(left[k], right[k], atol=atol, rtol=rtol)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"Length mismatch: {len(left)} vs {len(right)}"
        for a, b in zip(left, right):
            assert_nested_allclose(a, b, atol=atol, rtol=rtol)
    else:
        assert left == right, f"Value mismatch: {left} != {right}"


# -----------------------------------------------------------------------------
# Group 1: Zero Recipe Bloat & Schema Invariance
# -----------------------------------------------------------------------------

def test_recipe_fields_exact_tuple_invariance():
    """Prior manipulators retain the frozen base schema plus explicit controls."""
    additions = ('reg_real_weight', 'reg_fake_weight', 'joint_reconstruction_weight', 'conditional_energy_weight', 'conditional_energy_samples', 'masked_mixture_weight', 'masked_transport_weight', 'imputer_component_weight', 'rhbi_weight', 'rhbi_rho', 'rhbi_tau', 'rhbi_eps', 'rhbi_iters', 'rhbi_c_max', 'rhbi_beta_ell', 'rhbi_nn_quantile', 'rhbi_critic_weight', 'rhbi_self_coupling', 'rhbi_relativistic_weight', 'rhbi_lambda_floor', 'rhbi_gamma_gauge', 'rhbi_rapidity_screening', 'rhbi_scale_log_odds', 'rhbi_conditional_fibre', 'rhbi_akt_horizon', 'rhbi_akt_mode', 'rhbi_slack_gain', 'rhbi_chord_weight', 'rhbi_thermal_tau', 'rhbi_thermal_shock_mu', 'rhbi_settle_beta', 'rhbi_settle_mode', 'rhbi_fibre_rel_blend', 'rhbi_step_damp', 'rhbi_cond_ot_weight', 'rhbi_cond_ot_mode', 'rhbi_simplex_weight', 'rhbi_cvx_mu_gain', 'rhbi_simplex_chart_weight', 'rhbi_tail_spill_gain', 'rhbi_simplex_macro_settle')
    fields_dict = get_recipe_fields()
    field_names = tuple(fields_dict)
    assert len(field_names) == 118 + len(additions)
    assert tuple(name for name in field_names if name not in additions) == FROZEN_RECIPE_FIELDS
    assert tuple(field.name for field in dataclasses.fields(Recipe)) == field_names
    from particlegan.recipe_compat import SEARCH_RECIPE_DEFAULTS
    assert all(fields_dict[name] == SEARCH_RECIPE_DEFAULTS[name] for name in additions)


def test_recipe_no_toplevel_torch_import():
    """AC1 Gate 2: Ensure recipes.py can be imported and resolved without importing torch."""
    source = ROOT / "particlegan/recipes.py"
    script = """
import importlib.util, sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError(f'Configuration resolution imported Torch: {fullname}')
sys.meta_path.insert(0, BlockTorch())
spec = importlib.util.spec_from_file_location('standalone_recipes', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# Resolve fields across multiple presets without torch
fields_ka2 = module.get_recipe_fields('ka2')
assert len(fields_ka2) == 159
fields_bcap = module.get_recipe_fields('bcap')
assert len(fields_bcap) == 159
fields_bispf = module.get_recipe_fields('bispf_optimized')
assert len(fields_bispf) == 159

assert 'torch' not in sys.modules, 'torch leaked into sys.modules'
assert 'particlegan' not in sys.modules, 'particlegan leaked into sys.modules'
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, f"Subprocess failed with stderr:\n{result.stderr}"


@pytest.mark.parametrize("preset", ["ka2", "bcap", "bispf", "bispf_optimized", "e22"])
def test_recipe_serialization_roundtrip(preset):
    """Verify Recipe.to_dict() and Recipe.from_dict(recipe.to_dict()) roundtrips cleanly without data loss."""
    recipe = get_recipe(preset)
    as_dict = recipe.to_dict()
    reconstructed = Recipe.from_dict(as_dict)
    assert reconstructed == recipe
    assert get_recipe_fields(preset) == dataclasses.asdict(recipe)


def test_recipe_unknown_field_rejection():
    """Ensure get_recipe and get_recipe_fields reject unknown parameters with TypeError."""
    with pytest.raises(TypeError, match="Unknown recipe field"):
        get_recipe_fields(manipulators=[])
    with pytest.raises(TypeError, match="Unknown recipe field"):
        get_recipe(prior_manipulators=["simplex"])


# -----------------------------------------------------------------------------
# Group 2: Manipulator Protocol, Composition & Chaining
# -----------------------------------------------------------------------------

def test_prior_manipulator_protocol_defaults():
    """Verify PriorManipulator base protocol defaults to no-op behavior and row-local gradients."""
    manip = PriorManipulator()
    assert manip.row_local_gradients is True

    base = ParticlePrior(num_particles=16, z_dim=4)
    assert manip.sample_indices(base, 8) is None

    codes = torch.randn(8, 4)
    idx = torch.arange(8)
    assert torch.equal(manip.transform(codes, idx, base_prior=base), codes)

    gen = torch.randn(8, 3, 16, 16)
    assert torch.equal(manip.postprocess_generated(gen), gen)

    reg = manip.regularize(codes)
    assert reg == 0.0 or (torch.is_tensor(reg) and reg.item() == 0.0)


def test_manipulated_prior_initialization_and_delegations():
    """Verify ManipulatedPrior wraps base prior and cleanly delegates base attributes."""
    base_p = ParticlePrior(num_particles=64, z_dim=8, init_std=0.04)
    m = PriorManipulator()
    p = ManipulatedPrior(base_p, [m])

    assert p.base is base_p
    assert p.z is base_p.z
    assert p.num_particles == 64
    assert p.z_dim == 8
    assert p.init_std == 0.04
    assert len(p.manipulators) == 1
    assert p.manipulators[0] is m

    base_mog = MoGParticlePrior(num_particles=64, z_dim=8, sigma=0.025, standardize=True)
    p_mog = ManipulatedPrior(base_mog, [m])
    assert p_mog.sigma is base_mog.sigma
    assert p_mog.standardize is True
    assert p_mog.d0 is base_mog.d0
    assert torch.equal(p_mog.means(), base_mog.means())


def test_manipulate_fluent_chaining_and_recipe_integration():
    """Verify .manipulate(*manipulators) fluent API and Recipe.make_prior keyword integration."""
    base = ParticlePrior(num_particles=32, z_dim=4)
    m1 = StratifiedSamplingManipulator(32)
    m2 = SimplexRoutingManipulator(num_modes=4, temperature=0.5)

    # 1. Chaining on base prior
    mp1 = base.manipulate(m1)
    assert isinstance(mp1, ManipulatedPrior)
    assert len(mp1.manipulators) == 1

    # 2. Chaining on ManipulatedPrior flattens cleanly into single container
    mp2 = mp1.manipulate(m2)
    assert isinstance(mp2, ManipulatedPrior)
    assert len(mp2.manipulators) == 2
    assert mp2.manipulators[0] is m1
    assert mp2.manipulators[1] is m2

    # 3. Multi-argument manipulate
    mp3 = base.manipulate(m1, m2)
    assert len(mp3.manipulators) == 2

    # 4. Recipe.make_prior keyword
    recipe = get_recipe("bispf_optimized", num_particles=32, z_dim=4)
    p_rec = recipe.make_prior(manipulators=[m1, m2])
    assert isinstance(p_rec, ManipulatedPrior)
    assert len(p_rec.manipulators) == 2


def test_stratified_sampling_manipulator_permutation_and_cursor():
    """Verify StratifiedSamplingManipulator implements without-replacement sampling and cursor tracking."""
    N = 16
    B = 4
    base = ParticlePrior(num_particles=N, z_dim=4)
    m = StratifiedSamplingManipulator(num_particles=N)

    assert hasattr(m, "_perm") and hasattr(m, "_cursor")
    assert "_perm" in m.state_dict() and "_cursor" in m.state_dict()
    assert m._cursor.item() == 0
    assert len(m._perm) == N

    visited = []
    # 4 batches of size 4 cover all 16 particles
    for step in range(N // B):
        idx = m.sample_indices(base, batch_size=B)
        assert len(idx) == B
        assert m._cursor.item() == (step + 1) * B % N
        visited.extend(idx.tolist())

    assert len(visited) == N
    assert len(set(visited)) == N, "Stratified sampling produced duplicate indices in one epoch"
    assert set(visited) == set(range(N))


def test_simplex_routing_manipulator_invariants_and_gradients():
    """Verify SimplexRoutingManipulator mode weight normalization and gradient flow."""
    K = 5
    m = SimplexRoutingManipulator(num_modes=K, temperature=0.3, z_dim=4)
    base = ParticlePrior(num_particles=16, z_dim=4)

    codes = torch.randn(8, 4, requires_grad=True)
    idx = torch.arange(8)
    weights = m.transform(codes, idx, base_prior=base)

    assert weights.shape == (8, K)
    # Simplex Law 1: Non-negativity
    assert torch.all(weights >= 0.0)
    # Simplex Law 2: Partition of unity
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-6)

    # Autograd check: gradients must propagate to particle codes
    loss = (weights * torch.arange(K, dtype=torch.float32)).sum()
    loss.backward()
    assert codes.grad is not None
    assert torch.all(torch.isfinite(codes.grad))
    assert not torch.all(codes.grad == 0)


def test_capabilities_unwrapping_and_a2_truthfulness():
    """Verify prior_capabilities correctly unwraps ManipulatedPrior and honors manipulator flags."""
    base_row_local = ParticlePrior(32, 4)
    m_row_local = PriorManipulator()
    m_coupled = MockNonRowLocalManipulator()

    # 1. Base row-local + row-local manipulator -> a2_eligible = True
    p1 = base_row_local.manipulate(m_row_local)
    caps1 = prior_capabilities(p1)
    assert caps1["row_local_gradients"] is True
    assert caps1["a2_eligible"] is True

    # 2. Base row-local + coupled manipulator -> a2_eligible = False
    p2 = base_row_local.manipulate(m_coupled)
    caps2 = prior_capabilities(p2)
    assert caps2["row_local_gradients"] is False
    assert caps2["a2_eligible"] is False

    # 3. Base standardized MoG (already coupled) + row-local manipulator -> a2_eligible = False
    base_mog = MoGParticlePrior(32, 4, sigma=0.01, standardize=True)
    p3 = base_mog.manipulate(m_row_local)
    caps3 = prior_capabilities(p3)
    assert caps3["row_local_gradients"] is False
    assert caps3["a2_eligible"] is False

    # Verify prior_mechanisms diagnostic report
    mech = prior_mechanisms(p2, latent_damping_max_rate=0.1, prior_beta1=0.0)
    assert mech["a2"]["enabled"] is False


# -----------------------------------------------------------------------------
# Group 3: Optimizer Parameter Role Routing Across All Recipes
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["ka2", "bcap", "bispf", "bispf_optimized", "e22"])
def test_optimizer_parameter_roles_all_recipes(preset):
    """AC1 Gate 3: Verify prior.base.z is role='prior' and manipulator parameters are role='router'."""
    overrides = dict(num_particles=32, z_dim=4, batch_size=8)
    if preset != "e22":
        overrides["total_steps"] = 8
    recipe = get_recipe(preset, **overrides)
    gen = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    disc = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

    m_train = MockTrainableManipulator(in_features=4, out_features=4)
    prior = recipe.make_prior(manipulators=[StratifiedSamplingManipulator(32), m_train])

    ema_critic = deepcopy(disc)
    opt_g, opt_d = recipe.make_optimizers(gen, disc, prior, ema_critic=ema_critic)

    # Assert prior.base.z has its own parameter group
    prior_groups = [g for g in opt_g.param_groups if any(p is prior.base.z for p in g["params"])]
    assert len(prior_groups) == 1, "prior.base.z must belong to exactly one parameter group"
    prior_group = prior_groups[0]
    assert len(prior_group["params"]) == 1, "prior table group must contain only prior.base.z"
    assert prior_group["params"][0].ndim == 2, "prior table must be a 2D matrix"
    assert prior_group["lr"] == pytest.approx(recipe.lr * recipe.prior_lr_mult)

    # Assert manipulator trainable parameters are routed to role='router' with base lr
    router_groups = [g for g in opt_g.param_groups if any(p is m_train.matrix for p in g["params"])]
    assert len(router_groups) == 1, "Manipulator parameters must belong to exactly one parameter group"
    router_group = router_groups[0]
    assert router_group["lr"] == pytest.approx(recipe.lr)

    if recipe.optimizer_family not in ("adam", "formulation"):
        assert prior_group["role"] == "prior"
        assert router_group["role"] == "router"
        gen_groups = [g for g in opt_g.param_groups if any(p in set(gen.parameters()) for p in g["params"])]
        for g in gen_groups:
            assert g["role"] == "generator"

    # Verify UpdatePolicy invariant checking accepts the configuration cleanly
    policy = UpdatePolicy(recipe, gen, disc, prior=prior, generator_optimizer=opt_g, critic_optimizer=opt_d)
    assert "table" in policy.roles[0]
    if recipe.optimizer_family not in ("adam", "formulation"):
        assert "router" in policy.roles[0]
        assert "generator" in policy.roles[0]


def test_frozen_prior_freezes_manipulator_parameters():
    """prior_update='frozen' freezes the whole prior: table and manipulator (router) parameters."""
    recipe = get_recipe("bcap", prior_update="frozen", num_particles=32, z_dim=4, batch_size=8)
    gen = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    disc = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

    m_train = MockTrainableManipulator(in_features=4, out_features=4)
    prior = recipe.make_prior(manipulators=[m_train])

    assert prior.base.z.requires_grad is False
    assert m_train.matrix.requires_grad is True  # make_prior does not bind the update policy

    opt_g, opt_d = recipe.make_optimizers(gen, disc, prior)

    # Neither the table nor the router may train under a frozen prior policy.
    assert m_train.matrix.requires_grad is False
    trained = {id(p) for g in opt_g.param_groups for p in g["params"]}
    assert id(prior.base.z) not in trained
    assert id(m_train.matrix) not in trained
    assert not any(g.get("role") == "router" for g in opt_g.param_groups)


# -----------------------------------------------------------------------------
# Group 4: Checkpoint Continuation Parity
# -----------------------------------------------------------------------------

def test_checkpoint_continuation_parity_3step():
    """AC1 Gate 4: 3-step pause and continue checkpoint parity asserting atol=1e-6 on all states."""
    recipe = get_recipe("bispf_optimized", num_particles=32, z_dim=4, batch_size=8, total_steps=8)

    def build_trainer():
        torch.manual_seed(42)
        gen = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
        disc = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))

        m_sample = StratifiedSamplingManipulator(32)
        m_route = SimplexRoutingManipulator(num_modes=4, temperature=0.5)
        m_train = MockTrainableManipulator(in_features=4, out_features=4)
        m_post = MockPostprocessManipulator()

        prior = recipe.make_prior(manipulators=[m_sample, m_route, m_train, m_post])

        # Isolated deterministic streams
        latent_gen = torch.Generator().manual_seed(101)
        penalty_gen = torch.Generator().manual_seed(102)
        noise_gen = torch.Generator().manual_seed(103)

        return GANTrainer(
            recipe, gen, disc, prior=prior,
            latent_generator=latent_gen,
            penalty_generator=penalty_gen,
            noise_generator=noise_gen,
        )

    # 1. Uninterrupted run: step 6 times
    trainer_uninterrupted = build_trainer()
    batches = [
        torch.randn(8, 2, generator=torch.Generator().manual_seed(200 + i))
        for i in range(6)
    ]

    # Run first 3 steps
    for i in range(3):
        trainer_uninterrupted.step(batches[i])

    # Capture checkpoint after step 3
    checkpoint_at_3 = trainer_uninterrupted.state_dict()

    # Continue to step 6 uninterrupted
    for i in range(3, 6):
        trainer_uninterrupted.step(batches[i])
    final_uninterrupted = trainer_uninterrupted.state_dict()

    # 2. Restored run: build fresh, load checkpoint at 3, step to 6
    trainer_restored = build_trainer()
    trainer_restored.load_state_dict(checkpoint_at_3)

    for i in range(3, 6):
        trainer_restored.step(batches[i])
    final_restored = trainer_restored.state_dict()

    # 3. Assert exact continuation parity
    assert final_restored["completed_steps"] == 6 == final_uninterrupted["completed_steps"]

    # Assert generator parameters match with atol=1e-6
    for k in trainer_uninterrupted.G.state_dict():
        assert torch.allclose(
            trainer_uninterrupted.G.state_dict()[k],
            trainer_restored.G.state_dict()[k],
            atol=1e-6,
        ), f"Generator parameter mismatch at key {k}"

    # Assert discriminator parameters match with atol=1e-6
    for k in trainer_uninterrupted.D.state_dict():
        assert torch.allclose(
            trainer_uninterrupted.D.state_dict()[k],
            trainer_restored.D.state_dict()[k],
            atol=1e-6,
        ), f"Discriminator parameter mismatch at key {k}"

    # Assert prior base table matches with atol=1e-6
    assert torch.allclose(
        trainer_uninterrupted.prior.base.z,
        trainer_restored.prior.base.z,
        atol=1e-6,
    ), "prior.base.z table mismatch"

    # Assert manipulator buffers and parameters match
    # m_sample cursor and perm must match bit-for-bit
    assert torch.equal(
        trainer_uninterrupted.prior.manipulators[0]._cursor,
        trainer_restored.prior.manipulators[0]._cursor,
    ), "Manipulator _cursor mismatch"
    assert torch.equal(
        trainer_uninterrupted.prior.manipulators[0]._perm,
        trainer_restored.prior.manipulators[0]._perm,
    ), "Manipulator _perm mismatch"

    # m_train weights match with atol=1e-6
    assert torch.allclose(
        trainer_uninterrupted.prior.manipulators[2].matrix,
        trainer_restored.prior.manipulators[2].matrix,
        atol=1e-6,
    ), "Manipulator parameter matrix mismatch"

    # Full checkpoint recursive numerical parity check
    assert_nested_allclose(final_uninterrupted, final_restored, atol=1e-6)


def test_scale_learning_rates_router_schedule_agreement():
    """Verify scale_learning_rates scales role='router' groups with network_scale and matches UpdatePolicy.begin_step()."""
    # 1. Setup recipe where learning rate annealing produces net_scale != prior_scale
    recipe = get_recipe("ka2", num_particles=32, z_dim=8)
    step = 6000
    net_scale, prior_scale = learning_rate_scales(step, recipe)
    assert net_scale != prior_scale, (
        f"Precondition failed: net_scale ({net_scale}) == prior_scale ({prior_scale})"
    )

    # 2. Construct ManipulatedPrior with a trainable manipulator
    m_train = MockTrainableManipulator(in_features=8, out_features=8)
    prior = recipe.make_prior(manipulators=[m_train])
    assert isinstance(prior, ManipulatedPrior)
    assert prior.base.z.requires_grad is True
    assert m_train.matrix.requires_grad is True

    # 3. Models and Optimizers
    gen = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 2))
    disc = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    ema_critic = deepcopy(disc)
    opt_g, opt_d = recipe.make_optimizers(gen, disc, prior, ema_critic=ema_critic)

    # 4. UpdatePolicy and base rates
    policy = UpdatePolicy(recipe, gen, disc, generator_optimizer=opt_g, critic_optimizer=opt_d, prior=prior)
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in policy.optimizers]

    # Map parameter groups by parameter identity and cross-check with policy roles
    gen_group = next(g for g in opt_g.param_groups if any(p in set(gen.parameters()) for p in g["params"]))
    prior_group = next(g for g in opt_g.param_groups if any(p is prior.base.z for p in g["params"]))
    router_group = next(g for g in opt_g.param_groups if any(p is m_train.matrix for p in g["params"]))
    critic_group = opt_d.param_groups[0]

    assert opt_g.param_groups[policy.roles[0].index("generator")] is gen_group
    assert opt_g.param_groups[policy.roles[0].index("table")] is prior_group
    assert opt_g.param_groups[policy.roles[0].index("router")] is router_group
    assert opt_d.param_groups[policy.roles[1].index("critic")] is critic_group

    # Verify unscaled base learning rates
    assert gen_group["lr"] == pytest.approx(recipe.lr)
    assert prior_group["lr"] == pytest.approx(recipe.lr * recipe.prior_lr_mult)
    assert router_group["lr"] == pytest.approx(recipe.lr)
    assert critic_group["lr"] == pytest.approx(recipe.lr * recipe.d_lr_mult)

    # 5. Call scale_learning_rates
    ret_net, ret_prior = scale_learning_rates(step, recipe, policy.optimizers, base_rates, prior=prior)
    assert ret_net == pytest.approx(net_scale)
    assert ret_prior == pytest.approx(prior_scale)

    # 6. Assertions on scaled learning rates
    expected_gen_lr = recipe.lr * net_scale
    expected_prior_lr = recipe.lr * recipe.prior_lr_mult * prior_scale
    expected_router_lr = recipe.lr * net_scale
    expected_critic_lr = recipe.lr * recipe.d_lr_mult * net_scale

    # Router parameter group must match recipe.lr * net_scale (NOT prior_scale)
    assert router_group["lr"] == pytest.approx(expected_router_lr, rel=1e-5), (
        f"Router group LR was {router_group['lr']}, expected {expected_router_lr}"
    )
    assert abs(router_group["lr"] - (recipe.lr * prior_scale)) > 1e-4, (
        f"Router group LR incorrectly matches prior_scale ({recipe.lr * prior_scale})"
    )

    # Prior parameter group must match recipe.lr * recipe.prior_lr_mult * prior_scale
    assert prior_group["lr"] == pytest.approx(expected_prior_lr, rel=1e-5), (
        f"Prior group LR was {prior_group['lr']}, expected {expected_prior_lr}"
    )
    assert abs(prior_group["lr"] - (recipe.lr * recipe.prior_lr_mult * net_scale)) > 1e-4, (
        "Prior group LR was scaled by net_scale instead of prior_scale"
    )

    # Generator parameter group must match recipe.lr * net_scale
    assert gen_group["lr"] == pytest.approx(expected_gen_lr, rel=1e-5), (
        f"Generator group LR was {gen_group['lr']}, expected {expected_gen_lr}"
    )

    # Critic parameter group must match recipe.lr * recipe.d_lr_mult * net_scale
    assert critic_group["lr"] == pytest.approx(expected_critic_lr, rel=1e-5), (
        f"Critic group LR was {critic_group['lr']}, expected {expected_critic_lr}"
    )

    # 7. Record scale_learning_rates rates across all optimizers and groups
    scale_lrs = [[g["lr"] for g in opt.param_groups] for opt in policy.optimizers]

    # 8. Assert complete agreement with UpdatePolicy.begin_step()
    policy.completed_steps = step
    real = torch.randn(recipe.batch_size, 2)
    policy.begin_step(real)
    policy_lrs = [[g["lr"] for g in opt.param_groups] for opt in policy.optimizers]

    assert len(scale_lrs) == len(policy_lrs)
    for opt_idx, (s_opt, p_opt) in enumerate(zip(scale_lrs, policy_lrs)):
        assert len(s_opt) == len(p_opt)
        for grp_idx, (s_lr, p_lr) in enumerate(zip(s_opt, p_opt)):
            assert s_lr == pytest.approx(p_lr, rel=1e-6), (
                f"Schedule disagreement at optimizer {opt_idx}, group {grp_idx}: "
                f"scale_learning_rates gave {s_lr}, but UpdatePolicy.begin_step gave {p_lr}"
            )


def test_trainer_mog_stream_isolation_with_manipulated_prior():
    """Verify GANTrainer initializes prior_noise_generator and maintains stream isolation for ManipulatedPrior wrapping MoGParticlePrior."""
    recipe = get_recipe(num_particles=16, z_dim=4, prior_kind="mog", sigma_rel=0.05, standardize=False)
    m_train = MockTrainableManipulator(in_features=4, out_features=4)
    prior = recipe.make_prior(manipulators=[m_train])

    assert isinstance(prior, ManipulatedPrior)
    assert isinstance(prior.base, MoGParticlePrior)

    gen = nn.Linear(4, 8)
    disc = nn.Linear(8, 1)

    # 1. Automatic stream initialization
    trainer = GANTrainer(recipe, gen, disc, prior=prior)
    assert trainer.prior_noise_generator is not None
    assert "prior_noise_generator" in trainer._STREAMS

    # Check that prior_noise_generator is distinct from other streams
    assert trainer.prior_noise_generator is not trainer.latent_generator
    assert trainer.prior_noise_generator is not trainer.penalty_generator
    assert trainer.prior_noise_generator is not trainer.noise_generator
    assert trainer.prior_noise_generator is not trainer.eval_generator

    # 2. Checkpoint state dict includes prior_noise_generator
    sd = trainer.state_dict()
    assert "streams" in sd
    assert "prior_noise_generator" in sd["streams"]

    # 3. Explicit stream injection
    custom_stream = torch.Generator().manual_seed(9999)
    trainer_custom = GANTrainer(recipe, deepcopy(gen), deepcopy(disc), prior=prior, prior_noise_generator=custom_stream)
    assert trainer_custom.prior_noise_generator is custom_stream
    assert "prior_noise_generator" in trainer_custom._STREAMS

    # 4. Stream advance during training step
    initial_prior_state = trainer.prior_noise_generator.get_state().clone()
    initial_latent_state = trainer.latent_generator.get_state().clone()
    real = torch.randn(recipe.batch_size, 8)
    trainer.step(real)

    # Both latent generator and prior noise generator advance
    assert not torch.equal(trainer.prior_noise_generator.get_state(), initial_prior_state)
    assert not torch.equal(trainer.latent_generator.get_state(), initial_latent_state)


def test_make_prior_regularizer_initial_mmd_unwraps_base_prior():
    """Verify make_prior_regularizer with initial_mmd registers reference buffers on base prior."""
    recipe = get_recipe(num_particles=16, z_dim=4, prior_regularizer="initial_mmd")
    m_train = MockTrainableManipulator(in_features=4, out_features=4)
    prior = recipe.make_prior(manipulators=[m_train])

    recipe.make_prior_regularizer(prior=prior)
    assert hasattr(prior.base, "_initial_mmd_reference")
    assert hasattr(prior.base, "_initial_mmd_scale")
    assert hasattr(prior.base, "_initial_mmd_reference_kernel")

    # Verify fluent chaining preserves access to registered buffers on base
    chained = prior.manipulate(StratifiedSamplingManipulator(num_particles=16))
    reg2 = recipe.make_prior_regularizer(prior=chained)
    assert reg2 is not None

