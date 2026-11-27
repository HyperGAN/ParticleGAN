"""Law/gate discrimination and actual public-API caller software controls."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.toy_audit import api_vectors as api
from particlegan import GANTrainer, Recipe


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def definition(identifier):
    return next(case for case in api.list_cases() if case["id"] == identifier)


def oracle(case, *, n=None, step=None, seed=713):
    return api._target(case, n or case["eval_samples"], torch.Generator().manual_seed(seed),
                       case["default_steps"] if step is None else step)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        # Optimizer records use uninitialized NaN sentinels. Compare their
        # exact representation too; NaN != NaN is not trajectory divergence.
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_deterministic_prior_and_identical_batches_across_formulations():
    case = definition("api-vector-two-broad")
    fixtures = []
    for family in ("k3p", "bispf"):
        fixtures.append(api.VectorFixture(deepcopy(case), recipe_name=family, max_steps=3))
    left, right = fixtures
    expected = api.init.deterministic_orthogonal_(left.recipe.make_prior(init_std=.5))
    assert torch.equal(left.trainer.prior.z, expected.z)
    for role in ("G", "D"):
        assert_nested_equal(getattr(left.trainer, role).state_dict(), getattr(right.trainer, role).state_dict())
    # Recipe-owned regularizers may register different reference buffers, but
    # the task's initial particle locations and data streams remain shared.
    assert torch.equal(left.trainer.prior.z, right.trainer.prior.z)
    for _ in range(3):
        left.step()
        right.step()
        assert left.batch_sequence_sha256 == right.batch_sequence_sha256
        assert torch.equal(left.data_rng.get_state(), right.data_rng.get_state())
        assert torch.equal(left.generator_data_rng.get_state(), right.generator_data_rng.get_state())


def test_old_data_stream_checkpoint_is_rejected_before_mutation():
    fixture = api.build_case("api-vector-two-broad", max_steps=2)
    before = fixture.state_dict()
    legacy = deepcopy(before)
    legacy["version"] = "toy-public-api-vectors-v1"
    with pytest.raises(ValueError, match="identity/recipe"):
        fixture.load_state_dict(legacy)
    assert_nested_equal(before, fixture.state_dict())


@pytest.mark.parametrize("changed_condition", ["seed", "target"])
def test_checkpoint_cannot_transfer_to_a_different_comparison_condition(changed_condition):
    original = api.build_case("api-vector-two-broad", max_steps=2)
    case = definition("api-vector-two-broad")
    if changed_condition == "target":
        case["spec"]["means"][0][0] += 1.
    recipient = api.VectorFixture(case, seed=1 if changed_condition == "seed" else 0, max_steps=2)
    before = recipient.state_dict()
    with pytest.raises(ValueError, match="identity/recipe"):
        recipient.load_state_dict(original.state_dict())
    assert_nested_equal(before, recipient.state_dict())


def test_every_owned_historical_question_and_all_proposal_arms_are_mapped():
    cases = api.list_cases()
    # 54 plus api-zero-tanh-smoke, api-zero-tanh-rail-escape, api-grid100-mlp16,
    # api-grid100-res16-tanh, api-grid100-res16-scaled-tanh, and
    # api-grid100-res16-scaled-tanh-v2.
    assert len(cases) == len({case["id"] for case in cases}) == 60
    catalog = json.loads((Path(__file__).parents[1] / "reports/toy_audit/catalog.json").read_text())["cases"]
    expected = {case["id"] for case in catalog if case["id"].startswith("atlas-")
                or case.get("spec", {}).get("kind") in {"gaussian_mixture", "spiral", "annulus"}}
    expected |= {"develop-two_pole", "develop-mode_hold", "source-family-10", "source-family-16",
                 "zero-tanh-smoke", "zero-tanh-rail-escape", "deep100g-mlp16", "deep100g-res16-tanh",
                 "deep100g-res16-scaled-tanh", "deep100g-res16-scaled-tanh-v2"}
    assert {old for case in cases for old in case["legacy_ids"]} == expected
    for case in cases:
        assert case["goal"] and case["thresholds"] and case["sampling"] and case["scope"]
        assert case["default_steps"] > 0 and case["eval_samples"] > 0
    for pr in api.PROPOSAL_DEFINITIONS:
        assert {case["id"] for case in cases if case.get("comparison_group") == f"api-pr{pr}"} == {f"api-pr{pr}-published", f"api-pr{pr}-control"}
    # Metadata copies cannot silently change another observation's contract.
    cases[0]["legacy_ids"].clear()
    assert api.list_cases()[0]["legacy_ids"]


@pytest.mark.parametrize("case", api.list_cases(), ids=lambda case: case["id"])
def test_fixed_gate_accepts_independent_full_target_law(case):
    result = api.score_case(case, oracle(case), case["default_steps"])
    assert result["passed"], result["failed_bounds"]
    assert result["failed_bounds"] == []


def test_missing_rare_mass_is_rejected_even_below_resolved_covariance_floor():
    case = definition("api-vector-unequal-mass")
    changed = deepcopy(case)
    changed["spec"]["masses"] = [.55 / .98, .30 / .98, .13 / .98, 0.]
    result = api.score_case(case, oracle(changed), case["default_steps"])
    assert not result["passed"]
    assert "min_mass_ratio >= 0.25" in result["failed_bounds"]
    assert case["spec"]["masses"][3] * case["particles"] < vector_floor()


def vector_floor():
    return api.vector_tasks.PARTICLE_FLOOR


@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "api-vector-narrow", "api-stress-fast-critic"])
def test_correct_centers_and_masses_cannot_substitute_for_width(identifier):
    case = definition(identifier)
    centers = torch.tensor(case["spec"]["means"])
    fake = centers[torch.arange(case["eval_samples"]) % len(centers)]
    result = api.score_case(case, fake, case["default_steps"])
    assert not result["passed"]
    assert any("eigen" in failure for failure in result["failed_bounds"])
    assert result["metrics"]["projection_ks"] > .06


def test_overlapping_ring_centers_false_positive_fails_analytic_cdf():
    case = definition("api-stress-overlapping-data")
    centers = torch.tensor(case["spec"]["means"])
    fake = centers[torch.arange(4096) % 8]
    old = api.vector_tasks.score_samples(fake, case["spec"], case["default_steps"])
    assert api.vector_tasks.passes(old, case["spec"]["thresholds"])
    new = api.score_case(case, fake, case["default_steps"])
    assert not new["passed"]
    assert new["failed_bounds"] == ["projection_ks <= 0.06"]
    assert not any("component" in bound[0] for bound in case["thresholds"])


def test_anisotropic_trace_matched_wrong_axes_are_rejected():
    case = definition("api-vector-anisotropic")
    changed = deepcopy(case)
    covariance = np.asarray(changed["spec"]["covariances"])
    changed["spec"]["covariances"] = [(np.eye(2) * np.trace(c) / 2).tolist() for c in covariance]
    result = api.score_case(case, oracle(changed), case["default_steps"])
    assert not result["passed"]
    assert result["metrics"]["projection_ks"] > .06


def test_continuous_annulus_wrong_radial_mass_is_rejected():
    case = definition("api-reserved-annulus")
    angle = torch.arange(4096) * (2 * torch.pi / 4096)
    fake = case["spec"]["radius_max"] * torch.stack([angle.cos(), angle.sin()], 1)
    result = api.score_case(case, fake, case["default_steps"])
    assert not result["passed"]
    assert result["metrics"]["projection_ks"] > .06


def test_gaussian_gate_rejects_same_mean_covariance_non_gaussian_circle():
    case = definition("api-gaussian2d")
    angle = torch.arange(4096) * (2 * torch.pi / 4096)
    fake = 1 + .2 * np.sqrt(2) * torch.stack([angle.cos(), angle.sin()], 1)
    result = api.score_case(case, fake)
    assert not result["passed"]
    assert result["metrics"]["radial_ks"] > .075


def test_separate_unperturbed_twelve_atom_ring_witness_has_mass_and_width_obstruction():
    # This supplied finite population is not GANTrainer.sample's DV12 law.
    case = definition("api-ring8-resolution12")
    angle = torch.arange(8) * torch.pi / 4
    centers = 3 * torch.stack([angle.cos(), angle.sin()], 1)
    twelve = centers[torch.arange(12) % 8]
    fake = twelve[torch.arange(12288) % 12]
    result = api.score_case(case, fake)
    assert not result["passed"]
    assert result["metrics"]["mass_tv"] == pytest.approx(1 / 6)
    assert result["metrics"]["min_cov_eigen"] == 0
    assert case["resource_change"]["current_rows"] == 12
    assert definition("api-ring8-acquire")["resource_change"]["current_rows"] == 256


def test_256_atom_ring_has_a_constructive_gate_witness_not_a_trained_pass():
    case = definition("api-ring8-acquire")
    modes = torch.arange(8) * torch.pi / 4
    centers = 3 * torch.stack([modes.cos(), modes.sin()], 1)
    u = (torch.arange(16) + .5) / 16
    radii = torch.sqrt(-2 * torch.log(1 - u))
    angle = torch.arange(16) * torch.pi * (3 - np.sqrt(5))
    half = .07 * radii[:, None] * torch.stack([angle.cos(), angle.sin()], 1)
    offsets = torch.cat([half, -half])
    population = (centers[:, None] + offsets[None]).reshape(256, 2)
    # One predeclared iid draw from a fixed constructive population; never
    # used to initialize a GAN, tune a gate, select a seed or claim learning.
    samples = population[torch.randint(256, (4096,), generator=torch.Generator().manual_seed(1931))]
    result = api.score_case(case, samples)
    assert result["passed"], result["failed_bounds"]


def test_two_pole_exact_batch_law_rejects_centers_and_one_pole_travel():
    case = definition("api-two-pole-grid12")
    exact = oracle(case)
    assert api.score_case(case, exact)["passed"]
    for fake in (torch.tensor([-1.] * 6 + [1.] * 6)[:, None], torch.full((12, 1), .6)):
        assert not api.score_case(case, fake)["passed"]
    assert not api.score_case(case, exact.repeat(2, 1))["passed"]


def test_native_all_modes_with_zero_width_and_missing_one_mode_fail():
    case = definition("api-grid100")
    centers = api.problems.evaluation_geometry("grid100")[0]
    collapsed = centers[torch.arange(20000) % 100]
    assert not api.score_case(case, collapsed)["passed"]
    fake = oracle(case)
    ids = torch.cdist(fake, centers).argmin(1)
    fake[ids == 0] += centers[1] - centers[0]
    result = api.score_case(case, fake)
    assert not result["passed"]
    assert "modes >= 100" in result["failed_bounds"]


def test_moving_targets_use_exact_original_update_boundaries():
    case = definition("api-rotated100-moving")
    assert api._native_angle(case, 500) == 0
    assert api._native_angle(case, 501) == pytest.approx(np.pi / 6)
    assert api._native_angle(case, 1000) == pytest.approx(np.pi / 6)
    assert api._native_angle(case, 1001) == pytest.approx(np.pi / 3)
    assert not api.score_case(case, oracle(case, step=500), 1500)["passed"]
    drift = definition("api-vector-scale-drift")
    assert api.vector_tasks.target_scale(drift["spec"], 32) != api.vector_tasks.target_scale(drift["spec"], 1600)
    assert drift["law"]["schedule_budget"] == 1600


@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "api-pr49-published", "api-pr49-control", "api-ring8-acquire", "api-two-pole-grid12", "api-gaussian2d", "api-grid100"])
def test_real_public_trainer_update_short_schedule_and_pure_observation(identifier):
    fixture = api.build_case(identifier, recipe_name="auto", max_steps=2)
    assert isinstance(fixture.recipe, Recipe) and isinstance(fixture.trainer, GANTrainer)
    full = fixture.metadata["default_steps"]
    assert fixture.recipe.total_steps == (full if fixture.recipe.continuous_policy is None else None)
    assert fixture.step()["step"] == 1
    assert fixture.trainer.completed_steps == fixture.completed_steps == 1
    before = fixture.state_dict()
    global_rng = torch.get_rng_state().clone()
    observation = fixture.observe(n=64)
    assert isinstance(observation["passed"], bool)
    assert observation["failed_bounds"] and not observation["passed"]
    assert all(view["target"].numel() and view["samples"].numel() for view in observation["views"])
    assert torch.equal(global_rng, torch.get_rng_state())
    assert_nested_equal(before, fixture.state_dict())


@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "api-ring8-acquire", "api-gaussian2d"])
def test_checkpoint_resume_retains_data_rng_optimizers_and_next_update(identifier):
    straight = api.build_case(identifier, recipe_name="auto", max_steps=3)
    straight.step()
    checkpoint = straight.state_dict()
    resumed = api.build_case(identifier, recipe_name="auto", max_steps=3)
    resumed.load_state_dict(checkpoint)
    assert_nested_equal(straight.state_dict(), resumed.state_dict())
    straight.observe(n=64)  # independent read in only one continuation
    straight.step(); resumed.step()
    assert_nested_equal(straight.state_dict(), resumed.state_dict())


@pytest.mark.parametrize("phase,completed", [("hold", 1200), ("shift", 2400), ("shift", 2800)])
def test_failed_scientific_phase_prerequisite_stops_before_training(phase, completed):
    # A synthetic phase clock exercises only the protocol guard, with no
    # trained model or simulated convergence. Any attempt to access a trainer
    # after the failed held-out gate would raise independently in this fixture.
    fixture = object.__new__(api.VectorFixture)
    fixture.metadata = definition("api-ring8-" + phase)
    fixture.execution_steps = fixture.metadata["default_steps"]
    fixture.completed_steps = completed
    fixture.phase_receipts = {}
    fixture.observe = lambda **kwargs: {"metrics": {"mass_tv": .20}, "passed": False,
                                       "failed_bounds": ["mass_tv <= 0.075"]}
    with pytest.raises(RuntimeError, match="scientific prerequisite failed"):
        fixture.step()
    assert fixture.completed_steps == completed
    assert not fixture.phase_receipts[str(completed)]["passed"]


def test_half_frequency_d_loop_preserves_phase_and_uses_public_factories():
    fixture = api.build_case("api-reserved-alternating-critic-updates", recipe_name="bispf", max_steps=4)
    assert fixture.recipe.name == "bispf" and fixture.recipe.continuous_policy is None
    assert fixture.recipe.total_steps == 2400
    assert fixture.recipe.input_noise_std == fixture.recipe.output_noise_std == 0
    for step in range(1, 5):
        counts = fixture.step()
        assert counts["g"] == step and counts["d"] == (step + 1) // 2
    state = fixture.state_dict()
    fixture.observe(n=64)
    assert_nested_equal(state, fixture.state_dict())
    with pytest.raises(ValueError, match="cannot execute the intact recipe"):
        api.build_case("api-reserved-alternating-critic-updates", recipe_name="atlas", max_steps=1)
    with pytest.raises(ValueError, match="explicitly declared matching recipe"):
        api.build_case("api-stress-r1-r2", recipe_name="atlas", max_steps=1)


@pytest.mark.parametrize("identifier", ["api-vector-two-broad"])
def test_corrupted_phase_clock_is_rejected_before_resume_mutates_any_owner(identifier):
    fixture = api.build_case(identifier, recipe_name="auto", max_steps=3)
    fixture.step()
    before = fixture.state_dict()
    corrupt = deepcopy(before)
    corrupt["completed_steps"] += 1
    with pytest.raises(ValueError, match="phase clock"):
        fixture.load_state_dict(corrupt)
    assert_nested_equal(before, fixture.state_dict())


def test_short_draws_and_nonfinite_arrays_cannot_pass():
    case = definition("api-vector-two-broad")
    assert not api.score_case(case, oracle(case, n=64))["passed"]
    with pytest.raises(ValueError, match="finite"):
        api.score_case(case, torch.full((4096, 2), float("nan")))
    with pytest.raises(ValueError, match="prefix"):
        api.build_case(case["id"], max_steps=case["default_steps"] + 1)


def test_grid100_res16_scaled_tanh_v2_unneutered_generator_and_non_identity_control():
    """Verify v2 builds preserve_latent_norm=False and is genuinely sensitive to weight perturbations."""
    from experiments.forge import adapters, nativeprofiles

    root = Path(__file__).resolve().parents[1]
    v2_case = deepcopy(definition("api-grid100-res16-scaled-tanh-v2"))
    v2_case["particles"] = 64
    v2_case["batch_size"] = 16
    v2_fixture = api.VectorFixture(v2_case, recipe_name="k3p", max_steps=1)
    assert isinstance(v2_fixture.trainer.G, api.ScaledResidualMLPGenerator)
    assert v2_fixture.trainer.G.preserve_latent_norm is False
    assert isinstance(v2_fixture.trainer.G.norm, torch.nn.LayerNorm)

    legacy_case = deepcopy(definition("api-grid100-res16-scaled-tanh"))
    legacy_case["particles"] = 64
    legacy_case["batch_size"] = 16
    legacy_fixture = api.VectorFixture(legacy_case, recipe_name="k3p", max_steps=1)
    assert isinstance(legacy_fixture.trainer.G, api.ScaledResidualMLPGenerator)
    assert legacy_fixture.trainer.G.preserve_latent_norm is True

    # Forge nativeprofiles routing honors explicit preserve_latent_norm=False on v2
    # while preserving legacy preserve_latent_norm=True when omitted on v1.
    cand = {"recipe_preset": "k3p", "recipe_overrides": {}}
    v2_card = json.loads((root / "configs/forge/tasks/grid100_res16_scaled_tanh_v2.json").read_text())
    v1_card = json.loads((root / "configs/forge/tasks/grid100_res16_scaled_tanh.json").read_text())
    assert v2_card["execution"]["model"]["preserve_latent_norm"] is False
    assert adapters.adapter_preflight(v2_card, cand) == []
    ctx_v2 = adapters._context({"candidate": cand, "protocol": {"seed": 0}}, v2_card, "cpu", v2_card["execution"]["resources"])
    g_v2, d_v2 = nativeprofiles.build_declared_native_models(ctx_v2, v2_card["execution"]["model"])
    assert isinstance(g_v2, api.ScaledResidualMLPGenerator)
    assert g_v2.preserve_latent_norm is False
    assert isinstance(d_v2, api.SimpleMLPDiscriminator)

    ctx_v1 = adapters._context({"candidate": cand, "protocol": {"seed": 0}}, v1_card, "cpu", v1_card["execution"]["resources"])
    g_v1, _ = nativeprofiles.build_declared_native_models(ctx_v1, v1_card["execution"]["model"])
    assert isinstance(g_v1, api.ScaledResidualMLPGenerator)
    assert g_v1.preserve_latent_norm is True

    # Semantic non-identity control: +0.5 Gaussian weight perturbation must
    # strongly move G(z) away from z (max |G(z) - z| > 0.1).
    rng = torch.Generator().manual_seed(0)
    z = torch.randn((128, 2), generator=rng)
    with torch.no_grad():
        for param in g_v2.parameters():
            param.add_(0.5 * torch.randn(param.shape, generator=rng, dtype=param.dtype))
        out_v2 = g_v2(z)
    max_diff = float((out_v2 - z).abs().max())
    assert max_diff > 0.1, f"expected non-identity output after +0.5 weight perturbation, got max |G(z)-z|={max_diff}"


def test_grid100_res16_scaled_tanh_v2_passes_forge_validate():
    """Verify grid100_res16_scaled_tanh_v2 loads and passes full experiments.forge validate."""
    import subprocess
    import sys
    from experiments.forge import nativeprofiles, views

    root = Path(__file__).resolve().parents[1]
    tasks = views.load_tasks(root)
    assert "grid100_res16_scaled_tanh_v2" in tasks
    v2_task = tasks["grid100_res16_scaled_tanh_v2"]
    assert v2_task["execution"]["model"]["preserve_latent_norm"] is False
    assert nativeprofiles.resolve_native_spec(v2_task, root=root) is None

    proc = subprocess.run(
        [sys.executable, "-m", "experiments.forge", "validate"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"experiments.forge validate failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
