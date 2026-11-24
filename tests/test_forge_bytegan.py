"""Semantic controls, public-API continuation, and fail-closed grading for ByteGAN stretch tasks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.toy_audit.api_bytegan import (
    BYTE_CORNERS,
    ByteGANBiGANFixture,
    ByteGANToyFixture,
    HOST_DEFINITIONS,
    PRIOR_EXCEPTION_REASONS,
    THRESHOLDS,
    build_case,
    build_models_for_spec,
    bytegan_oracle_controls,
    sample_byte_target,
    score_bytegan_bigan_dictionary,
    score_bytegan_hypercube8,
)
import experiments.forge.api as forge_api
from experiments.forge.bytegan import (
    ADAPTER_TO_KIND,
    BYTEGAN_KINDS,
    bytegan_context,
    grade as bytegan_grade,
    preflight as bytegan_preflight,
    run as bytegan_run,
)
from experiments.forge.planning import load_idea
from experiments.forge.priors import prior_code_path, prior_noise_sigma
from experiments.forge.state import state_digest
from experiments.forge.trainer_families import current_family_candidates
from particlegan import ParticlePrior, unwrap_prior

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {"recipe_preset": "unified_v12", "claim_contract": {"sampling_law": "task_declared"}}
SELECTED_FAMILIES = (
    "unified",
    "rhbi",
    "empf",
    "bispf",
    "bcap-pure",
    "k3p",
    "ka2",
    "r1r2",
    "release07-gan-v3",
    "e22",
)


def _make_task(kind: str) -> dict:
    spec = deepcopy(HOST_DEFINITIONS[kind])
    is_bigan = kind == "bytegan_bigan_dictionary"
    adapter = "bytegan_bigan" if is_bigan else "bytegan"
    execution = {
        "initializer": "deterministic_orthogonal",
        "host": kind,
        "steps": spec["steps"],
        "prior": {
            "kind": "particle_cloud",
            "sigma": 0.0,
            "standardize": False,
            "exception_reason": PRIOR_EXCEPTION_REASONS[kind],
        },
        "protocol": "screening",
        "produces_state": True,
        "prior_contract": "recipe_owned_v1",
        "host_definition": spec,
    }
    if is_bigan:
        execution["execution_path"] = "public_components"
        execution["transport_consumer"] = "output_marginal_v1"
    task = {
        "schema_version": 1,
        "id": kind,
        "description": spec["target"],
        "adapter": adapter,
        "retained_question_ids": [f"stretch-{kind.replace('_', '-')}"],
        "execution": execution,
        "evaluation": {
            "sampling_contract_version": 1,
            "sampling_law": (
                "generated_and_paired_reconstructed_prior_without_output_noise"
                if is_bigan
                else "enumerated_prior_without_output_noise"
            ),
            "eval_output_noise": "clean",
            "scoring_weights": "live",
            "kind": kind,
            "eval_samples": 256,
            "observations": 24,
            "minimum_stable_checks": 5,
            "guards": {
                "finite_state": True,
                "optimizer_roles": (
                    ["generator", "encoder", "prior", "discriminator"]
                    if is_bigan
                    else ["generator", "discriminator", "prior"]
                ),
                "exact_optimizer_updates": True,
                "mechanism_exercised": True,
                "rng_isolation": True,
            },
            "thresholds": deepcopy(THRESHOLDS[kind]),
            "sources": {},
        },
        "resources": {
            "gpus": 1,
            "gpu_memory_mb": 2048,
            "cpu_threads": 1,
            "timeout_seconds": 900 if is_bigan else 600,
        },
        "requires_capabilities": (
            ["public_components", "checkpoint", "named_rng", "live_sampling", "particle_cloud"]
            if is_bigan
            else ["checkpoint", "named_rng", "live_sampling", "particle_cloud"]
        ),
        "dependencies": [],
    }
    if is_bigan:
        task["task_cohort"] = "component_output_transport_v1"
    return task


@pytest.fixture(autouse=True)
def _pre_merge_api_wiring(monkeypatch):
    """Bridge Forge API wiring when running in an isolated pre-merge worktree.

    Once ``forge-wiring`` lands ``bytegan``/``bytegan_bigan`` directly in
    ``experiments/forge/api.py``, this fixture becomes a complete no-op so all
    tests exercise the real shared registry without patching.
    """
    if "bytegan_bigan_dictionary" in forge_api.COMPONENT_OBJECTIVE_HOSTS:
        return

    monkeypatch.setattr(
        forge_api,
        "COMPONENT_OBJECTIVE_HOSTS",
        forge_api.COMPONENT_OBJECTIVE_HOSTS | {"bytegan_bigan_dictionary"},
    )
    monkeypatch.setattr(
        forge_api,
        "PUBLIC_TRAINER_POLICY_ADAPTERS",
        forge_api.PUBLIC_TRAINER_POLICY_ADAPTERS | {"bytegan", "bytegan_bigan"},
    )

    orig_resources = forge_api.task_recipe_resources

    def _patched_resources(task):
        if task.get("adapter") in {"bytegan", "bytegan_bigan"}:
            return deepcopy(task["execution"]["host_definition"]["resources"])
        return orig_resources(task)

    monkeypatch.setattr(forge_api, "task_recipe_resources", _patched_resources)

    orig_policy_blockers = forge_api.task_policy_blockers

    def _patched_policy_blockers(task, candidate):
        if task.get("adapter") == "bytegan_bigan" and task.get("execution", {}).get("transport_consumer") == "output_marginal_v1":
            proxy = deepcopy(task)
            proxy["adapter"] = "word_joint"
            return orig_policy_blockers(proxy, candidate)
        return orig_policy_blockers(task, candidate)

    monkeypatch.setattr(forge_api, "task_policy_blockers", _patched_policy_blockers)


def test_oracle_positive_and_destructive_negative_controls():
    assert BYTEGAN_KINDS == frozenset({"bytegan_hypercube8", "bytegan_bigan_dictionary"})
    assert ADAPTER_TO_KIND == {"bytegan": "bytegan_hypercube8", "bytegan_bigan": "bytegan_bigan_dictionary"}

    controls = bytegan_oracle_controls()
    assert set(controls) == {
        "hypercube8_exact_bijection_oracle",
        "hypercube8_single_collision_negative",
        "hypercube8_faint_barycentric_ghosts_negative",
        "hypercube8_exploded_amplitude_negative",
        "hypercube8_half_cube_collapse_negative",
        "hypercube8_nonfinite_negative",
        "bigan_exact_dictionary_oracle",
        "bigan_swapped_reconstruction_negative",
        "bigan_encoder_bypass_cloud_negative",
        "bigan_forward_collision_negative",
        "bigan_blurry_reconstruction_negative",
        "bigan_nonfinite_negative",
    }
    for name, item in controls.items():
        assert item["passed"] is item["expected_pass"], f"{name}: {item}"

    # Verify selectivity of every destructive failure mode.
    assert "valid_unique_bytes == 256" in controls["hypercube8_single_collision_negative"]["failed_bounds"]
    assert "valid_particle_fraction == 1.0" in controls["hypercube8_faint_barycentric_ghosts_negative"]["failed_bounds"]
    assert "rms_bit_error <= 0.12" in controls["hypercube8_faint_barycentric_ghosts_negative"]["failed_bounds"]
    assert "valid_particle_fraction == 1.0" in controls["hypercube8_exploded_amplitude_negative"]["failed_bounds"]
    assert "valid_unique_bytes == 256" in controls["hypercube8_half_cube_collapse_negative"]["failed_bounds"]
    assert "finite_fraction == 1.0" in controls["hypercube8_nonfinite_negative"]["failed_bounds"]

    assert "recon_valid_exact_bytes == 256" in controls["bigan_swapped_reconstruction_negative"]["failed_bounds"]
    assert "encoder_particle_bijection_bytes == 256" in controls["bigan_encoder_bypass_cloud_negative"]["failed_bounds"]
    assert "valid_unique_bytes == 256" in controls["bigan_forward_collision_negative"]["failed_bounds"]
    assert "recon_valid_exact_bytes == 256" in controls["bigan_blurry_reconstruction_negative"]["failed_bounds"]
    assert "recon_rms_bit_error <= 0.12" in controls["bigan_blurry_reconstruction_negative"]["failed_bounds"]
    assert "finite_fraction == 1.0" in controls["bigan_nonfinite_negative"]["failed_bounds"]

    # Confirm faint barycentric ghosts still have 256 raw unique signs yet 0 valid bytes.
    ghost_score = score_bytegan_hypercube8(BYTE_CORNERS, 0.05 * BYTE_CORNERS)
    assert ghost_score["metrics"]["unique_bytes"] == 256
    assert ghost_score["metrics"]["valid_unique_bytes"] == 0
    assert ghost_score["metrics"]["valid_particle_fraction"] == 0.0
    assert ghost_score["passed"] is False

    # Confirm encoder bypass achieves 256/256 forward + 256/256 recon yet fails particle bijection.
    enc_bypass = BYTE_CORNERS[0:1].expand(256, 8) + 1e-4 * BYTE_CORNERS
    bypass_score = score_bytegan_bigan_dictionary(BYTE_CORNERS, BYTE_CORNERS, enc_bypass, BYTE_CORNERS)
    assert bypass_score["metrics"]["valid_unique_bytes"] == 256
    assert bypass_score["metrics"]["recon_valid_exact_bytes"] == 256
    assert bypass_score["metrics"]["encoder_particle_bijection_bytes"] == 1
    assert bypass_score["passed"] is False


@pytest.mark.parametrize("kind", ["bytegan_hypercube8", "bytegan_bigan_dictionary"])
@pytest.mark.parametrize(
    "bad_prior",
    [
        {"kind": "mog", "sigma": 0.025, "standardize": False},
        {"kind": "particle_cloud", "sigma": 0.01, "standardize": False, "exception_reason": "nonzero"},
        {"kind": "particle_cloud", "sigma": 0.0, "standardize": True, "exception_reason": "standardized"},
        {"kind": "particle_cloud", "sigma": 0.0, "standardize": False, "exception_reason": ""},
    ],
)
def test_forbidden_priors_refused_by_preflight_and_graded_invalid(kind, bad_prior):
    task = _make_task(kind)
    task["execution"]["prior"] = bad_prior
    assert bytegan_preflight(task, CANDIDATE, root=ROOT)
    assert bytegan_grade(task, {})[0] == "INVALID"


@pytest.mark.parametrize("kind", ["bytegan_hypercube8", "bytegan_bigan_dictionary"])
def test_fresh_gaussian_code_law_is_refused(kind):
    task = _make_task(kind)
    fresh_candidate = {
        "recipe_preset": "empf",
        "recipe_overrides": {"prior_update": "frozen", "prior_code_law": "fresh_gaussian"},
        "claim_contract": {"sampling_law": "task_declared"},
    }
    blockers = bytegan_preflight(task, fresh_candidate, root=ROOT)
    assert blockers and any("fresh_gaussian" in b or "table" in b for b in blockers)


@pytest.mark.parametrize("kind", ["bytegan_hypercube8", "bytegan_bigan_dictionary"])
@pytest.mark.parametrize("family", [*SELECTED_FAMILIES, "unified_v12_preset"])
def test_every_selected_family_binds_zero_sigma_particle_prior_and_steps(kind, family):
    task = _make_task(kind)
    if family == "unified_v12_preset":
        candidate = deepcopy(CANDIDATE)
    else:
        choice = next(row for row in current_family_candidates(ROOT) if row["family"] == family)
        candidate = load_idea(ROOT, choice["candidate_id"])

    blockers = bytegan_preflight(task, candidate, root=ROOT)
    if blockers:
        assert family == "e22" and any(
            "policy" in b or "live_sampling" in b or "served-sampling" in b for b in blockers
        ), f"unexpected blocker for {family} on {kind}: {blockers}"
        return

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        context = bytegan_context({"candidate": candidate, "protocol": {"seed": 0}}, task, "cpu", root=ROOT)
        assert prior_noise_sigma(context.prior_config) == 0.0
        assert prior_code_path(context.prior_config) == "ParticlePrior"
        spec = task["execution"]["host_definition"]
        if kind == "bytegan_hypercube8":
            g = context.construct(lambda: build_models_for_spec(spec, z_dim=context.recipe.z_dim)[0], component="generator")
            d = context.construct(lambda: build_models_for_spec(spec, z_dim=context.recipe.z_dim)[1], component="discriminator")
            trainer = context.build_trainer(g, d, max_steps=10)
            assert isinstance(unwrap_prior(trainer.prior), ParticlePrior)
            rng = context.streams.generator("data", component="target", purpose="training", device="cpu")
            for _ in range(2):
                trainer.step(sample_byte_target(context.recipe.batch_size, rng, device="cpu"))
            assert trainer.completed_steps == 2
            assert all(torch.isfinite(p).all() for m in (g, d, trainer.prior) for p in m.parameters())
        else:
            fixture = ByteGANBiGANFixture(
                device="cpu",
                seed=0,
                recipe_name=None,
                max_steps=10,
                components=context,
            )
            assert isinstance(unwrap_prior(fixture.prior), ParticlePrior)
            for _ in range(2):
                out = fixture.step()
                assert torch.isfinite(out["loss_d"]) and torch.isfinite(out["loss_g"])
            assert fixture.completed_steps == 2
            assert all(
                torch.isfinite(p).all()
                for m in (fixture.G, fixture.E, fixture.D, fixture.prior)
                for p in m.parameters()
            )
    finally:
        torch.set_num_threads(old)


def test_checkpoint_continuation_and_deliberate_damage_both_fixtures():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # --- 1. ByteGANToyFixture (bytegan_hypercube8) ---
        direct = ByteGANToyFixture(recipe_name="unified_v12", seed=0, max_steps=4)
        for _ in range(4):
            direct.step()
        direct_obs = direct.observe(seed=10000)

        part1 = build_case("api-bytegan-hypercube8", recipe_name="unified_v12", seed=0, max_steps=4)
        for _ in range(2):
            part1.step()
        saved = deepcopy(part1.state_dict())

        resumed = build_case("api-bytegan-hypercube8", recipe_name="unified_v12", seed=99, max_steps=4)
        resumed.load_state_dict(saved)
        for _ in range(2):
            resumed.step()
        resumed_obs = resumed.observe(seed=10000)

        assert resumed.completed_steps == 4
        assert state_digest(direct.trainer.G.state_dict()) == state_digest(resumed.trainer.G.state_dict())
        assert state_digest(direct.trainer.D.state_dict()) == state_digest(resumed.trainer.D.state_dict())
        assert torch.equal(unwrap_prior(direct.trainer.prior).z, unwrap_prior(resumed.trainer.prior).z)
        assert direct_obs["metrics"] == resumed_obs["metrics"]

        # Deliberate damage to measured G immediately fails.
        with torch.no_grad():
            for p in resumed.trainer.G.parameters():
                p.fill_(50.0)
        damaged_g = resumed.observe(seed=10000)
        assert damaged_g["passed"] is False
        assert damaged_g["metrics"]["valid_unique_bytes"] == 0
        assert damaged_g["metrics"]["valid_particle_fraction"] == 0.0

        # --- 2. ByteGANBiGANFixture (bytegan_bigan_dictionary) ---
        b_direct = ByteGANBiGANFixture(recipe_name="unified_v12", seed=0, max_steps=4)
        for _ in range(4):
            b_direct.step()
        b_direct_obs = b_direct.observe()

        b_part1 = build_case("api-bytegan-bigan-dictionary", recipe_name="unified_v12", seed=0, max_steps=4)
        for _ in range(2):
            b_part1.step()
        b_saved = deepcopy(b_part1.state_dict())

        b_resumed = build_case("api-bytegan-bigan-dictionary", recipe_name="unified_v12", seed=99, max_steps=4)
        b_resumed.load_state_dict(b_saved)
        for _ in range(2):
            b_resumed.step()
        b_resumed_obs = b_resumed.observe()

        assert b_resumed.completed_steps == 4
        assert state_digest(b_direct.G.state_dict()) == state_digest(b_resumed.G.state_dict())
        assert state_digest(b_direct.E.state_dict()) == state_digest(b_resumed.E.state_dict())
        assert state_digest(b_direct.D.state_dict()) == state_digest(b_resumed.D.state_dict())
        assert torch.equal(unwrap_prior(b_direct.prior).z, unwrap_prior(b_resumed.prior).z)
        assert b_direct_obs["metrics"] == b_resumed_obs["metrics"]

        # Deliberate damage to measured E collapses encoder bijection and fails immediately.
        with torch.no_grad():
            for p in b_resumed.E.parameters():
                p.zero_()
        damaged_e = b_resumed.observe()
        assert damaged_e["passed"] is False
        assert damaged_e["metrics"]["encoder_particle_bijection_bytes"] == 1

        # Deliberate damage to measured G explodes outputs and fails immediately.
        with torch.no_grad():
            for p in b_resumed.G.parameters():
                p.fill_(50.0)
        damaged_bg = b_resumed.observe()
        assert damaged_bg["passed"] is False
        assert damaged_bg["metrics"]["valid_unique_bytes"] == 0
        assert damaged_bg["metrics"]["recon_valid_exact_bytes"] == 0

        # Also verify BiGAN checkpoint continuation under KA2 (EMA critic + input noise + joint recon + transport).
        from particlegan import get_recipe

        r_extra = get_recipe(
            "ka2",
            num_particles=256,
            z_dim=8,
            batch_size=128,
            prior_kind="particles",
            sigma_rel=0.0,
            standardize=False,
            joint_reconstruction_weight=0.5,
            kinetic_transport_weight=0.1,
            total_steps=4,
        )
        k_direct = ByteGANBiGANFixture(recipe=r_extra, seed=0, max_steps=4)
        for _ in range(4):
            out = k_direct.step()
            assert "joint_reconstruction" in out and torch.isfinite(out["joint_reconstruction"])
        k_part1 = ByteGANBiGANFixture(recipe=r_extra, seed=0, max_steps=4)
        for _ in range(2):
            k_part1.step()
        k_resumed = ByteGANBiGANFixture(recipe=r_extra, seed=99, max_steps=4)
        k_resumed.load_state_dict(deepcopy(k_part1.state_dict()))
        for _ in range(2):
            k_resumed.step()
        assert k_resumed.transport is not None and k_resumed.transport.calls == 4
        assert state_digest(k_direct.G.state_dict()) == state_digest(k_resumed.G.state_dict())
        assert state_digest(k_direct.E.state_dict()) == state_digest(k_resumed.E.state_dict())
        assert state_digest(k_direct.D.state_dict()) == state_digest(k_resumed.D.state_dict())
        assert torch.equal(unwrap_prior(k_direct.prior).z, unwrap_prior(k_resumed.prior).z)
    finally:
        torch.set_num_threads(old)


def test_fresh_process_import_order_isolation():
    code = (
        "import inspect, json, torch; "
        "torch.manual_seed(0); "
        "before = torch.get_rng_state()[:16].tolist(); "
        "{first}; {second}; "
        "import particlegan; "
        "from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator; "
        "r = particlegan.get_recipe(); "
        "print(json.dumps({{"
        "'default_recipe': r.name, "
        "'prior_cls': particlegan.ParticlePrior.__qualname__, "
        "'mog_cls': particlegan.MoGParticlePrior.__qualname__, "
        "'trainer_cls': particlegan.GANTrainer.__qualname__, "
        "'gen_sig': str(inspect.signature(SimpleMLPGenerator.__init__)), "
        "'disc_sig': str(inspect.signature(SimpleMLPDiscriminator.__init__)), "
        "'rng_unchanged': before == torch.get_rng_state()[:16].tolist()"
        "}}))"
    )
    out_a = subprocess.run(
        [sys.executable, "-c", code.format(first="import particlegan", second="import benchmarks.toy_audit.api_bytegan")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1]
    out_b = subprocess.run(
        [sys.executable, "-c", code.format(first="import benchmarks.toy_audit.api_bytegan", second="import particlegan")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1]
    parsed_a, parsed_b = json.loads(out_a), json.loads(out_b)
    assert parsed_a == parsed_b
    assert parsed_a["rng_unchanged"] is True


def test_public_api_contract_and_full_schedule_grade_acceptance():
    import math
    from benchmarks.toy_audit.api_bytegan import list_cases
    from benchmarks.toy_audit.api_contract import validate_case, validate_observation
    from experiments.forge.bytegan import _contract_payload

    for case in list_cases():
        validate_case(dict(case))

    validate_observation(score_bytegan_hypercube8(BYTE_CORNERS, BYTE_CORNERS))
    validate_observation(score_bytegan_hypercube8(BYTE_CORNERS, torch.full((256, 8), float("nan"))))
    validate_observation(score_bytegan_bigan_dictionary(BYTE_CORNERS, BYTE_CORNERS, BYTE_CORNERS, BYTE_CORNERS))
    validate_observation(
        score_bytegan_bigan_dictionary(BYTE_CORNERS, BYTE_CORNERS, BYTE_CORNERS, torch.full((256, 8), float("inf")))
    )

    for kind in ("bytegan_hypercube8", "bytegan_bigan_dictionary"):
        task = _make_task(kind)
        steps = task["execution"]["steps"]
        sched = sorted({math.ceil(i * steps / 24) for i in range(1, 25)})
        evidence = {
            "bytegan_contract": _contract_payload(kind, task["execution"]["host_definition"]),
            "actual_training_gif": {"path": "goal.gif", "sha256": "0" * 64, "frames": 24},
            "observations": [{"step": s} for s in sched],
            "live": {"step": steps},
        }
        assert bytegan_grade(task, evidence) is None


def test_short_software_prefixes_render_gifs_isolate_rng_and_grade_fail_closed(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for kind in ("bytegan_hypercube8", "bytegan_bigan_dictionary"):
            original = _make_task(kind)
            prefix = deepcopy(original)
            prefix["id"] += "-software-prefix"
            prefix["execution"]["steps"] = 3
            prefix["execution"]["host_definition"]["steps"] = 3
            prefix["evaluation"]["observations"] = 3
            request = {"candidate": CANDIDATE, "protocol": {"seed": 0}, "tasks": {prefix["id"]: prefix}}
            out_dir = tmp_path / kind
            raw = bytegan_run(request, prefix, out_dir, "cpu")

            assert raw["cost"]["completed_steps"] == 3
            assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
            assert raw["evidence"]["guards"]["all_finite"] is True
            assert raw["evidence"]["guards"]["hooks_exercised"] is True
            assert (out_dir / "goal.gif").stat().st_size > 100
            assert (out_dir / "state.pt").is_file()

            expected_roles = (
                {"generator", "encoder", "prior", "discriminator"}
                if kind == "bytegan_bigan_dictionary"
                else {"generator", "prior", "discriminator"}
            )
            assert set(raw["evidence"]["guards"]["optimizer_updates"]) == expected_roles
            assert all(raw["evidence"]["guards"]["optimizer_updates"][role] == 3 for role in expected_roles)

            # 3-step prefix graded against full 24-observation task must be INCOMPLETE.
            verdict = bytegan_grade(original, raw["evidence"])
            assert verdict is not None and verdict[0] == "INCOMPLETE"

            # Missing contract -> INCOMPLETE.
            missing_contract = deepcopy(raw["evidence"])
            del missing_contract["bytegan_contract"]
            verdict_missing = bytegan_grade(original, missing_contract)
            assert verdict_missing is not None and verdict_missing[0] == "INCOMPLETE"

            # Tampered contract -> INVALID.
            wrong_contract = deepcopy(raw["evidence"])
            wrong_contract["bytegan_contract"]["sigma"] = 0.025
            verdict_wrong = bytegan_grade(original, wrong_contract)
            assert verdict_wrong is not None and verdict_wrong[0] == "INVALID"
    finally:
        torch.set_num_threads(old)
