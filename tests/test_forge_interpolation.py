"""Semantic controls, public-API continuation, and fail-closed grading for particle-interpolation stretch toys."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_interpolation import (
    InterpolationToyFixture,
    build_case,
    interpolation_oracle_controls,
    list_cases,
    score_img_patch_shift,
    score_saddle_barycentric,
    score_sparse_arc,
)
from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.priors import prior_code_path, prior_noise_sigma
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view
from particlegan import ParticlePrior, unwrap_prior

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {"recipe_preset": "empf", "claim_contract": {"sampling_law": "task_declared"}}
INTERP_TASKS = ("interp_sparse_arc", "interp_img_patch_shift", "interp_saddle_barycentric")


def _load_task(name: str) -> dict:
    return json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())


def test_interpolation_stretch_preserves_view_chain_and_required_assignments():
    view = load_view(ROOT, "discriminator_stability")
    v16 = json.loads((ROOT / view["previous_revision"]["archive"]).read_text())
    v15 = json.loads((ROOT / v16["previous_revision"]["archive"]).read_text())
    assert view["revision"] == 17 and view["previous_revision"]["revision"] == 16
    assert [a for a in view["assignments"] if a["task"] not in INTERP_TASKS] == v16["assignments"]
    assert [a for a in view["assignments"] if a["importance"] == "required"] == [
        a for a in v15["assignments"] if a["importance"] == "required"
    ]
    interp_roles = {a["task"]: (a["qualification_tier"], a["importance"]) for a in view["assignments"] if a["task"] in INTERP_TASKS}
    assert interp_roles == {
        "interp_sparse_arc": (1, "stretch"),
        "interp_img_patch_shift": (2, "stretch"),
        "interp_saddle_barycentric": (2, "stretch"),
    }
    tasks = load_tasks(ROOT)
    for name in INTERP_TASKS:
        prior = tasks[name]["execution"]["prior"]
        assert prior["kind"] == "particle_cloud" and prior["sigma"] == 0.0 and prior["standardize"] is False


def test_oracle_positive_destructive_negative_and_deliberate_damage_controls():
    controls = interpolation_oracle_controls()
    assert set(controls) == {
        "arc_smooth_oracle",
        "arc_chord_cutter_negative",
        "arc_voronoi_snap_negative",
        "patch_smooth_oracle",
        "patch_ghost_blend_negative",
        "patch_voronoi_snap_negative",
        "saddle_smooth_oracle",
        "saddle_linear_simplex_negative",
        "saddle_flat_plane_negative",
        "saddle_voronoi_snap_negative",
    }
    for name, item in controls.items():
        assert item["passed"] is item["expected_pass"], f"{name}: {item}"

    # Verify specific failure selectivity across all failure modes.
    assert any("midpoint_novelty" in b for b in controls["arc_voronoi_snap_negative"]["failed_bounds"])
    assert any("interp_hq" in b for b in controls["arc_chord_cutter_negative"]["failed_bounds"])
    assert any("interp_peak" in b for b in controls["patch_ghost_blend_negative"]["failed_bounds"])
    assert any("midpoint_novelty" in b for b in controls["patch_voronoi_snap_negative"]["failed_bounds"])
    assert any("barycenter_hq" in b for b in controls["saddle_linear_simplex_negative"]["failed_bounds"])
    assert any("barycenter_surface_mae" in b for b in controls["saddle_flat_plane_negative"]["failed_bounds"])
    assert any("barycenter_novelty" in b for b in controls["saddle_voronoi_snap_negative"]["failed_bounds"])

    # Nonfinite outputs fail closed across all three scorers without raising.
    nan_z = torch.zeros((12, 2))
    bad_arc = score_sparse_arc(nan_z, torch.full((12, 2), float("nan")), lambda z: torch.zeros((len(z), 2)))
    assert bad_arc["passed"] is False and "finite_fraction == 1.0" in bad_arc["failed_bounds"]

    bad_patch = score_img_patch_shift(
        torch.zeros((16, 4)),
        torch.zeros((16, 1, 8, 8)),
        lambda z: torch.full((len(z), 1, 8, 8), float("inf")),
    )
    assert bad_patch["passed"] is False and "finite_fraction == 1.0" in bad_patch["failed_bounds"]

    bad_sad = score_saddle_barycentric(
        torch.zeros((24, 2)),
        torch.zeros((24, 3)),
        lambda z: torch.full((len(z), 3), float("nan")),
    )
    assert bad_sad["passed"] is False and "finite_fraction == 1.0" in bad_sad["failed_bounds"]


@pytest.mark.parametrize(
    "bad_prior",
    [
        {"kind": "mog", "sigma": 0.025, "standardize": False},
        {"kind": "particle_cloud", "sigma": 0.01, "standardize": False, "exception_reason": "nonzero"},
        {"kind": "particle_cloud", "sigma": 0.0, "standardize": True, "exception_reason": "standardized"},
        {"kind": "particle_cloud", "sigma": 0.0, "standardize": False, "exception_reason": ""},
    ],
)
def test_forbidden_mog_and_nonzero_sigma_priors_never_run(bad_prior):
    from experiments.forge.interpolation import grade

    value = _load_task("interp_sparse_arc")
    value["execution"]["prior"] = bad_prior
    assert adapter_preflight(value, CANDIDATE, root=ROOT)
    assert grade(value, {})[0] == "INVALID"


def test_fresh_gaussian_code_law_is_refused_for_particle_interpolation():
    value = _load_task("interp_sparse_arc")
    fresh_candidate = {
        "recipe_preset": "empf",
        "recipe_overrides": {"prior_update": "frozen", "prior_code_law": "fresh_gaussian"},
        "claim_contract": {"sampling_law": "task_declared"},
    }
    blockers = adapter_preflight(value, fresh_candidate, root=ROOT)
    assert blockers and any("fresh_gaussian" in b or "table" in b for b in blockers)


@pytest.mark.parametrize(
    "family",
    ["empf", "atlas", "bcap-pure", "bispf", "e22", "k3p", "ka2", "r1r2", "release07-gan-v3"],
)
def test_every_selected_family_binds_zero_sigma_particle_prior_or_declared_blocker(family):
    from benchmarks.toy_audit.api_interpolation import (
        build_models_for_spec,
        sample_target_for_kind,
    )
    from experiments.forge.interpolation import interpolation_context
    from experiments.forge.planning import load_idea
    from experiments.forge.trainer_families import current_family_candidates

    value = _load_task("interp_sparse_arc")
    choice = next(row for row in current_family_candidates(ROOT) if row["family"] == family)
    candidate = load_idea(ROOT, choice["candidate_id"])
    blockers = adapter_preflight(value, candidate, root=ROOT)
    if blockers:
        assert family == "e22" and any("live_sampling" in b or "policy" in b for b in blockers)
        return

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        context = interpolation_context({"candidate": candidate, "protocol": {"seed": 0}}, value, "cpu", root=ROOT)
        assert prior_noise_sigma(context.prior_config) == 0.0
        assert prior_code_path(context.prior_config) == "ParticlePrior"
        spec = value["execution"]["host_definition"]
        g = context.construct(lambda: build_models_for_spec(spec, z_dim=context.recipe.z_dim)[0], component="generator")
        d = context.construct(lambda: build_models_for_spec(spec, z_dim=context.recipe.z_dim)[1], component="discriminator")
        trainer = context.build_trainer(g, d, max_steps=10)
        assert isinstance(unwrap_prior(trainer.prior), ParticlePrior)
        rng = context.streams.generator("data", component="target", purpose="training", device="cpu")
        for _ in range(2):
            trainer.step(sample_target_for_kind("interp_sparse_arc", context.recipe.batch_size, rng, device="cpu"))
        assert trainer.completed_steps == 2
        assert all(torch.isfinite(p).all() for m in (g, d, trainer.prior) for p in m.parameters())
    finally:
        torch.set_num_threads(old)


def test_public_api_fixture_checkpoint_continuation_and_deliberate_damage():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        case = next(c for c in list_cases() if c["id"] == "api-interp-sparse-arc")
        direct = InterpolationToyFixture(case, recipe_name="empf", seed=0, max_steps=4)
        for _ in range(4):
            direct.step()
        direct_obs = direct.observe(n=256, seed=10000)

        part1 = build_case("api-interp-sparse-arc", recipe_name="empf", seed=0, max_steps=4)
        for _ in range(2):
            part1.step()
        saved = deepcopy(part1.state_dict())

        resumed = build_case("api-interp-sparse-arc", recipe_name="empf", seed=99, max_steps=4)
        resumed.load_state_dict(saved)
        for _ in range(2):
            resumed.step()
        resumed_obs = resumed.observe(n=256, seed=10000)

        assert resumed.completed_steps == 4
        assert state_digest(direct.trainer.G.state_dict()) == state_digest(resumed.trainer.G.state_dict())
        assert state_digest(direct.trainer.D.state_dict()) == state_digest(resumed.trainer.D.state_dict())
        assert torch.equal(unwrap_prior(direct.trainer.prior).z, unwrap_prior(resumed.trainer.prior).z)
        assert direct_obs["metrics"] == resumed_obs["metrics"]

        # Deliberate damage to the measured generator changes/degrades the evaluated output immediately.
        with torch.no_grad():
            for p in resumed.trainer.G.parameters():
                p.fill_(50.0)
        damaged = resumed.observe(n=256, seed=10000)
        assert damaged["passed"] is False
        assert damaged["metrics"]["endpoint_hq"] == 0.0 and damaged["metrics"]["interp_hq"] == 0.0
    finally:
        torch.set_num_threads(old)


def test_short_software_prefixes_render_gifs_isolate_rng_and_grade_incomplete(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name in INTERP_TASKS:
            original = _load_task(name)
            prefix = deepcopy(original)
            prefix["id"] += "-software-prefix"
            prefix["execution"]["steps"] = 3
            prefix["execution"]["host_definition"]["steps"] = 3
            prefix["evaluation"]["observations"] = 3
            request = {"candidate": CANDIDATE, "protocol": {"seed": 0}, "tasks": {prefix["id"]: prefix}}
            raw = run_task(request, {"task_id": prefix["id"]}, tmp_path / name, "cpu")
            assert raw["cost"]["completed_steps"] == 3
            assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
            assert raw["evidence"]["guards"]["all_finite"] is True
            assert (tmp_path / name / "goal.gif").stat().st_size > 100
            assert grade_result(original, raw)["gate_status"] == "INCOMPLETE"

            missing_contract = deepcopy(raw)
            del missing_contract["evidence"]["interpolation_contract"]
            assert grade_result(original, missing_contract)["gate_status"] == "INCOMPLETE"

            wrong_contract = deepcopy(raw)
            wrong_contract["evidence"]["interpolation_contract"]["sigma"] = 0.025
            assert grade_result(original, wrong_contract)["gate_status"] == "INVALID"
    finally:
        torch.set_num_threads(old)
