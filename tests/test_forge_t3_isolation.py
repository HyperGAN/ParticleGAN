"""Semantic controls, fresh-process continuation, import-order checks, and fail-closed grading for Theory-3 isolation toys."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.api_t3_isolation import (
    T3IsolationFixture,
    sample_target_for_kind,
    score_t3_isolation_kind,
    t3_isolation_oracle_controls,
)
from experiments.forge.adapters import _dispatch_task, adapter_preflight
from experiments.forge.priors import prior_code_path, prior_noise_sigma
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {"recipe_preset": "rgdt", "claim_contract": {"sampling_law": "task_declared"}}
ISO_TASKS = ("t3_iso_scale_mass_8x", "t3_iso_horizon_saddle_shift")


def _load_task(name: str) -> dict:
    return json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())


def test_t3_isolation_view_and_tasks_preserve_discriminator_stability_v17():
    main_view = load_view(ROOT, "discriminator_stability")
    assert main_view["revision"] == 18 and main_view["previous_revision"]["revision"] == 17
    stretch_map = {a["task"]: a["importance"] for a in main_view["assignments"] if a["task"] in ISO_TASKS}
    assert stretch_map == {name: "stretch" for name in ISO_TASKS}

    iso_view = load_view(ROOT, "t3_causal_isolation_stretch")
    assert iso_view["evidence_scope"] == "research_diagnostic"
    assert [a["task"] for a in iso_view["assignments"]] == list(ISO_TASKS)

    tasks = load_tasks(ROOT)
    for name in ISO_TASKS:
        t = tasks[name]
        assert t["adapter"] == "t3_isolation"
        assert t["execution"]["execution_path"] == "public_components"
        assert t["execution"]["prior_applicability"] == "not_sampled"
        prior = t["execution"]["prior"]
        assert prior["kind"] == "particle_cloud" and prior["sigma"] == 0.0 and prior["standardize"] is False


def test_oracle_positive_destructive_negative_and_deliberate_damage_controls():
    controls = t3_isolation_oracle_controls()
    assert set(controls) == {
        "scale_mass_oracle_pass",
        "scale_mass_uniform_impostor_fail",
        "scale_mass_sharp_critic_fail",
        "scale_mass_collapsed_narrow_fail",
        "saddle_shift_oracle_pass",
        "saddle_shift_trapped_2mode_fail",
        "saddle_shift_nan_fail",
    }
    for name, item in controls.items():
        assert item["passed"] is item["expected_pass"], f"{name}: {item}"

    assert "mass_tv <= 0.1" in controls["scale_mass_uniform_impostor_fail"]["failed_bounds"]
    assert "critic_grad_med <= 1.0" in controls["scale_mass_sharp_critic_fail"]["failed_bounds"]
    assert any("eigen_ratio" in b or "covariance_error" in b for b in controls["scale_mass_collapsed_narrow_fail"]["failed_bounds"])
    assert "modes == 4" in controls["saddle_shift_trapped_2mode_fail"]["failed_bounds"]
    assert "finite_fraction == 1.0" in controls["saddle_shift_nan_fail"]["failed_bounds"]


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
    from experiments.forge.t3_isolation import grade

    value = _load_task("t3_iso_scale_mass_8x")
    value["execution"]["prior"] = bad_prior
    assert adapter_preflight(value, CANDIDATE, root=ROOT)
    assert grade(value, {})[0] == "INVALID"


def test_fresh_gaussian_and_frozen_prior_are_refused():
    value = _load_task("t3_iso_scale_mass_8x")
    for overrides in (
        {"prior_update": "frozen", "prior_code_law": "fresh_gaussian"},
        {"prior_update": "frozen"},
    ):
        cand = {
            "recipe_preset": "rgdt",
            "recipe_overrides": overrides,
            "claim_contract": {"sampling_law": "task_declared"},
        }
        blockers = adapter_preflight(value, cand, root=ROOT)
        assert blockers


@pytest.mark.parametrize("family", ["empf", "bcap-pure", "bispf", "rhbi"])
def test_selected_families_bind_identity_generator_and_zero_sigma_prior(family):
    from experiments.forge.planning import load_idea
    from experiments.forge.t3_isolation import t3_isolation_context
    from experiments.forge.trainer_families import current_family_candidates

    value = _load_task("t3_iso_scale_mass_8x")
    choice = next(row for row in current_family_candidates(ROOT) if row["family"] == family)
    candidate = load_idea(ROOT, choice["candidate_id"])
    assert adapter_preflight(value, candidate, root=ROOT) == []

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ctx = t3_isolation_context({"candidate": candidate, "protocol": {"seed": 0}}, value, "cpu", root=ROOT)
        assert prior_noise_sigma(ctx.prior_config) == 0.0
        assert prior_code_path(ctx.prior_config) == "ParticlePrior"
        fix = T3IsolationFixture("t3_iso_scale_mass_8x", ctx.recipe, device="cpu", seed=0, max_steps=4)
        assert isinstance(fix.G, nn.Identity)
        assert sum(p.numel() for p in fix.G.parameters()) == 0
        for _ in range(2):
            fix.step()
        assert fix.completed_steps == 2
        assert torch.isfinite(fix.prior.z).all()
    finally:
        torch.set_num_threads(old)


def test_fresh_process_continuation_across_saddle_shift_boundary_and_damage_control(tmp_path):
    """Verify exact continuation across step-300 target shift in a fresh subprocess + deliberate damage."""
    ckpt_path = tmp_path / "shift_boundary.pt"
    out_direct = tmp_path / "direct.pt"
    out_resumed = tmp_path / "resumed.pt"
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"}

    script_direct = f"""
import torch
from particlegan import Recipe, get_recipe
from benchmarks.toy_audit.api_t3_isolation import HOST_DEFINITIONS, T3IsolationFixture, sample_target_for_kind
torch.set_num_threads(1)
spec = HOST_DEFINITIONS["t3_iso_horizon_saddle_shift"]
base = get_recipe("rgdt").to_dict()
base.update({{"num_particles": 128, "z_dim": 4, "batch_size": 16, "total_steps": 304, "prior_kind": "particles", "sigma_rel": 0.0, "standardize": False}})
r = Recipe.from_dict(base)
fix = T3IsolationFixture("t3_iso_horizon_saddle_shift", r, device="cpu", seed=0, max_steps=304)
for s in range(1, 305):
    fix.step()
    if s == 300:
        torch.save(fix.state_dict(), {str(ckpt_path)!r})
ref = sample_target_for_kind("t3_iso_horizon_saddle_shift", 128, torch.Generator().manual_seed(77), step=304)
obs = fix.observe(ref)
torch.save({{"z": fix.prior.z.detach().clone(), "d": fix.D.state_dict(), "metrics": obs["metrics"], "froz": fix.frozen_snapshot}}, {str(out_direct)!r})
"""
    subprocess.run([sys.executable, "-c", script_direct], check=True, env=env)

    script_resume = f"""
import torch
from particlegan import Recipe, get_recipe
from benchmarks.toy_audit.api_t3_isolation import T3IsolationFixture, sample_target_for_kind
torch.set_num_threads(1)
base = get_recipe("rgdt").to_dict()
base.update({{"num_particles": 128, "z_dim": 4, "batch_size": 16, "total_steps": 304, "prior_kind": "particles", "sigma_rel": 0.0, "standardize": False}})
r = Recipe.from_dict(base)
fix = T3IsolationFixture("t3_iso_horizon_saddle_shift", r, device="cpu", seed=999, max_steps=304)
state = torch.load({str(ckpt_path)!r}, weights_only=False)
fix.load_state_dict(state)
for _ in range(4):
    fix.step()
ref = sample_target_for_kind("t3_iso_horizon_saddle_shift", 128, torch.Generator().manual_seed(77), step=304)
obs = fix.observe(ref)
torch.save({{"z": fix.prior.z.detach().clone(), "d": fix.D.state_dict(), "metrics": obs["metrics"], "froz": fix.frozen_snapshot}}, {str(out_resumed)!r})
"""
    subprocess.run([sys.executable, "-c", script_resume], check=True, env=env)

    d_res = torch.load(out_direct, weights_only=False)
    r_res = torch.load(out_resumed, weights_only=False)
    assert torch.equal(d_res["z"], r_res["z"])
    assert torch.equal(d_res["froz"], r_res["froz"])
    assert state_digest(d_res["d"]) == state_digest(r_res["d"])
    assert d_res["metrics"] == r_res["metrics"]

    # Deliberate damage to measured particle cloud flips passing oracle state to immediate FAIL:
    rng = torch.Generator().manual_seed(9)
    ref_sm = sample_target_for_kind("t3_iso_scale_mass_8x", 512, rng)
    good = score_t3_isolation_kind("t3_iso_scale_mass_8x", ref_sm.clone(), ref_sm, critic_grad_med=0.3)
    assert good["passed"] is True
    damaged_pts = ref_sm.clone() + 25.0
    bad = score_t3_isolation_kind("t3_iso_scale_mass_8x", damaged_pts, ref_sm, critic_grad_med=0.3)
    assert bad["passed"] is False and bad["metrics"]["hq"] == 0.0


def test_fresh_process_import_order_independence():
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"}
    cmd_a = "import particlegan, benchmarks.toy_audit.api_t3_isolation as m; assert all(v['passed']==v['expected_pass'] for v in m.t3_isolation_oracle_controls().values())"
    cmd_b = "import benchmarks.toy_audit.api_t3_isolation as m, particlegan; assert all(v['passed']==v['expected_pass'] for v in m.t3_isolation_oracle_controls().values())"
    subprocess.run([sys.executable, "-c", cmd_a], check=True, env=env)
    subprocess.run([sys.executable, "-c", cmd_b], check=True, env=env)


def test_short_software_prefixes_render_gifs_and_tamper_checks_fail_closed(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name in ISO_TASKS:
            original = _load_task(name)
            prefix = deepcopy(original)
            prefix["id"] += "-software-prefix"
            prefix["execution"]["steps"] = 3
            prefix["execution"]["host_definition"]["steps"] = 3
            prefix["evaluation"]["observations"] = 3
            request = {"candidate": CANDIDATE, "protocol": {"seed": 0}, "tasks": {prefix["id"]: prefix}}
            raw = _dispatch_task(request, {"task_id": prefix["id"]}, tmp_path / name, "cpu")
            assert raw["cost"]["completed_steps"] == 3
            assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
            assert raw["evidence"]["guards"]["all_finite"] is True
            assert raw["evidence"]["guards"]["optimizer_updates"] == {"generator": 0, "prior": 3, "discriminator": 3}
            assert (tmp_path / name / "goal.gif").stat().st_size > 100
            assert grade_result(original, {**raw, "status": "COMPLETED"})["status"] == "INCOMPLETE"

            tampered_gen = deepcopy(raw)
            tampered_gen["evidence"]["t3_isolation_contract"]["generator_trainable_params"] = 128
            assert grade_result(prefix, {**tampered_gen, "status": "COMPLETED"})["status"] == "INVALID"
    finally:
        torch.set_num_threads(old)
