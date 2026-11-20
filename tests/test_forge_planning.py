from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, validate_idea
from experiments.forge.planning import plan_summary, resolve_idea


@pytest.fixture
def checkout(tmp_path):
    defaults = {"protocol": "screening", "prior": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}}
    atomic_json(tmp_path / "configs/forge/defaults.json", defaults)
    atomic_json(tmp_path / "configs/forge/protocols/screening.json", {
        "schema_version": 1, "id": "screening", "seed": 0, "rng": {"version": "forge-rng-v1"}, "scoring": {"weights": "live"}})
    atomic_json(tmp_path / "configs/forge/ideas/base.json", {
        "schema_version": 1, "id": "base", "goal": "stability", "hypothesis": "improve the penalty",
        "changed_factors": ["critic penalty"], "mechanism_class": "structural", "recipe_overrides": {},
        "claim_contract": {"sampling_law": "task_declared"}})
    assignments = []
    for index in (1, 2, 3):
        task = {"schema_version": 1, "id": f"t{index}", "adapter": "transfer_behavior",
                "execution": {"initializer": "deterministic_orthogonal", "steps": 80, "prior": defaults["prior"], "host": "mode_hold"},
                "evaluation": {"kind": "transfer_sustained", "thresholds": [["score", ">=", 1]],
                               "sampling_contract_version": 1, "sampling_law": "public_prior_without_output_noise",
                               "eval_output_noise": "clean"},
                "resources": {"gpus": 1, "gpu_memory_mb": 10, "cpu_threads": 1, "timeout_seconds": 10},
                "requires_capabilities": ["named_rng"], "dependencies": []}
        atomic_json(tmp_path / f"configs/forge/tasks/t{index}.json", task)
        assignments.append({"task": task["id"], "qualification_tier": index, "importance": "required", "order": 0})
    atomic_json(tmp_path / "configs/forge/views/stability.json", {
        "schema_version": 1, "id": "stability", "revision": 1, "goal": "stability", "assignments": assignments, "eligibility": {}})
    (tmp_path / "particlegan").mkdir()
    (tmp_path / "particlegan/mechanism.py").write_text("gain = 1\n")
    return tmp_path


def test_plan_does_not_write_and_honors_tier_cap(checkout):
    before = {str(p): p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    request = resolve_idea(checkout, "base")
    after = {str(p): p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    assert before == after
    summary = plan_summary(request)
    assert summary["worst_case_seconds"] == 10
    assert [r["permitted_by_tier_cap"] for r in summary["tasks"]] == [True, False, False]
    assert request["rng"]["bindings"]
    assert request["execution_policy"] == summary["execution_policy"] == {
        "schema_version": 1, "mode": "complete_current_tier"}


def test_planning_exposes_task_bound_values_and_protocol_ownership(checkout):
    request = resolve_idea(checkout, "base")
    summary = plan_summary(request, include_ownership=True)
    ownership = summary["tasks"][0]["field_ownership"]
    assert ownership["recipe_fields"]["total_steps"]["value"] == 80
    assert ownership["recipe_fields"]["total_steps"]["owner"] == "task"
    assert ownership["recipe_fields"]["batch_size"]["value"] == 128
    assert ownership["recipe_fields"]["batch_size"]["owner"] == "task"
    assert ownership["recipe_fields"]["lr"]["owner"] == "hyperparameter"
    assert ownership["protocol"]["seed"]["owner"] == "protocol"
    assert ownership["task_contract"]["initialization"]["value"] == "deterministic_orthogonal"
    assert request["jobs"][0]["science"]["task_initializers"] == {"t1": "deterministic_orthogonal"}


def test_planner_preserves_experiment_priors_over_candidate_reference(checkout):
    idea_path = checkout / "configs/forge/ideas/base.json"
    idea = read_json(idea_path)
    idea["prior"] = {"sigma": .1}
    atomic_json(idea_path, idea)
    particle_path = checkout / "configs/forge/tasks/t2.json"
    particle = read_json(particle_path)
    particle["execution"]["prior"] = {"kind": "particle_cloud", "sigma": 0,
        "standardize": False, "learnable": True, "exception_reason": "Explicit cloud fixture"}
    particle["requires_capabilities"].append("particle_cloud")
    atomic_json(particle_path, particle)
    request = resolve_idea(checkout, "base")
    assert request["candidate"]["prior"]["sigma"] == .1
    for name in ("t1", "t2", "t3"):
        declaration = read_json(checkout / f"configs/forge/tasks/{name}.json")
        assert request["tasks"][name]["execution"]["prior"] == declaration["execution"]["prior"]
        assert request["tasks"][name]["preflight_blockers"] == []


def test_changing_task_prior_changes_its_job_without_changing_other_experiments(checkout):
    old = resolve_idea(checkout, "base")
    path = checkout / "configs/forge/tasks/t1.json"
    value = read_json(path)
    value["execution"]["prior"]["sigma"] = .05
    atomic_json(path, value)
    new = resolve_idea(checkout, "base")
    assert old["candidate_revision"] == new["candidate_revision"]
    keys = lambda r: {j["task_id"]: j["compatibility_key"] for j in r["jobs"]}
    before, after = keys(old), keys(new)
    assert before["t1"] != after["t1"]
    assert before["t2"] == after["t2"] and before["t3"] == after["t3"]


def test_retiering_reuses_keys_but_freezes_old_policy(checkout):
    old = resolve_idea(checkout, "base")
    path = checkout / "configs/forge/views/stability.json"
    view = read_json(path)
    view["assignments"][0]["task"], view["assignments"][1]["task"] = "t2", "t1"
    view["revision"] = 2
    atomic_json(path, view)
    new = resolve_idea(checkout, "base")
    keys = lambda r: {j["task_id"]: j["compatibility_key"] for j in r["jobs"]}
    assert old["candidate_revision"] == new["candidate_revision"]
    assert keys(old) == keys(new)
    assert old["policy_fingerprint"] != new["policy_fingerprint"]
    assert old["view"]["assignments"][0]["task"] == "t1"


def test_metadata_rename_is_duplicate_but_code_change_invalidates(checkout):
    old = resolve_idea(checkout, "base")
    idea = read_json(checkout / "configs/forge/ideas/base.json")
    idea.update(id="renamed", hypothesis="same mechanism, new prose")
    atomic_json(checkout / "configs/forge/ideas/renamed.json", idea)
    new = resolve_idea(checkout, "renamed")
    assert old["candidate_revision"] == new["candidate_revision"]
    assert old["jobs"] == new["jobs"]
    (checkout / "particlegan/mechanism.py").write_text("gain = 2\n")
    changed = resolve_idea(checkout, "renamed")
    assert changed["candidate_revision"] != old["candidate_revision"]
    assert changed["jobs"][0]["compatibility_key"] != old["jobs"][0]["compatibility_key"]


def test_different_seed_or_sampling_law_never_reuses_evidence(checkout):
    old = resolve_idea(checkout, "base")
    path = checkout / "configs/forge/protocols/screening.json"
    protocol = read_json(path)
    protocol["seed"] = 1
    atomic_json(path, protocol)
    changed = resolve_idea(checkout, "base")
    assert changed["candidate_revision"] == old["candidate_revision"]
    assert changed["jobs"][0]["compatibility_key"] != old["jobs"][0]["compatibility_key"]
    path = checkout / "configs/forge/ideas/base.json"
    idea = read_json(path)
    idea["prior"] = {"sigma": .05}
    atomic_json(path, idea)
    assert resolve_idea(checkout, "base")["candidate_revision"] != old["candidate_revision"]


def test_seed_only_and_unfinished_proposals_do_not_enqueue(checkout):
    path = checkout / "configs/forge/ideas/base.json"
    idea = read_json(path)
    idea["changed_factors"] = ["seed"]
    with pytest.raises(ValueError, match="seed-only"):
        validate_idea(idea)
    idea["changed_factors"] = ["TODO: finish"]
    atomic_json(path, idea)
    with pytest.raises(ValueError, match="submission blocked"):
        resolve_idea(checkout, "base", freeze_source=True, queue_root=checkout / "runs/forge")
    assert not (checkout / "runs").exists()


def test_continuation_group_reserves_whole_execution_once(checkout):
    view_path = checkout / "configs/forge/views/stability.json"
    view = read_json(view_path)
    view["assignments"][2]["qualification_tier"] = 2
    view["assignments"][2]["order"] = 1
    atomic_json(view_path, view)
    for task_id in ("t2", "t3"):
        path = checkout / f"configs/forge/tasks/{task_id}.json"
        task = read_json(path)
        task["execution"]["execution_group"] = "continuation"
        task["execution"]["produces_state"] = True
        if task_id == "t3":
            task["dependencies"] = [{"task": "t2", "kind": "checkpoint"}]
            task["execution"]["continuation_of"] = "t2"
        atomic_json(path, task)
    request = resolve_idea(checkout, "base", through_tier=3)
    assert len(request["jobs"]) == 2
    assert request["jobs"][-1]["task_id"] == "t2"
    assert request["jobs"][-1]["task_ids"] == ["t2", "t3"]
    assert plan_summary(request)["worst_case_seconds"] == 20


def test_another_view_reuses_identical_task_evidence_with_extra_evaluators(checkout):
    extra = checkout / "reports/frozen/evaluator.py"
    extra.parent.mkdir(parents=True)
    extra.write_text("threshold = 1\n")
    task_path = checkout / "configs/forge/tasks/t3.json"
    task = read_json(task_path)
    task["evaluation"]["sources"] = {"reports/frozen/evaluator.py": file_hash(extra)}
    atomic_json(task_path, task)
    view = read_json(checkout / "configs/forge/views/stability.json")
    view.update(id="quality", goal="quality", assignments=view["assignments"][:1])
    atomic_json(checkout / "configs/forge/views/quality.json", view)
    stability = resolve_idea(checkout, "base")
    quality = resolve_idea(checkout, "base", view_id="quality")
    assert stability["candidate_revision"] == quality["candidate_revision"]
    assert stability["jobs"][0]["compatibility_key"] == quality["jobs"][0]["compatibility_key"]


def test_published_host_sources_are_frozen_independently_of_selected_view(checkout):
    from experiments.forge.sources import snapshot_source
    from experiments.forge.vectorprofiles import PROFILE_SOURCES, LEADING_SOURCE, resolve_vector_spec
    root = Path(__file__).resolve().parents[1]
    for relative in PROFILE_SOURCES:
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    task = read_json(root / "configs/forge/tasks/vector_unequal_mass_published.json")
    for relative in task["evaluation"]["sources"]:
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    atomic_json(checkout / f"configs/forge/tasks/{task['id']}.json", task)
    path = checkout / "configs/forge/views/stability.json"
    view = read_json(path)
    atomic_json(checkout / "configs/forge/views/quality.json", {
        **view, "id": "quality", "goal": "quality", "assignments": view["assignments"][:1]})
    view["assignments"].append({"task": task["id"], "qualification_tier": 2,
                                "importance": "diagnostic", "order": 1})
    atomic_json(path, view)
    stability = resolve_idea(checkout, "base", execution_backend="cpu")
    quality = resolve_idea(checkout, "base", view_id="quality", execution_backend="cpu")
    assert stability["source"]["files"][LEADING_SOURCE] == PROFILE_SOURCES[LEADING_SOURCE]
    assert stability["candidate_revision"] == quality["candidate_revision"]
    assert stability["jobs"][0]["compatibility_key"] == quality["jobs"][0]["compatibility_key"]
    frozen = snapshot_source(checkout, checkout / "runs/forge", stability["source"])
    assert resolve_vector_spec(stability["tasks"][task["id"]], root=frozen) == task["execution"]["host_definition"]
    (checkout / LEADING_SOURCE).unlink()
    missing = resolve_idea(checkout, "base", execution_backend="cpu")
    assert missing["tasks"][task["id"]]["preflight_blockers"]


@pytest.mark.parametrize("cached", [False, True])
def test_model_provider_resources_are_frozen_outside_selected_view(checkout, cached):
    from contextlib import nullcontext
    from experiments.forge.model_toy_sources import model_toy_source_paths
    from experiments.forge.planning import resolution_cache
    task = read_json(checkout / "configs/forge/tasks/t3.json")
    task["adapter"] = "public_api_model_toy"
    task["execution"]["host_definition"] = {"provider": "benchmarks.toy_audit.api_model_ae_ddgan"}
    atomic_json(checkout / "configs/forge/tasks/t3.json", task)
    support = model_toy_source_paths(task)
    for relative in support:
        resource = read_json(checkout / "configs/forge/tasks/t1.json")
        resource["id"] = Path(relative).stem
        atomic_json(checkout / relative, resource)
    view = read_json(checkout / "configs/forge/views/stability.json")
    atomic_json(checkout / "configs/forge/views/quality.json", {
        **view, "id": "quality", "goal": "quality", "assignments": view["assignments"][:1]})
    with resolution_cache(checkout) if cached else nullcontext():
        request = resolve_idea(checkout, "base", view_id="quality", execution_backend="cpu")
    for relative in support:
        assert request["source"]["files"][relative] == file_hash(checkout / relative)


def test_changed_prerequisite_invalidates_downstream_evidence_identity(checkout):
    for child, parent in (("t2", "t1"), ("t3", "t2")):
        path = checkout / f"configs/forge/tasks/{child}.json"
        value = read_json(path)
        value["dependencies"] = [{"task": parent, "kind": "gate"}]
        atomic_json(path, value)
    before = resolve_idea(checkout, "base")
    path = checkout / "configs/forge/tasks/t1.json"
    value = read_json(path)
    value["execution"]["steps"] += 1
    atomic_json(path, value)
    after = resolve_idea(checkout, "base")
    assert before["candidate_revision"] == after["candidate_revision"]
    assert all(a["compatibility_key"] != b["compatibility_key"] for a, b in zip(before["jobs"], after["jobs"]))


def test_unknown_adapter_is_visible_in_read_only_preflight(checkout):
    path = checkout / "configs/forge/tasks/t1.json"
    value = read_json(path)
    value["adapter"] = "unavailable_formulation"
    atomic_json(path, value)
    request = resolve_idea(checkout, "base")
    assert "no public adapter" in request["tasks"]["t1"]["preflight_blockers"][0]
    assert not (checkout / "runs").exists()


def test_actual_threads_and_parent_compute_participate_in_child_identity(checkout):
    child_path = checkout / "configs/forge/tasks/t2.json"
    child = read_json(child_path)
    child["resources"]["gpus"] = 0
    child["dependencies"] = [{"task": "t1", "kind": "gate"}]
    atomic_json(child_path, child)
    before = resolve_idea(checkout, "base")
    parent_path = checkout / "configs/forge/tasks/t1.json"
    parent = read_json(parent_path)
    parent["resources"]["gpus"] = 0
    atomic_json(parent_path, parent)
    after = resolve_idea(checkout, "base")
    assert before["jobs"][1]["science"]["compute"] == after["jobs"][1]["science"]["compute"]
    assert before["jobs"][1]["compatibility_key"] != after["jobs"][1]["compatibility_key"]
    child["resources"]["cpu_threads"] = 2
    atomic_json(child_path, child)
    threads = resolve_idea(checkout, "base")
    assert threads["jobs"][1]["science"]["compute"]["threads"] == 2
    assert threads["jobs"][1]["compatibility_key"] != after["jobs"][1]["compatibility_key"]


def test_new_planning_blocks_ambiguous_candidate_sampling_claim(checkout):
    path = checkout / "configs/forge/ideas/base.json"
    idea = read_json(path)
    idea["claim_contract"]["sampling_law"] = "public_noisy"
    atomic_json(path, idea)
    request = resolve_idea(checkout, "base")
    assert any("task_declared" in value for value in request["preflight_blockers"])
    with pytest.raises(ValueError, match="submission blocked"):
        resolve_idea(checkout, "base", freeze_source=True, queue_root=checkout / "runs")
    assert not (checkout / "runs").exists()


def test_new_planning_blocks_unversioned_tasks_despite_task_delegation(checkout):
    path = checkout / "configs/forge/tasks/t1.json"
    spec = read_json(path)
    spec["evaluation"].pop("sampling_contract_version")
    atomic_json(path, spec)
    request = resolve_idea(checkout, "base")
    assert any("sampling_contract_version" in value for value in request["tasks"]["t1"]["preflight_blockers"])
    assert not (checkout / "runs").exists()


@pytest.mark.parametrize("missing", [True, False])
def test_task_local_evaluator_source_blocker_does_not_block_freezing_peers(checkout, missing):
    relative = "reports/frozen/changed-evaluator.py"
    if not missing:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("threshold = 2\n")
    path = checkout / "configs/forge/tasks/t2.json"
    task = read_json(path)
    task["evaluation"]["sources"] = {relative: "0" * 64}
    atomic_json(path, task)
    request = resolve_idea(checkout, "base", through_tier=2, freeze_source=True, queue_root=checkout / "runs")
    assert request["preflight_blockers"] == []
    assert request["tasks"]["t1"]["preflight_blockers"] == []
    assert "evaluator source" in request["tasks"]["t2"]["preflight_blockers"][0]
    assert Path(request["source"]["snapshot_path"]).exists()
