"""Public adapter execution, independent timing grading and bounded admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge import adapters, scaling
from experiments.forge.api import task_formulation_context, task_policy_blockers
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.sampling import expected_policy
from experiments.forge.trainer_families import current_family_candidates
from experiments.forge.views import grade_result, load_tasks, load_view, qualify, validate_view
from test_forge_configuration_search import checkout


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_matrix_preserves_selected_recipes_and_clears_all_family_preflights():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "dcgan_performance")
    assert view["evidence_scope"] == "research_diagnostic"
    assert len(view["assignments"]) == 6
    selected = current_family_candidates(ROOT)
    from experiments.forge.planning import load_idea
    global_bindings = {}
    for choice in selected:
        candidate = load_idea(ROOT, choice["candidate_id"])
        for assignment in view["assignments"]:
            task = tasks[assignment["task"]]
            assert adapters.adapter_preflight(task, candidate, root=ROOT) == []
            context = task_formulation_context(candidate, task)
            recipe = context.recipe.to_dict()
            global_recipe = {key: value for key, value in recipe.items()
                             if key not in {"batch_size", "z_dim", "num_particles", "total_steps"}}
            if choice["family"] in global_bindings:
                assert global_recipe == global_bindings[choice["family"]]
            global_bindings[choice["family"]] = global_recipe
            assert context.prior_config["kind"] == "particle_cloud"
            assert context.prior_config["sigma"] == 0
            assert task["dependencies"] == []
            assert assignment["importance"] == "diagnostic"
    assert len(global_bindings) == len(selected) == 10


def test_policy_exception_requires_explicit_performance_contract():
    from experiments.forge.planning import load_idea
    candidate = load_idea(ROOT, "e22")
    task = load_tasks(ROOT)["dcgan_performance_r32"]
    assert task_policy_blockers(task, candidate) == []
    task["evaluation"]["qualification_input"] = True
    assert task_policy_blockers(task, candidate)


def test_public_cpu_adapter_actual_training_and_independent_trace_grade(tmp_path):
    task = load_tasks(ROOT)["dcgan_performance_cpu_smoke"]
    candidate = {"id": "software-test", "recipe_preset": "bcap_adam",
                 "recipe_overrides": {"lr": .0001, "input_noise_std": 0., "output_noise_std": 0.}}
    request = {"candidate": candidate, "candidate_revision": "software-test",
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    old = torch.get_rng_state().clone()
    raw = adapters.run_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert torch.equal(old, torch.get_rng_state())
    assert raw["status"] == "PASS"
    assert raw["candidate_id"] == candidate["id"]
    assert raw["evidence"]["artifacts"]["updates"]["sha256"] == file_hash(tmp_path / "updates.jsonl")
    assert all(raw["evidence"][key] == value for key, value in expected_policy(task).items())
    assert (tmp_path / "training.gif").is_file()
    checkpoint = torch.load(tmp_path / "state.pt", weights_only=False)
    assert checkpoint["data_cursor"]["completed_updates"] == task["execution"]["steps"]
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS"
    assert grade["qualification_input"] is False
    assert grade["metrics"]["measured_updates"] == 2
    # A status stamp cannot fill absent raw timing evidence.
    broken = deepcopy(raw)
    del broken["evidence"]["artifacts"]["updates"]
    assert grade_result(task, broken)["status"] == "INCOMPLETE"
    # Frozen trace hashes bind a success to its actual finite measured updates.
    with (tmp_path / "updates.jsonl").open("a") as stream:
        stream.write("{}\n")
    assert grade_result(task, raw)["status"] == "INVALID"


def test_registration_is_finite_recipe_pinned_and_task_pinned(tmp_path, monkeypatch):
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "dcgan_performance")
    candidate_path = tmp_path / "configs/forge/ideas/example.json"
    atomic_json(candidate_path, {"id": "example"})
    monkeypatch.setattr(scaling, "load_tasks", lambda root: deepcopy(tasks))
    monkeypatch.setattr(scaling, "load_view", lambda root, name: deepcopy(view))
    monkeypatch.setattr(scaling, "load_idea", lambda root, name: {"id": name})
    monkeypatch.setattr(scaling, "current_family_candidates", lambda root, family_ids: [
        {"family": "example", "label": "Example", "candidate_id": "example"}])
    registered = scaling.register(tmp_path, "bounded-test", execution_backend="cpu")
    assert registered["campaign"]["budget_seconds"] == 1200
    assert registered["campaign"]["candidate_budget_seconds"] == 1200
    assert registered["qualification_input"] is False
    manifest = scaling._load(tmp_path, "bounded-test")
    assert len(manifest["task_contract_sha256"]) == 6
    assert scaling.register(tmp_path, "bounded-test", execution_backend="cpu") == registered
    atomic_json(candidate_path, {"id": "example", "changed": True})
    with pytest.raises(ValueError, match="recipe declaration changed"):
        scaling._load(tmp_path, "bounded-test")


def test_explicit_candidate_registration_shares_budget_without_changing_selection(tmp_path, monkeypatch):
    tasks, view = load_tasks(ROOT), load_view(ROOT, "dcgan_performance")
    choices = [{"family": "example", "label": "Example", "candidate_id": "example"}]
    for name in ("example", "example-new"):
        atomic_json(tmp_path / "configs/forge/ideas" / f"{name}.json",
                    {"id": name, "trainer_family": "example"})
    selection = tmp_path / scaling.CURRENT_SELECTION
    atomic_json(selection, {"immutable": "existing selection"})
    original = selection.read_bytes()
    monkeypatch.setattr(scaling, "load_tasks", lambda root: deepcopy(tasks))
    monkeypatch.setattr(scaling, "load_view", lambda root, name: deepcopy(view))
    monkeypatch.setattr(scaling, "load_idea", lambda root, name: {"id": name})
    monkeypatch.setattr(scaling, "current_family_candidates", lambda root, family_ids: choices)
    monkeypatch.setattr(scaling, "family_for_candidate", lambda *args, **kwargs: {"id": "example"})
    campaign = {"id": "shared-performance", "candidate_budget_seconds": 1200, "budget_seconds": 2400}
    control = scaling.register(tmp_path, "control", family_ids=["example"], campaign=campaign)
    candidate = scaling.register(tmp_path, "candidate", family_ids=["example"], campaign=campaign,
                                 candidate_overrides={"example": "example-new"})
    assert control["campaign"] == candidate["campaign"] == campaign
    assert control["family_selections"][0]["candidate_id"] == "example"
    assert candidate["family_selections"][0]["candidate_id"] == "example-new"
    assert candidate["candidate_overrides"] == {"example": "example-new"}
    assert choices[0]["candidate_id"] == "example"
    assert selection.read_bytes() == original
    assert scaling._load(tmp_path, "candidate")["campaign"] == campaign
    assert scaling.register(tmp_path, "candidate", family_ids=["example"], campaign=campaign,
                            candidate_overrides={"example": "example-new"}) == candidate
    changed_campaign = {**campaign, "budget_seconds": 3600}
    with pytest.raises(ValueError, match="immutable"):
        scaling.register(tmp_path, "candidate", family_ids=["example"], campaign=changed_campaign,
                         candidate_overrides={"example": "example-new"})


@pytest.mark.parametrize("change", ["unknown_family", "another_family", "small_campaign", "wrong_candidate_cap"])
def test_explicit_performance_registration_rejects_wrong_family_and_budget(tmp_path, monkeypatch, change):
    monkeypatch.setattr(scaling, "load_tasks", lambda root: load_tasks(ROOT))
    monkeypatch.setattr(scaling, "load_view", lambda root, name: load_view(ROOT, "dcgan_performance"))
    monkeypatch.setattr(scaling, "current_family_candidates", lambda root, family_ids: [
        {"family": "example", "label": "Example", "candidate_id": "example"}])
    monkeypatch.setattr(scaling, "load_idea", lambda root, name: {"id": name})
    monkeypatch.setattr(scaling, "family_for_candidate", lambda *args, **kwargs: {
        "id": "another" if change == "another_family" else "example"})
    campaign = {"id": "shared", "candidate_budget_seconds": 1200, "budget_seconds": 2400}
    if change == "small_campaign":
        campaign["budget_seconds"] = 1199
    if change == "wrong_candidate_cap":
        campaign["candidate_budget_seconds"] = 1199
    overrides = {"unknown" if change == "unknown_family" else "example": "example-new"}
    with pytest.raises(ValueError, match="selected families|another trainer family|full selected matrix"):
        scaling.register(tmp_path, "invalid", candidate_overrides=overrides, campaign=campaign)
    assert not (tmp_path / "configs/forge/performance").exists()


def test_performance_view_never_qualifies_a_recipe():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "dcgan_performance")
    result = qualify(view, tasks, [], candidate={})
    assert result["status"] == "DIAGNOSTIC"
    assert result["qualified_tier"] == 0
    assert result["eligible"] is False
    assert result["qualification_input"] is False
    view.pop("evidence_scope")
    for assignment in view["assignments"]:
        assignment["importance"] = "required"
    with pytest.raises(ValueError, match="cannot grant qualification"):
        validate_view(view, tasks)


def test_actual_frozen_performance_enqueue_accepts_all_eight_selected_recipes_without_launch(tmp_path):
    from experiments.forge.queue import Queue
    from experiments.forge.sources import verify_snapshot
    manifest = scaling._load(ROOT, "dcgan-performance-cloud-v2")
    queue = Queue(tmp_path / "queue", report_root=ROOT / "reports/forge")
    result = scaling.enqueue(ROOT, queue.root, manifest["id"], queue=queue)
    assert len(result["submissions"]) == len(manifest["family_selections"]) == 8
    assert all(entry["status"] == "queued" for entry in result["submissions"])
    saved = queue.inspect()
    assert len(saved["submissions"]) == 8
    assert len(saved["jobs"]) == 48
    for submission in saved["submissions"].values():
        request = submission["request"]
        verify_snapshot(Path(request["source"]["snapshot_path"]), request["source"])
        assert request["study_review"]["status"] == "READY"
    assert len({entry["request"]["source"]["digest"] for entry in saved["submissions"].values()}) == 1
    assert saved["charges"] == []
    assert saved["campaigns"][manifest["campaign"]["id"]]["spent_seconds"] == 0
    assert saved["campaigns"][manifest["campaign"]["id"]]["reserved_seconds"] == 0
    assert all(job["attempts"] == [] and job["status"] == "pending" for job in saved["jobs"].values())


def test_legacy_recipe_exception_does_not_admit_other_diagnostics(checkout):
    from experiments.forge.contracts import read_json
    from experiments.forge.planning import resolve_idea
    from experiments.forge.promotion import validate_screening_submission
    from test_forge_studies import ready_study
    study = ready_study(checkout)
    view = read_json(checkout / "configs/forge/views/stability.json")
    view.update(id="probe", evidence_scope="research_diagnostic")
    view["assignments"] = [{"task": "t1", "qualification_tier": 1, "importance": "diagnostic", "order": 0}]
    atomic_json(checkout / "configs/forge/views/probe.json", view)
    study["scope"]["view"] = "probe"
    atomic_json(scaling.study_path(checkout, study["id"]), study)
    request = resolve_idea(checkout, "reusable", study=study["id"])
    request["candidate"]["schema_version"] = 1
    with pytest.raises(ValueError, match="research diagnostics require a ready v2"):
        validate_screening_submission(request)
