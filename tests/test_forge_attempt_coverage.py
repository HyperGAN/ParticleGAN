"""Attempt coverage cannot change qualification or duplicate durable work."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from experiments.forge import attempt_coverage as runner
from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.family_reports import _count, score


def test_coverage_retains_qualified_numerator_and_original_missing_counts():
    tasks = [
        dict(status="PASS", execution_recorded=True),
        dict(status="UNKNOWN", execution_recorded=False,
             run_attempt=dict(status="FAIL", attempted=True)),
        dict(status="BLOCKED", execution_recorded=False,
             run_attempt=dict(status="BLOCKED", attempted=True, block_kind="incompatible")),
        dict(status="UNKNOWN", execution_recorded=False,
             run_attempt=dict(status="BLOCKED", attempted=True, block_kind="prerequisite")),
        dict(status="UNKNOWN", execution_recorded=False,
             run_attempt=dict(status="PENDING", attempted=False)),
    ]
    before = deepcopy(tasks)
    result = _count(tasks)
    assert result["passed"] == 1 and result["not_run"] == 3 and result["incompatible"] == 1
    # One attempt reads "1 diagnostic attempt", as "1 error" and "1 timeout" already do.
    assert score(result) == "1/5 (1 not run, 1 incompatible, 1 blocked, 1 diagnostic attempt)"
    tasks.append(dict(status="UNKNOWN", execution_recorded=False, run_attempt=dict(status="PASS", attempted=True)))
    assert score(_count(tasks)) == "1/6 (1 not run, 1 incompatible, 1 blocked, 2 diagnostic attempts)"
    tasks.pop()
    assert tasks == before


def test_recovery_preserves_live_workers_and_never_replays_interrupted_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "process_identity", lambda pid: "live" if pid == 10 else None)
    live = dict(status="RUNNING", directory=str(tmp_path / "live"), pid=10, process_identity="live")
    gone = dict(status="RUNNING", directory=str(tmp_path / "gone"), pid=11, process_identity="old")
    finished = dict(status="RUNNING", directory=str(tmp_path / "finished"), pid=12, process_identity="old")
    atomic_json(tmp_path / "finished/result.json", dict(status="BLOCKED", attempted=True, reason="constructor"))
    (tmp_path / "finished/run.log").write_text("actual constructor traceback")
    state = dict(jobs=dict(live=live, gone=gone, finished=finished))
    runner.recover(state)
    assert live["status"] == "RUNNING"
    assert gone["status"] == "INTERRUPTED" and gone["attempted"]
    assert finished["status"] == "BLOCKED" and finished["result_sha256"] and finished["log_sha256"]
    runner.recover(state)
    assert gone["status"] == "INTERRUPTED"


def test_scheduler_caps_four_tests_two_per_gpu_and_resumes_without_replay(tmp_path, monkeypatch):
    root, directory = tmp_path / "repo", tmp_path / "runs"
    jobs = {str(i): dict(key=str(i), family="fixture", task=f"task{i}", status="PENDING",
                         resources=dict(gpus=1, gpu_memory_mb=1), dependencies=[]) for i in range(9)}
    atomic_json(directory / "state.json", dict(jobs=jobs))
    monkeypatch.setattr(runner, "device_slots", lambda *a, **kw: [
        dict(device=gpu, slot=slot, memory_mb=100) for gpu in ("0", "1") for slot in (0, 1)])
    monkeypatch.setattr(runner, "process_identity", lambda pid: str(pid))
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    starts = []

    class Process:
        def __init__(self, command, **kwargs):
            path = Path(command[-1]).parent
            starts.append(path)
            self.pid = 100 + len(starts)
            atomic_json(path / "result.json", dict(status="BLOCKED", attempted=True, reason="fixture"))

        def poll(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    assert runner.drain(root, directory, gpus=["0", "1"])["pending"] == 0
    active = {"0": 0, "1": 0}
    for event in map(json.loads, (directory / "events.jsonl").read_text().splitlines()):
        active[event["gpu"]] += 1 if event["event"] == "started" else -1
        assert sum(active.values()) <= 4 and max(active.values()) <= 2
    assert active == {"0": 0, "1": 0} and len(starts) == 9
    runner.drain(root, directory, gpus=["0", "1"])
    assert len(starts) == 9
    report = read_json(root / runner.REPORT)
    assert report["qualification_input"] is False
    assert all(r["evidence"] == runner.REPORT.as_posix() for r in report["records"])


def test_checkpoint_parent_failure_remains_a_blocker_but_failed_gate_does_not():
    state = dict(jobs={"parent": dict(family="f", task="parent", status="FAIL")})
    job = dict(family="f", dependencies=[dict(task="parent", kind="gate")])
    assert runner.dependency_blockers(job, state) == []
    job["dependencies"][0]["kind"] = "checkpoint"
    assert runner.dependency_blockers(job, state) == ["checkpoint parent parent: FAIL"]
    job["diagnostic_continuation"] = True
    assert runner.dependency_blockers(job, state) == [
        "checkpoint parent parent: certified artifact binding unavailable to this attempt"]


def test_uninterrupted_hold_cannot_be_restarted_from_a_passing_scalar(tmp_path):
    path = tmp_path / "raw.json"
    atomic_json(path, {"evidence": {"continuity": {"mode": "uninterrupted"}}})
    parent = dict(family="f", task="ring_hold", status="PASS", origin="new_attempt", directory=str(tmp_path))
    state = dict(jobs={"parent": parent})
    job = dict(family="f", dependencies=[dict(task="ring_hold", kind="checkpoint")])
    assert runner.prerequisite_bindings(job, state) == {}
    assert "certified artifact binding unavailable" in runner.dependency_blockers(job, state)[0]


def test_research_roster_excludes_historical_seed_studies():
    from experiments.forge.research_attempts import variants
    groups = variants(Path(__file__).resolve().parents[1])
    assert len(groups) == 19
    assert all(len(g["required"]) == 29 for g in groups)
    assert len({g["scope"] for g in groups}) == 19
    assert sum(len(g["records"]) for g in groups) == 52


def test_research_backfill_fills_empty_cells_without_replacing_original_verdicts(tmp_path):
    from experiments.forge.research_tiers import _row
    atomic_json(tmp_path / runner.REPORT, {"records": [
        dict(research_scope="frozen-scope", task="missing", status="PASS", attempted=True),
        dict(research_scope="frozen-scope", task="original", status="PASS", attempted=True),
        dict(research_scope="other-scope", task="unbound", status="PASS", attempted=True)]})
    original = {"original": "FAIL"}
    text = _row(tmp_path, tmp_path / "board.md", "Research", "details.md", original,
                {"1": {"original", "missing", "unbound"}, "2": set(), "3": set()}, scope="frozen-scope")
    assert "1/3 (1 not run)" in text
    assert original == {"original": "FAIL"}


def test_missing_completion_publisher_stops_without_restart_or_replaying_work(tmp_path, monkeypatch):
    directory = tmp_path / "runs"
    atomic_json(directory / "state.json", dict(jobs={}))
    monkeypatch.setattr(runner, "device_slots", lambda *args, **kwargs: [])
    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("a missing publisher must be detected before launching a subprocess")
    monkeypatch.setattr(runner.subprocess, "run", unexpected_subprocess)
    with pytest.raises(SystemExit) as stopped:
        runner.drain(tmp_path / "repo", directory, gpus=["0", "1"],
                     completion_hook=tmp_path / "missing-publisher.py")
    assert stopped.value.code == 2
    assert read_json(directory / "state.json")["jobs"] == {}
    template = Path(__file__).resolve().parents[1] / "ops/forge-attempt-coverage.service.example"
    assert "RestartPreventExitStatus=2" in template.read_text()


def test_policy_backfill_requires_exact_scope_and_keeps_original_verdicts(tmp_path):
    from experiments.forge.remaining_attempts import policy_scope
    from experiments.forge.research_tiers import _row
    scope = policy_scope("atlas", "source", "recipe")
    atomic_json(tmp_path / runner.REPORT, {"records": [
        dict(policy_scope=scope, task="missing_policy", parent_task="missing", status="PASS", attempted=True),
        dict(policy_scope=scope, task="original_policy", parent_task="original", status="PASS", attempted=True),
        dict(policy_scope=policy_scope("atlas", "other-source", "recipe"),
             task="unbound_policy", parent_task="unbound", status="PASS", attempted=True)]})
    original = {"original": "FAIL"}
    text = _row(tmp_path, tmp_path / "board.md", "Policy", "details.md", original,
                {"1": {"original", "missing", "unbound"}, "2": set(), "3": set()}, policy_scope=scope)
    assert "1/3 (1 not run)" in text and original == {"original": "FAIL"}
    original.update(original="INCOMPLETE", unbound="BLOCKED")
    text = _row(tmp_path, tmp_path / "board.md", "Policy", "details.md", original,
                {"1": {"original", "missing", "unbound"}, "2": set(), "3": set()}, policy_scope=scope)
    assert "1/3 (1 incompatible, 1 incomplete)" in text


def test_unbound_current_task_attempt_keeps_old_request_and_refuses_bound_replacement(tmp_path):
    from experiments.forge.remaining_attempts import prepare
    from experiments.forge.views import task_evaluation_fingerprint
    root, directory = tmp_path / "repo", tmp_path / "runs"
    card = dict(id="task", adapter="fixture", execution=dict(steps=10), evaluation=dict(threshold=1),
                resources=dict(gpus=1, gpu_memory_mb=1, timeout_seconds=10), dependencies=[])
    publication = {"rows": [dict(candidate_revision="recipe", bindings=dict(source_digest="source", task_contracts={}))],
        "family_progress": {"families": [dict(id="f", current_cohort_anchor="current", cohorts=[
            dict(anchor="current", row_index=0, views=[dict(id="view")])])],
            "views": [dict(id="view", assignments=[dict(task="task", importance="required")])]}}
    original = dict(candidate=dict(id="candidate"), candidate_revision="recipe",
                    source=dict(digest="source", snapshot_path=str(tmp_path / "frozen")),
                    tasks=dict(task={**card, "evaluation": dict(threshold=0)}))
    atomic_json(root / "reports/forge/technique-inventory.json", publication)
    atomic_json(root / "configs/forge/tasks/task.json", card)
    atomic_json(directory / "requests/f.json", original)
    atomic_json(directory / "remaining-sources.json", dict(unbound_current_tasks=[dict(family="f", tasks=["task"])]))
    job, = prepare(root, directory).values()
    assert job["source_digest"] == "source" and job["candidate_revision"] == "recipe"
    assert job["evaluation_sha256"] == task_evaluation_fingerprint(card)
    assert read_json(directory / "requests/f.json") == original
    publication["rows"][0]["bindings"]["task_contracts"]["task"] = "recorded"
    atomic_json(root / "reports/forge/technique-inventory.json", publication)
    with pytest.raises(ValueError, match="cannot replace a recorded task binding"):
        prepare(root, directory)


def test_checkpoint_repair_freezes_allowed_bytes_without_mutating_original(tmp_path):
    from experiments.forge.checkpoint_recovery import repaired_source
    from experiments.forge.contracts import file_hash, stable_hash
    baseline = tmp_path / "baseline"
    state_file = baseline / "experiments/forge/state.py"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("# original checkpoint exporter\n")
    model = baseline / "particlegan/recipes.py"
    model.parent.mkdir()
    model.write_text("# unchanged original formulation\n")
    manifest = dict(schema_version=1, origin_commit="unused", files={
        "experiments/forge/state.py": file_hash(state_file), "particlegan/recipes.py": file_hash(model)})
    manifest["digest"] = stable_hash(manifest["files"])
    current = Path(__file__).resolve().parents[1] / "experiments/forge/state.py"
    frozen, repair = repaired_source(baseline, manifest, tmp_path / "runs", {"experiments/forge/state.py": file_hash(current)})
    assert file_hash(frozen / "experiments/forge/state.py") == file_hash(current)
    assert file_hash(state_file) == manifest["files"]["experiments/forge/state.py"]
    assert file_hash(frozen / "particlegan/recipes.py") == file_hash(model)
    assert repair["digest"] != manifest["digest"]
    with pytest.raises(ValueError, match="cannot modify formulation sources"):
        repaired_source(baseline, manifest, tmp_path / "runs", {"particlegan/recipes.py": file_hash(model)})


def test_prior_ownership_repair_keeps_distribution_and_actual_recipe():
    from experiments.forge.checkpoint_recovery import align_prior_contracts
    parent = dict(id="parent", execution=dict(prior=dict(kind="mog", sigma=.025, standardize=False),
                                              prior_contract="recipe_owned_v1"))
    child = dict(id="child", execution=dict(continuation_of="parent",
        prior=dict(kind="mog", sigma=.025, standardize=False, learnable=True)))
    request = dict(candidate=dict(resolved_recipe=dict(prior_update="learned")), tasks=dict(parent=parent, child=child))
    recipe = deepcopy(request["candidate"])
    align_prior_contracts(request, ["child"])
    assert request["candidate"] == recipe
    assert child["execution"]["prior"] == parent["execution"]["prior"]
    assert child["execution"]["prior_contract"] == "recipe_owned_v1"
    assert request["task_metadata_repairs"]["child"]["original_execution"] != request["task_metadata_repairs"]["child"]["repaired_execution"]
    child["execution"]["prior"]["learnable"] = True
    request["candidate"]["resolved_recipe"]["prior_update"] = "frozen"
    with pytest.raises(ValueError, match="preserve the actual learned prior"):
        align_prior_contracts(request, ["child"])


def test_selected_universe_keeps_held_and_removed_tasks_and_drops_plain_diagnostics():
    note = {"explanation": "reports/forge/test-dispositions-20261015/README.md#x"}
    views = [{"id": "counted", "assignments": [
        {"task": "kept", "importance": "required"}, {"task": "held", "importance": "held", "hold": note},
        {"task": "removed", "importance": "diagnostic", "removal": note},
        {"task": "probe", "importance": "diagnostic"}]},
        {"id": "other", "assignments": [{"task": "elsewhere", "importance": "required"}]}]
    publication = {"rows": [{"candidate_id": "one"}], "family_progress": {"views": views, "families": [
        {"id": "family", "current_cohort_anchor": "a", "cohorts": [{"anchor": "a", "row_index": 0,
                                                                    "views": [{"id": "counted"}]}]}]}}
    assert runner.selected_tasks(publication) == [("family", {"candidate_id": "one"}, ["held", "kept", "removed"])]


def test_holding_or_removing_tests_keeps_the_committed_selected_universe():
    publication = read_json(Path(__file__).resolve().parents[1] / "reports/forge/technique-inventory.json")
    counted, uncounted = deepcopy(publication), deepcopy(publication)
    for view in counted["family_progress"]["views"]:
        for assignment in view["assignments"]:
            if assignment["importance"] == "held" or "removal" in assignment:
                assignment.pop("hold", None)
                assignment.pop("removal", None)
                assignment["importance"] = "required"
    for view in uncounted["family_progress"]["views"]:
        for assignment in view["assignments"]:
            # The decided dispositions: fifteen held tests and two removed from the count.
            if assignment["importance"] != "required":
                continue
            if assignment["task"] in {"ring_hold", "clockfree_audit"}:
                assignment.update(importance="diagnostic", removal={"reason": "phase"})
            elif assignment["task"].startswith(("vector_un", "vector_aniso", "vector_overlap", "img_b", "img_i",
                                                "grid100", "rotated100", "staggered100", "gaussian1d_stability",
                                                "target_shift")):
                assignment.update(importance="held", hold={"reason": "no_global_recipe"})
    assert counted["family_progress"]["views"] != uncounted["family_progress"]["views"]
    universe = runner.selected_tasks(publication)
    # View v15 appends two precision stretch tasks to the original 50-task
    # union. The FP32 diagnostic does not enter this counted/stretch universe.
    old = read_json(Path(__file__).resolve().parents[1] /
                    "configs/forge/leaderboards/technique-inventory-full-suite-2026-10-25.json")
    view = read_json(Path(__file__).resolve().parents[1] / "configs/forge/views/discriminator_stability.json")
    expected = set(old["tests"]) | {"ring_hold"} | {a["task"] for a in view["assignments"] if a["importance"] == "stretch"}
    # View17 replaces only the ring task; historical ring1600 evidence is not a ring6400 attempt.
    expected = (expected - {"ring16_acquisition"}) | {"ring16_acquisition_6400"}
    assert universe and all(set(tasks) == expected for _, _, tasks in universe)
    assert runner.selected_tasks(counted) == runner.selected_tasks(uncounted) == universe
