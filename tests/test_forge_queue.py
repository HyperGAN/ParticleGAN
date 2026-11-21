"""Scheduler invariant tests, without spending training compute."""
import copy
import multiprocessing
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.sources import inspect_source, snapshot_source


def grade(task, raw):
    # Test-only independent grader; production uses existing benchmark evaluators.
    return {"gate_status": "PASS" if raw.get("measured") == task["target"] else "FAIL"}


def request(tmp_path, candidate="a", *, cap=3, cost=10):
    from experiments.forge.api import FormulationContext
    protocol = {"id": "screening", "seed": 0}
    rng = FormulationContext().streams.manifest()
    source = tmp_path / "worktree"
    source.mkdir(exist_ok=True)
    (source / "particlegan").mkdir(exist_ok=True)
    (source / "particlegan" / "idea.py").write_text("mechanism = 1\n")
    manifest = inspect_source(source)
    manifest["snapshot_path"] = str(snapshot_source(source, tmp_path / "queue", manifest))
    tasks = {f"t{i}": {"id": f"t{i}", "target": i, "dependencies": []} for i in (1, 2, 3)}
    return {"candidate": {"id": candidate}, "candidate_revision": "same-code", "source": manifest,
            "protocol": protocol, "rng": rng,
            "view": {"goal": "stability", "assignments": [{"task": f"t{i}", "qualification_tier": i,
                       "importance": "required", "order": i} for i in (1, 2, 3)]}, "tasks": tasks,
            "through_tier": cap, "jobs": [{"task_id": f"t{i}", "compatibility_key": stable_hash(["code", i]),
                "science": {"protocol": protocol, "seed": 0, "rng": rng},
                "budget_seconds": cost, "resources": {"allow_cpu": True, "memory_mb": 1}} for i in (1, 2, 3)]}


def campaign(name="pilot", budget=100):
    return {"id": name, "budget_seconds": budget, "candidate_budget_seconds": budget}


SLOTS = [{"device": "cpu", "slot": i, "memory_mb": 100} for i in (0, 1)]


def finish(claim, *, measured=1, status="completed", elapsed=1):
    atomic_json(Path(claim["worker"]["directory"]) / "terminal.json", {
        "token": claim["worker"]["token"], "attempt_status": status,
        "result": {"measured": measured}, "elapsed_seconds": elapsed,
    })


def test_failed_smoke_never_reserves_downstream(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(request(tmp_path), campaign())
    claim = q.claim(SLOTS)
    assert claim["job"]["task_id"] == "t1"
    assert q.claim(SLOTS) is None
    finish(claim, measured=99)
    q.collect()
    assert q.claim(SLOTS) is None
    state = q.inspect()
    assert [j["status"] for j in state["jobs"].values()].count("pending") == 2
    assert next(iter(state["submissions"].values()))["status"] == "blocked"
    assert state["campaigns"]["pilot"]["spent_seconds"] == 1


def test_through_tier_and_budget_caps(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(request(tmp_path, cap=1), campaign())
    claim = q.claim(SLOTS)
    finish(claim)
    q.collect()
    assert q.claim(SLOTS) is None
    assert next(iter(q.inspect()["submissions"].values()))["status"] == "completed"
    q.submit(request(tmp_path, cap=3), campaign("small", budget=5))
    assert q.claim(SLOTS) is None
    assert q.inspect()["campaigns"]["small"]["spent_seconds"] == 0


def test_duplicate_subscribers_share_one_job_and_one_cost(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    first = q.submit(request(tmp_path, cap=1), campaign())
    second = q.submit(request(tmp_path, "renamed", cap=1), campaign("other"))
    assert first["request"]["request_id"] != second["request"]["request_id"]
    claim = q.claim(SLOTS)
    assert q.claim(SLOTS) is None
    finish(claim)
    assert q.collect() == 1
    state = q.inspect()
    assert all(e["status"] == "completed" for e in state["submissions"].values())
    assert sum(c["spent_seconds"] for c in state["campaigns"].values()) == 1
    assert len(state["charges"]) == 1
    assert q.collect() == 0
    assert q.inspect() == state


def test_required_infrastructure_error_is_incomplete_not_fail(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(request(tmp_path), campaign())
    claim = q.claim(SLOTS)
    finish(claim, status="error")
    q.collect()
    assert q.claim(SLOTS) is None
    job = q.inspect()["jobs"][claim["job"]["compatibility_key"]]
    assert job["result"]["task_results"][0]["gate_status"] == "INCOMPLETE"
    q.retry(claim["job"]["compatibility_key"], reason="repaired environment")
    assert q.claim(SLOTS) is not None


def test_scientific_failure_cannot_be_retried(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(request(tmp_path), campaign())
    claim = q.claim(SLOTS)
    finish(claim, measured=99)
    q.collect()
    with pytest.raises(ValueError, match="scientific failures"):
        q.retry(claim["job"]["compatibility_key"], reason="try again")


def test_task_placement_does_not_change_job_identity(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    req = request(tmp_path)
    first = q.submit(req, campaign())
    moved = copy.deepcopy(req)
    moved["view"]["assignments"][1]["qualification_tier"] = 1
    q.submit(moved, campaign("retier"))
    state = q.inspect()
    assert len(state["jobs"]) == 3
    assert len(state["submissions"]) == 2
    assert first["request"]["view"]["assignments"][1]["qualification_tier"] == 2
    assert all(j["status"] == "pending" for j in state["jobs"].values())


def test_snapshot_captures_edits_and_detects_tampering(tmp_path):
    req = request(tmp_path)
    snapshot = Path(req["source"]["snapshot_path"])
    (tmp_path / "worktree" / "particlegan" / "idea.py").write_text("mechanism = 2\n")
    assert (snapshot / "particlegan" / "idea.py").read_text() == "mechanism = 1\n"
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(req, campaign())
    (snapshot / "particlegan" / "idea.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="snapshot was changed"):
        q.claim(SLOTS)


def test_diagnostic_failure_does_not_veto(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    req = request(tmp_path)
    req["view"]["assignments"][0]["importance"] = "diagnostic"
    q.submit(req, campaign())
    claim = q.claim(SLOTS)
    finish(claim, measured=99)
    q.collect()
    assert q.claim(SLOTS)["job"]["task_id"] == "t2"


def submit_process(root, req):
    Queue(root).submit(req, campaign())


def test_concurrent_submissions_are_idempotent(tmp_path):
    req = request(tmp_path)
    context = multiprocessing.get_context("spawn")
    children = [context.Process(target=submit_process, args=(tmp_path / "queue", req)) for _ in range(6)]
    for child in children:
        child.start()
    for child in children:
        child.join(10)
        assert child.exitcode == 0
    state = Queue(tmp_path / "queue").inspect()
    assert len(state["submissions"]) == 1
    assert all(len(j["subscribers"]) == 1 for j in state["jobs"].values())


def test_budget_counts_outstanding_reservations(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    req = request(tmp_path, cost=10)
    q.submit(req, campaign(budget=15))
    second = copy.deepcopy(req)
    second["candidate_revision"] = "different-code"
    second["candidate"]["id"] = "different"
    for job in second["jobs"]:
        job["compatibility_key"] = stable_hash(["different-code", job["task_id"]])
    q.submit(second, campaign(budget=15))
    assert q.claim(SLOTS)
    assert q.claim(SLOTS) is None
    assert q.inspect()["campaigns"]["pilot"]["reserved_seconds"] == 10


def test_immutable_campaign_rejected(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    req = request(tmp_path)
    q.submit(req, campaign())
    with pytest.raises(ValueError, match="immutable"):
        q.submit(req, campaign(budget=999))


def test_tasks_within_smoke_fail_serially_before_claiming_later_screen(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    req = request(tmp_path)
    for i, assignment in enumerate(req["view"]["assignments"]):
        assignment.update(qualification_tier=1, order=i)
    q.submit(req, campaign())
    claim = q.claim(SLOTS)
    assert claim["job"]["task_id"] == "t1"
    assert q.claim(SLOTS) is None
    finish(claim, measured=99)
    q.collect()
    assert q.claim(SLOTS) is None
    assert sum(len(j["attempts"]) for j in q.inspect()["jobs"].values()) == 1


def test_retry_cannot_move_already_spent_cost_to_another_campaign(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    first = q.submit(request(tmp_path, cap=1), campaign("first"))
    claim = q.claim(SLOTS)
    finish(claim, status="error", elapsed=5)
    q.collect()
    q.cancel(first["request"]["request_id"])
    second = q.submit(request(tmp_path, cap=1), campaign("second"))
    q.retry(claim["job"]["compatibility_key"], reason="repaired runtime")
    retry = q.claim(SLOTS)
    assert retry["request"]["campaign_id"] == "second"
    finish(retry, elapsed=2)
    q.collect()
    state = q.inspect()
    assert state["campaigns"]["first"]["spent_seconds"] == 5
    assert state["campaigns"]["second"]["spent_seconds"] == 2
    assert [c["owner"]["campaign"] for c in state["charges"]] == ["first", "second"]
    assert q._available_budget(state, first["request"], 96)[0] is False
    assert q._available_budget(state, second["request"], 96)[0] is True


def test_completed_tier_one_subscriber_does_not_authorize_tier_two(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    owner = q.submit(request(tmp_path, cap=3), campaign())
    screen = q.submit(request(tmp_path, cap=1), campaign())
    claim = q.claim(SLOTS)
    finish(claim)
    q.collect()
    quality = q.claim(SLOTS)
    assert quality["job"]["task_id"] == "t2"
    assert screen["request"]["request_id"] not in q.inspect()["jobs"][quality["job"]["compatibility_key"]]["subscribers"]
    q.cancel(owner["request"]["request_id"])
    assert (Path(quality["worker"]["directory"]) / "cancel.json").exists()


def test_collector_recovers_torn_tail_and_replays_missing_campaign_copy(tmp_path):
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(request(tmp_path), campaign())
    q.flush_events()
    central = q.root / "events.jsonl"
    original = central.read_bytes()
    event = __import__("json").loads(original)
    event.pop("event_id")
    (q.root / "pilot/progress.jsonl").unlink()
    with central.open("ab") as stream:
        stream.write(b'{"event_id":"torn')
    with q.state() as state:
        state["events"].append(event)
    q.flush_events()
    assert central.read_bytes() == original
    assert (q.root / "pilot/progress.jsonl").read_bytes() == original
    assert list(q.root.glob("events.jsonl.partial-*"))


def test_retry_lineage_is_frozen_before_launch_and_preserved_in_durable_evidence(tmp_path):
    reports = tmp_path / "repo/reports/forge"
    q = Queue(tmp_path / "queue", grader=grade, report_root=reports)
    q.submit(request(tmp_path, cap=1), campaign())
    first = q.claim(SLOTS)
    finish(first, status="error", elapsed=3)
    q.collect()
    first_result = read_json(reports / "attempts" / first["worker"]["attempt"] / "result.json")
    q.retry(first["job"]["compatibility_key"], reason="fixed worker setup")
    second = q.claim(SLOTS)
    assert second["retry_of"]["attempt_id"] == first_result["attempt_id"]
    assert second["retry_of"]["result_hash"] == stable_hash(first_result)
    assert second["retry_of"]["reason"] == "fixed worker setup"
    finish(second, elapsed=2)
    q.collect()
    directory = reports / "attempts" / second["worker"]["attempt"]
    result = read_json(directory / "result.json")
    assert result["retry_of"] == read_json(directory / "request.json")["retry_of"]
    assert read_json(directory / "evidence.json")["result_hash"] == stable_hash(result)
    assert q.inspect()["campaigns"]["pilot"]["spent_seconds"] == 5
    from experiments.forge.knowledge import _attempts
    attempts, issues = _attempts(tmp_path / "repo")
    assert not issues
    old = next(a for a in attempts if a["attempt_id"] == first_result["attempt_id"])
    assert old["superseded_by"] == result["attempt_id"]


def test_ring_group_job_is_scheduled_through_a_removed_phase(tmp_path):
    # ring_hold leaves the count as a phase of ring_extension; both stay in one uninterrupted job.
    from experiments.forge.views import load_tasks
    declared = load_tasks(Path(__file__).resolve().parents[1])
    tasks = {name: {"id": name, "dependencies": declared[name]["dependencies"]}
             for name in ("mode_hold", "ring_hold", "ring_extension")}
    assert declared["ring_hold"]["execution"]["execution_group"] == declared["ring_extension"]["execution"]["execution_group"]
    assignments = [{"task": "mode_hold", "qualification_tier": 2, "importance": "required", "order": 0},
                   {"task": "ring_hold", "qualification_tier": 3, "importance": "diagnostic", "order": 0,
                    "removal": {"reason": "phase", "counted_by": "ring_extension"}},
                   {"task": "ring_extension", "qualification_tier": 3, "importance": "required", "order": 1}]
    jobs = [{"task_id": "mode_hold", "task_ids": ["mode_hold"], "compatibility_key": "mode"},
            {"task_id": "ring_hold", "task_ids": ["ring_hold", "ring_extension"], "execution_group": "ring_endurance",
             "compatibility_key": "ring"}]

    def state(mode_hold, ring=None):
        return {"jobs": {"mode": {"status": "terminal", "result": {"task_results": [
                    {"task_id": "mode_hold", "gate_status": mode_hold}]}},
                "ring": {"status": "terminal" if ring else "pending",
                         "result": {"task_results": [{"task_id": name, "gate_status": ring}
                                                     for name in ("ring_hold", "ring_extension")]} if ring else None}}}

    q = Queue(tmp_path / "queue", grader=grade)
    for mode, eligible in (("fail_fast", q._eligible), ("complete_current_tier", q._eligible_complete_tier)):
        submission = {"request": {"view": {"assignments": assignments}, "tasks": tasks, "jobs": jobs, "through_tier": 3,
                                  "execution_policy": {"schema_version": 1, "mode": mode}}}
        assert eligible(state("PASS"), submission) == (["ring"], None, False)
        # The counted verdict vetoes the tier once the group ran; a failed gate never schedules it.
        jobs_ran, reason, running = q._eligible(state("PASS", "FAIL"), submission)
        assert jobs_ran == [] and "ring_extension: FAIL" in reason and not running
        jobs_ran, reason, _ = q._eligible(state("FAIL"), submission)
        assert jobs_ran == [] and "mode_hold: FAIL" in reason
    assert not (tmp_path / "queue").exists()


def test_queue_state_context_skips_redundant_disk_write_when_unchanged(tmp_path, monkeypatch):
    from experiments.forge import queue as queue_mod
    q = Queue(tmp_path)
    with q.state() as state:
        state["campaigns"]["c1"] = {"definition": {"id": "c1"}, "spent_seconds": 0.0, "reserved_seconds": 0.0, "paused": False}
    writes = []
    orig_atomic = queue_mod.atomic_json
    monkeypatch.setattr(queue_mod, "atomic_json", lambda path, data: (writes.append(path), orig_atomic(path, data)))
    with q.state() as state:
        _ = state["campaigns"]["c1"]
    assert writes == []
