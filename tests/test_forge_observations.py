"""Bounded software checks; tiny training cannot qualify a formulation."""
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.forge import adapters, observations
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.sources import compute_profile, runtime_manifest
from experiments.forge.views import (
    grade_result,
    task_evaluation_fingerprint,
    task_execution_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actual_receipt(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    task = read_json(ROOT / "configs/forge/tasks/vector_two_broad.json")
    task["execution"]["steps"] = 2
    task["execution"]["host_definition"].update(hidden=8, layers=1, batch=4, particles=12)
    runtime = runtime_manifest()
    compute = compute_profile("cpu")
    protocol = {"id": "screening", "seed": 0}
    science = {"runtime": runtime, "compute": compute, "protocol": protocol, "seed": 0,
               "task_initializers": {task["id"]: task["execution"]["initializer"]},
               "execution": {task["id"]: task_execution_fingerprint(task)},
               "evaluation": {task["id"]: task_evaluation_fingerprint(task)}}
    job = {"task_id": task["id"], "science": science, "compatibility_key": stable_hash(science),
           "resources": {"cpu_threads": 1}}
    request = {"candidate": {"id": "software-fixture", "recipe_overrides": {"lr": .0001}},
               "candidate_revision": "a" * 64, "protocol": protocol, "tasks": {task["id"]: task},
               "source": {"digest": "b" * 64}, "runtime": runtime, "jobs": [job]}
    try:
        raw = adapters.run_task(request, job, tmp_path / "actual", "cpu")
        yield request, job, raw, {"runtime": runtime, "compute": compute}
    finally:
        torch.set_num_threads(old_threads)


def test_actual_adapter_witness_changes_neither_grade_nor_any_rng(actual_receipt):
    request, job, raw, environment = actual_receipt
    before = deepcopy(raw)
    torch_rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    observed = observations.attach_observations(request, job, raw, environment)
    witness = observed["observed_contract"]
    assert stable_hash(witness["resolved_recipe"]) == stable_hash(raw["recipe"])
    assert witness["sampling"]["scoring_weights"] == "live"
    assert witness["sampling"]["eval_output_noise"] == "clean"
    assert grade_result(request["tasks"][job["task_id"]], observed) == grade_result(
        request["tasks"][job["task_id"]], raw)
    assert raw == before
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert random.getstate() == python_rng
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[0] == np.random.get_state()[0]
    assert numpy_rng[2:] == np.random.get_state()[2:]
    witness["execution"]["steps"] = 99
    assert request["tasks"][job["task_id"]]["execution"]["steps"] == 2


@pytest.mark.parametrize("operational_limit", [False, True])
def test_runtime_automatically_emits_observation_using_actual_environment(
    actual_receipt, tmp_path, monkeypatch, operational_limit
):
    from experiments.forge import runtime
    request, job, raw, _ = actual_receipt
    request = {**request, "campaign_id": "software-fixture"}
    path = tmp_path / "request.json"
    task_id = job["task_id"]
    declared_limit = request["tasks"][task_id]["resources"]["timeout_seconds"]
    policy = {"ceiling_factor": 3, "tasks": {task_id: {"minimum_limit_seconds": declared_limit * 2}}}
    atomic_json(path, {"request": request, "job": job, "worker": {"attempt": "producer", "device": "cpu"},
                       "wallclock_policy": policy if operational_limit else "declared"})
    frozen = path.read_bytes()
    def adapter(executing_request, *args):
        expected = declared_limit * 6 if operational_limit else declared_limit
        assert executing_request["tasks"][task_id]["resources"]["timeout_seconds"] == expected
        assert task_execution_fingerprint(executing_request["tasks"][task_id]) == job["science"]["execution"][task_id]
        return deepcopy(raw)
    monkeypatch.setattr(adapters, "run_task", adapter)
    # A claim freezes its document; later default changes cannot alter it.
    monkeypatch.setenv("FORGE_WALLCLOCK_POLICY", "declared")
    rng = torch.get_rng_state().clone()
    assert runtime.execute(path) == 0
    recorded = read_json(tmp_path / "raw-result.json")
    assert recorded["observed_contract"]["compute"] == observations.execution_environment(torch, "cpu")["compute"]
    assert recorded["evidence"] == raw["evidence"]
    assert recorded["observed_contract"]["execution"] == request["tasks"][task_id]["execution"]
    assert path.read_bytes() == frozen
    assert torch.equal(rng, torch.get_rng_state())


def test_failed_environment_observation_discards_adapter_supplied_witness(actual_receipt, tmp_path, monkeypatch):
    from experiments.forge import runtime
    request, job, raw, _ = actual_receipt
    path = tmp_path / "request.json"
    atomic_json(path, {"request": {**request, "campaign_id": "fixture"}, "job": job,
                       "worker": {"attempt": "producer", "device": "cpu"}})
    raw = {**raw, "observed_contract": {"schema_version": 1, "forged": True}}
    monkeypatch.setattr(adapters, "run_task", lambda *args: deepcopy(raw))
    def unavailable(*args):
        raise ValueError("no hardware probe")
    monkeypatch.setattr(observations, "execution_environment", unavailable)
    assert runtime.execute(path) == 0
    recorded = read_json(tmp_path / "raw-result.json")
    assert "observed_contract" not in recorded
    assert recorded["observed_contract_gaps"]
    assert recorded["evidence"] == raw["evidence"]


def test_queue_automatically_exports_pointer_after_final_certificate(actual_receipt, tmp_path):
    from test_forge_queue import SLOTS, campaign
    from test_forge_queue import request as queue_request

    from experiments.forge.queue import Queue
    request, job, raw, environment = actual_receipt
    queued = queue_request(tmp_path, cap=1)
    for key in ("candidate", "candidate_revision", "protocol", "runtime", "tasks"):
        queued[key] = deepcopy(request[key])
    job = {**job, "budget_seconds": 10,
           "resources": {**job["resources"], "allow_cpu": True, "memory_mb": 1}}
    job["science"] = {**job["science"], "rng": queued["rng"]}
    job["compatibility_key"] = stable_hash(job["science"])
    queued["jobs"] = [job]
    queued["view"]["assignments"] = [{"task": job["task_id"], "qualification_tier": 1,
                                       "importance": "required", "order": 0}]
    raw = observations.attach_observations(queued, job, raw, environment)
    reports = tmp_path / "repo/reports/forge"
    # Test-only numerical grader exercises certification, not scientific quality.
    queue = Queue(tmp_path / "queue", grader=lambda *args: {"gate_status": "FAIL"}, report_root=reports)
    queue.submit(queued, campaign())
    claim = queue.claim(SLOTS)
    atomic_json(Path(claim["worker"]["directory"]) / "terminal.json", {
        "token": claim["worker"]["token"], "attempt_status": "completed", "elapsed_seconds": 1, "result": raw})
    assert queue.collect() == 1
    directory = reports / "attempts" / claim["worker"]["attempt"]
    result = read_json(directory / "result.json")
    record, = read_json(directory / "coverage.json")["records"]
    assert result["task_results"][0]["gate_status"] == record["status"] == "FAIL"
    assert record["producer"] == {"attempt_id": result["attempt_id"], "result_hash": stable_hash(result)}


@pytest.mark.parametrize("damage", ["recipe", "initializer", "seed", "sampling", "weights", "runtime",
                                   "compute", "execution", "evaluation", "missing_recipe", "missing_weights"])
def test_mismatched_or_missing_actual_facts_never_emit_witness(actual_receipt, damage):
    request, job, raw, environment = deepcopy(actual_receipt)
    if damage == "recipe":
        raw["recipe"]["lr"] *= 2
    elif damage == "initializer":
        raw["initializer"] = "supplied"
    elif damage == "seed":
        raw["rng"]["seed"] = 1
    elif damage == "sampling":
        raw["evidence"]["eval_output_noise"] = "learned_as_served"
    elif damage == "weights":
        raw["evidence"]["executed_scoring_weights"] = "ema"
    elif damage == "runtime":
        environment["runtime"]["python"] = "other"
    elif damage == "compute":
        environment["compute"]["threads"] = 2
    elif damage in {"execution", "evaluation"}:
        job["science"][damage][job["task_id"]] = "wrong"
    elif damage == "missing_recipe":
        del raw["recipe"]
    else:
        del raw["evidence"]["executed_scoring_weights"]
    observed = observations.attach_observations(request, job, raw, environment)
    assert "observed_contract" not in observed
    assert observed["observed_contract_gaps"]


def test_sampling_observer_does_not_change_training_state_or_consumed_streams(actual_receipt, tmp_path, monkeypatch):
    request, job, raw, _ = actual_receipt
    original_init = adapters._Run.__init__
    def unobserved_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.trainer.sample = self.trainer.sample.__wrapped__
    monkeypatch.setattr(adapters._Run, "__init__", unobserved_init)
    baseline = adapters.run_task(request, job, tmp_path / "baseline", "cpu")
    for field in ("live", "observations", "rng_audits"):
        assert baseline["evidence"][field] == raw["evidence"][field]
    for field in ("state_sha256", "named_stream_state_sha256"):
        assert baseline["evidence"]["provenance_checkpoint"][field] == raw["evidence"]["provenance_checkpoint"][field]


def test_swapped_served_weights_cannot_claim_live(actual_receipt, tmp_path, monkeypatch):
    request, job, _, environment = actual_receipt
    original_evaluate = adapters._Run.evaluate
    def swapped(self, function):
        # An explicit serving swap marker; the observer must inspect the owner,
        # rather than calling ema=False a live-weight observation.
        self.trainer.policy._fast = [p.detach().clone() for p in self.trainer.policy._served_parameters()]
        try:
            return original_evaluate(self, function)
        finally:
            self.trainer.policy._fast = None
    monkeypatch.setattr(adapters._Run, "evaluate", swapped)
    raw = adapters.run_task(request, job, tmp_path / "served", "cpu")
    assert raw["evidence"]["executed_scoring_weights"] == "served_average"
    observed = observations.attach_observations(request, job, raw, environment)
    assert "observed_contract" not in observed
    assert "sampling_or_weights_mismatch_or_missing" in observed["observed_contract_gaps"]


def certified(actual_receipt):
    request, job, raw, environment = actual_receipt
    raw = observations.attach_observations(request, job, raw, environment)
    # Test-only frozen numerical certificate: no claim that this tiny run
    # satisfies the real scientific task's budget/cadence gate.
    request = {**request, "requires_independent_grading": True}
    grade = {"gate_status": "FAIL"}
    result = {"attempt_id": "producer", "candidate_revision": request["candidate_revision"],
              "task_results": [{**raw, **grade, "task_id": job["task_id"],
                                "raw_status": "completed", "compatibility_key": job["compatibility_key"]}],
              "raw": {"attempt_status": "completed", "result": raw, "grading": {
                  "raw_hash": stable_hash(raw), "source_digest": request["source"]["digest"],
                  "grades": {job["task_id"]: grade}}}}
    certificate = {"result_hash": stable_hash(result), "source": request["source"], "runtime": request["runtime"]}
    resolved = {"request": request, "job": job, "worker": {"attempt": "producer"}}
    return resolved, result, certificate


def test_exact_export_satisfies_strict_consumer_and_retains_numerical_failure(actual_receipt, tmp_path):
    from ops.refresh_forge_readouts import _coverage_observation_errors
    resolved, result, certificate = certified(actual_receipt)
    directory = tmp_path / "reports/forge/attempts/producer"
    for name, value in (("request", resolved), ("result", result), ("evidence", certificate)):
        atomic_json(directory / f"{name}.json", value)
    exported = observations.coverage_export(resolved, result, certificate, family="fixture")
    assert not exported["gaps"]
    record, = exported["records"]
    assert record["status"] == "FAIL"
    assert record["producer"] == {"attempt_id": "producer", "result_hash": stable_hash(result)}
    request, job = resolved["request"], resolved["job"]
    assert _coverage_observation_errors(tmp_path, record, request, request["tasks"][job["task_id"]], job) == []


@pytest.mark.parametrize("damage", ["certificate", "grade", "raw_hash", "row_witness", "operational"])
def test_export_rejects_missing_or_conflicting_producer_facts(actual_receipt, damage):
    resolved, result, certificate = certified(actual_receipt)
    if damage == "certificate":
        certificate["result_hash"] = "wrong"
        with pytest.raises(ValueError, match="invalid producer"):
            observations.coverage_export(resolved, result, certificate)
        return
    if damage == "grade":
        result["raw"]["grading"]["grades"][resolved["job"]["task_id"]]["gate_status"] = "PASS"
    elif damage == "raw_hash":
        result["raw"]["grading"]["raw_hash"] = "wrong"
    elif damage == "row_witness":
        result["task_results"][0]["observed_contract"]["initializer"] = "supplied"
    else:
        result["task_results"][0]["gate_status"] = "BLOCKED"
    certificate["result_hash"] = stable_hash(result)
    exported = observations.coverage_export(resolved, result, certificate)
    assert exported["records"] == []
    assert exported["gaps"]
