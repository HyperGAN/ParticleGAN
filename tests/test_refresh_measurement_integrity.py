"""Current measurements and retained diagnostic integrity are distinct claims."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.forge import api
from experiments.forge import views
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.publication_integrity import (
    CORRECTIONS_FILES, IDENTITY_FIELDS, load_corrections,
)
from experiments.forge.sampling import validate_declaration
from ops import refresh_forge_readouts as refresh
from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = Path("reports/forge/attempt-coverage.json")


@pytest.fixture
def current_cell(tmp_path, monkeypatch):
    task = read_json(ROOT / "configs/forge/tasks/gaussian1d_smoke.json")
    recipe = get_recipe().to_dict()
    runtime = {"python": "test-python", "packages": {"torch": "test-torch"}}
    compute = {"backend": "cpu", "threads": 1, "model": "test-cpu"}
    candidate, revision, source = "candidate", "a" * 64, "b" * 64
    protocol = {"seed": 0}
    science = {"runtime": runtime, "compute": compute, "protocol": protocol, "seed": 0,
               "task_initializers": {task["id"]: task["execution"]["initializer"]}}
    job = {"task_id": task["id"], "science": science, "compatibility_key": stable_hash(science)}
    request = {"candidate": {"id": candidate}, "candidate_revision": revision,
               "source": {"digest": source}, "runtime": runtime, "protocol": protocol,
               "tasks": {task["id"]: task}, "jobs": [job]}
    sampling = validate_declaration(task, required=True)
    sampling["scoring_weights"] = task["evaluation"].get("scoring_weights", "live")
    record = {"family": "family", "candidate_id": candidate, "candidate_revision": revision,
              "source_digest": source, "task": task["id"], "attempted": True, "status": "FAIL",
              "diagnostic": False, "execution_sha256": refresh.task_execution_fingerprint(task),
              "evaluation_sha256": refresh.task_evaluation_fingerprint(task), "runtime": runtime,
              "observed_contract": {"schema_version": 1, "resolved_recipe": recipe,
                                    "runtime": runtime, "compute": compute, "sampling": sampling,
                                    "protocol": protocol, "initializer": task["execution"]["initializer"],
                                    "execution": task["execution"], "evaluation": task["evaluation"]}}
    record["key"] = stable_hash([record[field] for field in IDENTITY_FIELDS])
    monkeypatch.setattr(refresh, "current_family_candidates", lambda root: [
        {"family": "family", "candidate_id": candidate}])
    from reports.forge import regenerate_technique_inventory as inventory
    monkeypatch.setattr(inventory, "_resolve_idea_for_readout_check", lambda *args, **kwargs: request)
    monkeypatch.setattr(api, "task_formulation_context", lambda *args, **kwargs:
                        SimpleNamespace(recipe=SimpleNamespace(to_dict=lambda: recipe)))
    # This fixture tests receipt consumption, not the Gaussian scientific gate.
    # The producer grade is independently called; missing/full-budget evidence
    # and grades are varied below without executing a scientific experiment.
    monkeypatch.setattr(views, "grade_result", lambda task, raw, **kwargs:
                        {"status": "FAIL" if raw.get("evidence", {}).get("completed_steps") == 1000
                         else "INCOMPLETE"})
    producer = {"schema_version": 1, "attempt_id": "producer", "candidate_revision": revision,
                "task_results": [{"task_id": task["id"], "compatibility_key": job["compatibility_key"],
                                  "gate_status": "FAIL", "raw_status": "completed",
                                  "observed_contract": deepcopy(record["observed_contract"])}],
                "raw": {"attempt_status": "completed", "result": {"evidence": {"completed_steps": 1000},
                         "recipe": recipe, "initializer": task["execution"]["initializer"], "rng": {"seed": 0}}}}
    directory = tmp_path / "reports/forge/attempts/producer"
    atomic_json(directory / "request.json", {"request": request})
    atomic_json(directory / "result.json", producer)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(producer),
                                               "source": request["source"], "runtime": runtime})
    record["producer"] = {"attempt_id": "producer", "result_hash": stable_hash(producer)}

    def save(value):
        atomic_json(tmp_path / COVERAGE, {"summary": {"pending": 0, "running": 0}, "records": [value]})
    save(record)
    return tmp_path, record, save


def test_correctly_measured_formula_failure_is_complete_evidence(current_cell):
    root, _, _ = current_cell
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is True
    assert report["faithful_measured_cells"] == 1
    assert report["missing_by_family"] == {}


def test_faithful_current_diagnostic_measures_the_formula_without_qualification_credit(current_cell):
    root, original, save = current_cell
    save({**original, "diagnostic": True})
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is True
    assert report["qualification_reuse"] is False


def test_unmeasured_optional_precision_stays_visible_without_blocking_required_completion(current_cell):
    root, _, _ = current_cell
    from reports.forge import regenerate_technique_inventory as inventory
    request = inventory._resolve_idea_for_readout_check()
    task = deepcopy(next(iter(request["tasks"].values())))
    task["id"] = "optional-precision"
    task["execution"]["linear_precision"] = {"format": "nvfp4"}
    request["tasks"][task["id"]] = task
    request["view"] = {"assignments": [{"task": task["id"], "importance": "stretch"}]}
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is True
    assert report["optional_precision_tasks"] == ["optional-precision"]


@pytest.mark.parametrize("missing", ["resolved_recipe", "sampling", "runtime", "compute", "protocol",
                                    "initializer", "execution", "evaluation"])
def test_declared_identity_cannot_replace_missing_observed_witnesses(current_cell, missing):
    root, original, save = current_cell
    record = deepcopy(original)
    del record["observed_contract"][missing]
    save(record)
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is False
    assert report["faithful_measured_cells"] == 0
    assert report["gap_examples"]


@pytest.mark.parametrize("field,value", [("initializer", "supplied"), ("protocol", {"seed": 1})])
def test_other_initialization_or_seed_cannot_satisfy_current_contract(current_cell, field, value):
    root, original, save = current_cell
    record = deepcopy(original)
    record["observed_contract"][field] = value
    save(record)
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is False
    assert report["faithful_measured_cells"] == 0


def test_missing_or_mutated_producer_does_not_certify_displayed_failure(current_cell):
    root, original, save = current_cell
    save({key: value for key, value in original.items() if key != "producer"})
    assert refresh.check_attempt_coverage(root)["invalid_record_counts"]["missing_or_invalid_producer_receipt"] == 1
    save(original)
    result_path = root / "reports/forge/attempts/producer/result.json"
    result = read_json(result_path)
    result["task_results"][0]["gate_status"] = "PASS"
    atomic_json(result_path, result)
    assert refresh.check_attempt_coverage(root)["invalid_record_counts"]["invalid_producer_certificate"] == 1


def test_certified_display_label_cannot_replace_full_budget_producer_grade(current_cell):
    root, original, save = current_cell
    directory = root / "reports/forge/attempts/producer"
    result = read_json(directory / "result.json")
    result["raw"]["result"]["evidence"]["completed_steps"] = 500
    atomic_json(directory / "result.json", result)
    certificate = read_json(directory / "evidence.json")
    certificate["result_hash"] = stable_hash(result)
    atomic_json(directory / "evidence.json", certificate)
    record = deepcopy(original)
    record["producer"]["result_hash"] = stable_hash(result)
    save(record)
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is False
    assert report["invalid_record_counts"]["producer_grade_mismatch_or_incomplete"] == 1


def test_producer_actual_applied_recipe_cannot_be_replaced_by_a_matching_witness(current_cell):
    root, original, save = current_cell
    directory = root / "reports/forge/attempts/producer"
    result = read_json(directory / "result.json")
    result["raw"]["result"]["recipe"]["lr"] *= 2
    atomic_json(directory / "result.json", result)
    certificate = read_json(directory / "evidence.json")
    certificate["result_hash"] = stable_hash(result)
    atomic_json(directory / "evidence.json", certificate)
    record = deepcopy(original)
    record["producer"]["result_hash"] = stable_hash(result)
    save(record)
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is False
    assert report["invalid_record_counts"]["producer_applied_contract_mismatch_or_missing"] == 1


def test_frozen_independent_grading_certificate_must_bind_raw_evidence(current_cell):
    root, original, save = current_cell
    directory = root / "reports/forge/attempts/producer"
    frozen = read_json(directory / "request.json")
    frozen["request"]["requires_independent_grading"] = True
    atomic_json(directory / "request.json", frozen)
    result = read_json(directory / "result.json")
    result["raw"]["grading"] = {"source_digest": original["source_digest"], "raw_hash": "stale",
                                  "grades": {original["task"]: {"status": "FAIL"}}}
    atomic_json(directory / "result.json", result)
    certificate = read_json(directory / "evidence.json")
    certificate["result_hash"] = stable_hash(result)
    atomic_json(directory / "evidence.json", certificate)
    record = deepcopy(original)
    record["producer"]["result_hash"] = stable_hash(result)
    save(record)
    assert refresh.check_attempt_coverage(root)["invalid_record_counts"]["invalid_producer_grading_certificate"] == 1
    result["raw"]["grading"]["raw_hash"] = stable_hash(result["raw"]["result"])
    atomic_json(directory / "result.json", result)
    certificate["result_hash"] = stable_hash(result)
    atomic_json(directory / "evidence.json", certificate)
    record["producer"]["result_hash"] = stable_hash(result)
    save(record)
    assert refresh.check_attempt_coverage(root)["current"] is True


@pytest.mark.parametrize("field", ["source_digest", "candidate_revision", "execution_sha256", "evaluation_sha256"])
def test_current_completion_rejects_stale_scientific_bindings(current_cell, field):
    root, original, save = current_cell
    record = {**original, field: "f" * 64}
    record["key"] = stable_hash([record[name] for name in IDENTITY_FIELDS])
    save(record)
    report = refresh.check_attempt_coverage(root)
    assert report["current"] is False
    assert report["invalid_record_counts"][field + "_mismatch"] == 1


def test_recipe_substitution_and_ema_sampling_do_not_satisfy_live_formula(current_cell):
    root, original, save = current_cell
    for field in ("resolved_recipe", "sampling"):
        record = deepcopy(original)
        if field == "resolved_recipe":
            record["observed_contract"][field]["lr"] *= 2
        else:
            record["observed_contract"][field]["scoring_weights"] = "ema"
        save(record)
        assert refresh.check_attempt_coverage(root)["current"] is False


def test_scope_corrected_archived_pass_does_not_satisfy_measurement_completion(current_cell):
    root, _, save = current_cell
    for rel_path in CORRECTIONS_FILES:
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel_path).read_bytes())
    keys = load_corrections(ROOT)
    coverage_records = read_json(ROOT / COVERAGE)["records"]
    for rel_path in CORRECTIONS_FILES:
        reg = rel_path.as_posix()
        record = next(row for row in coverage_records
                      if row["key"] in keys and keys[row["key"]]["registry"] == reg
                      and row["status"] == "PASS")
        record = deepcopy(record)
        # Preserve exact original bytes/identity, even though the requested family
        # has a different current candidate and source.
        save(record)
        errors = refresh._coverage_observation_errors(root, record,
            {"candidate": {"id": record["candidate_id"]}, "candidate_revision": record["candidate_revision"],
             "source": {"digest": record["source_digest"]}},
            read_json(ROOT / "configs/forge/tasks" / (record["task"] + ".json")), {})
        assert "corrected_archived_diagnostic" in errors
        assert record["status"] == "PASS"


def test_published_scope_can_be_intact_while_current_measurements_are_missing(monkeypatch, tmp_path):
    strict = {"status": "NEEDS_REFRESH", "source_digest": "latest", "stale_stages": ["attempt_coverage"],
              "stages": {"queue": {"current": True}, "compiled_reports": {"current": True},
                         "attempt_coverage": {"current": False, "faithful_measured_cells": 0},
                         "retained_diagnostic_integrity": {"current": True, "scope": "retained_diagnostic_integrity"},
                         "scaling_performance": {"current": False, "published_source_current": True}}}
    monkeypatch.setattr(refresh, "check_status", lambda *args, **kwargs: strict)
    monkeypatch.setattr(refresh, "_registered_evidence_satisfies_selection", lambda root: True)
    report = refresh.check_published_status(tmp_path)
    assert report["status"] == "CURRENT"
    assert report["current_source_measurements"]["status"] == "NEEDS_REFRESH"
    assert report["stages"]["retained_diagnostic_integrity"]["current"] is True
    assert "attempt_coverage" not in report["stages"]
    assert report["qualification_reuse"] is False


def test_compiled_guard_accepts_real_blockers_but_rejects_a_changed_headline(tmp_path, monkeypatch):
    from reports.forge import regenerate_technique_inventory as inventory
    monkeypatch.setattr(refresh.knowledge, "freshness", lambda root: {"fresh": True})
    monkeypatch.setattr(refresh.history, "validate_inventory", lambda root: {"valid": True})
    monkeypatch.setattr(refresh, "current_family_candidates", lambda root: [])
    monkeypatch.setattr(refresh, "_validate_publication_registration_sync", lambda *args: True)
    monkeypatch.setattr(refresh, "_expected_selected_diagnostic_phrases", lambda *args: [])
    monkeypatch.setattr(refresh, "_attempt_coverage_markdown_synced", lambda *args: True)
    monkeypatch.setattr(refresh, "_generated_pages_synced", lambda *args: True)
    expected = (refresh.scaling_publication.START + "\n" + refresh.scaling_publication.END +
                "\nBlocked formula support: BLOCKED\n**0/0 distinct recipe/test pairs attempted (selected and research)**\n"
                "python ops/refresh_forge_readouts.py --check\npython ops/refresh_forge_readouts.py --reports-only\n")
    monkeypatch.setattr(inventory, "_current_markdown", lambda *args: expected)
    atomic_json(tmp_path / "reports/forge/compilation.json", {"pending_readout": [], "input_hashes": {}})
    perf = tmp_path / "reports/forge/scaling-performance.json"
    atomic_json(perf, {})
    atomic_json(tmp_path / "reports/forge/technique-inventory.json", {
        "family_progress": {"families": []}, "scaling_performance": {"manifest_sha256": file_hash(perf)}})
    atomic_json(tmp_path / COVERAGE, {"summary": {"attempted": 0, "total": 0}})
    page = tmp_path / "reports/forge/technique-inventory.md"
    page.write_text(expected)
    assert refresh.check_compiled_reports(tmp_path)["current"] is True
    page.write_text(expected.replace("BLOCKED", "FAIL"))
    report = refresh.check_compiled_reports(tmp_path)
    assert report["current"] is False
    assert report["current_source_tier1_synced"] is False
