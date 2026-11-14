"""Reporting controls: counts, evidence identity, criteria and navigable pages."""
from copy import deepcopy
from pathlib import Path
import re

import pytest

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.family_reports import (build_progress, certified_retry_successors,
                                              full_suite_outcomes, full_suite_tasks,
                                              generated_pages, render_leaderboard, score)
from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint


@pytest.fixture
def report(tmp_path):
    contracts = {}
    bindings = {}
    for name in ("shared", "failed", "missing", "held", "diagnostic"):
        task = {"schema_version": 1, "id": name, "adapter": "fixture",
                "execution": {"steps": 24}, "evaluation": {"kind": "transfer_sustained",
                "thresholds": [["score", ">=", .5]], "observations": 24, "minimum_stable_checks": 5},
                "resources": {"timeout_seconds": 60}, "requires_capabilities": [], "dependencies": []}
        atomic_json(tmp_path / f"configs/forge/tasks/{name}.json", task)
        contract = {"execution_sha256": task_execution_fingerprint(task),
                    "evaluation_sha256": task_evaluation_fingerprint(task), "timeout_seconds": 60}
        bindings[name] = stable_hash(contract)
        contracts[bindings[name]] = contract

    def view(name, assignments, **extra):
        definition = {"id": name, "revision": 1, "assignments": [
            {"task": task, "qualification_tier": tier, "importance": role, "order": order}
            for order, (task, tier, role) in enumerate(assignments)],
            "calibration": {"status": "provisional"}, **extra}
        atomic_json(tmp_path / f"configs/forge/views/{name}.json", definition)
    view("alpha", [("shared", 1, "required"), ("failed", 2, "required"),
                   ("missing", 3, "required"), ("diagnostic", 2, "diagnostic")])
    view("beta", [("shared", 1, "required"), ("held", 2, "required")])
    view("diagnostics", [("diagnostic", 1, "diagnostic")], evidence_scope="research_diagnostic")
    selected = {"candidate_id": "k3p-config", "candidate_revision": "revision-one", "trainer_family": "k3p",
                "technique": "K3P", "qualified_tier": 0, "attempt_ids": ["attempt-one"],
                "bindings": {"source_digest": "source-one", "task_contracts": bindings},
                "runtime_cohort": {"execution_backend": "cuda"},
                "tasks": [{"task_id": name, "status": status} for name, status in
                          (("shared", "PASS"), ("failed", "FAIL"), ("missing", "UNKNOWN"), ("held", "PASS"))]}
    summary = {"candidate_id": selected["candidate_id"], "candidate_revision": selected["candidate_revision"],
               "provenance": {"source_digest": "source-one"}, "task_results": [
                   {"task_id": "failed", "reason": "score below .5", "metrics": {"score": .25},
                    "evaluator_summary": {"metric_checks": {"score": {"value": .25, "op": ">=",
                        "threshold": .5, "status": "FAIL"}}}}]}
    atomic_json(tmp_path / "reports/forge/technique-receipts/attempt-one.json", summary)
    publication = {"view": "alpha", "view_revision": 1, "rows": [selected], "task_contracts": contracts,
                   "provenance": {"input_digest": "fixture"}}
    return tmp_path, publication


def generate(report):
    root, publication = report
    before = deepcopy(publication["rows"])
    publication["family_progress"] = build_progress(root, publication)
    assert publication["rows"] == before
    return publication["family_progress"]["families"][0]["cohorts"][0]


@pytest.mark.parametrize("source,attached,unrecorded", [
    ("source-one", True, False), ("different-source", False, False), ("source-one", True, True)])
def test_run_attempt_display_requires_exact_source_recipe_and_task_binding(report, source, attached, unrecorded):
    root, publication = report
    selected = publication["rows"][0]
    contract = publication["task_contracts"][selected["bindings"]["task_contracts"]["missing"]]
    if unrecorded:
        from experiments.forge.family_reports import COVERAGE_PREFLIGHT
        selected["bindings"]["task_contracts"].pop("missing")
        atomic_json(root / COVERAGE_PREFLIGHT, {
            "scope": "selected_formulation_task_preflight_only", "qualification_input": False,
            "qualification_reuse": False, "families": [{"candidate_id": selected["candidate_id"],
            "candidate_revision": selected["candidate_revision"], "source_digest": "source-one",
            "recipe_sha256": None, "tasks": {"missing": {**contract, "blockers": []}}}]})
    key = [source, selected["candidate_revision"], "missing", contract["execution_sha256"],
           contract["evaluation_sha256"]]
    atomic_json(root / "reports/forge/attempt-coverage.json", {
        "schema": "forge_attempt_coverage_v1", "qualification_input": False, "qualification_reuse": False,
        "summary": {"total": 1, "attempted": 1, "pending": 0, "running": 0},
        "records": [{"key": stable_hash(key), "source_digest": source,
                     "candidate_revision": selected["candidate_revision"], "task": "missing",
                     "execution_sha256": key[3], "evaluation_sha256": key[4],
                     "status": "FAIL", "attempted": True, "evidence": "reports/forge/attempt-coverage.json"}]})
    before = deepcopy(publication["rows"])
    cohort = generate(report)
    assert ("run_attempt" in cohort["tasks"]["missing"]) is attached
    assert cohort["tasks"]["missing"]["status"] == "UNKNOWN"
    assert publication["rows"] == before


def test_optimizer_alias_selects_whole_best_configuration_and_preserves_evidence(report):
    root, publication = report
    variant = deepcopy(publication["rows"][0])
    variant.update(candidate_id="k3p-optimizer-config", trainer_family="k3p-optimizer", attempt_ids=[])
    variant["bindings"]["source_digest"] = "source-two"
    variant["bindings"]["recipe_sha256"] = "recipe-two"
    variant["tasks"] = [{"task_id": name, "status": status} for name, status in
                        (("shared", "PASS"), ("failed", "PASS"), ("missing", "PASS"), ("held", "FAIL"))]
    publication["rows"].append(variant)
    publication["recipe_contracts"] = {"recipe-two": {
        "optimizer_family": "dualnorm", "optimizer_momentum": 0, "lr": .012,
        "d_lr_mult": 1.5, "prior_lr_mult": 2.5, "reg_arm": "b_cap"}}
    atomic_json(root / "configs/forge/trainer-families.json", {"families": [
        {"id": "k3p"}, {"id": "k3p-optimizer", "reporting_family": "k3p"}]})
    original = deepcopy(publication["rows"])
    cohort = generate(report)
    progress = publication["family_progress"]
    assert len(progress["families"]) == 1
    assert cohort["row_index"] == 1 and score(cohort["total"]) == "4/5"
    assert cohort["tasks"]["held"]["status"] == "FAIL"  # Cannot borrow the baseline PASS.
    assert publication["rows"] == original
    assert len(progress["configuration_families"]) == 2
    page = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert "Best recorded configuration" in page
    assert "optimizer_family=dualnorm" in page and "optimizer_momentum=0" in page
    assert "critic rate: **0.018**" in page and "prior rate: **0.03**" in page
    assert "source-two" in page and "source-one" in page


def test_optimizer_alias_keeps_exact_runtime_groups_and_stable_ties(report):
    root, publication = report
    variant = deepcopy(publication["rows"][0])
    variant.update(candidate_id="variant", trainer_family="optimizer", attempt_ids=[])
    publication["rows"].append(variant)
    atomic_json(root / "configs/forge/trainer-families.json", {"families": [
        {"id": "k3p"}, {"id": "optimizer", "reporting_family": "k3p"}]})
    assert generate(report)["row_index"] == 0  # Existing solution wins a complete tie.
    variant["runtime_cohort"] = {"execution_backend": "cpu"}
    generate(report)
    cohorts = publication["family_progress"]["families"][0]["cohorts"]
    assert {item["backend"]: item["row_index"] for item in cohorts} == {"cuda": 0, "cpu": 1}


def test_explicit_current_optimizer_wins_display_without_borrowing_or_hiding_history(report):
    from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin
    root, publication = report
    variant = deepcopy(publication["rows"][0])
    variant.update(candidate_id="selected-optimizer", trainer_family="optimizer", attempt_ids=[])
    variant["bindings"]["source_digest"] = "selected-source"
    variant["tasks"] = [{"task_id": name, "status": "FAIL"} for name in ("shared", "failed", "missing", "held")]
    publication["rows"].append(variant)
    old = deepcopy(variant)
    old["runtime_cohort"]["model"] = "older-runtime"
    old["bindings"]["source_digest"] = "archived-source"
    publication["historical_family_rows"] = [old]
    atomic_json(root / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [
        {"id": "k3p", "label": "K3P", "candidates": ["k3p-config"],
         "canonical_candidate": "k3p-config", "current_configuration_family": "optimizer"},
        {"id": "optimizer", "label": "Optimizer", "candidates": ["selected-optimizer"],
         "canonical_candidate": "selected-optimizer", "reporting_family": "k3p"}]})
    pin = family_row_pin(variant, selection_kind="historical_incumbent", reason="Explicit current optimizer.")
    atomic_json(root / CURRENT_SELECTION, {"schema_version": 1, "scope": "whole_candidate_family_current",
        "default_adoption": False, "view": "alpha", "policy_fingerprint": "fixture", "selections": [pin]})
    original = deepcopy(publication["rows"] + publication["historical_family_rows"])
    current = generate(report)
    family = publication["family_progress"]["families"][0]
    assert current["row_index"] == 1 and score(current["total"]) == "0/5"
    assert len(family["cohorts"]) == 2
    assert publication["rows"] + publication["historical_family_rows"] == original
    page = root / "reports/forge/technique-inventory.md"
    text = render_leaderboard(root, publication, page)
    assert sum(line.startswith("| **[") for line in text.splitlines()) == 1
    assert "older-runtime" not in text
    detail = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert "Current benchmark configuration" in detail and "Archived runtime cohort" in detail
    assert "selected-source" in detail and "archived-source" in detail
    pin["scientific_row_sha256"] = "different-row"
    atomic_json(root / CURRENT_SELECTION, {"schema_version": 1, "scope": "whole_candidate_family_current",
        "default_adoption": False, "view": "alpha", "policy_fingerprint": "fixture", "selections": [pin]})
    with pytest.raises(ValueError, match="exactly one verified scientific row"):
        build_progress(root, publication)


def test_hidden_inventory_entry_preserves_numerical_rows_without_navigation(report):
    root, publication = report
    hidden = deepcopy(publication["rows"][0])
    hidden.update(candidate_id="legacy", trainer_family="legacy", technique="Legacy", attempt_ids=[])
    publication["rows"].append(hidden)
    atomic_json(root / "configs/forge/trainer-families.json", {"families": [
        {"id": "k3p"}, {"id": "legacy", "inventory_visible": False}]})
    original = deepcopy(publication["rows"])
    generate(report)
    assert publication["rows"] == original
    assert len(publication["family_progress"]["families"]) == 1
    assert "Legacy" not in render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert root / "reports/forge/families/legacy.md" not in generated_pages(root, publication)


def test_shared_tasks_count_per_view_and_diagnostics_never_fill_required_totals(report):
    cohort = generate(report)
    assert {tier: score(count) for tier, count in cohort["tiers"].items()} == {
        "1": "2/2", "2": "1/2", "3": "0/1 (1 not run)"}
    assert score(cohort["total"]) == "3/5 (1 not run)"
    assert [score(view["total"]) for view in cohort["views"]] == ["1/3 (1 not run)", "2/2"]
    assert len(cohort["tasks"]) == 5  # Four unique requirements plus a diagnostic.
    assert cohort["tasks"]["diagnostic"]["status"] == "UNKNOWN"


def test_unrun_and_incompatible_counts_are_disjoint_and_preserve_executed_failures(report):
    root, publication = report
    publication["rows"][0]["tasks"][3].update(status="BLOCKED", reason="required capability unavailable")
    cohort = generate(report)
    assert cohort["total"]["counts"] == {"BLOCKED": 1, "FAIL": 1, "PASS": 2, "UNKNOWN": 1}
    assert cohort["total"]["not_run"] == cohort["total"]["incompatible"] == 1
    assert score(cohort["total"]) == "2/5 (1 not run, 1 incompatible)"
    assert score({"passed": 10, "total": 21, "not_run": 3, "incompatible": 6}) == "10/21 (3 not run, 6 incompatible)"
    assert cohort["tasks"]["failed"]["execution_recorded"] is True
    assert cohort["tasks"]["held"]["execution_recorded"] is False


def test_completed_failure_has_no_marker_and_new_result_refreshes_every_total(report):
    root, publication = report
    cohort = generate(report)
    assert score(cohort["views"][0]["tiers"]["2"]) == "0/1"
    publication["rows"][0]["tasks"][2]["status"] = "PASS"
    updated = generate(report)
    assert score(updated["total"]) == "4/5"
    assert score(updated["views"][0]["total"]) == "2/3"


def test_unrun_family_defaults_to_zero_with_full_denominators(report):
    root, publication = report
    selected = publication["rows"][0]
    selected["tasks"] = []
    selected["attempt_ids"] = []
    cohort = generate(report)
    assert score(cohort["total"]) == "0/5 (5 not run)"
    assert [score(cohort["tiers"][tier]) for tier in ("1", "2", "3")] == ["0/2 (2 not run)", "0/2 (2 not run)", "0/1 (1 not run)"]


def test_changed_gate_preserves_recorded_failure_without_unrun_marker(report):
    root, publication = report
    path = root / "configs/forge/tasks/failed.json"
    declaration = read_json(path)
    declaration["evaluation"]["thresholds"][0][2] = .75
    atomic_json(path, declaration)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["status"] == "FAIL"
    assert cohort["tasks"]["failed"]["current_contract"] == "CHANGED"
    assert cohort["tasks"]["failed"]["changed_contract_fields"] == ["evaluation (gates or sampling law)"]
    assert "Test definition changed since this run: evaluation" in cohort["tasks"]["failed"]["coverage_reason"]
    assert cohort["tasks"]["failed"]["execution_recorded"] is True
    assert score(cohort["views"][0]["tiers"]["2"]) == "0/1"
    text = next(iter(generated_pages(root, publication).values()))
    assert "| score | 0.25 | >= 0.5 | FAIL |" in text
    assert "| score | >= 0.75 |" in text
    assert "All 24 declared observations" in text and "5 consecutive passing terminal" in text


@pytest.mark.parametrize("status", ["PASS", "FAIL"])
def test_missing_recorded_definition_does_not_hide_an_executed_result(report, status):
    root, publication = report
    selected = publication["rows"][0]
    selected["tasks"][1]["status"] = status
    selected["bindings"]["task_contracts"].pop("failed")
    cohort = generate(report)
    result = cohort["tasks"]["failed"]
    assert result["current_contract"] == "unbound"
    assert result["execution_recorded"] is True
    assert score(cohort["views"][0]["tiers"]["2"]) == ("1/1" if status == "PASS" else "0/1")
    assert "Recorded test definition unavailable" in result["coverage_reason"]


@pytest.mark.parametrize("status,raw_status", [("ERROR", "error"), ("INVALID", "completed"),
                                               ("INCOMPLETE", "timeout"), ("INCOMPLETE", "cancelled")])
def test_attempted_errors_show_cause_attempt_and_cost_without_pass_credit(report, status, raw_status):
    root, publication = report
    publication["rows"][0]["tasks"][1]["status"] = status
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    receipt["task_results"][0].update(gate_status=status, raw_status=raw_status,
        reason="specific failure: missing evaluator certificate" if status == "INVALID" else "specific worker failure",
        cost={"execution_seconds": 12.5, "wall_seconds": 0, "charged_task": "shared"})
    atomic_json(path, receipt)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["execution_recorded"] is True
    assert cohort["tasks"]["failed"]["attempt_id"] == "attempt-one"
    assert score(cohort["views"][0]["tiers"]["2"]) == "0/1"
    assert cohort["views"][0]["tiers"]["2"]["counts"] == {status: 1}
    text = next(iter(generated_pages(root, publication).values()))
    expected_reason = "specific failure: missing evaluator certificate" if status == "INVALID" else "specific worker failure"
    assert expected_reason in text
    assert "Attempt: `attempt-one`; raw outcome: " + raw_status in text
    assert "Recorded execution seconds: 12.5; charged wall seconds: 0" in text
    assert "charged once to shared" in text
    assert "Compact metrics and receipt provenance" in text


@pytest.mark.parametrize("status", ["UNKNOWN", "NOT_RUN", "BLOCKED", "INVALID", "INCOMPLETE", "ERROR"])
def test_no_task_execution_evidence_retains_marker_even_when_cohort_has_an_attempt(report, status):
    root, publication = report
    publication["rows"][0]["tasks"][2]["status"] = status
    cohort = generate(report)
    assert cohort["tasks"]["missing"]["execution_recorded"] is False
    assert cohort["tasks"]["missing"]["attempt_id"] is None
    assert score(cohort["views"][0]["tiers"]["3"]) == ("0/1 (1 incompatible)" if status == "BLOCKED" else "0/1 (1 not run)")


def test_worker_launch_failure_with_zero_execution_retains_unrun_marker(report):
    root, publication = report
    publication["rows"][0]["tasks"][1]["status"] = "INCOMPLETE"
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    receipt["task_results"][0].update(raw_status="error", reason="worker launch failed",
                                     cost={"execution_seconds": 0, "wall_seconds": 0})
    atomic_json(path, receipt)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["execution_recorded"] is False
    assert score(cohort["views"][0]["tiers"]["2"]) == "0/1 (1 not run)"
    assert "worker launch failed" in next(iter(generated_pages(root, publication).values()))


@pytest.mark.parametrize("catalogued", [True, False])
def test_structured_setup_error_renders_without_changing_recorded_grade(report, catalogued):
    root, publication = report
    task = publication["rows"][0]["tasks"][1]
    task["status"] = "INCOMPLETE"
    reasons = [{"type": "ValueError", "message": "dualnorm supports matrix weights and vector/scalar biases only"}]
    if catalogued:
        task["reasons_sha256"] = stable_hash(reasons)
        publication["status_reasons"] = {stable_hash(reasons): reasons}
    else:
        task["reasons"] = reasons
    before = deepcopy(publication)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["status"] == "INCOMPLETE"
    assert "dualnorm supports matrix weights" in next(iter(generated_pages(root, publication).values()))
    assert publication["rows"] == before["rows"]
    assert publication.get("status_reasons") == before.get("status_reasons")


@pytest.mark.parametrize("field", ["candidate_id", "candidate_revision", "source_digest"])
def test_receipt_cannot_supply_metrics_from_another_configuration_or_source(report, field):
    root, publication = report
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    target = receipt["provenance"] if field == "source_digest" else receipt
    target[field] = "another-cohort"
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="selected configuration/source"):
        build_progress(root, publication)


def _compact_repair(report):
    root, publication = report
    parent = read_json(root / "reports/forge/technique-receipts/attempt-one.json")
    parent.update(attempt_id="attempt-one", attempt_status="error", certificate_validated=True,
                  qualification_input=False, qualification_reuse=False, runtime={"python": "3.12"})
    parent["provenance"].update(canonical_result_hash=stable_hash({"interrupted": True}), source_origin_commit="commit-one")
    parent["task_results"][0].update(compatibility_key="one-complete-task-law", gate_status="INCOMPLETE",
                                     raw_status="error", reason="worker stopped", cost={"wall_seconds": 12.5})
    child = deepcopy(parent)
    child.update(attempt_id="attempt-retry", attempt_status="completed",
                 retry_of={"attempt_id": "attempt-one", "result_hash": parent["provenance"]["canonical_result_hash"],
                           "reason": "Explicit environment repair", "authorized_at": "2026-10-05T19:35:12+00:00"})
    child["provenance"]["canonical_result_hash"] = stable_hash({"repaired": True})
    result = child["task_results"][0]
    result.update(gate_status="PASS", raw_status="completed", reason="complete repaired run",
                  metrics={"score": .9}, cost={"wall_seconds": 7.25})
    result["evaluator_summary"]["metric_checks"]["score"].update(value=.9, status="PASS")
    return {"attempt-one": parent, "attempt-retry": child}


@pytest.mark.parametrize("reverse", [False, True])
def test_certified_repair_uses_successor_metrics_and_preserves_original_history_without_raw_receipts(report, reverse):
    root, publication = report
    summaries = _compact_repair(report)
    for attempt, summary in summaries.items():
        atomic_json(root / f"reports/forge/technique-receipts/{attempt}.json", summary)
    selected = publication["rows"][0]
    selected["attempt_ids"] = list(reversed(summaries)) if reverse else list(summaries)
    selected["tasks"][1]["status"] = "PASS"
    selected["cost"] = {"wall_seconds": 19.75}
    original = deepcopy(publication["rows"])
    protected = {path: path.read_bytes() for path in (root / "reports/forge/technique-receipts").glob("*.json")}
    assert not (root / "reports/forge/attempts").exists()
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["status"] == "PASS"
    assert cohort["tasks"]["failed"]["attempt_id"] == "attempt-retry"
    assert cohort["tasks"]["failed"]["metrics"]["metrics"] == {"score": .9}
    assert cohort["retry_history"][0]["attempt_id"] == "attempt-one"
    assert cohort["retry_history"][0]["superseded_by"] == "attempt-retry"
    assert cohort["retry_history"][0]["original_task_costs"] == {"failed": {"wall_seconds": 12.5}}
    assert score(cohort["total"]) == "4/5 (1 not run)"
    text = next(iter(generated_pages(root, publication).values()))
    assert "Certified execution repair" in text and "attempt-one.json" in text and "attempt-retry.json" in text
    assert "original outcome and cost remain recorded" in text
    assert publication["rows"] == original
    assert all(path.read_bytes() == content for path, content in protected.items())


@pytest.mark.parametrize("tamper", ["hash", "parent_missing", "authorization", "reason", "certificate", "qualification",
                                    "parent_completed", "parent_fail", "runtime", "source", "origin", "candidate", "revision",
                                    "task_key", "duplicate_task", "extra_task", "identity"])
def test_compact_repair_rejects_unproven_or_changed_lineage(report, tamper):
    summaries = _compact_repair(report)
    parent, child = summaries["attempt-one"], summaries["attempt-retry"]
    if tamper == "hash":
        child["retry_of"]["result_hash"] = "wrong-hash"
    elif tamper == "parent_missing":
        del summaries["attempt-one"]
    elif tamper in {"authorization", "reason"}:
        child["retry_of"]["authorized_at" if tamper == "authorization" else "reason"] = ""
    elif tamper == "certificate":
        parent["certificate_validated"] = False
    elif tamper == "qualification":
        child["qualification_input"] = True
    elif tamper == "parent_completed":
        parent["attempt_status"] = "completed"
    elif tamper == "parent_fail":
        parent["task_results"][0]["gate_status"] = "FAIL"
    elif tamper == "runtime":
        child["runtime"]["python"] = "other"
    elif tamper in {"source", "origin"}:
        child["provenance"]["source_digest" if tamper == "source" else "source_origin_commit"] = "other"
    elif tamper in {"candidate", "revision"}:
        child["candidate_id" if tamper == "candidate" else "candidate_revision"] = "other"
    elif tamper == "task_key":
        child["task_results"][0]["compatibility_key"] = "different-task-law"
    elif tamper in {"duplicate_task", "extra_task"}:
        extra = deepcopy(child["task_results"][0])
        if tamper == "extra_task":
            extra["task_id"] = "other-task"
        child["task_results"].append(extra)
    elif tamper == "identity":
        child["attempt_id"] = "other-attempt"
    with pytest.raises(ValueError, match="compact retry"):
        certified_retry_successors(summaries)


def test_compact_repair_rejects_conflicting_branches_and_cycles(report):
    summaries = _compact_repair(report)
    sibling = deepcopy(summaries["attempt-retry"])
    sibling["attempt_id"] = "attempt-sibling"
    summaries["attempt-sibling"] = sibling
    with pytest.raises(ValueError, match="conflicting compact retry branches"):
        certified_retry_successors(summaries)
    del summaries["attempt-sibling"]
    parent, child = summaries["attempt-one"], summaries["attempt-retry"]
    child["attempt_status"] = "error"
    child["task_results"][0]["gate_status"] = "INCOMPLETE"
    parent["retry_of"] = {**child["retry_of"], "attempt_id": "attempt-retry",
                          "result_hash": child["provenance"]["canonical_result_hash"]}
    with pytest.raises(ValueError, match="cyclic compact retry"):
        certified_retry_successors(summaries)


def test_runtime_groups_never_sum_or_borrow_each_others_passes(report):
    root, publication = report
    cpu = deepcopy(publication["rows"][0])
    cpu.update(runtime_cohort={"execution_backend": "cpu"}, attempt_ids=[], tasks=[])
    publication["rows"].append(cpu)
    generate(report)
    cohorts = publication["family_progress"]["families"][0]["cohorts"]
    assert [score(cohort["total"]) for cohort in cohorts] == ["3/5 (1 not run)", "0/5 (5 not run)"]
    pages = generated_pages(root, publication)
    assert len([page for page in pages if page.parent.name == "families"]) == 1
    assert root / "reports/forge/problems.md" in pages
    publication["rows"].append(deepcopy(cpu))
    with pytest.raises(ValueError, match="multiple selected configurations"):
        build_progress(root, publication)


def test_cuda_labels_are_omitted_only_when_one_runtime_is_unambiguous(report):
    root, publication = report
    generate(report)
    page = root / "reports/forge/technique-inventory.md"
    text = render_leaderboard(root, publication, page)
    assert "**[K3P](" in text and "K3P (cuda)" not in text
    cpu = deepcopy(publication["rows"][0])
    cpu.update(runtime_cohort={"execution_backend": "cpu"}, attempt_ids=[], tasks=[])
    publication["rows"].append(cpu)
    generate(report)
    text = render_leaderboard(root, publication, page)
    assert "K3P (cuda)" in text and "K3P (cpu)" in text


def test_task_devices_and_policy_parent_remain_visible_in_provenance(report):
    root, publication = report
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    receipt["task_results"][0]["device"] = "cpu"
    atomic_json(path, receipt)
    task_path = root / "configs/forge/tasks/failed.json"
    task = read_json(task_path)
    task["policy_parent"] = {"id": "shared", "task_sha256": "original-parent-hash"}
    atomic_json(task_path, task)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["device"] == "cpu"
    text = next(iter(generated_pages(root, publication).values()))
    assert "Actual task device: `cpu`" in text
    assert "original-parent-hash" in text and "no cells to the parent clean cohort" in text


def test_scoped_task_variant_discovery_and_links_use_its_actual_declaration_path(report):
    root, publication = report
    original = root / "configs/forge/tasks/held.json"
    variant = root / "configs/forge/task-variants/policy-cohort/held.json"
    variant.parent.mkdir(parents=True)
    original.rename(variant)
    cohort = generate(report)
    assert cohort["tasks"]["held"]["current_contract"] == "matches"
    assert publication["family_progress"]["task_paths"]["held"] == variant.relative_to(root).as_posix()
    text = next(iter(generated_pages(root, publication).values()))
    assert "task-variants/policy-cohort/held.json" in text


def test_separate_policy_view_retains_required_results_without_expanding_parent_family_totals(report):
    root, publication = report
    original = generate(report)
    definition = read_json(root / "configs/forge/views/beta.json")
    definition.update(id="policy-coverage", reporting={"family_totals": False})
    atomic_json(root / "configs/forge/views/policy-coverage.json", definition)
    updated = generate(report)
    assert updated["total"] == original["total"] and updated["tiers"] == original["tiers"]
    assert publication["family_progress"]["scoped_views"] == ["policy-coverage"]
    assert updated["scoped_views"][0]["tiers"]["1"] == {"passed": 1, "total": 1,
        "incomplete": False, "not_run": 0, "incompatible": 0, "counts": {"PASS": 1}}
    text = next(iter(generated_pages(root, publication).values()))
    assert "Separate cohort coverage (excluded from family totals)" in text
    assert "This ordinary lane retains its own required gates" in text
    overview = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "↳ [policy-coverage]" not in overview


def test_leaderboard_clicks_resolve_to_family_tiers_including_empty_tiers(report):
    root, publication = report
    generate(report)
    pages = generated_pages(root, publication)
    for path, content in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    overview = root / "reports/forge/technique-inventory.md"
    text = render_leaderboard(root, publication, overview)
    assert "**[3/5 (1 not run)]" in text and "↳ [alpha]" in text
    table = text.split("| --- | ---: | ---: | ---: | ---: |", 1)[1].split("\n\n", 1)[0]
    for target in re.findall(r"\]\(([^)]+)\)", table):
        filename, _, fragment = target.partition("#")
        assert filename.startswith("families/")
        if fragment:
            assert f'<a name="{fragment}"></a>' in (overview.parent / filename).read_text()
        else:
            assert "## Technique overview" in (overview.parent / filename).read_text()
    assert "**[2/2]" in text  # A complete Tier 1 has no incomplete marker.
    assert publication["family_progress"] == build_progress(root, publication)
    assert pages == generated_pages(root, publication)


def test_committed_pages_and_every_drilldown_link_match_the_generator(tmp_path):
    # This is the current generated-artifact guard. Archived clock/selection
    # questions below deliberately use their original published task cards.
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    assert publication["family_progress"] == build_progress(root, publication)
    pages = generated_pages(root, publication)
    overview = root / "reports/forge/technique-inventory.md"
    from reports.forge.regenerate_technique_inventory import _current_markdown
    pages[overview] = _current_markdown(publication, root, overview)
    linked_text_cache = {}
    for page, text in pages.items():
        assert page.read_text() == text
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target:
                continue
            filename, _, fragment = target.partition("#")
            linked = (page.parent / filename).resolve()
            assert linked.is_file(), (page, target)
            if fragment:
                # dict.get evaluates its default eagerly. Read each external
                # target once while checking every original link and anchor.
                text = pages.get(linked)
                if text is None:
                    if linked not in linked_text_cache:
                        linked_text_cache[linked] = linked.read_text()
                    text = linked_text_cache[linked]
                assert f'<a name="{fragment}"></a>' in text, (page, target)


def test_current_clock_measurement_retains_original_evidence_alongside_policy_advance(tmp_path):
    from experiments.forge.trainer_families import scientific_row_hash
    from tests.archived_forge_contracts import published_develop_checkout
    root = published_develop_checkout(tmp_path)
    publication = read_json(root / "reports/forge/technique-inventory.json")
    recorded = deepcopy(publication["rows"])
    progress = build_progress(root, publication)
    assert publication["rows"] == recorded
    assert progress["qualification_input"] is False
    for family in progress["families"]:
        cohort = next(item for item in family["cohorts"] if item["anchor"] == family["current_cohort_anchor"])
        original = recorded[cohort["row_index"]]
        tasks = {task["task_id"]: task for task in original["tasks"] + original.get("nonrequired_tasks", [])}
        clock = cohort["tasks"]["clockfree_audit_measurement_v1"]
        assert clock["status"] == tasks["clockfree_audit_measurement_v1"]["status"]
        if original["attempt_ids"]:
            assert clock["status"] in {"PASS", "FAIL"} and clock["current_contract"] == "matches"
            assert cohort["tasks"]["five_word_joint_smoke"]["status"] == "PASS"
        else:
            assert clock["status"] in {"UNKNOWN", "BLOCKED", "NOT_RUN"}
        assert cohort["tasks"]["clockfree_audit"]["status"] == "UNKNOWN"
        assert cohort["tasks"]["clockfree_audit"]["current_contract"] == "unbound"
        # The clock diagnostic grants no qualification. A configured standard
        # can now qualify independently by passing every ordinary smoke gate.
        if original["qualified_tier"]:
            assert original["tiers"]["1"]["passed"] == original["tiers"]["1"]["total"]
            assert all(task["status"] == "PASS" for task in original["tasks"]
                       if task["task_id"] in publication["tier_requirements"]["1"])
    # Advancing the policy retains the exact previous selection and failed
    # clock verdict, even though the current BCAP DualNorm audit passes.
    previous = publication["archived_policies"][-1]
    card = read_json(root / previous["family_selection"]["path"])
    pin = next(item for item in card["selections"] if item["trainer_family"] == "bcap")
    archived = [{key: value for key, value in row.items() if key != "evidence_policy"}
                for row in publication["archived_evidence_rows"]
                if row["candidate_id"] == pin["candidate_id"]]
    archived = [row for row in archived if scientific_row_hash(row) == pin["scientific_row_sha256"]]
    assert len(archived) == 1
    clock = next(task for task in archived[0]["nonrequired_tasks"]
                 if task["task_id"] == "clockfree_audit_measurement_v1")
    assert clock["status"] == "FAIL"
    # Detailed historical clock diagnostics retain their original cohort.
    clock = next(cohort["tasks"]["clockfree_audit_measurement_v1"]
                 for family in progress["families"] + progress["configuration_families"] if family["id"] == "bcap"
                 for cohort in family["cohorts"] if "clock_audit" in cohort["tasks"]["clockfree_audit_measurement_v1"])
    assert clock["clock_audit"]["comparisons"]["step_label"]["digest_equal"] is False
    assert clock["clock_audit"]["comparisons"]["horizon"]["digest_equal"] is False
    assert clock["training_media"][0]["recorded_grade"] == "FAIL"


def test_unmeasured_current_incumbents_receive_no_execution_or_qualification():
    from experiments.forge.trainer_families import scientific_row_hash
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    rows = deepcopy(publication["rows"])
    progress = build_progress(root, publication)
    selection = read_json(root / "configs/forge/selections/family-current-v1.json")
    for pin in selection["selections"]:
        if pin["selection_kind"] != "historical_incumbent":
            continue
        selected = [row for row in rows if scientific_row_hash(row) == pin["scientific_row_sha256"]]
        assert len(selected) == 1
        row = selected[0]
        assert row["attempt_ids"] == [] and row["qualified_tier"] == 0
        assert row["selection"]["qualified"] is False and row["selection"]["default_adoption"] is False
        assert row["cost"]["wall_seconds"] is None
        assert all(task["status"] in {"UNKNOWN", "BLOCKED", "NOT_RUN"}
                   for task in row["tasks"] + row.get("nonrequired_tasks", []))
    publication["family_progress"] = progress
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "constructor refusals as incompatible" in text
    assert "earn no pass credit" in text
    assert publication["rows"] == rows


def test_current_measurement_families_complete_only_their_declared_view_scope(tmp_path):
    from experiments.forge.trainer_families import scientific_row_hash
    from tests.archived_forge_contracts import published_develop_checkout
    root = published_develop_checkout(tmp_path)
    publication = read_json(root / "reports/forge/technique-inventory.json")
    rows = deepcopy(publication["rows"])
    selection = read_json(root / "configs/forge/selections/family-current-v1.json")
    pins = {pin["trainer_family"]: pin for pin in selection["selections"]}
    progress = build_progress(root, publication)
    families = {family["id"]: family for family in progress["families"] + progress["configuration_families"]}
    unmeasured = {row["trainer_family"] for row in rows
                  if not row["attempt_ids"] and row["qualified_tier"] == 0}
    registry = read_json(root / "configs/forge/trainer-families.json")["families"]
    hidden = {family["id"] for family in registry if family.get("inventory_visible") is False}
    assert set(families) == (set(pins) | unmeasured) - hidden
    assert {family["id"] for family in registry
            if family.get("unmeasured_display_backend")} - hidden <= unmeasured
    for row in rows:
        if row["trainer_family"] in unmeasured:
            assert not row["attempt_ids"] and row["qualified_tier"] == 0
            assert row["selection"]["qualified"] is False and row["selection"].get("default_adoption", False) is False
            assert all(task["status"] in {"UNKNOWN", "BLOCKED", "NOT_RUN"}
                       for task in row["tasks"] + row.get("nonrequired_tasks", []))
    for name, pin in pins.items():
        if pin["selection_kind"] != "current_measurement":
            continue
        required = set(pin.get("measurement_tasks", []))
        for view_name in pin["measurement_views"]:
            view = read_json(root / f"configs/forge/views/{view_name}.json")
            required.update(assignment["task"] for assignment in view["assignments"]
                            if assignment["importance"] == "required" and assignment["qualification_tier"] == 1)
        assert required
        current = [cohort for cohort in families[name]["cohorts"]
                   if cohort["row_index"] < len(rows)
                   and scientific_row_hash(rows[cohort["row_index"]]) == pin["scientific_row_sha256"]]
        assert len(current) == 1
        for cohort in current:
            original = rows[cohort["row_index"]]
            recorded_tasks = {task["task_id"]: task for task in original["tasks"] + original.get("nonrequired_tasks", [])}
            assert all(recorded_tasks[task]["status"] in {"PASS", "FAIL"} for task in required)
            views = {view["id"]: view for view in cohort["views"] + cohort.get("scoped_views", [])}
            for view_name in pin["measurement_views"]:
                tier = views[view_name]["tiers"]["1"]
                assert tier["incomplete"] is False
                assert set(tier["counts"]) <= {"PASS", "FAIL"}
                assert all(cohort["tasks"][task]["current_contract"] == "matches" for task in required)
            # Other goal views retain any unknown cells; a bounded measurement
            # grants neither execution nor qualification outside its scope.
            for view in cohort["views"]:
                declaration = read_json(root / f"configs/forge/views/{view['id']}.json")
                statuses = [cohort["tasks"][assignment["task"]]["status"] for assignment in declaration["assignments"]
                            if assignment["importance"] == "required" and assignment["qualification_tier"] == 1]
                if any(status in {"UNKNOWN", "NOT_RUN", "BLOCKED"} for status in statuses):
                    assert view["tiers"]["1"]["incomplete"] is True
            assert (rows + publication.get("historical_family_rows", []))[cohort["row_index"]]["qualified_tier"] == 0
    assert publication["rows"] == rows


def test_legacy_gan_pages_preserve_original_whole_rows_and_do_not_enter_current_totals():
    from experiments.forge.trainer_families import scientific_row_hash
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    selection = read_json(root / "configs/forge/selections/family-current-v1.json")
    current = [row for row in publication["rows"] if row["trainer_family"] == "release07-gan-v3"]
    assert len(current) == 1
    old_pins = {}
    for pin in selection["historical_selections"]:
        old_pins.setdefault(pin["trainer_family"], set()).add(pin["scientific_row_sha256"])
    for row in publication["historical_family_rows"]:
        assert scientific_row_hash(row) in old_pins[row["trainer_family"]]
    aliases = [family for family in publication["family_progress"]["historical_families"]
               if family["id"].startswith("release07-gan-v3-")]
    historical_ids = {family["id"] for family in read_json(root / "configs/forge/trainer-families.json")["historical_families"]}
    assert {family["id"] for family in aliases} == {
        name for name in set(old_pins) & historical_ids if name.startswith("release07-gan-v3-")}
    assert not historical_ids & {family["id"] for family in publication["family_progress"]["families"]}
    pages = generated_pages(root, publication)
    for family in aliases:
        text = pages[root / family["page"]]
        assert "Historical cohort navigation" in text
        assert '<a name="cohort-cuda-7f9c23eb0e27-tier-1"></a>' in text
        assert "not pooled into it" in text


def test_first_case_snapshot_is_archival_and_missing_later_reports_add_no_cells(report, monkeypatch):
    root, publication = report
    from experiments.forge import family_reports
    historical = {"first_case": {"task_id": "two_pole", "status": "FAIL"},
                  "campaign_status": "PENDING_ADAPTER_AND_BUDGET", "remaining_not_run": 25}
    monkeypatch.setattr(family_reports, "_full_original_atlas_first_case", lambda root, load: deepcopy(historical))
    before = deepcopy(publication["rows"])
    progress = build_progress(root, publication)
    assert publication["rows"] == before
    assert "atlas_evidence_navigation" not in progress
    archived = progress["full_original_atlas_common26"]
    assert archived["archived_snapshot"] is True
    assert archived["counts_scope"] == "AT_FIRST_ACCEPTED_CASE"
    assert archived["remaining_not_run"] == historical["remaining_not_run"]
    assert score(progress["families"][0]["cohorts"][0]["total"]) == "3/5 (1 not run)"


EXPLANATIONS = "reports/forge/test-dispositions-20261015/README.md"


def hold_note(name, reason="no_global_recipe"):
    return {"schema": "forge_hold_v1", "previous_importance": "required", "reason": reason, "blocked_by": [],
            "summary": f"Fixture: {name} is measured and shown but not counted.",
            "explanation": f"{EXPLANATIONS}#{name}", "reentry": f"Count {name} again when one recipe passes it with the rest."}


def removal_note(name):
    return {"schema": "forge_count_removal_v1", "previous_importance": "required", "reason": "superseded",
            "counted_by": None, "summary": f"Fixture: {name} left the count of this view.",
            "explanation": f"{EXPLANATIONS}#{name}"}


def uncount(root, view, changes):
    """Rewrite a fixture view as a disposition apply would: {task: (importance, note key, note)}."""
    path = root / f"configs/forge/views/{view}.json"
    definition = read_json(path)
    for assignment in definition["assignments"]:
        if assignment["task"] in changes:
            importance, key, note = changes[assignment["task"]]
            assignment.update({"importance": importance, key: note})
    atomic_json(path, definition)


def hold_alpha(root):
    # alpha keeps one counted test; its Tier 2 and Tier 3 tests are held, its diagnostic is removed.
    uncount(root, "alpha", {"failed": ("held", "hold", hold_note("failed")),
                            "missing": ("held", "hold", hold_note("missing", "unsolved")),
                            "diagnostic": ("diagnostic", "removal", removal_note("diagnostic"))})


def assert_anchored_links_resolve(root, pages):
    """Every page-and-anchor link, including explanation links, reaches its anchor."""
    readme = root / EXPLANATIONS
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("".join(f'<a name="{name}"></a>\n\n' for name in ("failed", "missing", "diagnostic", "held")))
    for path, content in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    targets = [(path, target) for path, content in pages.items()
               for target in re.findall(r"\]\(([^)]+#[^)]+)\)", content)]
    assert any(EXPLANATIONS.rsplit("/", 2)[1] in target for _, target in targets)
    for path, target in targets:
        filename, _, fragment = target.partition("#")
        linked = (path.parent / filename).resolve()
        assert linked.is_file() and f'<a name="{fragment}"></a>' in linked.read_text(), (path, target)


def test_views_without_held_tasks_add_no_held_sections_and_count_every_requirement(report):
    root, publication = report
    generate(report)
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "**5 means 5 counted view requirements, including repeated tests**, from 4 distinct tests. These totals" in text
    for absent in ("Held and removed tests", "Held (measured, not counted)", "Counts follow the declared view revisions"):
        assert absent not in text
    page = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert "Held tests (still measured)" not in page and "Held (measured, not counted)" not in page
    assert "| [failed](k3p.md#" in page and "| required | FAIL |" in page
    # Refresh guidance names the registered view whose change alone switches the plain command.
    assert "always add `--refresh-publication`" in text and "only when alpha itself changed" in text


def test_held_and_removed_tests_leave_the_count_and_are_listed_from_the_views(report):
    root, publication = report
    hold_alpha(root)
    cohort = generate(report)
    alpha, beta = cohort["views"]
    assert score(alpha["total"]) == "1/1" and score(alpha["held"]) == "0/2 (1 not run)" and "held" not in beta
    assert score(cohort["total"]) == "3/3"
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    summed = [read_json(root / f"configs/forge/views/{name}.json") for name in ("alpha", "beta")]
    counted = sum(a["importance"] == "required" for view in summed for a in view["assignments"])
    assert f"**{counted} means {counted} counted view requirements, including repeated tests**" in text
    assert "2 held and 1 removed view assignments (2 held tests and 1 removed test) are still measured" in text
    section = text.split("## Held and removed tests", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    readme = "test-dispositions-20261015/README.md"
    assert rows == [
        "| `failed` | held: no one recipe passes it together with the counted tests | alpha (Tier 2) | "
        f"{hold_note('failed')['summary']} | {hold_note('failed')['reentry']} | 0 of 1 | [explanation]({readme}#failed) |",
        "| `missing` | held: no recipe has passed its current declaration | alpha (Tier 3) | "
        f"{hold_note('missing')['summary']} | {hold_note('missing')['reentry']} | 0 of 1 | [explanation]({readme}#missing) |",
        "| `diagnostic` | removed: replaced by newer tests | alpha (Tier 2) | "
        f"{removal_note('diagnostic')['summary']} | — | 0 of 1 | [explanation]({readme}#diagnostic) |"]
    assert "No new test was passed." in section
    assert "K3P's recorded measurement fails no counted test: 3/3 pass. Every test it fails is held." in section
    assert "K3P passes every counted test that any other selected recipe passes. It is not a new measurement of K3P." in section
    assert ("the highest tier of alpha, Tier 3, has only held tests, so the view never shows PASS: a recipe that "
            "passes every counted tier gets HELD, and a counted failure still shows as a failure.") in section
    assert "Selected recipes passing (recorded result)" in section and "diagnostic runs" not in section
    breakdown = text.split("<summary>Selected recipe breakdown by view", 1)[1].split("</details>", 1)[0]
    assert "| Total view requirements | Held (measured, not counted) |" in breakdown
    assert f"[0/2 (1 not run)](families/k3p.md#{cohort['anchor']}-held) |" in breakdown
    assert breakdown.rstrip().endswith("| — |")  # beta holds nothing.
    pages = generated_pages(root, publication)
    pages[root / "reports/forge/technique-inventory.md"] = text
    assert_anchored_links_resolve(root, pages)
    # A held pass stays visible and is named, but it no longer counts.
    publication["rows"][0]["tasks"][1]["status"] = "PASS"
    assert score(generate(report)["total"]) == "3/3"
    section = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md").split(
        "## Held and removed tests", 1)[1].split("\n## ", 1)[0]
    assert "| 1 of 1 | [explanation](test-dispositions-20261015/README.md#failed) |" in section
    assert "Selected recipes' recorded results pass 1 of the held or removed view requirements" in section
    assert "No new test was passed" not in section and "fails no counted test" not in section


def test_disclosure_names_counted_requirements_without_a_recorded_pass(report):
    root, publication = report
    hold_alpha(root)
    page = root / "reports/forge/technique-inventory.md"
    for status, detail in (("UNKNOWN", "has no recorded result yet (UNKNOWN)"), ("BLOCKED", "is BLOCKED")):
        publication["rows"][0]["tasks"][3]["status"] = status  # beta's counted task
        generate(report)
        assert (f"K3P's recorded measurement fails no counted test: 2/3 pass and the other 1 {detail}. "
                "Every test it fails is held.") in render_leaderboard(root, publication, page)


def test_view_counting_the_same_tests_as_another_while_its_top_tier_is_held_says_so(report):
    root, publication = report
    hold_alpha(root)
    uncount(root, "beta", {"held": ("held", "hold", hold_note("held"))})
    generate(report)
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert ("While missing is held, alpha counts the same tests as beta. Its highest tier, Tier 3, has only held "
            "tests, so it never shows PASS: a recipe that passes every counted tier gets HELD, and a counted failure "
            "still shows as a failure.") in text
    assert "While held is held, beta counts the same tests as alpha. Its highest tier, Tier 2" in text
    assert ("**2 means 2 counted view requirements, including repeated tests**, from 1 distinct test. 3 held and "
            "1 removed view assignments (3 held tests and 1 removed test)") in text


def test_family_page_links_held_roles_and_lists_each_held_test_with_its_recorded_result(report):
    root, publication = report
    hold_alpha(root)
    cohort = generate(report)
    page = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    readme = "../test-dispositions-20261015/README.md"
    assert f"| [held: no one recipe passes it together with the counted tests]({readme}#failed) | FAIL |" in page
    assert f"| [held: no recipe has passed its current declaration]({readme}#missing) | UNKNOWN |" in page
    assert f"| [removed: replaced by newer tests]({readme}#diagnostic) | UNKNOWN |" in page
    assert ("Required counts (Tier 1 / 2 / 3): **1 / 0 / 0**. Held counts (measured and shown, not counted): "
            "**0 / 1 / 1**. Its highest tier, Tier 3, has only held tests, so while they are held the view never "
            "shows PASS: a recipe that passes every counted tier gets HELD.") in page
    assert "| Total distinct tests | Held (measured, not counted) |" in page
    assert f"[0/2 (1 not run)](k3p.md#{cohort['anchor']}-held) |" in page
    held = page.split("### Held tests (still measured)", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in held.splitlines() if line.startswith("| [")]
    assert [row.split("](")[0] for row in rows] == ["| [failed", "| [missing"]
    assert f"| FAIL | [held: no one recipe passes it together with the counted tests]({readme}#failed) | " in rows[0]
    assert rows[1].endswith(f"| {hold_note('missing')['summary']} | {hold_note('missing')['reentry']} |")
    second = page.split("### Tier 2 across views", 1)[1].split("<a name=", 1)[0]
    counted, uncounted = second.split("Held at this tier (measured and shown, not counted):")
    assert "[held](k3p.md" in counted and "[failed](k3p.md" not in counted
    held_rows, removed_rows = uncounted.split("Removed from the count at this tier (still measured):")
    assert "[failed](k3p.md" in held_rows and "| FAIL | matches recorded run |" in held_rows
    assert f"[removed: replaced by newer tests]({readme}#diagnostic) | UNKNOWN |" in removed_rows


def test_declared_revision_note_follows_the_recorded_view_only_when_it_changed(report):
    root, publication = report
    path = root / "configs/forge/views/alpha.json"
    definition = read_json(path)
    definition["revision"] = 2
    atomic_json(path, definition)
    generate(report)
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert ("Counts follow the declared view revisions: alpha 2, beta 1. The recorded qualification keeps alpha "
            "revision 1 until new ordinary evidence is registered; no recorded verdict was regraded.") in text
    page = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert ("Their counts follow the declared view revisions (alpha 2, beta 1); the recorded qualification keeps "
            "alpha revision 1 until new ordinary evidence is registered.") in page


def attempt_passes(report, *names):
    """Record a separate diagnostic PASS of the selected recipe for each named task."""
    root, publication = report
    selected = publication["rows"][0]
    records = []
    for name in names:
        contract = publication["task_contracts"][selected["bindings"]["task_contracts"][name]]
        key = ["source-one", selected["candidate_revision"], name, contract["execution_sha256"], contract["evaluation_sha256"]]
        records.append({"key": stable_hash(key), "source_digest": key[0], "candidate_revision": key[1], "task": name,
                        "execution_sha256": key[3], "evaluation_sha256": key[4], "status": "PASS", "attempted": True,
                        "diagnostic": True, "evidence": "reports/forge/attempt-coverage.json"})
    atomic_json(root / "reports/forge/attempt-coverage.json", {
        "schema": "forge_attempt_coverage_v1", "qualification_input": False, "qualification_reuse": False,
        "summary": {"total": len(records), "attempted": len(records), "pending": 0, "running": 0}, "records": records})


def test_held_table_counts_recorded_results_and_names_separate_diagnostic_passes(report):
    root, publication = report
    hold_alpha(root)
    publication["rows"][0]["tasks"][3]["status"] = "UNKNOWN"  # beta's counted task has no recorded result.
    attempt_passes(report, "missing", "held")
    cohort = generate(report)
    assert cohort["tasks"]["missing"]["status"] == "UNKNOWN"
    section = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md").split(
        "## Held and removed tests", 1)[1].split("\n## ", 1)[0]
    assert "| Selected recipes passing (recorded result) | Details |" in section
    assert "| 0 of 1; separate diagnostic runs only: K3P | [explanation](test-dispositions-20261015/README.md#missing) |" in section
    assert "| 0 of 1 | [explanation](test-dispositions-20261015/README.md#failed) |" in section
    assert ("No selected recipe's recorded result passes a held or removed view requirement, so every pass count is "
            "unchanged") in section
    assert ("Separate diagnostic runs of 1 selected recipe pass some of these tests (named in the table); a diagnostic "
            "run never becomes a recorded result, so those passes count nowhere.") in section
    assert ("K3P's recorded measurement fails no counted test: 2/3 pass and the other 1 has no recorded result yet "
            "(UNKNOWN); a separate diagnostic run passes held, but a diagnostic run never becomes a recorded result. "
            "Every test it fails is held.") in section
    assert ("each details link opens that test's explanation in [the decision record]"
            "(test-dispositions-20261015/README.md), which also defines the words used.") in section


def test_one_global_recipe_sentence_needs_every_counted_pass_of_the_other_recipes():
    from experiments.forge.family_reports import _uncounted_disclosure
    view = {"id": "alpha", "assignments": [
        {"task": "first", "qualification_tier": 1, "importance": "required"},
        {"task": "second", "qualification_tier": 1, "importance": "required"},
        {"task": "uncounted", "qualification_tier": 2, "importance": "held", "hold": hold_note("uncounted")}]}

    def cohort(**statuses):
        return {"tasks": {name: {"status": status} for name, status in statuses.items()}}

    leader = ({"label": "Leader"}, cohort(first="PASS", second="UNKNOWN", uncounted="FAIL"))
    rule = "That is the one-global-recipe rule at work"
    weaker = ({"label": "Other"}, cohort(first="PASS", second="FAIL", uncounted="FAIL"))
    text = _uncounted_disclosure([view], [leader, weaker])[0]
    assert "Leader's recorded measurement fails no counted test: 1/2 pass" in text and text.count(rule) == 1
    # Another selected recipe passes a counted test the leader does not: the sentence is withheld.
    stronger = ({"label": "Other"}, cohort(first="PASS", second="PASS", uncounted="FAIL"))
    text = _uncounted_disclosure([view], [leader, stronger])[0]
    assert "Leader's recorded measurement fails no counted test: 1/2 pass" in text
    assert "Leader passes every counted test" not in text
    assert "Other passes every counted test that any other selected recipe passes" in text and text.count(rule) == 1


def test_view_whose_every_test_is_held_counts_nothing_but_stays_in_the_totals(report):
    root, publication = report
    uncount(root, "beta", {"shared": ("held", "hold", hold_note("shared")), "held": ("held", "hold", hold_note("held"))})
    cohort = generate(report)
    assert [view["id"] for view in cohort["views"]] == ["alpha", "beta"]
    beta = cohort["views"][1]
    assert score(beta["total"]) == "0/0" and score(beta["held"]) == "2/2"
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "Nothing in beta is counted while held, shared are held, so it reports HELD for every recipe." in text
    assert "beta counts the same tests" not in text
    breakdown = text.split("<summary>Selected recipe breakdown by view", 1)[1].split("</details>", 1)[0]
    assert f"[2/2](families/k3p.md#{cohort['anchor']}-held) |" in breakdown.splitlines()[-2]
    page = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert ("Required counts (Tier 1 / 2 / 3): **0 / 0 / 0**. Held counts (measured and shown, not counted): "
            "**1 / 1 / 0**. Nothing here is counted while these tests are held, so the view reports HELD.") in page


def test_research_rows_say_their_recorded_assignments_still_count_held_tests(report, monkeypatch):
    from experiments.forge import research_tiers
    root, publication = report
    generate(report)
    monkeypatch.setattr(research_tiers, "tier_rows", lambda root, page: {"k3p": ["| Research recipe | 1/1 | 0/2 | 0/1 | 1/4 |"]})
    monkeypatch.setattr(research_tiers, "historical_seed_rows", lambda root, page: {})
    publication["tier_requirements"] = {"1": ["shared"], "2": ["failed"], "3": ["missing"]}
    publication["declared_view"] = {"revision": 2, "tier_requirements": {"1": ["shared"], "2": [], "3": []},
                                    "no_longer_required_tasks": {"1": [], "2": ["failed"], "3": ["missing"]}}
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert ("Research recipes use their recorded revision 9 assignments. Those assignments still count the 2 tests "
            "that alpha revision 2 holds or removes from the count (3 tests per row, against 1 counted in revision 2), "
            "so research rows are not on the same scale as the selected-recipe totals.") in text
    publication["declared_view"]["no_longer_required_tasks"] = {"1": [], "2": [], "3": []}
    assert "not on the same scale" not in render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")


def test_full_suite_keeps_held_failures_and_deduplicates_shared_tests(report):
    root, publication = report
    hold_alpha(root)
    cohort = generate(report)
    # Qualification is perfect only because its failed and missing tests are held.
    assert score(cohort["total"]) == "3/3"
    names = full_suite_tasks(publication["family_progress"])
    assert names == ["failed", "held", "missing", "shared"]
    assert full_suite_outcomes(cohort, names) == {
        "passed": 2, "failed": 1, "other": {"UNKNOWN": 1}, "total": 4}
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    selected, full_suite = text.split("## Full-suite results — failures included", 1)
    assert "| 2 | **1** | 1 | 0 | " in full_suite and " | 4 | " in full_suite
    assert "A held FAIL is still a failure" in full_suite
    assert "[3/3]" in selected and "[Problem list:" in full_suite
    assert "<details>" not in selected and "<summary>Qualification subset" not in text


def test_full_suite_never_borrows_diagnostic_passes_or_calls_errors_failures(report):
    root, publication = report
    cohort = generate(report)
    cohort["tasks"]["missing"]["run_attempt"] = {"status": "PASS", "attempted": True}
    for status in ("BLOCKED", "TIMEOUT", "ERROR", "INVALID", "INCOMPLETE", "UNKNOWN"):
        cohort["tasks"]["missing"]["status"] = status
        counts = full_suite_outcomes(cohort, full_suite_tasks(publication["family_progress"]))
        assert counts == {"passed": 2, "failed": 1, "other": {status: 1}, "total": 4}


def test_problem_page_explains_intent_failures_and_scoped_passers(report):
    root, publication = report
    hold_alpha(root)
    definitions = {"schema": "forge_problem_definitions_v1", "evidence_date": "2026-10-15", "problems": [{
        "task": "failed", "title": "Keep a learned distribution stable",
        "intent": "Stay correct while training continues.", "problem": "The model drifts away from its target.",
        "pass_summary": "A research configuration passes; the selected recipe fails.",
        "passers": ["Research config A passes this test but loses another."],
        "next_step": "Preserve the repair on the other tests.",
        "evidence": [{"label": "Audit", "path": EXPLANATIONS, "anchor": "failed"}]}]}
    atomic_json(root / "reports/forge/problem-definitions.json", definitions)
    generate(report)
    pages = generated_pages(root, publication)
    page = pages[root / "reports/forge/problems.md"]
    assert "**What it tests.** Stay correct" in page
    assert "**Historical diagnosis (2026-10-15).** The model drifts" in page
    assert "**Historical passer evidence.** A research configuration passes" in page
    assert "No selected recorded PASS; see exact outcomes and dated diagnostic evidence below." in page
    assert "**The problem.**" not in page and "**Has any configuration beaten it?**" not in page
    assert "Research config A passes" in page and "Held for qualification; included in full-suite results." in page
    assert "[FAIL](families/k3p.md#" in page
    assert "problem-definitions.json" in str(publication["family_progress"]["input_hashes"])
    # Includes the new page and all task-specific navigation targets.
    assert_anchored_links_resolve(root, pages)

    # Current summaries follow the selected record while the dated diagnosis stays intact.
    publication["rows"][0]["tasks"][1]["status"] = "PASS"
    generate(report)
    updated = generated_pages(root, publication)[root / "reports/forge/problems.md"]
    assert "| Stay correct while training continues. | Selected recorded PASS: K3P. |" in updated
    assert "**Historical diagnosis (2026-10-15).** The model drifts" in updated
    assert "**Historical passer evidence.** A research configuration passes" in updated


def test_real_selected_bispf_headline_keeps_all_eleven_failures_after_restoring_gates():
    import json
    root = Path(__file__).resolve().parents[1]
    publication = json.loads((root / "reports/forge/technique-inventory.json").read_text())
    from experiments.forge.family_reports import _selected_cohorts
    progress = publication["family_progress"]
    cohort = next(cohort for family, cohort in _selected_cohorts(progress) if family["id"] == "bispf")
    counts = full_suite_outcomes(cohort, full_suite_tasks(progress))
    # New model-mode research keeps its own source; it cannot fill eight new
    # cells in the previously selected whole recipe's ordinary evidence.
    assert counts == {"passed": 18, "failed": 11, "other": {"UNKNOWN": 35}, "total": 64}
    definitions = json.loads((root / "reports/forge/problem-definitions.json").read_text())
    described = {item["task"] for item in definitions["problems"]}
    held = {a["task"] for view in progress["views"] for a in view["assignments"]
            if a["importance"] == "held" and view["id"] not in progress["scoped_views"]}
    assert held <= described
    failed = {name for _, cohort in _selected_cohorts(progress) for name in full_suite_tasks(progress)
              if cohort["tasks"][name]["status"] == "FAIL"}
    assert failed <= described


def test_comparison_projection_labels_actual_execution_policy_separately(report):
    root, publication = report
    publication['rows'][0]['executed_policy_qualification'] = {
        'view_revision': 11, 'qualified_tier': 3, 'status': 'PASS'}
    publication['family_progress'] = build_progress(root, publication)
    pages = generated_pages(root, publication)
    family_page = pages[root / 'reports/forge/families/k3p.md']
    assert 'Comparison policy: alpha revision 1, **tier 0**' in family_page
    assert 'Actual executed-policy qualification: **tier 3**, alpha revision 11 (PASS)' in family_page
    assert 'no qualification reuse' in family_page
    overview = render_leaderboard(root, publication, root / 'reports/forge/technique-inventory.md')
    assert 'comparison policy revision 1, tier 0; actual executed-policy qualification **tier 3**, revision 11 (PASS)' in overview


def test_stretch_experiments_table_renders_tier_columns_and_supports_stretch_role(report):
    root, publication = report
    # Without stretch or held tasks, the stretch table is absent.
    generate(report)
    no_stretch_text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "## Stretch experiments" not in no_stretch_text

    # Mark 'failed' as stretch (frontier) and 'missing' as held.
    uncount(root, "alpha", {
        "failed": ("stretch", "stretch", {
            "schema": "forge_stretch_v1",
            "reason": "frontier",
            "summary": "Fixture: failed is a frontier stretch task in Tier 2.",
        }),
        "missing": ("held", "hold", hold_note("missing", "unsolved")),
    })
    cohort = generate(report)
    assert score(cohort["total"]) == "3/3"
    assert score(cohort["views"][0]["held"]) == "0/2 (1 not run)"
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "## Stretch experiments" in text
    stretch_sec = text.split("## Stretch experiments", 1)[1].split("\n## ", 1)[0]
    assert "| Family / selected recipe | Tier 1 stretch | Tier 2 stretch | Tier 3 stretch | Total stretch |" in stretch_sec
    assert "**2 stretch view assignments** across **2 distinct stretch tests**" in stretch_sec
    assert "[0/1](families/k3p.md#" in stretch_sec
    assert "[0/2 (1 not run)](families/k3p.md#" in stretch_sec
    family_page = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert "stretch: frontier stretch benchmark beyond the required suite" in family_page

