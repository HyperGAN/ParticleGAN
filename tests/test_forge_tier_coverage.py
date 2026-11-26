"""Diagnostic measurements cannot grant qualification or conceal missing work."""
from pathlib import Path
import shutil

import pytest

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.research_tiers import coverage_report, COVERAGE
from experiments.forge.trainer_families import scientific_row_hash

ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_coverage_preserves_every_selected_scientific_row():
    report = coverage_report(ROOT)
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    archived = read_json(ROOT / "configs/forge/selections/history/family-current-138fe6eb6c13251128aed4e0171908cd676e578682f49026fc8fd4be6492b7c2.json")
    archived_pins = {pin["candidate_id"]: pin for pin in archived["selections"]}
    all_rows = (
        publication.get("rows", [])
        + publication.get("evidence_rows", [])
        + publication.get("archived_evidence_rows", [])
    )
    rows = {}
    for candidate_id, pin in archived_pins.items():
        matched = next(
            row for row in all_rows
            if row.get("candidate_id") == pin["candidate_id"]
            and row.get("candidate_revision") == pin["candidate_revision"]
            and row.get("cohort") == pin["cohort"]
        )
        rows[candidate_id] = {**matched, "trainer_family": pin["trainer_family"]}
    assert len(report["coverage"]) == 7
    for family in report["coverage"]:
        assert scientific_row_hash(rows[family["candidate_id"]]) == family["scientific_row_sha256"]
    panels = [row for row in report["diagnostic_rows"] if row["requirements"]["2"]]
    assert {row["family"]: sum(status == "PASS" for status in row["tasks"].values()) for row in panels} == {
        "k3p": 11, "ka2": 10, "r1r2": 9, "release07-gan-v3": 9}
    assert sum(len(row["tasks"]) for row in panels) == 84
    for family in publication["family_progress"]["families"]:
        for cohort in family["cohorts"]:
            for count in [cohort["total"], *cohort["tiers"].values()]:
                assert sum(count["counts"].values()) == count["total"]
                assert count["not_run"] + count["incompatible"] <= count["total"] - count["passed"]


def test_new_audit_remains_a_separate_runtime_and_does_not_fill_ordinary_tier3():
    report = coverage_report(ROOT)
    assert report["new_attempts"] == 1
    assert report["maximum_concurrent_tests"] == 4 and report["maximum_tests_per_gpu"] == 2
    audit = report["diagnostic_rows"][-1]
    assert audit["tasks"] == {"clockfree_audit": "PASS"}
    assert audit["runtime"]["packages"]["torch"] == "2.13.0+cu126"
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    family = next(f for f in publication["family_progress"]["families"] if f["id"] == "bcap-pure")
    cohort = next(c for c in family["cohorts"] if c["anchor"] == family["current_cohort_anchor"])
    assert cohort["tiers"]["3"]["passed"] == 0
    assert cohort["tasks"]["clockfree_audit"]["execution_recorded"] is False
    archived = read_json(ROOT / "configs/forge/selections/history/family-current-138fe6eb6c13251128aed4e0171908cd676e578682f49026fc8fd4be6492b7c2.json")
    archived_revs = {pin["trainer_family"]: pin["candidate_revision"] for pin in archived["selections"]}
    for other in publication["family_progress"]["families"]:
        if other["id"] not in {"k3p", "ka2", "r1r2", "release07-gan-v3"}:
            continue
        current = next(c for c in other["cohorts"] if c["anchor"] == other["current_cohort_anchor"])
        task = current["tasks"]["clockfree_audit"]
        if current.get("candidate_revision") == archived_revs.get(other["id"]):
            assert task["separate_diagnostic_executed"] is True
        else:
            # Refreshed source/revision cohorts must not inherit prior-cohort diagnostic execution.
            assert not task.get("separate_diagnostic_executed")
        assert task["execution_recorded"] is False
        assert current["tiers"]["3"]["passed"] == 0
        assert current["tiers"]["3"]["incompatible"] == 0


def test_saved_diagnostic_input_drift_is_rejected(tmp_path):
    report = coverage_report(ROOT)
    for relative in [str(COVERAGE), *report["inputs"]]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    altered = tmp_path / "reports/forge/tier-coverage-20261014/clockfree-receipt.json"
    value = read_json(altered)
    value["task_results"][0]["gate_status"] = "FAIL"
    atomic_json(altered, value)
    with pytest.raises(ValueError, match="input changed"):
        coverage_report(tmp_path)
