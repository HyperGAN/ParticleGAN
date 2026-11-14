"""Recorded formulaic scores survive source maintenance without new qualification."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.family_reports import _selected_cohorts
from reports.forge import regenerate_technique_inventory as inventory


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = Path("reports/forge/attempt-coverage.json")


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    path = tmp_path / COVERAGE
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / COVERAGE).read_bytes())
    # The latest source has no concluded matching Tier 1 measurement.
    monkeypatch.setattr(inventory, "_concluded_current_source_tier1_payload", lambda *args: None)
    return tmp_path, publication


def test_projection_preserves_exact_selected_cohort_and_archived_coverage(recorded):
    root, publication = recorded
    original = deepcopy(publication)
    projected = inventory._project_publication_for_family_pages(root, publication)
    assert projected == original
    for _, cohort in _selected_cohorts(projected["family_progress"]):
        # The new task cannot borrow any archived ring1600 result.
        assert cohort["tasks"]["ring16_acquisition_6400"]["status"] == "UNKNOWN"
    proof = inventory._recorded_source_binary_projection(publication, root)
    assert proof["qualification_input"] is False
    assert proof["qualification_reuse"] is False
    assert inventory._current_source_tier1_readouts_markdown(publication, root, root / "report.md") == ""


def test_recorded_overview_keeps_selected_grades_without_diagnostic_pass_substitution(monkeypatch):
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    monkeypatch.setattr(inventory, "_concluded_current_source_tier1_payload", lambda *args: None)
    markdown = inventory._current_markdown(publication, ROOT, ROOT / "reports/forge/technique-inventory.md")
    assert "Scientific scope corrections" in markdown
    assert "not replace an ordinary selected-cohort grade" in markdown
    assert "binary PASS or FAIL" not in markdown
    assert "### Current-source selected-recipe Tier 1 readouts" not in markdown
    assert "not run" in markdown or "not measured" in markdown


def test_recorded_projection_rejects_a_changed_coverage_digest(recorded):
    root, publication = recorded
    path = root / COVERAGE
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="coverage digest changed"):
        inventory._recorded_source_binary_projection(publication, root)


def test_diagnostic_verdict_cannot_replace_ordinary_blocked_status(recorded, monkeypatch):
    root, publication = recorded
    family, cohort = next((family, cohort) for family, cohort in
                          _selected_cohorts(publication["family_progress"]) if family["id"] == "bcap-pure")
    task = "gaussian1d_stability"
    cohort["tasks"][task]["status"] = "BLOCKED"
    monkeypatch.setattr(inventory, "_load_attempt_coverage_family_records", lambda root: {
        (family["id"], task): {"status": "PASS", "reason": "unrelated source", "source_digest": "f" * 64}})
    projected = inventory._project_publication_for_family_pages(root, publication)
    _, projected_cohort = next((family, cohort) for family, cohort in
                              _selected_cohorts(projected["family_progress"]) if family["id"] == "bcap-pure")
    assert projected_cohort["tasks"][task]["status"] == "BLOCKED"
    assert projected == publication


def test_recorded_families_keep_separate_revision_and_source_when_candidate_ids_match(recorded):
    root, publication = recorded
    proof = inventory._recorded_source_binary_projection(publication, root)["families"]
    task = next(task for task in sorted(proof["bcap-pure"]["records"])
                if proof["bcap-pure"]["records"][task]["status"] != proof["bispf"]["records"][task]["status"])
    cohorts = {family["id"]: cohort for family, cohort in _selected_cohorts(publication["family_progress"])}
    expected = {family: cohorts[family]["tasks"][task]["status"] for family in ("bcap-pure", "bispf")}
    bcap_id = publication["rows"][cohorts["bcap-pure"]["row_index"]]["candidate_id"]
    publication["rows"][cohorts["bispf"]["row_index"]]["candidate_id"] = bcap_id
    coverage = read_json(root / COVERAGE)
    for record in coverage["records"]:
        if record["family"] == "bispf":
            record["candidate_id"] = bcap_id
    atomic_json(root / COVERAGE, coverage)
    publication["family_progress"]["input_hashes"][COVERAGE.as_posix()] = file_hash(root / COVERAGE)
    binary = inventory._selected_recipe_binary_task_statuses(publication, root)
    assert {family: binary[family][task] for family in expected} == expected


@pytest.mark.parametrize("field", ["candidate_id", "candidate_revision", "source_digest", "repair_scope"])
def test_recorded_projection_never_substitutes_another_identity_or_scoped_run(recorded, field):
    root, publication = recorded
    coverage = read_json(root / COVERAGE)
    record = next(row for row in coverage["records"]
                  if row["family"] == "bispf" and row["task"] == "five_word_joint_smoke"
                  and not any(row.get(key) for key in inventory._SCOPED_COVERAGE_KEYS))
    record[field] = "different-candidate" if field in {"candidate_id", "repair_scope"} else "f" * 64
    record["key"] = stable_hash([record[key] for key in (
        "source_digest", "candidate_revision", "task", "execution_sha256", "evaluation_sha256")])
    atomic_json(root / COVERAGE, coverage)
    publication["family_progress"]["input_hashes"][COVERAGE.as_posix()] = file_hash(root / COVERAGE)
    assert inventory._recorded_source_binary_projection(publication, root) is None
    statuses = inventory._selected_recipe_binary_task_statuses(publication, root)
    for family, cohort in _selected_cohorts(publication["family_progress"]):
        assert statuses[family["id"]] == {name: entry["status"] for name, entry in cohort["tasks"].items()}


def test_recorded_projection_validates_each_attempt_key(recorded):
    root, publication = recorded
    coverage = read_json(root / COVERAGE)
    record = next(row for row in coverage["records"] if row["family"] == "bispf")
    record["key"] = "f" * 64
    atomic_json(root / COVERAGE, coverage)
    publication["family_progress"]["input_hashes"][COVERAGE.as_posix()] = file_hash(root / COVERAGE)
    with pytest.raises(ValueError, match="invalid attempt binding"):
        inventory._recorded_source_binary_projection(publication, root)


def test_historical_task_fingerprints_are_not_aliased_to_current_source(tmp_path, monkeypatch):
    from ops import refresh_forge_readouts
    old_source, live_source = "a" * 64, "b" * 64
    execution, evaluation = "c" * 64, "d" * 64
    atomic_json(tmp_path / COVERAGE, {"records": [{
        "source_digest": old_source, "task": "unit-task",
        "execution_sha256": execution, "evaluation_sha256": evaluation,
    }]})
    monkeypatch.setattr(refresh_forge_readouts, "catalog_source_manifest",
                        lambda root: {"digest": live_source})
    fingerprints = inventory._load_attempt_coverage_task_fps(tmp_path)
    assert fingerprints == {(old_source, "unit-task"): {(execution, evaluation)}}
    assert (live_source, "unit-task") not in fingerprints
