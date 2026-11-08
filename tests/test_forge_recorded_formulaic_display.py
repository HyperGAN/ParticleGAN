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
    proof = inventory._recorded_source_binary_projection(publication, root)
    assert proof["qualification_input"] is False
    assert proof["qualification_reuse"] is False
