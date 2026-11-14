"""The 6,400-update ring16 task is a separate revision; the original identity is untouched."""
import json
from copy import deepcopy
from pathlib import Path

from experiments.forge.contracts import file_hash

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "configs/forge/tasks/ring16_acquisition.json"
NEW = ROOT / "configs/forge/tasks/ring16_acquisition_6400.json"


def _load(path):
    return json.loads(path.read_text())


def test_original_ring16_identity_is_unchanged():
    old = _load(OLD)
    assert old["id"] == "ring16_acquisition"
    assert old["execution"]["steps"] == 1600
    assert old["execution"]["original_schedule_horizon"] == 400
    assert old["evaluation"]["observations"] == 96
    assert old["resources"]["timeout_seconds"] == 300


def test_revision_pins_its_predecessor():
    new = _load(NEW)
    revision = new["task_revision"]
    assert revision["revision_of"] == "ring16_acquisition"
    assert revision["previous_task_sha256"] == file_hash(OLD)


def test_revision_changes_only_declared_fields():
    old, new = _load(OLD), _load(NEW)
    assert new["id"] == "ring16_acquisition_6400"
    assert new["execution"]["steps"] == new["execution"]["original_schedule_horizon"] == 6400
    assert new["execution"]["host_definition"]["steps"] == 6400
    assert new["evaluation"]["observations"] == 384
    # Same cadence: one check every 16.67 updates, and the old 96-check grid is a subset.
    assert new["execution"]["steps"] / new["evaluation"]["observations"] == old["execution"]["steps"] / old["evaluation"]["observations"]
    projected = deepcopy(new)
    for key in ("task_revision", "research_artifacts", "description"):
        projected.pop(key)
        old.pop(key, None)
    projected["id"] = old["id"]
    projected["execution"]["steps"] = old["execution"]["steps"]
    projected["execution"]["host_definition"]["steps"] = old["execution"]["host_definition"]["steps"]
    projected["execution"]["original_schedule_horizon"] = old["execution"]["original_schedule_horizon"]
    projected["evaluation"]["observations"] = old["evaluation"]["observations"]
    projected["resources"]["timeout_seconds"] = old["resources"]["timeout_seconds"]
    assert projected == old
