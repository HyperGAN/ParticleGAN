"""Source-bound scope corrections without rewriting experiment observations.

A recorded numerical PASS is retained as such. It cannot establish a different
recipe, architecture or sampler merely because a display wants a complete row.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import read_json, stable_hash

CORRECTIONS = Path("reports/forge/evidence-corrections-2026-11-08.json")
CORRECTIONS_FILES = (
    CORRECTIONS,
    Path("reports/forge/evidence-corrections-2026-11-26.json"),
)
IDENTITY_FIELDS = ("source_digest", "candidate_revision", "task", "execution_sha256", "evaluation_sha256")


def load_corrections(root, load=None):
    corrections = {}
    for rel_path in CORRECTIONS_FILES:
        path = Path(root) / rel_path
        if not path.is_file():
            continue
        data = load(rel_path) if load else read_json(path)
        if data.get("schema") != "forge_evidence_corrections_v1":
            raise ValueError("unsupported evidence correction registry")
        for group in data["groups"]:
            for key in group["record_keys"]:
                if (not isinstance(key, str) or len(key) != 64
                        or any(c not in "0123456789abcdef" for c in key) or key in corrections):
                    raise ValueError("invalid or duplicate evidence correction key")
                corrections[key] = {
                    "id": group["id"],
                    "reason": group["reason"],
                    "registry": rel_path.as_posix(),
                    "eligible": False,
                    "record_sha256": group["record_sha256"][key],
                    "scope": "archived_contract_diagnostic",
                    "source_coverage": deepcopy(data["source_coverage"]),
                }
    return corrections


def _validate_record(record, correction):
    if record["key"] != stable_hash([record.get(field) for field in IDENTITY_FIELDS]):
        raise ValueError("corrected evidence has an invalid original identity")
    # Bind every retained runtime/provenance field, including explicitly absent
    # runtime information. The scientific key alone does not contain runtime.
    original = {key: value for key, value in record.items() if key != "evidence_correction"}
    if stable_hash(original) != correction["record_sha256"]:
        raise ValueError("corrected evidence differs from its original observation/runtime")


def coverage_disposition(root, record):
    """Return a correction only for the exact original scientific identity.

    Corrected-source execution has a new key and gets no inherited disposition.
    Neither a family label nor a task name selects a historical correction.
    """
    correction = load_corrections(root).get(record.get("key"))
    if correction is not None:
        _validate_record(record, correction)
        return deepcopy(correction)
    return None


def annotate_publication(root, publication):
    """Attach scope notes; never promote coverage over a selected-cohort grade."""
    corrections = load_corrections(root)
    if not corrections:
        return publication
    projected = deepcopy(publication)
    progress = projected.get("family_progress", {})
    paths = [rel.as_posix() for rel in CORRECTIONS_FILES if (Path(root) / rel).is_file()]
    progress["evidence_corrections"] = {
        "path": CORRECTIONS.as_posix(), "paths": paths, "count": len(corrections),
        "qualification_input": False, "qualification_reuse": False,
    }
    for family in (progress.get("families", []) + progress.get("configuration_families", [])
                   + progress.get("historical_families", [])):
        for cohort in family.get("cohorts", []):
            for cell in cohort.get("tasks", {}).values():
                record = cell.get("run_attempt")
                if record and record.get("key") in corrections:
                    _validate_record(record, corrections[record["key"]])
                    cell["evidence_correction"] = deepcopy(corrections[record["key"]])
    return projected
