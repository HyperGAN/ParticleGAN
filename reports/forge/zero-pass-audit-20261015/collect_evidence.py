"""Extract zero-pass coverage without training, sampling, or changing receipts.

Run from any directory with Python 3.9+. --check verifies the saved extraction.
The optional ring recheck needs the project's Python runtime and saved run files.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = "17df086e92e4ae700a7676993238094191d24dee"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def frozen_bytes(path):
    """Read the immutable publication named by this dated audit."""
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.check_output(["git", "show", f"{SNAPSHOT}:{relative}"], cwd=ROOT)


def collect():
    inventory_path = ROOT / "reports/forge/technique-inventory.json"
    attempts_path = ROOT / "reports/forge/attempt-coverage.json"
    inventory, attempts = json.loads(frozen_bytes(inventory_path)), json.loads(frozen_bytes(attempts_path))
    progress = inventory["family_progress"]
    excluded = set(progress["diagnostic_views"] + progress["scoped_views"])
    assignments = {}
    for view in progress["views"]:
        if view["id"] not in excluded:
            for assignment in view["assignments"]:
                if assignment["importance"] == "required":
                    assignments.setdefault(assignment["task"], []).append({
                        "view": view["id"], "revision": view["revision"],
                        "tier": assignment["qualification_tier"],
                    })
    selected = {family["id"]: next(cohort for cohort in family["cohorts"]
                if cohort["anchor"] == family["current_cohort_anchor"])
                for family in progress["families"]}
    task_ids = set(next(iter(selected.values()))["tasks"])
    zero = sorted(task for task in task_ids
                  if not any(cohort["tasks"][task]["status"] == "PASS"
                             for cohort in selected.values()))
    source_paths = [inventory_path, attempts_path]
    rows = []
    for task_id in zero:
        declaration_path = ROOT / progress["task_paths"][task_id]
        declaration = json.loads(frozen_bytes(declaration_path))
        source_paths.append(declaration_path)
        records = [record for record in attempts["records"] if record["task"] == task_id]
        selected_rows = {}
        for family, cohort in selected.items():
            row = cohort["tasks"][task_id]
            metrics = row.get("metrics", {})
            selected_rows[family] = {
                "status": row["status"], "reason": row.get("reason"),
                "current_contract": row.get("current_contract"),
                "receipt": row.get("receipt"), "attempt_id": row.get("attempt_id"),
                "endpoint": metrics.get("metrics", {}),
                "convergence": metrics.get("evaluator_summary", {}).get("convergence"),
            }
        rows.append({
            "task": task_id, "definition": progress["task_paths"][task_id],
            "scope": ("required" if task_id in assignments else
                      "policy" if "_tier1_policy_selected_cloud_v1" in task_id else "indexed_variant"),
            "required_in": assignments.get(task_id, []),
            "steps": declaration["execution"].get("steps"),
            "prior": declaration["execution"].get("prior"),
            "sampling": {key: declaration["evaluation"].get(key) for key in
                         ("sampling_law", "scoring_weights", "eval_output_noise")},
            "thresholds": declaration["evaluation"].get("thresholds"),
            "dependencies": declaration.get("dependencies", []),
            "selected": selected_rows,
            "backfill_counts": dict(sorted(Counter(r["status"] for r in records).items())),
            "backfill_passes": [{key: r.get(key) for key in
                ("family", "candidate_id", "source_digest", "candidate_revision",
                 "evaluation_sha256", "evidence", "original_attempt")} for r in records if r["status"] == "PASS"],
            "backfill_blockers": dict(sorted(Counter(r.get("reason", "See original receipt")
                for r in records if r["status"] in {"BLOCKED", "ERROR", "INCOMPLETE", "TIMEOUT"}).items())),
        })
    required_zero = [row for row in rows if row["scope"] == "required"]
    return {
        "schema": "zero_pass_test_audit_v1", "snapshot": SNAPSHOT,
        "qualification_input": False, "training_updates_added": 0, "sampling_draws_added": 0,
        "summary": {
            "selected_families": len(selected), "required_unique_tasks": len(assignments),
            "required_view_instances": sum(map(len, assignments.values())),
            "required_selected_zero": len(required_zero),
            "required_selected_zero_with_backfill_pass": sum(bool(r["backfill_passes"]) for r in required_zero),
            "required_zero_in_backfill": [r["task"] for r in required_zero if not r["backfill_passes"]],
            "policy_selected_zero": sum(r["scope"] == "policy" for r in rows),
            "other_indexed_selected_zero": sum(r["scope"] == "indexed_variant" for r in rows),
            "backfill": attempts["summary"],
        },
        "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(frozen_bytes(p)).hexdigest() for p in sorted(source_paths)},
        "tasks": rows,
    }


def recheck_ring(directory):
    """Recompute the existing gate, preserving the original BLOCKED index entry."""
    sources = ("experiments/forge/views.py",
               "reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py",
               "configs/forge/tasks/ring_extension.json", "reports/forge/attempt-coverage.json")
    if any((ROOT / name).read_bytes() != frozen_bytes(ROOT / name) for name in sources):
        raise SystemExit(f"Ring recheck requires evaluator, task and receipts from {SNAPSHOT}; "
                         "use the original snapshot and exact archived run, without training.")
    sys.path.insert(0, str(ROOT))
    from experiments.forge.views import _ring
    raw_path, result_path = directory / "raw.json", directory / "result.json"
    raw, result = read(raw_path), read(result_path)
    evidence = raw["evidence"]
    assert digest(raw_path) == result["raw_sha256"]
    attempts = read(ROOT / "reports/forge/attempt-coverage.json")["records"]
    parent = next(row for row in attempts if row.get("attempt_id") == directory.name)
    assert digest(result_path) == parent["result_sha256"]
    task_path = ROOT / "configs/forge/tasks/ring_extension.json"
    grade = _ring(read(task_path), evidence)
    manifest = evidence["provenance_checkpoint"]["artifact_manifest"]
    artifact_root = Path(evidence["provenance_checkpoint"]["artifact_root"])
    for name, item in manifest["files"].items():
        assert digest(artifact_root / name) == item["sha256"]
        assert (artifact_root / name).stat().st_size == item["size"]
    return {
        "schema": "saved_ring_extension_diagnostic_recheck_v1", "snapshot": SNAPSHOT,
        "qualification_input": False, "original_index_changed": False,
        "training_updates_added": 0, "sampling_draws_added": 0,
        "attempt_id": directory.name, "source_digest": parent["source_digest"],
        "candidate_id": parent["candidate_id"], "candidate_revision": parent["candidate_revision"],
        "raw_sha256": digest(raw_path), "result_sha256": digest(result_path),
        "task_sha256": digest(task_path), "evaluator_sha256": digest(ROOT / "experiments/forge/views.py"),
        "convergence_source_sha256": digest(ROOT / "reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py"),
        "artifact_files_verified": len(manifest["files"]),
        "artifact_manifest_sha256": manifest["sha256"], "continuity": evidence["continuity"],
        "grade": grade,
        "minimum_extension_hq": min(point["hq"] for point in evidence["dense"]
            if point["step"] > grade["metrics"]["converged_step"] + grade["metrics"]["hold_budget"]),
        "scope": "Saved observations meet the numerical gate; certification and indexed publication remain separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ring-run", type=Path)
    args = parser.parse_args()
    outputs = {HERE / "evidence.json": collect()}
    if args.ring_run:
        outputs[HERE / "ring-extension-recheck.json"] = recheck_ring(args.ring_run)
    for path, value in outputs.items():
        contents = encoded(value)
        if args.check:
            if path.read_text() != contents:
                raise SystemExit("Evidence differs: " + str(path))
        else:
            path.write_text(contents)
        print(("Verified " if args.check else "Wrote ") + str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
