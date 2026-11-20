"""Archive full concluded readouts and retain certified final summaries only."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from reports.forge.regenerate_technique_inventory import project_receipt


def compact(archive):
    archive.mkdir(parents=True, exist_ok=True)
    receipts = []
    for path in sorted((ROOT / "reports/forge/records").glob("*.json")):
        current = read_json(path)
        if not current.get("study_id", "").startswith("r1r2-repair-"):
            continue
        original_path = archive / path.name
        if current.get("full_readout_archive"):
            assert file_hash(original_path) == current["full_readout_archive"]["sha256"]
        elif original_path.exists():
            assert file_hash(path) == file_hash(original_path)
        else:
            shutil.copyfile(path, original_path)
        original = read_json(original_path)
        record = deepcopy(original)
        originals = {(row["compatibility_key"], row["task_id"]): row for row in original["task_results"]}
        rows = []
        for attempt in original["attempt_ids"]:
            for row in project_receipt(ROOT, attempt)["task_results"]:
                actual = originals[row["compatibility_key"], row["task_id"]]
                assert row["gate_status"] == actual["gate_status"]
                rows.append({**row, "attempt_id": attempt, "metrics": actual["metrics"],
                             "final_metrics_sha256": stable_hash(actual["metrics"])})
        assert len(rows) == len(original["task_results"])
        record["task_results"] = sorted(rows, key=lambda row: row["task_id"])
        record["original_source"] = original["source"]
        record["source"] = {"path": "reports/forge/r1r2-repair-20261022/README.md"}
        descriptor = dict(path=str(original_path), sha256=file_hash(original_path),
                          git_blob_oid=subprocess.check_output(
                              ["git", "hash-object", str(original_path)], cwd=ROOT, text=True).strip(),
                          bytes=original_path.stat().st_size, ever_committed=False)
        record.update(qualification_input=False, full_readout_sha256=descriptor["sha256"],
                      full_readout_archive=descriptor,
                      summary_curation="Certified final metrics, verdicts, decision outcomes and original receipt identities retained. Full per-update evidence is archived byte-exactly outside Git. This display summary supplies no qualification replacement.")
        atomic_json(path, record)
        receipts.append(dict(record_id=record["record_id"], original=descriptor,
                             compact_sha256=file_hash(path), cells=len(rows)))
    assert len(receipts) == 6
    atomic_json(Path(__file__).with_name("readout-archive.json"), dict(
        scope="lossless_original_archive_with_certified_final_display_projection",
        scientific_verdicts_changed=False, qualification_recomputed=False, records=receipts))
    print(f"Verified {len(receipts)} compact readouts; originals outside Git")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(
        "/mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/archive/readouts"))
    compact(parser.parse_args().archive.resolve())
