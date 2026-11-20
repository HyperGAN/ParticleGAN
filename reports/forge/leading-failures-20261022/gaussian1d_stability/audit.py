"""Audit retained BISPF Gaussian observations; performs no training or sampling."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import operator
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TASK = "gaussian1d_stability"
ATTEMPT = "6141a1c953cc41189a24685cf8080743"
SOURCE = "d9d72d53761c898bce20102e1995fc3d899c1447"
ARCHIVE_SHA = "50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe"
OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{SOURCE}:{path}"])


def failures(row, thresholds):
    return [f"{key} {op} {bound}" for key, op, bound in thresholds
            if not (type(row.get(key)) in (int, float) and math.isfinite(row[key])
                    and OPS[op](row[key], bound))]


def phase(rows, thresholds):
    counts = Counter()
    longest = current = suffix = 0
    passes = []
    for row in rows:
        failed = failures(row, thresholds)
        counts.update(failed)
        if not failed:
            passes.append(row["step"])
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        suffix = current
    return dict(checks=len(rows), passing_checks=len(passes), passing_steps=passes,
                first_step=rows[0]["step"], last_step=rows[-1]["step"],
                longest_passing_run=longest, passing_suffix=suffix,
                failing_checks_by_bound=dict(sorted(counts.items())),
                metric_ranges={key: dict(min=min(r[key] for r in rows), max=max(r[key] for r in rows))
                               for key in ("mean_error_sigma", "std_ratio", "cdf_ks")})


def audit(path):
    raw = path.read_bytes()
    require(sha(raw) == ARCHIVE_SHA, "Original full-readout digest differs")
    readout = json.loads(raw)
    quality = json.loads((ROOT / "reports/forge/bispf-optimized/quality.json").read_text())
    row = next(r for r in readout["task_results"] if r["task_id"] == TASK)
    selected = next(r for r in quality["rows"] if r["task"] == TASK)
    require(row["_attempt_id"] == selected["attempt"] == ATTEMPT, "Attempt differs")
    require(readout["candidate_id"] == quality["candidate"] == "bispf-optimized-quintic-v1", "Recipe differs")
    require(readout["candidate_revision"] == quality["candidate_revision"], "Revision differs")
    require(quality["source_commit"] == SOURCE and quality["seed"] == row["rng"]["seed"] == 0, "Source/seed differs")
    require(row["metrics"] == selected["metrics"] and row["evaluator_result"] == selected["evaluator_summary"],
            "Published metrics or stage summaries differ from the original")
    require(row["status"] == row["gate_status"] == selected["status"] == "FAIL", "Original verdict differs")
    task_path = f"configs/forge/tasks/{TASK}.json"
    task_bytes = frozen(task_path)
    task = json.loads(task_bytes)
    require(json.loads((ROOT / task_path).read_text()) == task, "Live task differs from frozen task")
    thresholds = task["evaluation"]["thresholds"]
    ev = row["evidence"]
    observations = ev["observations"]
    expected_steps = [base + math.ceil(i * 1000 / 24) for base in range(1000, 6000, 1000) for i in range(1, 25)]
    require([r["step"] for r in observations] == expected_steps, "Observation cadence differs")
    require([r["step"] for r in ev["frozen_observations"]] == expected_steps[72:], "Frozen cadence differs")
    stages = dict(stationary=phase(observations[:72], thresholds),
                  reacquisition_window=phase(observations[72:96], thresholds),
                  shifted_hold=phase(observations[96:], thresholds),
                  frozen_control=phase(ev["frozen_observations"], thresholds))
    summary = row["evaluator_result"]
    require(stages["stationary"]["passing_checks"] == summary["stationary_passes"], "Stationary count differs")
    require(stages["shifted_hold"]["passing_checks"] == summary["shift_hold_passes"], "Hold count differs")
    require(stages["frozen_control"]["passing_checks"] == summary["frozen_passes"], "Frozen count differs")
    require(stages["reacquisition_window"]["passing_suffix"] < 5 and summary["reacquisition_status"] == "FAIL",
            "Deadline decision differs")
    require(all(failures(r, thresholds) == [] or "cdf_ks <= 0.05" in failures(r, thresholds)
                for r in observations), "CDF no longer explains every failed observation")
    require(ev["continuity"]["restored_exactly"] and ev["continuity"]["history_reset"] is False,
            "Continuation contract differs")
    require(ev["guards"]["all_finite"] and ev["guards"]["unintended_rng_deviations"] == 0, "Guard failure")
    require(ev["guards"]["optimizer_updates"] == dict(generator=6000, discriminator=6000, prior=6000),
            "Update allowance differs")
    media = next(m for m in quality["media"] if m["task"] == TASK)
    require(sha((ROOT / media["path"]).read_bytes()) == media["sha256"], "Actual-training GIF differs")
    require(sha((ROOT / media["receipt"]["path"]).read_bytes()) == media["receipt"]["sha256"], "GIF receipt differs")
    source_files = [task_path, "experiments/forge/gaussian_tasks.py", "benchmarks/toy_audit/gaussian1d_quality.py",
                    "particlegan/recipes.py", "particlegan/bispf.py", "particlegan/optim/dualnorm.py"]
    return dict(schema_version=1, scope="retained_observation_analysis", qualification_input=False,
                new_training_updates=0, new_sampling_draws=0, recorded_verdict="FAIL",
                candidate=readout["candidate_id"], candidate_revision=readout["candidate_revision"],
                source_commit=SOURCE, source_digest=quality["source_digest"], protocol_seed=0,
                attempt=ATTEMPT, original_readout_sha256=ARCHIVE_SHA,
                source_files_sha256={name: sha(frozen(name)) for name in source_files},
                thresholds=thresholds, phases=stages,
                final_metrics=row["metrics"], endpoint_failed_bounds=failures(row["metrics"], thresholds),
                continuity=ev["continuity"], resolved_recipe=row["recipe"], prior=row["prior"],
                prior_policy=ev["prior_policy"], initial_model_hashes={k: v["initial_state_sha256"]
                                                                     for k, v in row["initialization"].items()},
                data_sha256=ev["data_sha256"], rng_audit_count=len(ev["rng_audits"]),
                unintended_rng_deviations=ev["guards"]["unintended_rng_deviations"],
                completed_steps=ev["completed_steps"], continuation_updates=5000,
                original_wall_seconds=row["cost"]["wall_seconds"],
                artifact_manifest=ev["artifact_manifest"], actual_training_media=media,
                causal_limit="Observation histories establish failed settling and recovery; unsupplied state/sample files prevent gradient or controller attribution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True, help="Byte-exact archived original full readout")
    parser.add_argument("--output", type=Path, help="Write compact result; no per-update trace is emitted")
    parser.add_argument("--check", action="store_true", help="Compare with committed compact analysis")
    args = parser.parse_args()
    result = audit(args.readout)
    if args.check:
        require(result == json.loads((HERE / "analysis.json").read_text()), "Committed analysis is stale")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(audit="PASS", task=TASK, original_gate="FAIL", new_updates=0,
                          stationary_passes=result["phases"]["stationary"]["passing_checks"],
                          reacquisition_passes=result["phases"]["reacquisition_window"]["passing_checks"],
                          shifted_hold_passes=result["phases"]["shifted_hold"]["passing_checks"])), flush=True)


if __name__ == "__main__":
    main()
