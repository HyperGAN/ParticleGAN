"""Execute the frozen Tier 1 img_conv_texture32 gate across selected recipes and emit compact receipts.

Raw checkpoints and logs stay under gitignored ``runs/conv_texture32_tier1/``.
Compact provenance receipts and actual-training GIFs are written to:
  - ``reports/toy_audit/api_contract/conv_texture32/publication.json``
  - ``reports/forge/conv-texture32-tier1/summary.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from benchmarks.toy_audit.api_conv_texture32 import list_cases
from benchmarks.toy_audit.api_run import source_identity
from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.planning import load_idea
from experiments.forge.views import grade_result

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ("rhbi-v12", "empf-v8", "bispf-throughput-v1")


def _runtime_info(device: str) -> dict:
    info = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "device": device,
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
    return info


def _summarize_result(candidate_id: str, candidate: dict, task: dict, raw_result: dict, *, wall_s: float | None = None) -> dict:
    verdict = grade_result(task, raw_result)
    ev = raw_result["evidence"]
    eval_res = verdict.get("evaluator_result", {})
    cost = raw_result.get("cost", {})
    effective_wall = wall_s if wall_s is not None else float(cost.get("adapter_loop_seconds", 0.0))
    return {
        "candidate_id": candidate_id,
        "trainer_family": candidate.get("trainer_family"),
        "recipe_preset": candidate.get("recipe_preset"),
        "status": verdict["status"],
        "reason": "; ".join(verdict.get("reasons", [])),
        "terminal_passed": eval_res.get("terminal_passed", 0),
        "minimum_stable_checks": eval_res.get("minimum_stable_checks", 5),
        "failed_bounds": eval_res.get("failed_bounds", []),
        "wall_seconds": round(effective_wall, 3),
        "cost": cost,
        "recipe": raw_result["recipe"],
        "guards": ev.get("guards", {}),
        "initial_parameters": ev.get("initial_parameters", {}),
        "batch_sequence_sha256": ev.get("batch_sequence_sha256"),
        "conv_texture32_contract": ev.get("conv_texture32_contract"),
        "live": ev.get("live"),
        "terminal_window": ev.get("observations", [])[-5:],
        "gif_source_path": ev["actual_training_gif"]["path"],
        "gif_sha256": ev["actual_training_gif"]["sha256"],
        "gif_frames": ev["actual_training_gif"]["frames"],
    }


def run_candidate(candidate_id: str, task: dict, run_root: Path, device: str, *, from_saved: bool = False) -> dict:
    candidate = load_idea(ROOT, candidate_id)
    candidate_for_run = {
        **candidate,
        "claim_contract": {
            **(candidate.get("claim_contract") if isinstance(candidate.get("claim_contract"), dict) else {}),
            "sampling_law": "task_declared",
        },
    }
    blockers = adapter_preflight(task, candidate_for_run, root=ROOT)
    if blockers:
        return {
            "candidate_id": candidate_id,
            "trainer_family": candidate.get("trainer_family"),
            "status": "BLOCKED",
            "blockers": blockers,
        }

    out_dir = run_root / candidate_id
    if from_saved:
        raw_result = read_json(out_dir / "adapter-receipt.json")
        return _summarize_result(candidate_id, candidate, task, raw_result)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    request = {
        "candidate": candidate_for_run,
        "protocol": {"seed": 0},
        "tasks": {task["id"]: task},
    }
    job = {"task_id": task["id"]}
    print(f"[conv_texture32] starting candidate={candidate_id} device={device} steps={task['execution']['steps']}", flush=True)
    t0 = time.perf_counter()
    raw_result = run_task(request, job, out_dir, device)
    wall_s = time.perf_counter() - t0
    row = _summarize_result(candidate_id, candidate, task, raw_result, wall_s=wall_s)
    print(
        f"[conv_texture32] finished candidate={candidate_id} status={row['status']} "
        f"terminal={row['terminal_passed']}/{row['minimum_stable_checks']} "
        f"wall={wall_s:.2f}s live={row['live']}",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/conv_texture32_tier1")
    parser.add_argument("--from-saved", action="store_true", help="Build compact publications from saved adapter-receipt.json files")
    args = parser.parse_args(argv)

    task = read_json(ROOT / "configs/forge/tasks/img_conv_texture32.json")
    src = source_identity()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime = _runtime_info(args.device)

    results = []
    for cid in args.candidates:
        results.append(run_candidate(cid, task, args.run_root, args.device, from_saved=args.from_saved))

    report_dir = ROOT / "reports/forge/conv-texture32-tier1"
    report_media = report_dir / "media"
    report_media.mkdir(parents=True, exist_ok=True)

    api_dir = ROOT / "reports/toy_audit/api_contract/conv_texture32"
    api_media = api_dir / "media"
    api_media.mkdir(parents=True, exist_ok=True)

    compact_rows = []
    for row in results:
        if "gif_source_path" in row:
            dst = report_media / f"{row['candidate_id']}.gif"
            shutil.copyfile(row["gif_source_path"], dst)
            assert file_hash(dst) == row["gif_sha256"]
            row_copy = {k: v for k, v in row.items() if k != "gif_source_path"}
            row_copy["gif"] = dst.relative_to(ROOT).as_posix()
            compact_rows.append(row_copy)
        else:
            compact_rows.append(row)

    summary = {
        "schema": "forge_conv_texture32_tier1_readout_v1",
        "schema_version": 1,
        "task_id": task["id"],
        "seed": 0,
        "qualification_input": False,
        "source_commit": commit,
        "execution_scientific_sources_sha256": {
            "benchmarks/toy_audit/api_conv_texture32.py": file_hash(ROOT / "benchmarks/toy_audit/api_conv_texture32.py"),
            "experiments/forge/conv_texture32.py": file_hash(ROOT / "experiments/forge/conv_texture32.py"),
            "configs/forge/tasks/img_conv_texture32.json": file_hash(ROOT / "configs/forge/tasks/img_conv_texture32.json"),
        },
        "runtime": runtime,
        "results": compact_rows,
    }
    atomic_json(report_dir / "summary.json", summary)

    # Populate supplemental API publication anchored on the primary rhbi-v12 failure witness.
    primary = next((r for r in results if r["candidate_id"] == "rhbi-v12"), results[0])
    case_def = {**list_cases()[0], "default_recipe": primary.get("recipe_preset", "rhbi_optimized")}
    api_gif_rel = "media/api-img-conv-texture32.gif"
    shutil.copyfile(primary["gif_source_path"], api_dir / api_gif_rel)
    recipe_dict = primary["recipe"]
    recipe_sha = hashlib.sha256(
        json.dumps(recipe_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    publication = {
        "schema": "forge_conv_texture32_api_publication_v1",
        "schema_version": 1,
        "seed": 0,
        "qualification_input": False,
        "cases": [case_def],
        "readouts": [
            {
                "id": case_def["id"],
                "task_id": task["id"],
                "candidate_id": primary["candidate_id"],
                "goal": case_def["goal"],
                "scope": case_def["scope"],
                "execution_status": "COMPLETE",
                "verdict": primary["status"],
                "completed_steps": task["execution"]["steps"],
                "completed_updates": task["execution"]["steps"],
                "default_updates": task["execution"]["steps"],
                "terminal_passes": primary["terminal_passed"],
                "failed_bounds": primary["failed_bounds"],
                "gif": api_gif_rel,
                "source_commit": commit,
                "qualification_input": False,
            }
        ],
        "runs": [
            {
                "id": case_def["id"],
                "task_id": task["id"],
                "candidate_id": primary["candidate_id"],
                "scope": case_def["scope"],
                "execution_status": "COMPLETE",
                "verdict": primary["status"],
                "completed_updates": task["execution"]["steps"],
                "default_updates": task["execution"]["steps"],
                "terminal_passes": primary["terminal_passed"],
                "failed_bounds": primary["failed_bounds"],
                "final_metrics": primary["live"],
                "gif": api_gif_rel,
                "gif_sha256": primary["gif_sha256"],
                "recipe": recipe_dict,
                "recipe_sha256": recipe_sha,
                "runtime": runtime,
                "source_commit": commit,
                "source_identity": src["files_sha256"]["benchmarks/toy_audit/api_conv_texture32.py"],
            }
        ],
    }
    atomic_json(api_dir / "publication.json", publication)
    print(f"[conv_texture32] wrote {report_dir / 'summary.json'} and {api_dir / 'publication.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
