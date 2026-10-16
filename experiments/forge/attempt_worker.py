"""Isolated, bounded public-API attempts for the inventory coverage runner.

Run by absolute path so the selected frozen checkout supplies every scientific
module. This file owns orchestration only; it never changes a recipe or task.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def _wallclock_module():
    """Load the orchestration-only wall-clock policy by path (no package imports)."""
    import importlib.util
    module_spec = importlib.util.spec_from_file_location(
        "experiments.forge.attempt_wallclock", Path(__file__).with_name("wallclock.py"))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "prepare_policy", "supervise", "execute", "reuse_group"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.path.read_text())
    # An absolute-script launch otherwise shadows stdlib queue with forge/queue.py.
    script_directory = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if str(Path(p).resolve()) != script_directory]
    sys.path.insert(0, spec["source_root"])
    from experiments.forge.contracts import atomic_json, file_lock, file_hash
    if args.mode == "prepare":
        from experiments.forge.planning import resolve_idea
        from experiments.forge.sources import snapshot_source
        from experiments.forge.views import load_tasks
        # The inventory spans several views; retain exact source-owned cards.
        # A scoped cohort intake names its own committed view explicitly.
        tasks = load_tasks(Path(spec["source_root"]))
        for item in spec.get("requests", [spec]):
            request = resolve_idea(Path(spec["source_root"]), item["candidate_id"],
                                   view_id=item.get("view_id", "clockfree_continuous"),
                                   through_tier=3, execution_backend="cuda")
            snapshot = snapshot_source(Path(spec["source_root"]),
                                       Path(item["output"]).parents[1] / "source", request["source"])
            request["source"]["snapshot_path"] = str(snapshot)
            request["tasks"].update({name: tasks[name] for name in item["tasks"] if name in tasks})
            atomic_json(Path(item["output"]), request)
        return
    if args.mode == "prepare_policy":
        from experiments.forge.tier1_policy import materialize
        from experiments.forge.sources import verify_snapshot
        original = Path(spec["original_request"])
        if file_hash(original) != spec["request_sha256"]:
            raise ValueError("original policy request bytes changed")
        request = json.loads(original.read_text())
        verify_snapshot(Path(spec["source_root"]), request["source"])
        request["source"]["snapshot_path"] = spec["source_root"]
        request["tasks"] = {}
        for parent in spec["parents"]:
            card = json.loads((Path(spec["source_root"]) / "configs/forge/tasks" / (parent + ".json")).read_text())
            card = materialize(card, Path(spec["source_root"]))
            request["tasks"][card["id"]] = card
        atomic_json(Path(spec["output"]), request)
        return
    directory = args.path.parent
    if args.mode == "reuse_group":
        # Only orchestration comes from the updated worker. The isolated frozen
        # package still supplies every scientific grader and source check.
        import importlib.util
        name = "experiments.forge.grouped_evidence"
        module_spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("grouped_evidence.py"))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        atomic_json(directory / "result.json", module.grade_saved_group(spec["parent"], spec["child"]))
        return
    if args.mode == "supervise":
        from experiments.forge.worker import stop_group
        with file_lock(directory / "execution.lock", blocking=False) as lease:
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1"}
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "execute", str(args.path)],
                                     env=env, start_new_session=True, pass_fds=(lease.fileno(),))
            started = time.monotonic()
            # Operational wall-clock limit (spec["wallclock_policy"], set to "default" by
            # the selected-intake runner): the task's raised minimum limit, then one
            # contention-aware extension. Absent or "declared", the declared limit is
            # enforced exactly as before.
            wallclock = None
            entry = None
            if spec.get("wallclock_policy"):
                wallclock_module = _wallclock_module()
                entry = wallclock_module.task_policy(spec["wallclock_policy"], spec["task"])
            if entry:
                base = wallclock_module.base_limit(spec["timeout_seconds"], entry)
                wallclock = {"declared_limit": spec["timeout_seconds"], "base_limit": base, "effective_limit": base,
                             "slowdown_factor": None, "progress_at_declared_limit": None, "decision": None,
                             "solo_seconds_per_update": entry.get("solo_seconds_per_update"),
                             "ceiling_factor": entry["ceiling_factor"]}
            try:
                try:
                    child.wait(timeout=wallclock["base_limit"] if wallclock else spec["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    if not wallclock:
                        raise
                    elapsed = time.monotonic() - started
                    progressed = wallclock_module.logged_progress(directory / "run.log")
                    decided = wallclock_module.decide_extension(spec["timeout_seconds"], elapsed, progressed, entry)
                    wallclock.update(effective_limit=decided["effective_limit"], slowdown_factor=decided["slowdown_factor"],
                                     decision=decided["decision"], projected_seconds=decided["projected_seconds"],
                                     progress_at_declared_limit=progressed)
                    remaining = decided["effective_limit"] - (time.monotonic() - started)
                    if remaining <= 0:
                        raise
                    child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                stop_group(child)
                value = {"status": "TIMEOUT", "attempted": True,
                    "reason": "original task wall-time limit exhausted", "updates": None,
                    "seconds": time.monotonic() - started}
                if wallclock:
                    value.update(reason="wall-clock limit exhausted; not measured (wall-clock); operational limit "
                                 f"{wallclock['effective_limit']:.0f} s ({wallclock['decision']})",
                                 wallclock=wallclock, outcome="NOT_MEASURED_WALLCLOCK")
                atomic_json(directory / "result.json", value)
            finally:
                stop_group(child, grace=0)
            if wallclock and (directory / "result.json").exists():
                value = json.loads((directory / "result.json").read_text())
                if "wallclock" not in value:
                    atomic_json(directory / "result.json", {**value, "wallclock": wallclock})
            if not (directory / "result.json").exists():
                atomic_json(directory / "result.json", {"status": "ERROR", "attempted": True,
                    "reason": f"worker exited {child.returncode}; see run.log", "updates": None,
                    "seconds": time.monotonic() - started})
        return
    started = time.monotonic()
    result = {"attempted": True, "updates": 0}
    try:
        from experiments.forge.sources import runtime_manifest, verify_snapshot
        result["runtime"] = runtime_manifest()
        if spec.get("request_sha256") and file_hash(Path(spec["request_path"])) != spec["request_sha256"]:
            raise ValueError("frozen attempt request bytes changed")
        request = json.loads(Path(spec["request_path"]).read_text())
        verify_snapshot(Path(spec["source_root"]), request["source"])
        if spec.get("dependency_blockers"):
            result.update(status="BLOCKED", block_kind="prerequisite", reason="; ".join(spec["dependency_blockers"]))
        else:
            import torch
            from experiments.forge.adapters import run_task
            from experiments.forge.api import CapabilityError
            from experiments.forge.views import grade_result
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = request["tasks"][spec["task"]]
            if spec.get("wallclock_policy"):
                # Adapters also stop themselves at resources.timeout_seconds, which
                # would pre-empt the supervisor's operational limit. Raise only that
                # in-process deadline to the hard ceiling; the supervisor enforces the
                # operational limit. resources are outside the execution/evaluation
                # fingerprints, and the frozen request file is not modified.
                entry = _wallclock_module().task_policy(spec["wallclock_policy"], spec["task"])
                if entry:
                    ceiling = _wallclock_module().adapter_allowance(spec["timeout_seconds"], entry)
                    task = request["tasks"][spec["task"]] = {**task, "resources": {**task["resources"], "timeout_seconds": ceiling}}
                    result["adapter_allowance_seconds"] = ceiling
            from experiments.forge.contracts import file_hash
            mismatched = [relative for relative, digest in task["evaluation"].get("sources", {}).items()
                          if not (Path(spec["source_root"]) / relative).is_file()
                          or file_hash(Path(spec["source_root"]) / relative) != digest]
            # Deliberately enter the actual adapter, even after a static refusal.
            # Constructor checks and the original task's checkpoint checks remain.
            try:
                if mismatched:
                    raise CapabilityError(["Original task source binding unavailable; do not port the formulation: " +
                                           ", ".join(mismatched)])
                raw = run_task(request, {"task_id": spec["task"], "prerequisites": spec.get("prerequisites", {}),
                                       "diagnostic_continuation": spec.get("diagnostic_continuation") is True},
                               directory, spec["device"])
                atomic_json(directory / "raw.json", raw)
                cost = raw.get("cost", {})
                result["updates"] = cost.get("completed_steps", cost.get("completed_updates", raw.get("raw", {}).get("steps")))
                result["grade"] = (grade_result(task, raw, diagnostic=True) if spec.get("diagnostic_continuation")
                                   else grade_result(task, raw))
                result["status"] = result["grade"]["status"]
                result["reason"] = "; ".join(result["grade"].get("reasons", []))
                result["raw_sha256"] = __import__("hashlib").sha256((directory / "raw.json").read_bytes()).hexdigest()
                # The renderer reads saved observations only; it does not sample
                # a model or alter the scientific grade.
                try:
                    import importlib.util
                    renderer_path = Path(__file__).with_name("tier1_media.py")
                    module_spec = importlib.util.spec_from_file_location("experiments.forge.attempt_media", renderer_path)
                    renderer = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(renderer)
                    display = {**raw, "gate_status": result["status"]}
                    if "dense" in raw.get("evidence", {}) and not raw["evidence"].get("observations"):
                        display["evidence"] = {**raw["evidence"], "observations": raw["evidence"]["dense"]}
                    try:
                        renderer.render(task, display, directory, directory / "training.gif")
                    except ValueError as sample_error:
                        if not display["evidence"].get("saved_observer_outputs") or not display["evidence"].get("observations"):
                            raise
                        display["evidence"] = dict(display["evidence"])
                        display["evidence"].pop("saved_observer_outputs")
                        renderer.render(task, display, directory, directory / "training.gif")
                        result["media_note"] = "Saved training metric curves; sample-panel validation failed: " + str(sample_error)
                    result["training_gif_sha256"] = file_hash(directory / "training.gif")
                except Exception as media_error:
                    result["media_error"] = f"{type(media_error).__name__}: {media_error}"
                    traceback.print_exc()
            except CapabilityError as error:
                result.update(status="BLOCKED", block_kind="incompatible", reason=str(error))
                traceback.print_exc()
    except Exception as error:
        result.update(status="ERROR", updates=None, reason=f"{type(error).__name__}: {error}"[:2000])
        traceback.print_exc()
    result["seconds"] = time.monotonic() - started
    atomic_json(directory / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
