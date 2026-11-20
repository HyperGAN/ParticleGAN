"""Independent timeout/cancellation supervisor for one immutable Forge attempt."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .contracts import atomic_json, read_json, utc_now
from .queue import process_identity
from .sources import compute_profile, runtime_manifest, verify_snapshot
from .telemetry import peak_rss


def stop_group(process: subprocess.Popen, grace: float = 2.0):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    # A leader can exit while leaving descendants alive. Kill the remaining group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def inherited_lease_fds():
    descriptors = [int(os.environ['FORGE_LEASE_FD'])]
    if 'FORGE_SHARED_SLOT_FD' in os.environ:
        descriptors.append(int(os.environ['FORGE_SHARED_SLOT_FD']))
    if os.environ.get('FORGE_SHARED_FLEET_FDS'):
        descriptors.extend(int(value) for value in os.environ['FORGE_SHARED_FLEET_FDS'].split(','))
    for fd in descriptors:
        os.fstat(fd)
    return tuple(dict.fromkeys(descriptors))


def execute(path: Path) -> int:
    resolved = read_json(path)
    worker, request, job = resolved["worker"], resolved["request"], resolved["job"]
    directory = path.parent
    started = time.monotonic()
    started_at = time.time()
    terminal = {"schema_version": 1, "token": worker["token"], "attempt_status": "error", "result": {}}
    cancelled = False
    child = None
    runner_seconds, grading_seconds = None, None
    phase_started = None
    phase = None

    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        verify_snapshot(Path(request["source"]["snapshot_path"]), request["source"])
        if request.get("runtime") != runtime_manifest():
            raise ValueError("runtime changed since submission; resolve a new request")
        expected_compute = job.get("science", {}).get("compute")
        if expected_compute and compute_profile("cpu" if worker["device"] == "cpu" else "cuda", expected_compute.get("model"),
                threads=job.get("resources", {}).get("cpu_threads", 1)) != expected_compute:
            raise ValueError("compute backend/hardware changed since submission; resolve a compatible request")
        inherited = inherited_lease_fds()
        # The runner inherits the lease too: SIGKILL of this supervisor cannot
        # convince recovery that a still-running training descendant has died.
        phase, phase_started = "runner", time.monotonic()
        child = subprocess.Popen([sys.executable, "-u", "-m", "experiments.forge.runtime", str(path)],
                                 start_new_session=True, pass_fds=inherited)
        atomic_json(directory / "child.json", {"pid": child.pid, "process_identity": process_identity(child.pid),
                    "supervisor_pid": os.getpid(), "token": worker["token"]})
        deadline = started + job["budget_seconds"]
        # Operational wall-clock limit (experiments/forge/wallclock.py DEFAULT_POLICY):
        # a listed task runs to its raised minimum limit, then gets one
        # contention-aware extension. FORGE_WALLCLOCK_POLICY=declared, or a resolved
        # request with wallclock_policy "declared", keeps the declared budget.
        from .wallclock import decide_extension, logged_progress, resolve_policy, task_policy
        policy = resolve_policy(resolved.get("wallclock_policy", "default"))
        entry = task_policy(policy, job["task_id"]) if policy else None
        if entry:
            from .wallclock import base_limit
            wallclock = {"declared_limit": job["budget_seconds"], "base_limit": base_limit(job["budget_seconds"], entry),
                         "effective_limit": None, "slowdown_factor": None, "decision": None,
                         "progress_at_base_limit": None, "solo_seconds_per_update": entry.get("solo_seconds_per_update"),
                         "ceiling_factor": entry["ceiling_factor"]}
            wallclock["effective_limit"] = wallclock["base_limit"]
            terminal["wallclock"] = wallclock
            deadline = started + wallclock["base_limit"]
        else:
            wallclock = None
        while child.poll() is None:
            atomic_json(directory / "heartbeat.json", {"timestamp": utc_now(), "pid": os.getpid(),
                "child_pid": child.pid, "token": worker["token"], "elapsed_seconds": time.monotonic() - started})
            if cancelled or (directory / "cancel.json").exists():
                stop_group(child)
                terminal.update(attempt_status="cancelled", reason="request cancelled")
                break
            if time.monotonic() >= deadline:
                if wallclock and wallclock["decision"] is None:
                    elapsed = time.monotonic() - started
                    progressed = logged_progress(directory / "run.log")
                    decided = decide_extension(job["budget_seconds"], elapsed, progressed, entry)
                    wallclock.update(effective_limit=decided["effective_limit"], slowdown_factor=decided["slowdown_factor"],
                                     decision=decided["decision"], projected_seconds=decided["projected_seconds"],
                                     progress_at_base_limit=progressed)
                    deadline = started + decided["effective_limit"]
                    if time.monotonic() < deadline:
                        continue
                stop_group(child)
                if wallclock:
                    terminal.update(attempt_status="timeout", outcome="NOT_MEASURED_WALLCLOCK",
                                    reason="wall-clock limit exhausted; not measured (wall-clock); operational limit "
                                           f"{wallclock['effective_limit']:.0f} s ({wallclock['decision']})")
                else:
                    terminal.update(attempt_status="timeout", reason="task wall-time budget exhausted")
                break
            time.sleep(.1)
        else:
            result_path = directory / "raw-result.json"
            if result_path.exists():
                terminal["result"] = read_json(result_path)
            if child.returncode == 0 and result_path.exists():
                terminal.update(attempt_status="completed")
            else:
                terminal.update(attempt_status="error", reason=f"runner exited {child.returncode}; inspect run.log",
                                exit_code=child.returncode)
        # Clean descendants on every path, including successful leader exit.
        stop_group(child, grace=0)
        runner_seconds = time.monotonic() - phase_started
        phase = None
        if terminal["attempt_status"] == "completed" and request.get("requires_independent_grading"):
            verify_snapshot(Path(request["source"]["snapshot_path"]), request["source"])
            phase, phase_started = "grading", time.monotonic()
            child = subprocess.Popen([sys.executable, "-u", "-m", "experiments.forge.evaluate", str(path)],
                                     start_new_session=True, pass_fds=inherited)
            atomic_json(directory / "child.json", {"pid": child.pid, "process_identity": process_identity(child.pid),
                        "supervisor_pid": os.getpid(), "token": worker["token"], "phase": "evaluation"})
            while child.poll() is None:
                if cancelled or (directory / "cancel.json").exists() or time.monotonic() >= deadline:
                    stop_group(child)
                    terminal.update(attempt_status="cancelled" if cancelled or (directory / "cancel.json").exists() else "timeout",
                                    reason="independent evaluation interrupted or over budget")
                    break
                time.sleep(.1)
            else:
                graded_path = directory / "graded-result.json"
                if child.returncode != 0 or not graded_path.exists():
                    terminal.update(attempt_status="error", reason="independent evaluator failed; raw evidence retained")
                else:
                    terminal["grading"] = read_json(graded_path)
            stop_group(child, grace=0)
            grading_seconds = time.monotonic() - phase_started
            phase = None
    except BaseException as error:
        if child:
            stop_group(child)
        traceback.print_exc()
        terminal.update(attempt_status="error", reason=f"{type(error).__name__}: {error}")
    finally:
        if phase == "runner":
            runner_seconds = time.monotonic() - phase_started
        elif phase == "grading":
            grading_seconds = time.monotonic() - phase_started
        terminal["telemetry"] = {"schema_version": 1,
            "interval": {"started_at": started_at, "finished_at": time.time(), "device": worker["device"],
                         "scope": "supervised attempt, including process startup and independent grading"},
            "runner_process_seconds": runner_seconds, "grading_process_seconds": grading_seconds,
            "supervisor_peak_rss_bytes": peak_rss(), "maximum_reaped_child_peak_rss_bytes": peak_rss(children=True),
            "memory_scope": "Peak RSS of supervisor and maximum individual reaped child, not summed concurrent process memory."}
        terminal.update(elapsed_seconds=time.monotonic() - started, finished_at=utc_now())
        atomic_json(directory / "terminal.json", terminal)
        print(f"{terminal['finished_at']} {job['task_id']} {terminal['attempt_status']} "
              f"{terminal['elapsed_seconds']:.3f}s {terminal.get('reason', '')}", flush=True)
    return 0 if terminal["attempt_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(execute(Path(sys.argv[1]).resolve()))
