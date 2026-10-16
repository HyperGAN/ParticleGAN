"""Task child entrypoint: one dispatch path for CLI run and queue workers."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import traceback

from .contracts import atomic_json, canonical, read_json, utc_now
from .telemetry import MemoryProbe, normalize_adapter_costs


def execute(path: Path) -> int:
    resolved = read_json(path)
    request, job, worker = resolved["request"], resolved["job"], resolved["worker"]
    output = path.parent
    started = time.monotonic()
    common = {"timestamp": utc_now(), "candidate": request["candidate"]["id"],
              "revision": request["candidate_revision"], "task": job["task_id"], "attempt": worker["attempt"],
              "gpu": worker["device"], "campaign": request["campaign_id"]}
    print(canonical({**common, "event": "started"}), flush=True)
    exit_code = 0
    probe = MemoryProbe()
    adapter_started = None
    try:
        import torch
        torch.set_num_threads(job["resources"].get("cpu_threads", 1))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from .api import CapabilityError
        blockers = request["tasks"][job["task_id"]].get("preflight_blockers", [])
        if blockers:
            raise CapabilityError(blockers)
        from .adapters import run_task
        probe = MemoryProbe(device="cpu" if worker["device"] == "cpu" else "cuda:0", torch_module=torch)
        execution_job = {**job, "prerequisites": resolved.get("prerequisites", {})}
        from .sampling import validate_request_sampling
        try:
            validate_request_sampling(request, task_ids=job.get("task_ids", [job["task_id"]]))
            from .hostprofiles import validate_request_host_profiles
            validate_request_host_profiles(request, task_ids=job.get("task_ids", [job["task_id"]]))
        except ValueError as exc:
            raise CapabilityError([str(exc)]) from exc
        # Adapters stop themselves at resources.timeout_seconds. Under the default
        # operational wall-clock policy raise only that in-process deadline (in
        # memory; the frozen request and every fingerprint are unchanged) to the
        # policy's hard ceiling; the supervisor enforces the operational limit.
        from .wallclock import adapter_allowance, resolve_policy, task_policy
        policy = resolve_policy(resolved.get("wallclock_policy", "default"))
        if policy:
            tasks = dict(request["tasks"])
            for member in job.get("task_ids", [job["task_id"]]):
                entry = task_policy(policy, member)
                if entry and member in tasks:
                    card = tasks[member]
                    tasks[member] = {**card, "resources": {**card["resources"], "timeout_seconds": adapter_allowance(
                        card["resources"]["timeout_seconds"], entry)}}
            request = {**request, "tasks": tasks}
        adapter_started = time.monotonic()
        result = run_task(request, execution_job, output, "cpu" if worker["device"] == "cpu" else "cuda:0")
        if not isinstance(result, dict):
            raise TypeError("adapter must return a raw evidence receipt")
    except Exception as exc:
        from .api import CapabilityError
        if isinstance(exc, CapabilityError):
            result = {"applicability": {"status": "unsupported", "reason": str(exc)},
                      "error": {"type": type(exc).__name__, "message": str(exc)}}
        else:
            result = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        traceback.print_exc()
        exit_code = 1
    result.setdefault("execution_path", request["candidate"].get("execution_path", "public_trainer"))
    result.setdefault("api_version", request["candidate"].get("api_version", "forge-api-v1"))
    result.setdefault("api_changes", request["candidate"].get("api_changes", []))
    result.setdefault("claim_contract", request["candidate"].get("claim_contract", {}))
    result = normalize_adapter_costs(result)
    if exit_code == 0:
        # Optional provenance must never turn a numerical result into an error.
        # Observe this process after execution, rather than echoing task labels.
        from .observations import attach_observations, execution_environment
        try:
            environment = execution_environment(torch, "cpu" if worker["device"] == "cpu" else "cuda:0")
        except Exception:  # noqa: BLE001 -- a probe failure must only withhold provenance
            environment = {}  # Fail closed, including any adapter-supplied witness.
        result = attach_observations(request, job, result, environment)
    result.setdefault("cost", {})
    elapsed = time.monotonic() - started
    result["cost"]["runner_seconds"] = elapsed
    result["cost"]["adapter_seconds"] = time.monotonic() - adapter_started if adapter_started is not None else None
    measured = result["cost"].get("phase_timing", {}).get("measured_seconds")
    result["telemetry"] = {"schema_version": 1, "memory": probe.snapshot(),
        "timing": {"runner_wall_seconds": elapsed, "adapter_wall_seconds": result["cost"]["adapter_seconds"],
            "unattributed_runner_seconds": max(0., elapsed - measured) if measured is not None else None,
            "scope": "Aggregate runner/adapter time includes setup, sampling, evaluation and artifact I/O; phase timing is separate."},
        "flops": {"kind": "unavailable", "value": None}}
    atomic_json(output / "raw-result.json", result)
    print(canonical({**common, "timestamp": utc_now(), "event": "recorded", "exit_code": exit_code,
                     "cost": result["cost"], "result": str(output / "raw-result.json")}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(execute(Path(sys.argv[1]).resolve()))
