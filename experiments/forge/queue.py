"""One-machine, file-backed scheduler with atomic reservations and durable attempts.

The queue lock covers subscriptions, claims and budgets. The drain lock permits
one collector. Kernel execution leases survive coordinator crashes through an
inherited worker descriptor; heartbeat age alone never triggers a relaunch.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .contracts import (atomic_json, canonical, file_hash, file_lock, identifier, positive_number,
                        read_json, stable_hash, utc_now)
from .sources import verify_snapshot
from .execution_policy import completes_tier, group_blockers, policy


def _optional_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_set(text):
    result = set()
    for part in (text or "").split(","):
        if not part:
            continue
        bounds = [int(value) for value in part.split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _cgroup_locations(proc_root, cgroup_root):
    """Resolve membership against mount roots, including cgroup namespaces."""
    memberships = []
    for row in (_optional_text(proc_root / "self/cgroup") or "").splitlines():
        _, controllers, member = row.split(":", 2)
        memberships.append((set(controllers.split(",")) - {""}, Path(member)))
    mounts = []
    for row in (_optional_text(proc_root / "self/mountinfo") or "").splitlines():
        left, right = row.split(" - ", 1)
        fields, filesystem = left.split(), right.split()
        if filesystem[0] not in {"cgroup", "cgroup2"}:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        if mount.is_relative_to("/sys/fs/cgroup"):
            mount = cgroup_root / mount.relative_to("/sys/fs/cgroup")
        mounts.append((Path(fields[3]), mount, set(filesystem[2].split(",")), filesystem[0] == "cgroup2"))
    if not mounts:
        # Linux's ordinary hierarchy when mountinfo is unavailable.
        mounts = [(Path("/"), cgroup_root if not controllers else cgroup_root / ",".join(sorted(controllers)),
                   controllers, not controllers) for controllers, _ in memberships]
    locations = []
    for controllers, member in memberships:
        for mounted_root, mount, supplied, unified in mounts:
            if unified != (not controllers) or (controllers and not controllers & supplied):
                continue
            relative = member.relative_to(mounted_root) if member.is_relative_to(mounted_root) else member.relative_to("/")
            path = mount / relative
            # Ancestor quotas/limits constrain descendants even if leaf says max.
            while path.is_relative_to(mount):
                locations.append((path, unified, controllers))
                if path == mount:
                    break
                path = path.parent
    return locations


def host_capacity(*, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup"), affinity=None):
    """Measured Linux capacity, capped by affinity and hierarchical cgroup limits.

    Available RAM excludes swap and leaves system headroom. Queue reservations
    are deducted again at claim time: deliberately conservative while workers
    have not yet allocated their full declared working set.
    """
    proc_root, cgroup_root = Path(proc_root), Path(cgroup_root)
    memory = {}
    for row in (_optional_text(proc_root / "meminfo") or "").splitlines():
        key, value = row.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    if not memory.get("MemTotal"):
        raise ValueError("cannot determine physical host memory from /proc/meminfo")
    total = memory["MemTotal"]
    available = min(total, memory.get("MemAvailable", memory.get("MemFree", 0)))
    if affinity is None:
        try:
            affinity = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            affinity = set(range(os.cpu_count() or 1))
    cpus = set(affinity)
    quotas = []
    for path, unified, controllers in _cgroup_locations(proc_root, cgroup_root):
        if unified or "cpuset" in controllers:
            declared = _optional_text(path / "cpuset.cpus.effective") or _optional_text(path / "cpuset.cpus")
            if declared:
                cpus &= _cpu_set(declared)
        if unified:
            rate = (_optional_text(path / "cpu.max") or "max 100000").split()
        else:
            rate = [_optional_text(path / "cpu.cfs_quota_us") or "-1",
                    _optional_text(path / "cpu.cfs_period_us") or "100000"]
        if (unified or "cpu" in controllers) and rate[0] not in {"max", "-1"}:
            quota, period = int(rate[0]), int(rate[1])
            if quota > 0 and period > 0:
                quotas.append(max(1, quota // period))
        if unified or "memory" in controllers:
            limit = _optional_text(path / ("memory.max" if unified else "memory.limit_in_bytes"))
            usage = _optional_text(path / ("memory.current" if unified else "memory.usage_in_bytes"))
            if limit and limit not in {"max", "-1"}:
                limit = int(limit)
                total = min(total, limit)
                available = min(available, max(0, limit - int(usage)) if usage is not None else 0)
    total_mb, available_mb = total // (1024 * 1024), available // (1024 * 1024)
    headroom = min(512, total_mb // 20)
    return {"cpu_threads": min([len(cpus), *quotas]), "memory_mb": max(0, total_mb - headroom),
            "available_memory_mb": max(0, available_mb - headroom), "memory_headroom_mb": headroom,
            "cpu_affinity": sorted(cpus)}


def _host_request(resources):
    threads = resources.get("cpu_threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    memory = resources.get("memory_mb", 0)
    host_memory = resources.get("host_memory_mb", max(512, memory) if type(memory) in (int, float) else None)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (memory, host_memory)):
        raise ValueError("memory budgets must be finite nonnegative MiB")
    if type(resources.get("gpus", 1)) is not int or resources.get("gpus", 1) not in (0, 1):
        raise ValueError("this worker supports exactly zero or one GPU")
    return {"cpu_threads": threads, "host_memory_mb": max(512, math.ceil(host_memory), math.ceil(memory))}


def _device_matches(slot, resources, *, maximum=False):
    return ((slot["device"] != "cpu" or resources.get("allow_cpu", False))
            and (not slot.get("sharing") or resources.get("memory_mb", 0) > 0)
            and (resources.get("gpus", 1) != 0 or slot["device"] == "cpu")
            and (not resources.get("backend") or resources["backend"] == ("cpu" if slot["device"] == "cpu" else "cuda"))
            and (not resources.get("gpu_model") or resources["gpu_model"] == slot.get("model"))
            and slot.get("capacity_mb" if maximum else "memory_mb", slot.get("memory_mb", 0)) >= resources.get("memory_mb", 0))


def process_identity(pid: int) -> str | None:
    """Linux start ticks protect against PID reuse; zombies are not live workers."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else f"{pid}:{fields[19]}"
    except (FileNotFoundError, ProcessLookupError):
        return None


def lease_held(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lease, fcntl.LOCK_UN)
        return False


def fence_orphan_group(child: dict, directory: Path) -> bool:
    """Find an inherited lease in the old session even after its leader dies."""
    group = child["pid"]
    owned = process_identity(group) == child.get("process_identity")
    if not owned:
        lease = (directory / "execution.lock").resolve()
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] == "Z" or int(fields[2]) != group or int(fields[3]) != group:
                    continue
                if any(fd.resolve() == lease for fd in (proc / "fd").iterdir()):
                    owned = True
                    break
            except (OSError, ValueError):
                continue
    if owned:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return owned


class Queue:
    def __init__(self, root: Path, *, report_root: Path | None = None, grader=None, on_completion=None):
        self.root = Path(root).resolve()
        self.report_root = Path(report_root).resolve() if report_root else None
        self.grader = grader
        self.on_completion = on_completion

    @contextmanager
    def state(self):
        with file_lock(self.root / "queue.lock"):
            path = self.root / "queue" / "state.json"
            existed = path.exists()
            state = read_json(path) if existed else {
                "schema_version": 1, "campaigns": {}, "submissions": {}, "jobs": {}, "events": [], "charges": [],
            }
            state.setdefault("charges", [])
            before = canonical(state) if existed else None
            yield state
            if not existed or canonical(state) != before:
                atomic_json(path, state)

    def event(self, state, event, **fields):
        # Submitters append to the locked inbox; only the collector emits streams.
        state["events"].append({"timestamp": utc_now(), "event": event, **fields})

    def submit(self, request: dict, campaign: dict) -> dict:
        """Request contains pinned view/tasks/protocol/source and fully resolved jobs."""
        policy(request)  # Reject unknown versions before any queue mutation.
        task_ids = None
        if completes_tier(request):
            request = deepcopy(request)
            from .preflight import recheck_request
            recheck_request(request)
            authorized = {a["task"] for a in request["view"]["assignments"]
                          if a["qualification_tier"] <= request["through_tier"]}
            if authorized - request["tasks"].keys():
                raise ValueError("authorized task lacks its frozen definition")
            task_ids = {member for job in request["jobs"]
                        if self._authorized(request, job) and not group_blockers(request, job)
                        for member in job.get("task_ids", [job["task_id"]])}
        from .decision_contracts import validate_admission as validate_decision_admission
        validate_decision_admission(request, campaign,
                                    root=self.report_root.parent.parent if self.report_root else None)
        from .promotion import validate_screening_submission, validate_submission
        if "calibration_lane" in request:
            from .calibration_lane import validate_submission as validate_calibration_submission
            if self.report_root is None:
                raise ValueError("calibration lane requires a repository-bound report_root")
            expected = validate_calibration_submission(self.report_root.parent.parent, request)
            if stable_hash(campaign) != stable_hash(expected):
                raise ValueError("calibration campaign differs from frozen budgets/policy")
        elif "promotion" in request:
            if self.report_root is None:
                raise ValueError("promotion requires a repository-bound report_root")
            expected = validate_submission(self.report_root.parent.parent, request)
            if stable_hash(campaign) != stable_hash(expected):
                raise ValueError("promotion campaign differs from frozen budgets/policy")
        else:
            validate_screening_submission(request,
                root=self.report_root.parent.parent if self.report_root else None, campaign=campaign)
        from .sampling import validate_request_sampling
        validate_request_sampling(request, task_ids=task_ids)
        from .hostprofiles import validate_request_host_profiles
        validate_request_host_profiles(request, task_ids=task_ids)
        identifier(campaign["id"], "campaign")
        positive_number(campaign["budget_seconds"], "campaign budget_seconds")
        positive_number(campaign["candidate_budget_seconds"], "candidate_budget_seconds")
        if not isinstance(request["through_tier"], int) or isinstance(request["through_tier"], bool) or request["through_tier"] < 1:
            raise ValueError("through_tier must be an integer >= 1")
        # Campaign is part of request identity, but not scientific compatibility.
        request = {**request, "campaign_id": campaign["id"], "queue_root": str(self.root)}
        request_id = stable_hash(request)[:24]
        request = {**request, "request_id": request_id}
        with self.state() as state:
            from .decision_contracts import register_round
            register_round(state, request)
            if self.report_root:
                from .lifecycle import ensure_open
                ensure_open(self.report_root.parent.parent, request)
            previous = state["campaigns"].get(campaign["id"])
            if previous and previous["definition"] != campaign:
                raise ValueError("campaign definition is immutable; use a new campaign id")
            if request_id in state["submissions"]:
                entry = state["submissions"][request_id]
                if entry["status"] == "cancelled":
                    for job in request["jobs"]:
                        saved = state["jobs"][job["compatibility_key"]]
                        if self._authorized(request, job) and request_id not in saved["subscribers"]:
                            saved["subscribers"].append(request_id)
                    entry.update(status="queued", lifecycle="ready", reason=None)
                    self.event(state, "resubmitted", request=request_id, campaign=campaign["id"],
                               candidate=request["candidate"]["id"], reason="explicit enqueue after cancellation")
                return entry
            if not previous:
                state["campaigns"][campaign["id"]] = {"definition": campaign, "spent_seconds": 0.0,
                    "reserved_seconds": 0.0, "paused": False}
            for job in request["jobs"]:
                key = job["compatibility_key"]
                positive_number(job["budget_seconds"], f"{job['task_id']} budget")
                if key in state["jobs"]:
                    existing = state["jobs"][key]
                    if stable_hash(existing["definition"]) != stable_hash(job):
                        raise ValueError("compatibility-key collision with a different resolved job")
                    if self._authorized(request, job):
                        existing["subscribers"].append(request_id)
                else:
                    state["jobs"][key] = {"definition": job, "subscribers": [request_id] if self._authorized(request, job) else [],
                        "status": "pending", "attempts": [], "result": None, "cost_owner": None}
            entry = {"request": request, "status": "queued", "submitted_at": time.time(),
                     "lifecycle": "ready", "reason": None}
            state["submissions"][request_id] = entry
            atomic_json(self.root / "queue" / "requests" / f"{request_id}.json", request)
            self.event(state, "submitted", request=request_id, campaign=campaign["id"],
                       candidate=request["candidate"]["id"], through_tier=request["through_tier"],
                       execution_policy=policy(request))
            return entry

    @staticmethod
    def _authorized(request, definition):
        members = set(definition.get("task_ids", [definition["task_id"]]))
        tiers = {a["task"]: a["qualification_tier"] for a in request["view"]["assignments"]}
        return all(member in tiers and tiers[member] <= request["through_tier"] for member in members)

    def inspect(self) -> dict:
        path = self.root / "queue/state.json"
        if not path.exists():
            return {"schema_version": 1, "campaigns": {}, "submissions": {}, "jobs": {}, "events": []}
        with file_lock(self.root / "queue.lock"):
            return read_json(path)

    def _grade(self, task: dict, raw: dict) -> dict:
        if self.grader:
            return self.grader(task, raw)
        from .views import grade_result
        return grade_result(task, raw)

    def _results(self, state, submission):
        result = []
        for job in submission["request"]["jobs"]:
            saved = state["jobs"][job["compatibility_key"]].get("result")
            if saved:
                result.extend(saved["task_results"])
        return result

    def _eligible(self, state, submission):
        """Order within a tier is frozen; no speculative downstream reservation."""
        request = submission["request"]
        if completes_tier(request):
            return self._eligible_complete_tier(state, submission)
        assignments = sorted(request["view"]["assignments"], key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"]))
        results = {r["task_id"]: r for r in self._results(state, submission)}
        jobs = {member: j for j in request["jobs"] for member in j.get("task_ids", [j["task_id"]])}
        any_running = False
        for tier in range(1, request["through_tier"] + 1):
            group = [a for a in assignments if a["qualification_tier"] == tier]
            for item in group:
                if item["importance"] != "required":
                    continue
                row = results.get(item["task"])
                if row and row["gate_status"] != "PASS":
                    return [], f"{item['task']}: {row['gate_status']} {row.get('reason', '')}", False
            missing_required = [a for a in group if a["importance"] == "required" and a["task"] not in results]
            eligible = []
            for item in group:
                task_id = item["task"]
                if task_id in results or task_id not in jobs:
                    continue
                task = request["tasks"][task_id]
                dependencies = [d["task"] if isinstance(d, dict) else d for d in task.get("dependencies", [])
                                if not request.get("calibration_lane") or
                                (isinstance(d, dict) and d.get("kind") != "gate")]
                if not all(results.get(d, {}).get("gate_status") == "PASS" for d in dependencies):
                    continue
                key = jobs[task_id]["compatibility_key"]
                job = state["jobs"][key]
                if job["status"] == "running":
                    any_running = True
                    if item["importance"] == "required":
                        return [], None, True
                elif job["status"] == "pending":
                    eligible.append(key)
                    if item["importance"] == "required":
                        break
            if missing_required:
                if not eligible and not any_running:
                    return [], "required evidence is unavailable or prerequisites are unsatisfied", False
                return eligible[:1], None, any_running
            # Complete optional work of this tier before proceeding when requested.
            if eligible or any_running:
                return eligible[:1], None, any_running
            if request.get("calibration_lane") and any(a["task"] not in results for a in group):
                return [], "selected calibration diagnostics lack required checkpoint/data prerequisites", False
        return [], None, any_running

    def _eligible_complete_tier(self, state, submission):
        """Finish runnable jobs in the current tier before applying its veto.

        A task-local blocker never creates scientific evidence or a paid attempt.
        Checkpoint/data and explicit gate dependencies remain mandatory. Whole
        execution groups are admitted together; their internal dependencies are
        enforced by the adapter rather than awaiting their own terminal receipt.
        """
        request = submission["request"]
        assignments = sorted(request["view"]["assignments"],
                             key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"]))
        results = {r["task_id"]: r for r in self._results(state, submission)}
        jobs = {member: job for job in request["jobs"]
                for member in job.get("task_ids", [job["task_id"]])}
        for tier in range(1, request["through_tier"] + 1):
            group = [a for a in assignments if a["qualification_tier"] == tier]
            eligible, seen, required_blockers, running = [], set(), [], False
            for item in group:
                task_id = item["task"]
                row = results.get(task_id)
                if row:
                    if item["importance"] == "required" and row["gate_status"] != "PASS":
                        required_blockers.append(f"{task_id}: {row['gate_status']} {row.get('reason', '')}".rstrip())
                    continue
                definition = jobs.get(task_id)
                reason = None
                if definition is None or not self._authorized(request, definition):
                    reason = "required evidence is unavailable or execution group exceeds tier cap"
                else:
                    reasons = group_blockers(request, definition)
                    if reasons:
                        reason = "BLOCKED " + "; ".join(reasons)
                    else:
                        members = set(definition.get("task_ids", [definition["task_id"]]))
                        dependencies = {d["task"] if isinstance(d, dict) else d
                                        for member in members
                                        for d in request["tasks"][member].get("dependencies", [])} - members
                        unsatisfied = sorted(d for d in dependencies
                                             if results.get(d, {}).get("gate_status") != "PASS")
                        if unsatisfied:
                            reason = "prerequisites are unsatisfied: " + ", ".join(unsatisfied)
                        else:
                            key = definition["compatibility_key"]
                            job = state["jobs"][key]
                            if job["status"] == "running":
                                running = True
                            elif job["status"] == "pending":
                                if key not in seen:
                                    eligible.append(key)
                                    seen.add(key)
                            else:
                                reason = "required evidence is unavailable"
                if reason and item["importance"] == "required":
                    required_blockers.append(f"{task_id}: {reason}")
            if eligible or running:
                return eligible, None, running
            if required_blockers:
                return [], f"tier {tier}: " + "; ".join(required_blockers), False
        return [], None, False

    def _refresh(self, state):
        for entry in state["submissions"].values():
            if entry["status"] in {"cancelled", "concluded"}:
                continue
            eligible, reason, running = self._eligible(state, entry)
            campaign = state["campaigns"][entry["request"]["campaign_id"]]
            if reason:
                entry.update(status="blocked", reason=reason, lifecycle="awaiting_readout")
            elif running:
                entry.update(status="running", reason=None, lifecycle="running")
            elif not eligible:
                entry.update(status="completed", reason=None, lifecycle="awaiting_readout")
            elif campaign["paused"]:
                entry.update(status="paused", reason="campaign paused")
            else:
                entry.update(status="queued", reason=None)

    def _available_budget(self, state, request, seconds, *, job_key=None, new_round_reservation=True):
        campaign = state["campaigns"][request["campaign_id"]]
        definition = campaign["definition"]
        if campaign["spent_seconds"] + campaign["reserved_seconds"] + seconds > definition["budget_seconds"]:
            return False, "campaign budget cannot reserve the full next task"
        candidate_spent = sum(charge["seconds"] for charge in state.get("charges", [])
                              if charge["owner"]["campaign"] == request["campaign_id"]
                              and charge["owner"]["revision"] == request["candidate_revision"])
        for job in state["jobs"].values():
            owner = job.get("cost_owner")
            if owner and owner["campaign"] == request["campaign_id"] and owner["revision"] == request["candidate_revision"]:
                candidate_spent += job.get("reserved_seconds", 0.0)
        if candidate_spent + seconds > definition["candidate_budget_seconds"]:
            return False, "candidate budget cannot reserve the full next task"
        from .decision_contracts import available_round_budget
        return available_round_budget(state, request, seconds if new_round_reservation else 0, job_key=job_key)

    def claim(self, slots: list[dict], *, campaign_filter=None, goal_filter=None, request_filter=None) -> dict | None:
        """Called only by the drain owner; reservation is atomic with submissions."""
        with self.state() as state:
            self._refresh(state)
            running = [j for j in state["jobs"].values() if j["status"] == "running"]
            from .policy_execution import active_reservations, device_key, recover_attempts
            recover_attempts(state)
            policy = active_reservations(state)
            capacity = host_capacity()
            reserved = [_host_request(j["definition"].get("resources", {})) for j in running]
            reserved += [entry["worker"]["host_reservation"] for entry in policy]
            free_threads = capacity["cpu_threads"] - sum(r["cpu_threads"] for r in reserved)
            free_host_memory = capacity["available_memory_mb"] - sum(r["host_memory_mb"] for r in reserved)
            busy = {(device_key(j["worker"]["device"]), j["worker"]["slot"]) for j in running}
            free = [s for s in slots if (device_key(s["device"]), s["slot"]) not in busy]
            exclusive = {device_key(entry["worker"]["device"]) for entry in policy
                         if entry["worker"].get("exclusive_device")}
            free = [slot for slot in free if device_key(slot["device"]) not in exclusive]
            pending = sorted(state["submissions"].items(), key=lambda kv: (
                -(kv[1]["request"].get("priority", 0) + (time.time() - kv[1]["submitted_at"]) / 300), kv[0]))
            open_cache: dict[tuple[str | None, str | None], str | None] = {}
            for request_id, entry in pending:
                if request_filter is not None and request_id != request_filter:
                    continue
                request = entry["request"]
                if entry["status"] not in {"queued", "running"}:
                    continue
                if self.report_root:
                    from .lifecycle import ensure_open
                    open_key = (request.get("candidate", {}).get("id"), request.get("candidate_revision"))
                    if open_key not in open_cache:
                        try:
                            ensure_open(self.report_root.parent.parent, request)
                            open_cache[open_key] = None
                        except ValueError as error:
                            open_cache[open_key] = str(error)
                    if open_cache[open_key] is not None:
                        entry.update(status="blocked", reason=open_cache[open_key], lifecycle="awaiting_readout")
                        continue
                if campaign_filter and request["campaign_id"] != campaign_filter:
                    continue
                if goal_filter and request["view"]["goal"] != goal_filter:
                    continue
                if state["campaigns"][request["campaign_id"]]["paused"]:
                    continue
                eligible, _, _ = self._eligible(state, entry)
                for key in eligible:
                    job = state["jobs"][key]
                    resources = job["definition"].get("resources", {})
                    try:
                        needed = _host_request(resources)
                        expected_threads = job["definition"].get("science", {}).get("compute", {}).get("threads")
                        if expected_threads is not None and expected_threads != needed["cpu_threads"]:
                            raise ValueError("CPU thread budget differs from the frozen scientific compute profile")
                    except ValueError as error:
                        entry.update(status="blocked", reason="resource: " + str(error), lifecycle="awaiting_readout")
                        continue
                    if (needed["cpu_threads"] > capacity["cpu_threads"]
                            or needed["host_memory_mb"] > capacity["memory_mb"]):
                        entry.update(status="blocked", reason="resource: requested CPU threads/host RAM exceed affinity or cgroup host capacity",
                                     lifecycle="awaiting_readout")
                        continue
                    if needed["cpu_threads"] > free_threads or needed["host_memory_mb"] > free_host_memory:
                        if not (running or policy):
                            entry.update(status="blocked", reason="resource: insufficient currently available host RAM; retry when capacity is available",
                                         lifecycle="awaiting_readout")
                        continue
                    suitable = [s for s in free if _device_matches(s, resources)]
                    if resources.get("gpus", 1):
                        suitable.sort(key=lambda s: s["device"] == "cpu")
                    if not suitable:
                        # Distinguish occupied capacity from an impossible allocation.
                        feasible = [s for s in slots if _device_matches(s, resources, maximum=True)]
                        if not feasible or not (running or policy):
                            entry.update(status="blocked", reason="resource: no currently available configured device satisfies task memory/device requirements",
                                         lifecycle="awaiting_readout")
                        continue
                    from .wallclock import reservation_seconds, resolve_policy
                    wallclock_policy = resolve_policy(request.get("wallclock_policy", "default"))
                    seconds = reservation_seconds(job["definition"], wallclock_policy)
                    ok, reason = self._available_budget(state, request, seconds, job_key=key)
                    if not ok:
                        entry.update(status="blocked", reason="resource: " + reason, lifecycle="awaiting_readout")
                        continue
                    snapshot = Path(request["source"]["snapshot_path"])
                    verify_snapshot(snapshot, request["source"])
                    attempt = uuid.uuid4().hex
                    directory = self.root / request["campaign_id"] / attempt
                    directory.mkdir(parents=True)
                    worker = {**suitable[0], "attempt": attempt, "directory": str(directory),
                              "token": uuid.uuid4().hex, "started_at": time.time(), "pid": None,
                              "host_reservation": needed, "host_capacity_at_claim": capacity}
                    resolved = {"schema_version": 1, "request_id": request_id, "request": request,
                                "job": job["definition"], "worker": worker,
                                "wallclock_policy": wallclock_policy, "reservation_seconds": seconds}
                    prerequisites = {}
                    for member in job["definition"].get("task_ids", [job["definition"]["task_id"]]):
                        for dependency in request["tasks"][member].get("dependencies", []):
                            required = dependency["task"] if isinstance(dependency, dict) else dependency
                            if required in job["definition"].get("task_ids", []):
                                continue
                            definition = next(j for j in request["jobs"] if required in j.get("task_ids", [j["task_id"]]))
                            result = state["jobs"][definition["compatibility_key"]]["result"]
                            if result is None and request.get("calibration_lane") and (
                                    not isinstance(dependency, dict) or dependency.get("kind") == "gate"):
                                continue
                            row = next(row for row in result["task_results"] if row["task_id"] == required)
                            prerequisites[required] = {"attempt_id": result["attempt_id"],
                                "result_hash": stable_hash(result), "result": row,
                                "candidate_revision": result["candidate_revision"],
                                "compatibility_key": definition["compatibility_key"]}
                    if prerequisites:
                        resolved["prerequisites"] = prerequisites
                    if job.get("retry_of"):
                        resolved["retry_of"] = job["retry_of"]
                    atomic_json(directory / "request.json", resolved)
                    job.update(status="running", worker=worker, cost_owner={"campaign": request["campaign_id"],
                               "revision": request["candidate_revision"], "request": request_id}, reserved_seconds=seconds)
                    job["attempts"].append({"attempt_id": attempt, "path": str(directory), "token": worker["token"]})
                    state["campaigns"][request["campaign_id"]]["reserved_seconds"] += seconds
                    entry.update(status="running", lifecycle="running")
                    self.event(state, "claimed", request=request_id, campaign=request["campaign_id"],
                               candidate=request["candidate"]["id"], task=job["definition"]["task_id"],
                               attempt=attempt, worker=attempt, gpu=worker["device"],
                               budget_seconds=seconds, declared_budget_seconds=job["definition"]["budget_seconds"],
                               log=str(directory / "run.log"))
                    return resolved
            return None

    def launch(self, resolved: dict, *, module="experiments.forge.worker", shared_slot_fd=None,
               shared_fleet_fds=()) -> subprocess.Popen:
        shared_fleet_fds = tuple(shared_fleet_fds)
        if (shared_slot_fd is not None or shared_fleet_fds) and module == "experiments.forge.worker":
            expected_worker = file_hash(Path(__file__).with_name("worker.py"))
            if resolved["request"]["source"].get("files", {}).get("experiments/forge/worker.py") != expected_worker:
                raise RuntimeError("external shared leases require the current FD-forwarding worker in the frozen source")
        worker = resolved["worker"]
        directory = Path(worker["directory"])
        # launch + registration remain under queue lock. Recovery also checks the
        # lease, closing the window between process creation and PID persistence.
        with self.state() as state:
            job = state["jobs"][resolved["job"]["compatibility_key"]]
            if job["status"] != "running" or job["worker"]["token"] != worker["token"]:
                raise RuntimeError("claim was fenced before launch")
            with (directory / "execution.lock").open("a+") as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                environment = os.environ.copy()
                environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                                   CUBLAS_WORKSPACE_CONFIG=":4096:8", FORGE_LEASE_FD=str(lease.fileno()))
                environment["CUDA_VISIBLE_DEVICES"] = "" if worker["device"] == "cpu" else str(worker["device"])
                environment["PYTHONPATH"] = resolved["request"]["source"]["snapshot_path"]
                inherited = (lease.fileno(),)
                environment.pop("FORGE_SHARED_SLOT_FD", None)
                environment.pop("FORGE_SHARED_FLEET_FDS", None)
                if shared_slot_fd is not None:
                    if type(shared_slot_fd) is not int or shared_slot_fd < 0:
                        raise ValueError('shared slot descriptor must be an open nonnegative integer')
                    os.fstat(shared_slot_fd)  # fail before process creation on a stale descriptor
                    environment["FORGE_SHARED_SLOT_FD"] = str(shared_slot_fd)
                    inherited += (shared_slot_fd,)
                if shared_fleet_fds:
                    for fd in shared_fleet_fds:
                        if type(fd) is not int or fd < 0:
                            raise ValueError('fleet descriptor must be an open nonnegative integer')
                        os.fstat(fd)
                    environment['FORGE_SHARED_FLEET_FDS'] = ','.join(map(str, shared_fleet_fds))
                    inherited = tuple(dict.fromkeys((*inherited, *shared_fleet_fds)))
                with (directory / "run.log").open("ab", buffering=0) as log:
                    process = subprocess.Popen([sys.executable, "-u", "-m", module, str(directory / "request.json")],
                        cwd=resolved["request"]["source"]["snapshot_path"], env=environment,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=inherited)
                job["worker"].update(pid=process.pid, process_identity=process_identity(process.pid))
                atomic_json(directory / "process.json", job["worker"])
                return process

    def _active_tier(self, state, submission) -> int | None:
        if submission["status"] in {"blocked", "completed", "cancelled", "concluded"}:
            return None
        eligible, reason, running = self._eligible(state, submission)
        if reason or (not eligible and not running):
            return None
        tiers = {a["task"]: a["qualification_tier"] for a in submission["request"]["view"]["assignments"]}
        if eligible:
            return min(tiers[state["jobs"][k]["definition"]["task_id"]] for k in eligible)
        return min((tiers[j["task_id"]] for j in submission["request"]["jobs"]
                    if state["jobs"][j["compatibility_key"]]["status"] == "running"
                    and j["task_id"] in tiers), default=None)

    def collect(self, *, only_orphans=False):
        completions = 0
        tier_completed = False
        with self.state() as state:
            # Dedicated admission may recover abandoned reservations, but must
            # not fence the collector's claim-to-launch registration window.
            if only_orphans and lease_held(self.root / "coordinator.lock"):
                return 0
            tiers_before = {req_id: self._active_tier(state, entry)
                            for req_id, entry in state["submissions"].items()}
            touched_requests = set()
            for key, job in state["jobs"].items():
                if job["status"] != "running":
                    continue
                worker = job["worker"]
                directory = Path(worker["directory"])
                self._collect_progress(state, job)
                path = directory / "terminal.json"
                if only_orphans and (path.exists() or lease_held(directory / "execution.lock")):
                    continue
                if not path.exists():
                    if lease_held(directory / "execution.lock"):
                        process = directory / "process.json"
                        child_path = directory / "child.json"
                        identity = read_json(process) if process.exists() else worker
                        pid = identity.get("pid")
                        supervisor_alive = pid and process_identity(pid) == identity.get("process_identity")
                        if not supervisor_alive and child_path.exists():
                            child = read_json(child_path)
                            # A killed supervisor cannot enforce the deadline.
                            # Fence descendants too, then wait for lease release.
                            fence_orphan_group(child, directory)
                        continue
                    # Terminal absence plus a released execution lease proves
                    # nobody can still publish a legitimate running completion.
                    raw = {"attempt_status": "error", "reason": "worker stopped without terminal receipt; explicit retry required",
                           "elapsed_seconds": max(0, time.time() - worker["started_at"]), "token": worker["token"]}
                    atomic_json(path, raw)
                raw = read_json(path)
                if raw.get("token") != worker["token"]:
                    raise RuntimeError(f"fenced worker published stale receipt: {worker['attempt']}")
                owner = job["cost_owner"]
                request = state["submissions"][owner["request"]]["request"]
                task_id = job["definition"]["task_id"]
                raw_result = raw.get("result", {})
                elapsed = max(0.0, float(raw.get("elapsed_seconds", 0)))
                rows = []
                members = job["definition"].get("task_ids", [task_id])
                for member in members:
                    member_raw = raw_result.get("task_results", {}).get(member, raw_result)
                    if raw["attempt_status"] == "completed":
                        if request.get("requires_independent_grading"):
                            grading = raw.get("grading", {})
                            if grading.get("raw_hash") != stable_hash(raw_result) or grading.get("source_digest") != request["source"]["digest"]:
                                graded = {"gate_status": "INVALID", "reason": "missing or mismatched frozen evaluator certificate"}
                            else:
                                graded = grading.get("grades", {}).get(member, {"gate_status": "INVALID", "reason": "missing frozen task evaluation"})
                        else:
                            graded = self._grade(request["tasks"][member], member_raw)
                        row = {**member_raw, **graded, "gate_status": graded.get("gate_status", graded.get("status", "INVALID"))}
                    else:
                        row = {**member_raw, "gate_status": "BLOCKED" if member_raw.get("applicability", {}).get("status") == "unsupported" else "INCOMPLETE",
                               "reason": raw.get("reason", raw["attempt_status"])}
                    row.setdefault("reason", "; ".join(row.get("reasons", [])))
                    row.update(task_id=member, compatibility_key=key, raw_status=raw["attempt_status"])
                    if raw["attempt_status"] == "timeout":
                        # Never graded: reported as NOT MEASURED (wall-clock), not FAIL.
                        row["wallclock_outcome"] = "NOT_MEASURED_WALLCLOCK"
                    if raw.get("wallclock"):
                        row["wallclock"] = raw["wallclock"]
                    row["cost"] = {**row.get("cost", {}), "wall_seconds": elapsed if member == task_id else 0,
                                   "execution_seconds": elapsed, "charged_task": task_id,
                                   "device": worker["device"], "flops": {"kind": "unavailable", "value": None}}
                    rows.append(row)
                campaign = state["campaigns"][owner["campaign"]]
                campaign["reserved_seconds"] -= job["reserved_seconds"]
                campaign["spent_seconds"] += elapsed
                state["charges"].append({"attempt_id": worker["attempt"], "owner": dict(owner), "seconds": elapsed})
                result = {"schema_version": 1, "attempt_id": worker["attempt"], "task_results": rows,
                          "raw": raw, "candidate_revision": request["candidate_revision"], "cost_owner": owner}
                if job.get("retry_of"):
                    result["retry_of"] = job["retry_of"]
                atomic_json(directory / "result.json", result)
                if self.report_root:
                    durable = self.report_root / "attempts" / worker["attempt"]
                    atomic_json(durable / "request.json", read_json(directory / "request.json"))
                    atomic_json(durable / "result.json", result)
                    certificate = {"result_hash": stable_hash(result), "source": request["source"],
                                "runtime": request.get("runtime"), "local_artifact_root": str(directory),
                                "portability": "local_artifacts_required" if any(r.get("evidence", {}).get("artifact_root") for r in rows) else "embedded"}
                    atomic_json(durable / "evidence.json", certificate)
                    from .observations import coverage_export
                    # A separate compact export can point to the final result
                    # hash without a circular self-reference in that result.
                    try:
                        exported = coverage_export(read_json(directory / "request.json"), result, certificate)
                    except Exception as exc:  # noqa: BLE001 -- an optional export cannot block certified collection
                        exported = {"schema": "forge_observed_coverage_v1", "records": [],
                                    "gaps": {task_id: [f"export_unavailable: {type(exc).__name__}: {exc}"]}}
                    atomic_json(durable / "coverage.json", exported)
                job.update(status="terminal", result=result, charged_seconds=job.get("charged_seconds", 0) + elapsed, reserved_seconds=0)
                touched_requests.update(job.get("subscribers", []))
                touched_requests.add(owner["request"])
                completions += 1
                self.event(state, "completed", candidate=request["candidate"]["id"], campaign=owner["campaign"],
                           attempt=worker["attempt"], worker=worker["attempt"], gpu=worker["device"], task=task_id,
                           verdicts={r["task_id"]: r["gate_status"] for r in rows},
                           cost={"wall_seconds": elapsed}, metrics={r["task_id"]: r.get("metrics", {}) for r in rows},
                           reason={r["task_id"]: r.get("reason") for r in rows})
            self._refresh(state)
            if completions:
                tier_completed = (not touched_requests or any(
                    self._active_tier(state, state["submissions"][req_id]) != tiers_before.get(req_id)
                    for req_id in touched_requests if req_id in state["submissions"]
                ))
        if tier_completed and self.on_completion:
            self.on_completion()
        return completions

    def _collect_progress(self, state, job):
        worker = job["worker"]
        path = Path(worker["directory"]) / "run.log"
        if not path.exists():
            return
        owner = job["cost_owner"]
        request = state["submissions"][owner["request"]]["request"]
        tier = next(a["qualification_tier"] for a in request["view"]["assignments"] if a["task"] == job["definition"]["task_id"])
        with path.open("rb") as stream:
            stream.seek(worker.get("log_offset", 0))
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    worker["log_offset"] = offset
                    break
                worker["log_offset"] = stream.tell()
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(row, dict) or "event" not in row:
                    continue
                event = str(row.pop("event"))
                row.update(campaign=owner["campaign"], candidate=request["candidate"]["id"],
                           revision=request["candidate_revision"], attempt=worker["attempt"],
                           worker=worker["attempt"], gpu=worker["device"], tier=tier,
                           log_offset=offset, log=str(path))
                row.setdefault("task", job["definition"]["task_id"])
                row.setdefault("cost", {"elapsed_seconds": time.time() - worker["started_at"]})
                self.event(state, "worker_" + event, **row)

    def flush_events(self):
        """Only the drain owner calls this; durable IDs permit crash deduplication."""
        with self.state() as state:
            central = self.root / "events.jsonl"
            seen_by_path = {}
            for event in state["events"]:
                event_id = stable_hash(event)
                row = {**event, "event_id": event_id}
                targets = [central]
                if event.get("campaign"):
                    targets.append(self.root / event["campaign"] / "progress.jsonl")
                    if event.get("attempt"):
                        targets.append(self.root / event["campaign"] / event["attempt"] / "events.jsonl")
                for target in targets:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target not in seen_by_path:
                        seen_by_path[target] = _event_ids(target)
                    if event_id in seen_by_path[target]:
                        continue
                    with target.open("a") as stream:
                        stream.write(canonical(row) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    seen_by_path[target].add(event_id)
            state["events"] = []
            atomic_json(self.root / "status.json", {"updated_at": utc_now(), "campaigns": state["campaigns"],
                "submissions": {k: {**{f: v[f] for f in ("status", "lifecycle", "reason")},
                                     "execution_policy": policy(v["request"])}
                                for k, v in state["submissions"].items()},
                "running": [j["worker"] for j in state["jobs"].values() if j["status"] == "running"]})

    def pause(self, campaign_id: str, paused: bool):
        with self.state() as state:
            state["campaigns"][campaign_id]["paused"] = paused
            self.event(state, "paused" if paused else "resumed", campaign=campaign_id)

    def cancel(self, request_id: str):
        with self.state() as state:
            entry = state["submissions"][request_id]
            # Withdraw before selecting an alternative payer/subscriber.
            entry.update(status="cancelled", lifecycle="awaiting_readout", reason="cancelled by request")
            for key, job in state["jobs"].items():
                if request_id not in job["subscribers"]:
                    continue
                job["subscribers"].remove(request_id)
                if job["status"] != "running":
                    continue
                active = [r for r in job["subscribers"] if state["submissions"][r]["status"] in {"queued", "running", "paused"}
                          and self._authorized(state["submissions"][r]["request"], job["definition"])]
                owner = job["cost_owner"]
                # Owner keeps the already authorized task reservation for an
                # active subscriber in its same campaign. Across campaigns the
                # payer must explicitly opt in to transfer via campaign policy.
                same_campaign = [r for r in active if state["submissions"][r]["request"]["campaign_id"] == owner["campaign"]]
                if active and (owner["request"] != request_id or same_campaign):
                    if owner["request"] == request_id:
                        owner["request"] = same_campaign[0]
                    continue
                if active:
                    transferred = False
                    for successor in active:
                        req = state["submissions"][successor]["request"]
                        campaign = state["campaigns"][req["campaign_id"]]
                        if campaign["definition"].get("accept_shared_cost_transfer") and self._available_budget(state, req, job["reserved_seconds"], job_key=key, new_round_reservation=False)[0]:
                            state["campaigns"][owner["campaign"]]["reserved_seconds"] -= job["reserved_seconds"]
                            campaign["reserved_seconds"] += job["reserved_seconds"]
                            job["cost_owner"] = {"request": successor, "campaign": req["campaign_id"], "revision": req["candidate_revision"]}
                            transferred = True
                            break
                    if transferred:
                        continue
                worker = job["worker"]
                atomic_json(Path(worker["directory"]) / "cancel.json", {"reason": "no authorized subscriber cost owner", "timestamp": utc_now()})
                pid = worker.get("pid")
                if pid and process_identity(pid) == worker.get("process_identity"):
                    os.kill(pid, signal.SIGTERM)
            self.event(state, "cancelled", request=request_id, campaign=entry["request"]["campaign_id"])

    def retry(self, key: str, *, reason: str):
        if not reason.strip():
            raise ValueError("retry requires an execution-repair reason")
        with self.state() as state:
            job = state["jobs"][key]
            if job["status"] != "terminal" or job["result"]["raw"]["attempt_status"] not in {"error", "timeout", "cancelled"}:
                raise ValueError("only failed execution attempts can be retried; scientific failures need a revised idea")
            if any(r["gate_status"] == "BLOCKED" for r in job["result"]["task_results"]):
                raise ValueError("known applicability blockers need an adapter/capability change, not retry")
            if len(job["attempts"]) >= 3:
                raise ValueError("maximum three attempts; repair the environment before a new request")
            open_subscribers, disposition_errors = [], []
            for request_id in job["subscribers"]:
                entry = state["submissions"][request_id]
                if entry["status"] == "cancelled" or not self._authorized(entry["request"], job["definition"]):
                    continue
                if self.report_root:
                    from .lifecycle import ensure_open
                    try:
                        ensure_open(self.report_root.parent.parent, entry["request"])
                    except ValueError as error:
                        disposition_errors.append(str(error))
                        continue
                open_subscribers.append(request_id)
            if not open_subscribers:
                raise ValueError("; ".join(disposition_errors) or
                                 "retry needs an open subscribed request, not a cancelled or abandoned one; enqueue the frozen request again after cancellation")
            previous = job["result"]
            job.update(status="pending", result=None, retry_of={
                "attempt_id": previous["attempt_id"], "result_hash": stable_hash(previous),
                "reason": reason, "authorized_at": utc_now(),
            })
            job["subscribers"] = open_subscribers
            for request_id in open_subscribers:
                entry = state["submissions"][request_id]
                if entry["status"] != "cancelled":
                    entry.update(status="queued", lifecycle="ready", reason=None)
            self.event(state, "retry", compatibility_key=key, reason=reason)


def _event_ids(path: Path) -> set[str]:
    """Repair only a torn final append; never hide corruption in complete rows."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        boundary = data.rfind(b"\n") + 1
        tail = data[boundary:]
        recovery = path.with_name(path.name + ".partial-" + stable_hash(tail.hex())[:12])
        recovery.write_bytes(tail)
        with path.open("r+b") as stream:
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())
        data = data[:boundary]
    try:
        return {json.loads(line)["event_id"] for line in data.splitlines()}
    except (ValueError, KeyError) as error:
        raise ValueError(f"corrupt complete event row in {path}") from error


def device_slots(devices: list[str], workers_per_gpu: int = 1, *, allow_sharing: bool = False) -> list[dict]:
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("select at least one device, without duplicates")
    if type(workers_per_gpu) is not int or workers_per_gpu < 1 or (workers_per_gpu > 1 and not allow_sharing):
        raise ValueError("GPU sharing requires --allow-sharing and explicit memory budgets")
    host = host_capacity()
    slots = []
    for device in devices:
        if device == "cpu":
            slots.append({"device": "cpu", "slot": 0, "memory_mb": host["available_memory_mb"],
                          "capacity_mb": host["memory_mb"], "model": None})
            continue
        else:
            if not device.isdigit():
                raise ValueError("GPU devices must be physical numeric indices")
            output = subprocess.check_output(["nvidia-smi", "-i", device, "--query-gpu=memory.free,memory.total,name", "--format=csv,noheader,nounits"], text=True)
            memory, total, model = [value.strip() for value in output.strip().split(",", 2)]
            free_mb = max(0, int(memory) - 512)
        for slot in range(workers_per_gpu):
            slots.append({"device": device, "slot": slot, "memory_mb": free_mb // workers_per_gpu,
                          "capacity_mb": max(0, int(total) - 512) // workers_per_gpu, "model": model,
                          "sharing": workers_per_gpu > 1})
    if "cpu" not in devices:
        slots.append({"device": "cpu", "slot": 0, "memory_mb": host["available_memory_mb"],
                      "capacity_mb": host["memory_mb"], "model": None})
    return slots


def drain(queue: Queue, devices: list[str], *, workers_per_gpu=1, allow_sharing=False,
          watch=False, campaign=None, goal=None, poll_seconds=.2, request_filter=None,
          shared_fleet_fds=()):
    # External bounded schedulers can retain ownership across this normal drain
    # and every training/evaluation child. Never infer ownership from inherited
    # environment variables or create/replace an owner's lease files.
    shared_fleet_fds = tuple(shared_fleet_fds)
    for fd in shared_fleet_fds:
        if type(fd) is not int or fd < 0:
            raise ValueError('fleet descriptor must be an open nonnegative integer')
        os.fstat(fd)
    with file_lock(queue.root / "coordinator.lock", blocking=False):
        processes = []
        try:
            while True:
                for process in processes[:]:
                    if process.poll() is not None:
                        processes.remove(process)
                queue.collect()
                slots = device_slots(devices, workers_per_gpu, allow_sharing=allow_sharing)
                while claim := queue.claim(slots, campaign_filter=campaign, goal_filter=goal,
                                           request_filter=request_filter):
                    try:
                        processes.append(queue.launch(claim, shared_fleet_fds=shared_fleet_fds))
                    except (OSError, RuntimeError) as error:
                        atomic_json(Path(claim["worker"]["directory"]) / "terminal.json", {
                            "schema_version": 1, "token": claim["worker"]["token"],
                            "attempt_status": "error", "reason": f"worker launch failed: {error}",
                            "elapsed_seconds": 0, "result": {}})
                        queue.collect()
                queue.flush_events()
                state = queue.inspect()
                active = [e for request_id, e in state["submissions"].items() if e["status"] in {"queued", "running"}
                          and (request_filter is None or request_id == request_filter)
                          and (not campaign or e["request"]["campaign_id"] == campaign)
                          and (not goal or e["request"]["view"]["goal"] == goal)]
                if not active and not watch:
                    break
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            # Workers have independent deadlines and leases. A later drain can
            # attach; leave their requests intact instead of starting duplicates.
            queue.flush_events()
            raise
