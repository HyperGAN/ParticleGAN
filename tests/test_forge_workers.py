"""Exercise real supervised processes without training a model."""
from pathlib import Path
import fcntl
import os
import shutil
import subprocess
import sys
import time

import pytest

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.queue import Queue, fence_orphan_group, lease_held, process_identity
from experiments.forge.sources import inspect_source, runtime_manifest, snapshot_source
from test_forge_queue import campaign, grade, request, SLOTS


def worker_request(tmp_path, program, *, budget=3):
    req = request(tmp_path, cap=1, cost=budget)
    checkout = tmp_path / "worktree"
    target = checkout / "experiments/forge"
    target.mkdir(parents=True)
    module_root = Path(__file__).resolve().parents[1] / "experiments/forge"
    for name in ("__init__.py", "contracts.py", "sources.py", "execution_policy.py", "queue.py", "worker.py", "telemetry.py",
                 "wallclock.py"):
        shutil.copyfile(module_root / name, target / name)
    # The real Forge initializer enforces serial autograd through this public
    # dependency. Keep the process fixture source-complete without importing
    # the unrelated training package or replacing the execution policy.
    execution_package = checkout / "particlegan"
    execution_package.mkdir(exist_ok=True)
    (execution_package / "__init__.py").write_text("")
    shutil.copyfile(module_root.parents[1] / "particlegan/execution.py",
                    execution_package / "execution.py")
    (target / "runtime.py").write_text(program)
    source = inspect_source(checkout)
    source["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "queue", source))
    req.update(source=source, runtime=runtime_manifest())
    return req


def await_file(path, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if path.exists():
            return read_json(path)
        time.sleep(.02)
    raise AssertionError(f"worker did not write {path}")


def test_real_worker_records_logs_receipt_and_releases_claim(tmp_path):
    req = worker_request(tmp_path, '''
from pathlib import Path
import sys
from .contracts import atomic_json
print("fixture-worker-output", flush=True)
atomic_json(Path(sys.argv[1]).parent / "raw-result.json", {"measured": 1})
''')
    q = Queue(tmp_path / "queue", report_root=tmp_path / "reports/forge", grader=grade)
    q.submit(req, campaign())
    claim = q.claim(SLOTS)
    process = q.launch(claim)
    assert process.wait(timeout=5) == 0
    assert q.collect() == 1
    result = q.inspect()["jobs"][claim["job"]["compatibility_key"]]["result"]
    assert result["task_results"][0]["gate_status"] == "PASS"
    directory = Path(claim["worker"]["directory"])
    assert "fixture-worker-output" in (directory / "run.log").read_text()
    assert (tmp_path / "reports/forge/attempts" / claim["worker"]["attempt"] / "evidence.json").is_file()
    q.flush_events()
    lines = (q.root / "events.jsonl").read_text().splitlines()
    assert len(lines) == 3
    q.flush_events()
    assert (q.root / "events.jsonl").read_text().splitlines() == lines


SLEEP_PROGRAM = '''
from pathlib import Path
import subprocess, sys, time
from .contracts import atomic_json
from .queue import process_identity
p = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"])
atomic_json(Path(sys.argv[1]).parent / "grandchild.json", {"pid": p.pid, "identity": process_identity(p.pid)})
time.sleep(20)
'''


def test_timeout_terminates_descendants_and_retains_attempt(tmp_path):
    # Importing Torch for the real Forge execution policy precedes this
    # fixture's descendant launch. Leave startup room, then time out the
    # 20-second workload and still verify the actual descendant is terminated.
    req = worker_request(tmp_path, SLEEP_PROGRAM, budget=5)
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(req, campaign())
    claim = q.claim(SLOTS)
    process = q.launch(claim)
    directory = Path(claim["worker"]["directory"])
    grandchild = await_file(directory / "grandchild.json", timeout=10)
    assert process.wait(timeout=10) == 1
    q.collect()
    assert process_identity(grandchild["pid"]) is None
    result = q.inspect()["jobs"][claim["job"]["compatibility_key"]]["result"]
    assert result["raw"]["attempt_status"] == "timeout"
    assert result["task_results"][0]["gate_status"] == "INCOMPLETE"
    assert q.claim(SLOTS) is None


def test_recovery_attaches_to_live_lease_and_cancel_kills_group(tmp_path):
    req = worker_request(tmp_path, SLEEP_PROGRAM)
    q = Queue(tmp_path / "queue", grader=grade)
    entry = q.submit(req, campaign())
    claim = q.claim(SLOTS)
    process = q.launch(claim)
    grandchild = await_file(Path(claim["worker"]["directory"]) / "grandchild.json")
    recovered = Queue(q.root, grader=grade)
    assert recovered.collect() == 0
    assert recovered.claim(SLOTS) is None
    recovered.cancel(entry["request"]["request_id"])
    assert process.wait(timeout=5) == 1
    recovered.collect()
    assert process_identity(grandchild["pid"]) is None
    assert read_json(Path(claim["worker"]["directory"]) / "terminal.json")["attempt_status"] == "cancelled"


def test_cancel_one_subscriber_keeps_shared_work(tmp_path):
    req = worker_request(tmp_path, '''
from pathlib import Path
import sys, time
from .contracts import atomic_json
time.sleep(.4)
atomic_json(Path(sys.argv[1]).parent / "raw-result.json", {"measured": 1})
''')
    q = Queue(tmp_path / "queue", grader=grade)
    first = q.submit(req, campaign())
    req = {**req, "candidate": {"id": "another-subscriber"}}
    second = q.submit(req, campaign())
    claim = q.claim(SLOTS)
    process = q.launch(claim)
    q.cancel(first["request"]["request_id"])
    assert process.wait(timeout=5) == 0
    q.collect()
    state = q.inspect()
    assert state["submissions"][first["request"]["request_id"]]["status"] == "cancelled"
    assert state["submissions"][second["request"]["request_id"]]["status"] == "completed"
    assert state["campaigns"]["pilot"]["spent_seconds"] > 0


def test_orphaned_descendant_is_fenced_after_session_leader_dies(tmp_path):
    script = '''
import json, os, sys, time
child = os.fork()
if child:
    with open(sys.argv[1] + ".tmp", "w") as stream:
        stream.write(json.dumps({"pid": child}))
    os.replace(sys.argv[1] + ".tmp", sys.argv[1])
    os._exit(0)
time.sleep(20)
'''
    lease_path = tmp_path / "execution.lock"
    with lease_path.open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path / "orphan.json")],
                                   start_new_session=True, pass_fds=(lease.fileno(),))
        identity = process_identity(process.pid)
        descendant = await_file(tmp_path / "orphan.json")
        assert process.wait(timeout=2) == 0
    assert process_identity(process.pid) is None
    assert lease_held(lease_path)
    assert fence_orphan_group({"pid": process.pid, "process_identity": identity}, tmp_path)
    until = time.monotonic() + 2
    while process_identity(descendant["pid"]) and time.monotonic() < until:
        time.sleep(.01)
    assert process_identity(descendant["pid"]) is None
    assert not lease_held(lease_path)


OBSERVING_PROGRAM = '''
from pathlib import Path
import json, sys, time
from .contracts import atomic_json
for step in range(1, {steps} + 1):
    time.sleep(0.2)
    print(json.dumps({{"event": "observation", "step": step}}), flush=True)
atomic_json(Path(sys.argv[1]).parent / "raw-result.json", {{"measured": 1}})
'''


def _launch_with_policy(tmp_path, steps, policy, budget):
    req = worker_request(tmp_path, OBSERVING_PROGRAM.format(steps=steps), budget=budget)
    q = Queue(tmp_path / "queue", grader=grade)
    q.submit(req, campaign())
    claim = q.claim(SLOTS)
    path = Path(claim["worker"]["directory"]) / "request.json"
    atomic_json(path, {**read_json(path), "wallclock_policy": policy})
    process = q.launch(claim)
    return q, claim, process


def test_queue_worker_extends_a_contended_run_to_its_update_budget(tmp_path):
    # About 5 updates/s against a 1,000 updates/s solo reference: one extension to the 4x ceiling.
    q, claim, process = _launch_with_policy(
        tmp_path, 40, {"solo_seconds_per_update": {"t1": 0.001}, "ceiling_factor": 4}, budget=4)
    assert process.wait(timeout=40) == 0
    q.collect()
    result = q.inspect()["jobs"][claim["job"]["compatibility_key"]]["result"]
    assert result["raw"]["attempt_status"] == "completed"
    assert result["raw"]["wallclock"]["decision"] == "extended"
    assert result["raw"]["wallclock"]["effective_limit"] == 16
    assert result["task_results"][0]["gate_status"] == "PASS"


def test_queue_worker_reports_a_wallclock_hit_as_not_measured(tmp_path):
    q, claim, process = _launch_with_policy(
        tmp_path, 400, {"solo_seconds_per_update": {"t1": 0.001}, "ceiling_factor": 1.5}, budget=4)
    assert process.wait(timeout=30) == 1
    q.collect()
    result = q.inspect()["jobs"][claim["job"]["compatibility_key"]]["result"]
    assert result["raw"]["attempt_status"] == "timeout" and result["raw"]["outcome"] == "NOT_MEASURED_WALLCLOCK"
    row = result["task_results"][0]
    assert row["gate_status"] == "INCOMPLETE" and row["wallclock_outcome"] == "NOT_MEASURED_WALLCLOCK"
    assert row["wallclock"]["effective_limit"] == 6 and "not measured" in row["reason"]
