"""Wall-clock limit hits are NOT MEASURED, never FAIL; the contention-aware limit is opt-in and capped."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from experiments.forge import wallclock as wc

ROOT = Path(__file__).resolve().parents[1]


def test_wallclock_hit_classification():
    assert wc.is_wallclock_hit({"status": "TIMEOUT"})
    assert wc.is_wallclock_hit({"status": "INCOMPLETE", "raw_terminal_status": "timeout",
                                "seconds": 900.2, "timeout_seconds": 900})
    # INCOMPLETE that completed every update (e.g. a missing confirmation) is not a wall-clock hit.
    assert not wc.is_wallclock_hit({"status": "INCOMPLETE", "raw_terminal_status": "completed"})
    assert not wc.is_wallclock_hit({"status": "INCOMPLETE", "raw_terminal_status": "timeout",
                                    "seconds": 500, "timeout_seconds": 900})
    assert not wc.is_wallclock_hit({"status": "FAIL", "seconds": 900, "timeout_seconds": 900})
    assert wc.outcome({"status": "TIMEOUT"}) == wc.NOT_MEASURED


def test_tally_keeps_wallclock_hits_and_their_downstream_blocks_out_of_fail():
    records = [{"status": "PASS"}, {"status": "FAIL"}, {"status": "TIMEOUT"},
               {"status": "BLOCKED", "reason": "checkpoint parent five_word_joint_smoke: TIMEOUT"},
               {"status": "BLOCKED", "reason": "constructor refused"}]
    value = wc.tally(records)
    assert value["passed"] == 1 and value["failed"] == 1
    assert value["not_measured_wallclock"] == 1 and value["not_measured_downstream"] == 1
    assert value["counts"]["BLOCKED"] == 1


def test_paired_delta_excludes_unmeasured_tasks():
    # The review's Confidence LR "+2": the baseline's smoke timed out and its hold was blocked downstream.
    tier1 = {t: "PASS" for t in ("gaussian1d_smoke", "two_pole", "unused_token_hold", "ring16_acquisition")}
    recipe = {**tier1, "five_word_joint_smoke": "PASS", "five_word_joint_hold": "PASS"}
    baseline = {**tier1, "five_word_joint_smoke": {"status": "INCOMPLETE", "raw_terminal_status": "timeout",
                                                   "seconds": 900.2, "timeout_seconds": 900},
                "five_word_joint_hold": {"status": "BLOCKED",
                                         "reason": "checkpoint parent five_word_joint_smoke: TIMEOUT/INCOMPLETE"}}
    value = wc.paired_delta(recipe, baseline)
    assert value["delta"] == 0 and value["compared"] == 4
    assert set(value["excluded"]) == {"five_word_joint_smoke", "five_word_joint_hold"}
    # A real FAIL still counts against its arm.
    value = wc.paired_delta({"two_pole": "PASS"}, {"two_pole": "FAIL"})
    assert value["delta"] == 1 and value["compared"] == 1


def test_effective_limit_is_capped_and_needs_progress():
    assert wc.effective_limit(900, 900, 11668, 0.04) == pytest.approx((900 * (900 / 11668) / 0.04, (900 / 11668) / 0.04))
    limit, factor = wc.effective_limit(900, 900, 2501, 0.04)  # 9x slower than solo
    assert limit == 2700 and factor == pytest.approx(9.0, rel=1e-3)
    assert wc.effective_limit(900, 900, 30000, 0.04) == (900.0, pytest.approx(0.75))  # faster: never shortened
    assert wc.effective_limit(3600, 3600, 1, 0.5, 3.0)[0] == 3600 * 3.0  # one step/hour still bounded
    assert wc.effective_limit(900, 900, 0, 0.04) == (900.0, None)  # no observation: defect, not contention
    with pytest.raises(ValueError):
        wc.effective_limit(900, 900, 10, 0.04, ceiling_factor=0.5)


def test_logged_progress_reads_plain_and_timestamped_logs(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        json.dumps({"event": "observation", "step": 834}),
        '{"t": 1.0}\t' + json.dumps({"event": "observation", "step": 1667}),
        "Traceback (most recent call last):",
        json.dumps({"event": "other", "step": 99999}),
    ]))
    assert wc.logged_progress(log) == 1667
    assert wc.logged_progress(tmp_path / "missing.log") == 0


def test_policy_resolution_is_opt_in():
    assert wc.policy_for(None, "five_word_joint_smoke") is None
    assert wc.policy_for({"solo_seconds_per_update": {}}, "five_word_joint_smoke") is None
    assert wc.policy_for({"solo_seconds_per_update": {"x": 0.04}}, "x") == (0.04, wc.DEFAULT_CEILING)


def _overlay(tmp_path, records):
    path = tmp_path / wc.RERUNS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "forge_wallclock_reruns_v1", "qualification_input": False,
                                "formulations_modified": False, "records": records}))
    return wc.load_reruns(tmp_path)


def test_reruns_replace_only_wallclock_hits_with_real_verdicts(tmp_path):
    reruns = _overlay(tmp_path, [
        {"key": "a", "status": "PASS", "attempt": "x", "seconds": 801.0, "updates": 20001},
        {"key": "b", "status": "TIMEOUT", "attempt": "y", "seconds": 900.3},
        {"key": "c", "status": "PASS", "attempt": "z", "seconds": 700.0},
        {"key": "d", "status": "PASS", "attempt": "w", "seconds": 760.0,
         "original_hit": {"raw_terminal_status": "timeout", "seconds": 900.2, "evidence": "e.json"}},
    ])
    reruns.update(_overlay(tmp_path, [
        {"key": "f", "status": "FAIL", "attempt": "v", "seconds": 290.0},
        {"key": "g", "status": "PASS", "attempt": "u", "seconds": 290.0}]))
    records = [{"key": "a", "status": "TIMEOUT"}, {"key": "b", "status": "TIMEOUT"},
               {"key": "c", "status": "FAIL"}, {"key": "d", "status": "INCOMPLETE", "timeout_seconds": 900},
               {"key": "e", "status": "TIMEOUT"},
               {"key": "f", "status": "BLOCKED", "reason": "checkpoint parent five_word_joint_smoke: TIMEOUT"},
               {"key": "g", "status": "BLOCKED", "reason": "constructor refused", "block_kind": "incompatible"}]
    out = wc.apply_reruns(records, reruns)
    assert [r["status"] for r in out] == ["PASS", "TIMEOUT", "FAIL", "PASS", "TIMEOUT", "FAIL", "BLOCKED"]
    assert out[0]["wallclock_rerun"]["superseded_status"] == "TIMEOUT"
    assert "wallclock_rerun" not in out[2]  # a recorded FAIL is never overwritten
    assert records[0]["status"] == "TIMEOUT"  # inputs are not mutated


def test_reruns_overlay_must_stay_diagnostic(tmp_path):
    path = tmp_path / wc.RERUNS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "forge_wallclock_reruns_v1", "qualification_input": True,
                                "formulations_modified": False, "records": []}))
    with pytest.raises(ValueError):
        wc.load_reruns(tmp_path)
    assert wc.load_reruns(tmp_path / "absent") == {}


FAKE_SOURCES = '''
import json, sys, time
def runtime_manifest():
    for step in range(1, 13):
        time.sleep(0.25)
        print(json.dumps({"event": "observation", "step": step * 10}), flush=True)
    return {}
def verify_snapshot(root, source):
    return None
'''


def _supervise(tmp_path, timeout, policy=None):
    source = tmp_path / "source"
    (source / "experiments/forge").mkdir(parents=True)
    (source / "experiments/__init__.py").write_text("")
    (source / "experiments/forge/__init__.py").write_text("")
    shutil.copy(ROOT / "experiments/forge/contracts.py", source / "experiments/forge/contracts.py")
    worker = (ROOT / "experiments/forge/worker.py").read_text()
    stop = worker[worker.index("def stop_group"):worker.index("\n\n\n", worker.index("def stop_group"))]
    (source / "experiments/forge/worker.py").write_text("import os, signal, subprocess\n\n" + stop + "\n")
    (source / "experiments/forge/sources.py").write_text(FAKE_SOURCES)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (tmp_path / "request.json").write_text(json.dumps({"source": {}, "tasks": {}}))
    spec = {"source_root": str(source), "task": "toy", "timeout_seconds": timeout,
            "request_path": str(tmp_path / "request.json"), "dependency_blockers": ["toy blocker"]}
    if policy:
        spec["wallclock_policy"] = policy
    (attempt / "request.json").write_text(json.dumps(spec))
    with (attempt / "run.log").open("w") as log:
        subprocess.run([sys.executable, "-u", str(ROOT / "experiments/forge/attempt_worker.py"), "supervise",
                        str(attempt / "request.json")], stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
    return json.loads((attempt / "result.json").read_text())


FAKE_EXECUTE = {
    "sources.py": "def runtime_manifest():\n    return {}\ndef verify_snapshot(root, source):\n    return None\n",
    "api.py": "class CapabilityError(Exception):\n    pass\n",
    "views.py": "def grade_result(task, raw, diagnostic=False):\n    return {'status': 'PASS', 'reasons': []}\n",
    "adapters.py": ("def run_task(request, job, directory, device):\n"
                    "    task = request['tasks'][job['task_id']]\n"
                    "    return {'allowance': task['resources']['timeout_seconds'], 'cost': {'completed_steps': 1}}\n"),
}


def _execute(tmp_path, policy=None):
    source = tmp_path / "source"
    (source / "experiments/forge").mkdir(parents=True)
    (source / "experiments/__init__.py").write_text("")
    (source / "experiments/forge/__init__.py").write_text("")
    shutil.copy(ROOT / "experiments/forge/contracts.py", source / "experiments/forge/contracts.py")
    for name, text in FAKE_EXECUTE.items():
        (source / "experiments/forge" / name).write_text(text)
    card = {"resources": {"timeout_seconds": 900}, "evaluation": {}}
    (tmp_path / "request.json").write_text(json.dumps({"source": {}, "tasks": {"toy": card}}))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    spec = {"source_root": str(source), "task": "toy", "timeout_seconds": 900, "device": "cpu",
            "request_path": str(tmp_path / "request.json")}
    if policy:
        spec["wallclock_policy"] = policy
    (attempt / "request.json").write_text(json.dumps(spec))
    subprocess.run([sys.executable, str(ROOT / "experiments/forge/attempt_worker.py"), "execute",
                    str(attempt / "request.json")], capture_output=True, timeout=120, check=True)
    return json.loads((attempt / "result.json").read_text()), json.loads((attempt / "raw.json").read_text())


def test_adapter_deadline_follows_the_policy_ceiling_only_when_opted_in(tmp_path):
    # Adapters stop themselves at resources.timeout_seconds; without lifting that in-process deadline a
    # supervisor extension cannot take effect (observed: "word declared task allowance exceeded" at 901 s).
    result, raw = _execute(tmp_path / "off")
    assert raw["allowance"] == 900 and "adapter_allowance_seconds" not in result
    result, raw = _execute(tmp_path / "on", {"solo_seconds_per_update": {"toy": 0.04}, "ceiling_factor": 1.5})
    assert raw["allowance"] == 1350 and result["adapter_allowance_seconds"] == 1350
    assert json.loads((tmp_path / "on/request.json").read_text())["tasks"]["toy"]["resources"]["timeout_seconds"] == 900


def test_supervisor_without_policy_keeps_the_declared_limit(tmp_path):
    result = _supervise(tmp_path, 1)
    assert result["status"] == "TIMEOUT" and result["reason"] == "original task wall-time limit exhausted"
    assert "wallclock" not in result and result["seconds"] < 2.5


def test_supervisor_extends_by_measured_slowdown(tmp_path):
    # About 40 updates/s against a 1,000 updates/s solo reference: the factor saturates the ceiling (5x).
    result = _supervise(tmp_path, 1, {"solo_seconds_per_update": {"toy": 0.001}, "ceiling_factor": 5})
    assert result["status"] == "BLOCKED"  # the toy run finished (~3 s) and reached its own grading step
    assert result["wallclock"]["effective_limit"] == 5 and result["wallclock"]["slowdown_factor"] > 5
    assert result["wallclock"]["progress_at_declared_limit"] > 0


def test_supervisor_extension_is_bounded_by_the_ceiling(tmp_path):
    result = _supervise(tmp_path, 1, {"solo_seconds_per_update": {"toy": 0.001}, "ceiling_factor": 1.5})
    assert result["status"] == "TIMEOUT" and "not measured" in result["reason"]
    assert result["wallclock"]["effective_limit"] == 1.5 and 1.4 < result["seconds"] < 3
