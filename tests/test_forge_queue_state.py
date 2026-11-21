"""Submission projections must not stand in for execution or cost ownership."""
from copy import deepcopy

import pytest
from test_forge_queue import SLOTS, campaign, finish, grade, request

from experiments.forge.queue import Queue, _submission_projection


@pytest.mark.parametrize("lifecycle", ["ready", "running", "awaiting_readout"])
@pytest.mark.parametrize("eligible,reason,running,paused,status,next_lifecycle,next_reason", [
    (False, None, False, False, "completed", "awaiting_readout", None),
    (False, None, False, True, "completed", "awaiting_readout", None),
    (False, None, True, False, "running", "running", None),
    (False, None, True, True, "running", "running", None),
    (True, None, False, False, "queued", None, None),
    (True, None, False, True, "paused", None, "campaign paused"),
    (True, None, True, False, "running", "running", None),
    (True, None, True, True, "running", "running", None),
    (False, "gate failed", False, False, "blocked", "awaiting_readout", "gate failed"),
    (False, "gate failed", False, True, "blocked", "awaiting_readout", "gate failed"),
    (False, "gate failed", True, False, "blocked", "awaiting_readout", "gate failed"),
    (False, "gate failed", True, True, "blocked", "awaiting_readout", "gate failed"),
    (True, "gate failed", False, False, "blocked", "awaiting_readout", "gate failed"),
    (True, "gate failed", False, True, "blocked", "awaiting_readout", "gate failed"),
    (True, "gate failed", True, False, "blocked", "awaiting_readout", "gate failed"),
    (True, "gate failed", True, True, "blocked", "awaiting_readout", "gate failed"),
])
def test_submission_projection_precedence(lifecycle, eligible, reason, running, paused,
                                          status, next_lifecycle, next_reason):
    assert _submission_projection(lifecycle=lifecycle, eligible=eligible, reason=reason,
                                  running=running, paused=paused) == {
        "status": status, "lifecycle": next_lifecycle or lifecycle, "reason": next_reason,
    }


@pytest.mark.parametrize("status", ["cancelled", "concluded"])
def test_refresh_preserves_explicit_closure_without_evaluating_jobs(tmp_path, monkeypatch, status):
    queue = Queue(tmp_path)
    state = {"submissions": {"closed": {
        "status": status, "lifecycle": "awaiting_readout", "reason": "explicit disposition",
    }}}
    before = deepcopy(state)

    def unexpected_eligibility(*_):
        raise AssertionError("closed submission must not be scheduled")

    monkeypatch.setattr(queue, "_eligible", unexpected_eligibility)
    queue._refresh(state)
    assert state == before


def test_pause_blocks_new_claims_while_live_work_and_cost_remain_owned(tmp_path):
    queue = Queue(tmp_path / "queue", grader=grade)
    entry = queue.submit(request(tmp_path, cap=2), campaign())
    request_id = entry["request"]["request_id"]
    first = queue.claim(SLOTS)
    queue.pause("pilot", True)
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert state["submissions"][request_id]["status"] == "running"
    assert state["campaigns"]["pilot"]["reserved_seconds"] == 10
    assert state["charges"] == []

    finish(first)
    assert queue.collect() == 1
    state = queue.inspect()
    assert {field: state["submissions"][request_id][field]
            for field in ("status", "lifecycle", "reason")} == {
        "status": "paused", "lifecycle": "running", "reason": "campaign paused",
    }
    assert state["campaigns"]["pilot"]["spent_seconds"] == 1
    assert state["campaigns"]["pilot"]["reserved_seconds"] == 0

    queue.pause("pilot", False)
    second = queue.claim(SLOTS)
    assert second["job"]["task_id"] == "t2"
    queue.pause("pilot", True)
    finish(second, measured=2)
    assert queue.collect() == 1
    state = queue.inspect()
    assert {field: state["submissions"][request_id][field]
            for field in ("status", "lifecycle", "reason")} == {
        "status": "completed", "lifecycle": "awaiting_readout", "reason": None,
    }
    assert state["campaigns"]["pilot"]["spent_seconds"] == 2
    assert state["campaigns"]["pilot"]["reserved_seconds"] == 0
    assert len(state["charges"]) == 2
    assert queue.collect() == 0
    assert queue.inspect() == state
