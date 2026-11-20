"""New-source grading support for the existing origin-Dirac smoke contract."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.forge import adapters, tanh_point_mass
from experiments.forge.priors import prior_policy_receipt
from experiments.forge.sampling import PUBLIC_PRIOR_CLEAN, executed_receipt
from experiments.forge.views import grade_result
from particlegan import Recipe


ROOT = Path(__file__).resolve().parents[1]


def task():
    return json.loads((ROOT / "configs/forge/tasks/zero_tanh_smoke.json").read_text())


def receipt(card, samples, *, earlier_samples=None):
    spec = card["execution"]["host_definition"]
    steps = card["execution"]["steps"]
    schedule = sorted({math.ceil(i * steps / 24) for i in range(1, 25)})
    final = tanh_point_mass.score_samples(samples, spec, steps)
    earlier = final if earlier_samples is None else tanh_point_mass.score_samples(earlier_samples, spec, 0)
    points = [{"step": step, **(final if step == steps else earlier)} for step in schedule]
    config = Recipe.from_dict({})
    prior = config.make_prior()
    return {"recipe": config.to_dict(), "evidence": {
        **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"),
        "prior_policy": prior_policy_receipt(config, prior),
        "observations": points, "live": {"step": steps, **final}}}


def test_origin_oracle_dispatches_and_does_not_invent_a_sustained_gate():
    card = task()
    original = deepcopy(card)
    raw = receipt(card, torch.zeros(4096, 2), earlier_samples=torch.full((4096, 2), .5))
    before = deepcopy(raw)
    grade = grade_result(card, raw)
    assert grade["status"] == "PASS"
    assert len(grade["evaluator_result"]["metrics"]) == 4
    assert "suffix" not in grade["reasons"][0]
    assert card == original and raw == before
    assert card["evaluation"]["thresholds"] == [["mean_abs_error", "<=", .005],
        ["rms_error", "<=", .01], ["p99_abs_error", "<=", .025], ["saturated_fraction", "==", 0.]]


@pytest.mark.parametrize("failure", ["mean_abs_error", "rms_error", "p99_abs_error", "saturated_fraction"])
def test_destructive_samples_fail_the_original_numerical_bound(failure):
    samples = torch.zeros(4096, 2)
    if failure == "mean_abs_error":
        samples.fill_(.006)
    elif failure == "rms_error":
        samples.flatten()[:32] = .2
    elif failure == "p99_abs_error":
        samples.flatten()[:164] = .03
    else:
        samples.fill_(.995)
    grade = grade_result(task(), receipt(task(), samples))
    assert grade["status"] == "FAIL"
    assert any(cell["metric"] == failure and cell["status"] == "FAIL"
               for cell in grade["evaluator_result"]["metrics"])


@pytest.mark.parametrize("missing", ["checkpoint", "budget", "sample_count", "live_metric"])
def test_missing_observation_or_final_contract_cannot_qualify(missing):
    card = task()
    raw = receipt(card, torch.zeros(4096, 2))
    if missing == "checkpoint":
        raw["evidence"]["observations"].pop(0)
    elif missing == "budget":
        raw["evidence"]["live"]["step"] -= 1
    elif missing == "sample_count":
        raw["evidence"]["live"]["sample_count"] = 32
    else:
        del raw["evidence"]["live"]["rms_error"]
    assert grade_result(card, raw)["status"] == "INCOMPLETE"


def test_nonfinite_outputs_and_unsupported_rail_host_never_pass():
    card = task()
    assert grade_result(card, receipt(card, torch.full((4096, 2), float("nan"))))["status"] == "INVALID"
    raw = receipt(card, torch.zeros(4096, 2))
    card["id"] = "zero_tanh_rail_escape"
    assert grade_result(card, raw)["status"] == "BLOCKED"


def test_prior_blocked_receipts_and_unsupported_contract_overrides_are_preserved():
    card = task()
    raw = receipt(card, torch.zeros(4096, 2))
    raw.update(gate_status="BLOCKED", reason="unsupported evaluator zero_tanh_smoke")
    before = deepcopy(raw)
    assert grade_result(card, raw)["status"] == "BLOCKED"
    assert raw == before
    del raw["gate_status"]
    for field, value in (("eval_samples", 32), ("minimum_stable_checks", 5)):
        changed = deepcopy(card)
        changed["evaluation"][field] = value
        assert grade_result(changed, raw)["status"] == "INVALID"


def test_public_one_update_adapter_is_independently_graded_end_to_end(tmp_path):
    # This software prefix preserves the public architecture/prior and scores
    # 4,096 actual outputs, but grants no 1,000-update scientific quality credit.
    card = task()
    card["execution"]["steps"] = 1
    request = {"candidate": {"recipe_overrides": {
        "input_noise_std": 0., "output_noise_std": 0., "lr": .0001}},
        "candidate_revision": "zero-tanh-grader-software", "protocol": {"seed": 0},
        "tasks": {card["id"]: card}}
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = adapters.run_task(request, {"task_id": card["id"]}, tmp_path, "cpu")
    finally:
        torch.set_num_threads(previous)
    assert raw["execution_path"] == "public_trainer"
    assert raw["evidence"]["observations"][0]["step"] == 1
    assert raw["evidence"]["live"]["sample_count"] == 4096
    grade = grade_result(card, raw)
    expected = all(raw["evidence"]["live"][name] <= threshold if operator == "<="
                   else raw["evidence"]["live"][name] == threshold
                   for name, operator, threshold in card["evaluation"]["thresholds"])
    assert grade["status"] == ("PASS" if expected else "FAIL")
    # The same one-update prefix cannot stand in for the original full budget.
    assert grade_result(task(), raw)["status"] == "INCOMPLETE"
