"""Independent acquisition, strict retention, and CUDA checkpoint contracts."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.reproducibility import reproducible_execution, construction_rng
from experiments.forge.contracts import read_json
from experiments.forge.mechanisms import NAMES
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, load_view
from experiments.forge.word_tasks import grade, schedule

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("PARTICLEGAN_TEST_CUDA_DEVICE", "cuda:0")
GOOD = dict(sample_count=1024, quality_fraction=1., modes=5, mass_tv=0.,
            reconstruction_exact=1, minimum_reconstruction_token_probability=1.)


def evidence(kind="smoke"):
    prefix, updates = (0, 20001) if kind == "smoke" else (834, 4000)
    steps = ([prefix] if prefix else []) + [prefix + step for step in schedule(updates)]
    guards = dict(all_finite=True, hooks_exercised=True, unintended_rng_deviations=0,
        optimizer_updates={role: prefix + updates for role in ("generator", "encoder", "prior", "discriminator")},
        mechanism_audit=dict(schema_version=1, mechanisms={name: dict(requested=False, enabled=False,
            calls=updates, eligible=0, applied=0) for name in NAMES}))
    confirms = [dict(step=step, metrics=deepcopy(GOOD), primary_state_sha256="a" * 64,
        confirmed_state_sha256="a" * 64, training_state_unchanged=True,
        independent_stream="eval/live/word_confirmation",
        reconstruction_stream="eval/live/word_reconstruction_confirmation") for step in steps]
    return dict(executed_updates=updates, completed_steps=prefix + updates,
        observations=[dict(step=step, **GOOD) for step in steps], confirmations=confirms, guards=guards,
        checkpoint=dict(completed_steps=834, training_state_sha256="a" * 64, selection="earliest_confirmed_passing_state"),
        continuity=dict(prefix_steps=prefix, parent_confirmed_step=prefix, restored_exactly=True,
                        history_reset=False, parent_smoke_status="PASS", same_recipe_prior_architecture=True))


def task(kind="smoke"):
    value = load_tasks(ROOT)["five_word_joint_" + kind]
    # Synthetic numerical reducer fixture under the explicit legacy contract.
    value["execution"].pop("prior_contract", None)
    value["execution"]["prior"]["learnable"] = True
    return value


def test_one_confirmed_joint_hit_passes_smoke_despite_later_collapse():
    raw = evidence()
    for row in raw["observations"][1:]:
        row["reconstruction_exact"] = 0
    result = grade(task(), raw)
    assert result["gate_status"] == "PASS"
    assert result["evaluator_result"]["confirmed_steps"] == [834]
    assert result["evaluator_result"]["endpoint_passed"] is False


@pytest.mark.parametrize("mutation,status", [
    ("short_budget", "INCOMPLETE"), ("short_optimizer", "INCOMPLETE"),
    ("missing_check", "INCOMPLETE"), ("missing_confirmation", "INCOMPLETE"),
    ("wrong_state", "INVALID"), ("same_stream", "INVALID"), ("later_checkpoint", "INVALID"),
    ("generation_only", "FAIL"), ("confirmation_fail", "FAIL")])
def test_acquisition_rejects_incomplete_unconfirmed_or_wrong_checkpoint(mutation, status):
    raw = evidence()
    if mutation == "short_budget": raw["executed_updates"] -= 1
    elif mutation == "short_optimizer": raw["guards"]["optimizer_updates"]["encoder"] -= 1
    elif mutation == "missing_check": raw["observations"].pop()
    elif mutation == "missing_confirmation": raw["confirmations"].pop()
    elif mutation == "wrong_state": raw["confirmations"][0]["confirmed_state_sha256"] = "b" * 64
    elif mutation == "same_stream": raw["confirmations"][0]["independent_stream"] = "eval/live/generated_words"
    elif mutation == "later_checkpoint": raw["checkpoint"]["completed_steps"] = 1667
    elif mutation == "generation_only":
        for row in raw["observations"]: row["reconstruction_exact"] = 0
    else:
        for row in raw["confirmations"]: row["metrics"]["mass_tv"] = .2
    assert grade(task(), raw)["gate_status"] == status


@pytest.mark.parametrize("failed", [0, 1, 12, 24])
def test_hold_requires_restored_state_and_every_continuation_check(failed):
    raw = evidence("hold")
    assert grade(task("hold"), raw)["gate_status"] == "PASS"
    raw["observations"][failed]["reconstruction_exact"] = 0
    assert grade(task("hold"), raw)["gate_status"] == "FAIL"


def test_task_policy_preserves_old_sustained_definition_and_checked_budgets():
    tasks, view = load_tasks(ROOT), load_view(ROOT, "discriminator_stability")
    old = tasks["five_word_joint_acquisition"]
    assert old["evaluation"]["kind"] == "transfer_sustained"
    assert old["evaluation"]["minimum_stable_checks"] == 5
    assignments = {row["task"]: row for row in view["assignments"]}
    assert "five_word_joint_acquisition" not in assignments
    assert assignments["five_word_joint_smoke"]["qualification_tier"] == 1
    assert assignments["five_word_joint_hold"]["qualification_tier"] == 2
    smoke, hold = task(), task("hold")
    assert smoke["execution"]["host_definition"] == hold["execution"]["host_definition"] == old["execution"]["host_definition"]
    assert smoke["evaluation"]["thresholds"] == hold["evaluation"]["thresholds"] == old["evaluation"]["thresholds"]
    assert hold["dependencies"] == [dict(task="five_word_joint_smoke", kind="checkpoint")]
    campaign = read_json(ROOT / "configs/forge/campaigns/technique-inventory-word-split-v1.json")
    # The historical campaign was reserved for view revision 9 (1,600-update ring16, 300 s);
    # revision 10 requires the 6,400-update ring16 (1,200 s) and its own adopt-20261016 campaigns.
    archived = read_json(ROOT / "configs/forge/view-history/discriminator_stability-v9.json")
    groups = {tasks[a["task"]]["execution"].get("execution_group", a["task"]):
              tasks[a["task"]]["resources"]["timeout_seconds"] for a in view["assignments"]}
    assert campaign["candidate_budget_seconds"] == 46620
    # The original campaign keeps its old allowance. New model-mode tasks add
    # an explicit reservation; they cannot silently fit the historical cap.
    assert sum(groups.values()) == 75120
    assert sum(value for key, value in groups.items() if key.startswith("precision_gaussian_")) == 900
    initial = read_json(ROOT / "reports/forge/precision-stretch/initial-view-v14.json")
    initial_keys = {tasks[a["task"]]["execution"].get("execution_group", a["task"]) for a in initial["assignments"]}
    original_groups = {key: value for key, value in groups.items() if key in initial_keys and not key.startswith("model_")}
    assert sum(original_groups.values()) == 58440


@pytest.mark.cuda
