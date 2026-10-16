"""Reference recipes delegate only explicitly declared, task-owned fields."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.forge.adapters import _context, adapter_preflight
from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.contracts import validate_idea
from experiments.forge.hostprofiles import _validate_candidate_identity, _validate_task
from experiments.forge.planning import candidate_revision_for, resolve_idea
from experiments.forge.taskrecipes import (adaptation_receipt, bind_task_candidate,
                                          delegated_fields)


ROOT = Path(__file__).resolve().parents[1]
NAME = "release07-gan-v3-task-adapted-v1"


def card():
    return json.loads((ROOT / f"configs/forge/ideas/{NAME}.json").read_text())


def task(name):
    value = json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())
    # Explicit legacy cohort: archived task-owned policy and host penalties.
    value["execution"].pop("prior_contract", None)
    value["execution"]["prior"]["learnable"] = True
    return value


def test_original_24_task_adaptations_stay_ready_and_original_reference_stays_blocked():
    original = resolve_idea(ROOT, "release07-gan-v3-mog-v1", through_tier=3)
    adapted = resolve_idea(ROOT, NAME, through_tier=3)
    assert not adapted["preflight_blockers"]
    # The recorded integration covered these 24 hosts. New required smoke hosts
    # retain their own compatibility checks; they do not inherit that readiness.
    additions = {"gaussian1d_smoke", "gaussian1d_stability", "ring16_acquisition", "five_word_joint_smoke",
                 "five_word_joint_hold", "clockfree_audit_measurement_v1", "zero_tanh_smoke",
                 "zero_tanh_rail_escape", "grid100_mlp16", "grid100_res16_tanh", "misgan_block_recovery",
                 "misgan_block_neural_imputer", "ring16_bigan_continuous"}
    original_view = json.loads((ROOT / "reports/forge/precision-stretch/initial-view-v14.json").read_text())
    original_hosts = set(adapted["tasks"]) & {a["task"] for a in original_view["assignments"]}
    original_hosts -= additions
    original_hosts -= {name for name, task in adapted["tasks"].items() if task["adapter"] == "public_api_model_toy"}
    original_hosts -= {name for name, task in adapted["tasks"].items() if "linear_precision" in task["execution"]}
