"""Qualification is evidence-derived, prerequisite-safe, and independent of tier labels."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

from experiments.forge.sampling import executed_receipt, PUBLIC_PRIOR_CLEAN
from experiments.forge.views import (
    grade_result, load_tasks, load_view, qualify, task_evaluation_fingerprint,
    task_execution_fingerprint, task_fingerprint, validate_view, view_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]

# Forge retains its full-component gates after the upstream v5 scorer migration.
# These literals intentionally do not inherit a mutable upstream task policy.
VECTOR_COMPONENT_BOUNDS_V1 = [["sw1_normalized", "<=", .18], ["mass_tv", "<=", .15],
                              ["hq", ">=", .85], ["component_covariance_error", "<=", .85],
                              ["component_min_eigen_ratio", ">=", .15]]
VECTOR_DISTRIBUTION_BOUNDS_V1 = [["sw1_normalized", "<=", .18], ["mean_error", "<=", .15],
                                 ["covariance_error", "<=", .45]]
VECTOR_THRESHOLDS_V1 = {
    "vector_two_broad": VECTOR_COMPONENT_BOUNDS_V1,
    "vector_unequal_mass": VECTOR_COMPONENT_BOUNDS_V1 + [["min_mass_ratio", ">=", .25]],
    "vector_unequal_width": VECTOR_COMPONENT_BOUNDS_V1,
    "vector_anisotropic": VECTOR_COMPONENT_BOUNDS_V1,
    "vector_overlap": VECTOR_DISTRIBUTION_BOUNDS_V1,
    "vector_spiral": VECTOR_DISTRIBUTION_BOUNDS_V1,
}


def simple_task(name):
    return dict(schema_version=1, id=name, adapter="test_host",
                execution=dict(initializer="deterministic_orthogonal", steps=24, produces_state=True,
                               prior=dict(kind="mog", sigma=.025, standardize=False, learnable=True)),
                evaluation=dict(kind="transfer_sustained", thresholds=[["error", "<=", 1.0]],
                                scoring_weights="live"), resources={},
                requires_capabilities=[], dependencies=[])


def simple_view(*assignments):
    return dict(schema_version=1, id="test", revision=1, goal="quality", eligibility={},
                assignments=[dict(task=n, qualification_tier=t, importance=i, order=k)
                             for k, (n, t, i) in enumerate(assignments)])


def receipt(task, *, bad_steps=(), final=.5):
    steps = task["execution"]["steps"]
    names = {r[0] for r in task["evaluation"]["thresholds"]}
    assert names == {"error"}
    observations = [dict(step=math.ceil(i * steps / 24), error=2. if i in bad_steps else .5)
                    for i in range(1, 25)]
    return dict(task_id=task["id"], compatibility_key="compatible", gate_status="PASS",
                evidence=dict(observations=observations, live=dict(error=final)),
                cost=dict(seconds=2.))


EXPLANATIONS = "reports/forge/test-dispositions-20261015/README.md"


def hold(name, *, reason="no_global_recipe", blocked_by=()):
    return dict(schema="forge_hold_v1", previous_importance="required", reason=reason, blocked_by=list(blocked_by),
                summary=f"Fixture: {name} is measured and shown, but not counted.",
                explanation=f"{EXPLANATIONS}#{name}", reentry=f"Count {name} again when one global recipe passes it.")


def removal(name, reason, counted_by=None):
    return dict(schema="forge_count_removal_v1", previous_importance="required", reason=reason,
                counted_by=counted_by, summary=f"Fixture: {name} leaves the count of this view.",
                explanation=f"{EXPLANATIONS}#{name}")


def phase_tasks():
    """A gate, then one uninterrupted run graded as a removed phase and a counted extension."""
    tasks = {n: simple_task(n) for n in ("gate", "phase", "extension")}
    for name in ("phase", "extension"):
        tasks[name]["execution"].update(execution_group="endurance", uninterrupted=True)
    tasks["phase"]["dependencies"] = [dict(task="gate", kind="gate")]
    tasks["extension"]["dependencies"] = [dict(task="phase", kind="checkpoint")]
    tasks["extension"]["execution"]["continuation_of"] = "phase"
    view = simple_view(("gate", 1, "required"), ("phase", 2, "diagnostic"), ("extension", 2, "required"))
    view["assignments"][1]["removal"] = removal("phase", "phase", "extension")
    return tasks, view


def same_run(task, **options):
    value = receipt(task, **options)
    value["evidence"]["continuity"] = dict(mode="uninterrupted", run_id="one")
    return value


def test_current_inventory_has_complete_quality_and_distinct_claim_views():
    tasks = load_tasks(ROOT)
    stability = load_view(ROOT, "discriminator_stability")
    quality = load_view(ROOT, "quality_coverage")
    assert len(quality["assignments"]) == 22
    assert sum(tasks[a["task"]]["evaluation"]["kind"] == "transfer_sustained"
               for a in quality["assignments"]) == 19
    transfer = load_view(ROOT, "host_profile_transfer")
    variant = next(a for a in transfer["assignments"] if a["task"] == "img_intensity2_residual16")
    assert variant["importance"] == "diagnostic"
    assert all(a["task"] != variant["task"] for a in stability["assignments"] + quality["assignments"])
    smoke = [a["task"] for a in stability["assignments"]
             if a["qualification_tier"] == 1 and a["importance"] == "required"]
    assert smoke[:7] == ["gaussian1d_smoke", "two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition",
                     "five_word_joint_smoke", "zero_tanh_smoke"]
    assert len(smoke) == 11 and all(tasks[name]["adapter"] == "public_api_model_toy" for name in smoke[7:])
    assert sum(tasks[n]["execution"]["steps"] for n in smoke[1:4]) == 530
    # 0.10 restores four accuracy/settling tasks; the former hold policy is archived.
    assert stability["revision"] == 17 and stability["previous_revision"]["revision"] == 16
