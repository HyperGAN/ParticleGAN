"""User decisions of 2026-10-16: counted ring16 at 6,400 updates, K3P/KA2 without the
network LR horizon cap, and operational wall-clock limits that end tests on their
declared update budget. Old task, view and recipe identities stay unchanged."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.forge import wallclock as wc
from experiments.forge.contracts import read_json
from experiments.forge.trainer_families import DECLARED_REVISIONS, load_declared_revisions
from experiments.forge.views import (load_tasks, load_view, task_evaluation_fingerprint,
                                     task_execution_fingerprint, validate_view)

ROOT = Path(__file__).resolve().parents[1]
ORDINARY_VIEWS = ("adaptation", "clockfree_continuous", "discriminator_stability", "formulation_comparison",
                  "host_profile_transfer", "quality_coverage")
TIER1 = ["gaussian1d_smoke", "two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition_6400",
         "five_word_joint_smoke"]


# --- ring16 at 6,400 updates -------------------------------------------------------------------------

def test_counted_ring16_is_the_6400_update_revision_in_every_ordinary_view():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "discriminator_stability")
    assert view["revision"] == 10
    tier1 = [a["task"] for a in view["assignments"] if a["qualification_tier"] == 1 and a["importance"] == "required"]
    assert tier1 == TIER1
    validate_view(view, tasks)
    for name in ORDINARY_VIEWS:
        required = {a["task"] for a in load_view(ROOT, name)["assignments"] if a["importance"] == "required"}
        assert "ring16_acquisition" not in required
    assert tasks["ring16_acquisition_6400"]["execution"]["steps"] == 6400


def test_revision9_is_archived_and_revision10_changes_only_the_ring16_question():
    archived = read_json(ROOT / "configs/forge/view-history/discriminator_stability-v9.json")
    current = load_view(ROOT, "discriminator_stability")
    assert archived["revision"] == 9
    assert "ring16_acquisition" in {a["task"] for a in archived["assignments"]}
    projected = deepcopy(current)
    projected["revision"] = 9
    projected["policy_change_reason"] = archived["policy_change_reason"]
    for assignment in projected["assignments"]:
        if assignment["task"] == "ring16_acquisition_6400":
            assignment["task"] = "ring16_acquisition"
    assert projected == archived


def test_ring16_bounds_and_sustained_window_rule_are_unchanged():
    tasks = load_tasks(ROOT)
    old, new = tasks["ring16_acquisition"], tasks["ring16_acquisition_6400"]
    for key in ("kind", "evaluator", "thresholds", "minimum_stable_checks", "gate_policy", "sample_evaluator",
                "sampling_law", "scoring_weights", "sources", "eval_output_noise"):
        assert new["evaluation"][key] == old["evaluation"][key], key
    # R1/R2's thin pass rests on this rule: the final 5 checks must all pass. Not weakened.
    assert new["evaluation"]["minimum_stable_checks"] == 5
    assert new["evaluation"]["kind"] == "transfer_sustained"
    # Same check cadence (one check every 16.67 updates); only the number of checks grows.
    assert new["execution"]["steps"] / new["evaluation"]["observations"] == \
        old["execution"]["steps"] / old["evaluation"]["observations"]
    assert task_evaluation_fingerprint(new) != task_evaluation_fingerprint(old)  # 384 vs 96 checks
    assert task_execution_fingerprint(old) != task_execution_fingerprint(new)


def test_scoped_policy_views_keep_their_own_ring16_variants():
    # The Atlas/E22 policy-cloud views are owned by their own fill and are not changed here.
    assert load_view(ROOT, "tier1_policy_coverage")["revision"] == 1
    for name in ("tier1_policy_coverage", "policy_cloud_coverage", "policy_cloud_coverage_as_served"):
        tasks = {a["task"] for a in load_view(ROOT, name)["assignments"]}
        assert any(task.startswith("ring16_acquisition_") and task != "ring16_acquisition_6400" for task in tasks)


# --- K3P / KA2 without network_lr_horizon_cap ---------------------------------------------------------

@pytest.mark.parametrize("family", ["k3p", "ka2"])
def test_new_selected_recipe_differs_only_by_the_dropped_cap(family):
    old = read_json(ROOT / f"configs/forge/ideas/recipe-prior-refactor-{family}-v1.json")
    new = read_json(ROOT / f"configs/forge/ideas/recipe-prior-refactor-{family}-v2.json")
    assert old["recipe_overrides"]["network_lr_horizon_cap"] == 1600  # old identity retained
    assert new["recipe_overrides"]["network_lr_horizon_cap"] is None
    assert {k: v for k, v in new["recipe_overrides"].items() if k != "network_lr_horizon_cap"} == \
        {k: v for k, v in old["recipe_overrides"].items() if k != "network_lr_horizon_cap"}
    assert new["parent"] == old["id"] and new["trainer_family"] == old["trainer_family"] == family
    editorial = {"id", "parent", "changed_factors", "mechanism_rationale", "recipe_overrides"}
    assert {k: v for k, v in new.items() if k not in editorial} == {k: v for k, v in old.items() if k not in editorial}


def test_family_selection_points_at_the_new_revisions():
    revisions = load_declared_revisions(ROOT)
    assert {family: r["candidate_id"] for family, r in revisions.items()} == {
        "k3p": "recipe-prior-refactor-k3p-v2", "ka2": "recipe-prior-refactor-ka2-v2"}
    assert all(r["status"] == "awaiting_measurement" for r in revisions.values())
    pins = {pin["trainer_family"]: pin["candidate_id"]
            for pin in read_json(ROOT / "configs/forge/selections/family-current-v1.json")["selections"]}
    # The measured pins keep their exact evidence until the revision is measured and published.
    assert pins["k3p"] == "recipe-prior-refactor-k3p-v1" and pins["ka2"] == "recipe-prior-refactor-ka2-v1"


def test_declared_revision_cannot_hide_other_recipe_changes(tmp_path):
    card = read_json(ROOT / DECLARED_REVISIONS)
    tampered = deepcopy(card)
    tampered["revisions"][0]["changed_recipe_fields"] = {"lr": [0.006375, 0.006375]}
    with pytest.raises(ValueError, match="beyond the declared fields"):
        load_declared_revisions(ROOT, card=tampered)
    tampered = deepcopy(card)
    tampered["revisions"][0]["supersedes"] = "recipe-prior-refactor-ka2-v1"
    with pytest.raises(ValueError, match="pinned candidate"):
        load_declared_revisions(ROOT, card=tampered)
    tampered = deepcopy(card)
    tampered["default_adoption"] = True
    with pytest.raises(ValueError):
        load_declared_revisions(ROOT, card=tampered)


# --- operational wall-clock limits --------------------------------------------------------------------

def test_default_policy_matches_task_declarations_and_never_changes_budgets():
    tasks = load_tasks(ROOT)
    for name, entry in wc.DEFAULT_POLICY["tasks"].items():
        task = tasks[name]
        assert entry["updates"] == task["execution"]["steps"], name
        assert entry["declared_seconds"] == task["resources"]["timeout_seconds"], name
        assert entry["minimum_limit_seconds"] >= entry["declared_seconds"]
        # A solo run with the projection margin fits the operational limit with room for contention.
        assert entry["solo_seconds_per_update"] * entry["updates"] * 1.5 < entry["minimum_limit_seconds"], name
    word = wc.DEFAULT_POLICY["tasks"]["five_word_joint_smoke"]
    assert word["minimum_limit_seconds"] == 1800  # hygiene readout: about 2 x the longest completed run (847 s)
    assert 0.0686 * word["updates"] < word["minimum_limit_seconds"]  # slowest measured word recipe finishes
    assert wc.DEFAULT_POLICY["ceiling_factor"] == 3.0


def test_policy_resolution_and_opt_out(monkeypatch):
    assert wc.resolve_policy() is wc.DEFAULT_POLICY
    assert wc.resolve_policy("declared") is None and wc.resolve_policy(None) is None
    monkeypatch.setenv(wc.POLICY_ENV, "declared")
    assert wc.resolve_policy() is None
    monkeypatch.delenv(wc.POLICY_ENV)
    entry = wc.task_policy("default", "five_word_joint_smoke")
    assert entry["minimum_limit_seconds"] == 1800 and entry["ceiling_factor"] == 3.0
    assert wc.task_policy("default", "two_pole") is None  # unlisted tasks keep their declared limit
    assert wc.base_limit(900, entry) == 1800 and wc.base_limit(900, None) == 900
    assert wc.adapter_allowance(900, entry) == 5400 and wc.adapter_allowance(120, None) == 120


def test_extension_decision_guards_defects_and_hopeless_runs():
    entry = wc.task_policy("default", "grid100")
    # Contended but able to finish: 3,500 of 7,000 updates at the 7,200 s base limit (2.06 s/update).
    decided = wc.decide_extension(3600, 7200, 3500, entry)
    assert decided["decision"] == "extended" and decided["effective_limit"] == 7200 * 3.0
    assert decided["projected_seconds"] < decided["effective_limit"]
    # One update per hour (the cell-quota defect) cannot finish within the ceiling: no extension.
    decided = wc.decide_extension(3600, 7200, 2, entry)
    assert decided["decision"] == "cannot_finish_within_ceiling" and decided["effective_limit"] == 7200
    # No observation at all: a defect, not contention.
    assert wc.decide_extension(3600, 7200, 0, entry)["decision"] == "no_progress"
    # Declared-only entries never extend.
    assert wc.decide_extension(900, 900, 100, None) == {"base_limit": 900.0, "effective_limit": 900.0,
                                                         "slowdown_factor": None, "projected_seconds": None,
                                                         "decision": "declared_only"}


def test_logged_progress_counts_continuation_updates(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        '{"t": 1.0}\t' + json.dumps({"event": "observation", "step": 834, "continuation_offset": 0}),
        '{"t": 2.0}\t' + json.dumps({"event": "observation", "step": 4501, "continuation_offset": 3667}),
    ]))
    assert wc.logged_progress(log) == 3667  # hold progress, not the inherited smoke step


def test_selected_attempt_runner_uses_the_default_policy_but_not_for_policy_cloud():
    source = (ROOT / "experiments/forge/attempt_coverage.py").read_text()
    assert 'wallclock_policy="default"' in source
    assert 'if initial.get("intake") == "policy-cloud":\n        wallclock_policy = None' in source


def test_queue_drain_can_skip_the_board_compile_hook():
    text = (ROOT / "experiments/forge/__main__.py").read_text()
    assert '"--no-publish"' in text and 'getattr(args, "no_publish", False)' in text


def test_adopt_campaigns_cover_the_operational_view_budget():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "discriminator_stability")
    declared, operational = {}, {}
    for assignment in view["assignments"]:
        task = tasks[assignment["task"]]
        group = task["execution"].get("execution_group", assignment["task"])
        declared[group] = max(declared.get(group, 0), task["resources"]["timeout_seconds"])
        entry = wc.task_policy("default", assignment["task"])
        operational[group] = max(operational.get(group, 0), wc.base_limit(task["resources"]["timeout_seconds"], entry))
    assert sum(declared.values()) == 47520 and sum(operational.values()) == 59520
    for group in ("k3p", "ka2", "bcap", "bispf", "r1r2", "gan-v3"):
        campaign = read_json(ROOT / f"configs/forge/campaigns/adopt-20261016-{group}-v1.json")
        assert campaign["id"] == f"adopt-20261016-{group}-v1"
        assert campaign["candidate_budget_seconds"] >= sum(operational.values())
