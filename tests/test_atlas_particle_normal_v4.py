"""Regression tests for atlas-particle-normal-v4 and LossSurprise critic-only witness guarding."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from experiments.forge import knowledge
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import read_json, validate_idea
from experiments.forge.planning import candidate_revision_for, load_idea, resolve_idea
from experiments.forge.studies import load_study
from particlegan.continuous import LossSurprise, SettledReopenGuard


ROOT = Path(__file__).resolve().parents[1]


def _seed_calm_detector(detector: LossSurprise, guard: SettledReopenGuard, steps: int = 80) -> float:
    for step in range(steps):
        loss = 0.5 + 0.02 * math.sin(step)
        detector.observe(loss)
        assert detector.decide(step, guard=guard) is False
    return detector.calm + 8.0 * math.sqrt(detector.var_calm)


def test_loss_surprise_blocks_pre_anchor_critic_only_witness_but_permits_post_anchor_and_other_witnesses():
    # 1. Pre-anchor critic-only witness (e.g. mode_hold step 35: ka2_anchor_started=False,
    # s_D=0.5, s_G=1.0, s_table=1.0): must NOT fire a reopen and must fold observations into calm.
    critic_only_detector = LossSurprise()
    critic_only_guard = SettledReopenGuard()
    critic_only_guard.ka2_anchor_started = False
    jump_loss = _seed_calm_detector(critic_only_detector, critic_only_guard)
    critic_only_guard.calm_network = {"1.0": {"role": "critic", "scale": 0.5}}
    calm_before = critic_only_detector.calm

    for offset in range(5):
        critic_only_detector.observe(jump_loss)
        assert critic_only_detector.decide(100 + offset, guard=critic_only_guard) is False
        assert critic_only_detector.streak == 0
        assert critic_only_detector.fires == 0
    assert critic_only_detector.calm > calm_before

    # 2. Post-anchor critic-only witness (ka2_anchor_started=True, e.g. settled equilibrium shift
    # when only the critic ladder contracted): must accumulate confirmation streak and fire on step 2.
    post_anchor_detector = LossSurprise()
    post_anchor_guard = SettledReopenGuard()
    post_anchor_guard.ka2_anchor_started = True
    jump_loss_post = _seed_calm_detector(post_anchor_detector, post_anchor_guard)
    post_anchor_guard.calm_network = {"1.0": {"role": "critic", "scale": 0.5}}

    post_anchor_detector.observe(jump_loss_post)
    assert post_anchor_detector.decide(100, guard=post_anchor_guard) is False
    assert post_anchor_detector.streak == 1

    post_anchor_detector.observe(jump_loss_post)
    assert post_anchor_detector.decide(101, guard=post_anchor_guard) is True
    assert post_anchor_detector.fires == 1
    assert post_anchor_guard.calm_network == {}

    # 3. Empty network witness (e.g. cold-start img_blobs4_residual16 / img_intensity2 at s=1.0):
    # must accumulate confirmation streak and fire on step 2 even while ka2_anchor_started=False.
    empty_detector = LossSurprise()
    empty_guard = SettledReopenGuard()
    empty_guard.ka2_anchor_started = False
    jump_loss_empty = _seed_calm_detector(empty_detector, empty_guard)
    assert empty_guard.calm_network == {}

    empty_detector.observe(jump_loss_empty)
    assert empty_detector.decide(100, guard=empty_guard) is False
    assert empty_detector.streak == 1

    empty_detector.observe(jump_loss_empty)
    assert empty_detector.decide(101, guard=empty_guard) is True
    assert empty_detector.fires == 1
    assert empty_detector.armed is False

    # 4. Settled generator + critic witness (post-settling abrupt distribution shift):
    # must accumulate confirmation streak and fire on step 2, clearing guard witnesses.
    settled_detector = LossSurprise()
    settled_guard = SettledReopenGuard()
    settled_guard.ka2_anchor_started = True
    jump_loss_settled = _seed_calm_detector(settled_detector, settled_guard)
    settled_guard.calm_network = {
        "0.0": {"role": "generator", "scale": 0.5},
        "1.0": {"role": "critic", "scale": 0.5},
    }

    settled_detector.observe(jump_loss_settled)
    assert settled_detector.decide(100, guard=settled_guard) is False
    assert settled_detector.streak == 1

    settled_detector.observe(jump_loss_settled)
    assert settled_detector.decide(101, guard=settled_guard) is True
    assert settled_detector.fires == 1
    assert settled_guard.calm_network == {}
    assert settled_guard.excursion is None



def test_atlas_particle_normal_v4_idea_and_study_contracts():
    idea = load_idea(ROOT, "atlas-particle-normal-v4")
    validate_idea(idea)
    assert idea["id"] == "atlas-particle-normal-v4"
    assert idea["parent"] == "atlas-particle-normal-v3"
    assert idea["recipe_preset"] == "atlas"
    assert idea["api_changes"] == [
        "particlegan/__init__.py",
        "particlegan/birth_death.py",
        "particlegan/continuous.py",
        "particlegan/feature_cells.py",
        "particlegan/feature_policy.py",
        "particlegan/ka2.py",
        "particlegan/policy.py",
        "particlegan/priors.py",
        "particlegan/recipes.py",
        "particlegan/training.py",
    ]
    assert idea["recipe_overrides"] == {
        "birth_death_cells": 256,
        "d_guard_min_steps": 0,
        "d_guard_ratio": 0.0,
        "output_noise_mode": "fixed",
        "output_noise_std": 0.0,
    }
    assert idea["prior_mechanism"] == {
        "kind": "particles",
        "manipulators": [{"kind": "normal_noise", "sigma": 0.0065}],
        "standardize": False,
    }

    study = load_study(ROOT, "atlas-particle-normal-replication-study-v4-reverified")
    assert study["candidate"] == "atlas-particle-normal-v4"
    assert study["control"]["candidate_id"] == "atlas-particle-normal-v2"
    assert study["prediction"]["task_id"] == "mode_hold"
    assert study["prediction"]["metric"] == "modes"
    assert study["prediction"]["threshold"] == 8
    assert study["scope"]["view"] == "prior-ablation-full-suite-v1"
    assert study["scope"]["execution_backend"] == "cuda"
    assert study["scope"]["through_tier"] == 1


def test_atlas_particle_normal_v4_readout_recovers_mode_hold_with_zero_regressions():
    idea = load_idea(ROOT, "atlas-particle-normal-v4")
    view = json.loads((ROOT / "configs/forge/views/prior-ablation-full-suite-v1.json").read_text())
    assert len(view["assignments"]) == 48
    for assignment in view["assignments"]:
        task = json.loads((ROOT / f"configs/forge/tasks/{assignment['task']}.json").read_text())
        ctx = task_formulation_context(idea, task)
        assert ctx.recipe is not None
        assert ctx.prior_config is not None

    v4_records = [
        read_json(path)
        for path in sorted((ROOT / "reports/forge/records").glob("readout-*.json"))
        if read_json(path).get("candidate_id") == "atlas-particle-normal-v4"
        and read_json(path).get("study_id") == "atlas-particle-normal-replication-study-v4-reverified"
    ]
    assert len(v4_records) == 1, f"Expected exactly 1 v4 full-suite readout record, found {len(v4_records)}"
    v4_readout = v4_records[0]

    v4_ordinary_records = [
        read_json(path)
        for path in sorted((ROOT / "reports/forge/records").glob("readout-*.json"))
        if read_json(path).get("candidate_id") == "atlas-particle-normal-v4"
        and read_json(path).get("study_id") == "atlas-particle-normal-v4-ordinary"
    ]
    assert len(v4_ordinary_records) == 1, (
        f"Expected exactly 1 v4 ordinary discriminator_stability readout, found {len(v4_ordinary_records)}"
    )
    v4_ordinary = v4_ordinary_records[0]
    assert all(item["valid_receipt"] for item in v4_ordinary["provenance"]["attempts"])
    ordinary_counts = Counter(row["gate_status"] for row in v4_ordinary["task_results"])
    assert len(v4_ordinary["task_results"]) == 21
    assert ordinary_counts["INCOMPLETE"] == 0
    assert ordinary_counts["PASS"] == 21
    assert ordinary_counts["FAIL"] == 0

    selection = read_json(ROOT / "configs/forge/selections/family-current-v1.json")
    atlas_pin = next(pin for pin in selection["selections"] if pin["trainer_family"] == "atlas")
    assert atlas_pin["candidate_id"] == "atlas-particle-normal-v4"
    assert atlas_pin["execution_backend"] == "cuda"

    inventory = read_json(ROOT / "reports/forge/technique-inventory.json")
    atlas_row = next(row for row in inventory["rows"] if row["trainer_family"] == "atlas")
    assert atlas_row["candidate_id"] == "atlas-particle-normal-v4"
    published_tasks = {
        item["task_id"]: item["status"]
        for key in ("tasks", "nonrequired_tasks")
        for item in atlas_row[key]
    }
    assert "INVALID" not in published_tasks.values()
    assert "ERROR" not in published_tasks.values()
    assert published_tasks["clockfree_audit_measurement_v1"] == "PASS"
    resolved_tasks = {
        "two_pole",
        "ring16_acquisition",
        "five_word_joint_smoke",
        "model_ae_gan_multimodal",
        "model_ae_ddgan_gaussian_chain",
        "model_ddgan_gaussian_chain",
        "model_ddgan_bimodal_chain",
        "model_ae_ddgan_bimodal_chain",
        "grid100",
        "rotated100",
        "staggered100",
        "mode_hold",
        "five_word_joint_hold",
    }
    for item in v4_ordinary["task_results"]:
        if item["task_id"] in resolved_tasks:
            assert item["gate_status"] == "PASS"
        else:
            assert published_tasks[item["task_id"]] == item["gate_status"]

    inventory_md = (ROOT / "reports/forge/technique-inventory.md").read_text()
    atlas_top_line = next(
        line
        for line in inventory_md.splitlines()
        if line.startswith(("| **[Atlas](families/atlas.md)** |", "| **[Atlas](families/atlas.md)**<br>"))
    )
    # The overview projects the certified base source cohort; the later
    # source-specific readout above retains its own unchanged measurements.
    from reports.forge import regenerate_technique_inventory as rti
    display = rti._formulaic_display_payload(inventory, ROOT)
    assert display is not None
    expected = "[18/27]" if display["scope"] == "recorded_source_formulaic_display" else "[27/27]"
    assert expected in atlas_top_line
    assert "invalid" not in atlas_top_line.lower()

    v3_readout = read_json(ROOT / "reports/forge/records/readout-efb8bacf93e19fc2b0d7f7bf.json")
    v2_readout = read_json(ROOT / "reports/forge/records/readout-4dc982d111099c9b0f33a18e.json")
    bispf_readout = read_json(ROOT / "reports/forge/records/readout-7b017c8ed900b677ebf6bf55.json")
    v1_readout = read_json(ROOT / "reports/forge/records/readout-9f0d62f87d1ff351cfef7f98.json")

    assert v4_readout["candidate_id"] == "atlas-particle-normal-v4"
    assert v4_readout["study_id"] == "atlas-particle-normal-replication-study-v4-reverified"
    assert all(item["valid_receipt"] for item in v4_readout["provenance"]["attempts"])

    resolved = resolve_idea(
        ROOT,
        "atlas-particle-normal-v4",
        study="atlas-particle-normal-replication-study-v4-reverified",
    )
    assert (
        candidate_revision_for(resolved["source"]["digest"], resolved["candidate"])
        == resolved["candidate_revision"]
    )
    assert (
        candidate_revision_for(atlas_row["bindings"]["source_digest"], resolved["candidate"])
        == v4_readout["candidate_revision"]
    )

    decision = v4_readout["decision_outcomes"][0]
    assert decision["binding_errors"] == []
    assert decision["observed"]["prediction"]["satisfied"] is True
    assert decision["observed"]["prediction"]["value"] >= 8
    assert decision["observed"]["falsifier"]["satisfied"] is False

    v4_by_task = {row["task_id"]: row["gate_status"] for row in v4_readout["task_results"]}
    v3_by_task = {row["task_id"]: row["gate_status"] for row in v3_readout["task_results"]}
    v2_by_task = {row["task_id"]: row["gate_status"] for row in v2_readout["task_results"]}

    v4_counts = Counter(v4_by_task.values())
    v3_counts = Counter(v3_by_task.values())
    v2_counts = Counter(v2_by_task.values())
    bispf_counts = Counter(row["gate_status"] for row in bispf_readout["task_results"])
    v1_counts = Counter(row["gate_status"] for row in v1_readout["task_results"])

    assert v4_counts["INCOMPLETE"] == 0
    assert len(v4_by_task) == 38 > len(v3_by_task) == 35
    assert v4_counts["PASS"] >= 24
    assert v4_counts["PASS"] > v3_counts["PASS"] == 23
    assert v4_counts["PASS"] > v2_counts["PASS"] == 22
    assert v4_counts["PASS"] > bispf_counts["PASS"] == 20
    assert v4_counts["PASS"] > v1_counts["PASS"] == 17

    # Zero regressions relative to BOTH v2 and v3 (including mode_hold + all 6 vector_* + all 4 img_*).
    prior_passes = {tid for tid, status in v2_by_task.items() if status == "PASS"} | {
        tid for tid, status in v3_by_task.items() if status == "PASS"
    }
    assert len(prior_passes) == 24
    for expected_pass in sorted(prior_passes):
        assert v4_by_task.get(expected_pass) == "PASS", (
            f"Regression on {expected_pass}: expected PASS in v4, got {v4_by_task.get(expected_pass)}"
        )

    # Confirm mode_hold unblocked its 3 downstream dependents so they executed rather than deferring.
    for unblocked_dependent in ("ring_hold", "ring_extension", "target_shift_recovery"):
        assert unblocked_dependent in v4_by_task


def test_v4_compilation_manifest_is_fresh_and_clean_checkout_safe():
    manifest = read_json(ROOT / "reports/forge/compilation.json")
    assert not any(
        key.startswith("reports/forge/attempts/") for key in manifest["input_hashes"]
    ), "compilation.json must not depend on gitignored reports/forge/attempts/ files"
    status = knowledge.freshness(ROOT)
    assert status["fresh"], status
