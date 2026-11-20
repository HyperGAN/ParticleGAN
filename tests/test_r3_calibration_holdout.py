"""Round-3 Track 4 tests: Tier-3 14k continuation & Tier-4 holdout declarations,
negative-control true-rejection sensitivity, criteria-v1 mathematical audit,
and public-default promotion blockers.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile

import pytest
import torch

import particlegan
from experiments.forge import calibration, promotion
from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.api import CapabilityError, FormulationContext
from experiments.forge.calibration_feasibility import decision_feasibility
from experiments.forge.contracts import atomic_json, file_hash, read_json, validate_idea
from experiments.forge.leaderboard_contract import family_search_budget
from experiments.forge.planning import resolution_cache, resolve_idea
from experiments.forge.studies import load_study, validate_study
from experiments.forge.vectorprofiles import STRETCH_VECTOR_HOSTS
from experiments.forge.views import grade_result, load_tasks, load_view, validate_view

ROOT = Path(__file__).resolve().parents[1]
CRITERIA_SHA256 = "6eac2531eed68e392c11a57328999ac5dce509f8fe51d23e7b9031b7a4e3f9d1"

CANDIDATE_SPECS = (
    ("empf-v8", "empf-v8"),
    ("rhbi-v12", "rhbi-v12"),
    ("rgdt-v3", "rgdt-v3"),
    ("unified-v3", "unified-v3"),
    ("t4-arch-v3", "t4-arch-v3"),
    ("negctrl", "r3-negctrl-unreg-adam"),
)

EXPECTED_T3_TASKS = [
    "two_pole",
    "unused_token_hold",
    "ae_gan_hold",
    "clockfree_audit",
    "zero_tanh_smoke",
    "zero_tanh_rail_escape",
    "five_word_joint_smoke",
    "five_word_joint_hold",
    "mode_hold",
    "ring_hold",
    "ring_extension",
    "target_shift_recovery",
    "grid100",
    "rotated100",
    "staggered100",
    "grid100_14k",
    "rotated100_14k",
    "staggered100_14k",
]

RUNNABLE_T4_TASKS = [
    "grid100_mlp16",
    "grid100_res16_tanh",
    "grid100_res16_scaled_tanh",
]


def test_default_recipe_and_frozen_criteria_preserved():
    assert particlegan.DEFAULT_RECIPE == "empf"
    assert particlegan.get_recipe().name == "empf"
    criteria_path = ROOT / "configs/forge/calibration/criteria-v1.json"
    assert file_hash(criteria_path) == CRITERIA_SHA256
    ds = load_view(ROOT, "discriminator_stability")
    assert ds["revision"] == 20
    assert ds["calibration"]["status"] == "provisional"
    cf = load_view(ROOT, "clockfree_continuous")
    assert cf["revision"] == 5
    assert cf["calibration"]["status"] == "provisional"


def test_holdout_views_negative_control_and_12_studies_resolve_ready():
    tasks = load_tasks(ROOT)
    view_t3 = load_view(ROOT, "r3-tier3-long-horizon-holdout-v1")
    view_t4 = load_view(ROOT, "r3-tier4-frontier-holdout-v1")
    validate_view(view_t3, tasks)
    validate_view(view_t4, tasks)

    assert view_t3["evidence_scope"] == "research_diagnostic"
    assert view_t4["evidence_scope"] == "research_diagnostic"
    assert view_t3["ranking"]["policy"] == "diagnostic_outcomes_only"
    assert view_t4["ranking"]["policy"] == "diagnostic_outcomes_only"
    assert [a["task"] for a in view_t3["assignments"]] == EXPECTED_T3_TASKS
    assert [a["task"] for a in view_t4["assignments"]] == (
        ["grid100", *RUNNABLE_T4_TASKS, *sorted(STRETCH_VECTOR_HOSTS)]
        if False
        else [
            "grid100",
            *RUNNABLE_T4_TASKS,
            "misgan_block_recovery",
            "misgan_block_neural_imputer",
            "ring16_bigan_continuous",
            "misgan_mcar_p50_recovery",
            "misgan_mcar_p80_anchored",
            "pacgan_unequal_mass",
            "pacgan8_grid100",
            "alice_spiral_bigan",
            "infogan_anisotropic_modes",
            "cgan_overlapping_classes",
            "cramer_energy_overlap",
            "cyclegan_unpaired_constellation",
        ]
    )
    assert set(STRETCH_VECTOR_HOSTS).issubset({a["task"] for a in view_t4["assignments"]})

    neg = read_json(ROOT / "configs/forge/ideas/r3-negctrl-unreg-adam.json")
    validate_idea(neg)
    assert neg["schema_version"] == 3
    assert "trainer_family" not in neg
    assert neg["recipe_overrides"]["reg_coeff"] == 0.0
    assert neg["recipe_overrides"]["prior_reg"] == 0.0
    assert neg["recipe_overrides"]["prior_regularizer"] == "none"
    assert neg["recipe_overrides"]["optimizer_family"] == "adam"
    # Negative control must not inflate EMPF's declared candidate card budget.
    assert family_search_budget(ROOT, "empf")["declared_candidates"] == 8

    with resolution_cache(ROOT):
        for short, cid in CANDIDATE_SPECS:
            sid_t3 = f"r3-t3-holdout-{short}-study"
            sid_t4 = f"r3-t4-holdout-{short}-study"
            st_t3 = load_study(ROOT, sid_t3)
            st_t4 = load_study(ROOT, sid_t4)
            validate_study(st_t3)
            validate_study(st_t4)
            req_t3 = resolve_idea(ROOT, cid, study=st_t3)
            req_t4 = resolve_idea(ROOT, cid, study=st_t4)
            assert req_t3["study_review"]["status"] == "READY"
            assert req_t4["study_review"]["status"] == "READY"
            assert req_t3["preflight_blockers"] == []
            assert req_t4["preflight_blockers"] == []
            assert req_t3["protocol"]["seed"] == 0
            assert req_t4["protocol"]["seed"] == 0
            # Verify grid100 shares the exact scientific compatibility_key across both holdout views.
            key_t3 = next(j["compatibility_key"] for j in req_t3["jobs"] if j["task_id"] == "grid100")
            key_t4 = next(j["compatibility_key"] for j in req_t4["jobs"] if j["task_id"] == "grid100")
            assert key_t3 == key_t4


def test_tier3_and_tier4_adapter_preflight_boundaries():
    tasks = load_tasks(ROOT)
    with resolution_cache(ROOT):
        for short, cid in CANDIDATE_SPECS:
            req_t3 = resolve_idea(ROOT, cid, study=f"r3-t3-holdout-{short}-study")
            req_t4 = resolve_idea(ROOT, cid, study=f"r3-t4-holdout-{short}-study")
            for tid in EXPECTED_T3_TASKS:
                assert req_t3["tasks"][tid]["preflight_blockers"] == [], f"{cid} unexpectedly blocked on {tid}"
            for tid in ["grid100", *RUNNABLE_T4_TASKS]:
                assert req_t4["tasks"][tid]["preflight_blockers"] == [], f"{cid} unexpectedly blocked on {tid}"
                if tid in RUNNABLE_T4_TASKS:
                    assert {"task": "grid100", "kind": "gate"} in tasks[tid]["dependencies"]
            for tid in STRETCH_VECTOR_HOSTS:
                blockers = req_t4["tasks"][tid]["preflight_blockers"]
                assert len(blockers) == 1
                assert "no registered public Forge adapter for this stretch host" in blockers[0]
                assert adapter_preflight(tasks[tid], req_t4["candidate"], root=ROOT) == blockers


def test_negative_control_semantic_rejection_clockfree_and_damage_control():
    tasks = load_tasks(ROOT)
    req = resolve_idea(
        ROOT,
        "r3-negctrl-unreg-adam",
        view_id="r3-tier3-long-horizon-holdout-v1",
        through_tier=1,
        freeze_source=False,
        execution_backend="cpu",
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        res_tp = run_task(req, {"task_id": "two_pole"}, tmp_path / "two_pole", "cpu")
        grade_tp = grade_result(tasks["two_pole"], {"task_id": "two_pole", **res_tp})
        assert res_tp["evidence"]["guards"]["all_finite"] is True
        assert res_tp["evidence"]["guards"]["hooks_exercised"] is True
        assert grade_tp["gate_status"] == "FAIL"
        assert grade_tp["metrics"]["mean_abs"] < 0.30

        res_cf = run_task(req, {"task_id": "clockfree_audit"}, tmp_path / "clockfree", "cpu")
        grade_cf = grade_result(tasks["clockfree_audit"], {"task_id": "clockfree_audit", **res_cf})
        assert grade_cf["gate_status"] == "PASS"
        assert grade_cf["metrics"]["parity_comparisons"] == 4

    # Semantic deliberate-damage control on public trainer sampling:
    # zeroing generator weights collapses pushforward spread, proving live weights are graded.
    t4_card = read_json(ROOT / "configs/forge/ideas/t4-arch-v3.json")
    ctx = FormulationContext(
        recipe_preset=None,
        recipe_overrides={**t4_card["recipe_overrides"], "z_dim": 2, "num_particles": 64, "batch_size": 32},
        prior={"kind": "mog", "sigma": 0.025, "standardize": False, "learnable": True},
        seed=0,
        requires_capabilities=t4_card["requires_capabilities"],
        initializer="deterministic_orthogonal",
    )
    from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator

    g = ctx.construct(lambda: SimpleMLPGenerator(2, 32, 3, 2), component="generator")
    d = ctx.construct(lambda: SimpleMLPDiscriminator(2, 32, 3, 2), component="discriminator")
    trainer = ctx.build_trainer(g, d, max_steps=10)
    intact_samples = trainer.sample(256, output_noise=False, generator=torch.Generator().manual_seed(0))
    intact_std = float(intact_samples.std().item())
    assert intact_std > 1e-3
    with torch.no_grad():
        for p in trainer.G.parameters():
            p.zero_()
    damaged_samples = trainer.sample(256, output_noise=False, generator=torch.Generator().manual_seed(0))
    assert float(damaged_samples.abs().max().item()) == 0.0


def test_criteria_v1_mathematical_audit_and_promotion_blockers():
    criteria = read_json(ROOT / "configs/forge/calibration/criteria-v1.json")
    initial = read_json(ROOT / "configs/forge/calibration/initial.json")
    smoke_tasks = initial["smoke_tasks"]
    ref_tasks = initial["reference_tasks"]
    assert len(smoke_tasks) == 3
    assert len(ref_tasks) == 16

    # 1. All 11 existing current-scope calibration reports are BLOCKED with 0 reference positives.
    current_reports = [
        read_json(p)
        for p in sorted((ROOT / "reports/forge/calibration").glob("*.json"))
        if read_json(p).get("evidence_scope") == "current" and "adoption" in read_json(p)
    ]
    assert len(current_reports) == 11
    for rep in current_reports:
        assert rep["adoption"] == "BLOCKED"
        assert rep["cohorts"][0]["stats"]["reference_positives"] == 0

    # 2. Verify exact complementarity vs conjunctive gap between unified-v3 (15/16) and t4-arch-v3 (13/16).
    u3 = read_json(ROOT / "reports/forge/records/readout-ca2ce5acd5856e8c69b00377.json")
    t4 = read_json(ROOT / "reports/forge/records/readout-95d61950559e6e35c77b9df3.json")
    u3_map = {r["task_id"]: r["gate_status"] for r in u3["task_results"]}
    t4_map = {r["task_id"]: r["gate_status"] for r in t4["task_results"]}
    u3_ref_fails = [t for t in ref_tasks if u3_map.get(t) != "PASS"]
    t4_ref_fails = [t for t in ref_tasks if t4_map.get(t) != "PASS"]
    assert u3_ref_fails == ["grid100"]
    assert set(t4_ref_fails) == {"unipolar", "vector_unequal_mass", "vector_anisotropic"}
    assert all(u3_map.get(t) == "PASS" or t4_map.get(t) == "PASS" for t in ref_tasks)

    # 3. Verify decision_feasibility and Clopper-Pearson bounds for (1 TA, 2 TR) vs (1 TA, 1 FA, k TR).
    feasible_min = [
        {"id": "pos", "cohort_sha256": "c", "smoke": {"decision": "PASS", "cost": {"wall_seconds": 40.0}},
         "reference": {"decision": "PASS", "cost": {"wall_seconds": 1400.0}}},
        {"id": "neg1", "cohort_sha256": "c", "smoke": {"decision": "FAIL", "cost": {"wall_seconds": 35.0}},
         "reference": {"decision": "FAIL", "cost": {"wall_seconds": 900.0}}},
        {"id": "neg2", "cohort_sha256": "c", "smoke": {"decision": "FAIL", "cost": {"wall_seconds": 35.0}},
         "reference": {"decision": "FAIL", "cost": {"wall_seconds": 900.0}}},
    ]
    assert decision_feasibility(feasible_min, criteria)["status"] == "POSSIBLE"
    cp_0_2 = calibration.clopper_pearson(0, 2)
    assert cp_0_2["estimate"] == 0.0 and cp_0_2["upper"] == pytest.approx(0.841886, abs=1e-4)

    # Adding 1 partial solver (smoke=PASS, ref=FAIL -> false_accept) makes 2 TR infeasible (1/3 > 0.10)
    # and requires >= 9 true_reject controls (1/10 <= 0.10).
    one_fa_eight_tr = [
        feasible_min[0],
        {"id": "fa", "cohort_sha256": "c", "smoke": {"decision": "PASS", "cost": {"wall_seconds": 40.0}},
         "reference": {"decision": "FAIL", "cost": {"wall_seconds": 1200.0}}},
        *[
            {"id": f"tr_{i}", "cohort_sha256": "c", "smoke": {"decision": "FAIL", "cost": {"wall_seconds": 35.0}},
             "reference": {"decision": "FAIL", "cost": {"wall_seconds": 900.0}}}
            for i in range(8)
        ],
    ]
    assert decision_feasibility(one_fa_eight_tr, criteria)["status"] == "INFEASIBLE"
    one_fa_nine_tr = [
        *one_fa_eight_tr,
        {"id": "tr_8", "cohort_sha256": "c", "smoke": {"decision": "FAIL", "cost": {"wall_seconds": 35.0}},
         "reference": {"decision": "FAIL", "cost": {"wall_seconds": 900.0}}},
    ]
    assert decision_feasibility(one_fa_nine_tr, criteria)["status"] == "POSSIBLE"

    # 4. Verify rev-19 11-task Tier-1 smoke cost ratio blocker against criteria-v1.json (<= 0.10 & <= 900s).
    ds = load_view(ROOT, "discriminator_stability")
    t1_req = [a["task"] for a in ds["assignments"] if a["qualification_tier"] == 1 and a["importance"] == "required"]
    t2_req = [a["task"] for a in ds["assignments"] if a["qualification_tier"] == 2 and a["importance"] == "required"]
    assert len(t1_req) == 11
    u3_t1_cost = sum(r["cost"]["wall_seconds"] for r in u3["task_results"] if r["task_id"] in t1_req)
    u3_t2_cost = sum(r["cost"]["wall_seconds"] for r in u3["task_results"] if r["task_id"] in t2_req and "cost" in r)
    assert u3_t1_cost > criteria["maximum_smoke_wall_seconds"]
    assert (u3_t1_cost / u3_t2_cost) > criteria["maximum_smoke_to_reference_wall_ratio"]

    # 5. Verify promotion.register deterministically blocks both:
    #    (a) schema-v1 legacy candidates lacking accepted view calibration, and
    #    (b) schema-v3 idea cards in promotion._resolve without study-free configuration/promotion resolution.
    with resolution_cache(ROOT):
        k3p_req = resolve_idea(ROOT, "k3p", view_id="discriminator_stability", through_tier=3, freeze_source=False)
        ka2_req = resolve_idea(ROOT, "ka2", view_id="discriminator_stability", through_tier=3, freeze_source=False)
        empf_req = resolve_idea(ROOT, "empf-v8", view_id="discriminator_stability", through_tier=3, freeze_source=False)
        ctrl_req = resolve_idea(ROOT, "rhbi-v12", view_id="discriminator_stability", through_tier=3, freeze_source=False)
    base_contract = {
        "schema_version": 1,
        "id": "uncalibrated-check",
        "qualification_view": "discriminator_stability",
        "candidate_revision": k3p_req["candidate_revision"],
        "seeds": [0, 1729],
        "tasks": ["two_pole"],
        "controls": [{
            "candidate_id": "ka2",
            "candidate_revision": ka2_req["candidate_revision"],
            "expected_statuses": {"two_pole": "PASS"},
        }],
        "budgets": {"task_seconds": {"two_pole": 300}, "candidate_seconds": 1200, "campaign_seconds": 2400},
        "scoring_weights": "live",
        "aggregation": "all_registered_cells",
        "acceptance": deepcopy(promotion.ACCEPTANCE),
        "no_tuning": True,
        "early_stop": "on_required_failure",
        "execution_backend": "cuda",
        "cuda_model": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        cpath = Path(tmp) / "promotion-check.json"
        atomic_json(cpath, base_contract)
        with pytest.raises(CapabilityError, match="accepted view calibration"):
            promotion.register(ROOT, "k3p", cpath)

        v3_contract = {
            **base_contract,
            "candidate_revision": empf_req["candidate_revision"],
            "controls": [{
                "candidate_id": "rhbi-v12",
                "candidate_revision": ctrl_req["candidate_revision"],
                "expected_statuses": {"two_pole": "PASS"},
            }],
        }
        atomic_json(cpath, v3_contract)
        with pytest.raises(CapabilityError, match="v3 candidate requires an explicit study"):
            promotion.register(ROOT, "empf-v8", cpath)
