"""Tests for ParticleGAN N-Tier infrastructure, contiguity, importance, and reporting."""
from copy import deepcopy
import math
from pathlib import Path
import pytest

from experiments.forge.views import (
    qualify, validate_view, load_tasks, load_view, IMPORTANCES
)
from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.tier_report import (
    build_report, render_markdown, tier_name, TIER_NAMES
)
from experiments.forge.family_reports import _table_header
from experiments.forge.__main__ import parser
from experiments.forge.dispositions import _tier_counts
from experiments.forge.configuration_search import select_configuration


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Test Helpers and Fixtures
# ---------------------------------------------------------------------------

def simple_task(name):
    """Generate a minimal valid task dictionary conforming to Forge schema."""
    return dict(
        schema_version=1, id=name, adapter="test_host",
        execution=dict(
            initializer="deterministic_orthogonal", steps=24, produces_state=True,
            prior=dict(kind="mog", sigma=0.025, standardize=False, learnable=True)
        ),
        evaluation=dict(kind="transfer_sustained", thresholds=[["error", "<=", 1.0]],
                        scoring_weights="live"),
        resources={}, requires_capabilities=[], dependencies=[]
    )


def simple_view(*assignments, view_id="test_view", revision=1):
    """Generate a minimal valid view dictionary with given (task, tier, importance) tuples."""
    return dict(
        schema_version=1, id=view_id, revision=revision, goal="quality", eligibility={},
        assignments=[dict(task=n, qualification_tier=t, importance=i, order=k)
                     for k, (n, t, i) in enumerate(assignments)]
    )


def receipt(task, *, bad_steps=(), final=0.5):
    """Generate a mock evaluation receipt for a task."""
    steps = task["execution"]["steps"]
    observations = [dict(step=math.ceil(i * steps / 24), error=2.0 if i in bad_steps else 0.5)
                    for i in range(1, 25)]
    return dict(
        task_id=task["id"], compatibility_key="compatible", gate_status="PASS",
        evidence=dict(observations=observations, live=dict(error=final)),
        cost=dict(seconds=2.0)
    )


def hold(name, *, reason="no_global_recipe", blocked_by=()):
    """Generate valid forge_hold_v1 metadata."""
    return dict(
        schema="forge_hold_v1", previous_importance="required", reason=reason,
        blocked_by=list(blocked_by),
        summary=f"Fixture: {name} is measured and shown, but not counted.",
        explanation=f"reports/forge/test-dispositions-20261015/README.md#{name}",
        reentry=f"Count {name} again when one global recipe passes it."
    )


# ---------------------------------------------------------------------------
# 1. Tier Validation Tests
# ---------------------------------------------------------------------------

def test_tier_validation_accepts_tier_4_and_tier_5():
    """Verify validate_view accepts qualification_tier 4 and 5 in contiguous views."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 6)}

    # Tier 4 view
    view4 = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required")
    )
    validate_view(view4, tasks)

    # Tier 5 view
    view5 = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required"),
        ("t5", 5, "required")
    )
    validate_view(view5, tasks)


def test_tier_validation_arbitrary_positive_integer():
    """Verify qualification_tier can scale dynamically up to arbitrary N >= 1."""
    n_tiers = 10
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, n_tiers + 1)}
    assignments = [(f"t{i}", i, "required") for i in range(1, n_tiers + 1)]
    view = simple_view(*assignments)
    validate_view(view, tasks)


@pytest.mark.parametrize("invalid_tier", [
    0, -1, -5, True, False, "1", "4", 4.0, None, [1], {"tier": 1}
])
def test_tier_validation_rejects_invalid_values(invalid_tier):
    """Verify qualification_tier rejects integers < 1, booleans, and non-integer types."""
    tasks = {"t1": simple_task("t1")}
    view = simple_view(("t1", 1, "required"))
    view["assignments"][0]["qualification_tier"] = invalid_tier
    with pytest.raises(ValueError, match="qualification_tier"):
        validate_view(view, tasks)


# ---------------------------------------------------------------------------
# 2. Contiguity Enforcement Tests
# ---------------------------------------------------------------------------

def test_contiguity_passes_for_complete_contiguous_tiers():
    """Verify contiguity passes when required/held tiers span 1..H continuously."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 6)}

    # Tiers 1..4
    v4 = simple_view(("t1", 1, "required"), ("t2", 2, "required"),
                     ("t3", 3, "required"), ("t4", 4, "required"))
    validate_view(v4, tasks)

    # Tiers 1..5
    v5 = simple_view(("t1", 1, "required"), ("t2", 2, "required"),
                     ("t3", 3, "required"), ("t4", 4, "required"),
                     ("t5", 5, "required"))
    validate_view(v5, tasks)


@pytest.mark.parametrize("tiers", [
    [1, 3],              # missing tier 2
    [1, 2, 4],           # missing tier 3
    [1, 2, 3, 5],        # missing tier 4
    [2, 3, 4],           # missing tier 1
    [1, 4],              # missing tiers 2, 3
    [1, 3, 5],           # missing tiers 2, 4
])
def test_contiguity_rejects_missing_intermediate_tiers(tiers):
    """Verify contiguity raises ValueError if any tier from 1 to H is omitted."""
    tasks = {f"t{t}": simple_task(f"t{t}") for t in tiers}
    assignments = [(f"t{t}", t, "required") for t in tiers]
    view = simple_view(*assignments)
    with pytest.raises(ValueError, match="empty required tier: qualification tiers must be contiguous from Tier 1"):
        validate_view(view, tasks)


def test_contiguity_held_tasks_satisfy_tier_presence():
    """Verify held tasks (being in COUNTED_OR_HELD) satisfy tier contiguity."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 5)}
    view = simple_view(
        ("t1", 1, "required"),
        ("t2", 2, "held"),
        ("t3", 3, "required"),
        ("t4", 4, "held"),
    )
    view["assignments"][1]["hold"] = hold("t2")
    view["assignments"][3]["hold"] = hold("t4")
    # This must pass contiguity because Tiers 1, 2, 3, 4 all have counted or held tasks
    validate_view(view, tasks)


def test_contiguity_ranking_or_diagnostic_cannot_bridge_gap():
    """Verify ranking or diagnostic tasks alone cannot satisfy tier contiguity."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 4)}

    # Tier 2 has only ranking task
    view_ranking = simple_view(
        ("t1", 1, "required"),
        ("t2", 2, "ranking"),
        ("t3", 3, "required"),
    )
    with pytest.raises(ValueError, match="empty required tier"):
        validate_view(view_ranking, tasks)

    # Tier 2 has only diagnostic task
    view_diag = simple_view(
        ("t1", 1, "required"),
        ("t2", 2, "diagnostic"),
        ("t3", 3, "required"),
    )
    with pytest.raises(ValueError, match="empty required tier"):
        validate_view(view_diag, tasks)


# ---------------------------------------------------------------------------
# 3. View Importance Semantics Tests across Tiers 1-5
# ---------------------------------------------------------------------------

def test_qualification_all_required_pass_through_tier_4():
    """Verify a candidate passing required tasks in Tiers 1-4 qualifies at Tier 4."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 5)}
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required")
    )
    results = [receipt(tasks[f"t{i}"]) for i in range(1, 5)]
    res = qualify(view, tasks, results)

    assert res["status"] == "PASS"
    assert res["qualified_tier"] == 4
    assert res["eligible"] is True
    assert res["required_passed"] == 4
    assert res["required_total"] == 4
    assert res["held_tiers"] == []
    assert res["counted_tiers"] == [1, 2, 3, 4]


def test_qualification_all_required_pass_through_tier_5():
    """Verify a candidate passing required tasks in Tiers 1-5 qualifies at Tier 5."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 6)}
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required"),
        ("t5", 5, "required")
    )
    results = [receipt(tasks[f"t{i}"]) for i in range(1, 6)]
    res = qualify(view, tasks, results)

    assert res["status"] == "PASS"
    assert res["qualified_tier"] == 5
    assert res["eligible"] is True
    assert res["counted_tiers"] == [1, 2, 3, 4, 5]


def test_qualification_required_failure_stops_progression_tier_4():
    """Verify failure of a required task in Tier 4 stops progression and reports FAIL."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 5)}
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required")
    )
    # Tiers 1-3 pass, Tier 4 fails
    results = [
        receipt(tasks["t1"]), receipt(tasks["t2"]),
        receipt(tasks["t3"]), receipt(tasks["t4"], final=2.0)
    ]
    res = qualify(view, tasks, results)

    assert res["status"] == "FAIL"
    assert res["qualified_tier"] == 3
    assert res["eligible"] is False
    assert res["required_passed"] == 3
    assert res["required_total"] == 4
    assert any(b["task_id"] == "t4" for b in res["blockers"])


def test_qualification_ranking_failure_does_not_block_tier_4():
    """Verify failure of a ranking task in Tier 4 does not block qualification."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 5)}
    tasks["r4"] = simple_task("r4")
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required"),
        ("r4", 4, "ranking")
    )
    results = [
        receipt(tasks["t1"]), receipt(tasks["t2"]),
        receipt(tasks["t3"]), receipt(tasks["t4"]),
        receipt(tasks["r4"], final=2.0)  # ranking failure
    ]
    res = qualify(view, tasks, results)

    assert res["status"] == "PASS"
    assert res["qualified_tier"] == 4
    assert res["eligible"] is True
    assert res["task_statuses"]["r4"] == "FAIL"
    assert res["task_statuses"]["t4"] == "PASS"
    assert res["required_passed"] == 4
    assert res["required_total"] == 4


def test_qualification_top_tier_held_semantics_tier_4():
    """Verify that when highest assigned tier (Tier 4) contains only held tasks, view reports HELD."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 5)}
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "held")
    )
    view["assignments"][3]["hold"] = hold("t4")

    # Lower counted tiers pass; held task passes
    results_pass = [receipt(tasks[f"t{i}"]) for i in range(1, 5)]
    res_pass = qualify(view, tasks, results_pass)
    assert res_pass["status"] == "HELD"
    assert res_pass["qualified_tier"] == 3
    assert res_pass["eligible"] is False
    assert res_pass["held_tiers"] == [4]
    assert res_pass["counted_tiers"] == [1, 2, 3]

    # Lower counted tiers pass; held task fails
    results_fail = [
        receipt(tasks["t1"]), receipt(tasks["t2"]),
        receipt(tasks["t3"]), receipt(tasks["t4"], final=2.0)
    ]
    res_fail = qualify(view, tasks, results_fail)
    assert res_fail["status"] == "HELD"
    assert res_fail["qualified_tier"] == 3
    assert res_fail["eligible"] is False
    assert res_fail["held_tiers"] == [4]


def test_qualification_top_tier_held_semantics_tier_5():
    """Verify top-tier HELD semantics for Tier 5."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 6)}
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required"),
        ("t5", 5, "held")
    )
    view["assignments"][4]["hold"] = hold("t5")

    results = [receipt(tasks[f"t{i}"]) for i in range(1, 6)]
    res = qualify(view, tasks, results)

    assert res["status"] == "HELD"
    assert res["qualified_tier"] == 4
    assert res["eligible"] is False
    assert res["held_tiers"] == [5]
    assert res["counted_tiers"] == [1, 2, 3, 4]


def test_qualification_counted_failure_never_masked_by_held_top_tier():
    """Verify a counted failure at lower tiers is never masked as HELD by a top held tier."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 5)}
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "held")
    )
    view["assignments"][3]["hold"] = hold("t4")

    # Tier 2 fails
    results = [
        receipt(tasks["t1"]), receipt(tasks["t2"], final=2.0),
        receipt(tasks["t3"]), receipt(tasks["t4"])
    ]
    res = qualify(view, tasks, results)

    # Must report FAIL, not HELD
    assert res["status"] == "FAIL"
    assert res["qualified_tier"] == 1
    assert res["eligible"] is False


def test_qualification_diagnostic_tasks_isolated_from_totals():
    """Verify diagnostic tasks in Tier 4 do not enter required_total or affect qualification."""
    tasks = {f"t{i}": simple_task(f"t{i}") for i in range(1, 5)}
    tasks["d4"] = simple_task("d4")
    view = simple_view(
        ("t1", 1, "required"), ("t2", 2, "required"),
        ("t3", 3, "required"), ("t4", 4, "required"),
        ("d4", 4, "diagnostic")
    )
    results = [
        receipt(tasks["t1"]), receipt(tasks["t2"]),
        receipt(tasks["t3"]), receipt(tasks["t4"]),
        receipt(tasks["d4"], final=2.0)  # diagnostic fails
    ]
    res = qualify(view, tasks, results)

    assert res["status"] == "PASS"
    assert res["qualified_tier"] == 4
    assert res["eligible"] is True
    assert res["required_passed"] == 4
    assert res["required_total"] == 4
    assert res["task_statuses"]["d4"] == "FAIL"


# ---------------------------------------------------------------------------
# 4. Dynamic Report & Table Generation Tests
# ---------------------------------------------------------------------------

def test_tier_name_mapping():
    """Verify tier_name maps 1-4 to specific names and N >= 5 to automatic tier-N fallback."""
    assert tier_name(1) == "smoke"
    assert tier_name(2) == "quality"
    assert tier_name(3) == "endurance"
    assert tier_name(4) == "stress_and_formulations"
    assert tier_name(5) == "tier-5"
    assert tier_name(6) == "tier-6"
    assert tier_name(10) == "tier-10"
    assert tier_name(99) == "tier-99"


def test_tier_report_renders_tier_4_and_tier_5(tmp_path):
    """Verify build_report and render_markdown dynamically render 4-tier and 5-tier tables."""
    # Scaffold temporary task catalog
    for i in range(1, 6):
        atomic_json(tmp_path / f"configs/forge/tasks/task{i}.json", simple_task(f"task{i}"))

    # Scaffold 4-tier view
    v4_assignments = [
        {"task": f"task{i}", "qualification_tier": i, "importance": "required", "order": 0}
        for i in range(1, 5)
    ]
    atomic_json(tmp_path / "configs/forge/views/v4.json", {
        "schema_version": 1, "id": "v4", "revision": 1, "goal": "quality",
        "eligibility": {}, "calibration": {"status": "provisional"},
        "assignments": v4_assignments
    })

    # Build report for 4-tier catalog
    report4 = build_report(tmp_path)
    view4 = report4["views"][0]
    tier_numbers = [t["qualification_tier"] for t in view4["tiers"]]
    assert 4 in tier_numbers
    t4 = next(t for t in view4["tiers"] if t["qualification_tier"] == 4)
    assert t4["name"] == "stress_and_formulations"

    md4 = render_markdown(report4, tmp_path, tmp_path / "reports/forge/EXPERIMENTS_BY_TIER.md")
    assert "| View | Revision | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Declared calibration |" in md4
    assert "### Tier 4: stress_and_formulations" in md4

    # Now add a 5-tier view
    v5_assignments = [
        {"task": f"task{i}", "qualification_tier": i, "importance": "required", "order": 0}
        for i in range(1, 6)
    ]
    atomic_json(tmp_path / "configs/forge/views/v5.json", {
        "schema_version": 1, "id": "v5", "revision": 1, "goal": "quality",
        "eligibility": {}, "calibration": {"status": "provisional"},
        "assignments": v5_assignments
    })

    report5 = build_report(tmp_path)
    md5 = render_markdown(report5, tmp_path, tmp_path / "reports/forge/EXPERIMENTS_BY_TIER.md")
    assert "Tier 5" in md5
    assert "### Tier 5: tier-5" in md5


def test_family_reports_table_header_dynamic_tiers():
    """Verify _table_header produces dynamic column headers across 3, 4, and 5 tiers."""
    # 4 tiers - view headers
    h4 = _table_header(tiers=("1", "2", "3", "4"), selected=False)
    assert "| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |" in h4[0]

    # 4 tiers - selected recipe headers
    h4_sel = _table_header(tiers=("1", "2", "3", "4"), selected=True)
    assert "| Family / selected recipe | Tier 1 requirements | Tier 2 requirements | Tier 3 requirements | Tier 4 requirements | Total view requirements |" in h4_sel[0]

    # 5 tiers - view headers
    h5 = _table_header(tiers=("1", "2", "3", "4", "5"), selected=False)
    assert "Tier 5 tests" in h5[0]

    # Held column appended properly
    h4_held = _table_header(tiers=("1", "2", "3", "4"), selected=False, held=True)
    assert "Held (measured, not counted)" in h4_held[0]
    # Check delimiter column count matches header column count
    header_cols = [c for c in h4_held[0].split("|") if c.strip()]
    divider_cols = [c for c in h4_held[1].split("|") if c.strip()]
    assert len(header_cols) == len(divider_cols)


# ---------------------------------------------------------------------------
# 5. CLI Argument Parsing Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcommand", ["plan", "enqueue", "run"])
def test_forge_cli_through_tier_flags_accept_4_and_5(subcommand):
    """Verify CLI subcommands plan/enqueue/run accept --through-tier 4 and 5."""
    cli_parser = parser()

    args4 = cli_parser.parse_args([subcommand, "base", "--through-tier", "4"])
    assert args4.through_tier == 4

    args5 = cli_parser.parse_args([subcommand, "base", "--through-tier", "5"])
    assert args5.through_tier == 5


def test_forge_cli_board_accepts_tier_4_and_5():
    """Verify forge board accepts --tier 4 and 5."""
    cli_parser = parser()
    args4 = cli_parser.parse_args(["board", "--tier", "4"])
    assert args4.tier == 4
    args5 = cli_parser.parse_args(["board", "--tier", "5"])
    assert args5.tier == 5


def test_forge_cli_retier_accepts_tier_4():
    """Verify forge retier accepts --tier 4."""
    cli_parser = parser()
    args = cli_parser.parse_args([
        "retier", "view", "--task", "grid100_mlp16", "--tier", "4", "--reason", "stress"
    ])
    assert args.tier == 4


def test_dispositions_tier_counts_accepts_tier4():
    """Verify dispositions _tier_counts dynamically sizes lists to highest tier."""
    view = simple_view(
        ("t1", 1, "required"),
        ("t2", 2, "required"),
        ("t3", 3, "required"),
        ("t4", 4, "held"),
    )
    counts = _tier_counts(view)
    assert len(counts["required"]) == 4
    assert counts["required"] == [1, 1, 1, 0]
    assert counts["held"] == [0, 0, 0, 1]


def test_configuration_search_accepts_tier4():
    """Verify select_configuration accepts through_tier=4."""
    trials = [
        {"configuration_id": "cfg1", "candidate_id": "c1", "submission_status": "completed",
         "tasks": [{"task": "t1", "importance": "required", "qualification_tier": 1, "gate_status": "PASS", "compatibility_key": "k1"},
                   {"task": "t2", "importance": "required", "qualification_tier": 2, "gate_status": "PASS", "compatibility_key": "k2"},
                   {"task": "t3", "importance": "required", "qualification_tier": 3, "gate_status": "PASS", "compatibility_key": "k3"},
                   {"task": "t4", "importance": "required", "qualification_tier": 4, "gate_status": "PASS", "compatibility_key": "k4"}]}
    ]
    summary = select_configuration(trials, through_tier=4)
    assert summary["selected_candidate_id"] == "c1"
    assert summary["required_passes_by_tier"] == [1, 1, 1, 1]


def test_cli_rejects_invalid_non_positive_tiers():
    """Verify CLI flags reject integers < 1 for --through-tier and < 0 for --tier."""
    p = parser()
    with pytest.raises(SystemExit):
        p.parse_args(["plan", "idea", "--through-tier", "0"])
    with pytest.raises(SystemExit):
        p.parse_args(["plan", "idea", "--through-tier", "-1"])
    with pytest.raises(SystemExit):
        p.parse_args(["board", "--tier", "-1"])
    with pytest.raises(SystemExit):
        p.parse_args(["retier", "view", "--task", "t", "--tier", "0", "--reason", "r"])


def test_task_recipe_resources_defensive_particle_fallback():
    """Verify task_recipe_resources gracefully defaults num_particles to 256 when omitted."""
    from experiments.forge.api import task_recipe_resources

    dummy_task = {
        "adapter": "transfer_vector",
        "execution": {
            "host_definition": {
                "z_dim": 8,
                "batch": 2048,
            }
        }
    }
    resources = task_recipe_resources(dummy_task)
    assert resources == {"num_particles": 256, "z_dim": 8, "batch_size": 2048}


def test_misgan_tasks_host_definition_particles_declared():
    """Verify misgan tasks explicitly declare particles: 256 in host_definition."""
    from pathlib import Path
    from experiments.forge.planning import load_tasks

    tasks = load_tasks(Path("."))
    for task_id in (
        "misgan_block_recovery",
        "misgan_block_neural_imputer",
        "misgan_mcar_p50_recovery",
        "misgan_mcar_p80_anchored",
    ):
        assert task_id in tasks
        host_def = tasks[task_id]["execution"]["host_definition"]
        assert "particles" in host_def, f"{task_id} missing particles in host_definition"
        assert host_def["particles"] == 256


def test_stretch_formulation_tasks_graders_and_preflight():
    """Unsupported hosts refuse admission; numerical graders retain their exact gates."""
    from experiments.forge.adapters import adapter_preflight
    from experiments.forge.interpolation import INTERPOLATION_KINDS, _contract_payload
    from experiments.forge.mechanisms import NAMES as MECHANISM_NAMES
    from experiments.forge.priors import RECIPE_PRIOR_CONTRACT
    from experiments.forge.sampling import ENUMERATED_PRIOR_CLEAN, PUBLIC_PRIOR_CLEAN, executed_receipt
    from experiments.forge.vectorprofiles import STRETCH_VECTOR_HOSTS
    from experiments.forge.views import STRETCH_EVALUATOR_KINDS, grade_result
    from ops.formulaic_coverage import STRETCH_FIXTURE_TASKS, adapt_candidate_for_task, load_candidate_card

    tasks = load_tasks(ROOT)
    assert STRETCH_VECTOR_HOSTS == STRETCH_FIXTURE_TASKS
    assert STRETCH_EVALUATOR_KINDS == STRETCH_FIXTURE_TASKS | {"zero_tanh_rail_escape"} | INTERPOLATION_KINDS
    for tid in sorted(STRETCH_FIXTURE_TASKS):
        t = deepcopy(tasks[tid])
        for cid in ("bispf-optimized-quintic-v1", "atlas-particle-normal-v4"):
            cand = adapt_candidate_for_task(load_candidate_card(ROOT, cid), t)
            assert cand == load_candidate_card(ROOT, cid)
            blockers = adapter_preflight(t, cand, root=ROOT)
            assert blockers, "a shape-valid host without a registered runner must not be admitted"

    for tid in sorted(STRETCH_EVALUATOR_KINDS):
        t = deepcopy(tasks[tid])
        steps = t["execution"]["steps"]
        obs_count = t["evaluation"].get("observations", 24)
        thresholds = t["evaluation"]["thresholds"]
        passing_metrics, failing_metrics = {}, {}
        for name, op, bound in thresholds:
            passing_metrics[name] = bound
            failing_metrics[name] = bound - 1.0 if op == ">=" else bound + 1.0
        ckpts = sorted({math.ceil(i * steps / obs_count) for i in range(1, obs_count + 1)})
        if tid == "zero_tanh_rail_escape":
            ckpts = sorted(
                set(ckpts)
                | {
                    t["execution"]["shift_step"] - 1,
                    t["execution"]["shift_step"] + t["evaluation"]["recovery_deadline"],
                    steps,
                }
            )
        sampling_law = ENUMERATED_PRIOR_CLEAN if tid in INTERPOLATION_KINDS else PUBLIC_PRIOR_CLEAN
        base_ev = {
            **executed_receipt(sampling_law, eval_output_noise="clean"),
            "scoring_weights": "live",
            "prior_policy": {
                "contract": RECIPE_PRIOR_CONTRACT,
                "applicability": "latent_prior",
                "update": "learned",
                "trainable_locations": True,
                "regularizer": "vicreg",
                "weight": 0.0,
                "target_std": 1.0,
                "eps": 1e-4,
                "l2": 0.0,
            },
            "guards": {
                "all_finite": True,
                "optimizer_updates": {"generator": steps, "discriminator": steps, "prior": steps},
                "hooks_exercised": True,
                "mechanism_audit": {
                    "mechanisms": {
                        name: {"requested": False, "enabled": False, "calls": 0, "eligible": 0, "applied": 0}
                        for name in MECHANISM_NAMES
                    }
                },
                "unintended_rng_deviations": 0,
            },
        }
        if tid in INTERPOLATION_KINDS:
            base_ev["interpolation_contract"] = _contract_payload(tid, t["execution"]["host_definition"])
            base_ev["actual_training_gif"] = {"sha256": "0" * 64, "frames": obs_count}
        obs_pass = [{"step": s, **passing_metrics} for s in ckpts]
        obs_fail = [{"step": s, **failing_metrics} for s in ckpts]
        live_pass = {"step": steps, **passing_metrics} if tid in INTERPOLATION_KINDS else passing_metrics
        live_fail = {"step": steps, **failing_metrics} if tid in INTERPOLATION_KINDS else failing_metrics
        assert grade_result(t, {"evidence": {**base_ev, "observations": obs_pass, "live": live_pass}})["status"] == "PASS"
        assert grade_result(t, {"evidence": {**base_ev, "observations": obs_fail, "live": live_fail}})["status"] == "FAIL"
