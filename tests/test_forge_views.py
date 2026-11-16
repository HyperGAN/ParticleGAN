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
    assert [a["task"] for a in stability["assignments"] if a["importance"] == "diagnostic"] == [
        "ring_hold", "clockfree_audit_measurement_v1", "precision_gaussian_fp32"]
    assert next(a for a in stability["assignments"] if a["task"] == "ring_hold")["removal"]["counted_by"] == "ring_extension"
    for role, counts in (("required", [11, 18, 1, 0]), ("held", [0, 7, 1, 5])):
        assert [sum(a["qualification_tier"] == tier and a["importance"] == role
                    for a in stability["assignments"]) for tier in (1, 2, 3, 4)] == counts
    archived = json.loads((ROOT / "configs/forge/view-history/discriminator_stability-v9.json").read_text())
    validate_view(archived, tasks)
    assert [sum(a["qualification_tier"] == tier and a["importance"] == "required"
                for a in archived["assignments"]) for tier in (1, 2, 3)] == [6, 21, 2]
    assert all("qualification_tier" not in t for t in tasks.values())
    assert tasks["ring_hold"]["execution"]["execution_group"] == tasks["ring_extension"]["execution"]["execution_group"]
    assert tasks["ring_hold"]["execution"]["max_total_steps"] == 7500
    assert "target_shift_recovery" not in {a["task"] for a in stability["assignments"]}
    assert "target_shift_recovery" in {a["task"] for a in load_view(ROOT, "adaptation")["assignments"]}
    assert "clockfree_audit" in {a["task"] for a in load_view(ROOT, "clockfree_continuous")["assignments"]}
    for name, task in tasks.items():
        prior = task["execution"]["prior"]
        if prior["kind"] == "particle_cloud":
            assert prior["kind"] == "particle_cloud" and prior["sigma"] == 0
            assert prior["exception_reason"]
        else:
            assert prior["kind"] == "mog" and prior["sigma"] > 0
            if task["execution"].get("prior_contract") == "recipe_owned_v1":
                assert "learnable" not in prior
            else:
                assert prior["learnable"]
    assert tasks["ae_gan_hold"]["execution"]["prior"]["kind"] == "mog"


def test_default_plan_and_report_include_acquisition_smoke_without_new_views():
    from experiments.forge.planning import plan_summary, resolve_idea
    from experiments.forge.tier_report import build_report

    request = resolve_idea(ROOT, "k3p", through_tier=1, execution_backend="cpu")
    plan = plan_summary(request)
    planned = {row["task"]: row for row in plan["tasks"]}
    report = build_report(ROOT)
    assert not {"ring16_acquisition", "five_word_joint"} & {row["id"] for row in report["views"]}
    view = next(row for row in report["views"] if row["id"] == "discriminator_stability")
    smoke = next(row for row in view["tiers"] if row["qualification_tier"] == 1)
    assert smoke["counts"] == {"required": 11, "held": 0, "ranking": 0, "diagnostic": 2, "stretch": 3}
    reported = {row["id"]: row for row in smoke["tasks"]}
    for name, updates, timeout in (("gaussian1d_smoke", 1000, 120),
                                   ("ring16_acquisition", 1600, 300),
                                   ("five_word_joint_smoke", 20001, 900)):
        assert planned[name]["qualification_tier"] == 1
        assert planned[name]["importance"] == "required" and planned[name]["permitted_by_tier_cap"]
        assert planned[name]["budget_seconds"] == timeout
        assert reported[name]["steps"] == updates and reported[name]["timeout_seconds"] == timeout
    # The tier cap permits both acquisitions and excludes every quality task.
    assert all(not row["permitted_by_tier_cap"] for row in plan["tasks"]
               if row["qualification_tier"] > 1)


def test_uninterrupted_execution_group_cannot_cross_tier_caps():
    tasks=load_tasks(ROOT)
    view=load_view(ROOT,"discriminator_stability")
    next(a for a in view["assignments"] if a["task"]=="ring_hold")["qualification_tier"]=2
    with pytest.raises(ValueError,match="uninterrupted execution group"):
        validate_view(view,tasks)


def test_frozen_thresholds_reference_the_declared_gate_policy_and_current_scorer():
    from benchmarks.transfer_suite.protocol import required_tasks
    from benchmarks.transfer_suite.vector_tasks import TASKS as vectors
    from benchmarks.transfer_suite.image_tasks import TASKS as images
    tasks = load_tasks(ROOT)
    from tests.archived_forge_contracts import archived_contract_bytes
    archived_shallow = {"gaussian1d_shallow_smoke", "gaussian1d_shallow_stability"}
    # Behavioral and image tasks still follow their existing task declarations.
    # Vector policy is independently frozen below; scorer upgrades cannot replace it.
    for spec in required_tasks() + [s for s in images if s["tier"] == "ranking"]:
        thresholds = spec["thresholds"]
        if isinstance(thresholds, dict):
            thresholds = [["modes", ">=", thresholds["modes"]], ["hq", ">=", thresholds["hq_min"]]]
        assert tasks[spec["name"]]["evaluation"]["thresholds"] == json.loads(json.dumps(thresholds))
    assert {s["name"] for s in vectors if s["tier"] == "ranking"} == set(VECTOR_THRESHOLDS_V1)
    from experiments.forge.tier1_policy import is_policy_task, validate as validate_policy
    # Separately scoped policy variants keep their parent's exact frozen gates.
    for task in tasks.values():
        if is_policy_task(task):
            parent = validate_policy(task)
            for field in ("thresholds", "gate_policy", "observations", "minimum_stable_checks"):
                assert task["evaluation"].get(field) == parent["evaluation"].get(field)
    assert {name for name, task in tasks.items() if task["adapter"] == "transfer_vector"
            and "vector_profile" not in task["execution"] and not is_policy_task(task)
            and not name.startswith("recipe-prior-refactor-legacy-")
            and task["evaluation"].get("gate_policy", {}).get("id") == "forge-vector-full-component-v1"} == set(VECTOR_THRESHOLDS_V1)
    for task in tasks.values():
        if task["adapter"] == "transfer_vector" and "vector_profile" in task["execution"]:
            assert task["evaluation"] == tasks[task["execution"]["host"]]["evaluation"]
    changed_upstream = set()
    for name, thresholds in VECTOR_THRESHOLDS_V1.items():
        task = tasks[name]
        evaluation = task["evaluation"]
        assert evaluation["thresholds"] == thresholds
        assert evaluation["gate_policy"]["id"] == "forge-vector-full-component-v1"
        assert evaluation["gate_policy"]["retained_from"] == "transfer-vectors-v2"
        assert evaluation["gate_policy"]["finite_atom_exemptions"] is False
        assert "nonzero-width MoG" in evaluation["gate_policy"]["rationale"]
        assert evaluation["sampling_law"] == "public_prior_without_output_noise"
        assert task["execution"]["prior"]["kind"] == "mog" and task["execution"]["prior"]["sigma"] > 0
        upstream = next(s for s in vectors if s["name"] == name)
        if upstream["thresholds"] != thresholds:
            changed_upstream.add(name)
    assert changed_upstream == {"vector_anisotropic", "vector_unequal_width", "vector_unequal_mass"}
    missing_history = set()
    for task in tasks.values():
        if task["id"].startswith("recipe-prior-refactor-legacy-"):
            # Inert binding controls retain exact original source identities.
            archive = ROOT / "reports/forge/recipe-prior-refactor"
            baseline = json.loads((archive / "baseline.json").read_text())
            original_id = task["id"].removeprefix("recipe-prior-refactor-legacy-")
            original_bytes = (archive / "legacy-task-cards" / f"{original_id}.json").read_bytes()
            assert hashlib.sha256(original_bytes).hexdigest() == baseline["task_cards"][original_id]["sha256"]
            original = json.loads(original_bytes)
            original["id"] = task["id"]
            if task != original:
                # PR399 rebinding: exactly one declared source-only layer over the frozen card.
                revision = task["evaluation"]["evaluator_revision"]
                assert revision["id"] == "pr399-host-device-replay-source-only-v1"
                assert revision["previous_revision"] == original["evaluation"].get("evaluator_revision")
                frozen = original["evaluation"].get("sources", {})
                live = task["evaluation"].get("sources", {})
                rebound = set(revision["previous_source_sha256"]) | set(revision.get("added_source_members", []))
                assert all(frozen.get(p) == d for p, d in revision["previous_source_sha256"].items())
                strip = lambda t: {**t, "evaluation": {k: v for k, v in t["evaluation"].items()
                                                       if k not in {"sources", "evaluator_revision"}}}
                assert strip(task) == strip(original)
                assert {p for p in set(live) | set(frozen) if live.get(p) != frozen.get(p)} <= rebound
            if subprocess.run(["git", "cat-file", "-e", baseline["source_commit"] + "^{commit}"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                for path, digest in original["evaluation"].get("sources", {}).items():
                    source = subprocess.check_output(["git", "show", baseline["source_commit"] + ":" + path], cwd=ROOT)
                    assert hashlib.sha256(source).hexdigest() == digest
            else:
                missing_history.add(baseline["source_commit"])
            continue
        historical_policy_sources = task["execution"].get("policy_contract", {}).get("sources", {})
        if historical_policy_sources:
            from experiments.forge.tier1_policy import validate
            parent = validate(task)
            historical_policy_sources = {
                **parent["evaluation"].get("sources", {}),
                **historical_policy_sources,
                "configs/forge/tasks/" + parent["id"] + ".json": task["policy_parent"]["task_sha256"],
            }
        for path, digest in task["evaluation"].get("sources", {}).items():
            if path in historical_policy_sources:
                # Frozen policy execution bytes may deliberately be blocked
                # by a newer package. Their source pins remain immutable.
                assert digest == historical_policy_sources[path]
                continue
            if task["id"] in archived_shallow and path == "experiments/forge/gaussian_tasks.py":
                original = json.loads(archived_contract_bytes(
                    "shallow_gaussian_cards", "configs/forge/tasks/" + task["id"] + ".json"))
                assert task == original
                assert hashlib.sha256(archived_contract_bytes("shallow_gaussian", path)).hexdigest() == digest
                from experiments.forge.preflight import task_preflight
                candidate = json.loads((ROOT / "configs/forge/ideas/k3p.json").read_text())
                blockers = task_preflight(deepcopy(task), candidate, {}, root=ROOT, tasks=tasks)
                assert f"{task['id']}: evaluator source changed; revise the task definition: {path}" in blockers
                continue
            if task["adapter"] == "public_api_model_toy" and path in {
                    "benchmarks/toy_audit/api_contract.py", "benchmarks/toy_audit/api_run.py"}:
                from pr_integration_compat import COMBINED_INTEGRATION, before_pr_integration
                original = before_pr_integration(
                    ROOT, task, "configs/forge/tasks/" + task["id"] + ".json")
                if task["id"] == "model_misgan_four_mode":
                    # This separate cohort originally declared no API source map.
                    # Preserve that exact coupled-state task; corrected v2 gains
                    # explicit current pins without inheriting historical quality.
                    receipt_path = "reports/forge/pr-integration-20261109/misgan-execution-v2.json"
                    receipt_raw = (ROOT / receipt_path).read_bytes()
                    assert receipt_raw == subprocess.check_output([
                        "git", "show", "ea75964cd20ed87e6f7a798e694205594b7e65a9:" + receipt_path], cwd=ROOT)
                    receipt = json.loads(receipt_raw)
                    archived = receipt["coupled_v1"]["task"]
                    archived_raw = subprocess.check_output([
                        "git", "show", archived["source_commit"] + ":" + archived["source_path"]], cwd=ROOT)
                    assert (ROOT / archived["archived_path"]).read_bytes() == archived_raw
                    assert hashlib.sha256(archived_raw).hexdigest() == archived["sha256"]
                    assert original == json.loads(archived_raw)
                    assert original["evaluation"]["sources"] == {}
                    assert task["evaluation"]["integration_source_binding"]["id"] == COMBINED_INTEGRATION
                    assert task["execution"]["host_definition"]["provider"] == "benchmarks.toy_audit.api_model_misgan"
                    assert task["execution"]["execution_contract"] == receipt["version"]
                    assert receipt["corrected_execution_quality"] == "unmeasured"
                    assert receipt["historical_quality_inherited"] is False
                    assert receipt["qualification_input"] is False
                    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
                    continue
                # The original model cohort retains its exact contract bytes.
                # Capability admission changes do not rebind that evidence.
                frozen = archived_contract_bytes("model_toy_api_callers", path)
                assert hashlib.sha256(frozen).hexdigest() == original["evaluation"]["sources"][path]
                if original != task:
                    # The receipt verifies each source-only parent and every
                    # current pin. Frozen evidence retains its archived bytes.
                    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
                    continue
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                    from experiments.forge.preflight import task_preflight
                    candidate = json.loads((ROOT / "configs/forge/ideas/k3p.json").read_text())
                    blockers = task_preflight(deepcopy(task), candidate, {}, root=ROOT, tasks=tasks)
                    assert f"{task['id']}: evaluator source changed; revise the task definition: {path}" in blockers
                continue
            if path in {"benchmarks/toy_audit/api_model_ae.py",
                        "benchmarks/toy_audit/api_model_ddgan.py",
                        "benchmarks/toy_audit/api_model_vae.py"}:
                from recipe_entrypoint_compat import frozen_restoration_source
                frozen_restoration_source(ROOT, path, digest)
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                    from experiments.forge.preflight import task_preflight
                    candidate = json.loads((ROOT / "configs/forge/ideas/k3p.json").read_text())
                    blockers = task_preflight(deepcopy(task), candidate, {}, root=ROOT, tasks=tasks)
                    assert f"{task['id']}: evaluator source changed; revise the task definition: {path}" in blockers
                continue
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    if missing_history:
        pytest.skip("frozen legacy-card source pins need archived commit(s) " + ", ".join(sorted(c[:12] for c in missing_history))
                    + " which this shallow checkout lacks; all non-history checks passed")


def test_forge_retains_rare_component_shape_failure_despite_upstream_finite_atom_exemption():
    from benchmarks.transfer_suite.vector_tasks import TASKS, component_resolved
    task = load_tasks(ROOT)["vector_unequal_mass"]
    upstream = next(s for s in TASKS if s["name"] == task["id"])
    assert component_resolved(upstream) == [True, True, True, False]
    # Explicit legacy numerical fixture; this does not attest a trained recipe.
    task["execution"].pop("prior_contract", None)
    task["execution"]["prior"]["learnable"] = True
    # Synthetic evaluator fixture, not a training receipt: the rare component
    # has correct mass but zero variance; all resolved components have good shape.
    metrics = dict(sw1_normalized=.1, mass_tv=0., hq=1., min_mass_ratio=1.,
                   component_covariance_error=.25, component_min_eigen_ratio=0.,
                   resolved_core_covariance_error=0., resolved_core_min_eigen_ratio=1.,
                   resolved_max_component_spill=0.)
    evidence = dict(**executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"),
                    live=metrics, scoring_weights="live", observations=[
        dict(metrics, step=math.ceil(i * task["execution"]["steps"] / 24)) for i in range(1, 25)])
    assert grade_result(task, {"evidence": evidence})["status"] == "FAIL"
    alternate = deepcopy(task)
    alternate["evaluation"]["thresholds"] = upstream["thresholds"]
    assert grade_result(alternate, {"evidence": evidence})["status"] == "PASS"
    assert task_evaluation_fingerprint(alternate) != task_evaluation_fingerprint(task)
    assert task_execution_fingerprint(alternate) == task_execution_fingerprint(task)


@pytest.mark.parametrize("field,value", [("gate_policy", {"id": "uncalibrated-v5"}),
                                         ("sampling_law", "public_prior_with_output_noise")])
def test_vector_gate_and_sampling_declarations_are_bound_in_evaluator_identity(field, value):
    task = load_tasks(ROOT)["vector_unequal_mass"]
    changed = deepcopy(task)
    changed["evaluation"][field] = value
    assert task_evaluation_fingerprint(changed) != task_evaluation_fingerprint(task)
    assert task_execution_fingerprint(changed) == task_execution_fingerprint(task)


def test_retiering_reuses_evidence_without_changing_execution_or_evaluator_identity():
    tasks = {n: simple_task(n) for n in "abcd"}
    view = simple_view(("a", 1, "required"), ("b", 2, "required"),
                       ("c", 2, "required"), ("d", 3, "required"))
    frozen = deepcopy(view)
    results = [receipt(t) for t in tasks.values()]
    before = qualify(view, tasks, results)
    fingerprints = {n: task_fingerprint(t) for n, t in tasks.items()}
    moved = deepcopy(view)
    moved["revision"] = 2
    moved["assignments"][1]["qualification_tier"] = 1
    after = qualify(moved, tasks, results)
    assert before["qualified_tier"] == after["qualified_tier"] == 3
    assert after["task_statuses"] == before["task_statuses"]
    assert after["next_task"] is None
    assert view == frozen
    assert {n: task_fingerprint(t) for n, t in tasks.items()} == fingerprints
    assert view_fingerprint(moved) != view_fingerprint(view)
    changed = deepcopy(tasks["a"])
    changed["evaluation"]["thresholds"][0][2] = .1
    assert task_execution_fingerprint(changed) == task_execution_fingerprint(tasks["a"])
    assert task_evaluation_fingerprint(changed) != task_evaluation_fingerprint(tasks["a"])


def test_current_clock_view_preserves_historical_policy_and_first_tier_budget():
    tasks = load_tasks(ROOT)
    path = ROOT / "configs/forge/view-history/clockfree_continuous-v2.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "68935adac00a08890c798e193af48ab8bfdddf719e7c66f52aec2f36806b989d"
    historical = json.loads(path.read_text())
    current = load_view(ROOT, "clockfree_continuous")
    validate_view(historical, tasks)
    assert current["revision"] == 5 and historical["revision"] == 2
    assert current["eligibility"] == historical["eligibility"]
    assert current["calibration"] == historical["calibration"]
    assignments = {a["task"]: a for a in current["assignments"]}
    assert assignments["clockfree_audit_measurement_v1"]["qualification_tier"] == 1
    assert assignments["clockfree_audit"]["qualification_tier"] == 3
    # Required continuations require their existing strict audit gate to count too.
    assert assignments["clockfree_audit"]["importance"] == "required"
    assert "removal" not in assignments["clockfree_audit"]
    for name in ("grid100_14k", "rotated100_14k", "staggered100_14k", "target_shift_recovery"):
        assert assignments[name]["importance"] == "required" and assignments[name]["qualification_tier"] == 3
    assert all("hold" not in assignments[name] for name in ("grid100_14k", "rotated100_14k", "staggered100_14k"))
    previous = json.loads((ROOT / current["previous_revision"]["archive"]).read_text())
    assert previous["revision"] == 4
    assert next(a for a in previous["assignments"] if a["task"] == "clockfree_audit")["importance"] == "diagnostic"
    assert next(a for a in historical["assignments"] if a["task"] == "clockfree_audit")["qualification_tier"] == 1
    # Shared execution groups reserve their maximum timeout once.
    def ceiling(view, tier=None):
        groups = {}
        for assignment in view["assignments"]:
            if tier is not None and assignment["qualification_tier"] != tier:
                continue
            task = tasks[assignment["task"]]
            group = task["execution"].get("execution_group", task["id"])
            groups[group] = max(groups.get(group, 0), task["resources"]["timeout_seconds"])
        return sum(groups.values())
    assert ceiling(current, 1) == ceiling(historical, 1)
    assert ceiling(current) == ceiling(historical) + 300


def test_passing_clock_probe_cannot_replace_original_long_run_prerequisite(monkeypatch):
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "clockfree_continuous")
    # Isolate prerequisite reduction from the separately tested artifact grader.
    def grade(task, result):
        status = "NOT_RUN" if result is None else "PASS"
        return {"status": status, "gate_status": status, "reasons": []}
    monkeypatch.setattr("experiments.forge.views.grade_result", grade)
    results = [{"task_id": a["task"]} for a in view["assignments"] if a["task"] != "clockfree_audit"]
    statuses = qualify(view, tasks, results)["task_statuses"]
    assert statuses["clockfree_audit_measurement_v1"] == "PASS"
    assert statuses["clockfree_audit"] == "NOT_RUN"
    for name in ("grid100_14k", "rotated100_14k", "staggered100_14k"):
        assert {"task": "clockfree_audit", "kind": "gate"} in tasks[name]["dependencies"]
        assert statuses[name] == "BLOCKED"


@pytest.mark.parametrize("mutation,match", [
    (lambda v,t: v["assignments"][0].update(task="missing"), "missing task"),
    (lambda v,t: v["assignments"][0].update(qualification_tier=2), "empty required tier"),
    (lambda v,t: v["assignments"].append(deepcopy(v["assignments"][0])), "duplicate assignment"),
    (lambda v,t: t["a"].update(qualification_tier=1), "only in view"),
    (lambda v,t: t["a"].update(dependencies=[dict(task="b", kind="gate")]), "later tier"),
    (lambda v,t: t["b"].update(dependencies=[dict(task="missing", kind="gate")]), "missing from view"),
    (lambda v,t: t["b"].update(dependencies=[dict(task="a", kind="checkpoint")]), "continuation_of"),
])
def test_invalid_retiering_and_dependency_edits_are_rejected(mutation, match):
    tasks = {n: simple_task(n) for n in "ab"}
    view = simple_view(("a",1,"required"),("b",2,"required"))
    mutation(view,tasks)
    with pytest.raises(ValueError, match=match):
        validate_view(view,tasks)


def test_same_tier_cycles_and_checkpoints_without_producer_are_rejected():
    tasks = {n:simple_task(n) for n in "ab"}
    view = simple_view(("a",1,"required"),("b",1,"required"))
    tasks["a"]["dependencies"] = [dict(task="b",kind="gate")]
    tasks["b"]["dependencies"] = [dict(task="a",kind="gate")]
    with pytest.raises(ValueError,match="cycle"):
        validate_view(view,tasks)
    tasks["a"]["dependencies"] = []
    tasks["a"]["execution"]["produces_state"] = False
    tasks["b"]["dependencies"][0]["kind"] = "checkpoint"
    tasks["b"]["execution"]["continuation_of"] = "a"
    with pytest.raises(ValueError,match="does not produce state"):
        validate_view(view,tasks)


def test_required_failure_stops_next_tier_and_keeps_partial_qualification():
    tasks = {n:simple_task(n) for n in "abc"}
    view = simple_view(("a",1,"required"),("b",2,"required"),("c",3,"required"))
    result = qualify(view,tasks,[receipt(tasks["a"]),receipt(tasks["b"],final=2.)])
    assert result["qualified_tier"] == 1
    assert result["status"] == "FAIL" and result["next_task"] is None
    assert result["task_statuses"]["c"] == "NOT_RUN"
    assert result["required_total"] == 3 and result["required_passed"] == 1
    assert result["cost_seconds"] == 4


def test_next_task_respects_prerequisites_within_tier():
    tasks = {n:simple_task(n) for n in "ab"}
    tasks["a"]["dependencies"] = [dict(task="b",kind="gate")]
    view = simple_view(("a",1,"required"),("b",1,"required"))
    assert qualify(view,tasks,[])["next_task"] == "b"
    assert qualify(view,tasks,[receipt(tasks["b"])])["next_task"] == "a"


def test_dependency_failures_propagate_before_ordered_leaderboard_rows():
    tasks = {n: simple_task(n) for n in "abc"}
    tasks["a"]["dependencies"] = [dict(task="b", kind="gate")]
    tasks["b"]["dependencies"] = [dict(task="c", kind="gate")]
    view = simple_view(("a", 1, "required"), ("b", 1, "required"), ("c", 1, "required"))
    result = qualify(view, tasks, [receipt(tasks["a"]), receipt(tasks["b"]), receipt(tasks["c"], final=2.)])
    assert result["task_statuses"] == {"a": "BLOCKED", "b": "BLOCKED", "c": "FAIL"}
    assert result["required_passed"] == 0


def test_diagnostic_failure_does_not_veto_or_change_required_denominator():
    tasks = {n:simple_task(n) for n in "ab"}
    view = simple_view(("a",1,"required"),("b",1,"diagnostic"))
    result = qualify(view,tasks,[receipt(tasks["a"]),receipt(tasks["b"],final=2.)])
    assert result["qualified_tier"] == 1 and result["eligible"]
    assert result["required_total"] == result["required_passed"] == 1
    assert result["task_statuses"]["b"] == "FAIL"


def test_registered_calibration_view_has_no_qualification_even_when_all_metrics_pass():
    tasks = {"a": simple_task("a")}
    view = simple_view(("a", 1, "diagnostic"))
    with pytest.raises(ValueError, match="empty required tier"):
        validate_view(view, tasks)
    view["evidence_scope"] = "calibration_diagnostic"
    validate_view(view, tasks)
    missing = qualify(view, tasks, [])
    assert missing["status"] == "DIAGNOSTIC" and not missing["diagnostic_complete"]
    result = qualify(view, tasks, [receipt(tasks["a"])])
    assert result["diagnostic_complete"] and result["task_statuses"] == {"a": "PASS"}
    assert result["qualified_tier"] == 0 and not result["eligible"]
    assert not result["current_qualification_reuse"] and not result["next_tasks"]
    result = qualify(view, tasks, [receipt(tasks["a"], final=2.)])
    assert result["diagnostic_complete"] and result["task_statuses"] == {"a": "FAIL"}
    assert result["status"] == "DIAGNOSTIC" and result["qualified_tier"] == 0


@pytest.mark.parametrize("importance,tier", [("required", 1), ("ranking", 1), ("diagnostic", 2)])
def test_calibration_scope_cannot_hide_required_or_higher_tier_qualification(importance, tier):
    view = simple_view(("a", tier, importance))
    view["evidence_scope"] = "calibration_diagnostic"
    with pytest.raises(ValueError, match="only diagnostic tasks in Tier 1"):
        validate_view(view, {"a": simple_task("a")})


def test_raw_api_error_becomes_blocked_without_erasing_error_or_denominator():
    task = simple_task("a")
    raw = dict(task_id="a",gate_status="ERROR",error="unsupported keyword",applicability=dict(status="unsupported",reason="host lacks new field"))
    result = qualify(simple_view(("a",1,"required")),{"a":task},[raw])
    assert result["status"] == "BLOCKED" and result["required_total"] == 1
    assert result["tasks"][0]["raw_status"] == "ERROR"
    assert raw["gate_status"] == "ERROR"
    raw.pop("applicability")
    assert grade_result(task,raw)["status"] == "INCOMPLETE"


def test_pass_stamp_and_ema_cannot_substitute_for_live_evidence():
    task = simple_task("a")
    assert grade_result(task,{"gate_status":"PASS"})["status"] == "INCOMPLETE"
    raw = receipt(task)
    raw["evidence"]["scoring_weights"] = "ema"
    assert grade_result(task,raw)["status"] == "INVALID"


@pytest.mark.parametrize("mutation,status", [
    (lambda e:e["observations"].pop(),"INCOMPLETE"),
    (lambda e:e["observations"].append(dict(e["observations"][-1])),"INVALID"),
    (lambda e:e["observations"][5].update(error=float("inf")),"INVALID"),
    (lambda e:e["observations"][5].update(error=True),"INVALID"),
    (lambda e:e["observations"][5].update(step=5.5),"INVALID"),
    (lambda e:e["observations"][5].pop("error"),"INVALID"),
    (lambda e:e["live"].clear(),"INCOMPLETE"),
])
def test_whole_observation_evidence_is_required(mutation,status):
    task=simple_task("a"); raw=receipt(task)
    mutation(raw["evidence"])
    assert grade_result(task,raw)["status"] == status


def test_transient_or_early_good_checkpoint_cannot_hide_terminal_failure():
    task=simple_task("a")
    assert grade_result(task,receipt(task,bad_steps=[20]))["status"] == "FAIL"
    assert grade_result(task,receipt(task,bad_steps=[19]))["status"] == "PASS"
    assert grade_result(task,receipt(task,final=2.))["status"] == "FAIL"


def test_smoke_requires_learning_and_rng_activation_guards():
    task=simple_task("a")
    task["evaluation"]["guards"]=dict(finite_state=True,optimizer_roles=["prior"],mechanism_exercised=True,rng_isolation=True)
    raw=receipt(task)
    assert grade_result(task,raw)["status"] == "INCOMPLETE"
    raw["evidence"]["guards"]=dict(all_finite=True,optimizer_updates=dict(prior=0),hooks_exercised=True,unintended_rng_deviations=0)
    assert grade_result(task,raw)["status"] == "FAIL"
    raw["evidence"]["guards"]["optimizer_updates"]["prior"]=24
    assert grade_result(task,raw)["status"] == "BLOCKED"  # A boolean cannot prove activation.
    from experiments.forge.mechanisms import NAMES
    mechanisms = {name: dict(requested=False, enabled=False, calls=0, eligible=0, applied=0)
                  for name in NAMES}
    mechanisms["critic_penalty"].update(requested=True, enabled=True, calls=24, eligible=24, applied=24)
    raw["evidence"]["guards"]["mechanism_audit"] = {"schema_version": 1, "mechanisms": mechanisms}
    assert grade_result(task,raw)["status"] == "PASS"
    raw["evidence"]["guards"]["unintended_rng_deviations"]=1
    assert grade_result(task,raw)["status"] == "INVALID"


def test_conflicting_compatible_attempts_cannot_cherry_pick_a_pass():
    task=simple_task("a"); view=simple_view(("a",1,"required"))
    result=qualify(view,{"a":task},[receipt(task),receipt(task,final=2.)])
    assert result["status"] == "INVALID"


def ring_evidence(*,bad_step=None,truncate=0,run_id="one"):
    # First 200 checks acquire at 1400, followed by 1200 hold and 300 extension.
    points=[dict(step=s,modes=8,hq=.8 if s==bad_step else .95) for s in range(1201,2901-truncate)]
    return dict(**executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"),
                dense=points,continuity=dict(mode="uninterrupted",run_id=run_id))


def test_ring_first_convergence_and_immediate_extension_are_separate_verdicts():
    tasks=load_tasks(ROOT)
    for name in ("ring_hold", "ring_extension"):
        tasks[name]["execution"].pop("prior_contract", None)  # Legacy numerical reducer fixture.
        tasks[name]["execution"]["prior"]["learnable"] = True
    raw=dict(evidence=ring_evidence(bad_step=2700))
    assert grade_result(tasks["ring_hold"],raw)["status"] == "PASS"
    assert grade_result(tasks["ring_extension"],raw)["status"] == "FAIL"
    assert grade_result(tasks["ring_extension"],dict(evidence=ring_evidence(truncate=1)))["status"] == "INCOMPLETE"
    raw=dict(evidence=ring_evidence(bad_step=1500))
    assert grade_result(tasks["ring_hold"],raw)["status"] == "FAIL"


def test_ring_missing_dense_check_and_restarted_state_are_rejected():
    tasks=load_tasks(ROOT)
    for name in ("ring_hold", "ring_extension"):
        tasks[name]["execution"].pop("prior_contract", None)  # Legacy numerical reducer fixture.
        tasks[name]["execution"]["prior"]["learnable"] = True
    raw=dict(evidence=ring_evidence())
    raw["evidence"]["dense"].pop(10)
    assert grade_result(tasks["ring_hold"],raw)["status"] == "INVALID"
    raw=dict(evidence=ring_evidence()); raw["evidence"].pop("continuity")
    assert grade_result(tasks["ring_extension"],raw)["status"] == "BLOCKED"


def test_checkpoint_continuation_requires_same_parent_state():
    tasks={n:simple_task(n) for n in "ab"}
    tasks["b"]["dependencies"]=[dict(task="a",kind="checkpoint")]
    tasks["b"]["execution"]["continuation_of"]="a"
    view=simple_view(("a",1,"required"),("b",2,"required"))
    a,b=receipt(tasks["a"]),receipt(tasks["b"])
    a["evidence"]["continuity"]=dict(mode="uninterrupted",run_id="parent")
    b["evidence"]["continuity"]=dict(mode="uninterrupted",run_id="other")
    assert qualify(view,tasks,[a,b])["status"] == "BLOCKED"
    b["evidence"]["continuity"]["run_id"]="parent"
    assert qualify(view,tasks,[a,b])["qualified_tier"] == 2
    a["evidence"]["continuity"]["state_sha256"] = "f" * 64
    b["evidence"]["continuity"] = dict(mode="verified_checkpoint", full_state_verified=True,
        parent_state_sha256="f" * 64, restore_proof={"status": "PASS", "source_sha256": "a" * 64})
    blocked = qualify(view, tasks, [a,b])
    assert blocked["qualified_tier"] == 1 and blocked["task_statuses"]["b"] == "BLOCKED"
    assert "no supported proof contract" in str(blocked["blockers"])


def test_native_gate_requires_artifacts_and_rejects_wrong_budget(tmp_path):
    task=load_tasks(ROOT)["grid100"]
    task["execution"].pop("prior_contract", None)  # Isolate the artifact/budget reducer.
    task["execution"]["prior"]["learnable"] = True
    assert grade_result(task,dict(gate_status="PASS",evidence=dict(coverage="PASS",accuracy="PASS")))["status"] == "INCOMPLETE"
    (tmp_path/"grid100").mkdir()
    (tmp_path/"grid100/config.json").write_text(json.dumps(dict(steps=1000)))
    from experiments.forge.artifacts import manifest_artifacts
    raw=dict(evidence=dict(**executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"),
                           artifact_root=str(tmp_path), artifact_manifest=manifest_artifacts(tmp_path)))
    assert grade_result(task,raw)["status"] == "INVALID"


@pytest.mark.parametrize("name,key,value", [
    ("two_pole", "minimum_stable_checks", 6),
    ("two_pole", "observations", 30),
    ("two_pole", "evaluator", "another.module:gate"),
    ("grid100", "minimum_stable_checks", 7),
    ("grid100", "holdout_samples", 200000),
    ("grid100", "accuracy_limits", {"mass_tv": .01}),
    ("grid100", "coverage_thresholds", {"min_modes": 101}),
    ("ring_hold", "thresholds", [["hq", ">=", .99]]),
    ("target_shift_recovery", "stationary_checks", 6),
    ("target_shift_recovery", "deadline_checks", 82),
    ("target_shift_recovery", "minimum_frozen_passing", 1),
    ("clockfree_audit", "conditions", ["horizon"]),
    ("clockfree_audit", "training_feedback", True),
])
def test_fixed_evaluator_metadata_cannot_silently_override_its_predicate(name, key, value):
    tasks = load_tasks(ROOT)
    tasks[name]["evaluation"][key] = value
    view = load_view(ROOT, "adaptation" if name == "target_shift_recovery" else
        "clockfree_continuous" if name == "clockfree_audit" else "discriminator_stability")
    with pytest.raises(ValueError, match="fixed|unsupported override"):
        validate_view(view, tasks)
    assert grade_result(tasks[name], {"evidence": {}})["status"] == "INVALID"


def test_clockfree_claim_metadata_and_independent_parity_are_both_required():
    task=load_tasks(ROOT)["clockfree_audit"]
    view=simple_view((task["id"],1,"required"))
    view["eligibility"]=dict(requires_capabilities=["named_rng"],claim_contract=dict(learning="clockfree",shared_settings=True))
    evidence=dict(comparisons=[dict(condition=c,permitted_state_sha256="0"*64,rng_state_sha256="1"*64,
                                   reference_sha256="2"*64,changed_sha256="2"*64)
                               for c in task["evaluation"]["conditions"]],
                  source_audit=dict(source_sha256={"learner.py":"3"*64},allowed_state=["parameters","moments"],unexplained_clock_dependencies=[]))
    raw=dict(task_id=task["id"],evidence=evidence)
    assert qualify(view,{task["id"]:task},[raw])["status"] == "BLOCKED"
    candidate=dict(capabilities=["named_rng"],claim_contract=dict(learning="clockfree",shared_settings=True))
    assert qualify(view,{task["id"]:task},[raw],candidate=candidate)["status"] == "INCOMPLETE"
    raw["evidence"]["comparisons"][0]["changed_sha256"]="4"*64
    assert qualify(view,{task["id"]:task},[raw],candidate=candidate)["status"] == "INCOMPLETE"


def test_paired_adaptation_rejects_unmatched_or_truncated_control():
    task=load_tasks(ROOT)["target_shift_recovery"]
    assert grade_result(task,dict(evidence=dict(active={},frozen={})))["status"] == "INCOMPLETE"


def test_native_checkpoint_dependency_binds_the_selected_parent(monkeypatch):
    # Isolate dependency reduction; real saved-state verification is covered by
    # the public native continuation adapter test and does not use this stub.
    from experiments.forge import views
    parent, child = simple_task("prefix"), simple_task("continuation")
    child.update(adapter="native100_continuation", dependencies=[{"task": "prefix", "kind": "checkpoint"}])
    child["execution"]["continuation_of"] = "prefix"
    view = simple_view(("prefix", 1, "required"), ("continuation", 2, "required"))
    monkeypatch.setattr(views, "grade_result", lambda *args: views._verdict("PASS", "independent grader fixture"))
    checkpoint, manifest = {"state_sha256": "fixed"}, {"sha256": "tree"}
    first = {"task_id": "prefix", "compatibility_key": "prefix-key", "_attempt_id": "prefix-attempt",
             "evidence": {"checkpoint": checkpoint, "artifact_manifest": manifest}}
    second = {"task_id": "continuation", "evidence": {"prefix_parity": {
        "checkpoint": checkpoint, "artifact_manifest": manifest,
        "prerequisite": {"compatibility_key": "prefix-key", "attempt_id": "prefix-attempt"}}}}
    tasks = {"prefix": parent, "continuation": child}
    assert qualify(view, tasks, [first, second])["qualified_tier"] == 2
    second["evidence"]["prefix_parity"]["prerequisite"]["attempt_id"] = "another-candidate"
    assert qualify(view, tasks, [first, second])["task_statuses"]["continuation"] == "BLOCKED"


def test_every_declared_and_archived_view_still_validates_and_legacy_views_count_every_tier():
    tasks = load_tasks(ROOT)
    paths = sorted((ROOT / "configs/forge/views").glob("*.json")) + sorted((ROOT / "configs/forge/view-history").glob("*.json"))
    assert len(paths) >= 22
    for path in paths:
        view = json.loads(path.read_text())
        validate_view(view, tasks)
        if view.get("evidence_scope") or any(a["importance"] == "held" or "removal" in a for a in view["assignments"]):
            continue
        # Without held or removed tasks the new rules are inert: every assigned tier is counted.
        result = qualify(view, tasks, [])
        assert result["held_total"] == 0 and result["held_tiers"] == [] and result["held_tasks"] == []
        assert result["counted_tiers"] == sorted({a["qualification_tier"] for a in view["assignments"]})
        assert result["status"] in {"INCOMPLETE", "BLOCKED"} and not result["eligible"]


def test_held_middle_tier_is_graded_and_shown_but_neither_passes_nor_stops_progression():
    tasks = {n: simple_task(n) for n in "abc"}
    view = simple_view(("a", 1, "required"), ("b", 2, "held"), ("c", 3, "required"))
    view["assignments"][1]["hold"] = hold("b")
    result = qualify(view, tasks, [receipt(tasks["a"]), receipt(tasks["b"], final=2.), receipt(tasks["c"])])
    assert result["status"] == "PASS" and result["eligible"] and result["qualified_tier"] == 3
    assert [t["status"] for t in result["tiers"]] == ["PASS", "HELD", "PASS"]
    assert result["tiers"][1] == {"qualification_tier": 2, "status": "HELD", "required_passed": 0, "required_total": 0,
                                  "counts": {}, "held_total": 1, "held_passed": 0, "diagnostic_total": 0}
    assert result["counted_tiers"] == [1, 3] and result["held_tiers"] == [2] and result["held_tasks"] == ["b"]
    assert (result["held_total"], result["held_passed"]) == (1, 0)
    assert result["required_total"] == result["required_passed"] == 2
    assert result["task_statuses"]["b"] == "FAIL" and result["blockers"] == []
    assert next(row for row in result["tasks"] if row["task_id"] == "b")["hold"] == hold("b")
    pending = qualify(view, tasks, [receipt(tasks["a"])])
    assert pending["status"] == "INCOMPLETE" and pending["qualified_tier"] == 1 and pending["next_tasks"] == ["c"]
    failed = qualify(view, tasks, [receipt(tasks["a"]), receipt(tasks["c"], final=2.)])
    assert failed["status"] == "FAIL" and failed["qualified_tier"] == 1
    assert [b["task_id"] for b in failed["blockers"]] == ["c"]


def test_held_top_tier_reports_held_never_pass_and_keeps_counted_failures_visible():
    tasks = {n: simple_task(n) for n in "abc"}
    view = simple_view(("a", 1, "required"), ("b", 2, "required"), ("c", 3, "held"))
    view["assignments"][2]["hold"] = hold("c", reason="unsolved")
    result = qualify(view, tasks, [receipt(tasks[n]) for n in "abc"])
    assert result["status"] == "HELD" and not result["eligible"] and result["qualified_tier"] == 2
    assert result["tiers"][2]["status"] == "HELD" and (result["held_total"], result["held_passed"]) == (1, 1)
    assert result["counted_tiers"] == [1, 2] and result["next_tasks"] == [] and result["blockers"] == []
    failed = qualify(view, tasks, [receipt(tasks["a"]), receipt(tasks["b"], final=2.)])
    assert failed["status"] == "FAIL" and failed["qualified_tier"] == 1
    gated = deepcopy(view)
    gated["eligibility"] = {"requires_capabilities": ["named_rng"]}
    assert qualify(gated, tasks, [receipt(tasks[n]) for n in "abc"])["status"] == "BLOCKED"
    nothing_counted = simple_view(("a", 1, "held"))
    nothing_counted["assignments"][0]["hold"] = hold("a")
    empty = qualify(nothing_counted, tasks, [receipt(tasks["a"])])
    assert empty["status"] == "HELD" and not empty["eligible"] and empty["qualified_tier"] == 0
    assert empty["counted_tiers"] == [] and empty["held_tiers"] == [1] and empty["required_total"] == 0


@pytest.mark.parametrize("mutation,match", [
    (lambda a: a.pop("hold"), "hold metadata"),
    (lambda a: a["hold"].update(extra="field"), "exactly"),
    (lambda a: a["hold"].update(reason="too_hard"), "hold reason"),
    (lambda a: a["hold"].update(previous_importance="diagnostic"), "previously required"),
    (lambda a: a["hold"].update(schema="forge_hold_v0"), "previously required"),
    (lambda a: a["hold"].update(summary="too short"), "20-280"),
    (lambda a: a["hold"].update(summary="x" * 281), "20-280"),
    (lambda a: a["hold"].update(explanation="/tmp/README.md#b"), "path#anchor"),
    (lambda a: a["hold"].update(explanation=EXPLANATIONS), "path#anchor"),
    (lambda a: a["hold"].update(explanation="../README.md#b"), "path#anchor"),
    (lambda a: a["hold"].update(explanation="reports/x/README.txt#b"), "path#anchor"),
    (lambda a: a["hold"].update(explanation="reports/x/README.md#"), "path#anchor"),
    (lambda a: a["hold"].update(explanation="reports/x/READ ME.md#b"), "path#anchor"),
    (lambda a: a["hold"].update(explanation="reports/x/.md#b"), "path#anchor"),
    (lambda a: a["hold"].update(explanation="https://example.com/README.md#b"), "path#anchor"),
    (lambda a: a["hold"].update(explanation="C:/Users/x/README.md#b"), "path#anchor"),
    (lambda a: a["hold"].update(explanation="~/notes/README.md#b"), "path#anchor"),
    (lambda a: a["hold"].update(summary=" Fixture: a leading space is not plain text."), "20-280"),
    (lambda a: a["hold"].update(reentry="later"), "re-entry"),
    (lambda a: a["hold"].update(blocked_by=["a"]), "exactly its held prerequisites"),
    (lambda a: a["hold"].update(reason="held_prerequisite"), "held_prerequisite"),
    (lambda a: a.update(importance="required"), "requires importance held"),
    (lambda a: a.update(importance="diagnostic", removal=removal("b", "superseded")), "requires importance held"),
])
def test_hold_metadata_must_be_complete_and_consistent(mutation, match):
    tasks = {n: simple_task(n) for n in "ab"}
    view = simple_view(("a", 1, "required"), ("b", 2, "held"))
    view["assignments"][1]["hold"] = hold("b")
    validate_view(deepcopy(view), tasks)
    mutation(view["assignments"][1])
    with pytest.raises(ValueError, match=match):
        validate_view(view, tasks)


def test_task_behind_a_held_prerequisite_is_held_too_and_names_it():
    tasks = {n: simple_task(n) for n in "ab"}
    tasks["b"]["dependencies"] = [dict(task="a", kind="checkpoint")]
    tasks["b"]["execution"]["continuation_of"] = "a"
    view = simple_view(("a", 1, "held"), ("b", 2, "required"))
    view["assignments"][0]["hold"] = hold("a", reason="unsolved")
    with pytest.raises(ValueError, match="depends on held task a; hold b with reason held_prerequisite"):
        validate_view(view, tasks)
    view["assignments"][1].update(importance="held", hold=hold("b", blocked_by=["a"]))
    with pytest.raises(ValueError, match="uses reason held_prerequisite"):
        validate_view(view, tasks)
    view["assignments"][1]["hold"].update(reason="held_prerequisite", blocked_by=[])
    with pytest.raises(ValueError, match="blocked_by must list exactly"):
        validate_view(view, tasks)
    view["assignments"][1]["hold"]["blocked_by"] = ["a"]
    validate_view(view, tasks)
    a, b = same_run(tasks["a"], final=2.), same_run(tasks["b"])
    result = qualify(view, tasks, [a, b])
    assert result["task_statuses"] == {"a": "FAIL", "b": "BLOCKED"}
    assert result["status"] == "HELD" and result["blockers"] == [] and result["counted_tiers"] == []


def test_removed_phase_keeps_one_counted_verdict_and_its_prerequisites():
    tasks, view = phase_tasks()
    validate_view(view, tasks)
    results = [receipt(tasks["gate"]), same_run(tasks["phase"]), same_run(tasks["extension"])]
    passed = qualify(view, tasks, results)
    assert passed["status"] == "PASS" and passed["qualified_tier"] == 2
    assert passed["required_total"] == 2 and passed["tiers"][1]["diagnostic_total"] == 1
    assert next(row for row in passed["tasks"] if row["task_id"] == "phase")["removal"]["counted_by"] == "extension"
    # A failing phase still blocks the counted verdict that replays it.
    failed_phase = qualify(view, tasks, [results[0], same_run(tasks["phase"], final=2.), results[2]])
    assert failed_phase["task_statuses"]["extension"] == "BLOCKED" and failed_phase["status"] == "BLOCKED"
    assert [b["task_id"] for b in failed_phase["blockers"]] == ["extension"]
    # The group is proposed through its counted member once the external gate passes.
    assert qualify(view, tasks, results[:1])["next_tasks"] == ["extension"]
    assert qualify(view, tasks, [])["next_tasks"] == ["gate"]
    assert qualify(view, tasks, [receipt(tasks["gate"], final=2.)])["next_tasks"] == []
    assert qualify(view, tasks, [results[0], same_run(tasks["phase"], final=2.)])["next_tasks"] == []
    counted = deepcopy(view)
    counted["assignments"][1]["importance"] = "required"
    counted["assignments"][1].pop("removal")
    assert qualify(counted, tasks, results[:1])["next_tasks"] == ["phase"]
    # In one tier the counted job still waits for the gate that its removed phase needs.
    same_tier = deepcopy(view)
    same_tier["assignments"][0]["qualification_tier"] = 2
    same_tier["assignments"].insert(0, dict(task="smoke", qualification_tier=1, importance="required", order=9))
    tiered_tasks = {**tasks, "smoke": simple_task("smoke")}
    assert qualify(same_tier, tiered_tasks, [receipt(tiered_tasks["smoke"])])["next_tasks"] == ["gate"]


@pytest.mark.parametrize("mutation,match", [
    (lambda t, v: v["assignments"][1]["removal"].update(counted_by="gate"), "nonrequired"),
    (lambda t, v: v["assignments"][1]["removal"].update(reason="duplicate"), "nonrequired"),
    (lambda t, v: [t[n]["execution"].pop("uninterrupted") for n in ("phase", "extension")], "nonrequired"),
    (lambda t, v: t["phase"]["execution"].update(execution_group="other"), "nonrequired"),
    (lambda t, v: t["extension"].update(dependencies=[dict(task="phase", kind="gate")]), "nonrequired"),
    (lambda t, v: v["assignments"][1].update(qualification_tier=1), "uninterrupted execution group"),
    (lambda t, v: v["assignments"][2].update(importance="held", hold=hold("extension")), "phase removal"),
    (lambda t, v: v["assignments"][1]["removal"].update(counted_by=None), "name counted_by"),
    (lambda t, v: v["assignments"][1]["removal"].update(reason="superseded"), "name counted_by"),
    (lambda t, v: v["assignments"][1]["removal"].update(reason="unneeded"), "removal reason"),
    (lambda t, v: v["assignments"][1]["removal"].update(summary="short"), "20-280"),
    (lambda t, v: v["assignments"][1]["removal"].pop("explanation"), "exactly"),
    (lambda t, v: v["assignments"][1].update(importance="ranking"), "stays assigned as diagnostic"),
])
def test_phase_exception_is_narrow(mutation, match):
    tasks, view = phase_tasks()
    mutation(tasks, view)
    with pytest.raises(ValueError, match=match):
        validate_view(view, tasks)


def test_removed_phase_prerequisites_must_be_counted_like_its_counted_verdict():
    """A held or uncounted gate behind a removed phase would decide the counted verdict."""
    tasks, view = phase_tasks()
    held_gate = deepcopy(view)
    held_gate["assignments"][0].update(importance="held", hold=hold("gate"))
    with pytest.raises(ValueError, match="removed phase phase depends on held task gate; count gate, "
                                         "or hold extension and phase with reason held_prerequisite"):
        validate_view(held_gate, tasks)
    # The suggested repair is valid: the phase and its counted verdict are held behind the gate.
    held_gate["assignments"][1].pop("removal")
    held_gate["assignments"][1].update(importance="held", hold=hold("phase", reason="held_prerequisite", blocked_by=["gate"]))
    held_gate["assignments"][2].update(importance="held", hold=hold("extension", reason="held_prerequisite",
                                                                    blocked_by=["phase"]))
    validate_view(held_gate, tasks)
    probed_tasks, probed = deepcopy(tasks), deepcopy(view)
    probed_tasks["probe"] = simple_task("probe")
    probed_tasks["phase"]["dependencies"].append(dict(task="probe", kind="gate"))
    probed["assignments"].append(dict(task="probe", qualification_tier=1, importance="diagnostic", order=3))
    with pytest.raises(ValueError, match="nonrequired task \\(its removed phase phase depends on probe\\)"):
        validate_view(probed, probed_tasks)
    # The same holds for the real ring group: mode_hold gates ring_hold, which ring_extension replays.
    stability, all_tasks = load_view(ROOT, "discriminator_stability"), load_tasks(ROOT)
    gate = next(a for a in stability["assignments"] if a["task"] == "mode_hold")
    gate.update(importance="held", hold=hold("mode_hold"))
    with pytest.raises(ValueError, match="ring_extension: its removed phase ring_hold depends on held task mode_hold"):
        validate_view(stability, all_tasks)


def test_duplicate_removal_names_an_earlier_counted_task_with_the_same_grader():
    tasks = {n: simple_task(n) for n in ("probe", "other", "copy")}
    view = simple_view(("probe", 1, "required"), ("other", 2, "required"), ("copy", 2, "diagnostic"))
    view["assignments"][2]["removal"] = removal("copy", "duplicate", "probe")
    validate_view(view, tasks)
    superseded = deepcopy(view)
    superseded["assignments"][2]["removal"] = removal("copy", "ill_posed")
    validate_view(superseded, tasks)
    cases = [
        lambda t, v: v["assignments"][2]["removal"].update(counted_by="copy"),
        lambda t, v: v["assignments"][2]["removal"].update(counted_by="missing"),
        lambda t, v: v["assignments"][0].update(importance="held", hold=hold("probe")),
        lambda t, v: [v["assignments"][0].update(qualification_tier=2), v["assignments"][1].update(qualification_tier=1),
                      v["assignments"][2].update(qualification_tier=1)],
        lambda t, v: t["probe"].update(adapter="another_host"),
        lambda t, v: t["copy"]["evaluation"].update(kind="custom_probe"),
        # Same grader family but a different question: another bound or another source pin.
        lambda t, v: t["copy"]["evaluation"].update(thresholds=[["error", "<=", 0.5]]),
        lambda t, v: t["probe"]["evaluation"].update(sources={"benchmarks/other_scorer.py": "0" * 64}),
    ]
    for mutation in cases:
        bad_tasks, bad = deepcopy(tasks), deepcopy(view)
        mutation(bad_tasks, bad)
        with pytest.raises(ValueError, match="duplicate removal"):
            validate_view(bad, bad_tasks)
    # A derived counted copy may pin the duplicate's own declaration as provenance.
    derived = deepcopy(tasks)
    derived["probe"]["evaluation"]["sources"] = {"configs/forge/tasks/copy.json": "0" * 64}
    validate_view(view, derived)


def test_duplicate_rule_accepts_the_clock_audit_copy_but_not_a_different_density_test():
    all_tasks = load_tasks(ROOT)
    clock = json.loads((ROOT / "configs/forge/view-history/clockfree_continuous-v4.json").read_text())
    validate_view(clock, all_tasks)
    assert next(a for a in clock["assignments"] if a["task"] == "clockfree_audit")["removal"]["counted_by"] == (
        "clockfree_audit_measurement_v1")
    quality = load_view(ROOT, "quality_coverage")
    spiral = next(a for a in quality["assignments"] if a["task"] == "vector_spiral")
    assert spiral["importance"] == "required"
    spiral.update(importance="diagnostic", removal=removal("vector_spiral", "duplicate", "vector_two_broad"))
    with pytest.raises(ValueError, match="vector_spiral: a duplicate removal"):
        validate_view(quality, all_tasks)


def test_previous_revision_names_its_exact_archive():
    tasks = {"a": simple_task("a")}
    view = simple_view(("a", 1, "required"))
    view["revision"] = 3
    view["previous_revision"] = dict(revision=2, archive="configs/forge/view-history/test-v2.json",
                                     policy_fingerprint="0" * 64)
    validate_view(view, tasks)
    for change, match in ((dict(revision=3), "earlier revision"), (dict(archive="configs/forge/views/test.json"), "view-history"),
                          (dict(policy_fingerprint="abc"), "SHA-256"), (dict(extra=1), "exactly")):
        bad = deepcopy(view)
        bad["previous_revision"].update(change)
        with pytest.raises(ValueError, match=match):
            validate_view(bad, tasks)


def test_qualify_exposes_counted_eligible_and_continuous_check_summary():
    from experiments.forge.views import check_summary
    tasks = {n: simple_task(n) for n in "ab"}
    view = simple_view(("a", 1, "required"), ("b", 2, "held"))
    view["assignments"][1]["hold"] = hold("b")
    res_a = receipt(tasks["a"])
    res_b = receipt(tasks["b"], final=2.)
    default_out = qualify(view, tasks, [res_a, res_b])
    assert default_out["status"] == "HELD"
    assert default_out["eligible"] is False
    assert default_out["counted_eligible"] is True
    allowed_out = qualify(view, tasks, [res_a, res_b], allow_held_highest_tier=True)
    assert allowed_out["status"] == "PASS"
    assert allowed_out["eligible"] is True
    assert allowed_out["counted_eligible"] is True
    summary = check_summary({"evaluator_result": {"checks": {"c1": True, "c2": False, "c3": True, "c4": True}}})
    assert summary == {"passed_checks": 3, "total_checks": 4, "check_pass_rate": 0.75}


def test_stretch_assignments_and_stretch_tiers_never_block_qualification_or_pass():
    from experiments.forge.views import is_stretch, stretch_note
    tasks = {n: simple_task(n) for n in "abc"}
    view = simple_view(("a", 1, "required"), ("b", 2, "stretch"), ("c", 2, "stretch"))
    view["assignments"][2]["stretch"] = {
        "schema": "forge_stretch_v1",
        "reason": "frontier",
        "summary": "Frontier extra-credit benchmark task for multi-tier stretch validation.",
    }
    validate_view(view, tasks)
    assert is_stretch("stretch") and is_stretch("held") and not is_stretch("required")
    assert is_stretch(view["assignments"][1]) and stretch_note(view["assignments"][2])["reason"] == "frontier"

    # Even when stretch tasks fail or are not run, a passing required tier qualifies the view as PASS.
    result = qualify(view, tasks, [receipt(tasks["a"]), receipt(tasks["b"], final=2.)])
    assert result["status"] == "PASS" and result["eligible"] is True and result["qualified_tier"] == 1
    assert result["tiers"][1]["status"] == "STRETCH"
    assert result["tiers"][1]["stretch_total"] == 2 and result["tiers"][1]["stretch_passed"] == 0
    assert (result["stretch_total"], result["stretch_passed"]) == (2, 0)
    assert result["stretch_tasks"] == ["b", "c"] and result["stretch_tiers"] == [2]

    # When stretch tasks pass, stretch_passed increments without altering required totals.
    both_pass = qualify(view, tasks, [receipt(tasks["a"]), receipt(tasks["b"]), receipt(tasks["c"])])
    assert both_pass["status"] == "PASS" and (both_pass["stretch_total"], both_pass["stretch_passed"]) == (2, 2)
    assert (both_pass["required_total"], both_pass["required_passed"]) == (1, 1)

    # A stretch-only view reports STRETCH and is not eligible for required qualification.
    stretch_only = simple_view(("b", 1, "stretch"))
    validate_view(stretch_only, tasks)
    so_out = qualify(stretch_only, tasks, [receipt(tasks["b"])])
    assert so_out["status"] == "STRETCH" and not so_out["eligible"] and so_out["qualified_tier"] == 0

    # Invalid stretch metadata is rejected.
    bad_role = deepcopy(view)
    bad_role["assignments"][0]["stretch"] = {"reason": "frontier"}
    with pytest.raises(ValueError, match="requires importance stretch"):
        validate_view(bad_role, tasks)
    missing_summary = deepcopy(view)
    missing_summary["assignments"][1]["stretch"] = {"reason": "frontier"}
    with pytest.raises(ValueError, match="stretch metadata needs at least"):
        validate_view(missing_summary, tasks)
    bad_reason = deepcopy(view)
    bad_reason["assignments"][1]["stretch"] = {
        "reason": "invalid_reason",
        "summary": "Frontier extra-credit benchmark task for multi-tier stretch validation.",
    }
    with pytest.raises(ValueError, match="stretch reason"):
        validate_view(bad_reason, tasks)

    # A pure STRETCH tier above a HELD tier must not flip a HELD view into PASS.
    held_plus_stretch = simple_view(("a", 1, "required"), ("b", 2, "held"), ("c", 3, "stretch"))
    held_plus_stretch["assignments"][1]["hold"] = hold("b", reason="unsolved")
    validate_view(held_plus_stretch, tasks)
    hps_out = qualify(held_plus_stretch, tasks, [receipt(tasks["a"]), receipt(tasks["b"]), receipt(tasks["c"])])
    assert hps_out["status"] == "HELD" and not hps_out["eligible"] and hps_out["counted_eligible"] is True
    assert hps_out["qualified_tier"] == 1 and hps_out["held_tiers"] == [2] and hps_out["stretch_tiers"] == [2, 3]

    # Stretch prerequisite reason requires blocked_by listing stretch prerequisites.
    dep_tasks = {n: simple_task(n) for n in "ab"}
    dep_tasks["b"]["dependencies"] = [{"task": "a", "kind": "gate"}]
    dep_view = simple_view(("a", 1, "stretch"), ("b", 2, "stretch"))
    dep_view["assignments"][1]["stretch"] = {
        "schema": "forge_stretch_v1",
        "reason": "stretch_prerequisite",
        "summary": "Depends on stretch task a and therefore requires blocked_by.",
    }
    with pytest.raises(ValueError, match="requires blocked_by"):
        validate_view(dep_view, dep_tasks)
    dep_view["assignments"][1]["stretch"]["blocked_by"] = ["a"]
    validate_view(dep_view, dep_tasks)


