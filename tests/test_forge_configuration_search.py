"""Configuration search identity, bounded execution and selection contracts."""
from copy import deepcopy
import math
from pathlib import Path
import shutil

import pytest

from experiments.forge import configuration_search as search
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.planning import declaration_paths, discover_candidate_ids, load_idea, resolve_idea
from experiments.forge.queue import Queue


@pytest.fixture
def checkout(tmp_path):
    prior = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
    protocol = {"schema_version": 1, "id": "screening", "seed": 0,
                "rng": {"version": "forge-rng-v1"}, "scoring": {"weights": "live"}}
    atomic_json(tmp_path / "configs/forge/defaults.json", {"protocol": "screening", "prior": prior})
    atomic_json(tmp_path / "configs/forge/protocols/screening.json", protocol)
    atomic_json(tmp_path / "configs/forge/ideas/base.json", {
        "schema_version": 1, "id": "base", "goal": "stability", "hypothesis": "improve critic stability",
        "changed_factors": ["critic penalty"], "mechanism_class": "structural", "recipe_overrides": {},
        "claim_contract": {"sampling_law": "task_declared"}})
    assignments = []
    for tier in (1, 2, 3):
        atomic_json(tmp_path / f"configs/forge/tasks/t{tier}.json", {
            "schema_version": 1, "id": f"t{tier}", "adapter": "transfer_behavior",
            "execution": {"steps": 80, "prior": prior, "host": "mode_hold",
                          "initializer": "deterministic_orthogonal"},
            "evaluation": {"kind": "transfer_sustained", "thresholds": [["score", ">=", 1]],
                           "observations": 24, "minimum_stable_checks": 5,
                           "sampling_contract_version": 1, "sampling_law": "public_prior_without_output_noise",
                           "eval_output_noise": "clean"},
            "resources": {"gpus": 0, "gpu_memory_mb": 10, "cpu_threads": 1, "timeout_seconds": 10},
            "requires_capabilities": ["named_rng"], "dependencies": []})
        assignments.append({"task": f"t{tier}", "qualification_tier": tier, "importance": "required", "order": 0})
    atomic_json(tmp_path / "configs/forge/views/stability.json", {
        "schema_version": 1, "id": "stability", "revision": 1,
        "goal": "stability", "assignments": assignments, "eligibility": {}})
    (tmp_path / "particlegan").mkdir()
    (tmp_path / "particlegan/fixture.py").write_text("mechanism = 1\n")
    return tmp_path


@pytest.fixture
def spec(checkout):
    return {"schema_version": 1, "id": "study", "trainer_family": "family", "base_candidate": "base",
            "grid": {"lr": [.003, .005], "penalty": [{"reg_coeff": 1.0, "reg_coeff_end": None},
                                                     {"reg_coeff": .1, "reg_coeff_end": None}]},
            "tuning_through_tier": 1, "view": "stability", "execution_backend": "cpu", "protocol": "screening",
            "protocol_hash": stable_hash(read_json(checkout / "configs/forge/protocols/screening.json")),
            "campaign": {"id": "search-study", "budget_seconds": 40, "candidate_budget_seconds": 10}}


def complete(queue, score, *, seconds=2, fabricated_gate=None):
    """A certified frozen-evaluator terminal envelope, no training process."""
    claimed = queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}])
    assert claimed is not None
    if callable(score):
        score = score(claimed)
    evidence = {"curve": [{"step": math.ceil(i * 80 / 24), "score": score} for i in range(1, 25)],
                "live": {"score": score},
        "sampling_contract_version": 1, "sampling_law": "public_prior_without_output_noise",
        "eval_output_noise": "clean"}
    raw = {"evidence": evidence, "metrics": {"score": score}}
    gate = fabricated_gate or ("PASS" if score >= 1 else "FAIL")
    task = claimed["job"]["task_id"]
    atomic_json(Path(claimed["worker"]["directory"]) / "terminal.json", {
        "token": claimed["worker"]["token"], "attempt_status": "completed", "elapsed_seconds": seconds,
        "result": raw, "grading": {"raw_hash": stable_hash(raw), "source_digest": claimed["request"]["source"]["digest"],
                                     "grades": {task: {"gate_status": gate}}}})
    assert queue.collect() == 1
    return claimed


def test_plan_readonly_grid_correlations_full_denominators_and_pending(checkout, spec):
    before = {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    report = search.plan_search(checkout, checkout / "runs", spec)
    assert report["declared_worst_case_seconds"] == 40
    assert len(report["trials"]) == 4
    assert report["holdout_tasks"] == ["t2", "t3"]
    assert report["selection"]["selection_kind"] == "pending"
    assert report["selection"]["selected_candidate_id"] is None
    assert {t["resolved_recipe"]["name"] for t in report["trials"]} == {"ka2"}
    assert all(len(t["tasks"]) == 3 for t in report["trials"])
    assert before == {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}


def test_grid_union_retains_exact_existing_configuration_identities(checkout, spec):
    first = deepcopy(spec)
    first["grid"] = {"lr": [.003]}
    second = deepcopy(spec)
    second["grid"] = {"lr": [.005], "d_lr_mult": [.5, 2.]}
    expected = search._declarations(checkout, first) + search._declarations(checkout, second)
    for card, _ in expected:
        atomic_json(checkout / f"configs/forge/configurations/{card['id']}.json", card)
    before = {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    spec["grid"] = [first["grid"], second["grid"]]
    report = search.plan_search(checkout, checkout / "runs", spec)
    assert len(report["trials"]) == 3
    assert {t["candidate_id"] for t in report["trials"]} == {card["id"] for card, _ in expected}
    assert {stable_hash(t["settings"]) for t in report["trials"]} == {
        stable_hash(settings) for _, settings in expected}
    assert report["declared_worst_case_seconds"] == 30
    assert before == {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}


@pytest.mark.parametrize("grid", [[], [{"lr": [.003]}, {"lr": [.003]}],
                                  [[{"lr": [.003]}]], [{"seed": [0]}],
                                  [{"lr": [.003]}, {"lr": list(range(256))}]])
def test_invalid_grid_unions_are_rejected_before_declarations(checkout, spec, grid):
    spec["grid"] = grid
    with pytest.raises(ValueError):
        search.materialize_search(checkout, spec)
    assert not (checkout / "configs/forge/configurations").exists()


def test_tier1_refresh_materializes_only_all_32_existing_configuration_cards(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "configs/forge", tmp_path / "configs/forge")
    roster = read_json(tmp_path / "configs/forge/rounds/tier1-existing-configs-v1.json")
    campaign = read_json(tmp_path / roster["campaign"])
    cards = tmp_path / "configs/forge/configurations"
    before = {path.name: path.read_bytes() for path in cards.glob("*.json")}
    expected = set(roster["configuration_ids"])
    assert len(expected) == 32
    assert expected <= {Path(name).stem for name in before}
    assert len(roster["candidate_ids"]) == len(set(roster["candidate_ids"])) == 47
    assert set(roster["candidate_ids"]) == expected | set(roster["idea_ids"])
    assert campaign["budget_seconds"] == 47 * 2100
    assert campaign["candidate_budget_seconds"] == 2100
    observed = []
    for study_id in roster["studies"]:
        spec = search._load_spec(tmp_path, study_id)
        assert spec["tuning_through_tier"] == 1
        assert spec["campaign"] == campaign
        assert spec["execution_backend"] == "cuda"
        paths = search.materialize_search(tmp_path, spec)
        assert len(paths) == roster["configuration_count_by_family"][spec["trainer_family"]]
        observed.extend(path.stem for path in paths)
        for path in paths:
            card = read_json(path)
            search.validate_configuration_declaration(card, root=tmp_path)
    assert len(observed) == len(set(observed)) == 32
    assert set(observed) == expected
    assert before == {path.name: path.read_bytes() for path in cards.glob("*.json")}
    registry = read_json(tmp_path / "configs/forge/trainer-families.json")
    family_pins = {family["id"]: family.get("active_search_by_backend", {})
                   for family in registry["families"] + registry.get("historical_families", [])}
    assert family_pins["r1r2"]["cpu"] == "r1r2-modern-toy-v1"
    # Later independently registered families do not alter this frozen roster.
    assert {family_pins[family]["cuda"] for family in roster["configuration_count_by_family"]} == set(roster["studies"])


@pytest.mark.parametrize("knob", ["seed", "prior", "task", "budget", "batch_size", "total_steps", "z_dim",
                                  "sigma_rel", "name", "output_noise_std", "initializer", "reg_arm", "optimizer_family"])
def test_forbidden_protocol_architecture_sampling_or_mechanism_axes(checkout, spec, knob):
    spec["grid"] = {knob: [1, 2]}
    with pytest.raises(ValueError, match="forbidden|unknown"):
        search.plan_search(checkout, checkout / "runs", spec)


@pytest.mark.parametrize("grid", [{"lr": []}, {"lr": [.003, .003]},
                                  {"lr": [.003], "group": [{"lr": .005}]},
                                  {"group": [{"lr": .003}, {"reg_coeff": 1}]},
                                  {"typo": [1]}, {"lr": [-1]}])
def test_invalid_grid_and_public_recipe_validation(checkout, spec, grid):
    spec["grid"] = grid
    with pytest.raises(ValueError):
        search.plan_search(checkout, checkout / "runs", spec)


@pytest.mark.parametrize("grid", [{"lr": [True]}, {"reg_coeff": [False]},
                                  {"reg_every": [1.0]}, {"betas": [[False, .99]]},
                                  {"prior_betas": [[0, True]]}, {"eps": [float("inf")]}])
def test_search_values_have_explicit_numeric_types(checkout, spec, grid):
    spec["grid"] = grid
    with pytest.raises(ValueError, match="invalid hyperparameter value"):
        search.materialize_search(checkout, spec)
    assert not (checkout / "configs/forge/configurations").exists()


@pytest.mark.parametrize("grid, mechanism", [({"reg_coeff": [0.]}, "critic_penalty"),
                                            ({"amsgrad": [True]}, "amsgrad"),
                                            ({"prior_reg": [.1]}, "prior_regularization")])
def test_new_trials_preserve_base_mechanisms_before_writing(checkout, spec, grid, mechanism):
    spec["grid"] = grid
    with pytest.raises(ValueError, match=mechanism):
        search.materialize_search(checkout, spec)
    assert not (checkout / "configs/forge/configurations").exists()


@pytest.mark.parametrize("field", ["reg_coeff_anneal_end", "beta2_anneal_end"])
def test_inactive_schedule_axes_are_rejected(checkout, spec, field):
    spec["grid"] = {field: [.1, .3]}
    with pytest.raises(ValueError, match="inactive or task-owned"):
        search.materialize_search(checkout, spec)


def _behavioral_prior_reg_base(checkout):
    path = checkout / "configs/forge/ideas/base.json"
    base = read_json(path)
    base["recipe_overrides"]["prior_reg"] = .05
    base["host_adaptation"] = {"schema_version": 1, "recipe_fields": ["prior_reg"]}
    atomic_json(path, base)
    path = checkout / "configs/forge/tasks/t1.json"
    task = read_json(path)
    task["execution"].update(host="two_pole", prior={"kind": "particle_cloud", "sigma": 0,
        "standardize": False, "learnable": True, "exception_reason": "frozen behavioral host"})
    atomic_json(path, task)


def test_task_delegated_axes_are_rejected_when_every_tuning_task_owns_them(checkout, spec):
    _behavioral_prior_reg_base(checkout)
    spec["grid"] = {"prior_reg": [.1, .2]}
    with pytest.raises(ValueError, match="prior_reg is inactive or task-owned"):
        search.materialize_search(checkout, spec)


def test_axis_keeps_honest_scoped_ownership_when_a_scalar_task_uses_it(checkout, spec):
    _behavioral_prior_reg_base(checkout)
    spec["grid"] = {"prior_reg": [.1, .2]}
    spec["tuning_through_tier"] = 2
    assert len(search.materialize_search(checkout, spec)) == 2


def test_frozen_priors_cannot_supply_a_prior_optimizer_search_axis(checkout, spec):
    path = checkout / "configs/forge/tasks/t1.json"
    task = read_json(path)
    task["execution"]["prior"]["learnable"] = False
    atomic_json(path, task)
    spec["grid"] = {"prior_lr_mult": [1, 4]}
    with pytest.raises(ValueError, match="prior_lr_mult is inactive"):
        search.materialize_search(checkout, spec)


def test_bound_budgets_and_protocol_are_preflight_requirements(checkout, spec):
    spec["campaign"]["budget_seconds"] = 39
    with pytest.raises(ValueError, match="all declared"):
        search.plan_search(checkout, checkout / "runs", spec)
    spec["campaign"]["budget_seconds"] = 40
    spec["campaign"]["candidate_budget_seconds"] = 9
    with pytest.raises(ValueError, match="candidate budget"):
        search.plan_search(checkout, checkout / "runs", spec)
    spec["campaign"]["candidate_budget_seconds"] = 10
    spec["protocol_hash"] = "wrong"
    with pytest.raises(ValueError, match="fixed protocol"):
        search.plan_search(checkout, checkout / "runs", spec)


def test_cards_have_global_hash_identity_and_ordinary_loader_validates_it(checkout, spec):
    paths = search.materialize_search(checkout, spec)
    assert len(paths) == 4 and len(declaration_paths(checkout)) == 5
    assert len(discover_candidate_ids(checkout)) == 5
    assert search.materialize_search(checkout, spec) == paths
    card = read_json(paths[0])
    assert card["configuration_id"] == search.configuration_id(card)
    assert load_idea(checkout, card["id"]) == card
    card["recipe_overrides"]["lr"] = .007
    atomic_json(paths[0], card)
    with pytest.raises(ValueError, match="hash|contradict"):
        load_idea(checkout, card["id"])
    with pytest.raises(ValueError, match="hash|contradict"):
        resolve_idea(checkout, card["id"], declaration=card, view_id="stability", execution_backend="cpu")


def test_family_membership_is_checked_against_registry(checkout, spec):
    atomic_json(checkout / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [
        {"id": "different", "label": "Different", "canonical_candidate": "base", "candidates": ["base"]}]})
    with pytest.raises(ValueError, match="registered trainer_family"):
        search.plan_search(checkout, checkout / "runs", spec)


def test_idempotent_requests_and_no_source_identity_change_from_cards(checkout, spec):
    planned = search.plan_search(checkout, checkout / "runs", spec)
    first = search.enqueue_search(checkout, checkout / "runs", spec)
    second = search.enqueue_search(checkout, checkout / "runs", spec)
    assert [t["request_id"] for t in first["trials"]] == [t["request_id"] for t in second["trials"]]
    assert first["source_digest"] == planned["source_digest"]
    state = Queue(checkout / "runs").inspect()
    assert len(state["submissions"]) == 4 and len(state["jobs"]) == 12
    assert all(job["attempts"] == [] for job in state["jobs"].values())
    assert state["campaigns"][spec["campaign"]["id"]]["spent_seconds"] == 0


def test_freeze_all_before_any_submission_and_detect_source_drift(checkout, spec, monkeypatch):
    real, calls = search.resolve_idea, 0
    def drift(*args, **kwargs):
        nonlocal calls
        if kwargs.get("freeze_source"):
            calls += 1
            if calls == 2:
                (checkout / "particlegan/fixture.py").write_text("mechanism = 2\n")
        return real(*args, **kwargs)
    monkeypatch.setattr(search, "resolve_idea", drift)
    with pytest.raises(ValueError, match="changed before submission"):
        search.enqueue_search(checkout, checkout / "runs", spec)
    assert Queue(checkout / "runs").inspect()["submissions"] == {}


def test_registered_source_and_spec_cannot_change(checkout, spec):
    search.enqueue_search(checkout, checkout / "runs", spec)
    changed = deepcopy(spec)
    changed["grid"]["lr"][0] = .007
    with pytest.raises(ValueError, match="immutable"):
        search.enqueue_search(checkout, checkout / "runs", changed)
    (checkout / "particlegan/fixture.py").write_text("mechanism = 2\n")
    with pytest.raises(ValueError, match="source or scientific"):
        search.enqueue_search(checkout, checkout / "runs", spec)


def test_partial_admission_crash_preserves_identity_and_restarts_without_duplicates(checkout, spec, monkeypatch):
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    real, calls = queue.submit, 0
    def fail_second(request, campaign):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated admission crash")
        return real(request, campaign)
    monkeypatch.setattr(queue, "submit", fail_second)
    with pytest.raises(RuntimeError, match="crash"):
        search.enqueue_search(checkout, checkout / "runs", spec, queue=queue)
    assert len(queue.inspect()["submissions"]) == 1
    changed = deepcopy(spec)
    changed["grid"]["lr"][0] = .007
    with pytest.raises(ValueError, match="immutable"):
        search.enqueue_search(checkout, checkout / "runs", changed, queue=queue)
    monkeypatch.setattr(queue, "submit", real)
    result = search.enqueue_search(checkout, checkout / "runs", spec, queue=queue)
    assert result["submitted_count"] == 4 and len(queue.inspect()["submissions"]) == 4
    assert result["cost"]["new_paid_wall_seconds"] == 0


def test_report_regrades_status_stamps_and_rejects_corrupt_certificate(checkout, spec):
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    search.enqueue_search(checkout, checkout / "runs", spec, queue=queue)
    claim = complete(queue, 0, fabricated_gate="PASS")
    report = search.report_search(checkout, checkout / "runs", spec, queue=queue)
    row = next(t for t in report["trials"] if t["candidate_id"] == claim["request"]["candidate"]["id"])
    assert row["tasks"][0]["gate_status"] == "FAIL"
    receipt = checkout / "reports/forge/attempts" / claim["worker"]["attempt"] / "evidence.json"
    cert = read_json(receipt)
    cert["result_hash"] = "bad"
    atomic_json(receipt, cert)
    report = search.report_search(checkout, checkout / "runs", spec, queue=queue)
    row = next(t for t in report["trials"] if t["candidate_id"] == claim["request"]["candidate"]["id"])
    assert row["tasks"][0]["gate_status"] == "INVALID"
    assert not report["selection"]["qualified"]


def test_run_ordinary_fail_stops_holdouts_and_selects_whole_terminal_config(checkout, spec, monkeypatch):
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    calls = []
    def fake_drain(received, devices, *, campaign):
        calls.append((received, devices, campaign))
        for _ in range(4):
            complete(received, 0)
        assert received.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}]) is None
    monkeypatch.setattr(search, "drain", fake_drain)
    report = search.run_search(checkout, checkout / "runs", spec, devices=["cpu"], queue=queue)
    assert calls == [(queue, ["cpu"], spec["campaign"]["id"])]
    assert report["selection"]["selection_kind"] == "best_observed"
    assert report["selection"]["selected_configuration_id"] == min(t["configuration_id"] for t in report["trials"])
    assert not report["selection"]["qualified"] and not report["default_adoption"]
    assert report["cost"]["new_paid_wall_seconds"] == report["cost"]["evidence_wall_seconds"] == 8
    assert all(t["tasks"][0]["gate_status"] == "FAIL" and
               [task["gate_status"] for task in t["tasks"]][1:] == ["UNKNOWN", "UNKNOWN"] for t in report["trials"])
    before = deepcopy(report)
    (checkout / "particlegan/fixture.py").write_text("mechanism = 2\n")
    assert search.report_search(checkout, checkout / "runs", spec, queue=queue) == before


def test_reused_completed_science_adds_no_paid_attempts(checkout, spec):
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    first = search.enqueue_search(checkout, checkout / "runs", spec, queue=queue)
    for _ in range(4):
        complete(queue, 0)
    second = deepcopy(spec)
    second["id"] = "reuse-study"
    second["campaign"]["id"] = "reuse-campaign"
    report = search.enqueue_search(checkout, checkout / "runs", second, queue=queue)
    # Refresh ordinary prerequisite decisions without launching a worker.
    assert queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}], campaign_filter="reuse-campaign") is None
    report = search.report_search(checkout, checkout / "runs", second, queue=queue)
    assert report["source_digest"] == first["source_digest"]
    assert report["cost"]["new_paid_wall_seconds"] == 0
    assert report["cost"]["evidence_wall_seconds"] == 8
    assert len(queue.inspect()["jobs"]) == 12
    assert sum(len(job["attempts"]) for job in queue.inspect()["jobs"].values()) == 4
    assert report["selection"]["selection_kind"] == "best_observed"


def test_report_digest_tampering_rejected(checkout, spec):
    search.enqueue_search(checkout, checkout / "runs", spec)
    path = checkout / "reports/forge/configuration-search/study.json"
    packet = read_json(path)
    packet["trials"][0]["cost"]["wall_seconds"] = 999
    atomic_json(path, packet)
    with pytest.raises(ValueError, match="input digest"):
        search.report_search(checkout, checkout / "runs", spec)


def test_pure_selector_lexicographic_counts_qualified_and_pending():
    trials = [{"candidate_id": name, "configuration_id": name, "submission_status": "completed",
               "tasks": [{"task": "a", "qualification_tier": 1, "importance": "required",
                          "compatibility_key": name, "gate_status": status},
                         {"task": "holdout", "qualification_tier": 2, "importance": "required",
                          "compatibility_key": name, "gate_status": "PASS"}]}
              for name, status in [("aaa", "FAIL"), ("bbb", "PASS")]]
    selected = search.select_configuration(trials, 1)
    assert selected["selected_candidate_id"] == "bbb" and selected["qualified"]
    assert selected["required_total"] == selected["required_pass_count"] == 1
    trials[0]["submission_status"] = "running"
    assert search.select_configuration(trials, 1)["selected_candidate_id"] is None
    trials[0]["submission_status"] = "cancelled"
    assert search.select_configuration(trials, 1)["selection_kind"] == "pending"



def test_held_tiers_never_complete_a_full_view_or_break_progression():
    def trial(*rows):
        return {"tasks": [{"task": name, "qualification_tier": tier, "importance": role, "gate_status": status}
                          for name, tier, role, status in rows]}
    top = search._qualification(trial(("a", 1, "required", "PASS"), ("b", 2, "required", "PASS"),
                                      ("c", 3, "held", "FAIL")), 3)
    assert top["qualified_tier"] == 2 and not top["full_view_qualified"]
    assert top["required_total"] == top["required_passed"] == 2 and (top["held_total"], top["held_passed"]) == (1, 0)
    middle = search._qualification(trial(("a", 1, "required", "PASS"), ("b", 2, "held", "FAIL"),
                                         ("c", 3, "required", "PASS")), 3)
    assert middle["qualified_tier"] == 3 and middle["full_view_qualified"]
    assert [row["tier"] for row in middle["tiers"]] == [1, 3]
    legacy = search._qualification(trial(("a", 1, "required", "PASS"), ("b", 2, "required", "PASS")), 2)
    assert legacy["full_view_qualified"] and "held_total" not in legacy

def test_git_metadata_advance_without_scientific_drift_reuses_requests(checkout, spec, monkeypatch):
    from experiments.forge import planning
    real = planning.inspect_source
    commit = "first"
    def source(*args, **kwargs):
        return {**real(*args, **kwargs), "origin_commit": commit}
    monkeypatch.setattr(planning, "inspect_source", source)
    first = search.enqueue_search(checkout, checkout / "runs", spec)
    commit = "second"
    second = search.enqueue_search(checkout, checkout / "runs", spec)
    assert [t["request_id"] for t in first["trials"]] == [t["request_id"] for t in second["trials"]]
    assert len(Queue(checkout / "runs").inspect()["submissions"]) == 4


def test_grouped_attempt_cost_deduplicates_repeated_member_charges():
    assert search._attempt_seconds({"task_results": [
        {"cost": {"wall_seconds": 3}}, {"cost": {"wall_seconds": 3}}]}) == 3


def test_frozen_configuration_identity_survives_later_public_defaults(checkout, spec, monkeypatch):
    path = search.materialize_search(checkout, spec)[0]
    card = read_json(path)
    def different_defaults(*args, **kwargs):
        raise AssertionError("historical identity must not resolve today's public Recipe")
    monkeypatch.setattr(search, "_resolved_recipe", different_defaults)
    assert search.configuration_id(card, resolved_recipe=card["resolved_configuration_recipe"]) == card["configuration_id"]
    search.validate_configuration_declaration(card)  # Frozen evidence has no current checkout lineage.


def test_ordinary_configuration_validation_rejects_rehashed_structural_ablation(checkout, spec):
    path = search.materialize_search(checkout, spec)[0]
    card = read_json(path)
    card["recipe_overrides"]["reg_coeff"] = 0.
    card["resolved_configuration_recipe"]["reg_coeff"] = 0.
    digest = search.configuration_id(card, resolved_recipe=card["resolved_configuration_recipe"])
    card.update(configuration_id=digest, id=f"{card['trainer_family']}--{digest}")
    search.validate_configuration_declaration(card)  # Internally consistent identity is insufficient.
    with pytest.raises(ValueError, match="critic_penalty"):
        search.validate_configuration_declaration(card, root=checkout)
    target = path.with_name(f"{card['id']}.json")
    atomic_json(target, card)
    with pytest.raises(ValueError, match="critic_penalty"):
        load_idea(checkout, card["id"])
    with pytest.raises(ValueError, match="critic_penalty"):
        resolve_idea(checkout, card["id"], declaration=card, view_id="stability", execution_backend="cpu")


def test_current_configuration_requires_an_acyclic_declared_parent(checkout, spec):
    card = read_json(search.materialize_search(checkout, spec)[0])
    card["parent"] = card["id"]
    with pytest.raises(ValueError, match="parent cycle"):
        search.validate_configuration_declaration(card, root=checkout)
    card["parent"] = "missing-technique"
    with pytest.raises(ValueError, match="parent must name"):
        search.validate_configuration_declaration(card, root=checkout)


def test_rehashed_configuration_cannot_change_fixed_laws_or_nonsearch_knobs(checkout, spec):
    card = read_json(search.materialize_search(checkout, spec)[0])
    for field, value, message in [("claim_contract", {"sampling_law": "changed"}, "fixed parent"),
                                  ("latent_damping_max_rate", .4, "outside the hyperparameter")]:
        changed = deepcopy(card)
        if field == "claim_contract":
            changed[field] = value
        else:
            changed["recipe_overrides"][field] = value
            changed["resolved_configuration_recipe"][field] = value
        digest = search.configuration_id(changed, resolved_recipe=changed["resolved_configuration_recipe"])
        changed.update(configuration_id=digest, id=f"{changed['trainer_family']}--{digest}")
        with pytest.raises(ValueError, match=message):
            search.validate_configuration_declaration(changed, root=checkout)


def test_existing_positive_floor_round_retains_all_configuration_identities():
    root = Path(__file__).resolve().parents[1]
    study = "r1r2-modern-family-round1-v1"
    paths = [path for path in (root / "configs/forge/configurations").glob("*.json")
             if read_json(path).get("search_study_id") == study]
    before = {path: path.read_bytes() for path in paths}
    declarations = search._declarations(root, search._load_spec(root, study))
    assert len(declarations) == len(paths) == 8
    assert {card["configuration_id"] for card, _ in declarations} == {
        read_json(path)["configuration_id"] for path in paths}
    for path in paths:
        search.validate_configuration_declaration(read_json(path), root=root)
    assert before == {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("base_overrides", [
    {}, {"ebi_beta": 30.0, "kappa_radial": 1.0},
    {"ebi_beta": 15.0}, {"kappa_radial": 2.0}, {"ebi_beta": 15.0, "kappa_radial": 2.0},
], ids=["implicit-defaults", "explicit-defaults", "ebi-nondefault", "radial-nondefault", "both-nondefault"])
def test_actual_context_recipe_matches_frozen_card_and_report(checkout, spec, base_overrides):
    base_path = checkout / "configs/forge/ideas/base.json"
    base = read_json(base_path)
    base["recipe_overrides"].update(base_overrides)
    atomic_json(base_path, base)
    planned = search.plan_search(checkout, checkout / "runs", spec)
    for trial in planned["trials"]:
        card = trial["declaration"]
        # Full frozen cards retain the added fields; legacy runtime packets
        # omit only their original default values. Compare the entire expected
        # packet independently of the production compatibility projector.
        expected_packet = deepcopy(card["resolved_configuration_recipe"])
        for name, default in (("ebi_beta", 30.0), ("kappa_radial", 1.0), ("generator_gradient_response", "none"), ('reg_real_weight', 1.0), ('reg_fake_weight', 1.0), ('joint_reconstruction_weight', 0.0), ('conditional_energy_weight', 0.0), ('conditional_energy_samples', 8), ('masked_mixture_weight', 0.0), ('masked_transport_weight', 0.0), ('imputer_component_weight', 0.0), ('rhbi_weight', 0.0), ('rhbi_rho', 2.0), ('rhbi_tau', 2.0), ('rhbi_eps', 0.1), ('rhbi_iters', 8), ('rhbi_c_max', 20.0), ('rhbi_beta_ell', 0.1), ('rhbi_nn_quantile', 0.25), ('rhbi_critic_weight', 0.0), ('rhbi_self_coupling', 'hollow'), ('rhbi_relativistic_weight', 0.0), ('rhbi_lambda_floor', 0.0), ('rhbi_gamma_gauge', 0.0), ('rhbi_rapidity_screening', True), ('rhbi_scale_log_odds', False), ('rhbi_conditional_fibre', 'joint'), ('rhbi_akt_horizon', False), ('rhbi_akt_mode', 'sqrt'), ('rhbi_slack_gain', 0.0), ('rhbi_chord_weight', 0.0), ('rhbi_thermal_tau', 0.0), ('rhbi_thermal_shock_mu', 0.15), ('rhbi_settle_beta', 0.0), ('rhbi_settle_mode', 'residual'), ('rhbi_fibre_rel_blend', 0.0), ('rhbi_step_damp', 0.0), ('rhbi_cond_ot_weight', 0.0), ('rhbi_cond_ot_mode', 'kernel_ed'), ('rhbi_simplex_weight', 0.0), ('rhbi_cvx_mu_gain', 0.40), ('rhbi_simplex_chart_weight', 0.0), ('rhbi_tail_spill_gain', 0.0), ('rhbi_simplex_macro_settle', False)):
            assert expected_packet[name] == base_overrides.get(name, default)
            if expected_packet[name] == default:
                expected_packet.pop(name)
        assert stable_hash(expected_packet) == stable_hash(trial["resolved_recipe"])
        assert trial["configuration_id"] == search.configuration_id(card, resolved_recipe=trial["resolved_recipe"])
        assert card["prior"] == planned["base_declaration"]["prior"]
        assert trial["resolved_recipe"]["prior_kind"] == "mog"
        assert trial["resolved_recipe"]["standardize"] is False
        assert trial["technique_signature"] == planned["technique_signature"]


def test_changed_unspecified_default_rejected_before_queue_admission(checkout, spec, monkeypatch):
    from experiments.forge import api
    from experiments.forge.hostprofiles import _validate_candidate_identity
    path = search.materialize_search(checkout, spec)[0]
    card = read_json(path)
    original = resolve_idea(checkout, card["id"], view_id="stability", execution_backend="cpu")
    assert card["resolved_configuration_recipe"]["prior_reg"] == 0
    real = api.resolve_public_recipe
    def changed(candidate, **overrides):
        recipe = real(candidate, **overrides)
        return recipe.replace(prior_reg=.013)
    monkeypatch.setattr(api, "resolve_public_recipe", changed)
    with pytest.raises(ValueError, match="frozen Recipe"):
        resolve_idea(checkout, card["id"], view_id="stability", execution_backend="cpu", queue_root=checkout / "runs", freeze_source=True)
    # A supplied request cannot bypass the same check at the worker boundary.
    with pytest.raises(ValueError, match="frozen Recipe"):
        _validate_candidate_identity(original)
    assert not (checkout / "runs").exists()
    assert Queue(checkout / "runs").inspect()["submissions"] == {}


def test_plan_refuses_mixed_source_cohorts_during_resolution(checkout, spec, monkeypatch):
    real, calls = search.resolve_idea, 0
    def drift(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            (checkout / "particlegan/fixture.py").write_text("mechanism = 2\n")
        return real(*args, **kwargs)
    monkeypatch.setattr(search, "resolve_idea", drift)
    with pytest.raises(ValueError, match="one frozen source/runtime/protocol cohort"):
        search.plan_search(checkout, checkout / "runs", spec)
    assert not (checkout / "runs").exists()


def test_cli_search_stages_parse_without_extra_protocol_knobs():
    from experiments.forge.__main__ import parser
    for stage in ("plan", "enqueue", "run", "report"):
        assert parser().parse_args(["search", stage, "study"]).stage == stage
    with pytest.raises(SystemExit):
        parser().parse_args(["search", "run", "study", "--seed", "1"])


def test_full_search_advances_every_smoke_survivor_and_keeps_unknowns(checkout, spec):
    spec["tuning_through_tier"] = 3
    spec["campaign"].update(budget_seconds=120, candidate_budget_seconds=30)
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    planned = search.enqueue_search(checkout, checkout / "runs", spec, queue=queue)
    ids = [trial["candidate_id"] for trial in planned["trials"]]
    observed = []
    def score(claimed):
        candidate, task = claimed["request"]["candidate"]["id"], claimed["job"]["task_id"]
        observed.append((candidate, task))
        return 0 if (candidate, task) in {(ids[0], "t1"), (ids[1], "t2")} else 1
    for _ in range(9):
        complete(queue, score)
    assert queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}]) is None
    report = search.report_search(checkout, checkout / "runs", spec, queue=queue)
    assert set(observed) == ({(ids[0], "t1"), (ids[1], "t1"), (ids[1], "t2")} |
                             {(candidate, task) for candidate in ids[2:] for task in ("t1", "t2", "t3")})
    assert report["progression"]["smoke_survivor_candidate_ids"] == sorted(ids[1:])
    assert report["progression"]["full_view_qualified_candidate_ids"] == sorted(ids[2:])
    assert report["progression"]["outcome"] == "full_winner"
    assert report["progression"]["full_view_winner_candidate_id"] == ids[2]
    assert report["selection"] == search.select_configuration(report["trials"], 3)
    first = report["trials"][0]["qualification"]
    assert first["required_total"] == 3 and first["required_statuses"] == {"FAIL": 1, "UNKNOWN": 2}
    assert report["trials"][1]["qualification"]["qualified_tier"] == 1
    assert report["cost"]["new_paid_wall_seconds"] == 18
    assert report["speed_selection"]["status"] == "UNAVAILABLE"
    assert report["speed_selection"]["selected_candidate_id"] is None
    assert report["default_adoption"] is False
    assert all(task["evaluator_timing"]["status"] == "UNAVAILABLE"
               for trial in report["trials"] for task in trial["tasks"])


def test_smoke_only_winner_is_not_full_view_qualified(checkout, spec):
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    search.enqueue_search(checkout, checkout / "runs", spec, queue=queue)
    for _ in range(4):
        complete(queue, 1)
    assert queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}]) is None
    report = search.report_search(checkout, checkout / "runs", spec, queue=queue)
    assert report["selection"]["selection_kind"] == "qualified_winner"
    assert report["progression"]["outcome"] == "tuning_only_winner"
    assert report["progression"]["full_view_winner_candidate_id"] is None
    assert report["progression"]["full_view_qualified_candidate_ids"] == []
    assert all(trial["qualification"]["required_statuses"] == {"PASS": 1, "UNKNOWN": 2}
               for trial in report["trials"])


def test_qualification_requires_required_prefix_and_retains_blocked_denominator():
    trial = {"tasks": [{"task": "smoke", "qualification_tier": 1, "importance": "required", "gate_status": "BLOCKED"},
                       {"task": "quality", "qualification_tier": 2, "importance": "required", "gate_status": "PASS"},
                       {"task": "hold", "qualification_tier": 3, "importance": "required", "gate_status": "UNKNOWN"},
                       {"task": "diagnostic", "qualification_tier": 1, "importance": "diagnostic", "gate_status": "PASS"}]}
    result = search._qualification(trial, 3)
    assert result["qualified_tier"] == 0
    assert result["required_total"] == 3
    assert result["required_statuses"] == {"BLOCKED": 1, "PASS": 1, "UNKNOWN": 1}
    assert not result["full_view_qualified"] and not result["tuning_qualified"]


def test_progression_rejects_dropped_or_duplicate_view_tasks(checkout, spec):
    report = search.plan_search(checkout, checkout / "runs", spec)
    report["trials"][0]["tasks"].pop()
    with pytest.raises(ValueError, match="complete view task denominator"):
        search._annotate_progression(report)
    report = search.plan_search(checkout, checkout / "runs", spec)
    for trial in report["trials"]:
        trial["tasks"].append(deepcopy(trial["tasks"][0]))
    with pytest.raises(ValueError, match="complete view task denominator"):
        search._annotate_progression(report)


def test_recorded_suffix_time_is_preserved_without_becoming_acquisition_speed():
    original = {"confirmed_step": 80, "confirmed_seconds": 1.5, "stable_from_seconds": .9}
    grade = {"evaluator_result": {"convergence": original}}
    result = search._evaluator_timing({"evaluation": {"kind": "transfer_sustained"}}, grade)
    assert result["status"] == "RECORDED_EVALUATOR_ONLY"
    assert result["original_evaluator_convergence"] == original
    assert not result["speed_qualified"]
    assert "terminal passing suffix" in result["semantics"]
    assert original["confirmed_seconds"] == 1.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_recorded_seconds_remain_unavailable_and_json_finite(value):
    original = {"confirmed_step": 80, "confirmed_seconds": value}
    result = search._evaluator_timing({"evaluation": {"kind": "transfer_sustained"}},
                                     {"evaluator_result": {"convergence": original}})
    assert result["status"] == "UNAVAILABLE" and not result["speed_qualified"]
    assert result["original_evaluator_convergence"]["confirmed_seconds"] is None
    assert result["invalid_timing_fields"] == ["confirmed_seconds"]
    stable_hash(result)
    assert original["confirmed_seconds"] is value


def test_hold_steps_and_native_coverage_time_cannot_supply_joint_quality_seconds():
    hold = search._evaluator_timing({"evaluation": {"kind": "ring_hold"}},
                                    {"metrics": {"converged_step": 1400, "hold_budget_complete": True}})
    assert hold["status"] == "UNAVAILABLE"
    assert hold["original_evaluator_convergence"]["converged_step"] == 1400
    native = search._evaluator_timing({"evaluation": {"kind": "native_accuracy"}},
        {"evaluator_result": {"coverage": {"confirmed_seconds": 3}}})
    assert native["status"] == "UNAVAILABLE" and not native["speed_qualified"]
    assert native["original_evaluator_convergence"] == {}
