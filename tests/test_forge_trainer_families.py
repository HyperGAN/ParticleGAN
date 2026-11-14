"""Family publication selects complete, verified configurations without training."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil

import pytest

from experiments.forge import configuration_search as search
from experiments.forge import trainer_families as families
from experiments.forge.api import resolve_public_recipe
from experiments.forge.contracts import atomic_json, read_json, stable_hash

ROOT = Path(__file__).resolve().parents[1]


def persist(root, report):
    report["input_digest"] = stable_hash({key: value for key, value in report.items() if key != "input_digest"})
    atomic_json(root / f"reports/forge/configuration-search/{report['study_id']}.json", report)


@pytest.fixture
def study(tmp_path):
    shutil.copytree(ROOT / "configs/forge", tmp_path / "configs/forge")
    (tmp_path / families.CURRENT_SELECTION).unlink(missing_ok=True)
    spec = read_json(tmp_path / "configs/forge/searches/r1r2-modern-toy-v1.json")
    declarations = search._declarations(tmp_path, spec)
    assignments = read_json(tmp_path / "configs/forge/views/discriminator_stability.json")["assignments"]
    policy = stable_hash(read_json(tmp_path / "configs/forge/views/discriminator_stability.json"))
    runtime = {"execution_backend": "cpu", "runtime": {"python": "frozen"},
               "compute_profiles": {"cpu": {"model": "fixture-cpu"}}}
    protocol = read_json(tmp_path / "configs/forge/protocols/screening.json")
    catalogs = {"task_contracts": {}, "recipe_contracts": {}, "protocol_contracts": {stable_hash(protocol): protocol}}
    task_contracts = {}
    for assignment in assignments:
        contract = {"host": assignment["task"], "prior": {"kind": "mog"},
                    "initialization": "fixed", "steps": 80, "sampling": {"eval_output_noise": "clean"}}
        digest = stable_hash(contract)
        catalogs["task_contracts"][digest] = contract
        task_contracts[assignment["task"]] = digest
    rows, trials, cards = [], [], {}
    def row_for(card, recipe, *, source="a" * 64, failure=True, model=None):
        keys = {a["task"]: stable_hash({"candidate": card["id"], "task": a["task"]}) for a in assignments}
        statuses = {a["task"]: "FAIL" if failure and a["qualification_tier"] == 1 else "UNKNOWN" for a in assignments}
        recipe_digest = stable_hash(recipe)
        catalogs["recipe_contracts"][recipe_digest] = recipe
        return {"candidate_id": card["id"], "candidate_revision": stable_hash(card), "cohort": stable_hash(keys),
                "technique": card["id"], "runtime_cohort": deepcopy(model or runtime), "qualified_tier": 0,
                "status": "FAIL" if failure else "UNKNOWN", "attempt_ids": [],
                "bindings": {"source_digest": source, "protocol_sha256": spec["protocol_hash"],
                             "recipe_sha256": recipe_digest, "task_keys_sha256": stable_hash(keys),
                             "task_contracts": deepcopy(task_contracts), "prior": {"kind": "mog"},
                             "initializer": "fixed", "rng_sha256": "rng", "claim_contract": {"sampling_law": "clean"}},
                "tasks": [{"task_id": a["task"], "status": statuses[a["task"]]} for a in assignments if a["importance"] == "required"],
                "nonrequired_tasks": [{"task_id": a["task"], "status": statuses[a["task"]]} for a in assignments if a["importance"] != "required"],
                "tiers": {str(tier): {"passed": 0, "total": sum(a["importance"] == "required" and a["qualification_tier"] == tier for a in assignments)}
                          for tier in (1, 2, 3)}}, keys, statuses
    for card, settings in declarations:
        atomic_json(tmp_path / f"configs/forge/configurations/{card['id']}.json", card)
        cards[card["id"]] = card
        recipe = deepcopy(card["resolved_configuration_recipe"])
        row, keys, statuses = row_for(card, recipe)
        rows.append(row)
        trials.append({"candidate_id": card["id"], "configuration_id": card["configuration_id"], "trainer_family": "r1r2",
                       "settings": settings, "declaration": deepcopy(card), "recipe_overrides": card["recipe_overrides"],
                       "resolved_recipe": recipe, "candidate_revision": row["candidate_revision"], "source_digest": "a" * 64,
                       "runtime_cohort": deepcopy(runtime), "protocol_hash": spec["protocol_hash"], "submission_status": "terminal",
                       "tasks": [{**a, "compatibility_key": keys[a["task"]], "gate_status": statuses[a["task"]]} for a in assignments]})
    base = search._base_declaration(tmp_path, spec)
    cards[base["id"]] = base
    canonical, _, _ = row_for(base, search._resolved_recipe(base), source="b" * 64, failure=False)
    rows.append(canonical)
    report = {"study_id": spec["id"], "trainer_family": "r1r2", "spec": spec, "spec_hash": stable_hash(spec),
              "base_declaration": deepcopy(base), "view": spec["view"], "execution_backend": "cpu",
              "tuning_through_tier": 1, "protocol_hash": spec["protocol_hash"], "policy_fingerprint": policy,
              "runtime_cohort": runtime, "source_digests": ["a" * 64], "trials": trials,
              "selection": search.select_configuration(trials, 1)}
    persist(tmp_path, report)
    return tmp_path, rows, catalogs, cards, report, policy


def select(study, **options):
    root, rows, catalogs, cards, _, policy = study
    return families.select_family_rows(root, rows, catalogs, declarations=cards,
                                      view_id="discriminator_stability", policy_fingerprint=policy, **options)


def test_all_four_declared_configuration_ids_bind_actual_forge_prior_context(study):
    from experiments.forge.hostprofiles import profile_source_paths
    from experiments.forge.planning import resolve_idea
    from experiments.forge.technique_board import request_bindings
    root, _, _, cards, report, _ = study
    for task_path in (root / "configs/forge/tasks").glob("*.json"):
        task = read_json(task_path)
        referenced = set(task["evaluation"].get("sources", {})) | set(profile_source_paths(task))
        for relative in referenced:
            if (ROOT / relative).is_file():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, root / relative)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert len(report["trials"]) == 4
    for trial in report["trials"]:
        card = cards[trial["candidate_id"]]
        request = resolve_idea(root, card["id"], declaration=card, view_id=report["view"],
                               through_tier=1, execution_backend="cpu")
        recipe = request["candidate"]["resolved_recipe"]
        bindings = request_bindings(request)
        identity = stable_hash(search.recipe_identity_fields(recipe))
        assert identity == stable_hash(search.recipe_identity_fields(card["resolved_configuration_recipe"]))
        assert identity == stable_hash(search.recipe_identity_fields(trial["resolved_recipe"]))
        assert recipe["loss"] == card["resolved_configuration_recipe"].get("loss", "relativistic")
        assert recipe["prior_kind"] == "mog" and recipe["standardize"] is False
        assert request["candidate"]["prior"] == card["prior"] == report["base_declaration"]["prior"]
        assert search.configuration_id(card, resolved_recipe=recipe) == trial["configuration_id"]
        assert bindings["recipe_sha256"] == stable_hash(recipe) == stable_hash(bindings["recipe"])
        # Raw preset defaults differ; only the actual FormulationContext is the
        # publication identity for these learned-MoG Forge requests.
        assert asdict(resolve_public_recipe(card))["prior_kind"] != recipe["prior_kind"]
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_failed_study_selects_one_whole_configuration_and_preserves_every_trial(study):
    result = select(study)
    selected = result["rows"][0]
    assert len(result["rows"]) == 1 and len(result["configuration_rows"]) == 5
    assert selected["candidate_id"] == study[4]["selection"]["selected_candidate_id"]
    assert selected["selection"]["selection_kind"] == "best_observed"
    assert not selected["selection"]["qualified"] and not selected["selection"]["default_adoption"]
    assert selected["tiers"] == next(row["tiers"] for row in study[1] if row["candidate_id"] == selected["candidate_id"])
    assert {row["alternative_scope"] for row in result["configuration_rows"]} == {"selected", "comparable_trial", "archived_alternative"}
    assignments = read_json(study[0] / "configs/forge/views/discriminator_stability.json")["assignments"]
    required = {row["task"] for row in assignments if row["importance"] == "required"}
    assert {row["task_id"] for row in selected["tasks"]} == required
    assert selected["tiers"]["1"] == {
        "passed": 0,
        "total": sum(row["importance"] == "required" and row["qualification_tier"] == 1
                     for row in assignments),
    }


def test_pending_search_uses_canonical_configuration(study):
    root, _, _, _, report, _ = study
    report["trials"][0]["submission_status"] = "running"
    report["selection"] = search.select_configuration(report["trials"], 1)
    persist(root, report)
    selected = select(study)["rows"][0]
    assert selected["candidate_id"] == "r3gan-stacked-training-toy-v1"
    assert selected["selection"]["selection_kind"] == "canonical_fallback"


def test_frozen_recipe_selection_survives_future_default_change(study, monkeypatch):
    monkeypatch.setattr(search, "_resolved_recipe", lambda *args: pytest.fail("frozen publication resolved today's Recipe defaults"))
    assert select(study)["rows"][0]["candidate_id"] == study[4]["selection"]["selected_candidate_id"]


def test_frozen_selection_survives_current_protocol_change(study):
    root = study[0]
    atomic_json(root / "configs/forge/defaults.json", {"protocol": "future-protocol"})
    atomic_json(root / "configs/forge/protocols/screening.json", {"changed": "future protocol"})
    assert select(study)["rows"][0]["candidate_id"] == study[4]["selection"]["selected_candidate_id"]


def modern_study(study):
    """Keep the frozen numerical rows while declaring protocol-by-name v2."""
    root, _, _, _, report, _ = study
    spec = report["spec"]
    assert report["protocol_hash"] == spec.pop("protocol_hash")
    spec.update(schema_version=2, hypothesis="A bounded public-recipe rate comparison may improve Tier 1 acquisition.")
    report["spec_hash"] = stable_hash(spec)
    atomic_json(root / f"configs/forge/searches/{spec['id']}.json", spec)
    persist(root, report)
    return study


def test_v2_generated_protocol_identity_selects_the_same_verified_whole_row(study):
    expected = select(study)["rows"][0]
    modern_study(study)
    actual = select(study)["rows"][0]
    assert actual["candidate_id"] == expected["candidate_id"]
    assert actual["tasks"] == expected["tasks"]
    assert actual["selection"]["selection_kind"] == "best_observed"
    assert not actual["selection"]["qualified"]
    assert not actual["selection"]["default_adoption"]


@pytest.mark.parametrize("generated_hash", [None, "not-a-hash", "0" * 64])
def test_v2_generated_protocol_hash_must_bind_the_original_evidence(study, generated_hash):
    modern_study(study)
    root, _, _, _, report, _ = study
    report["protocol_hash"] = generated_hash
    persist(root, report)
    with pytest.raises(ValueError, match="generated protocol hash|verified protocol contracts"):
        select(study)


def test_v2_protocol_name_must_match_its_frozen_contract(study):
    modern_study(study)
    root, _, _, _, report, _ = study
    report["spec"]["protocol"] = "different-protocol"
    report["spec_hash"] = stable_hash(report["spec"])
    atomic_json(root / f"configs/forge/searches/{report['study_id']}.json", report["spec"])
    persist(root, report)
    with pytest.raises(ValueError, match="verified protocol contracts"):
        select(study)


def test_v1_authored_protocol_hash_remains_authoritative(study):
    root, _, _, _, report, _ = study
    report["protocol_hash"] = "0" * 64
    persist(root, report)
    with pytest.raises(ValueError, match="frozen study contract"):
        select(study)


def test_explicit_recorded_policy_preserves_selection_after_current_view_changes(study):
    root = study[0]
    path = root / "configs/forge/views/discriminator_stability.json"
    recorded = read_json(path)
    current = deepcopy(recorded)
    current["revision"] += 1
    current["assignments"].append({"task": "img_intensity2_residual16", "qualification_tier": 1,
                                   "importance": "required", "order": 10})
    atomic_json(path, current)
    with pytest.raises(ValueError, match="frozen view task denominator"):
        select(study)
    selected = select(study, view_policy=recorded)["rows"][0]
    assert selected["candidate_id"] == study[4]["selection"]["selected_candidate_id"]
    assert selected["tiers"] == next(row["tiers"] for row in study[1]
                                      if row["candidate_id"] == selected["candidate_id"])
    # Supplying an edited policy cannot relabel frozen trial evidence.
    with pytest.raises(ValueError, match="recorded policy"):
        select(study, view_policy=current)


@pytest.mark.parametrize("tamper,message", [
    ("qualification", "selection differs"), ("status", "status differs"),
    ("omitted_trial", "omitted or changed"), ("tier", "frozen study contract"),
    ("task_omitted", "task denominator"), ("settings", "omitted or changed"),
    ("recipe", "configuration identity"), ("source", "different source/runtime"),
    ("prior", "different source/runtime"), ("digest", "input digest"),
])
def test_search_cannot_edit_selection_or_scientific_identity(study, tamper, message):
    root, rows, _, _, report, _ = study
    if tamper == "qualification":
        report["selection"]["qualified"] = True
    elif tamper == "status":
        report["trials"][0]["tasks"][0]["gate_status"] = "PASS"
    elif tamper == "omitted_trial":
        report["trials"].pop()
    elif tamper == "tier":
        report["tuning_through_tier"] = 2
    elif tamper == "task_omitted":
        report["trials"][0]["tasks"].pop()
    elif tamper == "settings":
        report["trials"][0]["settings"]["lr"] *= 10
    elif tamper == "recipe":
        report["trials"][0]["resolved_recipe"]["lr"] *= 10
    elif tamper == "source":
        report["trials"][0]["source_digest"] = rows[0]["bindings"]["source_digest"] = "c" * 64
        report["source_digests"].append("c" * 64)
    elif tamper == "prior":
        rows[0]["bindings"]["prior"] = {"kind": "cloud"}
    elif tamper == "digest":
        report["selection"]["qualified"] = True
        atomic_json(root / f"reports/forge/configuration-search/{report['study_id']}.json", report)
    if tamper != "digest":
        persist(root, report)
    with pytest.raises(ValueError, match=message):
        select(study)


def test_hardware_and_cpu_cuda_families_stay_separate(study):
    root, rows, _, _, report, _ = study
    gpu = deepcopy(rows[-1])
    gpu["runtime_cohort"] = {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "gpu-a"}}}
    second_gpu = deepcopy(gpu)
    second_gpu["runtime_cohort"]["compute_profiles"]["cuda"]["model"] = "gpu-b"
    rows.extend([gpu, second_gpu])
    with pytest.raises(ValueError, match="explicit whole-row family selection"):
        select(study)
    recorded = read_json(root / "configs/forge/views/discriminator_stability.json")
    result = select(study, view_policy=recorded)
    assert len(result["rows"]) == 3
    assert sum(row["selection"]["selection_kind"] == "best_observed" for row in result["rows"]) == 1


def unmeasured_study(study):
    root, rows, _, _, _, _ = study
    cpu = deepcopy(rows[-1])
    gpu = deepcopy(cpu)
    gpu["runtime_cohort"] = {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "gpu-a"}}}
    rows[:] = [cpu, gpu]
    registry = read_json(root / families.REGISTRY)
    family = next(family for family in registry["families"] if family["id"] == "r1r2")
    family.pop("active_search_by_backend", None)
    family.update(unmeasured_display_backend="cuda",
                  unmeasured_display_reason="Declaration-only CUDA placeholder; no execution or qualification credit.")
    atomic_json(root / families.REGISTRY, registry)
    return registry, family


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_unmeasured_backend_keeps_whole_scaffold_and_every_alternative(study, reverse):
    _, family = unmeasured_study(study)
    rows = study[1]
    before = {families.scientific_row_hash(row) for row in rows}
    if reverse:
        rows.reverse()
    result = select(study)
    assert len(result["rows"]) == 1 and len(result["configuration_rows"]) == 2
    selected = result["rows"][0]
    assert selected["runtime_cohort"]["execution_backend"] == "cuda"
    assert selected["selection"]["selection_kind"] == "unmeasured_declaration"
    assert selected["selection"]["reason"] == family["unmeasured_display_reason"]
    assert selected["selection"]["qualified"] is False and selected["selection"]["default_adoption"] is False
    assert not selected["attempt_ids"] and all(task["status"] == "UNKNOWN" for task in selected["tasks"])
    assert families.scientific_row_hash(selected) in before
    assert {families.scientific_row_hash(row) for row in result["configuration_rows"]} == before
    assert sum(row["selected_configuration"] for row in result["configuration_rows"]) == 1


@pytest.mark.parametrize("tamper", ["attempt", "task", "gate", "incomplete_task", "nonrequired", "qualified", "passed",
                                    "counts", "cost_tasks", "cost_seconds", "status", "malformed_status"])
def test_unmeasured_backend_cannot_hide_any_executed_or_forged_outcome(study, tamper):
    unmeasured_study(study)
    # The nonselected CPU alternative must also be unmeasured.
    row = study[1][0]
    if tamper == "attempt":
        row["attempt_ids"] = ["ordinary-attempt"]
    elif tamper == "task":
        row["tasks"][0]["status"] = "PASS"
    elif tamper == "gate":
        row["tasks"][0]["gate_status"] = "FAIL"
    elif tamper == "incomplete_task":
        row["status"] = row["tasks"][0]["status"] = "INCOMPLETE"
    elif tamper == "nonrequired":
        row["nonrequired_tasks"][0]["status"] = "INVALID"
    elif tamper == "qualified":
        row["qualified_tier"] = 1
    elif tamper == "passed":
        row["tiers"]["1"]["passed"] = 1
    elif tamper == "counts":
        row["tiers"]["1"]["counts"] = {"PASS": 1}
    elif tamper == "cost_tasks":
        row["cost"] = {"measured_tasks": 1}
    elif tamper == "cost_seconds":
        row["cost"] = {"wall_seconds": .01}
    elif tamper == "status":
        row["status"] = "FAIL"
    elif tamper == "malformed_status":
        row["status"] = "malformed"
    with pytest.raises(ValueError, match="family r1r2: unmeasured display requires exclusively declaration-only"):
        select(study)


def test_unmeasured_not_run_aggregate_incomplete_retains_its_original_grade(study):
    unmeasured_study(study)
    for row in study[1]:
        row["status"] = "INCOMPLETE"  # views.qualify() maps a wholly NOT_RUN aggregate to this label.
    result = select(study)
    assert result["rows"][0]["status"] == "INCOMPLETE"
    assert result["rows"][0]["selection"]["selection_kind"] == "unmeasured_declaration"


@pytest.mark.parametrize("tamper", ["missing_backend", "missing_reason", "invalid_backend", "empty_reason", "active_search"])
def test_unmeasured_backend_requires_explicit_valid_registry_and_no_active_search(study, tamper):
    registry, family = unmeasured_study(study)
    if tamper == "missing_backend":
        family.pop("unmeasured_display_backend")
    elif tamper == "missing_reason":
        family.pop("unmeasured_display_reason")
    elif tamper == "invalid_backend":
        family["unmeasured_display_backend"] = "gpu"
    elif tamper == "empty_reason":
        family["unmeasured_display_reason"] = " "
    else:
        family["active_search_by_backend"] = {"cuda": "active-study"}
    atomic_json(study[0] / families.REGISTRY, registry)
    with pytest.raises(ValueError, match="family r1r2: unmeasured display"):
        select(study)


@pytest.mark.parametrize("tamper", ["missing", "duplicate"])
def test_unmeasured_backend_requires_one_canonical_scientific_row_without_hash_ranking(study, tamper):
    unmeasured_study(study)
    rows = study[1]
    if tamper == "missing":
        rows.pop()
    else:
        alternative = deepcopy(rows[-1])
        alternative["runtime_cohort"]["compute_profiles"]["cuda"]["model"] = "gpu-b"
        rows.append(alternative)
    with pytest.raises(ValueError, match="family r1r2: unmeasured display requires exactly one canonical row"):
        select(study)


def test_unmeasured_backend_does_not_implicitly_select_when_no_selector_is_declared(study):
    registry, family = unmeasured_study(study)
    family.pop("unmeasured_display_backend")
    family.pop("unmeasured_display_reason")
    atomic_json(study[0] / families.REGISTRY, registry)
    with pytest.raises(ValueError, match="family r1r2: multiple current runtime cohorts"):
        select(study)


def test_unmeasured_backend_leaves_recorded_policy_cohorts_separate(study):
    unmeasured_study(study)
    policy = read_json(study[0] / "configs/forge/views/discriminator_stability.json")
    result = select(study, view_policy=policy)
    assert len(result["rows"]) == 2
    assert all(row["selection"]["selection_kind"] == "canonical_fallback" for row in result["rows"])


def current_pin(study, row, *, kind="historical_incumbent", measurement_views=None, measurement_tasks=None):
    root, _, _, _, _, policy = study
    row = deepcopy(row)
    row["trainer_family"] = "r1r2"
    card = {"schema_version": 1, "scope": "whole_candidate_family_current", "default_adoption": False,
            "view": "discriminator_stability", "policy_fingerprint": policy,
            "selections": [families.family_row_pin(row, selection_kind=kind, reason="Explicit whole-row choice.",
                                                    measurement_views=measurement_views, measurement_tasks=measurement_tasks)]}
    atomic_json(root / families.CURRENT_SELECTION, card)
    return card


def test_current_pin_selects_ordinary_idea_and_retains_all_other_cohorts_unranked(study):
    root, rows, _, cards, _, _ = study
    repaired = deepcopy(rows[0])
    repaired.update(candidate_id="r1r2-global-repair-v1", candidate_revision="new-revision")
    repaired["bindings"]["source_digest"] = "c" * 64
    repaired["runtime_cohort"]["execution_backend"] = "cuda"
    repaired["attempt_ids"] = ["ordinary-attempt"]
    cards[repaired["candidate_id"]] = {"id": repaired["candidate_id"], "trainer_family": "r1r2"}
    rows.append(repaired)
    current_pin(study, rows[0])
    result = select(study)
    assert len(result["rows"]) == 1
    assert result["rows"][0]["candidate_id"] == rows[0]["candidate_id"]
    assert result["rows"][0]["selection"]["qualified"] is False
    assert result["rows"][0]["tiers"] == rows[0]["tiers"]
    assert len(result["configuration_rows"]) == len(rows)
    assert {row["alternative_scope"] for row in result["configuration_rows"]} == {"selected", "archived_alternative"}
    assert next(row for row in result["configuration_rows"] if row["candidate_id"] == repaired["candidate_id"])["tasks"] == repaired["tasks"]


@pytest.mark.parametrize("field", ["candidate_revision", "cohort", "runtime_cohort", "source_digest", "recipe_sha256",
                                   "task_keys_sha256", "protocol_sha256", "rng_sha256", "prior", "initializer",
                                   "claim_contract", "tasks", "tiers", "attempt_ids"])
def test_current_pin_binds_entire_scientific_row(study, field):
    rows = study[1]
    current_pin(study, rows[0])
    if field in rows[0]["bindings"]:
        rows[0]["bindings"][field] = "changed"
    elif field == "runtime_cohort":
        rows[0][field]["runtime"]["python"] = "changed"
    else:
        rows[0][field] = "changed"
    with pytest.raises(ValueError, match="exact verified scientific row"):
        select(study)


def test_configured_standard_requires_complete_tier1_and_ordinary_measurement(study):
    root, rows, _, _, _, _ = study
    selected = rows[0]
    current_pin(study, selected, kind="configured_standard")
    with pytest.raises(ValueError, match="ordinary measured"):
        select(study)
    selected["attempt_ids"] = ["ordinary-attempt"]
    current_pin(study, selected, kind="configured_standard")
    with pytest.raises(ValueError, match="every required Tier 1"):
        select(study)
    tier1 = {a["task"] for a in read_json(root / "configs/forge/views/discriminator_stability.json")["assignments"]
             if a["qualification_tier"] == 1 and a["importance"] == "required"}
    for task in selected["tasks"]:
        if task["task_id"] in tier1:
            task["status"] = "PASS"
    selected["qualified_tier"] = 1
    selected["tiers"]["1"]["passed"] = len(tier1)
    current_pin(study, selected, kind="configured_standard")
    result = select(study)
    assert result["rows"][0]["selection"]["qualified"] is True
    assert result["rows"][0]["selection"]["default_adoption"] is False
    assert result["rows"][0]["tasks"] == selected["tasks"]


def test_current_pin_does_not_change_recorded_policy_selection(study):
    current_pin(study, study[1][-1])
    recorded = read_json(study[0] / "configs/forge/views/discriminator_stability.json")
    assert select(study)["rows"][0]["candidate_id"] == study[1][-1]["candidate_id"]
    assert select(study, view_policy=recorded)["rows"][0]["candidate_id"] == study[4]["selection"]["selected_candidate_id"]


def test_generic_family_selection_filters_mixed_rows_by_requested_backend(study):
    rows = study[1]
    gpu = deepcopy(rows[-1])
    gpu["runtime_cohort"] = {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "gpu-a"}}}
    rows.append(gpu)
    current_pin(study, rows[0])
    cpu = select(study, execution_backend="cpu")
    assert len(cpu["rows"]) == 1 and len(cpu["configuration_rows"]) == 5
    assert all(row["runtime_cohort"]["execution_backend"] == "cpu" for row in cpu["configuration_rows"])
    assert cpu["rows"][0]["candidate_id"] == rows[0]["candidate_id"]
    cuda = select(study, execution_backend="cuda")
    assert len(cuda["rows"]) == len(cuda["configuration_rows"]) == 1
    assert cuda["rows"][0]["runtime_cohort"]["execution_backend"] == "cuda"


def test_current_only_task_history_membership_preserves_recorded_family_identity():
    assert families.family_for_candidate(ROOT, "five-word-joint-ka2-v1", current_presentation=True)["id"] == "ka2"
    assert families.family_for_candidate(ROOT, "five-word-joint-ka2-v1")["id"] == "five-word-joint-ka2-v1"


def test_pure_adam_example_groups_only_for_current_presentation():
    candidate = "bcap-pure-adam-example-v3"
    declaration = read_json(ROOT / f"configs/forge/ideas/{candidate}.json")
    registry = families.load_families(ROOT)
    assert candidate not in registry["bcap-pure"]["candidates"]
    assert registry["bcap-pure"]["current_presentation_candidates"] == [candidate]
    assert registry["bcap-pure"]["canonical_candidate"] == "bcap-pure-adam-v2"
    current = families.family_for_candidate(ROOT, candidate, declaration, current_presentation=True)
    recorded = families.family_for_candidate(ROOT, candidate, declaration, current_presentation=False)
    assert current == registry["bcap-pure"]
    assert recorded == {"id": candidate, "label": candidate, "canonical_candidate": candidate,
                        "candidates": [candidate], "registration": "unclassified_independent_technique"}
    assert families.family_for_candidate(ROOT, "bcap-pure-adam-v2")["id"] == "bcap-pure"


def test_registry_groups_gan_v3_task_priors_and_keeps_original_historical_identities():
    registry = families.load_families(ROOT)
    retained = {"r1r2", "bcap", "k3p", "ka2", "e22", "atlas", "release07-gan-v3",
                "k3p-no-anchor", "k3p-no-penalty", "k3p-no-a2", "k3p-no-training-noise"}
    optimizer_families = {"bcap-sgda", "bcap-nsgda-global", "bcap-nsgda-layer", "bcap-ada-nsgda",
                          "bcap-dualnorm", "bcap-dualnorm-d-only", "bcap-particle-rownorm-only"}
    editorial_variants = {"bcap-develop-integration-" + name + "-v1" for name in ("combined", "winner")}
    editorial_variants.update("bcap-projection-baseline-review-" + name + "-v1" for name in ("baseline", "direction"))
    editorial_variants.update("bcap-three-phase-" + name + "-v1" for name in ("cap-margin", "finite-cap", "incumbent"))
    editorial_variants.update("bcap-tier1-stability-" + name + "-v1" for name in
                             ("combined", "incumbent", "projection-global", "projection-local", "projection",
                              "repairs-cap-margin", "repairs-finite-cap", "transport"))
    assert set(registry) == retained | {"bcap-pure", "halloween", "bispf", "empf", "rhbi"} | optimizer_families | editorial_variants
    assert registry["bispf"]["canonical_candidate"] == "bispf-v2"
    assert registry["empf"]["canonical_candidate"] == "empf-v7"
    assert registry["rhbi"]["canonical_candidate"] == "rhbi-v11"
    assert registry["halloween"]["canonical_candidate"] == "halloween-optimizer-loss-v1"
    assert registry["bcap-pure"]["canonical_candidate"] == "bcap-pure-adam-v2"
    assert set(registry["bcap-pure"]["candidates"]).isdisjoint(registry["bcap"]["candidates"])
    assert families.family_for_candidate(ROOT, "bcap-pure-adam-v1")["id"] == "bcap-pure"
    assert families.family_for_candidate(ROOT, "k3p-bcap-matched-v1")["id"] == "bcap"
    assert registry["r1r2"]["canonical_candidate"] == "r3gan-stacked-training-toy-v1"
    assert families.family_for_candidate(ROOT, "k3p-r1r2-matched-v1")["id"] == "r1r2"
    assert families.family_for_candidate(ROOT, "release07-gan-v3-cloud-v1")["id"] != families.family_for_candidate(ROOT, "release07-gan-v3-task-adapted-v1")["id"]
    for candidate in ("release07-gan-v3-cloud-v1", "release07-gan-v3-mog-v1", "release07-gan-v3-task-adapted-v1"):
        assert families.family_for_candidate(ROOT, candidate, current_presentation=True)["id"] == "release07-gan-v3"
    assert registry["release07-gan-v3"]["canonical_candidate"] == "release07-gan-v3-task-adapted-v1"
    assert families.family_for_candidate(ROOT, "forge-no-critic-penalty")["id"] != families.family_for_candidate(ROOT, "k3p")["id"]


def test_current_measurement_accepts_failures_without_qualification_and_requires_current_contracts(study):
    from experiments.forge.views import task_execution_fingerprint, task_evaluation_fingerprint
    root, rows, catalogs, _, _, _ = study
    selected = rows[0]
    selected["attempt_ids"] = ["ordinary-attempt"]
    required = {assignment["task"] for assignment in read_json(root / "configs/forge/views/discriminator_stability.json")["assignments"]
                if assignment["qualification_tier"] == 1 and assignment["importance"] == "required"}
    for name in required:
        task = read_json(root / "configs/forge/tasks" / (name + ".json"))
        contract = {"execution_sha256": task_execution_fingerprint(task),
                    "evaluation_sha256": task_evaluation_fingerprint(task),
                    "timeout_seconds": task["resources"]["timeout_seconds"]}
        digest = stable_hash(contract)
        selected["bindings"]["task_contracts"][name] = digest
        catalogs["task_contracts"][digest] = contract
    current_pin(study, selected, kind="current_measurement", measurement_views=["discriminator_stability"])
    result = select(study)
    metadata = result["rows"][0]["selection"]
    assert metadata["measurement_complete"] is True and metadata["qualified"] is False
    assert metadata["measured_required_tasks"] == sorted(required)
    first = next(task for task in selected["tasks"] if task["task_id"] in required)
    first["status"] = "UNKNOWN"
    current_pin(study, selected, kind="current_measurement", measurement_views=["discriminator_stability"])
    with pytest.raises(ValueError, match="every required Tier 1 task"):
        select(study)
    first["status"] = "FAIL"
    digest = selected["bindings"]["task_contracts"][first["task_id"]]
    catalogs["task_contracts"][digest]["evaluation_sha256"] = "old-law"
    current_pin(study, selected, kind="current_measurement", measurement_views=["discriminator_stability"])
    with pytest.raises(ValueError, match="current execution, evaluation and budget"):
        select(study)


def test_current_measurement_requires_explicit_views_and_additional_probe(study):
    selected = study[1][0]
    selected["attempt_ids"] = ["ordinary-attempt"]
    current_pin(study, selected, kind="current_measurement")
    with pytest.raises(ValueError, match="explicit distinct measurement views"):
        select(study)
    current_pin(study, selected, kind="current_measurement", measurement_views=["discriminator_stability"],
                measurement_tasks=["clockfree_audit"])
    with pytest.raises(ValueError, match="every required Tier 1 task"):
        select(study)


def test_prior_family_regrouping_selects_one_whole_row_without_borrowing(study):
    root, rows, catalogs, cards, _, policy = study
    selected, alternative = deepcopy(rows[0]), deepcopy(rows[1])
    selected.update(candidate_id="release07-gan-v3-task-adapted-v1", trainer_family="release07-gan-v3")
    alternative.update(candidate_id="release07-gan-v3-cloud-v1", trainer_family="release07-gan-v3")
    alternative["tasks"][0]["status"] = "PASS"
    for row in (selected, alternative):
        cards[row["candidate_id"]] = read_json(root / "configs/forge/ideas" / (row["candidate_id"] + ".json"))
    card = {"schema_version": 1, "scope": "whole_candidate_family_current", "default_adoption": False,
            "view": "discriminator_stability", "policy_fingerprint": policy,
            "selections": [families.family_row_pin(selected, selection_kind="historical_incumbent", reason="Whole source.")]}
    atomic_json(root / families.CURRENT_SELECTION, card)
    result = families.select_family_rows(root, [selected, alternative], catalogs, declarations=cards,
                                        view_id="discriminator_stability", policy_fingerprint=policy)
    assert len(result["rows"]) == 1
    assert result["rows"][0]["tasks"] == selected["tasks"]
    assert result["rows"][0]["tasks"][0]["status"] == "FAIL"
    assert result["configuration_rows"][1]["tasks"][0]["status"] == "PASS"


def test_original_prior_page_survives_in_archived_policy_without_filling_current_cells(study):
    root, rows, catalogs, cards, _, policy = study
    current, previous = deepcopy(rows[0]), deepcopy(rows[1])
    current.update(candidate_id="release07-gan-v3-task-adapted-v1", trainer_family="release07-gan-v3")
    previous.update(candidate_id="release07-gan-v3-cloud-v1", trainer_family="release07-gan-v3-cloud")
    previous["tasks"][0]["status"] = "PASS"
    previous["bindings"]["source_digest"] = "original-source"
    cards[current["candidate_id"]] = read_json(root / "configs/forge/ideas" / (current["candidate_id"] + ".json"))
    card = {"schema_version": 1, "scope": "whole_candidate_family_current", "default_adoption": False,
            "view": "discriminator_stability", "policy_fingerprint": policy,
            "selections": [families.family_row_pin(current, selection_kind="historical_incumbent", reason="Current whole source.")],
            "historical_selections": [families.family_row_pin(previous, selection_kind="historical_incumbent", reason="Exact archive.")]}
    atomic_json(root / families.CURRENT_SELECTION, card)
    result = families.select_family_rows(root, [current], catalogs, declarations=cards,
                                        view_id="discriminator_stability", policy_fingerprint=policy,
                                        historical_rows=[previous])
    assert result["rows"][0]["tasks"][0]["status"] == "FAIL"
    assert families.scientific_row_hash(result["historical_family_rows"][0]) == families.scientific_row_hash(previous)


def test_new_release_search_can_use_current_family_without_rewriting_old_search_ids(study):
    root = study[0]
    old = read_json(root / "configs/forge/searches/release07-gan-v3-mog-tier1-refresh-v1.json")
    spec = {**old, "trainer_family": "release07-gan-v3"}
    declarations = search._declarations(root, spec)
    assert all(card["trainer_family"] == "release07-gan-v3" for card, _ in declarations)
    assert all(card["id"].startswith("release07-gan-v3--") for card, _ in declarations)
    for card, _ in declarations:
        search.validate_configuration_declaration(card, root=root)
    assert read_json(root / "configs/forge/searches" / (old["id"] + ".json")) == old
