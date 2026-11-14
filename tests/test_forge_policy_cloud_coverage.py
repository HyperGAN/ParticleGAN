"""Software controls for the scoped Atlas/E22 policy-cloud coverage cohorts.

Two scoring laws: clean (learned output noise removed from measured samples)
and as served (kept exactly as GANTrainer serves it). No qualification
experiment runs here. Variants are materialized in memory from the current
parent files and sources, so later source edits cannot make these controls
depend on stale committed pins.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from experiments.forge import attempt_coverage as runner
from experiments.forge import grouped_evidence as reuse
from experiments.forge.adapters import adapter_preflight
from experiments.forge.api import CapabilityError
from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.policy_coverage import (BLOCKED_HOSTS, COHORT, COVERAGE, DIAGNOSTIC_CONTINUATIONS, LABEL,
                                               LAW_INDEPENDENT, PARENTS, SERVED, SERVED_COHORT, SERVED_LABEL,
                                               SERVED_PARENTS, SERVED_VIEW, SOURCES, TIERS, VIEW, served_row_task,
                                               served_view_declaration, variant_id, view_declaration)
from experiments.forge.tier1_policy import (TIER1, is_policy_task, materialize, schedule_free_policy, validate,
                                            validate_evidence)
from experiments.forge.views import grade_result, load_tasks, task_evaluation_fingerprint, task_execution_fingerprint

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("atlas", "e22")
COMPATIBLE = tuple(name for name in PARENTS if name not in BLOCKED_HOSTS)


def parent(name):
    return json.loads((ROOT / "configs/forge/tasks" / (name + ".json")).read_text())


def candidate(name):
    return json.loads((ROOT / "configs/forge/ideas" / (name + ".json")).read_text())


@pytest.fixture(scope="module")
def variants():
    return {name: materialize(parent(name), ROOT, COVERAGE) for name in PARENTS}


@pytest.fixture(scope="module")
def served():
    return {name: materialize(parent(name), ROOT, SERVED) for name in SERVED_PARENTS}


def _undo(variant, original):
    """Reverse exactly the declared transformation; anything left over changed."""
    task = deepcopy(variant)
    for key in ("task_cohort", "policy_parent"):
        task.pop(key)
    if "task_cohort" in original:
        task["task_cohort"] = original["task_cohort"]
    task["id"] = original["id"]
    task["requires_capabilities"] = deepcopy(original["requires_capabilities"])
    for dependency in task["dependencies"]:
        assert dependency["task"].endswith(COVERAGE.suffix)
        dependency["task"] = dependency["task"].removesuffix(COVERAGE.suffix)
    execution = task["execution"]
    for key in ("policy_parent_definition", "policy_contract", "policy_clock_horizon"):
        execution.pop(key, None)
    if "continuation_of" in execution:
        assert execution["continuation_of"].endswith(COVERAGE.suffix)
        execution["continuation_of"] = execution["continuation_of"].removesuffix(COVERAGE.suffix)
    for key in ("prior", "device", "resources"):
        if key in original["execution"]:
            execution[key] = deepcopy(original["execution"][key])
        else:
            execution.pop(key, None)
    task["resources"] = deepcopy(original["resources"])
    evaluation = task["evaluation"]
    added = set(evaluation["sources"]) - set(original["evaluation"].get("sources", {}))
    assert added <= set(SOURCES) | {"configs/forge/tasks/" + original["id"] + ".json"}
    for key, digest in original["evaluation"].get("sources", {}).items():
        if key not in SOURCES:
            assert evaluation["sources"][key] == digest  # Parent evaluator pins never change.
    evaluation["sources"] = deepcopy(original["evaluation"].get("sources", {}))
    if "sources" not in original["evaluation"]:
        evaluation.pop("sources")
    evaluation.pop("policy_observation")
    evaluation["scoring_weights"] = original["evaluation"]["scoring_weights"]
    evaluation["sampling_law"] = original["evaluation"]["sampling_law"]
    return task


@pytest.mark.parametrize("name", PARENTS)
def test_transformation_changes_only_declared_fields(name, variants):
    original, variant = parent(name), variants[name]
    assert _undo(variant, original) == original
    execution, evaluation = variant["execution"], variant["evaluation"]
    assert variant["id"] == variant_id(name) and variant["task_cohort"] == COHORT
    assert execution["policy_parent_definition"] == original
    prior = execution["prior"]
    assert (prior["kind"], prior["sigma"], prior["standardize"]) == ("particle_cloud", 0., False)
    # A recipe-owned prior leaves learnability to Recipe.prior_update; a legacy
    # task-owned prior keeps its explicit learned table.
    recipe_owned = original["execution"].get("prior_contract") == "recipe_owned_v1"
    assert ("learnable" not in prior) if recipe_owned else (prior["learnable"] is True)
    assert prior["exception_reason"].strip()
    contract = execution["policy_contract"]
    assert contract["recipe_overrides"] == {} and contract["original_parent_credit"] is False
    assert contract["schedule"] == "preserve_resolved_recipe_none" and contract["cohort"] == COHORT
    assert set(contract["sources"]) == set(SOURCES)
    assert evaluation["scoring_weights"] == "state_selected"
    assert evaluation["sampling_law"] == "policy_cloud_" + original["evaluation"]["sampling_law"]
    observation = evaluation["policy_observation"]
    assert observation["output_noise"] is False and observation["eval_streams"] == "forge-rng-v1_isolated"
    assert observation["sampler"] == ("served_snapshot" if name in COVERAGE.parameter_parents else "GANTrainer.sample")
    assert ("policy_clock_horizon" in execution) == (original["adapter"] == "clockfree_audit")
    # Budget, timeout, cadence and gates are inherited unchanged.
    assert execution["steps"] == original["execution"]["steps"]
    assert variant["resources"]["timeout_seconds"] == original["resources"]["timeout_seconds"]
    assert variant["resources"]["gpu_memory_mb"] == max(2048, original["resources"]["gpu_memory_mb"])
    for key in ("thresholds", "observations", "eval_interval", "conditions", "start_step"):
        assert evaluation.get(key) == original["evaluation"].get(key)
    caps = variant["requires_capabilities"]
    assert "live_sampling" not in caps and "mog_prior" not in caps
    assert {"policy_controls", "policy_serving"} <= set(caps)
    assert validate(variant, root=ROOT) == original


def test_original_tier1_cohort_keeps_its_exact_transformation():
    for name in ("two_pole", "clockfree_audit", "gaussian1d_acquisition"):
        variant = materialize(parent(name), ROOT)
        assert variant["task_cohort"] == TIER1.id and variant["dependencies"] == parent(name)["dependencies"]
        assert variant["evaluation"]["sampling_law"].startswith("tier1_selected_")
    committed = sorted((ROOT / TIER1.directory).glob("*.json"))
    assert len(committed) == 7 and all(validate(read_json(path)) for path in committed)


def test_committed_declarations_view_and_tiers_cover_all_selected_tests():
    tasks = load_tasks(ROOT)
    committed = {task["policy_parent"]["id"]: task for task in tasks.values() if task.get("task_cohort") == COHORT}
    assert set(committed) == set(PARENTS) and len(PARENTS) == 35
    assert {tier: len(names) for tier, names in TIERS.items()} == {1: 7, 2: 21, 3: 7}
    assert len(BLOCKED_HOSTS) == 9 and len(COMPATIBLE) == 26
    view = read_json(ROOT / "configs/forge/views" / (VIEW + ".json"))
    assert view == view_declaration() and view["reporting"] == {"family_totals": False, "families": ["atlas", "e22"]}
    assignments = {row["task"]: row for row in view["assignments"]}
    assert all(row["importance"] == "required" for row in assignments.values())
    for tier, names in TIERS.items():
        assert {assignments[variant_id(name)]["qualification_tier"] for name in names} == {tier}
    # The as-served cohort: 23 own variants; its view shares the 12 law-independent clean declarations.
    as_served = {task["policy_parent"]["id"]: task for task in tasks.values() if task.get("task_cohort") == SERVED_COHORT}
    assert set(as_served) == set(SERVED_PARENTS) and len(SERVED_PARENTS) == 23 and len(LAW_INDEPENDENT) == 12
    assert set(SERVED_PARENTS) | set(LAW_INDEPENDENT) == set(PARENTS)
    served_view = read_json(ROOT / "configs/forge/views" / (SERVED_VIEW + ".json"))
    assert served_view == served_view_declaration() and served_view["reporting"] == view["reporting"]
    rows = {row["task"]: row for row in served_view["assignments"]}
    for tier, names in TIERS.items():
        assert {rows[served_row_task(name)]["qualification_tier"] for name in names} == {tier}
    assert {name for name in rows if tasks[name]["task_cohort"] == COHORT} == {variant_id(n) for n in LAW_INDEPENDENT}
    # Every selected-view test keeps its recorded tier. Held and removed-from-count
    # tests still run (attempt_coverage.selected_tasks), so the cohort covers them too.
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    views = {v["id"]: v for v in publication["family_progress"]["views"]}
    family = next(f for f in publication["family_progress"]["families"] if f["id"] == "atlas")
    cohort = next(c for c in family["cohorts"] if c["anchor"] == family["current_cohort_anchor"])
    selected = {(a["task"], a["qualification_tier"]) for v in cohort["views"] for a in views[v["id"]]["assignments"]
                if a["importance"] in {"required", "held"} or "removal" in a}
    revision_13_new_tasks = {
        ("zero_tanh_smoke", 1),
        ("zero_tanh_rail_escape", 3),
        ("grid100_mlp16", 4),
        ("grid100_res16_tanh", 4),
        ("misgan_block_recovery", 4),
        ("misgan_block_neural_imputer", 4),
        ("ring16_bigan_continuous", 4),
    }
    # The frozen policy cohorts do not acquire new task variants implicitly.
    revision_14_new_tasks = {
        ("model_ae_reconstruction", 1),
        ("model_ae_gan_multimodal", 1),
        ("model_vae_hard", 1),
        ("model_vae_categorical", 1),
        ("model_ddgan_gaussian_chain", 2),
        ("model_ae_ddgan_gaussian_chain", 2),
        ("model_ddgan_bimodal_chain", 2),
        ("model_ae_ddgan_bimodal_chain", 2),
    }
    assert selected == (({(name, tier) for tier, names in TIERS.items() for name in names}
                         - {("ring16_acquisition", 1)}) | {("ring16_acquisition_6400", 1)}
                        | revision_13_new_tasks | revision_14_new_tasks)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", PARENTS)
def test_preflight_runs_compatible_variants_and_blocks_only_declared_hosts(family, name, variants):
    reasons = adapter_preflight(variants[name], candidate(family), root=ROOT)
    if name in BLOCKED_HOSTS:
        assert reasons == [variant_id(name) + ": " + BLOCKED_HOSTS[name]]
    else:
        assert reasons == []


@pytest.mark.parametrize("name", sorted(BLOCKED_HOSTS))
def test_blocked_hosts_refuse_before_any_host_setup(name, variants, tmp_path):
    from experiments.forge.adapters import _dispatch_task
    request = {"candidate": candidate("atlas"), "protocol": {"seed": 0}, "tasks": {variant_id(name): variants[name]}}
    with pytest.raises(CapabilityError) as refused:
        _dispatch_task(request, {"task_id": variant_id(name)}, tmp_path, "cpu")
    assert BLOCKED_HOSTS[name] in str(refused.value)
    assert not any(tmp_path.iterdir())


def test_routing_reaches_each_cohort_adapter(variants, served, tmp_path, monkeypatch):
    from experiments.forge import adapters, policy_behavior_adapters, policy_gaussian
    calls = []
    def recorder(label):
        def call(request, task, output, device, **kwargs):
            calls.append((label, task["policy_parent"]["id"], kwargs))
            return {"cost": {}}
        return call
    monkeypatch.setattr(adapters, "_ring", recorder("ring"))
    monkeypatch.setattr(adapters, "_native", recorder("native"))
    monkeypatch.setattr(adapters, "_vector", recorder("vector"))
    monkeypatch.setattr(adapters, "_image", recorder("image"))
    monkeypatch.setattr(policy_behavior_adapters, "run_behavior", recorder("policy_behavior"))
    monkeypatch.setattr(policy_gaussian, "run_gaussian", recorder("policy_gaussian"))
    expected = {"mode_hold": "ring", "ring_hold": "ring", "two_pole": "policy_behavior",
                "gaussian1d_smoke": "policy_gaussian", "gaussian1d_stability": "policy_gaussian",
                "grid100": "native", "grid100_14k": "native", "vector_spiral": "vector",
                "ring16_acquisition": "vector", "img_bars4": "image"}
    for name, label in expected.items():
        for cards in (variants, served):
            if name not in cards:
                assert name in LAW_INDEPENDENT  # Shared tests run once, in the clean cohort.
                continue
            task = cards[name]
            job = {"task_id": task["id"], "diagnostic_continuation": name in DIAGNOSTIC_CONTINUATIONS}
            adapters._dispatch_task({"candidate": candidate("e22"), "tasks": {task["id"]: task}}, job, tmp_path, "cpu")
            assert calls[-1][:2] == (label, name)
            if name in {"gaussian1d_stability", "grid100_14k"}:
                assert calls[-1][2]["diagnostic"] is True


def test_policy_gaussian_runs_unchanged_evaluator_code_with_parent_constants(variants, served):
    from experiments.forge import gaussian_tasks, policy_gaussian
    for name in ("gaussian1d_smoke", "gaussian1d_stability"):
        for task in (variants[name], served[name]):
            policy_gaussian.validate_task(task)
            with pytest.raises(ValueError, match="fixed by its evaluator"):
                gaussian_tasks.validate_task(task)
    # The evaluator's own code objects execute; only the validator and the
    # constructor's measurement-law binding are substituted in their scope.
    assert policy_gaussian._ORIGINAL_RUN.__code__ is gaussian_tasks.run_gaussian.__code__
    assert policy_gaussian._ORIGINAL_BUILD.__code__ is gaussian_tasks.build.__code__
    assert policy_gaussian.grade.__code__ is gaussian_tasks.grade.__code__
    assert policy_gaussian._SCOPE["build"] is policy_gaussian.build
    assert policy_gaussian._SCOPE["validate_task"] is policy_gaussian.validate_task
    assert policy_gaussian.build.__kwdefaults__ == gaussian_tasks.build.__kwdefaults__
    with pytest.raises(ValueError, match="registered scoped variant"):
        policy_gaussian.validate_task(parent("gaussian1d_smoke"))


def _policy_evidence(task, final):
    declaration = task["evaluation"]["policy_observation"]
    noise = declaration["output_noise"]
    draw = {"role": "measured", "model": "live", "n": 4096, "completed_steps": final,
            "output_noise": noise, "output_sigma": .029}
    summary = {"law": task["evaluation"]["eval_output_noise"], "output_noise": noise,
               "draws": 1, "draws_without_output_noise": 0 if noise else 1}
    evidence = {"policy_controls": {"cohort": task["task_cohort"], "implementation_observed": True,
                                    "requested_owners_bound": True, "lifecycle": {"complete": True},
                                    "completed_steps": final},
                "policy_observations": [{**declaration, "observed": True, "policy_owner": "particlegan.UpdatePolicy",
                                         "measured_draws": [draw]}],
                "policy_purity": [{"pure": True}], "served_output_noise": summary}
    if task["adapter"] == "paired_adaptation":
        evidence["frozen_served_output_noise"] = dict(summary)
    return evidence


def test_uninterrupted_endurance_attests_its_own_early_stop(variants):
    task = variants["ring_hold"]
    evidence = {**_policy_evidence(task, 2900), "dense": [{"step": s} for s in range(1201, 2901)]}
    assert validate_evidence(task, evidence) is None
    evidence["policy_controls"]["completed_steps"] = 7500
    assert validate_evidence(task, evidence)["status"] == "INCOMPLETE"
    ordinary = variants["vector_spiral"]
    assert validate_evidence(ordinary, _policy_evidence(ordinary, 1600)) is None
    assert validate_evidence(ordinary, _policy_evidence(ordinary, 800))["status"] == "INCOMPLETE"
    assert validate_evidence(ordinary, {**_policy_evidence(ordinary, 1600),
                                        "policy_controls": {"cohort": TIER1.id}})["status"] == "INVALID"


def test_schedule_free_helpers_bind_only_registered_policy_variants(variants):
    assert schedule_free_policy(variants["target_shift_recovery"]) and schedule_free_policy(variants["grid100_14k"])
    assert not schedule_free_policy(parent("target_shift_recovery"))
    assert is_policy_task(variants["two_pole"]) and not is_policy_task(parent("two_pole"))


def test_policy_two_pole_receipt_attests_recipe_owned_prior(variants, tmp_path):
    from experiments.forge.policy_behavior_adapters import run_behavior
    task = variants["two_pole"]
    request = {"candidate": candidate("e22"), "protocol": {"seed": 0}}
    result = run_behavior(request, task, tmp_path, "cpu")
    proof = result["evidence"]["prior_policy"]
    assert proof["update"] == "learned" and proof["trainable_locations"] is True and proof["l2"] == 0
    assert result["evidence"]["host"]["particle_l2"] == .02
    grade = grade_result(task, result)
    assert grade["status"] in {"PASS", "FAIL"}, grade


def _group(variants, tmp_path, cohort=COVERAGE):
    cards = {name: deepcopy(variants[name]) for name in ("ring_hold", "ring_extension")}
    source = tmp_path / "frozen"
    files = {}
    for card in cards.values():
        for relative in list(card["evaluation"]["sources"]) + ["experiments/forge/views.py"]:
            if relative in files:
                continue
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            files[relative] = file_hash(target)
    manifest = dict(files=files, digest=stable_hash(files))
    names = [variant_id("ring_hold", cohort), variant_id("ring_extension", cohort)]
    cards = {card["id"]: card for card in cards.values()}
    execution = {name: task_execution_fingerprint(cards[name]) for name in names}
    evaluation = {name: task_evaluation_fingerprint(cards[name]) for name in names}
    science = dict(execution=execution, evaluation=evaluation)
    request = dict(source=manifest, candidate_revision="recipe", tasks=cards, jobs=[
        dict(task_ids=names, compatibility_key=stable_hash(science), science=science)])
    request_path = tmp_path / "request.json"
    atomic_json(request_path, request)
    directory = tmp_path / "producer"
    artifact = directory / "provenance/state.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"retained fixture state, never loaded or trained")
    checkpoint = dict(path="state.pt", sha256=file_hash(artifact), completed_steps=2900,
                      artifact_root=str(artifact.parent), artifact_manifest=manifest_artifacts(artifact.parent))
    producer = cards[names[0]]
    raw = dict(cost=dict(completed_steps=2900), evidence={
        "sampling_contract_version": 1, "sampling_law": producer["evaluation"]["sampling_law"],
        "eval_output_noise": producer["evaluation"]["eval_output_noise"], "scoring_weights": "state_selected",
        "prior_policy": {"contract": "recipe_owned_v1", "update": "learned", "regularizer": "vicreg", "weight": 0.,
                         "target_std": 1., "eps": 1e-4, "l2": 0., "applicability": "latent_prior",
                         "trainable_locations": True},
        **_policy_evidence(producer, 2900),
        "dense": [dict(step=s, modes=8, hq=.95) for s in range(1201, 2901)],
        "continuity": dict(mode="uninterrupted", run_id="one-run", resume_count=0,
                           original_schedule_horizon=None, max_total_steps=7500),
        "provenance_checkpoint": checkpoint})
    atomic_json(directory / "raw.json", raw)
    atomic_json(directory / "result.json", dict(status="PASS", runtime={"fixture": True},
                raw_sha256=file_hash(directory / "raw.json")))
    jobs = []
    for name in names:
        job = dict(family="policy-cloud-fixture", candidate_id="candidate", source_digest=manifest["digest"],
                   candidate_revision="recipe", task=name, execution_sha256=execution[name],
                   evaluation_sha256=evaluation[name], request_path=str(request_path),
                   request_sha256=file_hash(request_path), source_root=str(source),
                   dependencies=cards[name]["dependencies"], attempted=True)
        job["key"] = stable_hash(reuse._identity(job))
        jobs.append(job)
    parent_job, child = jobs
    parent_job.update(status="PASS", origin="new_attempt", directory=str(directory), attempt_id="producer",
                      result_sha256=file_hash(directory / "result.json"))
    child.update(status="BLOCKED", block_kind="prerequisite")
    atomic_json(directory / "request.json", parent_job)
    return parent_job, child


def test_as_served_ring_group_regrades_its_own_saved_extension(served, tmp_path):
    parent_job, child = _group(served, tmp_path, SERVED)
    certificate = reuse.grade_saved_group(parent_job, child)
    assert certificate["grade"]["status"] == "PASS" and certificate["task"] == variant_id("ring_extension", SERVED)
    # A clean-law receipt cannot stand in for the as-served producer's draws.
    raw_path = Path(parent_job["directory"]) / "raw.json"
    raw = read_json(raw_path)
    for row in raw["evidence"]["policy_observations"]:
        row["measured_draws"][0]["output_noise"] = False
    atomic_json(raw_path, raw)
    result = read_json(Path(parent_job["directory"]) / "result.json")
    atomic_json(Path(parent_job["directory"]) / "result.json", dict(result, raw_sha256=file_hash(raw_path)))
    parent_job["result_sha256"] = file_hash(Path(parent_job["directory"]) / "result.json")
    with pytest.raises(ValueError, match="original gate"):
        reuse.grade_saved_group(parent_job, child)


def test_scoped_ring_group_regrades_saved_extension_without_training(variants, tmp_path):
    parent_job, child = _group(variants, tmp_path)
    certificate = reuse.grade_saved_group(parent_job, child)
    assert certificate["grade"]["status"] == "PASS"
    assert certificate["grade"]["metrics"]["extension_passing"] == 300
    # The scoped pair still needs its unchanged original parents to form one group.
    tampered = deepcopy(variants["ring_extension"])
    tampered["execution"]["policy_parent_definition"]["evaluation"]["hold_budget"] = 1
    with pytest.raises(ValueError):
        reuse._check_group(variants["ring_hold"], tampered)
    with pytest.raises(ValueError, match="cannot pool policy cohorts"):
        reuse.grade_saved_group(*_mixed_cohort(parent_job, child, tmp_path))


def _mixed_cohort(parent_job, child, tmp_path):
    request = read_json(Path(parent_job["request_path"]))
    request["tasks"][child["task"]]["task_cohort"] = TIER1.id
    path = tmp_path / "mixed-request.json"
    atomic_json(path, request)
    jobs = []
    for job in (parent_job, child):
        job = dict(job, request_path=str(path), request_sha256=file_hash(path),
                   execution_sha256=task_execution_fingerprint(request["tasks"][job["task"]]),
                   evaluation_sha256=task_evaluation_fingerprint(request["tasks"][job["task"]]))
        job["key"] = stable_hash(reuse._identity(job))
        jobs.append(job)
    return jobs


def test_intake_prepares_both_laws_once_and_never_writes_the_repository_report(tmp_path):
    repository, directory = tmp_path / "repository", tmp_path / "run"
    sentinel = repository / runner.REPORT
    atomic_json(sentinel, {"records": ["unchanged"]})
    state = runner.prepare(repository, ROOT, directory, intake="policy-cloud")
    assert read_json(sentinel) == {"records": ["unchanged"]}
    jobs = list(state["jobs"].values())
    # 35 clean jobs plus 23 as-served jobs per family; 12 law-independent tests run once.
    assert state["intake"] == "policy-cloud" and len(jobs) == 2 * (35 + 23)
    assert {job["family"] for job in jobs} == {"policy-cloud-atlas", "policy-cloud-e22",
                                               "policy-cloud-as-served-atlas", "policy-cloud-as-served-e22"}
    assert len({job["key"] for job in jobs}) == len(jobs) and len({(job["family"], job["task"]) for job in jobs}) == len(jobs)
    for family in FAMILIES:
        mine = [job for job in jobs if job["owning_family"] == family]
        clean = [job for job in mine if job["cohort_id"] == COHORT]
        as_served = [job for job in mine if job["cohort_id"] == SERVED_COHORT]
        assert sorted(job["parent_task"] for job in clean) == sorted(PARENTS)
        assert sorted(job["parent_task"] for job in as_served) == sorted(SERVED_PARENTS)
        clean_scope, = {job["cohort_scope"] for job in clean}
        served_scope, = {job["cohort_scope"] for job in as_served}
        assert clean_scope != served_scope
        assert clean_scope == stable_hash([family, clean[0]["source_digest"], clean[0]["candidate_revision"], COHORT])
        assert {(j["source_digest"], j["candidate_revision"]) for j in mine} == {(clean[0]["source_digest"],
                                                                                 clean[0]["candidate_revision"])}
        for job in clean:
            shared = job["parent_task"] in LAW_INDEPENDENT
            assert job["measurement_law"] == ("shared" if shared else "clean") and job["label"] == LABEL
            assert job.get("shared_scopes") == ([served_scope] if shared else None)
            assert job["task"] == variant_id(job["parent_task"])
        for job in as_served:
            assert job["measurement_law"] == "as_served" and job["label"] == SERVED_LABEL
            assert job["task"] == variant_id(job["parent_task"], SERVED) and "shared_scopes" not in job
            for dependency in job["dependencies"]:
                # Checkpoints come from the same-law producer; the only shared
                # dependency is a gate on the clean clock audit.
                expected = SERVED if dependency["kind"] == "checkpoint" else (
                    COVERAGE if dependency["task"].startswith("clockfree_audit") else SERVED)
                assert dependency["task"].endswith(expected.suffix)
    for job in jobs:
        assert job["status"] == "PENDING" and job["diagnostic"] is True
        assert job["reporting_tier"] == next(t for t, names in TIERS.items() if job["parent_task"] in names)
        assert job["diagnostic_continuation"] == (job["parent_task"] in DIAGNOSTIC_CONTINUATIONS)
        assert job.get("exclusive_gpu", False) == (job["parent_task"] in
                                                   {"gaussian1d_smoke", "gaussian1d_stability", "ring16_acquisition"})
        if job["parent_task"] in BLOCKED_HOSTS:
            assert job["dependencies"] == []
    published = read_json(directory / "attempt-coverage.json")
    assert published["publication"] == "state_directory_only" and len(published["records"]) == len(jobs)
    assert all({"cohort_scope", "cohort_id", "measurement_law", "owning_family", "reporting_tier", "parent_task"}
               <= set(r) for r in published["records"])
    # A second preparation keeps the same jobs; the state cannot change intake.
    assert runner.prepare(repository, ROOT, directory)["jobs"].keys() == state["jobs"].keys()
    with pytest.raises(ValueError, match="belongs to the policy-cloud intake"):
        runner.prepare(repository, ROOT, directory, intake="selected")
    with pytest.raises(ValueError, match="state directory"):
        runner.publish(repository, state)
    with pytest.raises(ValueError, match="never refresh"):
        runner.drain(repository, directory, gpus=["0", "1"], refresh=True)
    # A clean-only (first-build) state directory cannot silently gain a law.
    old = tmp_path / "old"
    atomic_json(old / "policy-cloud.json", {"schema_version": 1, "cohort": COHORT, "view": VIEW,
                                            "families": list(FAMILIES), "source_root": str(ROOT)})
    atomic_json(old / "state.json", {"intake": "policy-cloud", "jobs": {}})
    with pytest.raises(ValueError, match="clean-only intake"):
        runner.prepare(repository, ROOT, old)


def test_selected_state_directory_cannot_switch_to_the_cohort(tmp_path):
    atomic_json(tmp_path / "state.json", {"jobs": {"key": {"family": "atlas", "task": "two_pole"}}})
    with pytest.raises(ValueError, match="belongs to the selected intake"):
        runner.prepare(tmp_path / "repository", ROOT, tmp_path, intake="policy-cloud")


def test_short_limit_jobs_run_alone_on_their_gpu(tmp_path, monkeypatch):
    root, directory = tmp_path / "repo", tmp_path / "runs"
    jobs = {str(i): dict(key=str(i), family="fixture", task=f"task{i}", status="PENDING",
                         resources=dict(gpus=1, gpu_memory_mb=1), dependencies=[]) for i in range(8)}
    for key in ("0", "5"):
        jobs[key].update(exclusive_gpu=True, required_gpu="1")
    atomic_json(directory / "state.json", dict(jobs=jobs))
    monkeypatch.setattr(runner, "device_slots", lambda *a, **kw: [
        dict(device=gpu, slot=slot, memory_mb=100) for gpu in ("0", "1") for slot in (0, 1)])
    monkeypatch.setattr(runner, "process_identity", lambda pid: str(pid))
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    running = []

    class Process:
        def __init__(self, command, **kwargs):
            path = Path(command[-1]).parent
            self.path, self.pid = path, 100 + len(running)
            running.append(self)
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 2:
                atomic_json(self.path / "result.json", dict(status="BLOCKED", attempted=True, reason="fixture"))
            return 0 if self.polls >= 2 else None

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    runner.drain(root, directory, gpus=["0", "1"])
    occupancy, exclusive = {}, set()
    for event in map(json.loads, (directory / "events.jsonl").read_text().splitlines()):
        gpu = event["gpu"]
        if event["event"] == "started":
            occupancy[gpu] = occupancy.get(gpu, 0) + 1
            if event["task"] in {"task0", "task5"}:
                assert gpu == "1" and occupancy[gpu] == 1
                exclusive.add(gpu)
            else:
                assert gpu not in exclusive
        else:
            occupancy[gpu] -= 1
            if event["task"] in {"task0", "task5"}:
                exclusive.discard(gpu)
    assert all(job["status"] == "BLOCKED" for job in read_json(directory / "state.json")["jobs"].values())


def _records(statuses=None):
    """A complete synthetic two-family publication value (both laws)."""
    from experiments.forge.policy_cloud_attempts import cohort_scope
    statuses = statuses or {}
    records, recipes = [], {}
    for family in FAMILIES:
        scopes = {cohort: cohort_scope(family, "source", "recipe", cohort) for cohort in (COHORT, SERVED_COHORT)}
        common = dict(owning_family=family, source_digest="source", candidate_revision="recipe", attempted=True,
                      diagnostic=True, origin="new_attempt", candidate_id=family, execution_sha256="e", evaluation_sha256="v")
        for tier, names in sorted(TIERS.items()):
            for name in names:
                status = statuses.get(("clean", name), "BLOCKED" if name in BLOCKED_HOSTS else
                                      "PASS" if name == "img_stripes2" else "FAIL")
                shared = name in LAW_INDEPENDENT
                records.append(dict(common, cohort_scope=scopes[COHORT], cohort_id=COHORT, family="policy-cloud-" + family,
                                    task=variant_id(name), parent_task=name, reporting_tier=tier, status=status,
                                    label=LABEL, measurement_law="shared" if shared else "clean", updates=10,
                                    key=stable_hash([family, COHORT, name]), attempt_id="a" + name,
                                    **({"shared_scopes": [scopes[SERVED_COHORT]]} if shared else {}),
                                    **({"block_kind": "incompatible", "reason": variant_id(name) + ": " + BLOCKED_HOSTS[name]}
                                       if status == "BLOCKED" else {})))
                if not shared:
                    # The as-served law passes one more (width-limited) test here.
                    served_status = statuses.get(("as_served", name),
                                                 "PASS" if name in {"img_stripes2", "vector_two_broad"} else "FAIL")
                    records.append(dict(common, cohort_scope=scopes[SERVED_COHORT], cohort_id=SERVED_COHORT,
                                        family="policy-cloud-as-served-" + family, task=variant_id(name, SERVED),
                                        parent_task=name, reporting_tier=tier, status=served_status, updates=10,
                                        key=stable_hash([family, SERVED_COHORT, name]), attempt_id="b" + name,
                                        label=SERVED_LABEL, measurement_law="as_served"))
        recipes[family] = dict(verified=True, cohort_source_digest="source", cohort_candidate_revision="recipe",
                               selected_candidate_id=family, selected_candidate_revision="selected-" + family,
                               selected_source_digest="selected-source")
    from experiments.forge.policy_cloud_publication import SCHEMA
    return dict(schema=SCHEMA, qualification_input=False, qualification_reuse=False, formulations_modified=False,
                records=records, recipes=recipes)


def test_cohort_records_project_one_row_per_law_and_family_with_shared_verdicts(tmp_path):
    from experiments.forge import policy_cloud_publication as publication
    from experiments.forge.research_tiers import _attempt_registry, policy_cohort_rows
    _attempt_registry.cache_clear()
    assert policy_cohort_rows(tmp_path, tmp_path / "board.md") == {}  # Nothing published yet.
    value = _records()
    atomic_json(tmp_path / publication.RECORDS, value)
    page = tmp_path / "reports/forge/technique-inventory.md"
    rows = policy_cohort_rows(tmp_path, page)
    assert set(rows) == set(FAMILIES)
    for family in FAMILIES:
        clean, as_served = rows[family]
        assert LABEL in clean and SERVED_LABEL in as_served
        assert "12 law-independent verdicts shared with the clean row" in as_served and "shared" not in clean
        assert f"atlas-e22-policy-cloud-20261015/README.md#{family})" in clean
        # Both rows cover all 35 tests: the shared verdicts (two-pole, clock
        # audits, 9 incompatible hosts) appear in each row exactly once.
        assert "0/7 (3 incompatible)" in clean and "1/21 (6 incompatible)" in clean and "1/35 (9 incompatible)" in clean
        assert "0/7 (3 incompatible)" in as_served and "2/21 (6 incompatible)" in as_served
        assert "2/35 (9 incompatible)" in as_served
    # The family pages count the same cells as the leaderboard rows.
    row = {"candidate_id": "atlas", "candidate_revision": "selected-atlas", "bindings": {"source_digest": "selected-source"}}
    overlay = publication.family_overlay(value, "atlas", row, lambda name: ("e", "v"))
    assert len(overlay["cells"]) == 35 + 23 and overlay["publication"]["shared"] == 12
    from experiments.forge.family_reports import score
    for view in (view_declaration(), served_view_declaration()):
        names = [a["task"] for a in view["assignments"]]
        counts = publication.view_counts({n: overlay["cells"][n]["run_attempt"] for n in names}, names)
        assert score(counts) == ("1/35 (9 incompatible)" if view["id"] == VIEW else "2/35 (9 incompatible)")
    assert overlay["cells"][variant_id("two_pole")]["coverage_reason"].startswith("Separately scoped policy-cloud "
                                                                               "diagnostic (shared law; one run")
    assert publication.family_overlay(value, "atlas", dict(row, candidate_revision="other"), lambda name: ("e", "v")) is None
    assert publication.family_overlay(value, "atlas", row, lambda name: ("x", "v"))["cells"][
        variant_id("grid100")]["current_contract"] == "CHANGED"
    # Cohort records never live in the repository attempt report.
    atomic_json(tmp_path / runner.REPORT, {"records": [value["records"][0]]})
    _attempt_registry.cache_clear()
    with pytest.raises(ValueError, match="their own publication"):
        policy_cohort_rows(tmp_path, page)
    _attempt_registry.cache_clear()
    # A publication must be complete, correctly scoped and shared only where law-independent.
    for change, message in (("missing", "covers every test"), ("shared", "only a law-independent record"),
                            ("scope", "cohort scope differs"), ("credit", "unchanged-formulation"),
                            ("recipe", "verified recipe mapping")):
        broken = deepcopy(value)
        if change == "missing":
            broken["records"].pop()
        elif change == "shared":
            broken["records"][0].pop("shared_scopes")
        elif change == "scope":
            broken["records"][1]["cohort_scope"] = "other"
        elif change == "credit":
            broken["qualification_input"] = True
        else:
            broken["recipes"]["atlas"]["verified"] = False
        with pytest.raises(ValueError, match=message):
            publication.validate_records(broken)


def test_attempt_media_renders_saved_images_and_paired_curves_without_sampling(tmp_path, variants):
    import torch
    from experiments.forge.tier1_media import render
    task = variants["img_bars4"]
    records = [{"step": step, "samples": torch.rand(32, 1, 8, 8), "targets": torch.rand(4, 1, 8, 8),
                "metrics": {"modes": 1, "hq": .5}} for step in (25, 50)]
    path = tmp_path / "observed-images.pt"
    torch.save(records, path)
    evidence = {"observations": [{"step": r["step"], "modes": 1, "hq": .5} for r in records],
                "saved_observer_outputs": {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size},
                "scoring_weights": "state_selected", "sampling_law": task["evaluation"]["sampling_law"]}
    receipt = render(task, {"evidence": evidence, "gate_status": "FAIL"}, tmp_path, tmp_path / "images.gif")
    assert receipt["observation_count"] == 2 and receipt["sampling_draws_added"] == 0
    task = variants["target_shift_recovery"]
    curve = [{"step": step, "modes": 8, "hq": .95} for step in range(10, 101, 10)]
    evidence = {"active": {"diagnostic": curve}, "frozen": {"diagnostic": curve},
                "scoring_weights": "state_selected", "sampling_law": task["evaluation"]["sampling_law"]}
    receipt = render(task, {"evidence": evidence, "gate_status": "FAIL"}, tmp_path, tmp_path / "paired.gif")
    assert receipt["observation_count"] == 10 and (tmp_path / "paired.gif").is_file()


# --- As-served scoring law -------------------------------------------------

@pytest.mark.parametrize("name", SERVED_PARENTS)
def test_as_served_variant_differs_from_its_clean_variant_only_in_the_declared_measurement(name, variants, served):
    from experiments.forge.sampling import AS_SERVED_LAWS, SERVED_LEARNED_NOISE
    clean, task = variants[name], deepcopy(served[name])
    assert task["id"] == variant_id(name, SERVED) and task["task_cohort"] == SERVED_COHORT
    evaluation, observation = task["evaluation"], task["evaluation"]["policy_observation"]
    # The single declared difference, and the law names that follow from it.
    assert observation.pop("output_noise") is True and clean["evaluation"]["policy_observation"]["output_noise"] is False
    assert observation.pop("output_sigma").startswith("GANTrainer.sample(output_noise=True)")
    assert evaluation["eval_output_noise"] == SERVED_LEARNED_NOISE
    original_law = parent(name)["evaluation"]["sampling_law"]
    assert evaluation["sampling_law"] == "policy_cloud_as_served_" + AS_SERVED_LAWS[original_law]
    assert evaluation["sampling_law"] not in {original_law, clean["evaluation"]["sampling_law"]}
    # Undo exactly those fields and the cohort identity: the clean variant remains.
    observation["output_noise"] = False
    evaluation["eval_output_noise"] = clean["evaluation"]["eval_output_noise"]
    evaluation["sampling_law"] = clean["evaluation"]["sampling_law"]
    task.update(id=clean["id"], task_cohort=COHORT)
    task["execution"]["policy_contract"]["cohort"] = COHORT
    for dependency in task["dependencies"]:
        if dependency["task"].endswith(SERVED.suffix):
            dependency["task"] = dependency["task"].removesuffix(SERVED.suffix) + COVERAGE.suffix
        else:  # Only a gate may name a shared (law-independent) clean test.
            assert dependency["kind"] == "gate" and dependency["task"].removesuffix(COVERAGE.suffix) in LAW_INDEPENDENT
    if "continuation_of" in task["execution"]:
        assert task["execution"]["continuation_of"] == variant_id(parent(name)["execution"]["continuation_of"], SERVED)
        task["execution"]["continuation_of"] = clean["execution"]["continuation_of"]
    assert task == clean
    assert validate(served[name], root=ROOT) == parent(name)
    assert adapter_preflight(served[name], candidate("e22"), root=ROOT) == []


def test_law_independent_tests_are_shared_never_duplicated(variants):
    from experiments.forge.tier1_policy import PolicyCohort
    for name in LAW_INDEPENDENT:
        with pytest.raises(ValueError, match="law-independent"):
            materialize(parent(name), ROOT, SERVED)
        assert served_row_task(name) == variant_id(name)
        task = variants[name]
        # Its measurement scores no output samples, or a blocker refuses it first.
        assert (name in BLOCKED_HOSTS or task["adapter"] == "clockfree_audit"
                or task["evaluation"]["policy_observation"]["sampler"] == "served_snapshot")
    assert set(BLOCKED_HOSTS) <= set(LAW_INDEPENDENT)
    with pytest.raises(ValueError, match="same-law producer"):
        SERVED.dependency_task("two_pole", "checkpoint")
    with pytest.raises(ValueError, match="draws no output samples"):
        SERVED.sampling_law("learned_particles_and_critic_gradient")
    with pytest.raises(ValueError, match="law-independent parents stay"):
        PolicyCohort(id="x", parents=("two_pole",), sources=SOURCES, blocked_hosts={}, law_prefix="x_", label="x",
                     parameter_parents=frozenset(), clock_parents=frozenset(), measurement="as_served",
                     shared=frozenset({"two_pole"}), partner=COHORT)
    with pytest.raises(ValueError, match="clean or as_served"):
        PolicyCohort(id="x", parents=(), sources=(), blocked_hosts={}, law_prefix="x_", label="x", measurement="noisy")


class _FakeTrainer:
    completed_steps = 7

    def __init__(self):
        self.calls = []

    def output_sigma(self):
        return .031

    def sample(self, n, *, ema=False, generator=None, output_noise=False, fixed_first_n=False):
        import torch
        self.calls.append(output_noise)
        return torch.full((n, 2), float(output_noise))


def test_output_noise_is_permitted_only_for_the_as_served_cohort(variants, served):
    from experiments.forge.adapters import bind_measurement_law
    from experiments.forge.sampling import SERVED_LEARNED_NOISE, executed_receipt, task_blockers
    with pytest.raises(ValueError, match="registered policy task"):
        bind_measurement_law(_FakeTrainer(), parent("vector_spiral"))
    clean = _FakeTrainer()
    law = bind_measurement_law(clean, variants["vector_spiral"])
    clean.sample(8)
    clean.sample(8, output_noise=False)
    with pytest.raises(ValueError, match="without output noise"):
        clean.sample(8, output_noise=True)
    assert clean.calls == [False, False]
    assert law.receipt()["output_noise"] is False and law.receipt()["draws_without_output_noise"] == 2
    noisy = _FakeTrainer()
    served_law = bind_measurement_law(noisy, served["vector_spiral"])
    noisy.sample(8)
    noisy.sample(8, ema=True, output_noise=False)  # An evaluator's clean request takes the declared law.
    assert noisy.calls == [True, True]
    receipt = served_law.receipt()
    assert {key: receipt[key] for key in ("law", "output_noise", "draws", "draws_without_output_noise",
                                          "evaluator_clean_requests_served")} == {
        "law": SERVED_LEARNED_NOISE, "output_noise": True, "draws": 2, "draws_without_output_noise": 0,
        "evaluator_clean_requests_served": 1}
    assert receipt["output_sigma"] == {"first": .031, "last": .031, "min": .031, "max": .031}
    assert served_law.observed(1) == {"output_noise": True, "measured_draws": [
        {"role": "measured", "model": "ema", "n": 8, "completed_steps": 7, "output_noise": True, "output_sigma": .031}]}
    assert served_law.observed(2) == {}
    assert bind_measurement_law(noisy, served["vector_spiral"]) is served_law
    with pytest.raises(ValueError, match="another declared law"):
        bind_measurement_law(noisy, variants["vector_spiral"])
    # The as-served law name exists only under the registered as-served prefix.
    name = served["vector_spiral"]["evaluation"]["sampling_law"]
    assert executed_receipt(name, eval_output_noise=SERVED_LEARNED_NOISE)["sampling_law"] == name
    with pytest.raises(ValueError):
        executed_receipt(name, eval_output_noise="clean")
    with pytest.raises(ValueError):
        executed_receipt("public_prior_with_learned_output_noise", eval_output_noise=SERVED_LEARNED_NOISE)
    for task in (parent("vector_spiral"), variants["vector_spiral"]):
        impersonating = deepcopy(task)
        impersonating["evaluation"].update(sampling_law=name, eval_output_noise=SERVED_LEARNED_NOISE)
        assert task_blockers(impersonating)  # Neither an ordinary task nor the clean variant may claim it.
    assert task_blockers(served["vector_spiral"]) == [] and task_blockers(variants["vector_spiral"]) == []


def test_policy_evidence_requires_the_observed_output_noise_law(variants, served):
    task = served["vector_spiral"]
    evidence = _policy_evidence(task, 1600)
    assert validate_evidence(task, evidence) is None
    for change, status in (("drop_draws", "INCOMPLETE"), ("clean_draw", "INVALID"),
                           ("drop_summary", "INCOMPLETE"), ("observed_clean", "INVALID")):
        altered = deepcopy(evidence)
        row = altered["policy_observations"][0]
        if change == "drop_draws":
            row.pop("measured_draws")
        elif change == "clean_draw":
            row["measured_draws"][0]["output_noise"] = False
        elif change == "drop_summary":
            altered.pop("served_output_noise")
        else:
            row["output_noise"] = False
        assert validate_evidence(task, altered)["status"] == status, change
    ordinary = variants["vector_spiral"]
    evidence = _policy_evidence(ordinary, 1600)
    assert validate_evidence(ordinary, evidence) is None
    noisy = deepcopy(evidence)
    noisy["policy_observations"][0]["measured_draws"][0]["output_noise"] = True
    assert validate_evidence(ordinary, noisy)["status"] == "INVALID"
    # The clean policy-cloud law attests its sampled draws and receipt as well.
    for change in ("drop_draws", "drop_summary", "noisy_summary", "unsampled_summary"):
        altered = deepcopy(evidence)
        if change == "drop_draws":
            altered["policy_observations"][0].pop("measured_draws")
        elif change == "drop_summary":
            altered.pop("served_output_noise")
        elif change == "noisy_summary":
            altered["served_output_noise"].update(output_noise=True, draws_without_output_noise=0)
        else:
            altered["served_output_noise"]["draws"] = 0
        assert validate_evidence(ordinary, altered)["status"] == "INCOMPLETE", change
    # A parameter measurement records no sampled draws under either law.
    two_pole = variants["two_pole"]
    unsampled = _policy_evidence(two_pole, two_pole["execution"]["steps"])
    unsampled["policy_observations"][0].pop("measured_draws")
    unsampled.pop("served_output_noise")
    assert validate_evidence(two_pole, unsampled) is None
    # Archived Tier 1 receipts predate the draw ledger and keep their format.
    tier1 = materialize(parent("ring16_acquisition"), ROOT)
    legacy = _policy_evidence(tier1, tier1["execution"]["steps"])
    legacy["policy_observations"][0].pop("measured_draws")
    legacy.pop("served_output_noise")
    assert TIER1.attest_draws is False and validate_evidence(tier1, legacy) is None
    # The paired adaptation's frozen control is held to the same declared law.
    for cards in (variants, served):
        shift = cards["target_shift_recovery"]
        paired = _policy_evidence(shift, shift["execution"]["steps"])
        assert validate_evidence(shift, paired) is None
        for change in ("drop", "other_law"):
            altered = deepcopy(paired)
            if change == "drop":
                altered.pop("frozen_served_output_noise")
            else:
                noise = not altered["frozen_served_output_noise"]["output_noise"]
                altered["frozen_served_output_noise"].update(output_noise=noise, draws_without_output_noise=0 if noise else 1)
            assert validate_evidence(shift, altered)["status"] == "INCOMPLETE", change


def test_policy_gaussian_frozen_control_is_measured_under_the_same_law(served, monkeypatch):
    from experiments.forge import policy_gaussian
    monkeypatch.setattr(policy_gaussian, "_ORIGINAL_BUILD",
                        lambda request, task, device, *, max_steps=1000: (object(), _FakeTrainer(), {}))

    def run(request, task, output, device, *, prerequisites=None, diagnostic=False):
        # The unchanged evaluator builds its active trainer, then its frozen control.
        _, active, _ = policy_gaussian._SCOPE["build"](request, task, device)
        _, frozen, _ = policy_gaussian._SCOPE["build"](request, task, device, max_steps=6000)
        active.sample(4096)
        frozen.sample(4096, generator=None, output_noise=False)
        return active._forge_measurement_law, frozen._forge_measurement_law
    monkeypatch.setattr(policy_gaussian, "_ORIGINAL_RUN", run)
    primary, control = policy_gaussian.run_gaussian({}, served["gaussian1d_stability"], None, "cuda")
    assert control.draws is primary.draws and [d["role"] for d in primary.draws] == ["measured", "frozen_control"]
    assert all(d["output_noise"] for d in primary.draws)
    assert primary.receipt()["control_draws"] == 1 and primary.receipt()["evaluator_clean_requests_served"] == 1
    _, lone, _ = policy_gaussian.build({}, served["gaussian1d_smoke"], "cuda")
    assert lone._forge_measurement_law.role == "measured"


@pytest.mark.parametrize("name,steps", [("vector_spiral", 24), ("img_bars4", 24)])
def test_both_laws_train_identically_and_only_the_as_served_draws_carry_noise(name, steps, tmp_path):
    """Short CPU software control through the real adapters (never a result)."""
    import torch
    from experiments.forge.adapters import _dispatch_task
    torch.set_num_threads(1)
    short = parent(name)
    short["execution"]["steps"] = steps
    receipts = {}
    for cohort in (COVERAGE, SERVED):
        task = materialize(short, ROOT, cohort)
        request = {"candidate": candidate("e22"), "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
        raw = _dispatch_task(request, {"task_id": task["id"]}, tmp_path / cohort.id, "cpu")
        assert grade_result(task, raw)["status"] in {"PASS", "FAIL"}
        receipts[cohort.measurement] = raw["evidence"]
    clean, as_served = receipts["clean"], receipts["as_served"]
    assert clean["served_output_noise"]["draws_without_output_noise"] == clean["served_output_noise"]["draws"]
    assert as_served["served_output_noise"]["draws_without_output_noise"] == 0
    assert as_served["served_output_noise"]["output_sigma"]["min"] > 0
    assert all(row["output_noise"] is True for row in as_served["policy_observations"])
    assert all(row["output_noise"] is False for row in clean["policy_observations"])
    # Evaluation is pure: both laws end in the same training state.
    assert clean["training_state"] == as_served["training_state"]
    assert clean["observations"] != as_served["observations"]


def test_policy_cloud_views_display_only_on_the_atlas_and_e22_pages():
    from experiments.forge.family_reports import _displayed_for
    tier1 = read_json(ROOT / "configs/forge/views/tier1_policy_coverage.json")
    for view in (view_declaration(), served_view_declaration()):
        assert _displayed_for(view, "atlas") and _displayed_for(view, "e22")
        assert not _displayed_for(view, "bcap-pure") and not _displayed_for(view, "k3p")
    assert all(_displayed_for(tier1, family) for family in ("atlas", "bcap-pure", "k3p"))


# --- Build-fix controls: scoped continuations, retries, single GPU, publication ---

def test_diagnostic_continuation_stays_limited_to_registered_cohort_parents(variants, served, tmp_path, monkeypatch):
    from experiments.forge import adapters, gaussian_tasks, priors, views
    from experiments.forge import tier1_policy
    from experiments.forge.tier1_policy import diagnostic_continuation_allowed
    calls = []
    monkeypatch.setattr(gaussian_tasks, "run_gaussian",
                        lambda request, task, output, device, **kwargs: calls.append(kwargs) or {"cost": {}})
    ordinary = parent("gaussian1d_stability")
    request = {"candidate": candidate("e22"), "tasks": {ordinary["id"]: ordinary}}
    adapters._dispatch_task(request, {"task_id": ordinary["id"], "diagnostic_continuation": True}, tmp_path, "cpu")
    assert calls == [{"prerequisites": None}]  # Ordinary Gaussian tasks keep their own rules.
    smoke = variants["gaussian1d_smoke"]
    with pytest.raises(CapabilityError, match="declares no diagnostic continuation"):
        adapters._dispatch_task({"candidate": candidate("e22"), "tasks": {smoke["id"]: smoke}},
                                {"task_id": smoke["id"], "diagnostic_continuation": True}, tmp_path, "cpu")
    assert all(diagnostic_continuation_allowed(cards[name]) for cards in (variants, served)
               for name in DIAGNOSTIC_CONTINUATIONS)
    assert not diagnostic_continuation_allowed(ordinary) and not diagnostic_continuation_allowed(smoke)
    assert not diagnostic_continuation_allowed(materialize(parent("gaussian1d_acquisition"), ROOT))
    # Grading: only the registered stability variant may be graded as a diagnostic continuation.
    monkeypatch.setattr(priors, "validate_prior_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(views, "grade_sampling", lambda *args, **kwargs: None)
    monkeypatch.setattr(views, "_guards", lambda *args, **kwargs: None)
    monkeypatch.setattr(tier1_policy, "validate_evidence", lambda *args, **kwargs: None)
    grade = grade_result(ordinary, {"evidence": {"scoring_weights": "live"}}, diagnostic=True)
    assert grade["status"] == "INVALID" and grade["reasons"] == ["diagnostic checkpoint grading is limited to native continuations"]
    variant = variants["gaussian1d_stability"]
    grade = grade_result(variant, {"evidence": {"scoring_weights": "state_selected"}}, diagnostic=True)
    assert grade["reasons"] == ["diagnostic Gaussian continuation requires its explicit nonqualifying scope"]
    grade = grade_result(variant, {"evidence": {"scoring_weights": "state_selected", "continuity": {
        "diagnostic_continuation": True, "parent_smoke_status": "FAIL"}}}, diagnostic=True)
    assert "continuation" not in " ".join(grade["reasons"])  # Reaches the unchanged evaluator.


def _interrupted_state(tmp_path):
    jobs = {}

    def job(key, status, dependencies=(), **extra):
        directory = tmp_path / "attempts" / key
        directory.mkdir(parents=True)
        (directory / "run.log").write_text(key + " log\n")
        jobs[key] = dict(key=key, family="policy-cloud-atlas", task=key, status=status, attempted=True,
                         attempt_id="old-" + key, directory=str(directory), gpu="1", slot=0, pid=7,
                         process_identity="gone", dependencies=list(dependencies), request_path="frozen.json",
                         request_sha256="frozen", resources=dict(gpus=1, gpu_memory_mb=1), diagnostic=True,
                         cohort_scope="scope", **extra)
        return directory

    job("producer", "INTERRUPTED", reason="prior worker stopped without a terminal receipt")
    child = job("dependent", "BLOCKED", [{"task": "producer", "kind": "checkpoint"}], block_kind="prerequisite",
                reason="checkpoint parent producer: INTERRUPTED", origin="new_attempt", result_sha256="r",
                evidence=str(tmp_path / "attempts/dependent/result.json"))
    atomic_json(child / "result.json", {"status": "BLOCKED"})
    for key, status, extra in (("failed", "FAIL", {}), ("timeout", "TIMEOUT", {}), ("passed", "PASS", {}),
                               ("error", "ERROR", {}), ("refused", "BLOCKED", {"block_kind": "incompatible"})):
        directory = job(key, status, origin="new_attempt", evidence=str(tmp_path / "attempts" / key / "result.json"),
                        **extra)
        atomic_json(directory / "result.json", {"status": status})
    job("waits_on_failure", "BLOCKED", [{"task": "failed", "kind": "checkpoint"}], block_kind="prerequisite")
    return {"intake": "policy-cloud", "jobs": jobs}


def test_retry_interrupted_requeues_only_infrastructure_interruptions(tmp_path):
    state = _interrupted_state(tmp_path)
    untouched = {key: deepcopy(job) for key, job in state["jobs"].items() if key not in {"producer", "dependent"}}
    requeued = runner.requeue_interrupted(state)
    assert sorted(job["task"] for job in requeued) == ["dependent", "producer"]
    for key in ("producer", "dependent"):
        job = state["jobs"][key]
        assert job["status"] == "PENDING" and not set(runner.ATTEMPT_FIELDS) & set(job)
        assert (job["request_path"], job["request_sha256"], job["dependencies"]) == (
            "frozen.json", "frozen", untouched.get(key, job)["dependencies"])
        history, = job["retry_history"]
        assert history["attempt_id"] == "old-" + key and history["log_sha256"]
        assert history["status"] == ("INTERRUPTED" if key == "producer" else "BLOCKED")
        assert "same frozen request, task, recipe and seed" in history["retry_reason"]
    assert {key: state["jobs"][key] for key in untouched} == untouched  # Outcomes are never retried.
    assert runner.requeue_interrupted(state) == []
    # A receipt means it was not an infrastructure interruption.
    state = _interrupted_state(tmp_path / "second")
    atomic_json(Path(state["jobs"]["producer"]["directory"]) / "result.json", {"status": "FAIL"})
    with pytest.raises(ValueError, match="receipt or a live worker"):
        runner.requeue_interrupted(state)
    with pytest.raises(ValueError, match="only to the policy-cloud intake"):
        runner.requeue_interrupted({"jobs": {}})


def _fake_workers(monkeypatch, gpus, status="FAIL"):
    monkeypatch.setattr(runner, "device_slots", lambda devices, *a, **kw: [
        dict(device=gpu, slot=slot, memory_mb=100) for gpu in devices for slot in (0, 1)])
    monkeypatch.setattr(runner, "process_identity", lambda pid: str(pid))
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    running = []

    class Process:
        def __init__(self, command, **kwargs):
            self.path, self.pid, self.polls = Path(command[-1]).parent, 1000 + len(running), 0
            running.append(self)

        def poll(self):
            self.polls += 1
            if self.polls == 2:
                atomic_json(self.path / "result.json", dict(status=status, attempted=True, reason="fixture"))
            return 0 if self.polls >= 2 else None

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    return running


def test_policy_cloud_drain_falls_back_to_one_gpu_and_retries_interrupted_attempts(tmp_path, monkeypatch):
    root, directory = tmp_path / "repo", tmp_path / "runs"
    jobs = {str(i): dict(key=str(i), family="policy-cloud-fixture", task=f"task{i}", status="PENDING",
                         resources=dict(gpus=1, gpu_memory_mb=1), dependencies=[]) for i in range(6)}
    jobs["0"].update(exclusive_gpu=True, required_gpu="1")
    atomic_json(directory / "state.json", dict(intake="policy-cloud", jobs=jobs))
    _fake_workers(monkeypatch, ["1"])
    with pytest.raises(ValueError, match="pinned to GPU 1"):
        runner.drain(root, directory, gpus=["0"])
    runner.drain(root, directory, gpus=["1"])
    occupancy, peak = 0, 0
    for event in map(json.loads, (directory / "events.jsonl").read_text().splitlines()):
        assert event["gpu"] == "1"
        occupancy += 1 if event["event"] == "started" else -1
        peak = max(peak, occupancy)
    assert peak <= 2 and all(job["status"] == "FAIL" for job in read_json(directory / "state.json")["jobs"].values())
    # A pending diagnostic continuation must keep its producer's GPU.
    continuation = tmp_path / "continuation"
    atomic_json(continuation / "state.json", dict(intake="policy-cloud", jobs={
        "producer": dict(key="producer", family="f", task="producer", status="FAIL", gpu="0", dependencies=[],
                         resources=dict(gpus=1, gpu_memory_mb=1)),
        "child": dict(key="child", family="f", task="child", status="PENDING", diagnostic_continuation=True,
                      dependencies=[{"task": "producer", "kind": "checkpoint"}], resources=dict(gpus=1, gpu_memory_mb=1))}))
    with pytest.raises(ValueError, match="producer's GPU 0"):
        runner.drain(root, continuation, gpus=["1"])
    with pytest.raises(ValueError, match="exactly two distinct GPUs"):
        runner.drain(root, tmp_path / "selected", gpus=["1"]) if atomic_json(
            tmp_path / "selected/state.json", dict(jobs={})) is None else None
    # An explicit retry re-runs the interrupted producer and its refused dependent once.
    retry = tmp_path / "retry"
    state = _interrupted_state(retry)
    for job in state["jobs"].values():
        job["family"] = "policy-cloud-fixture"
    atomic_json(retry / "state.json", state)
    _fake_workers(monkeypatch, ["0", "1"], status="PASS")
    runner.drain(root, retry, gpus=["0", "1"], retry_interrupted=True)
    final = read_json(retry / "state.json")["jobs"]
    assert final["producer"]["status"] == "PASS" and final["producer"]["retry_history"][0]["status"] == "INTERRUPTED"
    assert final["dependent"]["status"] == "PASS" and final["dependent"]["retry_history"][0]["status"] == "BLOCKED"
    assert final["failed"]["status"] == "FAIL" and "retry_history" not in final["failed"]
    events = [json.loads(line) for line in (retry / "events.jsonl").read_text().splitlines()]
    assert sorted(e["task"] for e in events if e["event"] == "requeued") == ["dependent", "producer"]
    with pytest.raises(ValueError, match="only to the policy-cloud intake"):
        runner.drain(root, tmp_path / "selected", gpus=["0", "1"], retry_interrupted=True)


def test_plain_causes_state_the_failed_checks_in_words():
    from experiments.forge.policy_cloud_publication import plain_cause
    base = dict(task="t_policy_cloud_coverage_v1", status="FAIL")
    curve = dict(base, grade={"evaluator_result": {"metrics": [
        {"metric": "modes", "op": ">=", "threshold": 16, "value": 0, "status": "FAIL"},
        {"metric": "mass_tv", "op": "<=", "threshold": .15, "value": .08, "status": "PASS"}],
        "convergence": {"observations": 96, "passing_observations": 0, "passing_suffix": 0, "minimum_stable_checks": 5}}})
    assert plain_cause(curve) == "Final live checks fail: modes 0 (needs >= 16); 0/96 observations passed."
    native = dict(base, grade={"evaluator_result": {"coverage_status": "INVALID", "terminal_checks": [{"passed": False}] * 5,
                                                     "holdout_metrics": {"precision": .034, "mass_tv": .128}}})
    assert plain_cause(native).startswith("Coverage gate INVALID; final live accuracy checks 0/5 passed; independent holdout fails")
    assert "mass TV 0.128 (needs <= 0.06) FAIL" in plain_cause(native)
    gaussian = dict(base, status="PASS", grade={"metrics": {"cdf_ks": .03, "std_ratio": 1.3, "sample_count": 4096},
                                                "evaluator_result": dict(confirmed_steps=[84], first_confirmed_step=84,
                                                                         passing_observations=16, observations=24)})
    assert plain_cause(gaussian, [["sample_count", ">=", 4096], ["std_ratio", ">=", .8], ["std_ratio", "<=", 1.2],
                                  ["cdf_ks", "<=", .05]]).endswith(
        "Final: std_ratio 1.3 (needs 0.8 to 1.2) FAIL, cdf_ks 0.03 (needs <= 0.05).")
    stability = dict(base, grade={"evaluator_result": dict(stationary_passes=46, stationary_checks=72,
        reacquisition_status="FAIL", shift_hold_passes=17, shift_hold_checks=24, frozen_passes=0, frozen_checks=48)})
    assert "Stationary hold 46/72, deadline reacquisition FAIL, shifted hold 17/24" in plain_cause(stability)
    ring = dict(base, grade={"metrics": dict(name="first_convergence_then_hold_v1", status="NOT_CONVERGED",
        settling_failures=40, settling_checks=40, confirmation_checks=5, modes=8, min_hq=.9)})
    assert plain_cause(ring).startswith("No qualifying window: 40/40 settling checks failed")
    shift = dict(base, grade={"metrics": {"live_deadline": dict(passing_checks=0, checks=6, min_modes=8, min_hq=.64),
                                          "frozen_deadline": dict(passing_checks=0, checks=6)}})
    assert plain_cause(shift).startswith("Recovery window after the deadline: live 0/6 checks passed")
    # The complete grader rule names every window, also without the recomputed windows.
    assert all(part in plain_cause(shift) for part in ("stationary window", "continued hold", "frozen control passing none"))
    window = lambda first, last, passing, checks: dict(first_step=first, last_step=last, passing_checks=passing, checks=checks)
    windows = {"stationary": window(1000, 1200, 2, 5), "continued_hold": window(1210, 2400, 28, 120),
               "live_recovery": window(2800, 3600, 0, 6)}
    assert plain_cause(shift, windows=windows).startswith(
        "Live run: stationary window (steps 1,000-1,200) 2/5 checks passed, continued hold (steps 1,210-2,400) 28/120, "
        "recovery window after the deadline (steps 2,800-3,600) live 0/6 checks passed")
    blocked = dict(base, status="BLOCKED", block_kind="incompatible", reason="t_policy_cloud_coverage_v1: no rows")
    assert plain_cause(blocked) == "Not run (incompatible): no rows"
    assert plain_cause(dict(base, status="INTERRUPTED")).startswith("Worker stopped without any receipt")
    assert "120 s" in plain_cause(dict(base, status="TIMEOUT", timeout_seconds=120))


def test_native_width_ranges_come_from_the_recorded_holdouts():
    from experiments.forge.policy_cloud_publication import _width_text, native_width_summary
    value = _records()
    holdouts = {("atlas", "clean"): (.82, .99), ("e22", "clean"): (.36, .70), ("atlas", "as_served"): (.004, .01),
                ("e22", "as_served"): (.092, .106)}
    for record in value["records"]:
        if record["parent_task"] in {"grid100", "rotated100"} and record["measurement_law"] != "shared":
            radial, bias = holdouts[record["owning_family"], record["measurement_law"]]
            record["grade"] = {"evaluator_result": {"coverage_status": "FAIL", "holdout_metrics": dict(
                precision=.999 if record["parent_task"] == "grid100" else .994, radial_ks=radial,
                abs_cov_trace_bias=bias)}}
    summary = native_width_summary(value["records"])
    assert summary["clean"]["all"]["radial_ks"] == (.36, .82) and summary["clean"]["all"]["tests"] == 4
    assert summary["as_served"]["all"]["within_width_limits"] == 2
    text = _width_text(value["records"])
    assert "Their 4 clean holdouts" in text and "precision 0.994-0.999" in text
    assert "radial KS 0.36-0.82 against 0.04 (Atlas 0.82, E22 0.36)" in text and "|cov trace bias| 0.70-0.99" in text
    assert "radial KS 0.004-0.092; 2 of 4 are within both width limits" in text


def test_adaptation_windows_recompute_every_grader_window_from_the_raw_receipt(tmp_path):
    from experiments.forge import policy_cloud_publication as publication
    from benchmarks.toy100.continuous_probe import _window
    task = variant_id("target_shift_recovery")
    root = tmp_path / "repo"
    atomic_json(root / COVERAGE.directory / (task + ".json"),
                {"evaluation": {"shift_step": 2400, "steps": 3600, "recovery_deadline": 400}})

    def curve(good):
        return [{"step": step, "modes": 8 if good(step) else 7, "hq": .95} for step in range(10, 3601, 10)]
    active = {"diagnostic": curve(lambda step: step not in {1000, 1050, 1210} and step < 3000)}
    frozen = {"diagnostic": curve(lambda step: step <= 2400)}
    directory = tmp_path / "attempt"
    atomic_json(directory / "raw.json", {"evidence": {"active": active, "frozen": frozen}})
    late = lambda run: _window([p for p in run["diagnostic"] if p["step"] >= 2800])
    record = {"key": "k", "owning_family": "atlas", "task": task, "cohort_id": COHORT, "attempt_id": "a",
              "grade": {"metrics": {"live_deadline": late(active), "frozen_deadline": late(frozen)}}}
    found = publication.adaptation_windows(root, [{"key": "k", "directory": str(directory)}], [record])
    entry = found["atlas/" + task]
    assert (entry["stationary"]["passing_checks"], entry["stationary"]["checks"]) == (3, 5)
    assert (entry["continued_hold"]["passing_checks"], entry["continued_hold"]["checks"]) == (119, 120)
    assert (entry["live_recovery"]["passing_checks"], entry["live_recovery"]["checks"]) == (20, 81)
    assert entry["frozen_recovery"]["passing_checks"] == 0 and entry["raw_sha256"] == file_hash(directory / "raw.json")
    record["grade"]["metrics"]["live_deadline"] = dict(late(active), passing_checks=21)
    with pytest.raises(ValueError, match="recomputed recovery window differs"):
        publication.adaptation_windows(root, [{"key": "k", "directory": str(directory)}], [record])


def test_source_equivalent_names_exactly_the_files_that_differ_from_the_snapshot(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    from experiments.forge import policy_cloud_publication as publication
    repo = tmp_path / "repo"
    files = {"experiments/a.py": "a = 1\n", "experiments/b.py": "b = 1\n", "configs/x.json": "{}\n"}
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    manifest = {"origin_commit": "f" * 40, "digest": "d" * 64,
                "files": {path: hashlib.sha256(text.encode()).hexdigest() for path, text in files.items()}}
    (repo / "experiments/b.py").write_text("b = 2\n")

    def commit():
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
        subprocess.run(git + ["init", "-q"] if not (repo / ".git").exists() else ["true"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "c"], check=True)
        return subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    head = commit()
    declare = lambda differing: monkeypatch.setattr(publication, "SOURCE_EQUIVALENTS", {"f" * 40: {
        "commit": head, "branch": "b", "differing_files": {path: "why" for path in differing}}})
    declare(["experiments/b.py"])
    entry = publication.source_equivalent(repo, manifest)
    assert entry["equivalent_commit"] == head and entry["snapshot_files"] == 3
    assert list(entry["differing_files"]) == ["experiments/b.py"] and entry["files_added_in_snapshot_scope"] == []
    declare([])
    with pytest.raises(ValueError, match="differs from the frozen snapshot"):
        publication.source_equivalent(repo, manifest)
    (repo / "experiments/new.py").write_text("n = 1\n")
    head = commit()
    declare(["experiments/b.py"])
    with pytest.raises(ValueError, match="adds \\['experiments/new.py'\\]"):
        publication.source_equivalent(repo, manifest)
    assert publication.source_equivalent(repo, dict(manifest, origin_commit="e" * 40)) is None


def _cohort_state(tmp_path, value):
    """A drained synthetic policy-cloud run whose receipts back ``value``."""
    state_root = tmp_path / "state"
    jobs = {}
    for record in deepcopy(value["records"]):
        directory = state_root / "attempts" / record["key"][:16]
        directory.mkdir(parents=True)
        (directory / "run.log").write_text(record["task"] + "\n")
        (directory / "request.json").write_text("{}\n")
        raw = {"evidence": {"training_state": {"sha256": stable_hash([record["owning_family"], record["parent_task"]]),
                                               "completed_steps": 10}}} if record["status"] in {"PASS", "FAIL"} else None
        result = {"status": record["status"]}
        if raw is not None:
            atomic_json(directory / "raw.json", raw)
            result["raw_sha256"] = file_hash(directory / "raw.json")
        atomic_json(directory / "result.json", result)
        record.update(directory=str(directory), evidence=str(directory / "result.json"),
                      result_sha256=file_hash(directory / "result.json"), log_sha256=file_hash(directory / "run.log"),
                      dependencies=[{"task": variant_id("gaussian1d_smoke", SERVED if record["cohort_id"] == SERVED_COHORT
                                                        else COVERAGE), "kind": "checkpoint"}]
                      if record["parent_task"] == "gaussian1d_stability" else [],
                      diagnostic_continuation=record["parent_task"] in DIAGNOSTIC_CONTINUATIONS)
        if (record["owning_family"], record["cohort_id"], record["parent_task"]) == ("atlas", SERVED_COHORT, "grid100"):
            record["retry_history"] = [{"attempt_id": "interrupted", "status": "INTERRUPTED", "retry_reason": "fixture"}]
        jobs[record["key"]] = record
    atomic_json(state_root / "state.json", {"intake": "policy-cloud", "jobs": jobs})
    from experiments.forge.policy_cloud_attempts import LAWS
    atomic_json(state_root / "policy-cloud.json", {
        "schema_version": 2, "families": list(FAMILIES), "source_root": str(tmp_path / "frozen"),
        "laws": {law: {key: spec[key] for key in ("cohort", "view", "label")} for law, spec in LAWS.items()},
        "shared_tests": sorted(LAW_INDEPENDENT), "timeout_sensitive_gpu": "1"})
    return state_root


def test_publication_writes_records_results_and_readme_deterministically(tmp_path, monkeypatch):
    from experiments.forge import policy_cloud_publication as publication
    value = _records()
    monkeypatch.setattr(publication, "recipe_identity", lambda root, state_root, records: {
        family: dict(value["recipes"][family], selected_recipe_sha256="a" * 64, cohort_recipe_sha256="b" * 64,
                     restricted_to_selected_fields_sha256="a" * 64, cohort_source_origin_commit="c" * 40,
                     added_fields_at_registered_defaults={"prior_sigreg_knots": 17},
                     inert_because={"prior_regularizer": "vicreg", "prior_reg": 0.0, "prior_proximal_strength": 0.0})
        for family in FAMILIES})
    state_root = _cohort_state(tmp_path, value)
    # A ring extension regraded from its own passing hold: a verified certificate, no new training.
    state = read_json(state_root / "state.json")
    regraded = next(job for job in state["jobs"].values()
                    if job["owning_family"] == "e22" and job["task"] == variant_id("ring_extension"))
    certificate = state_root / "grouped-evidence" / "certificate.json"
    atomic_json(certificate, {"grade": {"status": "PASS"}, "original_attempt": "producer"})
    for key in ("directory", "log_sha256"):
        regraded.pop(key)
    regraded.update(status="PASS", origin="reused_execution", evidence=str(certificate),
                    result_sha256=file_hash(certificate), updates=0,
                    reused_execution={"producer_attempt_id": "producer", "optimizer_updates_added": 0})
    atomic_json(state_root / "state.json", state)
    root = tmp_path / "repo"
    summary = publication.publish(root, state_root)
    records = publication.validate_records(read_json(root / publication.RECORDS))
    assert len(records["records"]) == 2 * (35 + 23)
    assert all(r["evidence"] == publication.README.as_posix() for r in records["records"]
               if r["origin"] == "new_attempt")
    assert summary["summary"]["atlas"]["as_served"]["scores"]["total"] == "2/35 (9 incompatible)"
    assert summary["summary"]["e22"]["clean"]["scores"]["total"] == "2/35 (9 incompatible)"
    assert summary["training_state_pairs"] == {"match": 2 * 23 - 1, "no_own_training": 1}
    copied = root / publication.GROUPED / "certificate.json"
    assert copied.read_bytes() == certificate.read_bytes()
    record = next(r for r in records["records"] if r["owning_family"] == "e22" and r["task"] == variant_id("ring_extension"))
    assert record["evidence"] == (publication.GROUPED / "certificate.json").as_posix()
    results = read_json(root / publication.RESULTS)
    assert results["records_sha256"] == file_hash(root / publication.RECORDS)
    receipt = next(r for r in results["receipts"] if r["status"] == "FAIL")
    assert receipt["files"]["run.log"] == file_hash(Path(receipt["directory"]) / "run.log")
    continuation = [row for row in results["diagnostic_continuations"] if row["family"] == "atlas"]
    assert {(row["law"], row["producer"], row["producer_status"]) for row in continuation} == {
        ("clean", "gaussian1d_smoke", "FAIL"), ("as_served", "gaussian1d_smoke", "FAIL")}
    readme = (root / publication.README).read_text()
    assert "| 1 | two_pole | FAIL · shared run | FAIL · shared run | Law-independent:" in readme
    assert "Diagnostic continuation of its own same-law saved state after a FAIL parent gate" in readme
    assert "| [atlas](#atlas) | as served (12 shared with clean) |" in readme and '<a name="e22"></a>' in readme
    assert "Pairs: **45 match**" in readme and "prior_sigreg_knots=17" in readme
    assert "| 3 | ring_extension | PASS · regraded, no new training |" in readme
    assert "| 2 | grid100 | FAIL · 10 updates | FAIL · 10 updates | Re-run after an infrastructure interruption (as served)" in readme
    retried = next(r for r in results["receipts"] if r.get("retry_history"))
    assert retried["retry_history"][0]["status"] == "INTERRUPTED"
    before = {path: path.read_bytes() for path in (root / publication.REPORT_DIRECTORY).rglob("*") if path.is_file()}
    publication.publish(root, state_root)
    assert {path: path.read_bytes() for path in before} == before  # Deterministic rerun.
    # Changed local receipts or an undrained run refuse publication.
    log = Path(receipt["directory"]) / "run.log"
    log.write_text("edited\n")
    with pytest.raises(ValueError, match="differs from its recorded receipt"):
        publication.publish(root, state_root)
    state = read_json(state_root / "state.json")
    next(iter(state["jobs"].values()))["status"] = "PENDING"
    atomic_json(state_root / "state.json", state)
    with pytest.raises(ValueError, match="requires a drained run"):
        publication.publish(root, state_root)


def test_recipe_identity_maps_each_frozen_recipe_onto_its_selected_row(tmp_path):
    from experiments.forge.policy_cloud_publication import recipe_identity
    from particlegan.recipe_compat import SEARCH_RECIPE_DEFAULTS
    current = read_json(ROOT / "reports/forge/technique-inventory.json")
    archived = read_json(ROOT / "configs/forge/selections/history/family-current-138fe6eb6c13251128aed4e0171908cd676e578682f49026fc8fd4be6492b7c2.json")
    rows, fixture_rows, snapshots = {}, [], {}
    for family in FAMILIES:
        pin = next(item for item in archived["selections"] if item["trainer_family"] == family)
        incumbent = next(row for row in current["evidence_rows"]
                         if row.get("candidate_id") == pin["candidate_id"]
                         and row.get("candidate_revision") == pin["candidate_revision"]
                         and row.get("cohort") == pin["cohort"])
        incumbent = {**incumbent, "trainer_family": pin["trainer_family"]}
        rows[family] = incumbent
        fixture_rows.append(incumbent)
        key = incumbent["publication_key"]
        snapshots[key] = current["evidence_sources"][key]
    repo = tmp_path / "repo"
    progress = {
        "views": current["family_progress"]["views"],
        "families": [
            {"id": family, "current_cohort_anchor": "archived",
             "cohorts": [{"anchor": "archived", "row_index": index, "views": []}]}
            for index, family in enumerate(FAMILIES)
        ],
    }
    atomic_json(repo / "reports/forge/technique-inventory.json",
                {"rows": fixture_rows, "evidence_sources": snapshots, "family_progress": progress})
    for pointer in snapshots.values():
        target = repo / pointer["snapshot"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / pointer["snapshot"], target)
    recipes = {}
    for family in FAMILIES:
        row = rows[family]
        contract = read_json(repo / snapshots[row["publication_key"]]["snapshot"])[
            "recipe_contracts"][row["bindings"]["recipe_sha256"]]
        # A newer Recipe adds fields at their registered defaults.
        recipes[family] = {**contract, "prior_sigreg_knots": SEARCH_RECIPE_DEFAULTS["prior_sigreg_knots"],
                           "prior_proximal_strength": SEARCH_RECIPE_DEFAULTS["prior_proximal_strength"]}
    state_root = tmp_path / "state"

    def write(family, recipe):
        for prefix in ("policy-cloud-", "policy-cloud-as-served-"):
            atomic_json(state_root / "requests" / (prefix + family + ".json"), {
                "source": {"digest": "source", "origin_commit": "c" * 40}, "candidate_revision": "recipe",
                "protocol": {"seed": 0}, "candidate": {"recipe_overrides": {}, "resolved_recipe": recipe}})
    for family in FAMILIES:
        write(family, recipes[family])
    records = [{"owning_family": family, "source_digest": "source", "candidate_revision": "recipe"} for family in FAMILIES]
    mapping = recipe_identity(repo, state_root, records)
    for family in FAMILIES:
        assert mapping[family]["verified"] is True
        assert mapping[family]["restricted_to_selected_fields_sha256"] == rows[family]["bindings"]["recipe_sha256"]
        assert mapping[family]["selected_candidate_revision"] == rows[family]["candidate_revision"]
        assert mapping[family]["added_fields_at_registered_defaults"] == {"prior_proximal_strength": 0.0,
                                                                          "prior_sigreg_knots": 17}
        assert mapping[family]["inert_because"]["prior_reg"] == 0.0
    # A non-default addition, a changed recorded field or another seed is refused.
    for change, message in (({"prior_sigreg_knots": 18}, "differs from the selected recipe"),
                            ({"batch_size": recipes["atlas"]["batch_size"] + 1}, "differs from the selected recipe")):
        write("atlas", {**recipes["atlas"], **change})
        with pytest.raises(ValueError, match=message):
            recipe_identity(repo, state_root, records)
    write("atlas", recipes["atlas"])
    request = read_json(state_root / "requests/policy-cloud-atlas.json")
    request["protocol"]["seed"] = 1
    atomic_json(state_root / "requests/policy-cloud-atlas.json", request)
    with pytest.raises(ValueError, match="seed 0"):
        recipe_identity(repo, state_root, records)


def test_policy_cloud_drain_dispatches_the_longest_dependency_chains_first(tmp_path, monkeypatch):
    root, directory = tmp_path / "repo", tmp_path / "runs"

    def job(task, timeout, dependencies=()):
        return dict(key=task, family="policy-cloud-fixture", task=task, status="PENDING", timeout_seconds=timeout,
                    resources=dict(gpus=1, gpu_memory_mb=1), dependencies=list(dependencies))
    jobs = {j["key"]: j for j in [
        job("a_short", 300), job("b_short", 1800), job("c_short", 1800), job("d_short", 1800), job("e_short", 1800),
        job("z_native", 3600), job("z_gate", 300),
        job("z_native_14k", 7200, [{"task": "z_native", "kind": "checkpoint"}, {"task": "z_gate", "kind": "gate"}])]}
    state = dict(intake="policy-cloud", jobs=jobs)
    critical = runner.critical_path_seconds(state)
    # A gate dependency does not lengthen its producer's chain.
    assert (critical["z_native"], critical["z_native_14k"], critical["z_gate"]) == (10800, 7200, 300)
    atomic_json(directory / "state.json", state)
    _fake_workers(monkeypatch, ["0", "1"])
    runner.drain(root, directory, gpus=["0", "1"])
    started = [e["task"] for e in map(json.loads, (directory / "events.jsonl").read_text().splitlines())
               if e["event"] == "started"]
    assert started[0] == "z_native" and started.index("z_native_14k") < started.index("a_short")
    assert all(j["status"] == "FAIL" for j in read_json(directory / "state.json")["jobs"].values())
    # Other intakes keep their original order.
    selected = tmp_path / "selected"
    atomic_json(selected / "state.json", dict(jobs={k: dict(v, family="fixture") for k, v in jobs.items()}))
    runner.drain(root, selected, gpus=["0", "1"])
    started = [e["task"] for e in map(json.loads, (selected / "events.jsonl").read_text().splitlines())
               if e["event"] == "started"]
    assert started[0] == "a_short"


def _saved_run(directory, device, weight=1.0):
    """One attempt directory whose receipt digests a saved state bound to ``device``."""
    import torch
    from experiments.forge.adapters import training_state_receipt
    state = {"streams": {"manifest": {"bindings": {device: {"family": "training"}, "eval": {"family": "eval"}}},
                         "states": {device: torch.tensor([1, 2, 3]), "eval": torch.tensor([7])}},
             "trainer": {"G": torch.full((2,), weight), "cpu_rng": torch.tensor([5]), "device": device}}
    (directory / "provenance").mkdir(parents=True)
    path = directory / "provenance" / "provenance-state.pt"
    torch.save(state, path)
    receipt = training_state_receipt(state, 10)
    atomic_json(directory / "raw.json", {"evidence": {"training_state": receipt, "provenance_checkpoint": {
        "path": path.name, "sha256": file_hash(path)}}})
    return receipt["sha256"]


def test_training_state_pairs_normalize_the_device_label_only_across_gpus(tmp_path):
    from experiments.forge import policy_cloud_publication as publication

    def pair(name, clean_gpu, served_gpu, served_weight=1.0):
        jobs = []
        for cohort, gpu, weight in ((COHORT, clean_gpu, 1.0), (SERVED_COHORT, served_gpu, served_weight)):
            directory = tmp_path / name / cohort
            _saved_run(directory, f"cuda:{gpu}", weight)
            jobs.append(dict(owning_family="atlas", cohort_id=cohort, parent_task="grid100", key=name + cohort,
                             status="FAIL", attempt_id=cohort, gpu=gpu, directory=str(directory)))
        states = {job["key"]: publication._training_state(Path(job["directory"])) for job in jobs}
        (row,) = publication.training_state_pairs(jobs, states)
        return row
    across = pair("across", "0", "1")
    assert across["clean"]["sha256"] != across["as_served"]["sha256"]  # The receipt digest names the device.
    assert (across["comparison"], across["method"]) == ("match", "device_normalized_saved_state")
    assert across["clean"]["device_normalized"]["verified"] is True
    assert across["clean"]["device_normalized"]["sha256"] == across["as_served"]["device_normalized"]["sha256"]
    same = pair("same", "1", "1")
    assert (same["comparison"], same["method"]) == ("match", "receipt_digest")
    changed = pair("changed", "0", "1", served_weight=2.0)  # A real training difference survives normalization.
    assert (changed["comparison"], changed["method"]) == ("mismatch", "device_normalized_saved_state")
    differs_same_gpu = pair("samegpu", "0", "0", served_weight=2.0)
    assert (differs_same_gpu["comparison"], differs_same_gpu["method"]) == ("mismatch", "receipt_digest")
    # A saved state that does not reproduce its receipt is never trusted.
    tampered = tmp_path / "tampered"
    _saved_run(tampered, "cuda:0")
    raw = read_json(tampered / "raw.json")
    raw["evidence"]["training_state"]["sha256"] = "0" * 64
    atomic_json(tampered / "raw.json", raw)
    check = publication.device_independent_digest(publication._training_state(tampered))
    assert check["verified"] is False and check["sha256"] is None


def test_cohort_media_tiles_in_run_gifs_without_training_or_sampling(tmp_path):
    from PIL import Image
    from experiments.forge.policy_cloud_media import render_cohort_media
    trained = {}
    for law, parent, status, colour in (("clean", "grid100", "FAIL", (200, 0, 0)), ("shared", "two_pole", "FAIL", (0, 0, 200)),
                                        ("as_served", "grid100", "PASS", (0, 160, 0))):
        directory = tmp_path / "attempts" / (law + parent)
        directory.mkdir(parents=True)
        frames = [Image.new("RGB", (90, 60), tuple(min(255, c + 20 * k) for c in colour)) for k in range(3)]
        frames[0].save(directory / "training.gif", save_all=True, append_images=frames[1:], duration=100)
        trained[("atlas", law, parent)] = dict(directory=str(directory), status=status, measurement_law=law,
                                               attempt_id=law + parent)
    root = tmp_path / "repo"
    receipt = render_cohort_media(root, Path("report"), trained, families=("atlas",), laws=("clean", "as_served"),
                                  order=["two_pole", "grid100"])
    assert receipt["optimizer_updates_added"] == 0 and receipt["sampling_draws_added"] == 0
    clean, served = receipt["sheets"]
    assert [tile["test"] for tile in clean["tiles"]] == ["two_pole", "grid100"]  # Shared runs join the clean sheet.
    assert [tile["test"] for tile in served["tiles"]] == ["grid100"]
    assert clean["tiles"][0]["frames_used"][0] == 0 and clean["tiles"][0]["frames_used"][-1] == 2
    first = (root / clean["path"]).read_bytes()
    assert Image.open(root / clean["path"]).n_frames >= 2
    render_cohort_media(root, Path("report"), trained, families=("atlas",), laws=("clean", "as_served"),
                        order=["two_pole", "grid100"])
    assert (root / clean["path"]).read_bytes() == first  # Deterministic rerun.
