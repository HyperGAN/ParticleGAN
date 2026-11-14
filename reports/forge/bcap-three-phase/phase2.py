"""Explicit Phase 2 declaration, execution, and saved-evidence projection.

prepare/run/publish wrappers expose separate operations. Preparation resolves
admission without allocating a worker. Execution requires an explicit frozen
commit. Publication only reads certificates and already-scored training views.
The original revision-8 ordinary view remains authoritative, including its
optional clock audit; only six tasks enter the quality denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, identifier, read_json, stable_hash, utc_now, validate_idea
from experiments.forge.planning import FORMULATION_FIELDS, load_idea, plan_summary, resolve_idea
from experiments.forge.studies import validate_study
from experiments.forge.views import load_view

VIEW = "discriminator_stability"
TIMEOUTS = dict(gaussian1d_smoke=120, two_pole=300, unused_token_hold=300,
                ae_gan_hold=300, ring16_acquisition=300, five_word_joint_smoke=900)
PER_ARM = 2520
MAX_CAMPAIGN = 7560
ORIGINALS = "reports/forge/bcap-develop-integration/original-task-contracts.json"
OUTCOMES = dict(falsified="stop_revision", incomplete="request_missing_evidence",
                inconclusive="stop_and_readout", prediction_observed="review_saved_diagnostics")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def head(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def original_question(task):
    """Only previously declared optional hooks and source ancestry are metadata."""
    task = deepcopy(task)
    task.pop("task_cohort", None)
    for field in ("transport_consumer", "transport_contract"):
        task["execution"].pop(field, None)
    for field in ("sources", "evaluator_revision"):
        task["evaluation"].pop(field, None)
    return task


def ordinary_contract(root):
    view = load_view(root, VIEW)
    require(view["revision"] == 8 and view.get("evidence_scope", "ordinary") == "ordinary",
            "Phase 2 retains the original revision-8 ordinary view")
    active = [a for a in view["assignments"] if a["qualification_tier"] <= 1]
    required = [a for a in active if a["importance"] == "required"]
    require({a["task"] for a in required} == set(TIMEOUTS) and len(required) == 6,
            "All six original required Tier 1 questions must remain present")
    require(len(active) == 7 and [a["task"] for a in active if a["importance"] != "required"]
            == ["clockfree_audit_measurement_v1"], "Retain the separate optional clock audit")
    originals = read_json(root / ORIGINALS)["tasks"]
    tasks = {}
    for assignment in active:
        name = assignment["task"]
        task = read_json(root / f"configs/forge/tasks/{name}.json")
        if name in TIMEOUTS:
            require(original_question(task) == original_question(originals[name]),
                    f"{name}: architecture, law, prior, initializer, sampling, updates, cadence or gates changed")
            require(task["resources"]["timeout_seconds"] == TIMEOUTS[name], f"{name}: original timeout changed")
        tasks[name] = task
    require(sum(t["resources"]["timeout_seconds"] for t in tasks.values()) == PER_ARM,
            "Reserve the six original allowances plus the separate 300-second clock audit")
    return view, tasks


def rebind_sources(root, tasks, prefix):
    changes = {}
    for name, original in tasks.items():
        task = deepcopy(original)
        sources = task["evaluation"].get("sources", {})
        changed = {path: old for path, old in sources.items() if file_hash(root / path) != old}
        if not changed:
            continue
        for path in changed:
            sources[path] = file_hash(root / path)
        task["evaluation"]["evaluator_revision"] = dict(id=f"{prefix}-source-binding-v1",
            change="Bind current public implementation for fresh execution; original question, fixed fixtures, laws, initialization, budgets, cadence and all numerical gates retained. Earlier receipts keep their original source identity.",
            previous_source_sha256=changed, previous_revision=original["evaluation"].get("evaluator_revision"))
        require(original_question(task) == original_question(original), f"{name}: rebinding altered science")
        atomic_json(root / f"configs/forge/tasks/{name}.json", task)
        changes[name] = {path: dict(before=old, after=sources[path]) for path, old in changed.items()}
    return changes


def successor(root, arm):
    parent = load_idea(root, arm["parent"])
    fields = (*FORMULATION_FIELDS, "host_adaptation")
    candidate = {key: deepcopy(parent[key]) for key in fields if key in parent and key != "prior"}
    require("implementation" not in candidate, "Migrate custom implementation into the public API first")
    candidate.setdefault("recipe_overrides", {}).update(arm.get("recipe_overrides", {}))
    candidate.update(schema_version=3, id=arm["candidate_id"], parent=arm["parent"],
        api_version="forge-api-v1", changed_factors=arm["changed_factors"],
        mechanism_class=arm.get("mechanism_class", parent["mechanism_class"]),
        mechanism_rationale=arm["mechanism_rationale"], guide="reports/forge/bcap-three-phase/README.md")
    validate_idea(candidate)
    return candidate


def study_for(arm, prefix, campaign_budget):
    study = dict(schema_version=1, id=f'{arm["candidate_id"]}-study', status="ready",
        candidate=arm["candidate_id"], control=dict(candidate_id=arm["control"], task_map={}),
        hypothesis=arm["hypothesis"], competing_explanation=arm["competing_explanation"],
        scope=dict(view=VIEW, through_tier=1, execution_backend="cuda", cuda_model="NVIDIA RTX A6000"),
        campaign=dict(id=prefix, budget_seconds=campaign_budget, candidate_budget_seconds=PER_ARM),
        max_rounds=1, prior_evidence=arm["prior_evidence"], prediction=arm["prediction"],
        falsifier=arm["falsifier"], terminal_rules=OUTCOMES)
    validate_study(study)
    return study


def resolve_request(*args, **kwargs):
    """Bind this separate historical cohort to its original wall-clock limits."""
    request = resolve_idea(*args, **kwargs)
    request["wallclock_policy"] = "declared"
    return request


def request_summary(request):
    summary = plan_summary(request)
    active = [row for row in summary["tasks"] if row["permitted_by_tier_cap"]]
    require(request.get("study_review", {}).get("status") == "READY" and not request["preflight_blockers"],
            f'Admission blocked: {request.get("study_review", {})}; {request["preflight_blockers"]}')
    require(len(active) == 7 and not any(row["blockers"] or row["execution_group_blockers"] for row in active),
            f"Required/diagnostic active preflight blocked: {active}")
    require(summary["worst_case_seconds"] == PER_ARM, "Full original per-arm allowances must be reserved")
    return dict(candidate_id=request["candidate"]["id"], study_id=request["study"]["id"],
        admission=request["study_review"]["status"], source_digest=request["source"]["digest"],
        candidate_revision=request["candidate_revision"], study_admission=request["study_admission"],
        active_tasks=active, worst_case_seconds=summary["worst_case_seconds"])


def prepare(options):
    root = options.repository.resolve()
    spec = read_json(options.spec)
    prefix = identifier(spec["campaign_id"], "campaign")
    arms = spec["arms"]
    require(2 <= len(arms) <= 3 and len({a["role"] for a in arms}) == len(arms)
            and len({a["candidate_id"] for a in arms}) == len(arms), "Use an incumbent and one or two distinct repair arms")
    require(arms[0]["role"] == "incumbent", "First arm must be the one global incumbent")
    campaign_budget = PER_ARM * len(arms)
    require(campaign_budget <= MAX_CAMPAIGN, "Phase 2 may reserve at most 7,560 seconds")
    view, tasks = ordinary_contract(root)
    destination = options.registration.resolve()
    paths = [root / f'configs/forge/{kind}/{a["candidate_id"]}{suffix}.json'
             for a in arms for kind, suffix in (("ideas", ""), ("studies", "-study"))]
    require(not destination.exists() and not any(path.exists() for path in paths),
            "Use fresh candidate, study and registration identities; preserve earlier declarations")
    candidates = [successor(root, arm) for arm in arms]
    studies = [study_for(arm, prefix, campaign_budget) for arm in arms]
    # Write every candidate before resolving any study, so controls can select
    # another newly declared arm. READY is computed, never forged in a receipt.
    for candidate, study in zip(candidates, studies):
        atomic_json(root / f'configs/forge/ideas/{candidate["id"]}.json', candidate)
        atomic_json(root / f'configs/forge/studies/{study["id"]}.json', study)
    changes = rebind_sources(root, tasks, prefix)
    ordinary_contract(root)
    plans = {}
    for arm, study in zip(arms, studies):
        request = resolve_request(root, arm["candidate_id"], study=study)
        plans[arm["role"]] = request_summary(request)
    require(len({plan["source_digest"] for plan in plans.values()}) == 1, "All arms need identical source")
    registration = dict(schema_version=1, scope="phase2_fresh_ordinary_tier1", qualification_input=False,
        campaign_id=prefix, campaign_budget_seconds=campaign_budget, candidate_budget_seconds=PER_ARM,
        prepared_commit=head(root), view=VIEW, view_revision=view["revision"], through_tier=1,
        required_tasks=TIMEOUTS, optional_tasks=["clockfree_audit_measurement_v1"],
        protocol_seed=0, source_rebindings=changes, arms=plans,
        spec_sha256=file_hash(options.spec), source_digest=next(iter(plans.values()))["source_digest"],
        stop="Finish the complete declared Tier 1 round; no scientific retry, tuning, Tier 2, merge or default promotion.")
    atomic_json(destination, registration)
    print(json.dumps(dict(event="phase2_prepared", registration=str(destination), arms=plans), indent=2), flush=True)


def resolved(root, registration):
    ordinary_contract(root)
    require(registration["view"] == VIEW and registration["view_revision"] == 8 and registration["through_tier"] == 1,
            "Registration must bind the original ordinary Tier 1 view")
    requests = {}
    for role, plan in registration["arms"].items():
        request = resolve_request(root, plan["candidate_id"], study=plan["study_id"])
        require(request_summary(request) == plan, f"{role}: source/runtime/task/study binding changed; prepare a fresh round")
        requests[role] = request
    return requests


def committed_source(root, source, declarations=()):
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    files = set(source["files"]) | set(declarations)
    require(files <= tracked, "Commit all frozen public source inputs and declarations before admission")
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *sorted(files)], cwd=root)
    require(result.returncode == 0, "Commit all frozen public source inputs before admission")


def run(options):
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    require(head(root) == options.source_commit, "Execution must select the exact committed source")
    require(not artifacts.is_relative_to(root), "Bulk artifacts belong outside Git")
    registration = read_json(options.registration)
    requests = resolved(root, registration)
    declarations = {f'configs/forge/ideas/{request["candidate"]["id"]}.json' for request in requests.values()}
    declarations.update(f'configs/forge/studies/{request["study"]["id"]}.json' for request in requests.values())
    declarations.update(f"configs/forge/tasks/{name}.json" for name in next(iter(requests.values()))["tasks"])
    declarations.add(f"configs/forge/views/{VIEW}.json")
    committed_source(root, next(iter(requests.values()))["source"], declarations)
    progress_path = artifacts / "phase2-progress.json"
    if options.drain:
        require(progress_path.is_file(), "Drain requires explicit prior --submit admission")
        progress = read_json(progress_path)
        require(progress["registration_sha256"] == file_hash(options.registration)
                and progress["source_commit"] == options.source_commit, "Admitted registration/source differs")
    else:
        require(options.submit, "Choose explicit --submit or --drain")
        require(not progress_path.exists(), "Use a fresh artifact campaign; no unchanged rerun")
    from experiments.forge.queue import Queue, drain
    queue = Queue(artifacts / "phase2-queue", report_root=root / "reports/forge", on_completion=None)
    if options.submit:
        progress = dict(phase="submitting", updated_at=utc_now(), requests={}, source_commit=options.source_commit,
            registration_sha256=file_hash(options.registration), source=registration["source_digest"],
            log=str(artifacts / "logs/phase2-driver.log"))
        for role, plan in registration["arms"].items():
            request = resolve_request(root, plan["candidate_id"], study=plan["study_id"],
                                   queue_root=artifacts / "phase2-queue", freeze_source=True)
            require(request_summary(request) == plan, f"{role}: bindings changed while freezing")
            receipt = queue.submit(request, request["study"]["campaign"])
            progress["requests"][role] = receipt["request"]["request_id"]
            atomic_json(progress_path, progress)
            print(json.dumps(dict(event="phase2_submitted", role=role, request=progress["requests"][role])), flush=True)
        progress.update(phase="admitted", updated_at=utc_now())
        atomic_json(progress_path, progress)
    if options.drain:
        progress.update(phase="running", updated_at=utc_now())
        atomic_json(progress_path, progress)
        drain(queue, options.gpus.split(","), workers_per_gpu=1, allow_sharing=True,
              watch=False, campaign=registration["campaign_id"])
        progress.update(phase="ordinary_complete", updated_at=utc_now())
        atomic_json(progress_path, progress)
        print(json.dumps(dict(event="phase2_complete", time=utc_now())), flush=True)


def publisher(root):
    path = root / "reports/forge/bcap-develop-integration/publish.py"
    spec = importlib.util.spec_from_file_location("phase2_saved_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_saved_comparison(root, collection, roles):
    """Verify existing receipts for arbitrary arms; never construct a model."""
    path = root / "reports/forge/bcap-develop-integration/audit.py"
    spec = importlib.util.spec_from_file_location("phase2_saved_state_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    sources, requests = audit.source_bindings(collection, root)
    grouped, unavailable = {}, []
    for entry in collection["final"]:
        if entry["row"]["gate_status"] not in {"PASS", "FAIL"}:
            unavailable.append(dict(role=entry["item"]["role"], task_id=entry["row"]["task_id"],
                reason="Incomplete/invalid evidence cannot establish complete consumed-state matching"))
            continue
        for variant, saved in audit.saved_variants(entry):
            grouped.setdefault((entry["row"]["task_id"], variant), []).append(saved)
    proofs = []
    for (task_id, variant), entries in sorted(grouped.items()):
        require(len({entry["item"]["role"] for entry in entries}) == len(entries), "Ambiguous selected arm evidence")
        first = entries[0]
        initial = audit.metadata(first)
        first_bindings, first_states = audit.non_eval(first["saved"])
        steps = {entry["item"]["role"]: entry["item"]["provenance_checkpoint"]["completed_steps"] for entry in entries}
        equal_steps = len(set(steps.values())) == 1
        for entry in entries:
            require(audit.equal(initial, audit.metadata(entry)), f"{task_id}: actual initial models/priors differ")
            bindings, states = audit.non_eval(entry["saved"])
            require(audit.equal(first_bindings, bindings), f"{task_id}: named consumed-stream bindings differ")
            guards = entry["row"]["evidence"].get("guards", {})
            require(guards.get("unintended_rng_deviations", 0) == 0, f"{task_id}: unintended RNG consumption")
            require(all(point.get("unintended_rng_deviations", 0) == 0
                        for point in entry["row"]["evidence"].get("rng_audits", [])), f"{task_id}: RNG audit deviation")
            if equal_steps:
                require(audit.equal(first_states, states), f"{task_id}: actual consumed training RNG states differ")
                require(first["row"]["evidence"].get("data_sha256") == entry["row"]["evidence"].get("data_sha256"),
                        f"{task_id}: actual target-batch sequence differs")
            if "applied" in entry["saved"]:
                applied = entry["row"].get("applied", entry["row"]["evidence"].get("applied"))
                require(applied is not None and stable_hash(applied) == stable_hash(entry["saved"]["applied"]),
                        f"{task_id}: consumed component receipt differs from certified row")
        proofs.append(dict(task_id=task_id, saved_variant=variant, completed_roles=list(steps), completed_steps=steps,
            initial_state_proof_sha256=audit.state_digest(initial),
            saved_state_references={entry["item"]["role"]: entry["item"]["provenance_checkpoint"] for entry in entries},
            initialization_and_prior_equal=len(entries) >= 2, named_training_bindings_equal=len(entries) >= 2,
            all_declared_arms_present=set(steps) == set(roles),
            consumed_non_eval_streams_and_batches_equal=len(entries) >= 2 and equal_steps,
            consumption_comparison="Exact complete consumed states" if len(entries) >= 2 and equal_steps else
                "Only available complete states; missing arms/different completed budgets remain unverified"))
    return dict(status="PASS", scope="phase2_saved_evidence_audit", frozen_source_checks=sources,
        requests=requests, original_six_questions_preserved=True, saved_state_comparisons=proofs,
        unavailable_complete_states=unavailable, optimizer_updates_added=0, sampling_draws_added=0)


def publish(options):
    root, artifacts, output = options.repository.resolve(), options.artifacts.resolve(), options.output.resolve()
    registration = read_json(options.registration)
    progress_path = artifacts / "phase2-progress.json"
    progress = read_json(progress_path)
    require(progress["phase"] == "ordinary_complete"
            and progress["registration_sha256"] == file_hash(options.registration), "Publish only the complete registered campaign")
    requests = resolved(root, registration)
    require(set(requests) == set(progress["requests"]), "All declared arms must remain visible")
    module = publisher(root)
    module.ROLES = tuple(requests)
    view, _ = ordinary_contract(root)
    requirements = sorted((a for a in view["assignments"] if a["qualification_tier"] == 1
                           and a["importance"] == "required"), key=lambda a: a["order"])
    module.required_questions = lambda _root: requirements
    collection = module.collect(argparse.Namespace(repository=root, queue=artifacts / "phase2-queue",
        progress=progress_path, diagnostic_queue=None, diagnostic_progress=None, allow_partial=False))
    for context in collection["scopes"]:
        for role, submission in context["submissions"].items():
            request = submission["request"]
            require(request_summary(request) == registration["arms"][role], f"{role}: certified admission differs")
            require(request["source"]["digest"] == registration["source_digest"], "Certified source differs")
    audit = audit_saved_comparison(root, collection, requests)
    renderer = module._saved_renderer(root)
    media = []
    from PIL import Image
    for entry in collection["final"]:
        item = entry["item"]
        if item["gate_status"] not in {"PASS", "FAIL"}:
            continue
        gif = output / "media" / f'{item["role"]}-{item["task_id"]}.gif'
        with renderer.forbid_live_execution():
            receipt = module.render_saved(entry, gif, renderer)
        with Image.open(gif) as image:
            require(image.n_frames >= 2, "Actual-training GIF requires multiple certified saved states")
            frames = image.n_frames
        media.append({**receipt, "role": item["role"], "task_id": item["task_id"],
                      "importance": item["importance"], "attempt_id": item["attempt_id"],
                      "gif": str(gif.relative_to(output)), "frames": frames})
        print(json.dumps(dict(event="phase2_saved_media", role=item["role"], task=item["task_id"])), flush=True)
    required = [entry["item"] for entry in collection["final"] if entry["item"]["importance"] == "required"]
    optional = [entry["item"] for entry in collection["final"] if entry["item"]["importance"] != "required"]
    outcomes = {role: dict(Counter(cell["gate_status"] for cell in collection["cells"] if cell["role"] == role))
                for role in requests}
    histories = [dict(attempt["compact"], final_selected=aid in {e["item"]["attempt_id"] for e in collection["final"]})
                 for aid, attempt in sorted(collection["attempts"].items())]
    result = dict(schema_version=1, scope="phase2_fresh_ordinary_tier1", qualification_input=False,
        registration_sha256=file_hash(options.registration), source_digest=registration["source_digest"],
        required_counts=dict(tier1=6), required_task_cells=collection["cells"], task_results=required,
        optional_diagnostics=optional, outcomes=outcomes,
        complete_tier1_pass={role: counts == {"PASS": 6} for role, counts in outcomes.items()},
        accounting=collection["accounting"], paid_attempt_history=histories,
        actual_training_gifs=len(media), optimizer_updates_added=0, sampling_draws_added=0,
        interpretation="Each arm requires all six original Tier 1 gates; the optional clock audit does not enter quality qualification. No Tier 2 or default adoption claim.")
    atomic_json(output / "phase2-results.json", result)
    atomic_json(output / "phase2-audit.json", audit)
    atomic_json(output / "media/index.json", dict(schema_version=1, media=media, optimizer_updates_added=0, sampling_draws_added=0))
    print(json.dumps(dict(event="phase2_published", outcomes=outcomes, gifs=len(media))), flush=True)


def main(action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if action is None:
        parser.add_argument("action", choices=("prepare", "run", "publish"))
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--source-commit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--submit", action="store_true", help="Admit and freeze only; no worker starts")
    group.add_argument("--drain", action="store_true", help="Explicitly execute already admitted requests")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    action = action or options.action
    require(action != "prepare" or options.spec, "prepare requires --spec")
    require(action == "prepare" or options.artifacts, "run/publish require --artifacts")
    require(action != "run" or options.source_commit, "run requires --source-commit")
    require(action != "publish" or options.output, "publish requires --output")
    globals()[action](options)


if __name__ == "__main__":
    main()
