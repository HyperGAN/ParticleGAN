"""Prepare, explicitly admit/run, and publish one paired research diagnostic.

The caller supplies the baseline, hypothesis and global candidate delta. This
helper runs no research during preparation or saved-evidence publication. Its
separate diagnostic view preserves the original sixteen questions and each
candidate's own passing checkpoint dependencies; it cannot qualify a technique.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, atomic_text, file_hash, identifier, read_json, stable_hash, utc_now
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.studies import validate_study
from experiments.forge.views import load_view

_spec = importlib.util.spec_from_file_location("phase3_phase2_helpers", Path(__file__).with_name("phase2.py"))
phase2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase2)
require, head = phase2.require, phase2.head
ROLES = ("baseline", "candidate")
DEEP_TASKS = ("gaussian1d_stability", "five_word_joint_hold", "trajectory", "residual_student",
              "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic",
              "grid100", "rotated100", "staggered100")
PER_ARM, PAIRED_RESERVATION, PAID_CEILING, SOFTWARE_ALLOWANCE = 22920, 45840, 48000, 300


def original_contract(root, task_ids):
    view = load_view(root, phase2.VIEW)
    require(view["revision"] == 8 and view.get("evidence_scope", "ordinary") == "ordinary",
            "Preserve the original revision-8 ordinary view")
    original_tier1 = {a["task"] for a in view["assignments"]
                      if a["qualification_tier"] == 1 and a["importance"] == "required"}
    require(original_tier1 == set(phase2.TIMEOUTS), "Retain all six original required Tier 1 questions")
    require(isinstance(task_ids, list) and len(task_ids) == len(set(task_ids)) == 16
            and set(task_ids) == original_tier1 | set(DEEP_TASKS),
            "Declare exactly the six original Tier 1 and ten planned Tier 2 questions")
    requirements = [deepcopy(a) for a in view["assignments"] if a["task"] in task_ids]
    require(len(requirements) == 16 and all(a["importance"] == "required" for a in requirements)
            and Counter(a["qualification_tier"] for a in requirements) == {1: 6, 2: 10},
            "Resolve the complete diagnostic roster from the ordinary view")
    originals = read_json(root / phase2.ORIGINALS)["tasks"]
    tasks = {}
    for name in task_ids:
        task = read_json(root / f"configs/forge/tasks/{name}.json")
        require(phase2.original_question(task) == phase2.original_question(originals[name]),
                f"{name}: original architecture, laws, initialization, updates, cadence, dependencies or gates changed")
        require(task["execution"]["initializer"] == "deterministic_orthogonal",
                f"{name}: retain the public deterministic initializer")
        tasks[name] = task
    require(sum(t["resources"]["timeout_seconds"] for t in tasks.values()) == PER_ARM,
            "Reserve all original allowances: 22,920 seconds per arm")
    return requirements, tasks


def check_spec(spec):
    require(spec.get("schema_version") == 1, "Phase 3 input requires schema_version 1")
    prefix = identifier(spec["campaign_id"], "campaign")
    identifier(spec["view_id"], "diagnostic view")
    arms = spec["arms"]
    require(len(arms) == 2 and tuple(a["role"] for a in arms) == ROLES
            and len({a["candidate_id"] for a in arms}) == 2,
            "Declare exactly one global baseline and one substantive candidate")
    require(spec.get("paid_ceiling_seconds") == PAID_CEILING
            and spec.get("software_allowance_seconds") == SOFTWARE_ALLOWANCE,
            "Freeze the 48,000-second paid ceiling and separate 300-second software allowance")
    require(spec.get("candidate_budget_seconds") == PAID_CEILING // 2,
            "Each arm has a 24,000-second ceiling, including admitted execution repairs")
    require(arms[0]["control"] == arms[1]["candidate_id"]
            and arms[1]["control"] == arms[0]["candidate_id"], "Freeze reciprocal paired controls")
    require(not arms[0].get("recipe_overrides"), "The baseline inherits its chosen recipe unchanged")
    return prefix, arms


def study_for(arm, spec):
    study = dict(schema_version=1, id=f'{arm["candidate_id"]}-study', status="ready",
        candidate=arm["candidate_id"], control=dict(candidate_id=arm["control"], task_map={}),
        hypothesis=arm["hypothesis"], competing_explanation=arm["competing_explanation"],
        scope=dict(view=spec["view_id"], through_tier=1, execution_backend="cuda", cuda_model="NVIDIA RTX A6000"),
        campaign=dict(id=spec["campaign_id"], budget_seconds=PAID_CEILING,
                      candidate_budget_seconds=spec["candidate_budget_seconds"]),
        max_rounds=1, prior_evidence=arm["prior_evidence"], prediction=arm["prediction"],
        falsifier=arm["falsifier"], terminal_rules=phase2.OUTCOMES)
    validate_study(study)
    require(study["prediction"]["task_id"] in spec["task_ids"]
            and study["falsifier"]["task_id"] in spec["task_ids"], "Predictions must address the frozen roster")
    return study


def resolve_request(*args, **kwargs):
    """Preserve this separate study's declared original runtime allowances."""
    request = resolve_idea(*args, **kwargs)
    request["wallclock_policy"] = "declared"
    return request


def request_summary(request, task_ids):
    summary = plan_summary(request)
    active = [row for row in summary["tasks"] if row["permitted_by_tier_cap"]]
    require(request["view"].get("evidence_scope") == "research_diagnostic"
            and all(a["qualification_tier"] == 1 and a["importance"] == "diagnostic"
                    for a in request["view"]["assignments"]), "Only the separate diagnostic view may run")
    require(request.get("study_review", {}).get("status") == "READY" and not request["preflight_blockers"],
            f'Admission blocked: {request.get("study_review", {})}; {request["preflight_blockers"]}')
    require(len(active) == 16 and {r["task"] for r in active} == set(task_ids)
            and not any(r["blockers"] or r["execution_group_blockers"] for r in active),
            f"All original diagnostic host contracts must preflight: {active}")
    require(summary["worst_case_seconds"] == PER_ARM and request["protocol"]["seed"] == 0
            and request["candidate"].get("initializer", "deterministic_orthogonal") == "deterministic_orthogonal",
            "Retain full allowances, protocol seed zero and public initialization")
    require(request["candidate"]["schema_version"] == 3, "Research candidates require schema version 3")
    return dict(candidate_id=request["candidate"]["id"], study_id=request["study"]["id"],
        admission=request["study_review"]["status"], source_digest=request["source"]["digest"],
        candidate_revision=request["candidate_revision"], study_admission=request["study_admission"],
        active_tasks=active, full_reservation_seconds=summary["worst_case_seconds"])


def repository_relative(root, path):
    path = path.resolve()
    require(path.is_relative_to(root), "Commit the input spec and registration inside the research checkout")
    return str(path.relative_to(root))


def prepare(options):
    root = options.repository.resolve()
    require(head(root) == options.source_commit, "Preparation must select the exact current HEAD")
    spec = read_json(options.spec)
    prefix, arms = check_spec(spec)
    requirements, tasks = original_contract(root, spec["task_ids"])
    spec_path = repository_relative(root, options.spec)
    registration_path = repository_relative(root, options.registration)
    view_path = root / f'configs/forge/views/{spec["view_id"]}.json'
    paths = [root / f'configs/forge/{kind}/{a["candidate_id"]}{suffix}.json'
             for a in arms for kind, suffix in (("ideas", ""), ("studies", "-study"))]
    require(not options.registration.exists() and not view_path.exists() and not any(p.exists() for p in paths),
            "Use fresh track candidate, study, view and registration identities")
    candidates = [phase2.successor(root, arm) for arm in arms]
    studies = [study_for(arm, spec) for arm in arms]
    view = deepcopy(load_view(root, phase2.VIEW))
    view.update(id=spec["view_id"], revision=1, evidence_scope="research_diagnostic",
        assignments=[dict(task=t, qualification_tier=1, importance="diagnostic", order=i)
                     for i, t in enumerate(spec["task_ids"])],
        eligibility={}, ranking=dict(policy="diagnostic_outcomes_only", cost_separate=True),
        policy_change_reason="One fixed-seed paired research comparison. Original numerical questions and own checkpoint dependencies retained; independent deeper measurements confer no ordinary qualification.")
    atomic_json(view_path, view)
    for candidate, study in zip(candidates, studies):
        atomic_json(root / f'configs/forge/ideas/{candidate["id"]}.json', candidate)
        atomic_json(root / f'configs/forge/studies/{study["id"]}.json', study)
    changes = phase2.rebind_sources(root, tasks, prefix)
    original_contract(root, spec["task_ids"])
    plans = {arm["role"]: request_summary(resolve_request(root, arm["candidate_id"], study=study), spec["task_ids"])
             for arm, study in zip(arms, studies)}
    require(len({p["source_digest"] for p in plans.values()}) == 1, "Both arms require identical source")
    registration = dict(schema_version=1, scope="phase3_paired_research_diagnostic", qualification_input=False,
        campaign_id=prefix, view=spec["view_id"], view_revision=1, through_tier=1,
        prepared_commit=head(root), source_digest=plans["baseline"]["source_digest"],
        full_reservation_seconds=PAIRED_RESERVATION, per_arm_full_reservation_seconds=PER_ARM,
        campaign_ceiling_seconds=PAID_CEILING, candidate_ceiling_seconds=spec["candidate_budget_seconds"],
        software_allowance_seconds=SOFTWARE_ALLOWANCE, protocol_seed=0,
        task_ids=spec["task_ids"], original_requirements=requirements,
        task_contracts={t: stable_hash(read_json(root / f"configs/forge/tasks/{t}.json")) for t in spec["task_ids"]},
        source_rebindings=changes, arms=plans, spec_path=spec_path, spec_sha256=file_hash(options.spec),
        registration_path=registration_path,
        stop="One baseline and one global substantive candidate; full original questions, no seed experiments, scientific retry/tuning, ordinary qualification or automatic promotion.")
    atomic_json(options.registration, registration)
    print(json.dumps(dict(event="phase3_prepared", registration=str(options.registration),
        reservation=PAIRED_RESERVATION, ceiling=PAID_CEILING, arms=plans)), flush=True)


def resolved(root, registration):
    require(registration.get("scope") == "phase3_paired_research_diagnostic"
            and registration.get("qualification_input") is False, "Only the explicit diagnostic registration may run")
    spec = read_json(root / registration["spec_path"])
    check_spec(spec)
    require(repository_relative(root, root / registration["spec_path"]) == registration["spec_path"]
            and registration["campaign_id"] == spec["campaign_id"]
            and registration["schema_version"] == 1 and registration["protocol_seed"] == 0,
            "Frozen spec path, campaign or protocol identity changed")
    require(file_hash(root / registration["spec_path"]) == registration["spec_sha256"], "Frozen input spec changed")
    requirements, _ = original_contract(root, registration["task_ids"])
    require(requirements == registration["original_requirements"] and spec["task_ids"] == registration["task_ids"],
            "Frozen task roster or original placement changed")
    view = load_view(root, registration["view"])
    require(view["revision"] == 1 and view["id"] == spec["view_id"]
            and view.get("evidence_scope") == "research_diagnostic", "Diagnostic view identity changed")
    require(set(registration["arms"]) == set(ROLES)
            and registration["full_reservation_seconds"] == PAIRED_RESERVATION
            and registration["per_arm_full_reservation_seconds"] == PER_ARM
            and registration["campaign_ceiling_seconds"] == PAID_CEILING
            and registration["candidate_ceiling_seconds"] == PAID_CEILING // 2
            and registration["software_allowance_seconds"] == SOFTWARE_ALLOWANCE,
            "Frozen paired reservations/ceilings changed")
    require({r: p["candidate_id"] for r, p in registration["arms"].items()}
            == {a["role"]: a["candidate_id"] for a in spec["arms"]},
            "Frozen baseline/candidate role identities changed")
    require(set(registration["task_contracts"]) == set(registration["task_ids"]),
            "Retain every frozen task declaration")
    for name, digest in registration["task_contracts"].items():
        require(stable_hash(read_json(root / f"configs/forge/tasks/{name}.json")) == digest,
                f"{name}: frozen task declaration changed")
    requests = {}
    for role in ROLES:
        plan = registration["arms"][role]
        request = resolve_request(root, plan["candidate_id"], study=plan["study_id"])
        require(request_summary(request, registration["task_ids"]) == plan,
                f"{role}: source/runtime/task/study binding changed; prepare a fresh track")
        requests[role] = request
    require(all(r["source"]["digest"] == registration["source_digest"] for r in requests.values()),
            "Both certified arms must bind the declared scientific source")
    return requests


def run(options):
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    require(head(root) == options.source_commit, "Execution must select the exact committed HEAD")
    require(not artifacts.is_relative_to(root), "Bulk artifacts belong outside Git")
    registration = read_json(options.registration)
    require(repository_relative(root, options.registration) == registration["registration_path"],
            "Use the registered committed declaration path")
    requests = resolved(root, registration)
    declarations = {registration["registration_path"], registration["spec_path"],
                    f'configs/forge/views/{registration["view"]}.json'}
    declarations.update(f'configs/forge/tasks/{t}.json' for t in registration["task_ids"])
    for request in requests.values():
        declarations.update((f'configs/forge/ideas/{request["candidate"]["id"]}.json',
                             f'configs/forge/studies/{request["study"]["id"]}.json'))
    phase2.committed_source(root, requests["baseline"]["source"], declarations)
    progress_path = artifacts / "phase3-progress.json"
    if options.drain:
        require(progress_path.is_file(), "Drain requires explicit prior --submit admission")
        progress = read_json(progress_path)
        require(progress["phase"] in {"admitted", "running"} and set(progress["requests"]) == set(ROLES)
                and progress["registration_sha256"] == file_hash(options.registration)
                and progress["source_commit"] == options.source_commit,
                "Drain requires both arms admitted from this exact registration/source")
    else:
        require(options.submit and not progress_path.exists(), "Use a fresh --submit campaign; no unchanged rerun")
    from experiments.forge.queue import Queue, drain
    queue_root = artifacts / "phase3-queue"
    queue = Queue(queue_root, report_root=root / "reports/forge", on_completion=None)
    if options.submit:
        progress = dict(phase="submitting", updated_at=utc_now(), requests={}, source_commit=options.source_commit,
            registration_sha256=file_hash(options.registration), source_digest=registration["source_digest"],
            log=str(artifacts / "logs/phase3-driver.log"), qualification_input=False)
        for role in ROLES:
            plan = registration["arms"][role]
            request = resolve_request(root, plan["candidate_id"], study=plan["study_id"],
                                   queue_root=queue_root, freeze_source=True)
            require(request_summary(request, registration["task_ids"]) == plan, f"{role}: bindings changed while freezing")
            receipt = queue.submit(request, request["study"]["campaign"])
            progress["requests"][role] = receipt["request"]["request_id"]
            atomic_json(progress_path, progress)
            print(json.dumps(dict(event="phase3_submitted", role=role, request=progress["requests"][role])), flush=True)
        progress.update(phase="admitted", updated_at=utc_now())
        atomic_json(progress_path, progress)
    if options.drain:
        progress.update(phase="running", updated_at=utc_now())
        atomic_json(progress_path, progress)
        drain(queue, options.gpus.split(","), workers_per_gpu=1, allow_sharing=True,
              watch=False, campaign=registration["campaign_id"])
        progress.update(phase="diagnostic_complete", updated_at=utc_now())
        atomic_json(progress_path, progress)
        print(json.dumps(dict(event="phase3_complete", time=utc_now())), flush=True)


def diagnostic_scopes(module, artifacts, progress_path):
    """Runtime adapter for the saved publisher; archived implementation stays intact."""
    state = read_json(artifacts / "phase3-queue/queue/state.json")
    progress = read_json(progress_path)
    ids = progress["requests"]
    require(set(ids) == set(ROLES) and len(set(ids.values())) == 2, "Both distinct paired requests must remain visible")
    submissions = {role: state["submissions"][rid] for role, rid in ids.items()}
    jobs = [j for j in state["jobs"].values() if set(j.get("subscribers", [])) & set(ids.values())]
    active = [role for role, sub in submissions.items() if sub["status"] in module.ACTIVE]
    active += [j["definition"]["task_id"] for j in jobs if j["status"] in module.ACTIVE]
    require(not active, f"Publish only terminal certified diagnostic attempts: {active}")
    require(all(sub["request"]["view"].get("evidence_scope") == "research_diagnostic"
                for sub in submissions.values()), "Ordinary requests cannot enter the diagnostic report")
    return [dict(scope="research_diagnostic", queue=artifacts / "phase3-queue", state=state,
                 requests=ids, submissions=submissions, jobs=jobs, active=[], diagnostic_blockers={})]


def publish(options):
    root, artifacts, output = options.repository.resolve(), options.artifacts.resolve(), options.output.resolve()
    registration = read_json(options.registration)
    progress_path = artifacts / "phase3-progress.json"
    progress = read_json(progress_path)
    require(progress["phase"] == "diagnostic_complete"
            and progress["registration_sha256"] == file_hash(options.registration),
            "Publish only the complete registered diagnostic campaign")
    require(head(root) == progress["source_commit"] == options.source_commit,
            "Publish against the exact executed HEAD; source changes require an explicit separate followup")
    requests = resolved(root, registration)
    module = phase2.publisher(root)
    module.ROLES = ROLES
    module.required_questions = lambda _root: registration["original_requirements"]
    module.scopes = lambda _options: diagnostic_scopes(module, artifacts, progress_path)
    collection = module.collect(argparse.Namespace(repository=root, allow_partial=False))
    for context in collection["scopes"]:
        for role, sub in context["submissions"].items():
            require(request_summary(sub["request"], registration["task_ids"]) == registration["arms"][role],
                    f"{role}: certified admission differs from the frozen track")
    audit = phase2.audit_saved_comparison(root, collection, ROLES)
    audit.update(scope="phase3_paired_saved_evidence_audit", qualification_input=False)
    audit.pop("original_six_questions_preserved", None)
    audit["original_sixteen_questions_preserved"] = True
    audit_spec = importlib.util.spec_from_file_location("phase3_own_checkpoints", root / "reports/forge/bcap-develop-integration/audit.py")
    own = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(own)
    audit["own_checkpoint_producers"] = own.own_checkpoints(collection)
    renderer = module._saved_renderer(root)
    media = []
    from PIL import Image
    for entry in collection["final"]:
        item = entry["item"]
        require(item["scope"] == "research_diagnostic" and item["importance"] == "diagnostic",
                "Only the diagnostic arm evidence may be published")
        if item["gate_status"] not in {"PASS", "FAIL"}:
            continue
        gif = output / "media" / f'{item["role"]}-{item["task_id"]}.gif'
        with renderer.forbid_live_execution():
            receipt = module.render_saved(entry, gif, renderer)
        with Image.open(gif) as image:
            require(image.n_frames >= 2, "Actual-training GIF needs multiple certified saved states")
            frames = image.n_frames
        require(receipt.get("task_id", item["task_id"]) == item["task_id"],
                "Saved renderer receipt belongs to another task")
        media.append({**receipt, "role": item["role"], "task_id": item["task_id"],
                      "attempt_id": item["attempt_id"], "gif": str(gif.relative_to(output)),
                      "frames": frames})
        print(json.dumps(dict(event="phase3_saved_media", role=item["role"], task=item["task_id"])), flush=True)
    from experiments.forge.decision_contracts import evaluate
    decisions = {role: evaluate(request, [e["row"] for e in collection["final"] if e["item"]["role"] == role])
                 for role, request in requests.items()}
    histories = [dict(attempt["compact"], final_selected=aid in {e["item"]["attempt_id"] for e in collection["final"]})
                 for aid, attempt in sorted(collection["attempts"].items())]
    result = dict(schema_version=1, scope="phase3_paired_research_diagnostic", qualification_input=False,
        source_commit=progress["source_commit"], source_digest=registration["source_digest"], protocol_seed=0,
        registration_sha256=file_hash(options.registration), original_placement_counts=dict(tier1=6, tier2=10),
        task_cells=collection["cells"], task_results=[e["item"] for e in collection["final"]],
        outcomes={role: dict(Counter(c["gate_status"] for c in collection["cells"] if c["role"] == role)) for role in ROLES},
        study_outcomes=decisions, accounting=collection["accounting"], paid_attempt_history=histories,
        full_reservation_seconds=PAIRED_RESERVATION, paid_ceiling_seconds=PAID_CEILING,
        actual_training_gifs=len(media), optimizer_updates_added=0, sampling_draws_added=0,
        interpretation="Original gates measured only within a paired research diagnostic subset. No ordinary qualification, 21/21 Tier 2, default adoption or automatic promotion claim.")
    atomic_json(output / "phase3-results.json", result)
    atomic_json(output / "phase3-audit.json", audit)
    atomic_json(output / "media/index.json", dict(schema_version=1, qualification_input=False, media=media,
                optimizer_updates_added=0, sampling_draws_added=0))
    # Scoped research readout, not another generated goal leaderboard.
    lines = ["# Paired research diagnostic", "", result["interpretation"], "",
             "| Original tier | Task | Baseline | Candidate |", "| --- | --- | --- | --- |"]
    cells = {(c["role"], c["task_id"]): c for c in collection["cells"]}
    for assignment in registration["original_requirements"]:
        name = assignment["task"]
        lines.append(f'| {assignment["qualification_tier"]} | {name} | '
                     + " | ".join(cells[(role, name)]["gate_status"] for role in ROLES) + " |")
    lines += ["", "[Metrics, certificates, blockers and study outcomes](phase3-results.json) · "
              "[Paired state audit](phase3-audit.json) · [Actual-training GIF receipts](media/index.json)", ""]
    atomic_text(output / "README.md", "\n".join(lines))
    print(json.dumps(dict(event="phase3_published", outcomes=result["outcomes"], gifs=len(media))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "submit", "drain", "publish"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", default="0,1")
    options = parser.parse_args()
    require(not options.prepare or options.spec, "Preparation requires the reviewed input --spec")
    require(options.prepare or options.artifacts, "Execution/publication requires an external --artifacts archive")
    require(not options.publish or options.output, "Publication requires --output")
    if options.prepare:
        prepare(options)
    elif options.publish:
        publish(options)
    else:
        run(options)


if __name__ == "__main__":
    main()
