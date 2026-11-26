"""Bounded image-performance studies submitted through the ordinary Forge queue."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import atomic_json, file_hash, identifier, read_json, stable_hash
from .planning import load_idea, plan_summary, resolve_idea
from .studies import scaffold, study_path, validate_study
from .trainer_families import CURRENT_SELECTION, current_family_candidates, family_for_candidate
from .views import load_tasks, load_view


SCHEMA = "forge_image_performance_campaign_v1"


def manifest_path(root, registration):
    path = Path(registration)
    return (Path(root) / "configs/forge/performance" / f"{identifier(registration, 'performance registration')}.json"
            if path.suffix != ".json" and len(path.parts) == 1 else Path(root) / path)


def _write_immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"performance declaration is immutable; use a new id: {path}")
        return
    atomic_json(path, value)


def register(root, registration, *, view_id="dcgan_performance", execution_backend="cuda",
             cuda_model=None, family_ids=None, candidate_overrides=None, campaign=None):
    """Freeze one exact selected recipe per family and the complete reservation.

    The old image toy is only a host-change declaration control. It is never a
    throughput baseline: every numerical comparison uses the same new task.
    """
    root = Path(root).resolve()
    registration = identifier(registration, "performance registration")
    view = load_view(root, view_id)
    tasks = load_tasks(root)
    if (view.get("evidence_scope") != "research_diagnostic"
            or any(tasks[a["task"]]["adapter"] != "dcgan_performance_v1" for a in view["assignments"])):
        raise ValueError("image performance registration requires only explicit performance diagnostics")
    if execution_backend not in {"cpu", "cuda"} or (cuda_model and execution_backend != "cuda"):
        raise ValueError("select a CPU or CUDA performance cohort")
    choices = deepcopy(current_family_candidates(root, family_ids=family_ids))
    if not choices or (family_ids is not None and set(family_ids) != {c["family"] for c in choices}):
        raise ValueError("performance registration must select known current families")
    overrides = dict(candidate_overrides or {})
    if set(overrides) - {choice["family"] for choice in choices}:
        raise ValueError("performance candidate overrides must name selected families")
    for choice in choices:
        if choice["family"] not in overrides:
            continue
        candidate_id = identifier(overrides[choice["family"]], "performance candidate override")
        declaration = load_idea(root, candidate_id)
        family = family_for_candidate(root, candidate_id, declaration, current_presentation=True)
        if family.get("reporting_family", family["id"]) != choice["family"]:
            raise ValueError("performance candidate override belongs to another trainer family")
        choice["candidate_id"] = candidate_id
    task_ids = [a["task"] for a in view["assignments"]]
    allowance = sum(tasks[name]["resources"]["timeout_seconds"] for name in task_ids)
    if campaign is None:
        campaign = {"id": registration, "candidate_budget_seconds": allowance,
                    "budget_seconds": allowance * len(choices)}
    else:
        campaign = deepcopy(campaign)
        if (set(campaign) != {"id", "candidate_budget_seconds", "budget_seconds"}
                or type(campaign["candidate_budget_seconds"]) not in (int, float)
                or campaign["candidate_budget_seconds"] != allowance
                or type(campaign["budget_seconds"]) not in (int, float)
                or not allowance * len(choices) <= campaign["budget_seconds"] < float("inf")):
            raise ValueError("performance campaign must reserve the full selected matrix and exact per-candidate allowance")
        identifier(campaign["id"], "performance campaign")
    studies, selections = [], []
    for choice in choices:
        candidate = load_idea(root, choice["candidate_id"])
        if "decision_contract" in candidate:
            raise ValueError("a legacy embedded-contract candidate needs an explicit v3 declaration before this new study")
        study_id = f"{registration}-{choice['family']}"
        recipe_scope = "declared" if overrides else "selected"
        study = scaffold(study_id, choice["candidate_id"], choice["candidate_id"], view_id,
                         hypothesis=f"Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact {recipe_scope} global recipe.",
                         execution_backend=execution_backend)
        study["status"] = "ready"
        if cuda_model:
            study["scope"]["cuda_model"] = cuda_model
        study["campaign"] = campaign
        study["control"]["task_map"] = {name: "img_stripes2" for name in task_ids}
        study["prior_evidence"] = [{"path": CURRENT_SELECTION.as_posix(), "selector": [],
                                    "identity": {"scope": "whole_candidate_family_current", "default_adoption": False},
                                    "use": "motivation_only"}]
        signature = {"task_id": task_ids[0], "metric": "finite_update_fraction", "phase": "final", "threshold": 1.}
        study["prediction"] = {**signature, "op": ">="}
        study["falsifier"] = {**signature, "op": "<"}
        study["competing_explanation"] = (
            "Synthetic update throughput omits data loading and image quality. Initialization, policy phases, "
            "thermal state and shared hardware can affect timing. Warmup, isolated devices, explicit hardware "
            "receipts and completion checks constrain this diagnostic; no scientific promotion follows. "
            "The original image toy control establishes the declared host change only and is never a speed comparator.")
        validate_study(study)
        _write_immutable(study_path(root, study_id), study)
        studies.append(study_id)
        selections.append({"family": choice["family"], "label": choice["label"],
                           "candidate_id": choice["candidate_id"], "study": study_id,
                           "candidate_declaration_sha256": file_hash(root / "configs/forge/ideas" / f"{choice['candidate_id']}.json")})
    manifest = {"schema": SCHEMA, "id": registration, "view": view_id,
                "execution_backend": execution_backend, "cuda_model": cuda_model,
                "task_ids": task_ids, "family_selections": selections, "studies": studies,
                "task_contract_sha256": {name: stable_hash(tasks[name]) for name in task_ids},
                "campaign": campaign, "protocol_seed": 0,
                "qualification_input": False, "qualification_reuse": False,
                "comparison_scope": "same task, runtime and hardware; exact selected global recipes; synthetic data; no image quality claim"}
    if overrides:
        manifest["candidate_overrides"] = overrides
        manifest["comparison_scope"] = "same task, runtime and hardware; exact declared global recipes including explicit candidate overrides; synthetic data; no image quality claim"
    _write_immutable(manifest_path(root, registration), manifest)
    _write_immutable(root / "configs/forge/campaigns" / f"{campaign['id']}.json", campaign)
    return {"registration": str(manifest_path(root, registration)), **manifest, "training_launched": False}


def _load(root, registration):
    manifest = read_json(manifest_path(root, registration))
    if manifest.get("schema") != SCHEMA or manifest.get("protocol_seed") != 0:
        raise ValueError("unsupported image performance campaign")
    if manifest.get("qualification_input") is not False or manifest.get("qualification_reuse") is not False:
        raise ValueError("performance measurements cannot supply qualification")
    for selected in manifest["family_selections"]:
        path = Path(root) / "configs/forge/ideas" / f"{selected['candidate_id']}.json"
        if file_hash(path) != selected["candidate_declaration_sha256"]:
            raise ValueError("selected performance recipe declaration changed; use a new registration id")
    tasks = load_tasks(root)
    if manifest.get("task_contract_sha256") != {name: stable_hash(tasks[name]) for name in manifest["task_ids"]}:
        raise ValueError("performance task contract changed; use a new registration id")
    return manifest


def prepare(root, queue_root, registration, *, queue):
    from .planning import resolution_cache
    with resolution_cache(root):
        manifest = _load(root, registration)
        requests, rows = [], []
        state = queue.inspect()
        for selected in manifest["family_selections"]:
            request = resolve_idea(root, selected["candidate_id"], study=selected["study"])
            if request["protocol"]["seed"] != 0 or request["view"]["evidence_scope"] != "research_diagnostic":
                raise ValueError("performance lane requires diagnostic seed-zero execution")
            requests.append(request)
            row = plan_summary(request, state)
            rows.append({"family": selected["family"], "candidate": selected["candidate_id"],
                         "study": selected["study"], "preflight_blockers": row["preflight_blockers"],
                         "tasks": row["tasks"], "worst_case_seconds": row["worst_case_seconds"],
                         "source_digest": request["source"]["digest"],
                         "study_status": request["study_review"]["status"]})
        return manifest, requests, {"schema": SCHEMA, "stage": "planned", "registration": manifest["id"],
                                    "campaign": manifest["campaign"], "view": manifest["view"],
                                    "execution_backend": manifest["execution_backend"], "families": rows,
                                    "declared_worst_case_seconds": sum(row["worst_case_seconds"] for row in rows),
                                    "qualification_input": False, "logs": str(Path(queue_root) / "events.jsonl")}


def plan(root, queue_root, registration, *, queue):
    return prepare(root, queue_root, registration, queue=queue)[2]


def enqueue(root, queue_root, registration, *, queue):
    from .sources import snapshot_source
    manifest, requests, summary = prepare(root, queue_root, registration, queue=queue)
    submitted = []
    for selected, request in zip(manifest["family_selections"], requests):
        if request["preflight_blockers"]:
            submitted.append({"family": selected["family"], "candidate": selected["candidate_id"],
                              "status": "BLOCKED", "reasons": request["preflight_blockers"]})
            continue
        request["source"]["snapshot_path"] = str(snapshot_source(Path(root), Path(queue_root), request["source"]))
        entry = queue.submit(request, deepcopy(manifest["campaign"]))
        submitted.append({"family": selected["family"], "candidate": selected["candidate_id"],
                          "request_id": entry["request"]["request_id"], "status": entry["status"]})
    return {**summary, "stage": "enqueued", "submissions": submitted}
