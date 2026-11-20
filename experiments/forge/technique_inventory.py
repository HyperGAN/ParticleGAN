"""Discover declared techniques and submit a bounded ordinary Forge inventory.

This is a batch interface to the existing planner and queue, not a second
execution lane. Tier prerequisites, frozen recipes and scientific reuse retain
their ordinary semantics. Unsupported declarations remain visible in the plan
without manufacturing an attempt or spending on known blocked tasks.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import identifier, positive_number, read_json, stable_hash
from .execution_policy import completes_tier, group_blockers
from .planning import plan_summary, resolve_idea
from .queue import Queue, drain


DEFAULT_CAMPAIGN = Path("configs/forge/campaigns/technique-inventory-word-split-v1.json")


def discover_techniques(root: Path) -> list[str]:
    """New idea cards enter the next inventory without a maintained name list."""
    # Discovery lists ideas; the current family roster selects exact saved
    # configurations separately. Never expand a benchmark into historical grids.
    names = sorted(path.stem for path in (Path(root) / "configs/forge/ideas").glob("*.json"))
    if not names:
        raise ValueError("technique inventory requires at least one declared idea")
    return names


def _campaign(root, campaign):
    definition = deepcopy(campaign) if isinstance(campaign, dict) else read_json(Path(root) / campaign)
    identifier(definition["id"], "campaign")
    positive_number(definition["budget_seconds"], "campaign budget_seconds")
    positive_number(definition["candidate_budget_seconds"], "candidate_budget_seconds")
    return definition


def _first_task_blockers(request):
    assignments = sorted((a for a in request["view"]["assignments"]
                          if a["qualification_tier"] <= request["through_tier"]),
                         key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"]))
    if not assignments:
        return []
    if completes_tier(request):
        # Reject only a wholly blocked initial tier. One unsupported task must
        # not erase independent measurements from this ordinary batch.
        tier = assignments[0]["qualification_tier"]
        by_task = {member: job for job in request["jobs"]
                   for member in job.get("task_ids", [job["task_id"]])}
        first_tier = [item for item in assignments if item["qualification_tier"] == tier]
        reasons = []
        for item in first_tier:
            blocked = group_blockers(request, by_task[item["task"]])
            if not blocked:
                return []
            reasons.extend(blocked)
        return list(dict.fromkeys(reasons))
    first = assignments[0]
    if first["importance"] != "required":
        return []
    return [f"{first['task']}: {reason}" for reason in
            request["tasks"][first["task"]].get("preflight_blockers", [])]


def _signature(request):
    value = {key: request[key] for key in (
        "candidate", "candidate_revision", "policy_fingerprint", "tasks", "jobs", "through_tier",
        "protocol", "rng", "runtime", "execution_backend", "compute_profiles")}
    if "execution_policy" in request:
        value["execution_policy"] = request["execution_policy"]
    return stable_hash(value)


def _prepare(root, queue_root, *, view_id, through_tier, execution_backend, cuda_model, campaign, queue, technique_ids=None):
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    definition = _campaign(root, campaign)
    state = queue.inspect()
    existing = state.get("campaigns", {}).get(definition["id"])
    if existing and existing["definition"] != definition:
        raise ValueError("campaign definition is immutable; use a new campaign id")
    discovered = discover_techniques(root)
    names = discovered
    family_choices = None
    if technique_ids is None:
        from .trainer_families import CURRENT_SELECTION, current_family_candidates, load_declared_revisions
        if (root / CURRENT_SELECTION).is_file():
            family_choices = current_family_candidates(root)
            # A declared revision (configs/forge/selections/family-revisions-*.json)
            # is the recipe a family is measured with next; its measured pin is
            # replaced only by a later publication of that measurement.
            revisions = load_declared_revisions(root)
            for index, choice in enumerate(family_choices):
                revision = revisions.get(choice["configuration_family"])
                if revision and revision["status"] == "awaiting_measurement":
                    family_choices[index] = {**choice, "candidate_id": revision["candidate_id"],
                                             "selection_kind": "declared_revision", "supersedes": revision["supersedes"]}
            names = [choice["candidate_id"] for choice in family_choices]
    if technique_ids is not None:
        if (not isinstance(technique_ids, (list, tuple)) or not technique_ids
                or any(not isinstance(name, str) or not name for name in technique_ids)):
            raise ValueError("explicit technique_ids must be a nonempty list of declared idea IDs")
        if len(set(technique_ids)) != len(technique_ids):
            raise ValueError("explicit technique_ids must be unique")
        unknown = sorted(set(technique_ids) - set(discovered))
        if unknown:
            raise ValueError("explicit technique_ids contain undeclared ideas: " + ", ".join(unknown))
        names = list(technique_ids)
    requests = [resolve_idea(root, name, view_id=view_id, through_tier=through_tier,
                            execution_backend=execution_backend, cuda_model=cuda_model)
                for name in names]
    sources = {request["source"]["digest"] for request in requests}
    rows = []
    for request in requests:
        row = plan_summary(request, state)
        row["source_digest"] = request["source"]["digest"]
        blockers = list(request["preflight_blockers"]) + _first_task_blockers(request)
        by_task = {member: job for job in request["jobs"] for member in job["task_ids"]}
        permitted = {t["task"] for t in row["tasks"] if t["permitted_by_tier_cap"]}
        row["declared_worst_case_seconds"] = sum(job["budget_seconds"] for job in request["jobs"]
                                                 if set(job["task_ids"]) <= permitted)
        row["submission_blockers"] = blockers
        row["submission_status"] = "BLOCKED" if blockers else "READY"
        row["candidate_budget_covers_declared_ceiling"] = (
            definition["candidate_budget_seconds"] >= row["declared_worst_case_seconds"])
        max_tier = max([through_tier] + [t["qualification_tier"] for t in row["tasks"]])
        row["required_tier_totals"] = {
            str(tier): sum(t["qualification_tier"] == tier and t["importance"] == "required"
                           for t in row["tasks"]) for tier in range(1, max_tier + 1)}
        for task in row["tasks"]:
            saved = state.get("jobs", {}).get(by_task[task["task"]]["compatibility_key"], {})
            receipt = next((r for r in (saved.get("result") or {}).get("task_results", [])
                            if r["task_id"] == task["task"]), None)
            task["evidence_status"] = receipt["gate_status"] if receipt else "UNKNOWN"
            task["preflight_status"] = "BLOCKED" if request["preflight_blockers"] or task["blockers"] else "READY"
        rows.append(row)
    declared = sum(row["declared_worst_case_seconds"] for row in rows)
    summary = {"schema_version": 1, "stage": "planned", "view": view_id, "through_tier": through_tier,
               "execution_backend": execution_backend, "source_digests": sorted(sources),
               "source_digest": next(iter(sources)) if len(sources) == 1 else None,
               "campaign": definition, "queue_root": str(queue_root),
               "logs": str(queue_root / "events.jsonl"), "technique_count": len(rows), "candidates": rows,
               "declared_worst_case_seconds": declared,
               "worst_case_seconds": sum(row["worst_case_seconds"] for row in rows
                                         if not row["submission_blockers"]),
               "campaign_budget_covers_declared_ceiling": definition["budget_seconds"] >= declared,
               "guide": "EXPERIMENTATION.md"}
    if technique_ids is not None:
        summary["explicit_technique_ids"] = names
        summary["unrequested_technique_ids"] = sorted(set(discovered) - set(names))
    if family_choices is not None:
        summary["selection_scope"] = "one_current_configuration_per_family"
        summary["family_selections"] = family_choices
        summary["unrequested_technique_ids"] = sorted(set(discovered) - set(names))
    return requests, summary


def plan_inventory(root: Path, queue_root: Path, *, view_id="discriminator_stability", through_tier=3,
                   execution_backend="cuda", cuda_model=None, campaign=DEFAULT_CAMPAIGN, queue=None, technique_ids=None) -> dict:
    """Plan current family configurations, or an explicit historical idea roster."""
    queue = queue or Queue(queue_root)
    _, summary = _prepare(root, queue_root, view_id=view_id, through_tier=through_tier,
                          execution_backend=execution_backend, cuda_model=cuda_model,
                          campaign=campaign, queue=queue, technique_ids=technique_ids)
    return summary


def enqueue_inventory(root: Path, queue_root: Path, *, view_id="discriminator_stability", through_tier=3,
                      execution_backend="cuda", cuda_model=None, campaign=DEFAULT_CAMPAIGN, queue=None, technique_ids=None) -> dict:
    """Freeze every executable row before admitting any request; launch nothing.

    Repeated calls attach to the queue's exact request/job identities. Validation
    errors, source drift and lifecycle refusals propagate rather than becoming
    scientific BLOCKED receipts.
    """
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    if queue.root != queue_root:
        raise ValueError("inventory queue_root differs from the supplied Queue")
    if queue.report_root is None:
        queue.report_root = root / "reports/forge"
    elif queue.report_root.parent.parent != root:
        raise ValueError("inventory Queue report_root differs from its research repository")
    requests, summary = _prepare(root, queue_root, view_id=view_id, through_tier=through_tier,
                                 execution_backend=execution_backend, cuda_model=cuda_model,
                                 campaign=campaign, queue=queue, technique_ids=technique_ids)
    frozen = []
    for request, row in zip(requests, summary["candidates"]):
        if row["submission_blockers"]:
            continue
        resolved = resolve_idea(root, request["candidate"]["id"], view_id=view_id,
                                through_tier=through_tier, execution_backend=execution_backend,
                                cuda_model=cuda_model, queue_root=queue_root, freeze_source=True)
        if (resolved["source"]["digest"] != request["source"]["digest"]
                or _signature(resolved) != _signature(request)):
            raise ValueError("inventory source or scientific declarations changed before submission; plan and submit again")
        frozen.append((resolved, row))
    for request, row in frozen:
        entry = queue.submit(request, summary["campaign"])
        row.update(request_id=entry["request"]["request_id"], submission_status=entry["status"],
                   request=str(queue_root / "queue/requests" / f"{entry['request']['request_id']}.json"))
    summary["stage"] = "enqueued"
    summary["submitted_count"] = len(frozen)
    summary["blocked_count"] = len(requests) - len(frozen)
    return summary


def run_inventory(root: Path, queue_root: Path, *, devices=None, view_id="discriminator_stability", through_tier=3,
                  execution_backend="cuda", cuda_model=None, campaign=DEFAULT_CAMPAIGN, queue=None, technique_ids=None) -> dict:
    """Submit, then drain only this bounded campaign with ordinary prerequisites."""
    devices = list(devices or (["cpu"] if execution_backend == "cpu" else ["0", "1"]))
    if (execution_backend == "cpu") != (devices == ["cpu"]):
        raise ValueError("inventory execution backend must agree with devices")
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    summary = enqueue_inventory(root, queue_root, view_id=view_id, through_tier=through_tier,
                                execution_backend=execution_backend, cuda_model=cuda_model,
                                campaign=campaign, queue=queue, technique_ids=technique_ids)
    if summary["submitted_count"]:
        drain(queue, devices, campaign=summary["campaign"]["id"])
    state = queue.inspect()
    for row in summary["candidates"]:
        if "request_id" in row:
            entry = state["submissions"][row["request_id"]]
            row.update(submission_status=entry["status"], reason=entry.get("reason"))
    summary["stage"] = "drained"
    summary["campaign_accounting"] = state.get("campaigns", {}).get(summary["campaign"]["id"])
    return summary
