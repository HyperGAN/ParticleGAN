"""Prepare, execute and summarize one full-tier inventory through Forge.

No training loop lives here. The selected recipes, frozen public tasks, queue,
independent evaluator and existing source publication own the experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.boundaries import TUNABLE_FIELDS
from experiments.forge.configuration_search import _declarations, _load_spec, enqueue_search
from experiments.forge.decision_contracts import validate_admission
from experiments.forge.planning import load_idea, plan_summary, resolve_idea
from experiments.forge.queue import Queue, drain
from experiments.forge.trainer_families import load_families
from experiments.forge.views import load_tasks, load_view

ROUND = Path("configs/forge/rounds/tier1-completion-v1.json")
CAMPAIGN = Path("configs/forge/campaigns/tier1-completion-v1.json")
REPORT = Path("reports/forge/tier1-completion")
ID = "tier1-completion-v1"


def singleton_search(root, row, campaign):
    """Register the selected card unchanged, without another configuration choice."""
    card = load_idea(root, row["candidate_id"])
    if "configuration_id" not in card:
        return None
    names = set(card["recipe_overrides"]) & TUNABLE_FIELDS
    axis = next((name for name in ("lr", "d_lr_mult") if name in names), None)
    if axis is None:
        axis = next(iter(sorted(names)), None)
    if axis is None:
        raise ValueError("selected configuration needs an already declared tunable setting")
    protocol = read_json(root / "configs/forge/defaults.json")["protocol"]
    spec = {"schema_version": 1, "id": ID + "-" + row["family"],
            "trainer_family": card["trainer_family"], "base_candidate": card["id"],
            "grid": {axis: [card["recipe_overrides"][axis]]}, "tuning_through_tier": 1,
            "view": row["view"], "execution_backend": "cuda", "protocol": protocol,
            "protocol_hash": stable_hash(read_json(root / "configs/forge/protocols" / (protocol + ".json"))),
            "campaign": campaign,
            "rationale": "Admission registration for one existing selected configuration; no tuning, new card, recipe change or independent confirmation."}
    declarations = _declarations(root, _load_spec(root, spec))
    if len(declarations) != 1 or stable_hash(declarations[0][0]) != row["declaration_sha256"]:
        raise ValueError("singleton registration must retain the exact selected card")
    return spec


def registration_specs(root, definition):
    """Revalidate every frozen registration before any admission can mutate the queue."""
    rows = {row["candidate_id"]: row for row in definition["candidate_roster"]}
    expected = {name for name in rows if "configuration_id" in load_idea(root, name)}
    references = definition["configuration_searches"]
    if len(references) != len(expected) or {item["candidate_id"] for item in references} != expected:
        raise ValueError("registration roster must cover exactly the selected configuration cards")
    specs = []
    for item in references:
        spec = _load_spec(root, item["spec"])
        if stable_hash(spec) != item["spec_sha256"] or spec != singleton_search(root, rows[item["candidate_id"]], definition["campaign"]):
            raise ValueError("singleton registration differs from the frozen roster")
        specs.append(spec)
    return specs


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True, allow_nan=False), flush=True)


def prepare(root=ROOT):
    """Freeze the current whole-configuration roster; launch no work."""
    root = Path(root).resolve()
    tasks = load_tasks(root)
    audit = "clockfree_audit_measurement_v1"
    if audit not in tasks:
        raise ValueError("merge Tier 1 support before preparing this round")
    view_path = root / "configs/forge/views/discriminator_stability.json"
    view = read_json(view_path)
    if not any(a["task"] == audit for a in view["assignments"]):
        view["revision"] += 1
        view["assignments"].append({"task": audit, "qualification_tier": 1,
                                    "importance": "diagnostic", "order": 100})
        view["policy_change_reason"] = "Revision 5 retains the six required Tier 1 tests and adds a separately scoped clock-audit measurement diagnostic. This full-tier measurement round does not calibrate placement or promote a default; original clock-free eligibility and archived view grades remain unchanged."
        atomic_json(view_path, view)
    families = load_families(root)
    selections = read_json(root / "configs/forge/selections/family-current-v1.json")
    pins = {pin["trainer_family"]: pin for pin in selections["selections"]}
    roster = []
    for family, card in sorted(families.items()):
        candidate = pins.get(family, {}).get("candidate_id", card["canonical_candidate"])
        # Old cloud cards stay historical; the registry canonical owns the
        # forward task adaptation, with the selected global recipe preserved.
        declaration = load_idea(root, candidate)
        policy = family in {"atlas", "e22"}
        view_id = "tier1_policy_coverage" if policy else "discriminator_stability"
        roster.append({"family": family, "candidate_id": candidate, "view": view_id,
                       "declaration_sha256": stable_hash(declaration),
                       "measurement_tasks": [audit] if not policy else []})
    allowance = 0
    maximum = 0
    for row in roster:
        view = load_view(root, row["view"])
        names = {a["task"] for a in view["assignments"] if a["qualification_tier"] == 1}
        budget = sum(tasks[name]["resources"]["timeout_seconds"] for name in names)
        row.update(task_ids=sorted(names), view_fingerprint=stable_hash(view), allowance_seconds=budget,
                   task_definition_sha256={name: stable_hash(tasks[name]) for name in sorted(names)})
        allowance += budget
        maximum = max(maximum, budget)
    campaign = {"id": ID, "budget_seconds": 2 * allowance,
                "candidate_budget_seconds": 2 * maximum,
                "description": "One fixed recipe per family; complete Tier 1 only. One documented infrastructure-repair retry per task at most; no scientific retry or tuning."}
    definition = {"schema_version": 1, "id": ID, "through_tier": 1,
                  "execution_backend": "cuda", "gpus": ["0", "1"], "workers_per_gpu": 1,
                  "candidate_roster": roster, "declared_first_attempt_ceiling_seconds": allowance,
                  "max_infrastructure_retries_per_task": 1, "campaign": campaign,
                  "scope": "Current Tier 1 coverage; scoped policy and clock diagnostics retain separate identities. No default adoption or later-tier execution."}
    definition["configuration_searches"] = []
    for row in roster:
        spec = singleton_search(root, row, campaign)
        if spec is None:
            continue
        relative = Path("configs/forge/searches") / (spec["id"] + ".json")
        path = root / relative
        if path.exists() and read_json(path) != spec:
            raise ValueError("frozen singleton registration already differs: " + str(relative))
        atomic_json(path, spec)
        definition["configuration_searches"].append({"candidate_id": row["candidate_id"],
            "spec": relative.as_posix(), "spec_sha256": stable_hash(spec)})
    for relative, data in ((ROUND, definition), (CAMPAIGN, campaign)):
        path = root / relative
        if path.exists() and read_json(path) != data:
            previous = read_json(path)
            # The stopped pre-training round can add admission references only;
            # its exact recipe/task roster and both budget caps remain frozen.
            if relative != ROUND or "configuration_searches" in previous or previous != {
                    key: value for key, value in data.items() if key != "configuration_searches"}:
                raise ValueError(f"frozen round already differs: {relative}; do not replenish its budget")
        atomic_json(path, data)
    emit("prepared", families=len(roster), first_attempt_ceiling_seconds=allowance,
         retry_inclusive_ceiling_seconds=campaign["budget_seconds"])
    return definition


def requests(root, queue_root, *, freeze=False):
    from experiments.forge.planning import resolution_cache
    definition = read_json(root / ROUND)
    registration_specs(root, definition)
    with resolution_cache(root):
        tasks = load_tasks(root)
        resolved = []
        for row in definition["candidate_roster"]:
            if stable_hash(load_idea(root, row["candidate_id"])) != row["declaration_sha256"]:
                raise ValueError("recipe declaration changed after roster freeze")
            if stable_hash(load_view(root, row["view"])) != row["view_fingerprint"]:
                raise ValueError("view changed after roster freeze")
            if {name: stable_hash(tasks[name]) for name in row["task_ids"]} != row["task_definition_sha256"]:
                raise ValueError("task definition changed after roster freeze")
            req = resolve_idea(root, row["candidate_id"], view_id=row["view"], through_tier=1,
                               execution_backend="cuda", queue_root=queue_root, freeze_source=freeze)
            if req.get("execution_policy", {}).get("mode") != "complete_current_tier":
                raise ValueError("merge complete-tier execution before this round")
            if req["preflight_blockers"]:
                raise ValueError(f"candidate admission blocked: {row['family']}: {req['preflight_blockers']}")
            allowance = sum(job["budget_seconds"] for job in req["jobs"] if Queue._authorized(req, job))
            if allowance != row["allowance_seconds"]:
                raise ValueError("resolved task allowance differs from frozen budget")
            resolved.append((row, req))
        if len({req["source"]["digest"] for _, req in resolved}) != 1:
            raise ValueError("one frozen implementation source is required across this round")
        return definition, resolved


def plan(root, queue_root):
    definition, resolved = requests(root, queue_root)
    queue = Queue(queue_root, report_root=root / "reports/forge")
    state = queue.inspect()
    return {"round": definition["id"], "source": resolved[0][1]["source"],
            "campaign": definition["campaign"], "logs": str(queue_root / "events.jsonl"),
            "candidates": [{"family": row["family"], **plan_summary(req, state)} for row, req in resolved]}


def enqueue(root, queue_root):
    from experiments.forge.planning import resolution_cache
    with resolution_cache(root):
        definition, resolved = requests(root, queue_root, freeze=True)
        verify_committed_source(root, resolved[0][1]["source"])
        queue = Queue(queue_root, report_root=root / "reports/forge")
        # Check every ordinary legacy declaration before registering configurations.
        # Search admission itself retains the existing exact bounded-registration
        # validation; the shared campaign and scientific requests do not change.
        for _, req in resolved:
            if "configuration_id" not in req["candidate"]:
                validate_admission(req, definition["campaign"], root=root)
        for spec in registration_specs(root, definition):
            registered = enqueue_search(root, queue_root, spec, queue=queue)
            if registered["submitted_count"] != 1:
                raise ValueError("singleton configuration registration was not admitted")
            emit("registered", study=spec["id"], candidate=spec["base_candidate"])
        entries = []
        for row, req in resolved:
            entry = queue.submit(req, definition["campaign"])
            entries.append({"family": row["family"], "candidate_id": row["candidate_id"],
                            "request_id": entry["request"]["request_id"]})
            emit("submitted", **entries[-1])
        atomic_json(queue_root / ID / "roster.json", entries)
        return queue


def verify_committed_source(root, source):
    """Every captured byte must exist at the reviewed Git origin, even ignored files."""
    names = list(source["files"])
    if any("\n" in name or "\r" in name for name in names):
        raise ValueError("source paths cannot contain line delimiters")
    references = "".join(source["origin_commit"] + ":" + name + "\n" for name in names)
    data = subprocess.run(["git", "cat-file", "--batch"], input=references.encode(),
                          cwd=root, stdout=subprocess.PIPE, check=True).stdout
    cursor = 0
    for name in names:
        end = data.index(b"\n", cursor)
        header = data[cursor:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("captured source is not committed: " + name)
        size = int(header[2])
        cursor = end + 1
        content = data[cursor:cursor + size]
        cursor += size + 1
        if hashlib.sha256(content).hexdigest() != source["files"][name]:
            raise ValueError("captured source differs from reviewed Git bytes: " + name)


def summarize(root, queue_root):
    definition = read_json(root / ROUND)
    queue = Queue(queue_root, report_root=root / "reports/forge")
    state = queue.inspect()
    rows = []
    for row in read_json(queue_root / ID / "roster.json"):
        entry = state["submissions"][row["request_id"]]
        request = entry["request"]
        names = {a["task"] for a in request["view"]["assignments"] if a["qualification_tier"] == 1}
        results = {r["task_id"]: r for r in queue._results(state, entry)}
        owners = {member: state["jobs"][job["compatibility_key"]].get("result") or {}
                  for job in request["jobs"] for member in job.get("task_ids", [job["task_id"]])}
        tasks = []
        for name in sorted(names):
            result = results.get(name, {})
            blockers = request["tasks"][name].get("preflight_blockers", [])
            tasks.append({"task_id": name, "status": result.get("gate_status", "BLOCKED" if blockers else "UNKNOWN"),
                          "reason": result.get("reason") or "; ".join(blockers),
                          "metrics": result.get("metrics", {}), "cost": result.get("cost", {}),
                          "device": result.get("device", result.get("cost", {}).get("device")),
                          "attempt_id": owners[name].get("attempt_id"),
                          "canonical_result_hash": stable_hash(owners[name]) if owners[name] else None,
                          "compatibility_key": next(j["compatibility_key"] for j in request["jobs"] if name in j.get("task_ids", [j["task_id"]]))})
        rows.append({**row, "source": request["source"]["digest"],
                     "source_commit": request["source"].get("origin_commit"), "view": request["view"]["id"],
                     "submission_status": entry["status"], "reason": entry.get("reason"),
                     "counts": dict(Counter(t["status"] for t in tasks)), "tasks": tasks})
    report = {"schema_version": 1, "round": ID, "qualification_input": False,
              "campaign": definition["campaign"], "accounting": state["campaigns"][ID], "candidates": rows,
              "logs": str(queue_root / "events.jsonl")}
    atomic_json(root / REPORT / "results.json", report)
    emit("readout", counts=dict(Counter(t["status"] for r in rows for t in r["tasks"])),
         charged_seconds=report["accounting"]["spent_seconds"])
    return report


def archive(root, queue_root, destination=None):
    """Archive byte-exact originals locally; commit only the hash receipt."""
    state = Queue(queue_root, report_root=root / "reports/forge").inspect()
    entries = read_json(queue_root / ID / "roster.json")
    attempts, sources, attempt_paths = set(), set(), {}
    for row in entries:
        request = state["submissions"][row["request_id"]]["request"]
        sources.add(request["source"]["digest"])
        for job in request["jobs"]:
            if not Queue._authorized(request, job):
                continue
            for attempt in state["jobs"][job["compatibility_key"]]["attempts"]:
                attempts.add(attempt["attempt_id"])
                attempt_paths[attempt["attempt_id"]] = Path(attempt["path"])
    destination = Path(destination).resolve() if destination is not None else root / "artifacts/forge" / (ID + ".tar.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("archive already exists; preserve its exact published identity")
    members = {}
    with tarfile.open(destination, "w:gz") as bundle:
        def add_tree(path, prefix):
            if not path.exists():
                return
            paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for member in paths:
                if member.is_symlink() or "__pycache__" in member.parts:
                    continue
                name = str(Path(prefix) / member.relative_to(path)) if path.is_dir() else prefix
                bundle.add(member, arcname=name, recursive=False)
                members[name] = {"sha256": file_hash(member), "bytes": member.stat().st_size}
        for attempt in sorted(attempts):
            add_tree(root / "reports/forge/attempts" / attempt, "durable/" + attempt)
            add_tree(attempt_paths[attempt], "attempts/" + attempt)
        add_tree(queue_root / ID, "queue/" + ID)
        for source in sorted(sources):
            add_tree(queue_root / "snapshots" / source, "snapshots/" + source)
        add_tree(queue_root / "events.jsonl", "queue/events.jsonl")
        for row in entries:
            add_tree(queue_root / "queue/requests" / (row["request_id"] + ".json"), "queue/requests/" + row["request_id"] + ".json")
    receipt = {"schema_version": 1, "archive": {"path": str(destination), "sha256": file_hash(destination),
                                               "bytes": destination.stat().st_size},
               "manifest": {"executed_commit": read_json(root / REPORT / "results.json")["candidates"][0]["source_commit"],
                            "original_receipts_sha256": {name: row["sha256"] for name, row in members.items()}},
               "attempt_ids": sorted(attempts), "source_digests": sorted(sources)}
    atomic_json(root / REPORT / "artifact-inventory.json", receipt)
    emit("archived", attempts=len(attempts), bytes=destination.stat().st_size, path=str(destination))
    return receipt


def export_media(root, queue_root):
    from experiments.forge.tier1_media import export_attempt
    queue = Queue(queue_root, report_root=root / "reports/forge")
    state = queue.inspect()
    media = []
    for row in read_json(queue_root / ID / "roster.json"):
        request = state["submissions"][row["request_id"]]["request"]
        seen = set()
        for job in request["jobs"]:
            if not Queue._authorized(request, job):
                continue
            for attempt in state["jobs"][job["compatibility_key"]]["attempts"]:
                name = attempt["attempt_id"]
                if name in seen:
                    continue
                seen.add(name)
                original = root / "reports/forge/attempts" / name
                if not (original / "evidence.json").is_file():
                    continue
                output = root / REPORT / "media" / row["family"] / name
                for receipt in export_attempt(original, output):
                    task = receipt["task_id"]
                    media.append({"family": row["family"], "attempt_id": name, **receipt,
                                  "gif": (output / (task + ".gif")).relative_to(root).as_posix()})
                    emit("media", family=row["family"], task=task, attempt=name)
    atomic_json(root / REPORT / "media.json", {"schema_version": 1, "qualification_input": False,
                                                "items": media})
    return media


def retry(root, queue_root, compatibility_key, reason):
    queue = Queue(queue_root, report_root=root / "reports/forge")
    state = queue.inspect()
    entries = read_json(queue_root / ID / "roster.json")
    allowed = {job["compatibility_key"] for row in entries
               for job in state["submissions"][row["request_id"]]["request"]["jobs"]
               if Queue._authorized(state["submissions"][row["request_id"]]["request"], job)}
    if compatibility_key not in allowed or len(state["jobs"][compatibility_key]["attempts"]) != 1:
        raise ValueError("this round permits one documented infrastructure retry per authorized task")
    if not reason or not reason.strip():
        raise ValueError("document the completed infrastructure repair")
    return queue.retry(compatibility_key, reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "enqueue", "run", "report", "media", "archive", "retry"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--compatibility-key")
    parser.add_argument("--reason")
    parser.add_argument("--archive-path", type=Path)
    args = parser.parse_args()
    root, queue_root = args.root.resolve(), args.queue_root.resolve()
    if args.stage in {"enqueue", "run"}:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if not args.expected_commit or head != args.expected_commit:
            raise ValueError("execution requires the reviewed --expected-commit")
        subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=root, check=True)
    if args.stage == "prepare":
        prepare(root)
    elif args.stage == "plan":
        print(json.dumps(plan(root, queue_root), sort_keys=True, allow_nan=False))
    elif args.stage in {"enqueue", "run"}:
        queue = enqueue(root, queue_root)
        if args.stage == "run":
            drain(queue, ["0", "1"], campaign=ID)
            summarize(root, queue_root)
    elif args.stage == "report":
        summarize(root, queue_root)
    elif args.stage == "archive":
        archive(root, queue_root, args.archive_path)
    elif args.stage == "media":
        export_media(root, queue_root)
    elif args.stage == "retry":
        retry(root, queue_root, args.compatibility_key, args.reason)


if __name__ == "__main__":
    main()
