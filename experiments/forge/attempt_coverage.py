"""Resume missing selected and research run attempts without altering qualification.

Examples:
  python -m experiments.forge.attempt_coverage plan --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY
  python -m experiments.forge.attempt_coverage run --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --watch
  python -m experiments.forge.attempt_coverage plan --intake policy-cloud --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY
  python -m experiments.forge.attempt_coverage run --intake policy-cloud --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --gpus 0,1
  python -m experiments.forge.attempt_coverage run --intake policy-cloud --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --gpus 0,1 --retry-interrupted

The policy-cloud intake schedules only the scoped Atlas/E22 policy-cloud
coverage cohorts (clean and as-served scoring laws; law-independent tests run
once) and keeps their records inside RUN_DIRECTORY, never in the repository
report. ``python -m experiments.forge.policy_cloud_publication`` publishes a
drained run into its own committed cohort report directory. This intake also
accepts a single GPU (``--gpus 1``, at most two concurrent tests) as a
fallback, and ``--retry-interrupted`` re-queues attempts whose worker stopped
without any receipt (infrastructure interruptions only).

Completed evidence is reused. Every new attempt keeps the original task and
recipe, including constructor checks, checkpoints and timeout. Gate-only tier
barriers do not prevent diagnostic attempts. Raw artifacts stay outside Git.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .contracts import atomic_json, atomic_text, file_hash, file_lock, read_json, stable_hash
from .queue import device_slots, process_identity, lease_held
from .views import COUNTED_OR_HELD, task_execution_fingerprint, task_evaluation_fingerprint

REPORT = Path("reports/forge/attempt-coverage.json")
WORKER = Path(__file__).with_name("attempt_worker.py")
DONE = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "ERROR", "TIMEOUT", "BLOCKED", "INTERRUPTED"}
INTAKES = ("selected", "policy-cloud")


def identity(source, revision, task, execution, evaluation):
    return stable_hash([source, revision, task, execution, evaluation])


def selected_tasks(publication):
    """Every test a selected recipe's views measure, counted or not.

    Holding a test or removing it from the count changes what is counted, not
    what runs, so held and removal-tagged assignments stay in the universe.
    """
    result = []
    for family in publication["family_progress"]["families"]:
        current = next((c for c in family["cohorts"] if c["anchor"] == family.get("current_cohort_anchor")), None)
        if current is None:
            continue
        row = publication["rows"][current["row_index"]]
        names = {v["id"] for v in current["views"]}
        tasks = sorted({a["task"] for v in publication["family_progress"]["views"] if v["id"] in names
                        for a in v["assignments"] if a["importance"] in COUNTED_OR_HELD or "removal" in a})
        result.append((family["id"], row, tasks))
    return result


def retained_attempts(root):
    """Read the already validated, source-bound ordinary/diagnostic receipts."""
    from .research_tiers import coverage_report
    report = coverage_report(root)  # Verifies its pinned original inputs.
    result = {}
    preflight = read_json(root / "reports/forge/tier-coverage-20261014/preflight.json")
    bindings = {f["family"]: f for f in preflight["families"]}
    for family in report["coverage"]:
        pin = bindings[family["family"]]
        for task, value in family["tasks"].items():
            if value["status"] not in {"PASS", "FAIL"}:
                continue
            contract = pin["tasks"][task]
            key = identity(pin["source_digest"], pin["candidate_revision"], task,
                           contract["execution_sha256"], contract["evaluation_sha256"])
            result[key] = {"status": value["status"], "attempted": True, "origin": "retained_receipt",
                           "evidence": "reports/forge/tier-coverage-20261014/results.json",
                           "original_attempt": value["attempt_id"], "diagnostic": value["scope"] != "ordinary"}
    for record in read_json(root / "reports/forge/actual-missing-diagnostics-20261014/results.json")["records"]:
        if record["lane"] != "ordinary_diagnostics":
            continue
        key = identity(record["source_digest"], record["candidate_revision"], record["task"],
                       record["task_execution_sha256"], record["task_evaluation_sha256"])
        result[key] = {"status": "BLOCKED" if record["status"] == "NOT_RUN" else record["status"],
                       "block_kind": "prerequisite" if record["status"] == "NOT_RUN" else None,
                       "attempted": True, "origin": "retained_attempt", "reason": record["reason"],
                       "evidence": "reports/forge/actual-missing-diagnostics-20261014/results.json",
                       "original_attempt": record["raw_receipt"], "diagnostic": True}
    audit = read_json(root / "reports/forge/tier-coverage-20261014/clockfree-receipt.json")
    pin = bindings["bcap-pure"]; contract = pin["tasks"]["clockfree_audit"]
    key = identity(pin["source_digest"], pin["candidate_revision"], "clockfree_audit",
                   contract["execution_sha256"], contract["evaluation_sha256"])
    result[key] = {"status": "PASS", "attempted": True, "origin": "retained_receipt", "diagnostic": True,
                   "evidence": "reports/forge/tier-coverage-20261014/clockfree-receipt.json",
                   "original_attempt": audit.get("attempt_id")}
    return result


def _intake(previous, requested):
    """A state directory belongs to exactly one intake for its whole life."""
    recorded = previous.get("intake") or ("selected" if previous.get("jobs") else None)
    intake = requested or recorded or "selected"
    if intake not in INTAKES:
        raise ValueError(f"unknown attempt intake {intake!r}")
    if recorded is not None and recorded != intake:
        raise ValueError(f"state directory belongs to the {recorded} intake, not {intake}")
    return intake


def prepare_policy_cloud(root, source, directory, previous):
    """Plan only the scoped policy-cloud cohort; no other intake is consulted."""
    from .policy_cloud_attempts import prepare as prepare_cohort
    jobs = deepcopy(previous["jobs"])
    for key, job in prepare_cohort(root, source, directory).items():
        old = previous["jobs"].get(key)
        if old:
            job.update({k: v for k, v in old.items() if k not in job or k == "status"})
        jobs[key] = job
    state = {"schema_version": 1, "intake": "policy-cloud", "jobs": jobs, "maximum_tests": 4,
             "maximum_per_gpu": 2, "include_research": False}
    atomic_json(directory / "state.json", state)
    publish(root, state, directory)
    return state


def prepare(root, source, directory, *, include_research=False, intake=None):
    directory.mkdir(parents=True, exist_ok=True)
    previous = read_json(directory / "state.json") if (directory / "state.json").exists() else {"jobs": {}}
    if _intake(previous, intake) == "policy-cloud":
        if include_research:
            raise ValueError("the policy-cloud intake cannot include research variants")
        return prepare_policy_cloud(root, source, directory, previous)
    publication = read_json(root / "reports/forge/technique-inventory.json")
    retained = retained_attempts(root)
    pins = {f["family"]: f for f in read_json(root / "reports/forge/tier-coverage-20261014/preflight.json")["families"]}
    directory.mkdir(parents=True, exist_ok=True)
    previous = read_json(directory / "state.json") if (directory / "state.json").exists() else {"jobs": {}}
    jobs = deepcopy(previous["jobs"])  # Retain attempts against earlier task definitions.
    selected = selected_tasks(publication)
    source_map_path = directory / "source-roots.json"
    source_map = read_json(source_map_path) if source_map_path.is_file() else {}
    missing = [{"candidate_id": selection["candidate_id"], "tasks": tasks,
                "source_root": source_map.get(family, str(source)),
                "output": str(directory / "requests" / f"{family}.json")}
               for family, selection, tasks in selected
               if not (directory / "requests" / f"{family}.json").exists()
               or not read_json(directory / "requests" / f"{family}.json").get("source", {}).get("snapshot_path")]
    if missing:
        for source_path in sorted({item["source_root"] for item in missing}):
            spec = directory / "requests" / (stable_hash(source_path)[:12] + "-prepare.json")
            atomic_json(spec, {"source_root": source_path, "requests": [item for item in missing if item["source_root"] == source_path]})
            subprocess.run([sys.executable, str(WORKER), "prepare", str(spec)], check=True,
                           stdout=subprocess.DEVNULL, timeout=600)
    for family, selection, tasks in selected:
        request_path = directory / "requests" / f"{family}.json"
        request = read_json(request_path)
        if (request["source"]["digest"] != selection["bindings"]["source_digest"]
                or request["candidate_revision"] != selection["candidate_revision"]):
            raise ValueError(f"{family}: frozen source/recipe does not match the selected row")
        for task in tasks:
            card = request["tasks"][task]
            execution, evaluation = task_execution_fingerprint(card), task_evaluation_fingerprint(card)
            binding = selection["bindings"]["task_contracts"].get(task)
            pin = (pins[family]["tasks"][task] if family in pins else publication["task_contracts"].get(binding))
            if pin and (execution != pin["execution_sha256"] or evaluation != pin["evaluation_sha256"]):
                raise ValueError(f"{family}/{task}: task differs from the selected frozen contract")
            key = identity(request["source"]["digest"], request["candidate_revision"], task, execution, evaluation)
            job = {"key": key, "family": family, "candidate_id": selection["candidate_id"],
                   "candidate_revision": request["candidate_revision"], "source_digest": request["source"]["digest"],
                   "task": task, "execution_sha256": execution, "evaluation_sha256": evaluation,
                   "request_path": str(request_path), "source_root": request["source"]["snapshot_path"],
                   "request_sha256": file_hash(request_path),
                   "timeout_seconds": card["resources"]["timeout_seconds"], "resources": card["resources"],
                   "dependencies": card.get("dependencies", []), "status": "PENDING"}
            old = previous["jobs"].get(key)
            if old:
                job.update({k: v for k, v in old.items() if k not in job or k == "status"})
            elif key in retained:
                job.update(retained[key])
            else:
                recorded = next((r for r in selection["tasks"] + selection.get("nonrequired_tasks", [])
                                 if r["task_id"] == task and r["status"] in {"PASS", "FAIL"}), None)
                if recorded:
                    job.update(status=recorded["status"], attempted=True, origin="retained_selected_row",
                               diagnostic=False, evidence="reports/forge/technique-inventory.json")
            jobs[key] = job
    include_research = include_research or previous.get("include_research", False)
    if include_research:
        from .research_attempts import prepare as prepare_research
        for key, job in prepare_research(root, directory).items():
            old = previous["jobs"].get(key)
            if old:
                job.update({k: v for k, v in old.items() if k not in job or k == "status"})
            jobs[key] = job
    from .remaining_attempts import prepare as prepare_remaining
    from .checkpoint_recovery import prepare as prepare_repairs
    additions = {**prepare_remaining(root, directory), **prepare_repairs(root, directory)}
    for key, job in additions.items():
        old = previous["jobs"].get(key)
        if old:
            job.update({k: v for k, v in old.items() if k not in job or k == "status"})
        jobs[key] = job
    state = {"schema_version": 1, "jobs": jobs, "maximum_tests": 4, "maximum_per_gpu": 2,
             "include_research": include_research}
    atomic_json(directory / "state.json", state)
    publish(root, state, directory)
    return state


def summary(state):
    counts = Counter(j["status"] for j in state["jobs"].values())
    return {"total": len(state["jobs"]), "attempted": sum(j.get("attempted", False) for j in state["jobs"].values()),
            "pending": counts.get("PENDING", 0), "running": counts.get("RUNNING", 0), "counts": dict(sorted(counts.items()))}


RECORD_FIELDS = ("key", "family", "candidate_id", "candidate_revision", "source_digest", "task",
                 "execution_sha256", "evaluation_sha256", "status", "attempted", "block_kind", "reason",
                 "origin", "diagnostic", "evidence", "original_attempt", "attempt_id", "updates", "seconds",
                 "result_sha256", "log_sha256", "runtime", "timeout_seconds", "setup_retries", "research_scope", "label",
                 "training_gif_sha256", "media_error", "media_note", "grade", "policy_scope", "parent_task", "retry_history",
                 "saved_observations", "repair_scope", "owning_family", "reporting_tier", "diagnostic_continuation",
                 "repair_origin_source", "required_gpu", "repair_superseded", "reused_execution",
                 "cohort_scope", "dependency_note", "exclusive_gpu", "cohort_id", "measurement_law",
                 "shared_scopes", "shared_reason")


def coverage_value(state, *, evidence=REPORT.as_posix()):
    """The published record format, shared by the repository and cohort runs."""
    value = {"schema": "forge_attempt_coverage_v1", "qualification_input": False, "qualification_reuse": False,
             "formulations_modified": False, "maximum_tests": 4, "maximum_per_gpu": 2,
             "summary": summary(state), "records": [{k: j[k] for k in RECORD_FIELDS if k in j}
                        for j in sorted(state["jobs"].values(), key=lambda j: (j["family"], j["task"]))]}
    for record in value["records"]:
        if record.get("origin") == "new_attempt":
            record["original_attempt"] = record["evidence"]
            record["evidence"] = evidence
    return value


def publish_records(path, state):
    """Write a cohort run's records beside its state; no repository report."""
    from .policy_cloud_publication import README
    value = {**coverage_value(state, evidence=README.as_posix()), "intake": state["intake"],
             "publication": "state_directory_only"}
    if not path.exists() or read_json(path) != value:
        atomic_json(path, value)
    return value


def publish(root, state, directory=None):
    if state.get("intake") == "policy-cloud":
        # Scoped cohort records stay with their run until the explicit cohort
        # publication (policy_cloud_publication) writes its own report directory;
        # the repository attempt report is never rewritten here.
        if directory is None:
            raise ValueError("policy-cloud records publish only inside their state directory")
        return publish_records(Path(directory) / "attempt-coverage.json", state)
    if any(job.get("cohort_scope") for job in state["jobs"].values()):
        raise ValueError("scoped cohort records never enter the repository attempt report")
    value = coverage_value(state)
    path = root / REPORT
    if not path.exists() or read_json(path) != value:
        atomic_json(path, value)
        lines = ["# Selected-family and non-seed research run-attempt coverage", "",
            "Every record identifies an attempt to run an unchanged selected or archived research recipe and task. "
            "PASS/FAIL diagnostics remain separate from ordinary qualification. BLOCKED means an actual "
            "start was refused by its constructor or required checkpoint; it is not a numerical failure.", "",
            "| Family or exact recipe | Attempted / distinct tests | Pending | Outcome counts |",
            "| --- | ---: | ---: | --- |"]
        for family in sorted({j["family"] for j in state["jobs"].values()}):
            jobs = [j for j in state["jobs"].values() if j["family"] == family]
            counts = Counter(j["status"] for j in jobs)
            label = jobs[0].get("label", family)
            lines.append(f"| {label} | {sum(j.get('attempted', False) for j in jobs)}/{len(jobs)} | "
                         f"{counts.get('PENDING', 0)} | " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + " |")
        lines += ["", "[Exact per-task attempts](attempt-coverage.json) · "
                  "[Main technique leaderboard](technique-inventory.md)", "",
                  "The runner reuses matched source/recipe/task receipts, resumes pending work, and never "
                  "replays a completed attempt automatically. It uses at most four workers, two per GPU. "
                  "Each child retains the task's original timeout. Raw logs and checkpoints are stored "
                  "outside Git in the configured state directory.", "",
                  "```sh", "python -m experiments.forge.attempt_coverage plan --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --include-research",
                  "python -m experiments.forge.attempt_coverage run --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --include-research --refresh",
                  "python -m experiments.forge.attempt_coverage status --state-root RUN_DIRECTORY", "```", ""]
        lines += ["For a durable user service, adapt "
                  "[the systemd template](../../ops/forge-attempt-coverage.service.example), then run "
                  "`systemctl --user enable --now particlegan-attempt-fill.service`. "
                  "Follow its starts and completions with "
                  "`journalctl --user -u particlegan-attempt-fill.service -f` or "
                  "`tail -f RUN_DIRECTORY/events.jsonl`. Each event names the family, task and attempt directory ID.", "",
                  "`--watch` rescans the local selected rows every 30 seconds after draining. "
                  "`--refresh` regenerates the local main leaderboard during work and when the queue drains. "
                  "An optional `--completion-hook ops/publish_attempt_coverage.py` publishes the completed batch "
                  "through a generated-report-only commit and a checked GitHub PR on the dedicated branch. "
                  "For multiple selected source cohorts, `RUN_DIRECTORY/source-roots.json` maps family IDs "
                  "to their frozen checkouts; every source digest and candidate revision is checked. "
                  "A changed source or recipe requires its matching frozen checkout; "
                  "a mismatched source stops preparation. Interrupted attempts are retained for review, not replayed. "
                  "The research intake includes the 19 seed-zero variants already shown in the leaderboard; "
                  "historical seed-only repetitions are excluded. Original pinned requests supply each recipe and "
                  "snapshot, and original source task declarations supply the missing questions. A mismatch with "
                  "a frozen evaluator source blocks the attempt. Completed new same-source measurements fill "
                  "previously empty research cells; archived verdicts and selected-family qualification never change. "
                  "Checkpoint/data dependencies still require passing certified artifacts; gate-only failures are "
                  "bypassed solely for this authorized diagnostic lane. No new qualification is granted.", ""]
        remaining = [j for j in state["jobs"].values() if j.get("policy_scope") or (j.get("retry_history") and not j.get("repair_scope"))
                     or j.get("label", "").endswith(" current unbound task")]
        if remaining:
            lines += ['<a name="remaining-attempts"></a>', "", "## Follow-up attempts of remaining cells", "",
                      "These cover the three previously empty Atlas policy cells, three current BISPF continuation "
                      "declarations, and one unchanged retry after a GPU-memory failure. Earlier receipts remain intact. "
                      "The 324 missing pairs in historical seed-only studies are excluded from this backfill.", "",
                      "| Family / recipe | Test | Outcome | Updates / saved observations | Cause |",
                      "| --- | --- | --- | --- | --- |"]
            for job in sorted(remaining, key=lambda j: (j["family"], j["task"])):
                saved = job.get("saved_observations", {})
                updates = saved.get("completed_updates", job.get("updates"))
                observed = saved.get("count", "—")
                reason = str(job.get("reason", "Pending attempt.")).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {job.get('label', job['family'])} | {job.get('parent_task', job['task'])} | "
                             f"{job['status']} | {updates if updates is not None else '—'} / {observed} | {reason} |")
            lines += ["", "Export errors retain their saved numerical observations and actual-training GIFs in the "
                      "local attempt directories. Those observations do not replace the recorded gate. "
                      "The original BISPF attempts were blocked by failed parent gates; the checkpoint recovery section "
                      "below rebuilds their states and runs explicitly nonqualifying continuations past those gates. "
                      "Original timeouts and incompatible formulations are not changed to obtain a score.", ""]
        repairs = [j for j in state["jobs"].values() if j.get("repair_scope")]
        if repairs:
            lines += ['<a name="checkpoint-recovery"></a>', "", "## Checkpoint export repair and BISPF rebuilds", "",
                      "The checkpoint hasher now preserves diagnostic NaN/Inf values while retaining every finite-state hash. "
                      "Atlas/E22 rerun their unchanged Gaussian and Ring16 questions with the export fix. "
                      "BISPF rebuilds each original 7k producer and continues its exact own state to 14k. "
                      "A failed parent quality gate does not stop these user-authorized diagnostics. "
                      "Recipes, seeds, initialization, prior laws, numerical gates and original budgets remain fixed; "
                      "original failures stay recorded and these repaired-source diagnostics grant no qualification.", "",
                      "| Family | Recovery run | Test | Outcome | Completed updates | Cause |",
                      "| --- | --- | --- | --- | ---: | --- |"]
            for job in sorted(repairs, key=lambda j: (j["owning_family"], j["reporting_tier"], j["task"])):
                reason = str(job.get("reason", "Queued" if job["status"] == "PENDING" else "Running")).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {job['owning_family']} | {job['label']} | {job['task']} | {job['status']} | "
                             f"{job.get('updates') if job.get('updates') is not None else '—'} | {reason} |")
            lines += ["", "Continuation checks still require a complete own checkpoint, matching recipe and RNG identity, "
                      "unchanged 7k scheduling horizon, exact prefix restoration and the full 14k update count. "
                      "Ordinary qualification rejects the explicitly marked diagnostic continuation receipts.", ""]
        reused = [j for j in state["jobs"].values() if j.get("origin") == "reused_execution"]
        if reused:
            lines += ["## Results recovered from existing uninterrupted runs", "",
                      "These task results reuse their producer's already measured observations. No new training "
                      "or sampling was performed. Original refused-start receipts remain linked in each certificate; "
                      "ordinary qualification is unchanged.", "",
                      "| Family | Test | Result | Passing extension checks | Evidence |",
                      "| --- | --- | --- | ---: | --- |"]
            for job in sorted(reused, key=lambda j: (j["family"], j["task"])):
                metrics = job["grade"]["metrics"]
                evidence = job["evidence"].removeprefix("reports/forge/")
                lines.append(f"| {job['family']} | {job['task']} | {job['status']} | "
                             f"{metrics['extension_passing']}/{metrics['extension_checks']} | "
                             f"[Verified saved run]({evidence}) |")
            lines.append("")
        blocked = [j for j in state["jobs"].values() if j.get("origin") == "new_attempt"
                   and j["status"] in {"BLOCKED", "ERROR", "TIMEOUT", "INCOMPLETE", "INVALID"}]
        if blocked:
            lines += ["## Attempted blockers and incomplete runs", "",
                      "These are recorded start attempts, not missing runs. A source or formulation change is not applied to make them pass.", "",
                      "| Recipe | Task | Outcome | Reason |", "| --- | --- | --- | --- |"]
            for job in sorted(blocked, key=lambda j: (j.get("label", j["family"]), j["task"])):
                reason = str(job.get("reason", "See exact attempt receipt.")).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {job.get('label', job['family'])} | {job['task']} | {job['status']} | {reason} |")
            lines.append("")
        atomic_text(root / "reports/forge/ATTEMPT_COVERAGE.md", "\n".join(lines))


def dependency_blockers(job, state):
    result = []
    for dep in job["dependencies"]:
        if not isinstance(dep, dict) or dep.get("kind") == "gate":
            continue  # Authorized diagnostic attempts may follow a failed gate.
        parent = next((j for j in state["jobs"].values()
                       if j["family"] == job["family"] and j["task"] == dep["task"]), None)
        allowed = {"PASS", "FAIL"} if job.get("diagnostic_continuation") else {"PASS"}
        if parent is None or parent["status"] not in allowed:
            result.append(f"{dep['kind']} parent {dep['task']}: {parent['status'] if parent else 'unavailable'}")
        elif dep["task"] not in prerequisite_bindings(job, state):
            # A passing scalar stamp is not a usable checkpoint. Never fabricate
            # an artifact binding or restart a continuation from initialization.
            result.append(f"{dep['kind']} parent {dep['task']}: certified artifact binding unavailable to this attempt")
    return result


def prerequisite_bindings(job, state):
    bindings = {}
    for dep in job["dependencies"]:
        if not isinstance(dep, dict) or dep.get("kind") == "gate":
            continue
        allowed = {"PASS", "FAIL"} if job.get("diagnostic_continuation") else {"PASS"}
        parent = next((j for j in state["jobs"].values()
                       if j["family"] == job["family"] and j["task"] == dep["task"] and j["status"] in allowed), None)
        if parent is None:
            continue
        if parent.get("certified_parent"):
            evidence = parent["certified_parent"]["result"].get("evidence", {})
            if evidence.get("checkpoint") and evidence.get("artifact_manifest"):
                bindings[dep["task"]] = parent["certified_parent"]
        elif parent.get("origin") == "new_attempt" and (Path(parent["directory"]) / "raw.json").is_file():
            raw = read_json(Path(parent["directory"]) / "raw.json")
            if not raw.get("evidence", {}).get("checkpoint") or not raw.get("evidence", {}).get("artifact_manifest"):
                continue
            request = read_json(job["request_path"])
            expected = next(j["compatibility_key"] for j in request["jobs"]
                            if dep["task"] in j.get("task_ids", [j["task_id"]]))
            bindings[dep["task"]] = dict(attempt_id=parent["attempt_id"], result_hash=stable_hash(raw),
                result={**raw, "task_id": dep["task"], "compatibility_key": expected, "gate_status": parent["status"]},
                candidate_revision=parent["candidate_revision"],
                compatibility_key=expected)
    return bindings


def recover(state):
    for job in state["jobs"].values():
        if job["status"] != "RUNNING":
            continue
        directory = Path(job["directory"])
        terminal = directory / "result.json"
        if terminal.exists() and not lease_held(directory / "execution.lock"):
            job.update(read_json(terminal), origin="new_attempt", diagnostic=True,
                       evidence=str(terminal), status=read_json(terminal)["status"],
                       result_sha256=file_hash(terminal), log_sha256=file_hash(directory / "run.log"))
        elif (not lease_held(directory / "execution.lock") and
              (not process_identity(job.get("pid", -1)) or
               process_identity(job.get("pid", -1)) != job.get("process_identity"))):
            job.update(status="INTERRUPTED", attempted=True, diagnostic=True,
                       reason="prior worker stopped without a terminal receipt; retained without automatic replay")


def reconcile_grouped(root, directory, state):
    """Recover completed grouped measurements, without granting restart eligibility."""
    recovered = []
    for child in state["jobs"].values():
        if not (child["status"] == "PENDING" or
                child["status"] == "BLOCKED" and child.get("block_kind") == "prerequisite"):
            continue
        if not child.get("request_path") or not str(child.get("task", "")).startswith("ring_extension"):
            continue
        request = read_json(Path(child["request_path"]))
        card = request["tasks"][child["task"]]
        # Scoped variants keep the evaluator kind; their ids carry a cohort suffix.
        if (card["adapter"] != "ring_endurance" or card["evaluation"].get("kind") != "ring_extension"
                or not card["execution"].get("execution_group")):
            continue
        parents = [job for job in state["jobs"].values()
                   if job["task"] == card["execution"].get("continuation_of")
                   and all(job.get(key) == child.get(key) for key in
                           ("family", "candidate_revision", "source_digest", "request_sha256"))
                   and job["status"] == "PASS" and job.get("origin") == "new_attempt"]
        if len(parents) != 1:
            continue
        parent = parents[0]
        reuse_id = stable_hash([child["key"], parent["result_sha256"], child["request_sha256"],
                               file_hash(Path(__file__).with_name("grouped_evidence.py"))])
        target = directory / "reused-groups" / reuse_id
        spec = {"source_root": parent["source_root"], "parent": parent, "child": child}
        atomic_json(target / "request.json", spec)
        subprocess.run([sys.executable, str(WORKER), "reuse_group", str(target / "request.json")],
                       check=True, timeout=60, stdout=subprocess.DEVNULL)
        certificate = read_json(target / "result.json")
        previous = {key: child[key] for key in ("status", "attempt_id", "result_sha256", "reason", "evidence")
                    if key in child}
        certificate["previous_attempt"] = previous
        if state.get("intake") == "policy-cloud":
            # Scoped cohort certificates stay in their run until publication.
            evidence_file = directory / "grouped-evidence" / (reuse_id + ".json")
            evidence_value = str(evidence_file)
        else:
            evidence_path = Path("reports/forge/grouped-evidence") / (reuse_id + ".json")
            evidence_file, evidence_value = root / evidence_path, evidence_path.as_posix()
        atomic_json(evidence_file, certificate)
        for key in ("block_kind", "pid", "process_identity", "gpu", "slot", "directory", "log_sha256"):
            child.pop(key, None)
        child.update(status=certificate["grade"]["status"], grade=certificate["grade"],
                     reason="verified observations from the same uninterrupted producer; no new execution",
                     origin="reused_execution", diagnostic=True, attempted=True,
                     evidence=evidence_value, original_attempt=certificate["original_attempt"],
                     attempt_id="reuse_" + reuse_id[:32], result_sha256=file_hash(evidence_file),
                     updates=0, seconds=0, runtime=certificate["runtime"],
                     reused_execution={"producer_attempt_id": parent["attempt_id"],
                         "producer_updates": certificate["producer_updates"], "previous_attempt": previous,
                         "optimizer_updates_added": 0, "sampling_draws_added": 0})
        recovered.append(child["key"])
    return recovered


# Attempt-specific fields; planning fields (request, task, scope, law,
# dependencies, resources) are never touched by a retry.
ATTEMPT_FIELDS = ("attempt_id", "directory", "pid", "process_identity", "gpu", "slot", "announced", "attempted",
                  "reason", "block_kind", "origin", "evidence", "original_attempt", "result_sha256", "log_sha256",
                  "updates", "seconds", "runtime", "grade", "raw_sha256", "training_gif_sha256", "media_error",
                  "media_note")


def requeue_interrupted(state):
    """Re-queue infrastructure-interrupted attempts once, on explicit request.

    Only INTERRUPTED records qualify: their supervisor stopped without writing
    any terminal receipt (a crash, reboot or kill), so no outcome exists.
    PASS, FAIL, TIMEOUT, ERROR, INCOMPLETE, INVALID and ordinary BLOCKED
    records are never retried. A dependent that was refused (BLOCKED,
    prerequisite) because a producer it waits on was interrupted is re-queued
    with that producer. Every prior record is kept in ``retry_history``; the
    frozen request, task, recipe and seed are unchanged.
    """
    if state.get("intake") != "policy-cloud":
        raise ValueError("re-running interrupted attempts is available only to the policy-cloud intake")
    requeued, reasons = [], {}
    for job in state["jobs"].values():
        if job["status"] != "INTERRUPTED":
            continue
        directory = Path(job["directory"])
        if (directory / "result.json").exists() or lease_held(directory / "execution.lock"):
            raise ValueError(f"{job['family']}/{job['task']}: interrupted attempt has a receipt or a live worker")
        reasons[job["key"]] = ("explicit --retry-interrupted: the worker stopped without a terminal receipt "
                               "(infrastructure interruption); same frozen request, task, recipe and seed")
    changed = True
    while changed:  # Dependents of re-queued producers, transitively.
        changed = False
        for job in state["jobs"].values():
            if job["key"] in reasons or job["status"] != "BLOCKED" or job.get("block_kind") != "prerequisite":
                continue
            producers = [parent for dep in job["dependencies"] if isinstance(dep, dict) and dep.get("kind") != "gate"
                         for parent in state["jobs"].values()
                         if parent["family"] == job["family"] and parent["task"] == dep["task"] and parent["key"] in reasons]
            if producers:
                reasons[job["key"]] = ("re-queued with its interrupted producer " + producers[0]["task"] +
                                       "; same frozen request, task, recipe and seed")
                changed = True
    for key, reason in reasons.items():
        job = state["jobs"][key]
        entry = {field: job[field] for field in ("attempt_id", "status", "block_kind", "reason", "gpu", "slot",
                                                 "directory", "result_sha256", "log_sha256", "seconds", "updates")
                 if job.get(field) is not None}
        log = Path(job["directory"]) / "run.log" if job.get("directory") else None
        if "log_sha256" not in entry and log is not None and log.is_file():
            entry["log_sha256"] = file_hash(log)
        entry["retry_reason"] = reason
        for field in ATTEMPT_FIELDS:
            job.pop(field, None)
        job["retry_history"] = [*job.get("retry_history", []), entry]
        job["status"] = "PENDING"
        requeued.append(job)
    return requeued


def _slot_allowed(job, slot, active, pending_exclusive):
    """Short fixed task limits run alone; reserve their GPU until they start."""
    device = str(slot["device"])
    occupied = {str(j["gpu"]) for j in active}
    exclusive = {str(j["gpu"]) for j in active if j.get("exclusive_gpu")}
    if job.get("exclusive_gpu"):
        return device not in occupied
    return device not in exclusive and device not in pending_exclusive


def critical_path_seconds(state):
    """Each job's own time limit plus its longest chain of same-family dependents.

    The policy-cloud intake dispatches the longest chains first (for example the
    7k native parents before short tests), so continuations are not the tail.
    Only the dispatch order changes; requests, limits and results do not."""
    children = {}
    for job in state["jobs"].values():
        for dep in job["dependencies"]:
            if isinstance(dep, dict) and dep.get("kind") != "gate":
                children.setdefault((job["family"], dep["task"]), []).append(job)
    memo = {}

    def chain(job):
        key = (job["family"], job["task"])
        if key not in memo:
            memo[key] = 0  # Guards a malformed cycle.
            memo[key] = int(job.get("timeout_seconds") or 0) + max(
                (chain(child) for child in children.get(key, [])), default=0)
        return memo[key]

    return {key: chain(job) for key, job in state["jobs"].items()}


def drain(root, directory, *, gpus, source=None, watch=False, refresh=False, completion_hook=None,
          retry_interrupted=False, wallclock_policy="default"):
    initial = read_json(directory / "state.json")
    # Operational wall-clock limits (experiments/forge/wallclock.py DEFAULT_POLICY) apply
    # by default to the selected intake. The scoped policy-cloud intake keeps its own
    # declared fixed limits; pass "declared" to restore declared limits everywhere.
    if initial.get("intake") == "policy-cloud":
        wallclock_policy = None
    elif wallclock_policy is not None and wallclock_policy != "declared":
        from .wallclock import resolve_policy
        wallclock_policy = resolve_policy(wallclock_policy)
    if initial.get("intake") == "policy-cloud":
        # The cohort intake may also drain on one GPU (two tests at a time).
        if not 1 <= len(gpus) <= 2 or len(set(gpus)) != len(gpus):
            raise ValueError("select one or two distinct GPUs; at most 2 tests run per GPU")
        pinned = {str(job["required_gpu"]) for job in initial["jobs"].values()
                  if job.get("required_gpu") is not None and job["status"] in {"PENDING", "INTERRUPTED"}}
        if pinned - {str(gpu) for gpu in gpus}:
            raise ValueError("pending short fixed-limit tests are pinned to GPU " + ",".join(sorted(pinned)) +
                             "; include it in --gpus")
        # A diagnostic continuation restores its producer's state on the same GPU.
        producers = {str(parent["gpu"]) for job in initial["jobs"].values()
                     if job["status"] in {"PENDING", "INTERRUPTED"} and job.get("diagnostic_continuation")
                     for dep in job["dependencies"] if isinstance(dep, dict) and dep.get("kind") == "checkpoint"
                     for parent in initial["jobs"].values()
                     if parent["family"] == job["family"] and parent["task"] == dep["task"] and "gpu" in parent}
        if producers - {str(gpu) for gpu in gpus}:
            raise ValueError("pending diagnostic continuations must run on their producer's GPU " +
                             ",".join(sorted(producers)) + "; include it in --gpus")
        if refresh or completion_hook is not None:
            raise ValueError("policy-cloud runs never refresh or publish the repository leaderboard")
    elif len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("select exactly two distinct GPUs; the fixed limit is 4 tests / 2 per GPU")
    if retry_interrupted and initial.get("intake") != "policy-cloud":
        raise ValueError("re-running interrupted attempts is available only to the policy-cloud intake")
    processes = {}
    refreshed = None
    refresh_at = 0.0
    published = None
    with file_lock(directory / "coordinator.lock", blocking=False):
        try:
            while True:
                state = read_json(directory / "state.json")
                recover(state)
                if retry_interrupted:
                    # Once per invocation, after stopped workers are recognized.
                    for job in requeue_interrupted(state):
                        event(directory, "requeued", {**job, "reason": job["retry_history"][-1]["retry_reason"]})
                    retry_interrupted = False
                reconcile_grouped(root, directory, state)
                for job in state["jobs"].values():
                    if job["status"] in DONE and job.get("origin") == "new_attempt" and not job.get("announced"):
                        event(directory, "finished", job); job["announced"] = True
                slots = [s for s in device_slots(gpus, 2, allow_sharing=True) if s["device"] != "cpu"]
                active = [j for j in state["jobs"].values() if j["status"] == "RUNNING"]
                busy = {(j["gpu"], j["slot"]) for j in active}
                if state.get("intake") == "policy-cloud":
                    critical = critical_path_seconds(state)
                    pending = [j for _, j in sorted(
                        ((k, j) for k, j in state["jobs"].items() if j["status"] == "PENDING"),
                        key=lambda kj: (-critical[kj[0]], len(kj[1]["dependencies"]), kj[1]["family"], kj[1]["task"]))]
                else:
                    pending = sorted((j for j in state["jobs"].values() if j["status"] == "PENDING"),
                                     key=lambda j: (len(j["dependencies"]), j["family"], j["task"]))
                def waiting(job):
                    return any(parent["status"] in {"PENDING", "RUNNING"}
                               for dep in job["dependencies"] if isinstance(dep, dict) and dep.get("kind") != "gate"
                               for parent in state["jobs"].values()
                               if parent["family"] == job["family"] and parent["task"] == dep["task"])
                pending_exclusive = {str(j["required_gpu"]) for j in pending
                                     if j.get("exclusive_gpu") and j.get("required_gpu") is not None and not waiting(j)}
                for job in pending:
                    if len(active) >= 4:
                        break
                    if waiting(job):
                        continue
                    parent_gpus = {str(parent["gpu"]) for dep in job["dependencies"]
                                   if job.get("diagnostic_continuation") and isinstance(dep, dict) and dep.get("kind") == "checkpoint"
                                   for parent in state["jobs"].values()
                                   if parent["family"] == job["family"] and parent["task"] == dep["task"] and "gpu" in parent}
                    slot = next((s for s in slots if (job.get("required_gpu") is None or str(s["device"]) == str(job["required_gpu"]))
                                 and (not parent_gpus or str(s["device"]) in parent_gpus)
                                 and (s["device"], s["slot"]) not in busy
                                 and _slot_allowed(job, s, active, pending_exclusive)
                                 and s["memory_mb"] >= job["resources"]["gpu_memory_mb"]), None)
                    if slot is None:
                        continue
                    attempt = uuid.uuid4().hex
                    target = directory / "attempts" / attempt
                    target.mkdir(parents=True)
                    spec = {**job, "dependency_blockers": dependency_blockers(job, state),
                            "prerequisites": prerequisite_bindings(job, state),
                            "device": f"cuda:{slot['device']}" if job["resources"].get("gpus", 1) else "cpu"}
                    if wallclock_policy and wallclock_policy != "declared":
                        spec["wallclock_policy"] = wallclock_policy
                    atomic_json(target / "request.json", spec)
                    log = (target / "run.log").open("w")
                    child = subprocess.Popen([sys.executable, "-u", str(WORKER), "supervise", str(target / "request.json")],
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    log.close(); processes[child.pid] = child
                    job.update(status="RUNNING", attempted=True, attempt_id=attempt, directory=str(target),
                               pid=child.pid, process_identity=process_identity(child.pid), gpu=slot["device"], slot=slot["slot"])
                    active.append(job); busy.add((slot["device"], slot["slot"]))
                    event(directory, "started", job)
                atomic_json(directory / "state.json", state); publish(root, state, directory)
                for pid, child in list(processes.items()):
                    if child.poll() is not None:
                        processes.pop(pid)
                fingerprint = stable_hash({k: j["status"] for k, j in state["jobs"].items()})
                drained = not any(j["status"] in {"PENDING", "RUNNING"} for j in state["jobs"].values())
                if refresh and fingerprint != refreshed and (drained or time.monotonic() - refresh_at >= 60):
                    subprocess.run([sys.executable, "reports/forge/regenerate_technique_inventory.py", "--refresh-publication"],
                                   cwd=root, check=True, timeout=300)
                    refreshed = fingerprint; refresh_at = time.monotonic()
                if drained:
                    if completion_hook is not None and fingerprint != published:
                        if not completion_hook.is_file():
                            print(f"Completion publisher is missing: {completion_hook}. "
                                  "Results are retained; restore the publisher before restarting.",
                                  file=sys.stderr, flush=True)
                            raise SystemExit(2)
                        subprocess.run([sys.executable, str(completion_hook)], cwd=root, check=True, timeout=600,
                                       env={**os.environ, "FORGE_ATTEMPT_STATE_ROOT": str(directory)})
                        published = fingerprint
                    if not watch:
                        return summary(state)
                    time.sleep(30)
                    if source is not None:
                        prepare(root, source, directory)
                else:
                    time.sleep(.25)
        finally:
            # Independent supervisors keep their original time limits and durable
            # receipts. A resumed coordinator attaches instead of duplicating them.
            pass


def event(directory, action, job):
    value = {"time": time.time(), "event": action, "family": job["family"], "task": job["task"],
             "status": job["status"], "attempt": job.get("attempt_id"), "gpu": job.get("gpu"),
             "slot": job.get("slot"), "reason": job.get("reason")}
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status", "reconcile"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--include-research", action="store_true", help="include original seed-zero research variants")
    parser.add_argument("--refresh", action="store_true", help="refresh the ordinary leaderboard after draining")
    parser.add_argument("--completion-hook", type=Path, help="trusted Python publisher to run after the queue drains")
    parser.add_argument("--intake", choices=INTAKES, help="selected (default) or only the scoped policy-cloud cohorts "
                        "(clean and as-served laws); a state directory keeps the intake it was first prepared with")
    parser.add_argument("--retry-interrupted", action="store_true",
                        help="policy-cloud run only: re-queue INTERRUPTED attempts (worker stopped without any receipt) "
                        "once, with the same frozen request and seed, plus dependents refused only because of them; "
                        "PASS, FAIL, TIMEOUT and ERROR records are never retried and every prior record is kept")
    parser.add_argument("--wallclock-policy", default="default",
                        help="operational wall-clock limits for the selected intake: 'default' (raised task minimums plus "
                             "a contention-aware extension, experiments/forge/wallclock.py), 'declared' (declared limits "
                             "only) or a JSON policy file")
    args = parser.parse_args(); root=args.root.resolve(); directory=args.state_root.resolve()
    if args.action == "status":
        state = read_json(directory / "state.json")
        value = summary(state)
        if state.get("intake") == "policy-cloud":
            from .policy_cloud_attempts import plan_summary
            value = {**value, "intake": "policy-cloud", "jobs": plan_summary(state)}
        print(json.dumps(value, sort_keys=True)); return
    if args.action == "reconcile":
        with file_lock(directory / "coordinator.lock", blocking=False):
            state = read_json(directory / "state.json")
            recovered = reconcile_grouped(root, directory, state)
            atomic_json(directory / "state.json", state)
            publish(root, state, directory)
        print(json.dumps({"recovered": recovered, **summary(state)}, sort_keys=True)); return
    if args.source_root is None:
        parser.error("--source-root is required for plan/run")
    if args.retry_interrupted and args.action != "run":
        parser.error("--retry-interrupted applies to run only")
    # Preparing cannot race an active coordinator or replace its live state.
    with file_lock(directory / "coordinator.lock", blocking=False):
        state = prepare(root, args.source_root.resolve(), directory, include_research=args.include_research,
                        intake=args.intake)
    if args.action == "run":
        value = drain(root, directory, gpus=args.gpus.split(","), source=args.source_root.resolve(),
                      watch=args.watch, refresh=args.refresh, completion_hook=args.completion_hook,
                      retry_interrupted=args.retry_interrupted,
                      wallclock_policy=(args.wallclock_policy if args.wallclock_policy in {"default", "declared"}
                                        else read_json(Path(args.wallclock_policy))))
    else:
        value = summary(state)
        if state.get("intake") == "policy-cloud":
            from .policy_cloud_attempts import plan_summary
            value = {**value, "intake": "policy-cloud", "jobs": plan_summary(state)}
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
