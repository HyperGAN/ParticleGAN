"""Generated navigation over recorded results, never a qualification reducer.

One selected scientific row supplies every view in a runtime group. Views share
task results, so aggregate counts describe view requirements, not unique runs.
Earlier task verdicts retain their contracts and cannot qualify newer tasks.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from .contracts import file_hash, identifier, read_json, stable_hash
from . import family_documentation
from .views import (COUNTED_OR_HELD, is_stretch, stretch_note,
                    task_evaluation_fingerprint, task_execution_fingerprint)


FAMILY_DIRECTORY = Path("reports/forge/families")
PROBLEM_DEFINITIONS = Path("reports/forge/problem-definitions.json")
PROBLEM_PAGE = Path("reports/forge/problems.md")
COMPLETE = {"PASS", "FAIL"}
BASE_TIERS = ("1", "2", "3")
TIERS = BASE_TIERS
COVERAGE_PREFLIGHT = Path("reports/forge/tier-coverage-20261014/preflight.json")


def evaluated_tiers(views) -> tuple[str, ...]:
    """Return tuple of tier strings from 1 to max tier present (at least 3)."""
    highest = max([3] + [a["qualification_tier"] for view in views for a in view.get("assignments", [])])
    return tuple(str(t) for t in range(1, highest + 1))


FIRST_FULL_ATLAS_RESULT = Path("reports/forge/common26-first-two-pole-full-atlas-20261004/results.json")
FIRST_FULL_ATLAS_RESULT_SHA256 = "66551a0018a22cc96ffe88bd27754634ddb5768c5fd896cf6f5dd262e72aa38b"
FULL_ORIGINAL_ATLAS_CONFIG_SHA256 = "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"


# These immutable public reports supply navigation only, never selected-row cells.
ATLAS_EVIDENCE_REPORTS = (
    ("live_clean_common26", "reports/forge/common26-full-original-diagnostic-20261005/results.json",
     "02cee5ca6877a4fd8c8b7bca78bcc245d905b0482fee2626a0c8b1220480b6e5", 922104,
     "pg_common26_full_original_diagnostic_checkpoint_v1"),
    ("native_restoration", "reports/forge/atlas-native-restoration-checkpoint-20261005/results.json",
     "8d1b91ea5326731176edc68df06ed36d38007e62d4ebf704fde1d5bf35e4b698", 34254,
     "pg_atlas_restoration_public_report_v1"),
    ("ae_sourceguard", "reports/forge/atlas-ae-sourceguard-checkpoint-20261005/results.json",
     "e6d93f4a850e5c4afd2ab2c4fa18d5b9dab4b34968b5c89e320fd16626d622a6", 15492,
     "pg_ae633_sourceguard_public_checkpoint_v1"),
)


def _full_original_atlas_first_case(root, load):
    """Navigation over one immutable accepted result; never a selected-row grade."""
    if not (root / FIRST_FULL_ATLAS_RESULT).is_file():
        return None
    if file_hash(root / FIRST_FULL_ATLAS_RESULT) != FIRST_FULL_ATLAS_RESULT_SHA256:
        raise ValueError("retained first Full Atlas result changed")
    record = load(FIRST_FULL_ATLAS_RESULT)
    result = record.get("result", {})
    certificate = result.get("certificate", {})
    grade = certificate.get("grade", {})
    candidate = record.get("candidate", {})
    binding = record.get("binding", {})
    source = record.get("source", {})
    repeat = record.get("protocol", {}).get("scientific_repeat", {})
    convergence = grade.get("evaluator_result", {}).get("convergence", {})
    if (record.get("schema") != "pg_canonical_two_pole_first_case_public_v1"
            or record.get("task_id") != "two_pole" or result.get("task_id") != "two_pole"
            or candidate.get("id") != "atlas-full-original-common26-ember552"
            or candidate.get("trainer_family") != "atlas"
            or binding.get("config_sha256") != FULL_ORIGINAL_ATLAS_CONFIG_SHA256
            or repeat.get("config_sha256") != FULL_ORIGINAL_ATLAS_CONFIG_SHA256
            or repeat.get("source_digest") != source.get("digest")
            or repeat.get("source_origin_commit") != source.get("origin_commit")
            or type(repeat.get("execution_updates")) is not int or repeat["execution_updates"] != 80
            or type(repeat.get("seed")) is not int or repeat["seed"] != 0
            or certificate.get("full_protocol_complete") is not True
            or result.get("status") not in COMPLETE
            or grade.get("status") != result["status"] or grade.get("gate_status") != result["status"]
            or grade.get("evaluator_result", {}).get("status") != result["status"]
            or convergence.get("complete") is not True
            or type(convergence.get("observations")) is not int or convergence["observations"] != 24):
        raise ValueError("retained first Full Atlas result has conflicting scope or completeness")
    checks = grade.get("evaluator_result", {}).get("metrics", [])
    if {check.get("metric") for check in checks} != {"mean_abs", "grad_med"} or len(checks) != 2:
        raise ValueError("retained first Full Atlas result has incomplete metric receipts")
    return {"schema": "pg_full_original_atlas_common26_navigation_v1", "qualification_input": False,
            "configuration_id": candidate["id"], "config_sha256": binding["config_sha256"],
            "readout": FIRST_FULL_ATLAS_RESULT.with_name("README.md").as_posix(),
            "public_result": FIRST_FULL_ATLAS_RESULT.as_posix(), "public_result_sha256": FIRST_FULL_ATLAS_RESULT_SHA256,
            "source": {key: source[key] for key in ("origin_commit", "digest")},
            "first_case": {"task_id": "two_pole", "status": result["status"], "completed_updates": 80,
                           "observations": 24, "seed": 0, "metric_receipts": [
                               {key: deepcopy(check[key]) for key in ("metric", "value", "op", "threshold", "status")}
                               for check in checks]},
            "completed": 1, "required": 26, "remaining_not_run": 25,
            "campaign": None, "campaign_status": "PENDING_ADAPTER_AND_BUDGET",
            "current_selected_configuration_credit": False, "prerequisite_credit": False,
            "default_adoption": False, "speed_ranking": False}


def _atlas_evidence_navigation(root, load):
    """Read separate byte-bound public scopes without importing grades into a row."""
    cohorts = []
    for scope, relative, digest, size, schema in ATLAS_EVIDENCE_REPORTS:
        path = Path(relative)
        if not (root / path).is_file():
            continue  # Historical fixtures need not contain later publications.
        if (root / path).stat().st_size != size or file_hash(root / path) != digest:
            raise ValueError("Atlas evidence navigation public report changed: " + scope)
        record = load(path)
        if record.get("schema") != schema:
            raise ValueError("Atlas evidence navigation has an unsupported public schema")
        entry = {"scope_id": scope, "public_schema": schema, "qualification_input": False,
                 "selected_row_credit": False, "readout": path.with_name("README.md").as_posix(),
                 "public_result": relative, "public_result_sha256": digest, "public_result_bytes": size}
        if scope == "live_clean_common26":
            if (record.get("config_sha256") != FULL_ORIGINAL_ATLAS_CONFIG_SHA256
                    or any(record.get(key) is not False for key in
                           ("qualification_credit", "default_adoption", "speed_ranking",
                            "same_source_whole26_claim", "measured_whole26_ranking"))
                    or not isinstance(record.get("campaign_status"), str)
                    or type(record.get("halt_required")) is not bool):
                raise ValueError("retained common26 navigation changed its scope")
            entry.update(label="Retained live-clean common-26 diagnostic", status=record["campaign_status"],
                         halt_required=record["halt_required"], counts=deepcopy(record["counts"]),
                         source=deepcopy(record["current_source"]),
                         retained_first_case_source=deepcopy(record["retained_first_case_source"]))
        elif scope == "native_restoration":
            cases = record.get("native_cases", [])
            if (not isinstance(cases, list) or [case.get("task_id") for case in cases]
                    != ["grid100", "rotated100", "staggered100"]
                    or any(record.get("claims", {}).get(key) is not False for key in
                           ("qualification_credit", "original_common26_credit", "default_adoption", "speed_ranking"))):
                raise ValueError("native restoration navigation changed its roster or scope")
            for case in cases:
                source = case.get("source", {})
                grade = case.get("grade", {})
                if (case.get("candidate_id") != "atlas-native-original-representation-selected-seed0-v1"
                        or case.get("recipe_sha256") != "152ee607b4a2b18d99e9f440985b920363a481a14236ae12409747e9c283738a"
                        or source.get("origin_commit") != "9da0e927a0a4242ee813ceb0340e9fee70ccdc40"
                        or source.get("execution_digest") != "00c19ef01f68e3222888a9b0edc6fcc5f635a384b982260c4e005832c1ed09ae"
                        or grade.get("status") not in COMPLETE or grade.get("full_protocol_complete") is not True):
                    raise ValueError("native restoration navigation changed its binding or completion")
            entry.update(label="Restored native Atlas — raw particles, public selected/noisy",
                         candidate_id=cases[0]["candidate_id"], recipe_sha256=cases[0]["recipe_sha256"],
                         source={"origin_commit": cases[0]["source"]["origin_commit"],
                                 "digest": cases[0]["source"]["execution_digest"]},
                         task_statuses=[{"task_id": case["task_id"], "status": case["grade"]["status"]} for case in cases])
        else:
            grade = record.get("accepted_evidence", {}).get("grade", {})
            if (record.get("candidate_id") != "atlas-full-original-ae-sourceguard-repair-v1"
                    or record.get("task_id") != "ae_gan_hold"
                    or record.get("recipe_sha256") != "a26eb000fbb7616c90265a4bf31c5126e75f86a4cbd04912d71dc0292167e250"
                    or record.get("source") != {"origin_commit": "36718f9ecbffe6a217914de7a9c22cff1349079e",
                                               "digest": "2f71661f3407a3af3b166493fa0cb1f6730e8ac4689c2b37a8c8a9e424c5c233"}
                    or record.get("complete_protocol") is not True or record.get("completed_updates") != 250
                    or record.get("observation_count") != 24 or record.get("status") not in COMPLETE
                    or grade.get("status") != record["status"] or grade.get("gate_status") != record["status"]
                    or any(record.get(key) is not False for key in
                           ("qualification_credit", "default_adoption", "speed_ranking"))):
                raise ValueError("repaired AE navigation changed its binding or completion")
            entry.update(label="Repaired Atlas AE — auxiliary MoG, live scheduled-noise",
                         candidate_id=record["candidate_id"], recipe_sha256=record["recipe_sha256"],
                         source=deepcopy(record["source"]),
                         task_statuses=[{"task_id": record["task_id"], "status": record["status"]}])
        cohorts.append(entry)
    if not cohorts:
        return None
    return {"schema": "pg_atlas_evidence_navigation_v1", "qualification_input": False,
            "selected_row_credit": False, "cohorts": cohorts}


def _full_original_atlas_status(root, page, progress):
    lines = []
    navigation = progress.get("atlas_evidence_navigation")
    if navigation is not None:
        links = []
        for cohort in navigation["cohorts"]:
            outcome = (", ".join(row["task_id"] + " " + row["status"] for row in cohort["task_statuses"])
                       if "task_statuses" in cohort else cohort["status"])
            links.append(link(root, page, cohort["label"] + ": " + outcome, cohort["readout"]))
        if navigation.get("next_steps"):
            links.append(link(root, page, "Atlas inventory gaps and bounded next steps", navigation["next_steps"]["readout"]))
        lines += ["**Atlas measured evidence — separate configurations and contracts.** " + " · ".join(links) + ". " +
                  "These records do not add cells to the selected Atlas row or grant prerequisite credit, "
                  "default adoption or speed ranking.", ""]
    context = progress.get("full_original_atlas_common26")
    if context is None:
        return lines
    first = context["first_case"]
    metrics = "; ".join(cell(check["metric"]) + " " + number(check["value"]) + " " + cell(check["op"]) + " " +
                        number(check["threshold"]) + " (" + cell(check["status"]) + ")"
                        for check in first["metric_receipts"])
    return lines + ["**Archived first Full Atlas case: two_pole " + first["status"] + ".** " + metrics + ". " +
                    "This accepted first case completed 80 updates and 24 ordinary live observations at seed 0. " +
                    link(root, page, "Archived first-case result and goal GIF", context["readout"]) + " · " +
                    link(root, page, "Pinned result, full Recipe and source", context["public_result"]) + ". " +
                    "Its first-case status does not describe the later diagnostic scopes or the canonical Atlas "
                    "configuration selected in the recorded table. No selected-table cells, prerequisite credit, "
                    "default adoption or speed ranking are awarded.", ""]

def cell(value):
    return str(value if value is not None else "unavailable").replace("|", "\\|").replace("\n", " ")


def number(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return cell(value)


def link(root, page, label, target, anchor=None):
    relative = os.path.relpath(root / target, page.parent)
    return f"[{cell(label)}]({relative}" + (f"#{anchor}" if anchor else "") + ")"


# A stretch or held task is measured and shown but not counted; a task removed
# from the count stays assigned as diagnostic (experiments/forge/views.py). Both
# carry a note in the view naming the reason and linking its explanation.
HOLD_REASON_TEXT = {"no_global_recipe": "no one recipe passes it together with the counted tests",
                    "unsolved": "no recipe has passed its current declaration",
                    "frontier": "frontier stretch benchmark beyond the required suite",
                    "extra_credit": "extra-credit stretch benchmark",
                    "optional": "optional stretch benchmark"}
REMOVAL_REASON_TEXT = {"phase": "phase of {} (counted there)", "duplicate": "duplicate of {} (counted there)",
                       "superseded": "replaced by newer tests", "ill_posed": "cannot run as declared"}


def role_note(assignment):
    """The stretch, hold or removal note of an uncounted assignment, else None."""
    return stretch_note(assignment) if is_stretch(assignment) else assignment.get("removal")


def role_label(assignment):
    """Plain role text; stretch, held and removed roles name their reason."""
    note = role_note(assignment)
    if not note:
        return assignment["importance"]
    reason = note["reason"]
    if not is_stretch(assignment):
        return "removed: " + REMOVAL_REASON_TEXT.get(reason, reason.replace("_", " ")).format(note.get("counted_by"))
    prefix = "stretch" if assignment["importance"] == "stretch" else "held"
    parents = note.get("blocked_by") or []
    if reason in {"held_prerequisite", "stretch_prerequisite"} and parents:
        return (f"{prefix}: prerequisite {parents[0]} is {prefix}" if len(parents) == 1 else
                f"{prefix}: prerequisites {', '.join(parents)} are {prefix}")
    return f"{prefix}: " + HOLD_REASON_TEXT.get(reason, reason.replace("_", " "))


def role_cell(root, page, assignment):
    """The role, linked to its explanation when the assignment is stretch, held or removed."""
    note = role_note(assignment)
    if not note:
        return cell(assignment["importance"])
    if "explanation" not in note:
        return cell(role_label(assignment))
    path, _, anchor = note["explanation"].partition("#")
    return link(root, page, role_label(assignment), path, anchor)


def uncounted_entries(views):
    """One row per stretch, held or removed task and note, built from the view declarations.

    The same disposition normally appears in every view that counted the task;
    differing notes stay separate rows rather than being merged.
    """
    entries = {}
    for view in views:
        for assignment in view["assignments"]:
            note = role_note(assignment)
            if not note:
                continue
            key = (assignment["task"], assignment["importance"], json.dumps(note, sort_keys=True))
            entry = entries.setdefault(key, {"task": assignment["task"], "assignment": assignment,
                                             "note": note, "views": []})
            entry["views"].append((view["id"], assignment["qualification_tier"]))
    return sorted(entries.values(), key=lambda entry: (not is_stretch(entry["assignment"]),
                                                       min(tier for _, tier in entry["views"]), entry["task"]))


def evaluation_rows(evaluation):
    """Readable bounds from the declaration, without an invented combined score."""
    rows = [(name, op, bound) for name, op, bound in evaluation.get("thresholds", [])]
    for name, bound in evaluation.get("coverage_thresholds", {}).items():
        op = ">=" if name.startswith("min_") else "<=" if name.startswith("max_") else "=="
        rows.append(("coverage." + name, op, bound))
    rows += [("accuracy." + name, "<=", bound) for name, bound in evaluation.get("accuracy_limits", {}).items()]
    return rows


def evaluation_notes(evaluation):
    notes = []
    if evaluation.get("minimum_stable_checks"):
        notes.append(f"At least {evaluation['minimum_stable_checks']} consecutive passing terminal observations.")
    if evaluation.get("kind") == "transfer_sustained":
        notes.append(f"All {evaluation.get('observations', 24)} declared observations and final live metrics are required.")
    if evaluation.get("kind") == "native_accuracy":
        notes.append("Both sustained coverage and independent holdout accuracy must pass.")
    if evaluation.get("conditions"):
        notes.append("Exact state/output parity for: " + ", ".join(evaluation["conditions"]) + "; bound source audit required.")
    for key, label in (("confirmation_checks", "Confirmation checks"), ("hold_budget", "Hold updates"),
                       ("extension_steps", "Extension updates"), ("recovery_deadline", "Recovery deadline in updates")):
        if key in evaluation:
            notes.append(f"{label}: {evaluation[key]}.")
    if evaluation.get("kind") == "paired_adaptation":
        notes.append("Active quality must hold before the shift and throughout the post-deadline window; "
                     "the matched frozen control must have zero passing post-deadline checks.")
    guards = evaluation.get("guards", {})
    if guards:
        notes.append("Execution guards: " + "; ".join(
            key.replace("_", " ") + " = " + (", ".join(value) if isinstance(value, list) else str(value))
            for key, value in guards.items()) + ".")
    return notes


def _execution_recorded(status, receipt):
    """Distinguish a recorded run from preflight rejection or missing evidence."""
    if status in COMPLETE:
        return True  # The immutable selected row already records a measurement.
    result = receipt.get("result", {})
    cost = result.get("cost", {})
    return (result.get("raw_status") == "completed"
            or any(isinstance(cost.get(key), (int, float)) and cost[key] > 0
                   for key in ("execution_seconds", "wall_seconds")))


def _count(tasks):
    statuses = Counter(task["status"] for task in tasks)
    incompatible = sum(_incompatible(task) for task in tasks)
    not_run = sum(not task["execution_recorded"] for task in tasks) - incompatible
    value = {"passed": statuses.get("PASS", 0), "total": len(tasks),
            "incomplete": bool(not_run or incompatible), "not_run": not_run, "incompatible": incompatible,
            "counts": dict(sorted(statuses.items()))}
    if any("run_attempt" in task for task in tasks):
        coverage = Counter()
        for task in tasks:
            attempt = task.get("run_attempt", {})
            if task["execution_recorded"]:
                continue
            if not attempt.get("attempted"):
                coverage["incompatible" if _incompatible(task) else "not_run"] += 1
            elif attempt["status"] == "BLOCKED":
                coverage["incompatible" if attempt.get("block_kind") == "incompatible" else "blocked"] += 1
            else:
                coverage["running" if attempt["status"] == "RUNNING" else "diagnostic_attempts"] += 1
        value["attempt_coverage"] = dict(coverage)
    return value


def _incompatible(task):
    return (not task["execution_recorded"] and not task.get("separate_diagnostic_executed")
            and (task["status"] == "BLOCKED" or bool(task.get("preflight_blockers"))))


def _sum(counts):
    statuses = Counter()
    for count in counts:
        statuses.update(count["counts"])
    value = {"passed": sum(c["passed"] for c in counts), "total": sum(c["total"] for c in counts),
            "incomplete": any(c["incomplete"] for c in counts),
            "not_run": sum(c["not_run"] for c in counts),
            "incompatible": sum(c["incompatible"] for c in counts), "counts": dict(sorted(statuses.items()))}
    if any("attempt_coverage" in c for c in counts):
        coverage = Counter()
        for count in counts:
            coverage.update(count.get("attempt_coverage", {k: count.get(k, 0) for k in ("not_run", "incompatible")}))
        value["attempt_coverage"] = dict(coverage)
    return value


def score(count, *, marker=True):
    value = f"{count['passed']}/{count['total']}"
    coverage = count.get("attempt_coverage", count)
    missing = [f"{count[key]} {label}" for key, label in (("not_run", "not run"), ("incompatible", "incompatible"))
               if count.get(key)]
    if "attempt_coverage" in count:
        missing = [f"{coverage[key]} {label}" for key, label in (
            ("not_run", "not run"), ("incompatible", "incompatible"), ("blocked", "blocked"),
            ("errors", "error" if coverage.get("errors") == 1 else "errors"),
            ("timeouts", "timeout" if coverage.get("timeouts") == 1 else "timeouts"),
            ("incomplete_attempts", "incomplete"), ("interrupted_attempts", "interrupted"),
            ("queued", "queued"), ("running", "running"),
            ("diagnostic_attempts", "diagnostic attempt" if coverage.get("diagnostic_attempts") == 1 else "diagnostic attempts"))
            if coverage.get(key)]
    return value + (" (" + ", ".join(missing) + ")" if marker and missing else "")


def _definition_label(result):
    return {"matches": "matches recorded run", "CHANGED": "changed since run",
            "unbound": "recorded definition unavailable"}[result["current_contract"]]


def _same_active_runtime(left, right):
    """Match served hardware/software while retaining separate cohort identities."""
    backend = left.get("execution_backend")
    runtime = left.get("runtime")
    profile = left.get("compute_profiles", {}).get(backend)
    return (backend in {"cpu", "cuda"} and backend == right.get("execution_backend")
            and isinstance(runtime, dict) and bool(runtime) and runtime == right.get("runtime")
            and isinstance(profile, dict) and bool(profile)
            and profile == right.get("compute_profiles", {}).get(backend))


def certified_retry_successors(summaries: dict) -> dict:
    """Resolve certified execution repairs using compact receipts alone.

    Compatibility keys bind the complete task execution, evaluator, protocol,
    initializer, RNG and compute law. Scientific outcomes cannot be retried.
    Eligibility comes from the saved explicit authorization; this projection
    does not infer the cause of an error or authorize a new execution.
    """
    successors, parents = {}, {}

    def task_keys(summary):
        results = summary.get("task_results", [])
        keys = {row.get("task_id"): row.get("compatibility_key") for row in results}
        if (not keys or len(keys) != len(results)
                or any(not isinstance(name, str) or not name or not isinstance(key, str) or not key
                       for name, key in keys.items())):
            raise ValueError("compact retry lacks complete unique task compatibility keys")
        return keys

    for child, summary in summaries.items():
        link = summary.get("retry_of")
        if not link:
            continue
        if (not isinstance(link, dict)
                or set(link) != {"attempt_id", "result_hash", "reason", "authorized_at"}
                or not isinstance(link.get("reason"), str) or not link["reason"].strip()
                or not isinstance(link.get("authorized_at"), str) or not link["authorized_at"].strip()):
            raise ValueError("compact retry lacks explicit repair authorization")
        parent = link["attempt_id"]
        if not isinstance(parent, str) or parent not in summaries:
            raise ValueError("compact retry lacks its original receipt")
        previous = summaries[parent]
        for attempt, receipt in ((parent, previous), (child, summary)):
            if (receipt.get("attempt_id") != attempt or receipt.get("certificate_validated") is not True
                    or receipt.get("qualification_input") is not False or receipt.get("qualification_reuse") is not False
                    or any(not isinstance(receipt.get(key), str) or not receipt[key]
                           for key in ("candidate_id", "candidate_revision"))
                    or not receipt.get("provenance", {}).get("canonical_result_hash")):
                raise ValueError("compact retry requires certified nonqualifying receipt identities")
        original_hash = previous.get("provenance", {}).get("canonical_result_hash")
        if not isinstance(original_hash, str) or not original_hash or link["result_hash"] != original_hash:
            raise ValueError("compact retry original result hash mismatch")
        if (previous.get("attempt_status") not in {"error", "timeout", "cancelled"}
                or not previous.get("task_results")
                or any(row.get("gate_status") != "INCOMPLETE" for row in previous["task_results"])):
            raise ValueError("compact retry cannot supersede scientific or invalid evidence")
        if (any(previous.get(key) != summary.get(key) for key in ("candidate_id", "candidate_revision", "runtime", "campaign_id"))
                or not previous.get("runtime")
                or not previous.get("provenance", {}).get("source_digest")
                or previous["provenance"]["source_digest"] != summary.get("provenance", {}).get("source_digest")
                or previous["provenance"].get("source_origin_commit") != summary.get("provenance", {}).get("source_origin_commit")
                or task_keys(previous) != task_keys(summary)):
            raise ValueError("compact retry changes the original scientific identity")
        if parent in successors and successors[parent] != child:
            raise ValueError("conflicting compact retry branches require investigation")
        successors[parent], parents[child] = child, parent
    for child in parents:
        seen, cursor = set(), child
        while cursor in parents:
            if cursor in seen:
                raise ValueError("cyclic compact retry lineage is invalid")
            seen.add(cursor)
            cursor = parents[cursor]
    return successors


def _solution_families(root, families, load, selected_rows, current_row_count, *, selection_card=None):
    """Select whole recorded configurations for display, preserving evidence IDs.

    Optimizer and tuning registries may predate the solution taxonomy. Their
    original identities still bind receipts; reporting aliases only group the
    navigation. This is a best-observed display, not scientific qualification.
    """
    registry_path = Path("configs/forge/trainer-families.json")
    registry = load(registry_path) if (root / registry_path).is_file() else {}
    definitions = {item["id"]: item for item in registry.get("families", [])}
    from .trainer_families import CURRENT_SELECTION, current_family_candidates, scientific_row_hash
    groups, configurations = {}, []
    for family in families.values():
        definition = definitions.get(family["id"], {})
        if definition.get("inventory_visible", True) is False:
            continue
        target = definition.get("reporting_family", family["id"])
        identifier(target, "reporting family")
        if target != family["id"]:
            if target not in definitions or definitions[target].get("reporting_family", target) != target:
                raise ValueError("reporting family must reference a registered solution without another alias")
            if definitions[target].get("inventory_visible", True) is False:
                raise ValueError("reporting family cannot reference a hidden solution")
        groups.setdefault(target, []).append(family)
    choices = ({choice["family"]: choice for choice in current_family_candidates(
        root, family_ids=set(groups), selection_card=selection_card)}
               if selection_card is not None or (root / CURRENT_SELECTION).is_file() else {})
    if choices:
        load(CURRENT_SELECTION)  # Bind the display choice in publication provenance.
    solutions = []
    for target, members in groups.items():
        base = next((family for family in members if family["id"] == target), None)
        if base is None:
            raise ValueError("reporting family needs its own recorded solution row")
        solution = deepcopy(base)
        choice = choices.get(target)

        def is_current(cohort):
            if choice is None or cohort["row_index"] >= current_row_count:
                return False
            row = selected_rows[cohort["row_index"]]
            if choice["selection_kind"] == "unmeasured_declaration":
                from .trainer_families import _declaration_only_row
                return row["candidate_id"] == choice["candidate_id"] and _declaration_only_row(row)
            return (row["candidate_id"] == choice["candidate_id"]
                    and scientific_row_hash(row) == choice["scientific_row_sha256"])

        if len(members) > 1:
            solution["cohorts"] = []
            runtimes = {}
            for member in members:
                for cohort in member["cohorts"]:
                    runtimes.setdefault(cohort["anchor"], []).append((member, cohort))
            for candidates in runtimes.values():
                # All candidates share the displayed required denominator.
                # Choose a complete row; never assemble passing task cells.
                def rank(item):
                    member, cohort = item
                    counts = cohort["total"]
                    return (-counts["passed"],
                            -sum(counts["counts"].get(status, 0) for status in COMPLETE),
                            member["id"] != target, member["id"])

                # Explicit current choices take precedence over pass counts.
                # Old runtimes remain navigation under their original evidence.
                ordered = sorted(candidates, key=lambda item: (not is_current(item[1]), rank(item)))
                winner, cohort = ordered[0]
                selected = deepcopy(cohort)
                selected["configuration_family"] = winner["id"]
                selected["configuration_documentation"] = winner.get("documentation")
                selected["configuration_alternatives"] = [
                    {"family": member["id"], "page": member["page"], "row_index": candidate["row_index"],
                     "total": deepcopy(candidate["total"]), "selected": candidate is cohort}
                    for member, candidate in ordered]
                solution["cohorts"].append(selected)
            solution["configuration_selection"] = "best_recorded_required_passes"
            if any(cohort["configuration_family"] != target for cohort in solution["cohorts"]):
                configuration = deepcopy(base)
                configuration["page"] = (FAMILY_DIRECTORY / (target + "-configuration.md")).as_posix()
                configuration["solution_page"] = base["page"]
                configuration["solution_label"] = base["label"]
                configurations.append(configuration)
                for cohort in solution["cohorts"]:
                    for alternative in cohort["configuration_alternatives"]:
                        if alternative["family"] == target:
                            alternative["page"] = configuration["page"]
            for member in members:
                if member["id"] != target:
                    configuration = deepcopy(member)
                    configuration["solution_page"] = base["page"]
                    configuration["solution_label"] = base["label"]
                    configurations.append(configuration)
        if choice is not None:
            current = [cohort for cohort in solution["cohorts"] if is_current(cohort)]
            if len(current) != 1:
                raise ValueError(f"family {target}: current benchmark must match exactly one verified scientific row")
            solution["current_cohort_anchor"] = current[0]["anchor"]
            solution["benchmark_configuration"] = deepcopy(choice)
            solution["cohorts"].sort(key=lambda cohort: cohort["anchor"] != current[0]["anchor"])
            if len(members) > 1:
                solution["configuration_selection"] = "explicit_current_configuration"
        solutions.append(solution)
    return solutions, configurations


def build_progress(root: Path, publication: dict, *, selection_card: dict | None = None) -> dict:
    """Project current view placement over immutable selected task outcomes."""
    root = Path(root)
    inputs = {}

    def load(path):
        from .trainer_families import CURRENT_SELECTION
        if selection_card is not None and path == CURRENT_SELECTION:
            text = json.dumps(selection_card, sort_keys=True, indent=2) + "\n"
            inputs[path.as_posix()] = hashlib.sha256(text.encode()).hexdigest()
            return deepcopy(selection_card)
        inputs[path.as_posix()] = file_hash(root / path)
        return read_json(root / path)

    views = [load(path.relative_to(root)) for path in sorted((root / "configs/forge/views").glob("*.json"))]
    task_files = sorted([*(root / "configs/forge/tasks").glob("*.json"),
                         *(root / "configs/forge/task-variants").rglob("*.json")])
    tasks = {path.stem: load(path.relative_to(root)) for path in task_files}
    if len(tasks) != len(task_files):
        raise ValueError("family report task declarations have duplicate ids")
    task_paths = {path.stem: path.relative_to(root).as_posix() for path in task_files}
    from .scoped_publications import load_publications
    attachments = load_publications(root, load)
    # The policy-cloud cohorts publish their own records file (never the
    # repository attempt report); it fills only their scoped view cells.
    from .policy_cloud_publication import VIEWS as POLICY_CLOUD_VIEWS, family_overlay, load_records, view_counts
    policy_cloud = load_records(root, load)
    support = load(COVERAGE_PREFLIGHT) if (root / COVERAGE_PREFLIGHT).is_file() else None
    attempt_path = Path("reports/forge/attempt-coverage.json")
    from .publication_integrity import load_corrections
    evidence_corrections = load_corrections(root, load)
    run_attempts = {}
    attempt_summary = None
    if (root / attempt_path).is_file():
        registry = load(attempt_path)
        attempt_summary = deepcopy(registry["summary"])
        if (registry.get("schema") != "forge_attempt_coverage_v1"
                or registry.get("qualification_input") is not False
                or registry.get("qualification_reuse") is not False):
            raise ValueError("run-attempt coverage must remain diagnostic only")
        from .wallclock import RERUNS, apply_reruns, load_reruns
        if (root / RERUNS).is_file():
            load(RERUNS)  # Record the rerun overlay among the publication inputs.
        for attempt in apply_reruns(registry["records"], load_reruns(root)):
            key = (attempt["source_digest"], attempt["candidate_revision"], attempt["task"],
                   attempt["execution_sha256"], attempt["evaluation_sha256"])
            if attempt["key"] != stable_hash(list(key)) or key in run_attempts:
                raise ValueError("invalid or duplicate run-attempt binding")
            if attempt["status"] not in {"PENDING", "RUNNING", "PASS", "FAIL", "INCOMPLETE", "INVALID",
                                         "ERROR", "TIMEOUT", "BLOCKED", "INTERRUPTED"}:
                raise ValueError("invalid run-attempt status")
            attempt = deepcopy(attempt)
            if attempt["key"] in evidence_corrections:
                attempt["evidence_correction"] = deepcopy(evidence_corrections[attempt["key"]])
            run_attempts[key] = attempt
    followup = Path("reports/forge/actual-missing-diagnostics-20261014/results.json")
    diagnostic_executions = set()
    if (root / followup).is_file():
        measured = load(followup)  # Display evidence, never a qualification input.
        if (measured.get("qualification_input") is not False
                or measured.get("qualification_reuse") is not False
                or measured.get("formulations_modified") is not False):
            raise ValueError("actual execution coverage must remain diagnostic only")
        diagnostic_executions = {(r["source_digest"], r["candidate_revision"], r["task"],
            r["task_execution_sha256"], r["task_evaluation_sha256"])
            for r in measured["records"] if r["lane"] == "ordinary_diagnostics"
            and r["raw_status"] == "COMPLETE" and r["status"] in COMPLETE}
    if support is not None and (support.get("scope") != "selected_formulation_task_preflight_only"
                                or support.get("qualification_input") is not False
                                or support.get("qualification_reuse") is not False):
        raise ValueError("coverage preflight is display-only task compatibility")
    scoped_publications, media = attachments["cohorts"], attachments["media"]
    inputs.update({item["gif"]: item["gif_sha256"] for item in media.values()})
    for view in views:
        identifier(view["id"], "view")
        names = [assignment["task"] for assignment in view["assignments"]]
        if len(names) != len(set(names)):
            raise ValueError("family report view has duplicate assignments")
        if tasks and set(names) - tasks.keys():
            raise ValueError("family report view references a missing task")
        shown = view.get("reporting", {}).get("families")
        if shown is not None and (view.get("reporting", {}).get("family_totals") is not False
                                  or not isinstance(shown, list) or not shown
                                  or not all(isinstance(name, str) and name for name in shown)):
            raise ValueError("only a scoped view may name the families whose pages display it")
    scoped = [view for view in views if view.get("reporting", {}).get("family_totals") is False]
    # A view whose tests are all held still belongs to the totals; it counts nothing.
    ordinary = [view for view in views if view not in scoped
                and view.get("evidence_scope") not in {"research_diagnostic", "calibration_diagnostic"}
                and any(a["importance"] in COUNTED_OR_HELD for a in view["assignments"])]
    diagnostics = [view for view in views if view not in ordinary and view not in scoped]
    families = {}
    selected_rows = publication["rows"] + publication.get("historical_family_rows", [])
    for row_index, selected in enumerate(selected_rows):
        family_id = identifier(selected.get("trainer_family", selected["candidate_id"]), "trainer family")
        family = families.setdefault(family_id, {"id": family_id, "label": selected["technique"],
                                                "page": (FAMILY_DIRECTORY / (family_id + ".md")).as_posix(), "cohorts": [],
                                                "historical_only": row_index >= len(publication["rows"])})
        backend = selected.get("runtime_cohort", {}).get("execution_backend", "unrecorded")
        cohort_anchor = "cohort-" + backend + "-" + stable_hash(selected.get("runtime_cohort", {}))[:12]
        if any(cohort["anchor"] == cohort_anchor for cohort in family["cohorts"]):
            raise ValueError("family report cannot pool multiple selected configurations in one runtime")
        recorded = {task["task_id"]: task for task in selected.get("tasks", [])}
        recorded.update({task["task_id"]: task for task in selected.get("nonrequired_tasks", [])})
        bindings = dict(selected.get("bindings", {}).get("task_contracts", {}))
        catalogs = dict(publication.get("task_contracts", {}))
        reason_catalog = dict(publication.get("status_reasons", {}))
        support_rows = [item for item in (support or {}).get("families", [])
                        if item["candidate_id"] == selected["candidate_id"]
                        and item["candidate_revision"] == selected.get("candidate_revision")
                        and item["source_digest"] == selected.get("bindings", {}).get("source_digest")]
        if len(support_rows) > 1:
            raise ValueError("duplicate selected formulation preflight")
        task_support = support_rows[0]["tasks"] if support_rows else {}
        if support_rows and support_rows[0]["recipe_sha256"] != selected.get("bindings", {}).get("recipe_sha256"):
            raise ValueError("coverage preflight changed the selected recipe")
        separate = [item for item in scoped_publications if item["family"] == family_id
                    and item["row"]["candidate_id"] == selected["candidate_id"]
                    and item["row"].get("candidate_revision") == selected.get("candidate_revision")
                    and _same_active_runtime(item["row"].get("runtime_cohort", {}), selected.get("runtime_cohort", {}))
                    and item["row"]["bindings"]["source_digest"] == selected.get("bindings", {}).get("source_digest")]
        scoped_names = {assignment["task"] for view in scoped for assignment in view["assignments"]}
        scoped_attempts = []
        final_measurements = {task: measured for (family, task), measured in attachments["final_measurements"].items()
                              if family == family_id and measured["candidate_id"] == selected["candidate_id"]
                              and measured["candidate_revision"] == selected.get("candidate_revision")
                              and measured["source_digest"] == selected.get("bindings", {}).get("source_digest")}
        for item in separate:
            measured = item["row"].get("tasks", []) + item["row"].get("nonrequired_tasks", [])
            if any(result["task_id"] not in scoped_names or result["task_id"] in recorded for result in measured):
                raise ValueError("separate cohort results cannot replace selected parent cells")
            recorded.update({result["task_id"]: result for result in measured})
            bindings.update(item["row"]["bindings"].get("task_contracts", {}))
            catalogs.update(item["task_contracts"])
            reason_catalog.update(item["status_reasons"])
            scoped_attempts.extend(item["row"].get("attempt_ids", []))
        receipts, summaries = {}, {}
        for attempt in selected.get("attempt_ids", []) + scoped_attempts:
            identifier(attempt, "attempt")
            path = Path("reports/forge/technique-receipts") / (attempt + ".json")
            summary = load(path)
            if (summary.get("candidate_id") != selected["candidate_id"]
                    or summary.get("candidate_revision") != selected.get("candidate_revision")
                    or summary.get("provenance", {}).get("source_digest") != selected.get("bindings", {}).get("source_digest")):
                raise ValueError("family report receipt differs from the selected configuration/source")
            summaries[attempt] = summary
        successors = certified_retry_successors(summaries)
        retry_history = [{"attempt_id": parent, "superseded_by": child,
                          "retry_of": deepcopy(summaries[child]["retry_of"]),
                          "receipt": f"reports/forge/technique-receipts/{parent}.json",
                          "retry_receipt": f"reports/forge/technique-receipts/{child}.json",
                          "original_task_costs": {row["task_id"]: deepcopy(row.get("cost", {}))
                                                  for row in summaries[parent]["task_results"]}}
                         for parent, child in sorted(successors.items())]
        for attempt, summary in summaries.items():
            if attempt in successors:
                continue
            path = Path("reports/forge/technique-receipts") / (attempt + ".json")
            for result in summary.get("task_results", []):
                # Grades belong to the selected scientific row. Compact metrics
                # enrich its navigation only; a receipt never creates a pass.
                if result["task_id"] in recorded:
                    final = final_measurements.get(result["task_id"])
                    if final and final["attempt_id"] != attempt:
                        continue
                    prior = receipts.get(result["task_id"])
                    if prior and prior["result"] != result:
                        raise ValueError("family report has conflicting compact task results")
                    receipts[result["task_id"]] = {"path": path.as_posix(), "attempt_id": attempt, "result": result}

        results = {}
        assigned = {assignment["task"] for view in views if _displayed_for(view, family_id)
                    for assignment in view["assignments"]}
        for name in sorted(assigned):
            previous = recorded.get(name, {})
            contract_hash = bindings.get(name)
            contract = catalogs.get(contract_hash, {})
            task = tasks.get(name, {})
            match = "unbound"
            changed = []
            if task and contract:
                if contract.get("execution_sha256") != task_execution_fingerprint(task):
                    changed.append("execution (host, recipe binding, prior, initialization or budget)")
                if contract.get("evaluation_sha256") != task_evaluation_fingerprint(task):
                    changed.append("evaluation (gates or sampling law)")
                if contract.get("timeout_seconds") != task.get("resources", {}).get("timeout_seconds"):
                    changed.append("timeout reservation")
                match = "CHANGED" if changed else "matches"
            receipt = receipts.get(name, {})
            reasons = reason_catalog.get(previous.get("reasons_sha256"), [])
            status = previous.get("status", "UNKNOWN")
            reason = ("; ".join(item.get("message", item.get("reason", json.dumps(item, sort_keys=True)))
                                if isinstance(item, dict) else str(item)
                                for item in previous.get("reasons", reasons)) or previous.get("reason")
                      or receipt.get("result", {}).get("reason"))
            if not reason:
                reason = ("; ".join(selected.get("blockers", [])) if status == "BLOCKED" else None)
            reason = reason or {"UNKNOWN": "No recorded result for this selected configuration and source.",
                                "NOT_RUN": "This task was not executed for the selected configuration and source.",
                                "BLOCKED": "The frozen selected cohort lacks task support; no scientific measurement was admitted.",
                                "INVALID": "The recorded execution violated its frozen contract; it supplies no valid measurement.",
                                "ERROR": "The recorded attempt encountered an execution error; see its receipt for the cause.",
                                "INCOMPLETE": "The recorded execution or required observations did not complete."}.get(
                                    status, "Recorded scientific gate " + status + ".")
            coverage = ("Test definition changed since this run: " + "; ".join(changed) + ". Earlier verdict preserved." if changed else
                        "Recorded test definition unavailable; compatibility with today's declaration cannot be checked." if match == "unbound" else
                        "Test definition matches the recorded conditions.")
            results[name] = {"task_id": name, "status": previous.get("status", "UNKNOWN"),
                             "execution_recorded": _execution_recorded(status, receipt),
                             "attempt_id": receipt.get("attempt_id"),
                             "current_contract": match, "receipt": receipt.get("path"),
                             "metrics": deepcopy(receipt.get("result", {})), "recorded_contract": contract_hash,
                             "recorded_conditions": {key: deepcopy(contract.get(key, {})) for key in ("prior", "sampling")},
                             "reason": reason, "coverage_reason": coverage, "changed_contract_fields": changed,
                             "device": receipt.get("result", {}).get("device", receipt.get("result", {}).get("cost", {}).get("device")),
                             "policy_parent": deepcopy(task.get("policy_parent"))}
            attempt_key = (selected.get("bindings", {}).get("source_digest"), selected["candidate_revision"],
                           name, contract.get("execution_sha256", task_execution_fingerprint(task) if task else None),
                           contract.get("evaluation_sha256", task_evaluation_fingerprint(task) if task else None))
            if attempt_key in run_attempts:
                results[name]["run_attempt"] = deepcopy(run_attempts[attempt_key])
            supported = task_support.get(name)
            if supported is not None:
                if contract and (supported["execution_sha256"] != contract.get("execution_sha256")
                                 or supported["evaluation_sha256"] != contract.get("evaluation_sha256")):
                    raise ValueError("coverage preflight differs from recorded task: " + name)
                results[name]["preflight_blockers"] = supported["blockers"]
                results[name]["preflight_source_digest"] = support_rows[0]["source_digest"]
                if not results[name]["execution_recorded"] and supported["blockers"]:
                    results[name]["reason"] = "Skipped incompatible test: " + "; ".join(supported["blockers"])
                diagnostic_key = (support_rows[0]["source_digest"], selected["candidate_revision"], name,
                                  supported["execution_sha256"], supported["evaluation_sha256"])
                if diagnostic_key in run_attempts:
                    results[name]["run_attempt"] = deepcopy(run_attempts[diagnostic_key])
                if diagnostic_key in diagnostic_executions and not results[name]["execution_recorded"]:
                    results[name]["separate_diagnostic_executed"] = True
                    results[name]["reason"] = ("Executable: completed in a separate-runtime diagnostic. "
                        "This ordinary cohort remains not run; see actual-missing-diagnostics-20261014.")
            matching_media = [item for (family, task_id, attempt), item in media.items()
                              if family == family_id and task_id == name
                              and attempt in selected.get("attempt_ids", []) + scoped_attempts
                              and (name not in final_measurements or attempt == final_measurements[name]["attempt_id"])
                              and item["recorded_grade"] == status]
            if matching_media:
                results[name]["training_media"] = matching_media
            audit = attachments["clock_audits"].get((family_id, name))
            if (audit and name in final_measurements
                    and audit["attempt_id"] in selected.get("attempt_ids", []) + scoped_attempts
                    and audit["recorded_grade"] == status):
                results[name]["clock_audit"] = audit

        overlay = family_overlay(policy_cloud, family_id, selected, lambda name: (
            (task_execution_fingerprint(tasks[name]), task_evaluation_fingerprint(tasks[name])) if name in tasks else (None, None)))
        for name, cell in (overlay or {}).get("cells", {}).items():
            if name in results:
                results[name].update(cell)

        progress_tiers = evaluated_tiers(views)
        def view_row(view):
            tiers = {}
            assignments = sorted(view["assignments"], key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"]))
            for tier in progress_tiers:
                required = [results[a["task"]] for a in assignments if str(a["qualification_tier"]) == tier
                            and a["importance"] == "required"]
                if overlay is not None and view["id"] in POLICY_CLOUD_VIEWS:
                    # The same counts as the cohort's leaderboard row.
                    tiers[tier] = view_counts({result["task_id"]: result["run_attempt"] for result in required
                                               if "run_attempt" in result}, [result["task_id"] for result in required])
                else:
                    tiers[tier] = _count(required)
            row = {"id": view["id"], "revision": view["revision"], "tiers": tiers,
                   "total": _sum(list(tiers.values()))}
            held = [results[a["task"]] for a in assignments if is_stretch(a)]
            if held:
                row["held"] = _count(held)  # Measured and shown, never in a total.
            return row

        view_rows = [view_row(view) for view in ordinary]
        family["cohorts"].append({"anchor": cohort_anchor, "backend": backend,
                                   "row_index": row_index, "tasks": results, "views": view_rows,
                                   "scoped_views": [view_row(view) for view in scoped if _displayed_for(view, family_id)],
                                   "scoped_publications": [{**{key: item[key] for key in ("path", "view", "source_commit")},
                                                            "cohort": item["row"].get("cohort"),
                                                            "runtime_cohort_sha256": stable_hash(item["row"].get("runtime_cohort", {}))}
                                                           for item in separate],
                                   "tiers": {tier: _sum([view["tiers"][tier] for view in view_rows]) for tier in progress_tiers},
                                   "total": _sum([view["total"] for view in view_rows])})
        if retry_history:
            family["cohorts"][-1]["retry_history"] = retry_history
        if overlay is not None:
            family["cohorts"][-1]["policy_cloud_publication"] = overlay["publication"]
    documented, tag_definitions = family_documentation.load_family_documentation(root, families, load)
    for family in families.values():
        family.update(documented.get(family["id"], {"tags": [], "documentation": None}))
        illustration = (family.get("documentation") or {}).get("illustration")
        if illustration is not None:
            inputs[illustration["path"]] = file_hash(root / illustration["path"])
    active, configurations = _solution_families(
        root, {name: family for name, family in families.items() if not family["historical_only"]}, load,
        selected_rows, len(publication["rows"]), selection_card=selection_card)
    atlas_navigation = _atlas_evidence_navigation(root, load)
    next_steps = Path("reports/forge/atlas-inventory-next-steps-20261005.md")
    if atlas_navigation is not None and (root / next_steps).is_file():
        inputs[next_steps.as_posix()] = file_hash(root / next_steps)
        atlas_navigation["next_steps"] = {"readout": next_steps.as_posix(),
                                          "sha256": inputs[next_steps.as_posix()]}
    first_full_atlas = _full_original_atlas_first_case(root, load)
    from .research_tiers import coverage_report, COVERAGE
    coverage = coverage_report(root)
    if coverage is not None:
        inputs[COVERAGE.as_posix()] = file_hash(root / COVERAGE)
        inputs.update(coverage["inputs"])
    problems = load(PROBLEM_DEFINITIONS) if (root / PROBLEM_DEFINITIONS).is_file() else None
    # A larger current suite cannot invalidate the certified older matrix or
    # turn its missing new tests into failures. Pin the exact original scope,
    # never copies of its rendered scores. Fixtures and clean checkouts can
    # verify these bytes without hydrating any training artifacts.
    from .leaderboard_contract import load_fixed_test_list
    fixed = load_fixed_test_list(root)
    if fixed is not None:
        inputs[fixed["path"]] = fixed["sha256"]
    recorded_scope = None
    binding = (fixed or {}).get("recorded_display_scope")
    if binding is not None:
        relative = Path(binding["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("recorded display scope must be repository-relative")
        text = (root / relative).read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != binding["sha256"]:
            raise ValueError("recorded display scope digest changed")
        inputs[relative.as_posix()] = digest
        recorded_scope = {"path": relative.as_posix(), "sha256": digest, "text": text}
    progress = {"schema_version": 1, "scope": "recorded_view_progress", "qualification_input": False,
                "qualification_reuse": False, "views": views, "diagnostic_views": [view["id"] for view in diagnostics],
                "scoped_views": [view["id"] for view in scoped], "tiers": list(progress_tiers),
                "families": active,
                "configuration_families": configurations,
                "historical_families": [family for family in families.values() if family["historical_only"]],
                "tag_definitions": tag_definitions,
                "task_paths": task_paths,
                "problem_definitions": problems,
                "input_hashes": dict(sorted(inputs.items())),
                "renderer_sha256": file_hash(Path(__file__)),
                "integrity_renderer_sha256": file_hash(Path(__file__).with_name("publication_integrity.py")),
                "research_renderer_sha256": file_hash(Path(__file__).with_name("research_tiers.py")),
                "pr399_renderer_sha256": file_hash(Path(__file__).with_name("pr399_scores.py")),
                "policy_cloud_renderer_sha256": file_hash(Path(__file__).with_name("policy_cloud_publication.py")),
                "documentation_loader_sha256": file_hash(Path(family_documentation.__file__))}
    if recorded_scope is not None:
        progress["recorded_display_scope"] = recorded_scope
    if first_full_atlas is not None:
        progress["full_original_atlas_common26"] = {**first_full_atlas, "archived_snapshot": True,
                                                   "counts_scope": "AT_FIRST_ACCEPTED_CASE"}
    if atlas_navigation is not None:
        progress["atlas_evidence_navigation"] = atlas_navigation
    if attempt_summary is not None:
        progress["run_attempt_coverage"] = attempt_summary
    return progress


def _table_header(tiers=None, *, selected=False, held=False):
    if tiers is None:
        tiers = TIERS
    req_cols = " | ".join(f"Tier {t} requirements" for t in tiers)
    test_cols = " | ".join(f"Tier {t} tests" for t in tiers)
    header = (f"| Family / selected recipe | {req_cols} | Total view requirements |"
              if selected else f"| View | {test_cols} | Total distinct tests |")
    div_cols = ["---"] + ["---:"] * (len(tiers) + 1)
    if held:
        header += " Held (measured, not counted) |"
        div_cols.append("---:")
    return [header, "| " + " | ".join(div_cols) + " |"]


def _held_cell(root, page, family, cohort, view):
    """Passes / held tests of one view, linked to the family's held-test list."""
    if "held" not in view:
        return "—"
    return link(root, page, score(view["held"]), family["page"], cohort["anchor"] + "-held")


def _summed_views(progress):
    """The views whose counted requirements add up to the selected-recipe totals."""
    excluded = set(progress.get("diagnostic_views", [])) | set(progress.get("scoped_views", []))
    return [view for view in progress.get("views", []) if view["id"] not in excluded]


def _selected_cohorts(progress):
    """One displayed cohort per family: its current benchmark when one is pinned."""
    for family in progress["families"]:
        for cohort in family["cohorts"]:
            if family.get("current_cohort_anchor") and cohort["anchor"] != family["current_cohort_anchor"]:
                continue
            yield family, cohort


def full_suite_tasks(progress):
    """Distinct ordinary questions, including stretch/held tests; omit duplicate phases.

    This is a performance display, independent of tier qualification. A shared
    test is counted once even when several views require or hold it.
    """
    return sorted({a["task"] for view in _summed_views(progress) for a in view["assignments"]
                   if a["importance"] in COUNTED_OR_HELD})


def full_suite_outcomes(cohort, names):
    """Count original selected-row grades without borrowing diagnostic passes."""
    statuses = Counter(cohort["tasks"].get(name, {}).get("status", "UNKNOWN") for name in names)
    return {"passed": statuses.pop("PASS", 0), "failed": statuses.pop("FAIL", 0),
            "other": dict(sorted(statuses.items())), "total": len(names)}


def _outcome_text(counts):
    labels = {"UNKNOWN": "not measured", "NOT_RUN": "not run", "BLOCKED": "blocked",
              "ERROR": "error", "INVALID": "invalid", "INCOMPLETE": "incomplete", "TIMEOUT": "timeout"}
    return "; ".join(f"{count} {labels.get(status, status.lower())}" for status, count in counts.items()) or "0"


def _selected_row(publication, cohort):
    rows = publication.get("rows", [])
    index = cohort.get("row_index")
    return rows[index] if isinstance(index, int) and 0 <= index < len(rows) else {}


def _cohort_tier_statuses(progress, cohort):
    """Required-task statuses per tier, one entry per distinct task.

    A task required by several summed views counts once, at the lowest tier
    any of them assigns (so a task that could have selected the
    configuration is never counted as holdout evidence).
    """
    tier_of = {}
    for view in _summed_views(progress):
        for assignment in view["assignments"]:
            if assignment["importance"] == "required":
                tier = int(assignment["qualification_tier"])
                tier_of[assignment["task"]] = min(tier, tier_of.get(assignment["task"], tier))
    statuses = {}
    for task, tier in sorted(tier_of.items()):
        statuses.setdefault(tier, []).append(cohort.get("tasks", {}).get(task, {}).get("status", "UNKNOWN"))
    return statuses


def _full_suite_section(root, publication, page, selected):
    from .leaderboard_contract import (budget_text, family_search_budget, fixed_outcomes, holdout_claim,
                                       load_fixed_test_list, selection_lineage)
    progress = publication["family_progress"]
    current = full_suite_tasks(progress)
    fixed = load_fixed_test_list(root)
    names = list(fixed["tests"]) if fixed else current
    counted = {a["task"] for view in _summed_views(progress) for a in view["assignments"]
               if a["importance"] == "required"}
    not_counted = [name for name in names if name not in counted]
    if fixed:
        intro = (f"**{len(names)} tests from the fixed test list `{fixed['id']}` (dated {fixed['dated']})**: "
                 + link(root, page, "list", fixed["path"]) + f", sha256 `{fixed['sha256'][:12]}`. "
                 "The denominator never shrinks: every listed test counts in every row, including tests that a "
                 "view now holds or removed from its count. ")
    else:
        intro = (f"**{len(names)} distinct tests** from the current views (no fixed test list is registered "
                 "in this checkout, so this denominator follows view revisions). ")
    lines = ["## Full-suite results — failures included", "",
             intro + "Each test counts once. "
             "A held FAIL is still a failure in this table. Only duplicate questions and phases already included "
             "in another test are excluded. These are recorded results for the selected recipe and source, "
             "not new measurements or current-source qualification.", "",
             link(root, page, "Problem list: what each test asks, what fails, and which configurations have passed",
                  PROBLEM_PAGE.as_posix()) + ".", "",
             "| Family / selected recipe | PASS | FAIL | Not measured | Other outcomes "
             "| Held or not counted (of these) | Total tests | Tuning budget and selection |",
             "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |"]
    claims = []
    for family, cohort in selected:
        name = family["label"]
        if cohort["backend"] != "cuda" or (len(family["cohorts"]) > 1 and not family.get("current_cohort_anchor")):
            name += " (" + cohort["backend"] + ")"
        label = link(root, page, name, family["page"])
        if cohort.get("configuration_family"):
            configuration = next(item for item in progress.get("configuration_families", [])
                                 if item["id"] == cohort["configuration_family"])
            label += " · " + link(root, page, configuration["label"], configuration["page"])
        statuses = {test: cohort.get("tasks", {}).get(test, {}).get("status", "UNKNOWN") for test in names}
        counts = fixed_outcomes(statuses, names, not_counted)
        held = counts["held"]
        held_other = held["total"] - held["passed"] - held["failed"] - held["not_measured"]
        held_text = (f"{held['total']}: {held['passed']} PASS, {held['failed']} FAIL, "
                     f"{held['not_measured']} not measured" + (f", {held_other} other" if held_other else "")
                     ) if held["total"] else "0"
        candidate = _selected_row(publication, cohort).get("candidate_id")
        lineage = selection_lineage(root, candidate) if candidate else {"search_id": None}
        budget = budget_text(family_search_budget(root, family["id"]), lineage)
        lines.append(f"| {label} | {counts['passed']} | **{counts['failed']}** | {counts['not_measured']} | "
                     f"{_outcome_text(counts['other'])} | {held_text} | {counts['total']} | {budget} |")
        if lineage.get("search_id"):
            claim = holdout_claim(_cohort_tier_statuses(progress, cohort), lineage.get("tuning_through_tier"))
            claims.append(f"- **{family['label']}:** {claim['claim']}")
    lines += ["", "Not measured and other outcomes are not scientific failures and earn no pass credit. "
              "Separate diagnostic and historical passes remain on the problem and family pages; "
              "they do not replace a selected recipe's result. Tuning budgets count the family's registered "
              "configuration searches (declared grid sizes) and every declared candidate card (ideas and "
              "configurations, including hand-declared variants); a family with no search and a single card "
              "is labelled untuned. Rows with different budgets are not equally tuned comparisons.", ""]
    if claims:
        lines += ["Selection versus holdout for recipes that descend from a registered search:", "", *claims, ""]
    if fixed:
        added = sorted(set(current) - set(names))
        retired = sorted(set(names) - set(current))
        if added:
            lines += [f"Tests added to the views after {fixed['dated']} (outside this fixed denominator; "
                      "a new dated list adds them): " + ", ".join(f"`{name}`" for name in added) + ".", ""]
        if retired:
            lines += ["Listed tests no longer assigned to a counted view still count above: "
                      + ", ".join(f"`{name}`" for name in retired) + ".", ""]
    return lines


def render_problem_list(root, publication, page):
    """Explain unresolved questions alongside every selected recipe's actual grade."""
    progress = publication["family_progress"]
    selected = list(_selected_cohorts(progress))
    registry = progress.get("problem_definitions") or {}
    if registry and registry.get("schema") != "forge_problem_definitions_v1":
        raise ValueError("unsupported problem definitions schema")
    definitions = registry.get("problems", [])
    names = set(full_suite_tasks(progress))
    if len({item["task"] for item in definitions}) != len(definitions):
        raise ValueError("problem definitions contain duplicate tasks")
    if any(item["task"] not in names for item in definitions):
        raise ValueError("problem definition is outside the ordinary full suite")
    held = {a["task"] for view in _summed_views(progress) for a in view["assignments"]
            if is_stretch(a)}
    lines = ["<!-- Generated Forge problem list -->", "# Unresolved training problems", "",
             link(root, page, "Full-suite leaderboard", "reports/forge/technique-inventory.md") + " · " +
             link(root, page, "All test/protocol results", "reports/forge/test-performance.md"), "",
             "This page explains the outstanding tests, including the held failures. **Held means excluded from "
             "qualification, not solved:** these failures are included in the leaderboard's full-suite comparison. "
             "Passing one problem does not establish that the same configuration passes the whole suite.", "",
             "Selected results below use one recorded recipe/source/runtime cohort per family. Research, diagnostic "
             "and historical configurations are named separately, with links to their original evidence. "
             "An oracle or hand-built solution demonstrates feasibility, not successful training. "
             "Not measured, blocked, timeout and error are never relabelled FAIL.", "",
             *( ["Passer summaries reflect the evidence audit dated **" + registry["evidence_date"] +
                  "**. Selected-result tables are regenerated from the registered publication.", ""]
                if registry.get("evidence_date") else []),
             "| Problem | What it tests | Has any configuration passed? |",
             "| --- | --- | --- |"]
    for item in definitions:
        passers = [family["label"] for family, cohort in selected
                   if cohort["tasks"].get(item["task"], {}).get("status") == "PASS"]
        summary = ("Selected recorded PASS: " + ", ".join(passers) + "." if passers else
                   "No selected recorded PASS; see exact outcomes and dated diagnostic evidence below.")
        lines.append("| " + link(root, page, item["title"], PROBLEM_PAGE.as_posix(), item["task"]) +
                     f" | {cell(item['intent'])} | {cell(summary)} |")
    for item in definitions:
        task = item["task"]
        lines += ["", f'<a name="{task}"></a>', "", f"## {item['title']}", "",
                  f"Test: `{task}` · " + ("Held for qualification; included in full-suite results." if task in held else
                                          "Required for qualification and included in full-suite results."), "",
                  "**What it tests.** " + item["intent"], "",
                  "**Historical diagnosis" + (" (" + registry["evidence_date"] + ")" if registry.get("evidence_date") else "") +
                  ".** " + item["problem"], "",
                  "**Historical passer evidence.** " + item["pass_summary"], ""]
        lines += ["- " + passer for passer in item["passers"]]
        corrections = {}
        for _, cohort in selected:
            entry = cohort["tasks"].get(task, {})
            correction = entry.get("evidence_correction") or entry.get("run_attempt", {}).get("evidence_correction")
            if correction:
                corrections[correction["id"]] = correction
        for correction in corrections.values():
            lines += ["", "**Scope correction for retained diagnostic observations.** " +
                      correction["reason"] + " Original outcomes remain archived; they do not establish the "
                      "declared unassisted recipe/task claim. " +
                      link(root, page, "Exact observation identities", correction["registry"]) + "."]
        lines += ["", "**Selected recipes' recorded results:**", "",
                  "| Selected recipe | Result |", "| --- | --- |"]
        for family, cohort in selected:
            result = cohort["tasks"].get(task, {})
            status = result.get("status", "UNKNOWN")
            result_label = "NOT MEASURED" if status == "UNKNOWN" else status
            lines.append("| " + link(root, page, family["label"], family["page"], cohort["anchor"]) +
                         " | " + link(root, page, result_label, family["page"], cohort["anchor"] + "-experiment-" + task) + " |")
        lines += ["", "**Recommendation from the dated diagnosis.** " + item["next_step"], "",
                  "Evidence: " + " · ".join(link(root, page, ref["label"], ref["path"], ref.get("anchor"))
                                               for ref in item["evidence"]) + ".", ""]
    return "\n".join(lines)


def _views_cell(memberships, summed):
    tiers = {tier for _, tier in memberships}
    if len(summed) > 1 and len(tiers) == 1 and sorted(view for view, _ in memberships) == sorted(v["id"] for v in summed):
        return f"all {len(summed)} views (Tier {tiers.pop()})"
    return ", ".join(f"{cell(view)} (Tier {tier})" for view, tier in memberships)


def _diagnostic_passers(selected, task):
    """Selected recipes whose separate diagnostic run passes a task their recorded result does not."""
    return [family["label"] for family, cohort in selected
            if cohort["tasks"][task]["status"] != "PASS"
            and cohort["tasks"][task].get("run_attempt", {}).get("status") == "PASS"
            and not (cohort["tasks"][task].get("evidence_correction")
                     or cohort["tasks"][task].get("run_attempt", {}).get("evidence_correction"))]


def _uncounted_disclosure(summed, selected):
    """What holding changed and what it did not, computed from recorded statuses."""
    counted = [a["task"] for view in summed for a in view["assignments"] if a["importance"] == "required"]
    uncounted = [
        a for view in summed for a in view["assignments"]
        if a.get("importance") == "held" or "removal" in a
    ]
    if not uncounted:
        return []
    passes = sum(cohort["tasks"][a["task"]]["status"] == "PASS" for _, cohort in selected for a in uncounted)
    if passes:
        sentences = [f"Selected recipes' recorded results pass {passes} of the held or removed view requirements; "
                     "those passes stay visible here and on the family pages but no longer count."]
    else:
        sentences = ["No new test was passed. No selected recipe's recorded result passes a held or removed view "
                     "requirement, so every pass count is unchanged; the totals shrank only because those requirements "
                     "left the count."]
    diagnostic = {name for a in uncounted for name in _diagnostic_passers(selected, a["task"])}
    if diagnostic:
        sentences.append(f"Separate diagnostic runs of {len(diagnostic)} selected "
                         + ("recipe pass" if len(diagnostic) == 1 else "recipes pass")
                         + " some of these tests (named in the table); a diagnostic run never becomes a recorded "
                           "result, so those passes count nowhere.")
    for family, cohort in selected:
        status = {name: result["status"] for name, result in cohort["tasks"].items()}
        failed = {("held" if is_stretch(a) else a["importance"]) for view in summed for a in view["assignments"]
                  if status[a["task"]] == "FAIL" and (a["importance"] in COUNTED_OR_HELD or "removal" in a)}
        if "required" in failed or "held" not in failed:
            continue
        passed = sum(status[name] == "PASS" for name in counted)
        others = Counter(status[name] for name in counted if status[name] != "PASS")
        label = cell(family["label"])
        text = f"{label}'s recorded measurement fails no counted test: {passed}/{len(counted)} pass"
        rest = len(counted) - passed
        if rest:
            if set(others) == {"UNKNOWN"}:
                detail = ("has" if rest == 1 else "have") + " no recorded result yet (UNKNOWN)"
            else:
                detail = ("is " if rest == 1 else "are ") + " and ".join(
                    f"{value} {name}" if len(others) > 1 else name for name, value in sorted(others.items()))
            text += f" and the other {rest} {detail}"
            passed_elsewhere = sorted({name for name in counted if status[name] != "PASS"
                                       and cohort["tasks"][name].get("run_attempt", {}).get("status") == "PASS"})
            if passed_elsewhere:
                text += (f"; a separate diagnostic run passes {', '.join(passed_elsewhere)}, but a diagnostic run "
                         "never becomes a recorded result")
        text += ". Every test it fails is " + ("held" if failed == {"held"} else "held or removed from the count") + ". "
        others_pass = {name for _, peer in selected if peer is not cohort
                       for name in counted if peer["tasks"][name]["status"] == "PASS"}
        if others_pass <= {name for name in counted if status[name] == "PASS"}:
            text += ("That is the one-global-recipe rule at work (one recipe must pass every counted test): "
                     f"{label} passes every counted test that any other selected recipe passes. ")
        sentences.append(text + f"It is not a new measurement of {label}.")
    return [" ".join(sentences), ""]


def _held_top_tier_notes(summed):
    """A view whose highest non-stretch tier has only held tests never shows PASS."""
    counted = {view["id"]: sorted((a["task"], a["qualification_tier"]) for a in view["assignments"]
                                  if a["importance"] == "required") for view in summed}
    notes = []
    for view in summed:
        non_stretch_tiers = [a["qualification_tier"] for a in view["assignments"] if a["importance"] != "stretch"]
        if not non_stretch_tiers:
            continue
        top = max(non_stretch_tiers)
        highest = [a for a in view["assignments"] if a["qualification_tier"] == top]
        held = sorted(a["task"] for a in highest if a["importance"] == "held")
        if not held or any(a["importance"] == "required" for a in highest):
            continue
        if not counted[view["id"]]:
            # Nothing counted: name every held test of the view, not only its top tier.
            held = sorted(a["task"] for a in view["assignments"] if a["importance"] == "held")
        names, verb = ", ".join(held), "is" if len(held) == 1 else "are"
        same = [other["id"] for other in summed if other is not view and counted[other["id"]] == counted[view["id"]]]
        rule = "a recipe that passes every counted tier gets HELD, and a counted failure still shows as a failure."
        if not counted[view["id"]]:
            notes.append(f"Nothing in {view['id']} is counted while {names} {verb} held, so it reports HELD for every recipe.")
        elif same:
            notes.append(f"While {names} {verb} held, {view['id']} counts the same tests as {' and '.join(same)}. "
                         f"Its highest tier, Tier {top}, has only held tests, so it never shows PASS: " + rule)
        else:
            notes.append(f"While {names} {verb} held, the highest tier of {view['id']}, Tier {top}, has only held "
                         "tests, so the view never shows PASS: " + rule)
    return [" ".join(notes), ""] if notes else []


def _uncounted_section(root, page, summed, selected):
    entries = uncounted_entries(summed)
    if not entries:
        return []
    records = sorted({entry["note"]["explanation"].partition("#")[0] for entry in entries if "explanation" in entry["note"]})
    record = (" in " + link(root, page, "the decision record", records[0]) + ", which also defines the words used."
              if len(records) == 1 else ".")
    lines = ["## Held and removed tests", "",
             "A **held** test is still declared, run and shown with its history. It is excluded from the qualification "
             "totals above, but included in the full-suite results. It counts toward qualification again once its re-entry "
             "rule is met. A test **removed from the "
             "count** stays in its views as a diagnostic, because another counted test already carries its question "
             "or it cannot run as declared. Rows come from the notes stored in the view declarations; each details "
             "link opens that test's explanation" + record, "",
             *_uncounted_disclosure(summed, selected), *_held_top_tier_notes(summed),
             "| Test | Role and reason | Views (tier) | Summary | Re-entry rule | Selected recipes passing (recorded result) | Details |",
             "| --- | --- | --- | --- | --- | ---: | --- |"]
    for entry in entries:
        note = entry["note"]
        passing = f"{sum(cohort['tasks'][entry['task']]['status'] == 'PASS' for _, cohort in selected)} of {len(selected)}"
        diagnostic = _diagnostic_passers(selected, entry["task"])
        if diagnostic:
            passing += "; separate diagnostic runs only: " + ", ".join(diagnostic)
        if "explanation" in note:
            path, _, anchor = note["explanation"].partition("#")
            details = link(root, page, "explanation", path, anchor)
        else:
            details = "—"
        lines.append("| " + " | ".join([
            "`" + cell(entry["task"]) + "`", cell(role_label(entry["assignment"])), _views_cell(entry["views"], summed),
            cell(note["summary"]), cell(note.get("reentry", note.get("graduation", "—"))), cell(passing),
            details]) + " |")
    return lines + [""]


def _stretch_section(root, page, progress, selected, report_tiers):
    summed = _summed_views(progress)
    stretch_assignments = [a for view in summed for a in view["assignments"] if is_stretch(a)]
    if not stretch_assignments:
        return []
    stretch_total = len(stretch_assignments)
    stretch_distinct = len({a["task"] for a in stretch_assignments})
    cols = " | ".join(f"Tier {t} stretch" for t in report_tiers)
    div = " | ".join(["---"] + ["---:"] * (len(report_tiers) + 1))
    lines = [
        "## Stretch experiments", "",
        "Stretch experiments (extra-credit, optional, frontier, and held benchmark tasks) are assigned to a tier, "
        "executed, and graded under the same deterministic protocol as required tasks, but never block tier "
        f"progression or enter required denominators: **{stretch_total} stretch view assignments** across "
        f"**{stretch_distinct} distinct stretch test{'' if stretch_distinct == 1 else 's'}**.", "",
        f"| Family / selected recipe | {cols} | Total stretch |",
        f"| {div} |",
    ]
    for family, cohort in selected:
        name = family["label"]
        if cohort["backend"] != "cuda" or (len(family["cohorts"]) > 1 and not family.get("current_cohort_anchor")):
            name += " (" + cohort["backend"] + ")"
        label = link(root, page, name, family["page"])
        if cohort.get("configuration_family"):
            configuration = next(item for item in progress.get("configuration_families", [])
                                 if item["id"] == cohort["configuration_family"])
            label += " · " + link(root, page, configuration["label"], configuration["page"])
        tier_counts = {
            t: _count([cohort["tasks"][a["task"]] for view in summed for a in view["assignments"]
                       if is_stretch(a) and str(a["qualification_tier"]) == t])
            for t in report_tiers
        }
        total_count = _sum(list(tier_counts.values()))
        cells = [label]
        for t in report_tiers:
            cells.append(link(root, page, score(tier_counts[t]), family["page"], f"{cohort['anchor']}-tier-{t}"))
        cells.append(link(root, page, score(total_count), family["page"], f"{cohort['anchor']}-held"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def _score_cells(root, page, family, cohort, counts, *, tiers=None, view_id=None, bold=False):
    if tiers is None:
        tiers = tuple(counts["tiers"].keys()) if "tiers" in counts else TIERS
    base = cohort["anchor"] + ("-" + view_id if view_id else "")
    cells = []
    for tier in (*tiers, "total"):
        count = counts["total"] if tier == "total" else counts["tiers"][tier]
        target = base + ("-tier-" + tier if tier != "total" else "")
        text = link(root, page, score(count), family["page"], target)
        cells.append("**" + text + "**" if bold else text)
    return cells


def render_leaderboard(root: Path, publication: dict, page: Path) -> str:
    progress = publication["family_progress"]
    report_tiers = tuple(progress.get("tiers", TIERS))
    from .research_tiers import tier_rows, historical_seed_rows
    research = tier_rows(root, page)
    historical_seeds = historical_seed_rows(root, page)
    summed = _summed_views(progress)
    assignments = [a for view in summed for a in view["assignments"]]
    counted = sum(a["importance"] == "required" for a in assignments)
    distinct = len({a["task"] for a in assignments if a["importance"] == "required"})
    roles = [(sum(a["importance"] == "stretch" for a in assignments),
              len({a["task"] for a in assignments if a["importance"] == "stretch"}), "stretch"),
             (sum(a["importance"] == "held" for a in assignments),
              len({a["task"] for a in assignments if a["importance"] == "held"}), "held"),
             (sum("removal" in a for a in assignments), len({a["task"] for a in assignments if "removal" in a}), "removed")]
    uncounted = " and ".join(f"{value} {label}" for value, _, label in roles if value)
    uncounted_tests = " and ".join(f"{tests} {label} test" + ("" if tests == 1 else "s") for _, tests, label in roles if tests)
    lines = ["# Forge family leaderboard", "",
             "A **family** can have several **recipes**. Each score belongs to one exact recipe and source. "
             "For example, BCAP's selected recipe uses DualNorm; Confidence LR, Finite critic cap and Positive evidence "
             "are separate BCAP research recipes, with their own scores.", "",
             "Click a count for its gates, measurements and evidence. Required lower tiers must pass before advancing.", "",
             "Counts describe screening and selection evidence. Scientific superiority or public-default adoption "
             "requires report-bound accepted calibration and independently frozen confirmation evidence. Software "
             "repairs grant neither; unmeasured confirmation remains unmeasured. Tuning effort and task holdout "
             "disclosures accompany the full-suite comparison below.", "",
             "## Selected recipes", "",
             "Passes / required-view instances for one selected recipe per family. Shared tests count once in every view "
             f"that requires them: **{counted} means {counted} counted view requirements, including repeated tests**, "
             f"from {distinct} distinct test" + ("" if distinct == 1 else "s") + ". " +
             (uncounted + f" view assignments ({uncounted_tests}) are still measured and listed under "
              "**Held and removed tests** below. " if uncounted else "") +
             "These totals project recorded task outcomes onto the current requirements; historical qualifications "
             "keep their original policy revisions. Held failures remain visible in the "
             "full-suite results below; separate diagnostic runs appear below.", ""]
    selected = list(_selected_cohorts(progress))
    from .publication_integrity import CORRECTIONS
    if (root / CORRECTIONS).is_file():
        lines += ["**Scientific scope corrections (2026-11-08):** assisted or recipe-adapted coverage remains "
                  "archived diagnostic evidence under its exact original identity. It does not replace an ordinary "
                  "selected-cohort grade. Constructor refusals, blocked dependencies, execution errors and missing "
                  "measurements retain their recorded status and earn no pass credit. " +
                  link(root, page, "Source-bound correction registry", CORRECTIONS.as_posix()) + ".", ""]
    lines += _table_header(tiers=report_tiers, selected=True)
    for family, cohort in selected:
        name = family["label"]
        if cohort["backend"] != "cuda" or (len(family["cohorts"]) > 1 and not family.get("current_cohort_anchor")):
            name += " (" + cohort["backend"] + ")"
        cells = ["**" + link(root, page, name, family["page"]) + "**"]
        if cohort.get("configuration_family"):
            configuration = next(item for item in progress.get("configuration_families", [])
                                 if item["id"] == cohort["configuration_family"])
            cells[0] += " · " + link(root, page, configuration["label"], configuration["page"])
        cells += _score_cells(root, page, family, cohort, cohort, tiers=report_tiers, bold=True)
        lines.append("| " + " | ".join(cells) + " |")
    lines += [""]
    for row in publication.get("rows", []):
        actual = row.get("executed_policy_qualification")
        if actual:
            lines += [f"**{row['technique']}:** comparison policy revision {publication['view_revision']}, "
                      f"tier {row.get('qualified_tier', 0)}; actual executed-policy qualification "
                      f"**tier {actual['qualified_tier']}**, revision {actual['view_revision']} "
                      f"({actual['status']}). These are separate policy results; no qualification reuse.", ""]
    declared = {view["id"]: view["revision"] for view in progress.get("views", [])}
    recorded = publication.get("view")
    if recorded in declared and declared[recorded] != publication.get("view_revision"):
        lines += ["Counts follow the declared view revisions: " +
                  ", ".join(f"{view['id']} {view['revision']}" for view in summed) +
                  f". The recorded qualification keeps {recorded} revision {publication['view_revision']} "
                  "until new ordinary evidence is registered; no recorded verdict was regraded.", ""]
    if (root / "reports/forge/attempt-coverage.json").is_file():
        attempts = progress.get("run_attempt_coverage", {})
        lines += [f"Backfill coverage: **{attempts.get('attempted', 0)}/{attempts.get('total', 0)} distinct "
                  f"recipe/test pairs attempted (selected and research)**; {attempts.get('pending', 0)} pending and {attempts.get('running', 0)} running. " +
                  link(root, page, "Outcomes, blocked attempts and setup", "reports/forge/ATTEMPT_COVERAGE.md") +
                  ". This counts attempts, including start refusals, rather than passes or qualification.", ""]
    lines += _full_suite_section(root, publication, page, selected)
    lines += _stretch_section(root, page, progress, selected, report_tiers)
    lines += _uncounted_section(root, page, summed, selected)
    from .model_toy_publication import board_section
    lines += board_section(root, page)
    model_diagnostics = []
    for model in ("vae", "ddgan"):
        readout = f"reports/forge/{model}-failure-investigation/README.md"
        if (root / readout).is_file():
            model_diagnostics.append(link(root, page, f"{model.upper()} saved-state investigation", readout))
    if model_diagnostics:
        lines += ["Saved-state investigations: " + "; ".join(model_diagnostics)
                  + ". These diagnostics add no training, qualification or replacement for the recorded failures.", ""]
    from .scaling_publication import performance_section
    lines += performance_section(root, publication, page)
    if any(research.values()):
        declared_view = publication.get("declared_view", {})
        no_longer = sum(len(names) for names in declared_view.get("no_longer_required_tasks", {}).values())
        scale = (f"Those assignments still count the {no_longer} tests that {publication['view']} revision "
                 f"{declared_view['revision']} holds or removes from the count "
                 f"({sum(len(names) for names in publication['tier_requirements'].values())} tests per row, against "
                 f"{sum(len(names) for names in declared_view['tier_requirements'].values())} counted in revision "
                 f"{declared_view['revision']}), so research rows are not on the same scale as the selected-recipe "
                 "totals. " if no_longer else "")
        lines += ["## Research recipes and separate diagnostics", "",
                  "Each row names its owning family and its own recipe or diagnostic scope. "
                  "Counts are **passes / distinct required tests within that row**: 29 means 29 different tests. "
                  "These scores are excluded from the selected-recipe totals above. "
                  "Compare rows only when their test, source and sampling contracts match.", "",
                  "Research recipes use their recorded revision 9 assignments. " + scale + "Atlas selected-cloud has a separate "
                  "seven-question Tier 1 policy scope. Same-source backfill results fill previously unmeasured research "
                  "cells while retaining original verdicts and qualification. Historical seed-only studies are "
                  "retained in the expandable section below and excluded from this backfill.", "",
                  "| Family | Recipe / diagnostic scope | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for family in progress["families"]:
            for row in research.get(family["id"], []):
                lines.append("| " + link(root, page, family["label"], family["page"]) + " | " + row.removeprefix("| "))
        lines += [""]
    if any(historical_seeds.values()):
        lines += ["<details>", "<summary>Historical seed-only studies (excluded from backfill)</summary>", "",
                  "These 12 archived seed 1–6 rows retain **324 unrun recipe/test pairs**. "
                  "They are excluded by the repository instruction against seed-only experiments. "
                  "Their original counts and evidence remain visible here; they are outside the completed backfill scope.", "",
                  "| Family | Historical seed-only recipe | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for family in progress["families"]:
            for row in historical_seeds.get(family["id"], []):
                lines.append("| " + link(root, page, family["label"], family["page"]) + " | " + row.removeprefix("| "))
        lines += ["", "</details>", ""]
    held = any("held" in view for _, cohort in selected for view in cohort["views"])
    lines += ["<details>", "<summary>Selected recipe breakdown by view (shared tests repeat across views)</summary>", "",
              "Each view uses the selected recipe above. View scores are the requirements summed in its selected-recipe total." +
              (" Held cells count passes / held tests of that view; they are in no total." if held else ""), "",
              f"| Family / selected recipe | View | {' | '.join(f'Tier {t}' for t in report_tiers)} | Total view requirements |" +
              (" Held (measured, not counted) |" if held else ""),
              f"| --- | --- | {' | '.join(['---:'] * len(report_tiers))} | ---: |" + (" ---: |" if held else "")]
    for family, cohort in selected:
        for view in cohort["views"]:
            label = "↳ " + link(root, page, view["id"], family["page"], cohort["anchor"] + "-" + view["id"])
            lines.append("| " + " | ".join([link(root, page, family["label"], family["page"]), label,
                                           *_score_cells(root, page, family, cohort, view, tiers=report_tiers, view_id=view["id"]),
                                           *([_held_cell(root, page, family, cohort, view)] if held else [])]) + " |")
    lines += ["", "</details>", "",
              "## Reading the results", "",
              "Counts show passes / required tasks; selected-row pass counts remain unchanged. For rows with run-attempt coverage, the suffix counts "
              "genuinely unattempted tasks as not run, constructor refusals as incompatible, missing required artifacts "
              "as blocked, and other separate-runtime attempts as diagnostic attempts. A start refusal is not a "
              "numerical failure. Rows without that registry retain their recorded missing/preflight counts. "
              "No formulation is changed to fit a test. "
              "Recorded PASS and FAIL are both executed results. Attempted ERROR, INVALID "
              "or INCOMPLETE results show their cause and execution evidence on the family page and earn no pass credit. "
              "Zero recorded passes always displays as 0, including unrun families.", "",
              "Counts retain each original recipe, prior, initialization, budget, serving law and source. "
              "Family pages retain exact metrics, phase checks, failures, blockers, historical comparisons and "
              "calibration requirements. Observation checks and catalog attempt counts add no benchmark passes; "
              "untiered historical studies remain on those detail pages.", ""]
    if (root / "reports/forge/actual-missing-diagnostics-20261014/results.json").is_file():
        lines += [link(root, page, "Actual attempts of unrun and blocked tests",
                       "reports/forge/actual-missing-diagnostics-20261014/README.md") +
                  ". The new diagnostic rows show executions beyond failed tier gates and Atlas/E22 public-policy tests. "
                  "Atlas's original 19/19 CUDA passes remain valid under their original task and sampling contracts.", ""]
    if (root / "reports/forge/tier-coverage-20261014/results.json").is_file():
        lines += [link(root, page, "Diagnostic completion, new strict audit and every skip reason",
                       "reports/forge/tier-coverage-20261014/README.md") + ". "
                  "The selected-recipe Tier 2 diagnostic rows retain their original 21-task scope. "
                  "The new strict BCAP audit has a separate runtime and changes no qualification.", ""]
    if (root / "reports/forge/bcap-prior-search/findings.md").is_file():
        lines += [link(root, page, "Prior-search measurements and gates", "reports/forge/bcap-prior-search/README.md") + " · " +
                  link(root, page, "Prior-search findings", "reports/forge/bcap-prior-search/findings.md"), ""]
    lines += _tag_directory(root, page, progress)
    lines += ["## Refresh", "", "```sh", "python reports/forge/regenerate_technique_inventory.py", "```", "",
              "This regenerates the leaderboard, family pages and experiments-by-tier report from committed evidence "
              "and declarations. Register newly measured evidence with `--source-commit <executed-commit>`; "
              "advancing the recorded view policy also requires `--advance-policy`.", "",
              "After any view revision (for example a held or removed test), always add `--refresh-publication`. "
              "It rebuilds the counts and pages from the declared views and keeps every recorded verdict and the "
              "recorded qualification. Without the flag, the command switches to that mode only when "
              f"{publication.get('view', 'the registered view')} itself changed; any other view change would take "
              "the ordinary publishing path, which re-reads earlier sources from Git.", "",
              link(root, page, "Experiments, criteria and tier assignments", "reports/forge/EXPERIMENTS_BY_TIER.md") + " · " +
              link(root, page, "Complete numerical publication and provenance", "reports/forge/technique-inventory.json"), "",
              f"Publication input digest `{publication['provenance']['input_digest']}`.", ""]
    return "\n".join(lines)


def _tag_directory(root, page, progress):
    families = progress["families"] + progress.get("configuration_families", []) + progress.get("historical_families", [])
    used = {tag for family in families for tag in family.get("tags", [])}
    if not used:
        return []
    lines = ["## Browse by technique tag", "",
             "Tags describe properties shared by a family's supported variants, not performance or qualification. "
             "Configuration-dependent behavior is explained on each family page. Historical pages retain their own scope.", ""]
    for tag in sorted(used):
        lines += [f'<a name="tag-{tag}"></a>', "", f"### `{tag}`", "",
                  progress["tag_definitions"][tag], ""]
        for family in sorted(families, key=lambda item: (item["label"], item["id"])):
            if tag in family.get("tags", []):
                lines.append("- " + link(root, page, family["label"], family["page"]) +
                             (" (historical cohort)" if family.get("historical_only") else ""))
        lines.append("")
    return lines


def _technique_description(root, page, family):
    documentation = family.get("documentation")
    if documentation is None:
        return ["## Technique overview", "",
                "Maintained technique documentation is unavailable in this historical or synthetic report source. "
                "Consult the recorded recipe and source bindings below; no training behavior is inferred from its name.", ""]
    lines = ["**Tags:** " + " · ".join(link(root, page, tag, "reports/forge/technique-inventory.md", "tag-" + tag)
                                       for tag in family["tags"]), "",
             "## Technique overview", "", documentation["overview"], ""]
    illustration = documentation.get("illustration")
    if illustration is not None:
        alt = illustration["alt"].replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")
        target = os.path.relpath(root / illustration["path"], page.parent)
        if any(character.isspace() for character in target):
            target = "<" + target + ">"
        lines += [f"![{alt}]({target})", "", illustration["caption"], ""]
    equations = documentation.get("equations", [])
    if equations:
        lines += ["## Mathematical formulation", "", documentation["symbols"], ""]
        for equation in equations:
            lines += ["**" + equation["label"] + "**", "", "$$", equation["latex"], "$$", "",
                      equation["explanation"], ""]
        lines += ["## Simplified pseudocode", ""]
    else:
        lines += ["## Simplified pseudocode", "", documentation["symbols"], ""]
    lines += ["```text", *documentation["pseudocode"], "```", ""]
    if not family.get("configuration_selection"):
        lines += ["## Training details", "", "| Characteristic | Behavior |", "| --- | --- |"]
        lines += ["| " + label + " | " + cell(documentation["training_details"][key]) + " |"
                  for key, label in family_documentation.TRAINING_DETAILS]
    lines += ["", "## Configuration differences", ""]
    lines += ["- " + note for note in documentation["configuration_notes"]]
    lines += ["", "<details>", "<summary>Implementation and recipe sources</summary>", "",
              "These links support the explanation. Recorded results below remain bound to their own executed source.", ""]
    for source in documentation["sources"]:
        lines.append("- " + (f"[{source}]({source})" if "://" in source else link(root, page, source, source)))
    lines += ["", "</details>", ""]
    return lines


def _technique_references(family):
    documentation = family.get("documentation")
    if documentation is None:
        return []
    lines = ["## References", ""]
    if not documentation["references"]:
        lines += ["This technique is repository-specific; no dedicated paper reference is declared. "
                  "Its implementation and recipe sources are linked above.", ""]
    for reference in documentation["references"]:
        lines.append("- [" + reference["title"] + "](" + reference["url"] + "). " + reference["component"])
    if documentation["references"]:
        lines.append("")
    return lines


def _existing_link(root, page, label, target):
    return link(root, page, label, target) if target and (root / target).is_file() else cell(label)


def _current_gate(evaluation):
    bounds = evaluation_rows(evaluation)
    lines = ["| Metric | Required bound |", "| --- | --- |"] if bounds else []
    lines += [f"| {cell(name)} | {cell(op)} {number(bound)} |" for name, op, bound in bounds]
    if lines:
        lines.append("")
    lines += evaluation_notes(evaluation)
    return lines


def _heading(level, title, anchor):
    return [f'<a name="{anchor}"></a>', "", "#" * level + " " + title, ""]


def _metric_values(metrics, prefix=""):
    for name, value in sorted(metrics.items()):
        key = prefix + name
        if isinstance(value, dict):
            yield from _metric_values(value, key + ".")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            yield key, value


def _selected_configuration_details(root, page, publication, family, cohort, row):
    if not family.get("configuration_selection"):
        return []
    explicit = (family.get("configuration_selection") == "explicit_current_configuration"
                and cohort["anchor"] == family.get("current_cohort_anchor"))
    lines = ["## " + ("Current benchmark configuration" if explicit else "Best recorded configuration"), "",
             ("Uses the explicitly selected family configuration, regardless of alternative pass counts. " if explicit else
              "Selected by recorded required passes, then completed measurements. ") +
             "Each count comes from this one complete configuration. Source differences preserve separate evidence "
             "contracts; the selection does not establish a controlled win or default adoption.", ""]
    recipe = publication.get("recipe_contracts", {}).get(row.get("bindings", {}).get("recipe_sha256"))
    if recipe is not None:
        fields = (
            ("Adversarial loss", ("loss", "loss_labels")),
            ("Optimizer", ("optimizer_family", "optimizer_momentum", "optimizer_smoothing", "optimizer_adam_lr", "adam_variant")),
            ("Adam parameters (when used)", ("betas", "d_betas", "prior_betas", "eps", "d_eps", "prior_eps")),
            ("Learning rates", ("lr", "d_lr_mult", "prior_lr_mult")),
            ("Rate schedule", ("lr_schedule", "lr_floor", "network_lr_floor", "network_lr_horizon_cap")),
            ("Critic penalty", ("reg_arm", "reg_coeff", "reg_kappa", "reg_every", "reg_anchor_weight")),
            ("Damping and guards", ("d_guard_ratio", "latent_damping_max_rate", "direct_particle_gain")),
            ("Training noise", ("input_noise_std", "output_noise_std", "output_noise_mode")),
            ("Averaging", ("ema_decay", "serve_average")),
        )
        lines += ["Recorded trainer recipe; task-owned architecture, prior, initialization, budget and sampling "
                  "remain in the experiment receipts below. Null role overrides inherit the shared value. "
                  "Optimizer parameters only apply to optimizers that consume them.", "",
                  "| Setting | Selected value |", "| --- | --- |"]
        for label, keys in fields:
            optimizer = recipe.get("optimizer_family", "adam")
            if label.startswith("Adam parameters") and optimizer not in {"adam", "ada_nsgda", "dualnorm_d_only", "particle_rownorm_only"}:
                continue
            if label == "Optimizer" and optimizer not in {"adam", "ada_nsgda", "dualnorm_d_only", "particle_rownorm_only"}:
                keys = ("optimizer_family", "optimizer_momentum", "optimizer_smoothing")
            values = "; ".join(f"{key}={recipe[key]}" for key in keys if key in recipe)
            if values:
                lines.append("| " + label + " | " + cell(values) + " |")
        lines.append("")
        if recipe.get("lr") is not None:
            lr = recipe["lr"]
            lines += [f"Base network rate: **{lr:g}**; critic rate: **{lr * recipe.get('d_lr_mult', 1):g}**; "
                      f"prior rate: **{lr * recipe.get('prior_lr_mult', 1):g}** before any declared schedule or host adaptation.", ""]
    documentation = cohort.get("configuration_documentation")
    if documentation:
        lines += ["Selected optimizer rule: " + documentation["training_details"]["optimizer"], ""]
    lines += ["<details>", "<summary>Other recorded configurations in this family</summary>", "",
              "These are whole configurations under their original sources. Their individual passing cells "
              "do not contribute to the selected result.", "",
              "| Configuration | Required passes | Executed source | Display selection |", "| --- | ---: | --- | --- |"]
    rows = publication["rows"] + publication.get("historical_family_rows", [])
    for alternative in cohort["configuration_alternatives"]:
        candidate = rows[alternative["row_index"]]
        label = candidate["candidate_id"].rsplit("--", 1)
        label = " · ".join([label[0], label[1][:12]]) if len(label) == 2 else label[0]
        lines.append("| " + " | ".join([
            link(root, page, label, alternative["page"]), score(alternative["total"]),
            "`" + str(candidate.get("bindings", {}).get("source_digest", "unbound"))[:12] + "`",
            "Selected" if alternative["selected"] else "Alternative"]) + " |")
    return lines + ["", "</details>", ""]


def _displayed_for(view, family_id):
    """A scoped view may name the families whose pages display it; others show everywhere."""
    families = view.get("reporting", {}).get("families")
    return families is None or family_id in families


def render_family(root: Path, publication: dict, family: dict) -> str:
    page = root / family["page"]
    progress = publication["family_progress"]
    report_tiers = tuple(progress.get("tiers", TIERS))
    views = {view["id"]: view for view in progress["views"]}
    lines = ["<!-- Generated Forge family report -->", "", f"# {family['label']}", "",
             link(root, page, "← Family leaderboard", "reports/forge/technique-inventory.md"), "",
             *_technique_description(root, page, family),
             "Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific "
             "contracts; grouping them under current views grants no new qualification.", ""]
    if family.get("historical_only"):
        lines += ["**Historical cohort navigation.** This original recipe/prior row is retained for existing links. " +
                  link(root, page, "Current GAN v3 solution family", FAMILY_DIRECTORY / "release07-gan-v3.md") +
                  " uses one whole selected configuration and task-declared priors; these historical cells are not pooled into it.", ""]
    if family.get("solution_page"):
        lines += ["**Configuration detail for " + link(root, page, family["solution_label"], family["solution_page"]) +
                  ".** Its optimizer or settings do not create a separate solution family. "
                  "This page preserves the original configuration evidence and diagnostics.", ""]
    if family["id"] == "atlas":
        lines += _full_original_atlas_status(root, page, progress)
    if publication.get("scaling_performance") and not family.get("historical_only"):
        from .scaling_publication import render_performance
        lines += render_performance(root, publication, page, family_id=family["id"])
    declared = views.get(publication["view"], {}).get("revision", publication["view_revision"])
    declared_note = ("" if declared == publication["view_revision"] else
                     " Their counts follow the declared view revisions (" +
                     ", ".join(f"{view['id']} {view['revision']}" for view in _summed_views(progress)) +
                     f"); the recorded qualification keeps {publication['view']} revision {publication['view_revision']} "
                     "until new ordinary evidence is registered.")
    # Stretch and held tests stay listed with this family's recorded result; diagnostic-only views count nothing.
    held_entries = [entry for entry in uncounted_entries([view for view in progress["views"]
                                                          if view["id"] not in progress["diagnostic_views"]])
                    if is_stretch(entry["assignment"])]
    for cohort in family["cohorts"]:
        row = (publication["rows"] + publication.get("historical_family_rows", []))[cohort["row_index"]]
        base = cohort["anchor"]
        held_views = any("held" in view for view in cohort["views"])
        config = Path("configs/forge/configurations") / (row["candidate_id"] + ".json")
        if not (root / config).is_file():
            config = Path("configs/forge/ideas") / (row["candidate_id"] + ".json")
        label = row["candidate_id"].rsplit("--", 1)
        configuration_label = " · ".join([label[0], label[1][:12]]) if len(label) == 2 else label[0]
        source = row.get("bindings", {}).get("source_digest")
        pointer = publication.get("evidence_sources", {}).get(row.get("publication_key"), {})
        # publish_current validates this snapshot's complete identity before
        # rendering, then writes a pending snapshot alongside the pages. Its
        # first-write link must not depend on that later filesystem mutation.
        frozen_evidence = (link(root, page, "Frozen numerical evidence", pointer["snapshot"])
                           if pointer.get("snapshot") and pointer.get("json_sha256") else
                           _existing_link(root, page, "Frozen numerical evidence", pointer.get("snapshot")))
        current_anchor = family.get("current_cohort_anchor")
        cohort_label = ("Current benchmark" if cohort["anchor"] == current_anchor else
                        "Archived runtime cohort") if current_anchor else cohort["backend"].upper() + " results"
        actual = row.get("executed_policy_qualification")
        qualification = (
            f"Comparison policy: {publication['view']} revision {publication['view_revision']}, "
            f"**tier {row.get('qualified_tier', 0)}**. Actual executed-policy qualification: "
            f"**tier {actual['qualified_tier']}**, {publication['view']} revision {actual['view_revision']} "
            f"({actual['status']}). These are separate policy results; no qualification reuse."
            if actual else
            f"Recorded qualification: **tier {row.get('qualified_tier', 0)}**, "
            f"{publication['view']} revision {publication['view_revision']}. "
            "Other view rows below are navigation over recorded task evidence, not recomputed qualification." + declared_note)
        lines += [*_heading(2, cohort_label, base), f"Runtime: **{cohort['backend']}**. Selected configuration: " +
                  _existing_link(root, page, configuration_label, config) + ".", "",
                  qualification, "",
                  "<details>", "<summary>Configuration, source and runtime provenance</summary>", "",
                  f"Source digest: `{source or 'unbound'}`. Candidate revision: `{row.get('candidate_revision') or 'unbound'}`. "
                  f"Runtime cohort: `{row.get('cohort') or 'unbound'}`.", "",
                  frozen_evidence + " · " +
                  link(root, page, "Complete recipe, prior, initialization and sampling bindings", "reports/forge/technique-inventory.json"), "",
                  "Selection: " + cell(row.get("selection", {}).get("selection_kind", "canonical fallback")) + ". " +
                  cell(row.get("selection", {}).get("reason", "No qualified configuration has been selected.")), "",
                  "</details>", "", *_table_header(tiers=report_tiers, held=held_views)]
        if row.get("selection", {}).get("measurement_complete"):
            lines[-2:-2] = ["Complete current Tier 1 measurement in: " +
                            ", ".join(row["selection"]["measurement_views"]) +
                            ("; additional scoped probes: " + ", ".join(row["selection"]["measurement_tasks"])
                             if row["selection"].get("measurement_tasks") else "") +
                            ". PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.", ""]
        details = _selected_configuration_details(root, page, publication, family, cohort, row)
        if details:
            lines[-2:-2] = details
        for view in cohort["views"]:
            lines.append("| " + " | ".join([link(root, page, view["id"], family["page"], base + "-" + view["id"]),
                                            *_score_cells(root, page, family, cohort, view, tiers=report_tiers, view_id=view["id"]),
                                            *([_held_cell(root, page, family, cohort, view)] if held_views else [])]) + " |")
        if cohort.get("scoped_views"):
            held_scoped = any("held" in view for view in cohort["scoped_views"])
            lines += ["", "Separate cohort coverage (excluded from family totals):", "", *_table_header(tiers=report_tiers, held=held_scoped)]
            for view in cohort["scoped_views"]:
                lines.append("| " + " | ".join([link(root, page, view["id"], family["page"], base + "-" + view["id"]),
                                                *_score_cells(root, page, family, cohort, view, tiers=report_tiers, view_id=view["id"]),
                                                *([_held_cell(root, page, family, cohort, view)] if held_scoped else [])]) + " |")
            policy = cohort.get("policy_cloud_publication")
            if policy:
                lines += ["", "Policy-cloud cells come from the " + link(root, page, "cohort report", policy["readme"]) +
                          " (" + link(root, page, "records", policy["records"]) + "): frozen source `" +
                          policy["source_digest"][:12] + "`, candidate revision `" + policy["candidate_revision"][:12] +
                          "`, verified there to run this selected recipe. Each law is its own row; the " +
                          str(policy["shared"]) + " law-independent verdicts come from one run and appear in both. "
                          "These diagnostics give no parent-cohort credit."]
            for item in cohort.get("scoped_publications", []):
                lines += ["", link(root, page, "Frozen separate-cohort numerical evidence", item["path"]) +
                          "; source commit `" + item["source_commit"] + "`; exact scoped cohort `" +
                          cell(item.get("cohort")) + "` (runtime SHA256 `" + cell(item.get("runtime_cohort_sha256")) +
                          "`). These cells give no parent-cohort credit."]
        lines += ["", "With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; "
                  "blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. "
                  "Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. "
                  "PASS and FAIL both count as executed. Attempted errors retain their status and cause; "
                  "changes to today's test definition are shown separately from execution coverage.", ""]
        for repair in cohort.get("retry_history", []):
            lines += ["Certified execution repair: " +
                      link(root, page, repair["attempt_id"], repair["receipt"]) + " → " +
                      link(root, page, repair["superseded_by"], repair["retry_receipt"]) +
                      ". The original outcome and cost remain recorded; displayed task metrics use the certified successor.", ""]
        for tier in report_tiers:
            members, held, removed = {}, {}, {}
            for definition in progress["views"]:
                if definition["id"] in progress["diagnostic_views"] + progress.get("scoped_views", []):
                    continue
                for assignment in definition["assignments"]:
                    if str(assignment["qualification_tier"]) != tier:
                        continue
                    group = (members if assignment["importance"] == "required" else
                             held if is_stretch(assignment) else removed if "removal" in assignment else None)
                    if group is not None:
                        group.setdefault(assignment["task"], []).append((definition["id"], assignment))

            def rows(group, tail):
                return ["| " + " | ".join([
                    link(root, page, name, family["page"], base + "-experiment-" + name),
                    ", ".join(link(root, page, view_id, family["page"], base + "-" + view_id + "-tier-" + tier)
                              for view_id, _ in memberships), *tail(cohort["tasks"][name], memberships[0][1])]) + " |"
                    for name, memberships in sorted(group.items())]

            def outcome(result, _):
                return [result["status"], _definition_label(result)]

            lines += [*_heading(3, "Tier " + tier + " across views", base + "-tier-" + tier),
                      "Shared experiments appear once in this list; the family numerator/denominator count their view requirements.", "",
                      "| Experiment | Required by | Recorded result | Test definition |", "| --- | --- | --- | --- |",
                      *rows(members, outcome), ""]
            if held:
                lines += ["Held at this tier (measured and shown, not counted):", "",
                          "| Experiment | Held in | Recorded result | Test definition |", "| --- | --- | --- | --- |",
                          *rows(held, outcome), ""]
            if removed:
                lines += ["Removed from the count at this tier (still measured):", "",
                          "| Experiment | Removed in | Reason | Recorded result |", "| --- | --- | --- | --- |",
                          *rows(removed, lambda result, assignment: [role_cell(root, page, assignment), result["status"]]), ""]
        if held_entries:
            lines += [*_heading(3, "Held tests (still measured)", base + "-held"),
                      "These tests keep running and keep this recipe's recorded result, but they do not count toward its "
                      "totals or qualification. Each counts again once its re-entry rule is met.", "",
                      "| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for entry in held_entries:
                lines.append("| " + " | ".join([
                    link(root, page, entry["task"], family["page"], base + "-experiment-" + entry["task"]),
                    ", ".join(link(root, page, view_id + " / Tier " + str(tier), family["page"],
                                   base + "-" + view_id + "-tier-" + str(tier)) for view_id, tier in entry["views"]),
                    cohort["tasks"][entry["task"]]["status"], role_cell(root, page, entry["assignment"]),
                    cell(entry["note"]["summary"]), cell(entry["note"].get("reentry", entry["note"].get("graduation", "—")))]) + " |")
            lines.append("")
        elif held_views:
            lines += [f'<a name="{base}-held"></a>', ""]
        for definition in progress["views"]:
            if not _displayed_for(definition, family["id"]):
                continue
            view_id = definition["id"]
            anchor = base + "-" + view_id
            diagnostic = view_id in progress["diagnostic_views"]
            separate_cohort = view_id in progress.get("scoped_views", [])
            lines += [*_heading(2, view_id, anchor), "**" + view_id + f" — revision {definition['revision']}**. " +
                      _existing_link(root, page, "View declaration", Path("configs/forge/views") / (view_id + ".json")) + ".", ""]
            if diagnostic:
                lines += ["Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.", ""]
            else:
                if separate_cohort:
                    lines += ["Separately scoped cohort. This ordinary lane retains its own required gates and execution "
                              "policy; its measurements are excluded from family totals and give no parent-cohort credit.", ""]
                def per_tier(role):
                    match = is_stretch if role == "held" else (lambda a: a["importance"] == role)
                    return [sum(match(a) and str(a["qualification_tier"]) == tier
                                for a in definition["assignments"]) for tier in report_tiers]

                requirements, held_counts = per_tier("required"), per_tier("held")
                tier_label = " / ".join(report_tiers)
                text = ("Qualification requires every required experiment to pass, with all lower tiers and "
                        f"task dependencies passed first. Required counts (Tier {tier_label}): **{' / '.join(map(str, requirements))}**.")
                if any(held_counts):
                    text += f" Held counts (measured and shown, not counted): **{' / '.join(map(str, held_counts))}**."
                    non_stretch_tiers = [a["qualification_tier"] for a in definition["assignments"] if a["importance"] != "stretch"]
                    top = max(non_stretch_tiers, default=0)
                    has_legacy_held = any(a["importance"] == "held" for a in definition["assignments"])
                    if not any(requirements) and has_legacy_held:
                        text += " Nothing here is counted while these tests are held, so the view reports HELD."
                    elif top and not requirements[top - 1] and any(a["importance"] == "held" and a["qualification_tier"] == top for a in definition["assignments"]):
                        text += (f" Its highest tier, Tier {top}, has only held tests, so while they are held the view "
                                 "never shows PASS: a recipe that passes every counted tier gets HELD.")
                lines += [text, ""]
            calibration = definition.get("calibration", {})
            lines += ["Calibration: **" + cell(calibration.get("status", "undeclared")) + "**. " +
                      cell(calibration.get("adoption_blocker", "Calibration and robustness are separate from recorded task passes.")), ""]
            eligibility = definition.get("eligibility", {})
            if eligibility:
                lines += ["Additional eligibility requirements:", ""]
                lines += ["- Capability: " + cell(name) for name in eligibility.get("requires_capabilities", [])]
                lines += [f"- Claim {cell(name)}: {cell(value)}" for name, value in eligibility.get("claim_contract", {}).items()]
                lines.append("")
            for tier in report_tiers:
                assignments = sorted((a for a in definition["assignments"] if str(a["qualification_tier"]) == tier),
                                     key=lambda a: (a.get("order", 0), a["task"]))
                lines += _heading(3, "Tier " + tier, anchor + "-tier-" + tier)
                if not assignments:
                    lines += ["No experiments assigned.", ""]
                    continue
                lines += ["| Experiment | Role | Recorded result | Test definition |", "| --- | --- | --- | --- |"]
                for assignment in assignments:
                    name = assignment["task"]
                    result = cohort["tasks"][name]
                    lines.append("| " + " | ".join([
                        link(root, page, name, family["page"], base + "-experiment-" + name),
                        role_cell(root, page, assignment), result["status"], _definition_label(result)]) + " |")
                lines.append("")
        lines += [*_heading(2, "Experiment metrics and pass criteria", base + "-experiments"), "One evidence entry per experiment is shared by its view rows. "
                  "Test-definition changes describe differences from the recorded run, independently of whether it was executed. "
                  "Earlier verdicts are preserved.", ""]
        for name, result in cohort["tasks"].items():
            task_path = Path(progress.get("task_paths", {}).get(name, "configs/forge/tasks/" + name + ".json"))
            task = read_json(root / task_path) if (root / task_path).is_file() else {}
            lines += [*_heading(3, name, base + "-experiment-" + name), f"**{name}: {result['status']}**. " +
                      _existing_link(root, page, "Current experiment declaration", task_path) + ".", "",
                      "Test definition: **" + _definition_label(result) + "**. " + cell(result["coverage_reason"]) +
                      " " + cell(result["reason"]), ""]
            attempt = result.get("run_attempt")
            correction = result.get("evidence_correction") or (attempt or {}).get("evidence_correction")
            if correction:
                lines += ["**Scientific scope correction:** " + cell(correction["reason"]) + " " +
                          "The original observation remains recorded below as an archived diagnostic; it does not "
                          "establish faithful execution of this selected recipe/task contract. " +
                          link(root, page, "Source-bound correction", correction["registry"]) + ".", ""]
            if attempt and attempt.get("attempted"):
                lines += ["Run attempt: **" + cell(attempt["status"]) + "** (" +
                          cell(attempt.get("origin")) + "; qualification unchanged). " +
                          cell(attempt.get("reason", "Matched existing execution receipt.")) + " " +
                          link(root, page, "Attempt evidence", attempt.get("evidence", "reports/forge/attempt-coverage.json")) + ".", ""]
            if result["status"] not in COMPLETE and not (attempt and attempt.get("attempted")):
                lines += ["Execution: **" + ("recorded" if result["execution_recorded"] else
                           "incompatible; skipped" if _incompatible(result) else "not run") + "**."]
                if result["attempt_id"]:
                    lines.append("Attempt: `" + result["attempt_id"] + "`; raw outcome: " +
                                 cell(result["metrics"].get("raw_status", "unavailable")) + ".")
                    cost = result["metrics"].get("cost", {})
                    lines.append("Recorded execution seconds: " + number(cost.get("execution_seconds")) +
                                 "; charged wall seconds: " + number(cost.get("wall_seconds")) +
                                 " (shared execution is charged once to " + cell(cost.get("charged_task", name)) + ").")
                lines.append("")
            if result["device"]:
                lines += ["Actual task device: `" + cell(result["device"]) + "` (recorded execution receipt).", ""]
            if result["policy_parent"]:
                parent = result["policy_parent"]
                parent_id = parent.get("id", "unbound")
                lines += ["Policy-cohort variant of " + _existing_link(root, page, parent_id,
                           Path("configs/forge/tasks") / (parent_id + ".json")) +
                          "; parent task SHA256 `" + cell(parent.get("task_sha256")) +
                          "`. This measurement supplies no cells to the parent clean cohort.", ""]
            memberships = [link(root, page, view["id"] + " / Tier " + str(a["qualification_tier"]), family["page"],
                                base + "-" + view["id"] + "-tier-" + str(a["qualification_tier"]))
                           for view in progress["views"] if _displayed_for(view, family["id"])
                           for a in view["assignments"] if a["task"] == name]
            lines += ["Used by: " + " · ".join(memberships) + ".", ""]
            measured = result["metrics"]
            checks = measured.get("evaluator_summary", {}).get("metric_checks", {})
            if checks:
                lines += ["Recorded final metric checks:", "", "| Metric | Measured | Recorded bound | Recorded check |",
                          "| --- | ---: | --- | --- |"]
                lines += [f"| {cell(metric)} | {number(check.get('value'))} | {cell(check.get('op'))} "
                          f"{number(check.get('threshold'))} | {cell(check.get('status'))} |" for metric, check in sorted(checks.items())]
                lines.append("")
            elif measured.get("metrics"):
                values = measured["metrics"]
                lines += ["Recorded final metrics:", "", "| Metric | Measured |", "| --- | ---: |"]
                lines += [f"| {cell(metric)} | {number(value)} |" for metric, value in _metric_values(values)]
                lines.append("")
            convergence = measured.get("evaluator_summary", {}).get("convergence", {})
            if "passing_suffix" in convergence:
                lines += [f"Recorded terminal passing observations: **{convergence['passing_suffix']}**; "
                          f"required: {convergence.get('minimum_stable_checks', 'unavailable')}.", ""]
            if result["receipt"]:
                lines += [link(root, page, "Compact metrics and receipt provenance", result["receipt"]), ""]
            for artifact in result.get("training_media", []):
                lines += [link(root, page, "Actual-training GIF", artifact["gif"]) +
                          "; " + str(artifact["observation_count"]) + " saved observations; "
                          "no new optimizer updates or sampling draws.", ""]
            if result.get("clock_audit"):
                audit = result["clock_audit"]
                lines += ["Recorded clock parity diagnostics:", "", "| Condition | Exact state digest equality |",
                          "| --- | --- |"]
                lines += ["| " + cell(condition) + " | " + ("equal" if check["digest_equal"] else "different") + " |"
                          for condition, check in audit["comparisons"].items()]
                lines += ["", "Recorded unexplained clock dependencies: **" + str(audit["clock_dependency_count"]) + "**.", ""]
                lines += ["- " + cell(dependency) for dependency in audit["unexplained_clock_dependencies"]]
                lines += ["", link(root, page, "Certified parity digests and source audit", audit["path"]) +
                          ". These display diagnostics preserve the recorded gate " + cell(audit["recorded_grade"]) + ".", ""]
            if result["recorded_contract"]:
                old = publication.get("task_contracts", {}).get(result["recorded_contract"], result.get("recorded_conditions", {}))
                sampling = old.get("sampling", {})
                prior = old.get("prior", {})
                lines += ["Recorded conditions: " + cell(prior.get("kind", "unbound")) +
                          f" prior (sigma {number(prior.get('sigma'))}); " + cell(sampling.get("sampling_law", "unbound")) +
                          "; weights " + cell(sampling.get("scoring_weights", "unbound")) +
                          "; output noise " + cell(sampling.get("eval_output_noise", "unbound")) + ".", ""]
            execution = task.get("execution", {})
            budget = execution.get("max_total_steps", execution.get("steps", "unavailable"))
            evaluation = task.get("evaluation", {})
            prior = execution.get("prior", {})
            lines += ["Current pass criteria:", "", *_current_gate(task.get("evaluation", {})), "",
                      f"Declared budget: {budget} updates; timeout {task.get('resources', {}).get('timeout_seconds', 'unavailable')} seconds.", "",
                      "Current measurement: " + cell(prior.get("kind", "unbound")) +
                      f" prior (sigma {number(prior.get('sigma'))}); " + cell(evaluation.get("sampling_law", "unbound")) +
                      "; weights " + cell(evaluation.get("scoring_weights", "unbound")) +
                      "; output noise " + cell(evaluation.get("eval_output_noise", "unbound")) + ".", ""]
            if task.get("dependencies"):
                lines += ["Dependencies: " + ", ".join(dep["task"] + " (" + dep["kind"] + ")" for dep in task["dependencies"]) + ".", ""]
            explanation = task.get("research_artifacts", {}).get("readout")
            if explanation:
                lines += [_existing_link(root, page, "Explanation and existing training artifacts", explanation), ""]
    lines += ["## Historical and diagnostic evidence", "",
              "Separate configurations, API variants and serving laws retain their own scopes and supply no cells above.", ""]
    from .research_tiers import family_research_lines
    if not family.get("solution_page"):
        lines += family_research_lines(root, page, family["id"])
    if family["id"] == "bcap-pure" and not family.get("solution_page"):
        pure = root / "reports/forge/pure-bcap"
        if any((pure / name).is_file() for name in ("publication.json", "original-initial-readout.json")):
            import importlib.util
            spec = importlib.util.spec_from_file_location("pure_bcap_family_navigation", pure / "publish.py")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            lines.append(helper.display_section(root, page))
        else:
            retained_notes = root / "reports/forge/leaderboard-retained-source-notes.md"
            if retained_notes.is_file():
                notes = retained_notes.read_text()
                target = os.path.relpath(pure, page.parent)
                lines.append(notes.replace("](pure-bcap/", "](" + target + "/"))
    if family["id"] == "release07-gan-v3":
        for previous in progress.get("historical_families", []):
            if previous["id"].startswith("release07-gan-v3-"):
                lines.append("- " + link(root, page, previous["label"] + " original cohort", previous["page"]))
    if family["id"] == "atlas":
        original = publication.get("original_pr223_atlas", {})
        for label, record in [("Original Atlas recipe and serving-law evidence", original), *[(name, original.get(name, {})) for name in
                              ("fresh_retest", "native3_continuation", "native3_repaired_continuation")]]:
            if record.get("readout"):
                lines.append("- " + _existing_link(root, page, label, record["readout"]))
        progress_atlas = publication.get("atlas_unblocking_progress", {})
        for record in [progress_atlas.get("baseline", {}), *progress_atlas.get("adaptations", []), progress_atlas.get("word", {})]:
            if record.get("readout"):
                lines.append("- " + _existing_link(root, page, "Atlas diagnostic", record["readout"]))
        for key, label in (("baseline_debugging", "C6 baseline selection and retained diagnosis"),
                           ("word_half_base_diagnostic", "Word half-base rate contrast")):
            record = publication.get(key, {})
            if record.get("readout"):
                lines.append("- " + _existing_link(root, page, label, record["readout"]))
    for study in publication.get("completed_api_studies", {}).get("rows", []):
        if study.get("family_id", study.get("family")) == family["id"] or family["id"] == "atlas":
            if study.get("readout"):
                lines.append("- " + _existing_link(root, page, study.get("id", "Completed study"), study["readout"]))
    for study in publication.get("standalone_api_scores", []):
        if study.get("trainer_family") == family["id"]:
            label = "Standalone API evidence: " + family["id"] + " · " + study["case"]["title"]
            lines.append("- " + _existing_link(root, page, label, study["readout"]) + " · " +
                         _existing_link(root, page, "actual-training GIF", study["gif"]))
    lines += ["- " + link(root, page, "Other configurations and original evidence bindings", "reports/forge/technique-inventory.json"),
              "- " + link(root, page, "Compiled experiment memory", "reports/forge/EXPERIMENT_MEMORY.md"), "",
              "## Refresh", "", "```sh", "python reports/forge/regenerate_technique_inventory.py", "```", "",
              "This page is generated alongside the leaderboard. Register new source evidence before refreshing; "
              "editing a page cannot change a verdict or earn qualification.", ""]
    lines += _technique_references(family)
    return "\n".join(lines)


def generated_pages(root: Path, publication: dict) -> dict[Path, str]:
    pages = {root / family["page"]: render_family(root, publication, family)
            for family in (publication["family_progress"]["families"] +
                           publication["family_progress"].get("configuration_families", []) +
                           publication["family_progress"].get("historical_families", []))}
    pages[root / PROBLEM_PAGE] = render_problem_list(root, publication, root / PROBLEM_PAGE)
    return pages
