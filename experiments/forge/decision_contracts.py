"""Versioned, bounded hypothesis-to-decision contracts for new research ideas.

Mechanical validation binds a question to its actual public task conditions.
Numerical signatures are reviewed predictions, never qualification or causal
proof. No function here trains, changes a gate, or authorizes another round.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import re

from .contracts import canonical, file_hash, identifier, positive_number, read_json, stable_hash

VERSION = 1
OUTCOMES = {
    "falsified": "stop_revision", "prediction_observed": "review_saved_diagnostics",
    "inconclusive": "stop_and_readout", "incomplete": "request_missing_evidence",
}
OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
       ">": lambda a, b: a > b, "<": lambda a, b: a < b, "==": lambda a, b: a == b}


def scaffold(parent: str, goal: str) -> dict:
    """A reviewable draft, deliberately ineligible for execution."""
    return {"schema_version": VERSION, "status": "draft",
            "prior_evidence": [{"path": "TODO: source-bound JSON", "sha256": "TODO", "selector": [],
                                "identity": {"record_id": "TODO"}, "use": "motivation_only"}],
            "control": {"candidate_id": parent, "task_map": {}, "binding_sha256": "TODO"},
            "candidate_binding_sha256": "TODO", "substantive_delta": [],
            "scope": {"view": goal, "through_tier": 1, "task_ids": [], "max_rounds": 1,
                      "candidate_budget_seconds": 900, "campaign_budget_seconds": 900,
                      "protocol_sha256": "TODO", "source_digest": "TODO", "execution_backend": "cuda", "runtime_cohort_sha256": "TODO", "jobs_sha256": "TODO"},
            "prediction": {"task_id": "TODO", "metric": "TODO", "op": ">=", "threshold": 0.0, "phase": "final"},
            "falsifier": {"task_id": "TODO", "metric": "TODO", "op": "<", "threshold": 0.0, "phase": "final"},
            "competing_explanation": "TODO: state the alternative explanation this observation distinguishes",
            "terminal_rules": deepcopy(OUTCOMES)}


def _hash(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"decision_contract {label} requires a SHA-256")


def validate_shape(contract: dict) -> None:
    if not isinstance(contract, dict) or type(contract.get("schema_version")) is not int or contract.get("schema_version") != VERSION:
        raise ValueError("decision_contract requires schema_version 1")
    canonical(contract)
    if contract.get("status") == "draft":
        if not isinstance(contract.get("control"), dict) or not contract["control"].get("candidate_id"):
            raise ValueError("decision_contract draft requires the generated control candidate scaffold")
        identifier(contract["control"]["candidate_id"], "decision control")
        return
    required = {"schema_version", "status", "prior_evidence", "control", "candidate_binding_sha256",
                "substantive_delta", "scope", "prediction", "falsifier", "competing_explanation", "terminal_rules"}
    if set(contract) != required or contract.get("status") != "ready":
        raise ValueError("decision_contract must be a complete ready contract; use the generated draft scaffold")
    if not isinstance(contract["prior_evidence"], list) or not contract["prior_evidence"]:
        raise ValueError("decision_contract requires source-bound prior_evidence")
    for evidence in contract["prior_evidence"]:
        if (not isinstance(evidence, dict) or set(evidence) != {"path", "sha256", "selector", "identity", "use"}
                or evidence["use"] != "motivation_only" or not evidence["identity"]
                or not isinstance(evidence["identity"], dict) or not isinstance(evidence["selector"], list)):
            raise ValueError("decision_contract prior evidence requires an exact identity; summaries are motivation_only")
        _hash(evidence["sha256"], "prior_evidence.sha256")
    control = contract["control"]
    if not isinstance(control, dict) or set(control) != {"candidate_id", "task_map", "binding_sha256"}:
        raise ValueError("decision_contract requires a control candidate, task_map and binding_sha256")
    identifier(control["candidate_id"], "decision control")
    if not isinstance(control["task_map"], dict):
        raise ValueError("decision_contract control.task_map must be a mapping")
    _hash(control["binding_sha256"], "control.binding_sha256")
    _hash(contract["candidate_binding_sha256"], "candidate_binding_sha256")
    scope = contract["scope"]
    if (not isinstance(scope, dict) or set(scope) != {"view", "through_tier", "task_ids", "max_rounds",
                                                    "candidate_budget_seconds", "campaign_budget_seconds",
                                                    "protocol_sha256", "source_digest", "execution_backend", "runtime_cohort_sha256", "jobs_sha256"}
            or type(scope["through_tier"]) is not int or isinstance(scope["through_tier"], bool) or scope["through_tier"] < 1
            or type(scope["max_rounds"]) is not int or scope["max_rounds"] != 1
            or not isinstance(scope["task_ids"], list) or not scope["task_ids"]
            or len(scope["task_ids"]) != len(set(scope["task_ids"]))):
        raise ValueError("decision_contract scope must freeze one bounded round and explicit task_ids")
    identifier(scope["view"], "decision view")
    _hash(scope["protocol_sha256"], "scope.protocol_sha256")
    _hash(scope["source_digest"], "scope.source_digest")
    _hash(scope["runtime_cohort_sha256"], "scope.runtime_cohort_sha256")
    _hash(scope["jobs_sha256"], "scope.jobs_sha256")
    if scope["execution_backend"] not in {"cpu", "cuda"}:
        raise ValueError("decision_contract scope must bind its CPU/CUDA evidence cohort")
    for task in scope["task_ids"]:
        identifier(task, "decision task")
    positive_number(scope["candidate_budget_seconds"], "decision candidate budget")
    positive_number(scope["campaign_budget_seconds"], "decision campaign budget")
    if scope["candidate_budget_seconds"] > scope["campaign_budget_seconds"]:
        raise ValueError("decision candidate budget exceeds campaign budget")
    if not isinstance(contract["substantive_delta"], list) or not contract["substantive_delta"]:
        raise ValueError("decision_contract requires an exact substantive delta")
    for signature in (contract["prediction"], contract["falsifier"]):
        if (not isinstance(signature, dict) or set(signature) != {"task_id", "metric", "op", "threshold", "phase"}
                or signature["task_id"] not in scope["task_ids"] or signature["op"] not in OPS
                or signature["phase"] != "final" or not isinstance(signature["metric"], str)
                or not signature["metric"].strip() or "TODO" in signature["metric"] or type(signature["threshold"]) not in (int, float)
                or not math.isfinite(signature["threshold"])):
            raise ValueError("decision_contract prediction/falsifier require a scoped finite numerical final-metric signature")
    explanation = contract["competing_explanation"]
    if not isinstance(explanation, str) or not explanation.strip() or "TODO" in explanation:
        raise ValueError("decision_contract requires a competing explanation for review")
    if contract["terminal_rules"] != OUTCOMES:
        raise ValueError("decision_contract terminal_rules must retain deterministic stop/review actions; no automatic continuation")


def _authorized_tasks(request):
    tiers = {item["task"]: item["qualification_tier"] for item in request["view"]["assignments"]}
    return sorted(name for name in request["tasks"] if tiers[name] <= request["through_tier"])


def _bindings(candidate, tasks, protocol, root):
    """Actual public-API bindings; per-task memoized inside a bound resolution cache.

    The memo key includes the candidate id and every declared override value
    (``planning._binding_sig``), so declarations differing only in a tunable
    value never share a cached binding.
    """
    from .resolution_scope import active
    cache = active(root)
    if cache is not None:
        return cache.bindings(candidate, tasks, protocol, root, _resolve_bindings)
    return _resolve_bindings(candidate, tasks, protocol, root)


def _resolve_bindings(candidate, tasks, protocol, root):
    from .api import task_formulation_context
    result = {name: {} for name in ("recipe", "prior", "initialization", "budget", "sampling", "host", "evaluation", "task_identity")}
    for name, task in sorted(tasks.items()):
        receipt = task_formulation_context(candidate, task, protocol, root=root).receipt()
        ownership = receipt["field_ownership"]
        result["recipe"][name] = {key: field["value"] for key, field in ownership["recipe_fields"].items()
                                 if field["status"] == "effective"}
        prior_contract = ownership["task_contract"]["prior"]
        if "technique_mechanism" in prior_contract:
            result["prior"][name] = {"task_prior": prior_contract["value"],
                                     "technique_mechanism": prior_contract["technique_mechanism"]["value"]}
        else:
            result["prior"][name] = prior_contract["value"]
        result["initialization"][name] = deepcopy(ownership["task_contract"]["initialization"])
        result["budget"][name] = {"execution": ownership["task_contract"]["budget"]["value"],
                                 "resources": deepcopy(task["resources"])}
        result["host"][name] = {"bound": ownership["task_contract"]["host"]["value"],
                                "adapter_parameters": {key: deepcopy(value) for key, value in task["execution"].items()
                                                       if key not in {"host_definition", "prior", "initializer", "fixed_initialization", "host_initialization",
                                                                      "steps", "incremental_steps", "preserve_prefix_steps", "original_schedule_horizon",
                                                                      "resources", "model", "host_source", "problem", "native_profile"}}}
        from .views import task_execution_fingerprint, task_evaluation_fingerprint
        result["task_identity"][name] = {"execution": task_execution_fingerprint(task),
                                        "evaluation": task_evaluation_fingerprint(task)}
        result["evaluation"][name] = deepcopy(task["evaluation"])
        result["sampling"][name] = {key: task["evaluation"].get(key) for key in
                                    ("sampling_contract_version", "sampling_law", "eval_output_noise", "scoring_weights")}
    return result


def delta(before, after, path=()):
    """Exact leaf differences; missing values remain distinguishable from null."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            missing = {"absent": True}
            result.extend(delta(before.get(key, missing), after.get(key, missing), (*path, key)))
        return result
    if canonical(before) == canonical(after):
        return []
    return [{"path": list(path), "before": before, "after": after}]


def _evidence(root, evidence):
    if not isinstance(evidence["path"], str):
        raise ValueError("decision_contract prior evidence path must be repository-relative")
    path = root / evidence["path"]
    if (Path(evidence["path"]).is_absolute()
            or ".." in Path(evidence["path"]).parts or not path.resolve().is_relative_to(root.resolve())
            or path.suffix != ".json" or not path.is_file() or file_hash(path) != evidence["sha256"]):
        raise ValueError("decision_contract prior evidence is missing, unsafe or has changed; retain the original exact identity")
    selected = read_json(path)
    try:
        for key in evidence["selector"]:
            if type(key) not in (str, int) or (type(key) is int and key < 0):
                raise ValueError("invalid selector")
            selected = selected[key]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("decision_contract prior evidence selector does not identify a published object") from exc
    if not isinstance(selected, dict) or any(selected.get(key) != value for key, value in evidence["identity"].items()):
        raise ValueError("decision_contract prior evidence identity contradicts its bound source")


def inspect_contract(root: Path, request: dict, *, binding_resolver=None) -> dict | None:
    """Expose actual bindings in plans and block unfinished/mismatched launches."""
    candidate = request["candidate"]
    contract = candidate.get("decision_contract")
    if contract is None:
        return None  # Immutable v1 requests retain their original semantics.
    validate_shape(contract)
    root = Path(root)
    active = _authorized_tasks(request)
    from .planning import load_idea
    from .views import load_tasks
    defaults = read_json(root / "configs/forge/defaults.json")
    parent = load_idea(root, contract["control"]["candidate_id"])
    parent = {**parent, "prior": {**defaults["prior"], **parent.get("prior", {})}}
    catalog = load_tasks(root)
    mapping = contract["control"].get("task_map") or {name: name for name in active}
    if set(mapping) != set(active) or any(name not in catalog for name in mapping.values()):
        raise ValueError("decision_contract control.task_map must bind every authorized task to a declared control task")
    bind = binding_resolver or _bindings
    before = bind(parent, {name: catalog[control] for name, control in mapping.items()}, request["protocol"], root)
    after = bind(candidate, {name: request["tasks"][name] for name in active}, request["protocol"], root)
    changed = delta(before, after)
    expected = {"control_binding_sha256": stable_hash(before), "candidate_binding_sha256": stable_hash(after),
                "substantive_delta": changed, "task_map": mapping, "task_ids": active,
                "protocol_sha256": stable_hash(request["protocol"]), "source_digest": request["source"]["digest"],
                "execution_backend": request.get("execution_backend"),
                "runtime_cohort_sha256": stable_hash({"runtime": request.get("runtime"), "compute_profiles": request.get("compute_profiles")}),
                "jobs_sha256": stable_hash(sorted((job for job in request["jobs"]
                                                   if set(job.get("task_ids", [job["task_id"]])) <= set(active)),
                                                  key=lambda job: job["compatibility_key"]))}
    blockers = []
    if contract["status"] != "ready":
        blockers.append("decision_contract is draft: bind prior evidence, copy reviewed actual bindings/delta, freeze scope and numerical falsifier, then set status=ready")
    else:
        scope = contract["scope"]
        if scope["view"] != request["view"]["id"] or scope["through_tier"] != request["through_tier"] or sorted(scope["task_ids"]) != active:
            blockers.append("decision_contract scope differs from the requested view/tier/tasks")
        if any(scope[name] != expected[name] for name in ("protocol_sha256", "source_digest", "execution_backend", "runtime_cohort_sha256", "jobs_sha256")):
            blockers.append("decision_contract protocol/source/backend cohort differs from the frozen question")
        if contract["control"]["binding_sha256"] != expected["control_binding_sha256"]:
            blockers.append("decision_contract control actual recipe/prior/initialization/budget/sampling binding changed")
        if contract["candidate_binding_sha256"] != expected["candidate_binding_sha256"]:
            blockers.append("decision_contract candidate actual task-owned binding changed")
        if canonical(contract["substantive_delta"]) != canonical(changed) or not changed:
            blockers.append("decision_contract substantive_delta must exactly describe the exercised change; prose/seed-only/inactive changes are insufficient")
        if not any(item["path"][0] in {"recipe", "prior", "budget", "sampling", "host"}
                   or (item["path"][0] == "initialization" and len(item["path"]) > 2
                       and item["path"][2] in {"value", "component_policies", "host_initialization"}) for item in changed):
            blockers.append("decision_contract changes no exercised mechanism or host condition")
        total = sum(job["budget_seconds"] for job in request["jobs"]
                    if set(job.get("task_ids", [job["task_id"]])) <= set(active))
        if total > scope["candidate_budget_seconds"]:
            blockers.append("decision_contract candidate cap cannot reserve the complete bounded task allowances")
        for evidence in contract["prior_evidence"]:
            try:
                _evidence(root, evidence)
            except ValueError as exc:
                blockers.append(str(exc))
    return {"schema_version": VERSION, "status": "BLOCKED" if blockers else "READY", "blockers": blockers,
            "expected": expected, "actual_bindings": {"control": before, "candidate": after},
            "receipt": {"contract_sha256": stable_hash(contract), "binding_sha256": stable_hash(after),
                        "control_binding_sha256": stable_hash(before), "protocol_sha256": stable_hash(request["protocol"])},
            "qualification_input": False, "causal_judgment": "requires_review"}


def validate_admission(request: dict, campaign: dict, *, root: Path | None) -> None:
    if "study" in request:
        from .studies import validate_admission as validate_study_admission
        validate_study_admission(root, request, campaign)
        return
    contract = request["candidate"].get("decision_contract")
    if contract is None:
        if request["candidate"].get("schema_version") == 2:
            raise ValueError("v2 research admission requires a decision_contract")
        if root is not None and "calibration_lane" in request:
            from .calibration_lane import validate_submission
            if stable_hash(validate_submission(root, request)) != stable_hash(campaign):
                raise ValueError("calibration campaign differs from its exact frozen registration")
            return
        if root is not None and "promotion" in request:
            from .promotion import validate_submission
            if stable_hash(validate_submission(root, request)) != stable_hash(campaign):
                raise ValueError("promotion campaign differs from its exact frozen registration")
            return
        if root is not None and "configuration_id" in request["candidate"]:
            validate_registered_search(root, request, campaign)
            return
        if request["candidate"].get("schema_version") == 3:
            raise ValueError("v3 research admission requires an explicit study")
        validate_legacy_admission(request, root=root)
        return
    if root is None:
        raise ValueError("decision_contract admission requires repository-bound report_root")
    review = inspect_contract(root, request)
    if review["blockers"] or request.get("decision_admission") != review["receipt"]:
        raise ValueError("decision_contract admission blocked: " + "; ".join(review["blockers"] or ["missing/mismatched frozen decision receipt"]))
    scope = contract["scope"]
    if (campaign["candidate_budget_seconds"] > scope["candidate_budget_seconds"]
            or campaign["budget_seconds"] > scope["campaign_budget_seconds"]):
        raise ValueError("campaign exceeds the decision_contract frozen bounded-round caps")


def _row_producer_study_id(row: dict) -> str | None:
    if not isinstance(row, dict):
        return None
    direct = row.get("producer_study_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    prov = row.get("provenance")
    if isinstance(prov, dict):
        nested = prov.get("producer_study_id")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _row_contended(row: dict) -> bool | None:
    if not isinstance(row, dict):
        return None
    cost = row.get("cost")
    if isinstance(cost, dict):
        energy = cost.get("energy")
        if isinstance(energy, dict) and isinstance(energy.get("contended"), bool):
            return energy["contended"]
    hardware = row.get("hardware")
    if isinstance(hardware, dict):
        status = hardware.get("contention_status")
        if status == "clear":
            return False
        if status == "contended":
            return True
    if isinstance(row.get("contended"), bool):
        return row["contended"]
    return None


def _resolve_task_rows(request: dict, rows: list[dict], task_ids: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Select one row per scoped task, scoping by study and uncontended status.

    Never mutates input rows or historical receipts. Genuine duplicate attempts
    within the same study and contention tier remain reported as duplicates.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("task_id"), str):
            grouped[row["task_id"]].append(row)
    study = request.get("study") if isinstance(request, dict) else None
    target_study_id = study.get("id") if isinstance(study, dict) and isinstance(study.get("id"), str) else None

    by_task: dict[str, dict] = {}
    duplicates: list[str] = []
    for name in sorted(set(task_ids)):
        candidates = list(grouped.get(name, ()))
        if len(candidates) > 1 and target_study_id:
            study_ids = [_row_producer_study_id(row) for row in candidates]
            in_study = [row for row, sid in zip(candidates, study_ids) if sid == target_study_id]
            has_foreign_study = any(sid is not None and sid != target_study_id for sid in study_ids)
            if in_study and has_foreign_study:
                candidates = in_study
        if len(candidates) > 1:
            contention_flags = [_row_contended(row) for row in candidates]
            uncontended = [row for row, flag in zip(candidates, contention_flags) if flag is False]
            has_contended = any(flag is True for flag in contention_flags)
            if uncontended and has_contended:
                candidates = uncontended
        if len(candidates) == 1:
            by_task[name] = candidates[0]
        elif len(candidates) > 1:
            duplicates.append(name)
    return by_task, duplicates


def evaluate(request: dict, rows: list[dict]) -> dict:
    """Evaluate final scalar predictions, preserving scientific gates separately."""
    from .studies import contract_for
    contract = contract_for(request)
    validate_shape(contract)
    if contract["status"] != "ready":
        raise ValueError("cannot conclude a draft decision_contract")
    by_task, duplicates = _resolve_task_rows(request, rows, contract["scope"]["task_ids"])
    observed = {}
    scope = contract["scope"]
    cohort_errors = []
    current = {"protocol_sha256": stable_hash(request["protocol"]), "source_digest": request["source"]["digest"],
               "execution_backend": request.get("execution_backend"),
               "runtime_cohort_sha256": stable_hash({"runtime": request.get("runtime"), "compute_profiles": request.get("compute_profiles")})}
    if any(scope[name] != current[name] for name in current):
        cohort_errors.append("saved request differs from the frozen decision cohort")
    complete = not cohort_errors and all(name in by_task and by_task[name].get("gate_status") in {"PASS", "FAIL"}
                   for name in contract["scope"]["task_ids"])
    for name in ("prediction", "falsifier"):
        signature = contract[name]
        value = by_task.get(signature["task_id"], {}).get("metrics")
        value = value.get(signature["metric"]) if isinstance(value, dict) else None
        valid = type(value) in (int, float) and math.isfinite(value)
        observed[name] = {"value": value if valid else None,
                          "satisfied": OPS[signature["op"]](value, signature["threshold"]) if valid else None}
        complete &= valid
    outcome = ("incomplete" if not complete else "falsified" if observed["falsifier"]["satisfied"] else
               "prediction_observed" if observed["prediction"]["satisfied"] else "inconclusive")
    return {"schema_version": VERSION, "contract_sha256": stable_hash(contract), "outcome": outcome,
            "next_action": contract["terminal_rules"][outcome], "observed": observed,
            "duplicate_task_ids": duplicates, "binding_errors": cohort_errors, "max_rounds": 1,
            "qualification_input": False, "causal_judgment": "requires_review", "execution_authorized": False}


def validate_legacy_admission(request, *, root):
    """Only exact pinned original declarations retain the v1 admission path.

    Lightweight coordinator protocols without an idea schema are unchanged;
    production research requests carry schema_version and a declared candidate.
    Saved evidence readers never invoke admission and need no migration.
    """
    candidate = request["candidate"]
    if "schema_version" not in candidate and not request.get("requires_independent_grading"):
        return
    if root is None:
        raise ValueError("research admission requires repository-bound report_root")
    manifest_path = Path(root) / "configs/forge/legacy-ideas-v1.json"
    if not manifest_path.is_file():
        raise ValueError("v1 admission requires its pinned legacy declaration manifest; new ideas require v2 decision_contract")
    manifest = read_json(manifest_path)
    paths = manifest.get("declarations", {})
    matching = [relative for relative in paths if Path(relative).stem == candidate["id"]]
    if len(matching) != 1:
        raise ValueError("new v1 declaration is not immutable legacy evidence; use a v2 decision_contract")
    path = Path(root) / matching[0]
    if not path.is_file() or file_hash(path) != paths[matching[0]]:
        raise ValueError("legacy v1 declaration changed; retain its identity and create a v2 successor")
    original = read_json(path)
    defaults = read_json(Path(root) / "configs/forge/defaults.json")
    expected = {**original, "prior": {**defaults["prior"], **original.get("prior", {})},
                "claim_contract": original.get("claim_contract", {"schedule": "scheduled", "scoring_weights": "live"})}
    actual = {key: value for key, value in candidate.items() if key not in {"resolved_recipe", "capabilities"}}
    if canonical(actual) != canonical(expected):
        raise ValueError("request differs from its immutable legacy v1 declaration; use a v2 successor")


def round_definition(request):
    from .studies import contract_for
    contract = contract_for(request)
    if contract is None:
        return None
    active = set(_authorized_tasks(request))
    keys = sorted(job["compatibility_key"] for job in request["jobs"]
                  if set(job.get("task_ids", [job["task_id"]])) <= active)
    return {"round_id": stable_hash({"source": contract["scope"]["source_digest"],
                                     "binding": contract["candidate_binding_sha256"],
                                     "tasks": sorted(contract["scope"]["task_ids"]),
                                     "protocol": contract["scope"]["protocol_sha256"],
                                     "runtime": contract["scope"]["runtime_cohort_sha256"],
                                     "backend": contract["scope"]["execution_backend"]}),
            "job_keys": keys, "budget_seconds": contract["scope"]["candidate_budget_seconds"], "max_rounds": 1}


def register_round(state, request):
    """Register under the queue lock; prose/campaign changes cannot reset caps."""
    definition = round_definition(request)
    if definition is None:
        return
    rounds = state.setdefault("decision_rounds", {})
    previous = rounds.get(definition["round_id"])
    if previous is not None and previous != definition:
        raise ValueError("decision round budget is immutable across campaigns and contracts")
    from .studies import register
    register(state, request)
    rounds[definition["round_id"]] = definition


def available_round_budget(state, request, seconds, *, job_key=None):
    """Cumulative paid retries/reservations across campaign owners in one queue.

    Check even an old subscriber when its compatible jobs belong to a bounded
    round. Shared grouped jobs are counted once. Separate queue roots cannot
    supply a global campaign limit and are explicitly outside this guarantee.
    """
    keys = set(job["compatibility_key"] for job in request["jobs"])
    for definition in state.get("decision_rounds", {}).values():
        selected = set(definition["job_keys"])
        if not keys.intersection(selected) or (job_key is not None and job_key not in selected):
            continue
        attempts = {attempt["attempt_id"] for key in selected for attempt in state["jobs"].get(key, {}).get("attempts", [])}
        spent = sum(charge["seconds"] for charge in state.get("charges", []) if charge["attempt_id"] in attempts)
        reserved = sum(state["jobs"].get(key, {}).get("reserved_seconds", 0) for key in selected)
        if spent + reserved + seconds > definition["budget_seconds"]:
            return False, "decision round budget cannot reserve the full next task across campaigns/retries"
    return True, None


def concluded_outcomes(attempts):
    """Conclude each frozen contract/compute cohort, never reconstruct science.

    Certified infrastructure repairs may supersede outcomes, while original
    attempts/costs retain their identity in the ordinary readout provenance.
    Duplicate unsuperseded rows remain incomplete, regardless of agreement.
    """
    groups = defaultdict(list)
    for attempt in attempts:
        request = attempt["request"]
        from .studies import contract_for
        contract = contract_for(request)
        if contract is None:
            continue
        cohort = {"contract_sha256": stable_hash(contract), "round": round_definition(request),
                  "source_digest": request.get("source", {}).get("digest"), "runtime": request.get("runtime"),
                  "execution_backend": request.get("execution_backend"), "compute_profiles": request.get("compute_profiles")}
        if "study" in request:
            cohort["study_id"] = request["study"]["id"]
            cohort["study_sha256"] = stable_hash(request["study"])
        groups[stable_hash(cohort)].append((attempt, cohort))
    results = []
    for identity, group in sorted(groups.items()):
        selected = [attempt for attempt, _ in group]
        active = [attempt for attempt in selected if not attempt.get("superseded_by")]
        rows = []
        for attempt in active:
            attempt_study_id = ((attempt.get("request") or {}).get("study") or {}).get("id")
            for row in attempt["task_results"]:
                if isinstance(attempt_study_id, str) and attempt_study_id and "producer_study_id" not in row:
                    rows.append({**row, "producer_study_id": attempt_study_id})
                else:
                    rows.append(row)
        outcome = evaluate(selected[0]["request"], rows)
        outcome.update(cohort_sha256=identity, cohort=group[0][1],
                       attempt_ids=sorted(attempt["attempt_id"] for attempt in selected),
                       provenance=[{"attempt_id": attempt["attempt_id"], "result_hash": attempt["result_hash"],
                                    "valid_receipt": attempt["valid_receipt"]} for attempt in selected])
        results.append(outcome)
    return results


def validate_registered_search(root, request, campaign):
    """Retain the existing finite-search contract, requiring exact registration.

    A caller-provided configuration_id alone grants nothing. Registration binds
    the complete resolved candidate/tasks/jobs, source, runtime, protocol, scope,
    finite grid and both existing budget caps before the first queue admission.
    """
    from .configuration_search import _load_spec, _check_spec_registration, _declarations, validate_configuration_declaration
    from .technique_inventory import _signature
    candidate = request["candidate"]
    validate_configuration_declaration(candidate, root=root)
    for path in sorted((Path(root) / "reports/forge/configuration-search").glob("*.json")):
        saved = read_json(path)
        if saved.get("campaign") != campaign:
            continue
        spec = _load_spec(root, saved.get("spec"))
        registered = _check_spec_registration(root, spec)
        if registered is None or registered.get("stage") not in {"registered", "enqueued", "completed", "reported"}:
            continue
        if candidate.get("schema_version") == 3 and request.get("search_plan") != {
                "study_id": spec["id"], "spec_sha256": stable_hash(spec), "declaration": spec}:
            continue
        matches = [trial for trial in registered["trials"] if trial["candidate_id"] == candidate["id"]]
        if len(matches) != 1:
            continue
        trial = matches[0]
        if (trial["scientific_signature"] != _signature(request)
                or trial["source_digest"] != request["source"]["digest"]
                or registered["campaign"] != spec["campaign"]):
            raise ValueError("configuration request differs from its exact bounded search registration")
        declared = {idea["id"]: idea for idea, _ in _declarations(root, spec)}
        if candidate["id"] not in declared or stable_hash(declared[candidate["id"]]) != stable_hash(trial["declaration"]):
            raise ValueError("configuration declaration differs from its frozen finite grid")
        if spec["schema_version"] == 2:
            from .planning import resolve_idea
            current = resolve_idea(root, candidate["id"], declaration=declared[candidate["id"]],
                                   view_id=spec["view"], through_tier=spec["tuning_through_tier"],
                                   execution_backend=spec["execution_backend"], cuda_model=spec.get("cuda_model"))
            if (_signature(current) != _signature(request) or request.get("view") != current["view"]
                    or any(request.get("source", {}).get(key) != current["source"][key] for key in ("digest", "files"))):
                raise ValueError("configuration request differs from the generated source/task/view binding")
        return
    raise ValueError("v1 configuration admission requires an exact bounded search registration; new bare ideas require v2 decision_contract")
