"""Regeneratable technique-by-tier reports over Forge's qualified evidence.

This is a display reducer. The knowledge board selects compatible receipts and
regrades current evidence; this module never launches work or changes gates.
Pinned, calibration and historical outcomes retain their separate scopes.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

from .contracts import atomic_text, file_hash, read_json, stable_hash
from .views import is_stretch, task_evaluation_fingerprint, task_execution_fingerprint

REDUCER_VERSION = "forge-technique-board-v1"
DEFAULT_LABELS = {
    "k3p": "K3P",
    "ka2": "KA2",
    "k3p-r1r2-matched-v1": "R1/R2 standard penalty (matched K3P recipe)",
    "r3gan-stacked-training-toy-v1": "R3GAN Stacked-MNIST recipe (toy-host adaptation)",
    "k3p-bcap-matched-v1": "BCap (matched K3P recipe)",
    "release07-gan-v3-mog-v1": "GAN v3 release 0.7 (MoG adaptation)",
    "release07-gan-v3-task-adapted-v1": "GAN v3 release 0.7 (task adaptation)",
    "release07-gan-v3-cloud-v1": "GAN v3 release 0.7 (cloud)",
    "e22": "E22",
    "atlas": "Atlas",
    "forge-onboarding-anchor-ablation": "K3P without critic anchor",
    "forge-no-critic-penalty": "K3P without critic penalty",
    "k3p-a2-off-native-diagnostic": "K3P without A2",
    "k3p-no-output-noise-diagnostic": "K3P without training output noise",
    "rhbi-v1": "RHBI (v1)",
    "rhbi-v2": "RHBI (v2)",
    "rhbi-v3": "RHBI (v3)",
    "rhbi-v4": "RHBI (v4)",
    "rhbi-v5": "RHBI (v5)",
    "rhbi-v6": "RHBI (v6)",
    "rhbi-v7": "RHBI (v7)",
    "rhbi-v8": "RHBI (v8)",
    "rhbi-v9": "RHBI (v9)",
    "rhbi-v10": "RHBI (v10)",
    "rhbi-v11": "RHBI",
}
ARCHIVED_SCOPES = ("pinned", "calibration_diagnostic", "research_diagnostic", "historical")


def request_bindings(request: dict) -> dict:
    """Compact explicit contracts, backed by full scientific fingerprints.

    Saved request/result receipts retain observed tensor and stream hashes. The
    report binds the declared recipe, host, prior, initialization, complete steps,
    sampling/evaluator and timeout instead of inferring equivalence from labels.
    """
    candidate = request.get("candidate", {})
    task_keys = {name: job.get("compatibility_key") for job in request.get("jobs", [])
                 for name in job.get("task_ids", [job["task_id"]])}
    task_bindings = {}
    for name, task in sorted(request.get("tasks", {}).items()):
        execution, evaluation = task.get("execution", {}), task.get("evaluation", {})
        host = execution.get("host_definition", {})
        task_bindings[name] = {
            "adapter": task.get("adapter"),
            "execution_sha256": task_execution_fingerprint(task),
            "evaluation_sha256": task_evaluation_fingerprint(task),
            "compatibility_key": task_keys.get(name),
            "host": execution.get("host", execution.get("problem")),
            "host_definition_sha256": stable_hash(host),
            "host_profiles": {key: value for key, value in execution.items() if key.endswith("_profile")},
            "prior": execution.get("prior"),
            "initialization": execution.get("fixed_initialization", host.get("initialization")),
            "steps": execution.get("steps"),
            "timeout_seconds": task.get("resources", {}).get("timeout_seconds"),
            "sampling": {key: evaluation.get(key) for key in (
                "sampling_contract_version", "sampling_law", "eval_output_noise", "scoring_weights")},
        }
        if candidate.get("host_adaptation") is not None:
            from .taskrecipes import adaptation_receipt
            task_bindings[name]["host_adaptation"] = adaptation_receipt(candidate, task)
    return {"trainer_family": candidate.get("trainer_family"),
            "configuration_id": candidate.get("configuration_id"),
            "recipe": candidate.get("resolved_recipe"),
            "recipe_sha256": stable_hash(candidate.get("resolved_recipe")),
            "prior": candidate.get("prior"),
            "initializer": candidate.get("initializer", "deterministic_orthogonal"),
            "claim_contract": candidate.get("claim_contract"),
            "protocol": request.get("protocol"),
            "rng_sha256": stable_hash(request.get("rng")),
            "source_digest": request.get("source", {}).get("digest"),
            "source_origin_commit": request.get("source", {}).get("origin_commit"),
            "tasks": task_bindings}


def _status(value):
    # UNKNOWN is a presentation alias only; INCOMPLETE and INVALID stay visible.
    return "UNKNOWN" if value in {None, "NOT_RUN"} else value


def _counts(statuses):
    return dict(sorted(Counter(_status(status) for status in statuses).items()))


def _catalog_bindings(original, catalog, recipes, protocols):
    bindings = deepcopy(original.get("scientific_bindings", {}))
    task_bindings = bindings.pop("tasks", {})
    bindings["task_contracts"] = {}
    task_keys = {}
    for name, contract in sorted(task_bindings.items()):
        # Compatibility keys contain candidate/runtime identity. Keep them on
        # the row; identical host contracts can share a catalog entry.
        contract = deepcopy(contract)
        key = contract.pop("compatibility_key", None)
        digest = stable_hash(contract)
        catalog[digest] = contract
        bindings["task_contracts"][name] = digest
        task_keys[name] = key
    bindings["task_keys_sha256"] = stable_hash(task_keys)
    if "recipe" in bindings:
        recipe = bindings.pop("recipe")
        recipes[bindings["recipe_sha256"]] = recipe
    if "protocol" in bindings:
        protocol = bindings.pop("protocol")
        digest = stable_hash(protocol)
        protocols[digest] = protocol
        bindings["protocol_sha256"] = digest
    bindings["available"] = bool(original.get("scientific_bindings"))
    return bindings


def _tier_cells(row, assignments, reason_catalog):
    qualified = {task["task_id"]: task for task in row.get("qualification", {}).get("tasks", [])}
    cells, tasks, nonrequired = {}, [], []
    unresolvable = not row.get("qualification") and row.get("status") == "BLOCKED"
    for tier in sorted({item["qualification_tier"] for item in assignments}):
        required, held = [], []
        for assignment in assignments:
            if assignment["qualification_tier"] != tier:
                continue
            task = qualified.get(assignment["task"], {})
            status = task.get("status", task.get("gate_status", "BLOCKED" if unresolvable else "NOT_RUN"))
            receipt = {"task_id": assignment["task"], "status": _status(status), "gate_status": status}
            reasons = task.get("reasons", [])
            if reasons:
                digest = stable_hash(reasons)
                reason_catalog[digest] = reasons
                receipt["reasons_sha256"] = digest
            (tasks if assignment["importance"] == "required" else nonrequired).append(receipt)
            if assignment["importance"] == "required":
                required.append(status)
            elif is_stretch(assignment):
                held.append(status)
        counts = _counts(required)
        cells[str(tier)] = {"passed": counts.get("PASS", 0), "total": len(required), "counts": counts,
                            # Held tasks are measured and shown, never counted.
                            **({"held": len(held), "held_passed": held.count("PASS")} if held else {})}
    return cells, tasks, nonrequired


def _compact_cost(cost):
    return {key: cost.get(key) for key in ("wall_seconds", "measured_tasks", "unmeasured_tasks")}


def _source_pointer(source):
    if not isinstance(source, dict):
        return source
    return {key: value for key, value in source.items() if key in {
        "path", "url", "revision", "git_blob", "sha256", "digest", "origin_commit"}}


def _compact_prior(prior):
    if not isinstance(prior, dict):
        return prior
    return {key: value for key, value in prior.items() if key in {
        "kind", "sigma", "sigma_units", "standardize", "learnable", "trainable"}}


def _compact_claim(claim):
    if not isinstance(claim, dict):
        return claim
    result = {key: value for key, value in claim.items() if key in {
        "schedule", "scoring_weights", "sampling_law", "serve_average", "ema_decay"}}
    if isinstance(result.get("sampling_law"), dict):
        law = result["sampling_law"]
        result["sampling_law"] = {key: value for key, value in law.items() if key in {
            "prior_kind", "sigma_rel", "eval_output_noise", "scoring_weights", "evaluation_generate"}}
    result["full_contract_sha256"] = stable_hash(claim)
    return result


def reduce_board(board_result: dict, view: dict, *, execution_backend: str | None = None,
                 labels: dict | None = None) -> dict:
    """Convert a qualified board to compact rows without pooling any cohorts."""
    if execution_backend not in {None, "cpu", "cuda"}:
        raise ValueError("execution_backend must be cpu, cuda, or None")
    label_map = {**DEFAULT_LABELS, **(labels or {})}
    assignments = view["assignments"]
    tiers = sorted({item["qualification_tier"] for item in assignments})
    binding_catalog, recipes, protocols, reason_catalog, rows = {}, {}, {}, {}, []
    for original in board_result.get("current_rows", []):
        if original.get("evidence_scope") == "research_diagnostic":
            continue
        runtime = original.get("runtime_cohort", {})
        if execution_backend and runtime.get("execution_backend") != execution_backend:
            continue
        cells, tasks, nonrequired = _tier_cells(original, assignments, reason_catalog)
        bindings = _catalog_bindings(original, binding_catalog, recipes, protocols)
        row = {"technique": label_map.get(original["candidate_id"], original["candidate_id"]),
               "candidate_id": original["candidate_id"], "candidate_revision": original.get("candidate_revision"),
               "cohort": original.get("cohort"), "evidence_scope": "current", "runtime_cohort": runtime,
               "status": original.get("status"), "qualified_tier": original.get("qualified_tier", 0),
               "tiers": cells, "tasks": tasks, "nonrequired_tasks": nonrequired,
               "cost": _compact_cost(original.get("cost", {})), "attempt_ids": original.get("attempt_ids", []),
               "bindings": bindings,
               "blockers": original.get("preflight_blockers", original.get("blockers", [])),
               "lifecycle": original.get("lifecycle"), "pending_readout": original.get("pending_readout", False)}
        rows.append(row)
    # Preserve the source board's attained-tier ordering. Display names do not
    # establish a second ranking or choose among outcomes of the same method.
    archived = {scope: [] for scope in ARCHIVED_SCOPES}
    for original in board_result.get("rows", []):
        scope = original.get("evidence_scope")
        if scope not in archived:
            continue
        runtime = original.get("runtime_cohort")
        if execution_backend and runtime and runtime.get("execution_backend") != execution_backend:
            continue
        archived[scope].append({
            "technique": label_map.get(original["candidate_id"], original["candidate_id"]),
            "candidate_id": original["candidate_id"], "candidate_revision": original.get("candidate_revision"),
            "cohort": original.get("cohort"), "evidence_scope": scope, "qualified_tier": None,
            "recorded_counts": {_status(key): value for key, value in original.get("counts", {}).items()},
            "recorded_total": original.get("recorded_tasks", len(original.get("task_results", []))),
            "runtime_cohort": runtime, "cost": _compact_cost(original.get("cost", {})),
            "attempt_ids": original.get("attempt_ids", []), "record_id": original.get("record_id"),
            "source": _source_pointer(original.get("source")), "prior": _compact_prior(original.get("prior")),
            "claim_contract": _compact_claim(original.get("claim_contract")), "qualification_reuse": False})
    result = {"schema_version": 1, "reducer_version": REDUCER_VERSION,
              "view": board_result["view"], "view_revision": board_result["view_revision"],
              "policy_fingerprint": board_result["policy_fingerprint"],
              "execution_backend": execution_backend, "calibration": board_result.get("calibration"),
              "tier_requirements": {str(tier): [item["task"] for item in assignments
                    if item["qualification_tier"] == tier and item["importance"] == "required"] for tier in tiers},
              "rows": rows, "task_contracts": dict(sorted(binding_catalog.items())),
              "recipe_contracts": dict(sorted(recipes.items())), "protocol_contracts": dict(sorted(protocols.items())),
              "status_reasons": dict(sorted(reason_catalog.items())),
              "archived_rows": archived, "conflicts": board_result.get("conflicts", []),
              "notes": ["Tier cells show PASS/full declared required task total; diagnostics and ranking tasks are separate.",
                        "UNKNOWN means NOT_RUN or missing evidence; FAIL, BLOCKED, INVALID and INCOMPLETE remain distinct.",
                        "All required lower-tier gates must pass before higher-tier work is eligible. Unknown later tiers are not failures.",
                        "Only the knowledge board's independently graded compatible current receipts fill current cells.",
                        "Recipe, prior, initialization, budget, sampling, source and runtime bind each cohort. Clean/noisy serving never pools.",
                        "Pinned, calibration diagnostics and historical evidence are unranked and grant no current qualification."]}
    result["provenance"] = {"source_board_sha256": stable_hash(board_result),
                            "view_sha256": stable_hash(view), "reducer_sha256": file_hash(Path(__file__))}
    result["provenance"]["input_digest"] = stable_hash(result)
    return result


def technique_board(root: Path | str, view_id: str = "discriminator_stability", *,
                    execution_backend: str | None = None, labels: dict | None = None) -> dict:
    """Read compatible Forge evidence and declarations, without starting work."""
    from . import knowledge, views
    root = Path(root)
    result = knowledge.board(root, view_id, include_bindings=True)
    return reduce_board(result, views.load_view(root, view_id),
                        execution_backend=execution_backend, labels=labels)


def _cell(value):
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def _compute(runtime):
    backend = runtime.get("execution_backend", "unrecorded")
    models = sorted({profile.get("model") for profile in runtime.get("compute_profiles", {}).values()
                     if profile and profile.get("model")})
    return backend + (" / " + ", ".join(models) if models else "")


def render_markdown(result: dict, *, json_link: str | None = None, repo_link_prefix: str = "../..") -> str:
    tiers = list(result["tier_requirements"])
    lines = ["# Forge technique inventory by tier", "",
             f"View `{result['view']}` revision {result['view_revision']}; "
             f"requested device `{result.get('execution_backend') or 'all'}`. "
             f"Calibration remains `{(result.get('calibration') or {}).get('status', 'unrecorded')}`.", "",
             "Each cell is **passes / full required total** for that exact current cohort. "
             "Rows follow Forge's attained-tier order; no aggregate quality or speed ranking is declared.", "",
             "| Technique | Cohort / exact revision | Compute | " + " | ".join(f"Tier {tier}" for tier in tiers) +
             " | Qualified tier | Other outcomes | Paid seconds |",
             "| --- | --- | --- | " + " | ".join("---:" for _ in tiers) + " | ---: | --- | ---: |"]

    def tier_cell(cell):
        held = cell.get("held", 0)
        if held and not cell["total"]:
            return f"held ({held})"
        return f"{cell['passed']}/{cell['total']}" + (f" + {held} held" if held else "")

    for row in result["rows"]:
        counts = Counter(task["status"] for task in row["tasks"] if task["status"] != "PASS")
        others = ", ".join(f"{key} {value}" for key, value in sorted(counts.items())) or "all required tasks PASS"
        held = sum(cell.get("held", 0) for cell in row["tiers"].values())
        if held:
            others += f"; {held} held (measured, not counted)"
        seconds = row["cost"].get("wall_seconds")
        identity = f"{(row.get('cohort') or 'unknown')[:12]} / {(row.get('candidate_revision') or 'unknown')[:12]}"
        label = f"[{row['technique']}]({repo_link_prefix}/configs/forge/ideas/{row['candidate_id']}.json)"
        # HELD means the view's highest tier is held; it is never a PASS.
        qualified = f"{row['qualified_tier']} (HELD)" if row.get("status") == "HELD" else row["qualified_tier"]
        values = [label, identity, _compute(row["runtime_cohort"]),
                  *[tier_cell(row["tiers"][tier]) for tier in tiers],
                  qualified, others, round(seconds, 3) if seconds is not None else "unknown"]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "UNKNOWN means missing or unrun evidence. A failed or blocked prerequisite stops later-tier "
              "spending while every declared task stays in its denominator. Nonrequired diagnostics never fill required cells.", "",
              "Current rows bind the resolved recipe, prior, initialization, full budget, sampling law, source and runtime. "
              "Different clean/noisy sampling and hardware cohorts stay separate.", ""]
    details = []
    for row in result["rows"]:
        measured = [task for task in row["tasks"] if task["status"] in {"PASS", "FAIL", "INVALID", "INCOMPLETE"}]
        blocked = next((task for task in row["tasks"] if task["status"] == "BLOCKED"), None)
        shown = measured + ([blocked] if blocked else [])
        if not shown and not row["blockers"]:
            continue
        fragments = []
        for task in shown:
            reasons = result["status_reasons"].get(task.get("reasons_sha256"), [])
            fragments.append(f"`{task['task_id']}` {task['status']}" +
                             (" — " + "; ".join(str(reason) for reason in reasons) if reasons else ""))
        if not blocked and row["blockers"]:
            reason = row["blockers"][0]
            fragments.append("preflight BLOCKED — " + (reason.get("reason", str(reason)) if isinstance(reason, dict) else str(reason)))
        evidence = ", ".join(f"[receipt `{name[:12]}`]({repo_link_prefix}/reports/forge/attempts/{name}/result.json)"
                             for name in row["attempt_ids"])
        details.append(f"- **{row['technique']} / {(row.get('cohort') or 'unknown')[:12]}:** " +
                       "; ".join(fragments) + (". " + evidence if evidence else "."))
    if details:
        lines += ["## Measured cells and first blockers", "",
                  "The earliest required blocker is shown for each cohort; full task statuses and reasons are in JSON. "
                  "A smoke failure describes this frozen profile and recipe, without measuring unrun quality tiers.", "", *details, ""]
    if not result["rows"]:
        lines += ["No current cohorts match the requested device.", ""]
    lines += ["Pinned, calibration diagnostic and historical evidence remains separate and unranked: " +
              "; ".join(f"{scope} {len(result['archived_rows'][scope])} cohorts" for scope in ARCHIVED_SCOPES) + ". "
              "Overlapping historical summaries are not independent trials or a combined cost total. "
              f"[Historical memory and original evidence]({repo_link_prefix}/reports/forge/EXPERIMENT_MEMORY.md).", ""]
    if json_link:
        lines += [f"[Full task statuses, exact scientific bindings, archived cohorts and provenance]({json_link}).", ""]
    lines += ["Regenerate after new Forge receipts or declarations with:", "", "```sh",
              f"python -m experiments.forge techniques --goal {result['view']}" +
              (f" --device {result['execution_backend']}" if result.get("execution_backend") else "") +
              " --output reports/forge/technique-inventory", "```", "",
              f"Reducer `{result['reducer_version']}`; input digest `{result['provenance']['input_digest']}`. "
              "Report generation launches no training.", ""]
    return "\n".join(lines)


def write_report(root: Path | str, view_id: str = "discriminator_stability", *,
                 execution_backend: str | None = None, labels: dict | None = None,
                 output_prefix: Path | str | None = None) -> dict:
    """Materialize compact Markdown/JSON; unchanged reports retain their bytes."""
    root = Path(root)
    prefix = Path(output_prefix) if output_prefix is not None else Path("reports/forge/technique-inventory")
    if not prefix.is_absolute():
        prefix = root / prefix
    json_path, markdown_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
    if json_path.is_file() and read_json(json_path).get("publication_scope") == "current_technique_inventory":
        raise ValueError("registered current leaderboard must be updated with "
                         "python reports/forge/regenerate_technique_inventory.py")
    result = technique_board(root, view_id, execution_backend=execution_backend, labels=labels)
    markdown = render_markdown(result, json_link=json_path.name,
                               repo_link_prefix=os.path.relpath(root.resolve(), markdown_path.parent.resolve()))
    if not json_path.exists() or read_json(json_path) != result:
        atomic_text(json_path, json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    if not markdown_path.exists() or markdown_path.read_text() != markdown:
        atomic_text(markdown_path, markdown)
    return {"report": str(markdown_path), "json": str(json_path),
            "input_digest": result["provenance"]["input_digest"], "rows": len(result["rows"])}
