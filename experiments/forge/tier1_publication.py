"""Stage and publish complete, source-bound Tier 1 measurements without training.

Run ``stage`` to inspect independently reconstructed evidence and exact pins.
Run ``publish`` after the campaign, media export and archive are complete.
Only the existing current leaderboard is updated; staged reports stay local.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from .contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from .planning import load_idea
from .scoped_publications import REGISTRY, SCHEMA, clock_audit_summary
from .trainer_families import (CURRENT_SELECTION, _current_pin, family_for_candidate,
                               family_row_pin, load_families, scientific_row_hash)
from .views import load_tasks, load_view

MAIN_VIEW = "discriminator_stability"
POLICY_VIEW = "tier1_policy_coverage"
ROUND = Path("configs/forge/rounds/tier1-completion-v1.json")
POLICY_FAMILIES = {"atlas", "e22"}
POLICY_BLOCKED_PARENTS = {"unused_token_hold", "ae_gan_hold", "five_word_joint_acquisition"}


def _publisher(root):
    path = root / "reports/forge/regenerate_technique_inventory.py"
    spec = importlib.util.spec_from_file_location("_tier1_source_publisher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_row(report, candidate, source_digest):
    rows = [row for row in report["rows"] if row["candidate_id"] == candidate
            and row.get("bindings", {}).get("source_digest") == source_digest
            and row.get("runtime_cohort", {}).get("execution_backend") == "cuda"]
    if len(rows) != 1:
        raise ValueError("publication needs one exact frozen-source row per roster candidate")
    return deepcopy(rows[0])


def prepare_selection(root, definition, results, main_report, scoped_report):
    """Require complete clean measurements; never turn measured FAIL into PASS."""
    from .planning import resolution_cache
    root = Path(root)
    with resolution_cache(root):
        return _prepare_selection_inner(root, definition, results, main_report, scoped_report)


def _prepare_selection_inner(root, definition, results, main_report, scoped_report):
    families = load_families(root)
    declarations = load_tasks(root)
    roster = {row["family"]: row for row in definition["candidate_roster"]}
    observed = {row["family"]: row for row in results["candidates"]}
    if (len(roster) != len(definition["candidate_roster"]) or len(observed) != len(results["candidates"])
            or set(roster) != set(families) or set(observed) != set(roster)
            or results["round"] != definition["id"]):
        raise ValueError("results must retain the complete frozen family roster")
    sources = {row["source"] for row in observed.values()}
    commits = {row["source_commit"] for row in observed.values()}
    if len(sources) != 1 or len(commits) != 1:
        raise ValueError("one executed source and origin are required across the round")
    source_digest = next(iter(sources))
    source_commit = next(iter(commits))
    for report, view in ((main_report, MAIN_VIEW), (scoped_report, POLICY_VIEW)):
        if (report.get("publication_scope") != "frozen_source" or report["view"] != view
                or report.get("frozen_source", {}).get("commit") != source_commit
                or report["policy_fingerprint"] != stable_hash(load_view(root, view))):
            raise ValueError("independent reports must bind the executed source and current view")
    selections, clean_rows, scoped_rows = [], {}, {}
    for family, frozen in sorted(roster.items()):
        summary = observed[family]
        if summary["candidate_id"] != frozen["candidate_id"] or summary["view"] != frozen["view"]:
            raise ValueError("results changed the frozen candidate or cohort")
        tasks = {task["task_id"]: task for task in summary["tasks"]}
        required = {assignment["task"] for assignment in load_view(root, frozen["view"])["assignments"]
                    if assignment["qualification_tier"] == 1 and assignment["importance"] == "required"}
        expected_tasks = required | set(frozen["measurement_tasks"])
        if (len(tasks) != len(summary["tasks"]) or set(tasks) != set(frozen["task_ids"])
                or set(tasks) != expected_tasks):
            raise ValueError("results changed the frozen Tier 1 task denominator")
        row = _one_row(main_report, frozen["candidate_id"], source_digest)
        row["trainer_family"] = family_for_candidate(root, row["candidate_id"], load_idea(root, row["candidate_id"]),
                                                     current_presentation=True)["id"]
        if row["trainer_family"] != family:
            raise ValueError("roster candidate differs from its registered solution family")
        if family in POLICY_FAMILIES:
            for name, task in tasks.items():
                blocked = declarations[name].get("policy_parent", {}).get("id") in POLICY_BLOCKED_PARENTS
                if task["status"] not in ({"BLOCKED"} if blocked else {"PASS", "FAIL"}):
                    raise ValueError("every runnable policy question must be measured; ownership blockers remain separate")
            if row.get("status") != "BLOCKED" or row.get("attempt_ids"):
                raise ValueError("policy parent must remain its exact unmeasured blocked clean cohort")
            pin = family_row_pin(row, selection_kind="historical_incumbent",
                reason="Exact executed-source parent cohort remains blocked. Separate selected-policy/cloud measurements grant no parent clean qualification.")
            scoped = _one_row(scoped_report, frozen["candidate_id"], source_digest)
            scoped["trainer_family"] = family
            scoped_rows[family] = scoped
            graded = {task["task_id"]: task["status"] for task in scoped["tasks"] + scoped.get("nonrequired_tasks", [])}
        else:
            if any(task["status"] not in {"PASS", "FAIL"} for task in tasks.values()):
                raise ValueError("every clean Tier 1 question and scoped clock probe must be measured PASS or FAIL")
            pin = family_row_pin(row, selection_kind="current_measurement",
                reason="Complete current Tier 1 measurement at one frozen recipe and executed source. FAIL completes a measurement; calibration and confirmation remain separate.",
                measurement_views=[MAIN_VIEW], measurement_tasks=frozen["measurement_tasks"])
            _current_pin(root, family, [row], pin, view_id=MAIN_VIEW, catalogs=main_report)
            clean_rows[family] = row
            graded = {task["task_id"]: task["status"] for task in row["tasks"] + row.get("nonrequired_tasks", [])}
        if any(graded.get(name) != task["status"] for name, task in tasks.items()):
            raise ValueError("campaign statuses differ from the independent frozen-source grade")
        selections.append(pin)
    previous = read_json(root / CURRENT_SELECTION)
    card = {**previous, "view": MAIN_VIEW, "policy_fingerprint": main_report["policy_fingerprint"],
            "selections": selections}
    return card, clean_rows, scoped_rows, source_digest, source_commit


def validate_media(root, results, rows, index):
    """Every final PASS/FAIL must have its exact certified actual-training GIF."""
    root = Path(root).resolve()
    if index.get("qualification_input") is not False or not isinstance(index.get("items"), list):
        raise ValueError("actual-training media index is required")
    media = {}
    for item in index["items"]:
        key = item["family"], item["task_id"], item["attempt_id"]
        if key in media:
            raise ValueError("duplicate actual-training GIF identity")
        media[key] = item
    verified = {}
    for candidate in results["candidates"]:
        row = rows[candidate["family"]]
        for task in candidate["tasks"]:
            if task["status"] not in {"PASS", "FAIL"}:
                continue
            attempt = task.get("attempt_id")
            if not attempt or attempt not in row.get("attempt_ids", []):
                raise ValueError("final measurement lacks its exact selected attempt")
            compact = read_json(root / "reports/forge/technique-receipts" / (attempt + ".json"))
            original = compact.get("provenance", {})
            measured = [item for item in compact["task_results"] if item["task_id"] == task["task_id"]]
            if (compact["candidate_id"] != row["candidate_id"]
                    or compact["candidate_revision"] != row["candidate_revision"]
                    or original.get("source_digest") != candidate["source"]
                    or original.get("canonical_result_hash") != task.get("canonical_result_hash")
                    or len(measured) != 1 or measured[0]["gate_status"] != task["status"]
                    or measured[0]["compatibility_key"] != task["compatibility_key"]):
                raise ValueError("final attempt, source, grade or compatibility certificate differs")
            item = media.get((candidate["family"], task["task_id"], attempt))
            if (not item or item.get("recorded_grade") != task["status"]
                    or item.get("kind") != "actual_training_saved_observations_gif"
                    or item.get("optimizer_updates_added") != 0 or item.get("sampling_draws_added") != 0
                    or type(item.get("observation_count")) is not int or item["observation_count"] < 1):
                raise ValueError("every final PASS/FAIL needs actual saved-training media")
            gif = (root / item["gif"]).resolve()
            if not gif.is_relative_to(root) or not gif.is_file() or file_hash(gif) != item["gif_sha256"]:
                raise ValueError("actual-training GIF is missing or has changed")
            if gif.read_bytes()[:6] not in {b"GIF87a", b"GIF89a"}:
                raise ValueError("actual-training media must be a GIF")
            sidecar = read_json(gif.with_suffix(".json"))
            if sidecar != {key: value for key, value in item.items() if key not in {"family", "attempt_id", "gif"}}:
                raise ValueError("actual-training GIF index differs from its provenance receipt")
            verified[attempt] = original["canonical_result_hash"]
    return verified


def stage(root, source_commit, results_dir, *, stage_dir=None, publisher=None):
    root = Path(root).resolve()
    results_dir = Path(results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    results_dir = results_dir.resolve()
    if not results_dir.is_relative_to(root):
        raise ValueError("compact results and publication media must live in the repository")
    commit = subprocess.check_output(["git", "rev-parse", str(source_commit) + "^{commit}"], cwd=root, text=True).strip()
    definition = json.loads(subprocess.check_output(["git", "show", commit + ":" + ROUND.as_posix()], cwd=root, text=True))
    if definition != read_json(root / ROUND):
        raise ValueError("current roster differs from the frozen executed-source declaration")
    results, media = [read_json(results_dir / name) for name in ("results.json", "media.json")]
    if {row["source_commit"] for row in results["candidates"]} != {commit}:
        raise ValueError("requested source commit differs from the completed campaign")
    publisher = publisher or _publisher(root)
    local_stages = root / "runs/forge"
    local_stages.mkdir(parents=True, exist_ok=True)
    if stage_dir:
        stage_dir = Path(stage_dir)
        stage_dir = (root / stage_dir if not stage_dir.is_absolute() else stage_dir).resolve()
        if stage_dir.is_relative_to(root) and not stage_dir.is_relative_to(local_stages):
            raise ValueError("staged reports must stay under ignored runs/forge or outside the repository")
    else:
        stage_dir = Path(tempfile.mkdtemp(prefix="tier1-publication-", dir=local_stages))
    stage_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for view in (MAIN_VIEW, POLICY_VIEW):
        metadata = publisher.regenerate(root, view_id=view, execution_backend="cuda", source_commit=commit,
                                         output_prefix=stage_dir / view)
        reports[view] = read_json(metadata["json"])
    card, clean, scoped, digest, origin = prepare_selection(root, definition, results, reports[MAIN_VIEW], reports[POLICY_VIEW])
    verified = validate_media(root, results, {**clean, **scoped}, media)
    final_measurements = []
    clock_audits = []
    for candidate in results["candidates"]:
        row = (clean | scoped)[candidate["family"]]
        for task in candidate["tasks"]:
            if task["status"] not in {"PASS", "FAIL"}:
                continue
            measured = {key: task[key] for key in ("task_id", "status", "attempt_id", "canonical_result_hash", "compatibility_key")}
            measured.update(family=candidate["family"], candidate_id=candidate["candidate_id"],
                            candidate_revision=row["candidate_revision"], source_digest=digest)
            final_measurements.append(measured)
            if task["task_id"].startswith("clockfree_"):
                audit = clock_audit_summary(root, measured)
                if audit is None:
                    raise ValueError("final clock measurement lacks its certified comparison evidence")
                clock_audits.append(audit)
    scoped_report = reports[POLICY_VIEW]
    for row in scoped.values():
        for attempt in row.get("attempt_ids", []):
            summary = read_json(root / "reports/forge/technique-receipts" / (attempt + ".json"))
            verified[attempt] = summary["provenance"]["canonical_result_hash"]
    document = {"schema": "forge_tier1_scoped_evidence_v1", "qualification_input": False,
                "parent_qualification_credit": False, "source_commit": origin, "source_digest": digest,
                "view": POLICY_VIEW, "policy_fingerprint": scoped_report["policy_fingerprint"],
                "rows": scoped, "scientific_rows_sha256": {family: scientific_row_hash(row) for family, row in scoped.items()},
                "task_contracts": scoped_report["task_contracts"], "status_reasons": scoped_report.get("status_reasons", {}),
                "receipt_result_hashes": verified, "frozen_report_input_digest": scoped_report["provenance"]["input_digest"],
                "final_measurements": final_measurements, "clock_audits": clock_audits,
                "projector_sha256": file_hash(Path(__file__))}
    document_path = results_dir / "scoped-evidence.json"
    data = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    registry = read_json(root / REGISTRY) if (root / REGISTRY).is_file() else {"schema": SCHEMA, "qualification_input": False, "cohorts": []}
    registry["cohorts"] = [entry for entry in registry["cohorts"] if entry.get("view") != POLICY_VIEW] + [
        {"view": POLICY_VIEW, "path": document_path.relative_to(root).as_posix(),
         "sha256": hashlib.sha256(data.encode()).hexdigest()}]
    registry["media"] = {"path": (results_dir / "media.json").relative_to(root).as_posix(), "sha256": file_hash(results_dir / "media.json")}
    packet = {"schema": "forge_tier1_publication_stage_v1", "root": str(root), "source_commit": origin,
              "results_dir": str(results_dir), "selection": card, "scoped_document": document,
              "registry": registry, "clean_families": sorted(clean), "policy_families": sorted(scoped),
              "measured_counts": dict(Counter(task["status"] for row in results["candidates"] for task in row["tasks"])),
              "tracked_destinations": [CURRENT_SELECTION.as_posix(), REGISTRY.as_posix(), document_path.relative_to(root).as_posix()]}
    atomic_json(stage_dir / "publication-stage.json", packet)
    return packet, stage_dir


def publish(packet, *, publisher=None):
    """Apply an already verified packet; rejected publication restores staged inputs."""
    root = Path(packet["root"])
    publisher = publisher or _publisher(root)
    paths = [root / relative for relative in packet["tracked_destinations"]]
    before = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        for path, value in zip(paths, (packet["selection"], packet["registry"], packet["scoped_document"])):
            atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        manifest = read_json(root / publisher.EVIDENCE_MANIFEST)
        advancing = manifest["policy_fingerprint"] != packet["selection"]["policy_fingerprint"]
        result = publisher.publish_current(root, source_commit=packet["source_commit"], execution_backend="cuda",
                                           advance_policy=advancing)
    except Exception:
        for path, original in before.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                atomic_text(path, original.decode())
        raise
    receipt = {"schema": "forge_tier1_publication_receipt_v1", "qualification_input": False,
               "source_commit": packet["source_commit"], "publication": result,
               "selection_sha256": file_hash(root / CURRENT_SELECTION),
               "scoped_registry_sha256": file_hash(root / REGISTRY), "measured_counts": packet["measured_counts"],
               "clean_families": packet["clean_families"], "policy_families": packet["policy_families"]}
    atomic_json(Path(packet["results_dir"]) / "publication.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage", "publish"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path)
    args = parser.parse_args(argv)
    packet, directory = stage(args.root, args.source_commit, args.results_dir, stage_dir=args.stage_dir)
    output = publish(packet) if args.stage == "publish" else {"stage_dir": str(directory),
             "source_commit": packet["source_commit"], "measured_counts": packet["measured_counts"],
             "clean_families": packet["clean_families"], "policy_families": packet["policy_families"]}
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
