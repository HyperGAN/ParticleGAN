"""Update the single current technique leaderboard from validated evidence.

Default regeneration uses committed numerical snapshots and compact receipts.
Current selection pins one whole ordinary evidence row per formulation family.
--source-commit independently regrades hydrated original receipts and registers
new measured rows before updating the same leaderboard. No command trains.
--recorded-policy rebuilds existing rows under their exact archived view policy.
--advance-policy explicitly archives an earlier policy before registering new
source evidence and rebinding the same selected configurations under the current
view. Archived outcomes and selection cards are preserved without regrade.
--refresh-publication updates display evidence and declared-view coverage while
preserving verified selected rows and their recorded qualification policy.
Always pass it after a view revision, such as a held or removed test: without
it, only a change to the registered view itself switches to the refresh.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.forge.contracts import atomic_text, file_hash, identifier, read_json, stable_hash
from experiments.forge.publication_storage import compact_current_publication, publication_json_text
from experiments.forge.technique_board import render_markdown, write_report


@contextmanager
def _prefer_recorded_and_canonical_declarations(root, recorded_candidates=()):
    """Keep real receipt identities, then canonical unmeasured aliases."""
    from experiments.forge import planning
    from experiments.forge.trainer_families import load_families
    canonical = {family["canonical_candidate"] for family in load_families(root).values()}
    recorded = set(recorded_candidates)
    original = planning.declaration_paths
    def preferred(directory):
        return sorted(original(directory),
                      key=lambda path: (path.stem not in recorded, path.stem not in canonical, path.stem))
    planning.declaration_paths = preferred
    try:
        yield
    finally:
        planning.declaration_paths = original


@contextmanager
def _missing_declaration_enumeration(root, requested):
    """Read missing metadata only; keep the full catalog for exact lookup."""
    from experiments.forge import planning
    requested = set(requested)
    original_paths, original_lookup = planning.declaration_paths, planning.load_idea
    def missing(directory):
        paths = original_paths(directory)
        if Path(directory).resolve() != root:
            return paths
        paths = [p for p in paths if read_json(p).get("id", p.stem) in requested]
        if {read_json(p).get("id", p.stem) for p in paths} != requested:
            raise ValueError("current fallback declaration roster is incomplete")
        return paths
    def lookup(directory, idea_id):
        previous = planning.declaration_paths
        planning.declaration_paths = original_paths
        try:
            return original_lookup(directory, idea_id)
        finally:
            planning.declaration_paths = previous
    planning.declaration_paths, planning.load_idea = missing, lookup
    try:
        yield
    finally:
        planning.declaration_paths, planning.load_idea = original_paths, original_lookup


@contextmanager
def _unchanged_fallback_source_inspection(root, manifests):
    """Memoize source inspection during one read-only metadata reduction."""
    if not manifests:
        yield None
        return
    from experiments.forge import sources, planning
    def verify():
        for manifest in manifests.values():
            if stable_hash(manifest["files"]) != manifest["digest"]:
                raise ValueError("current fallback source manifest differs")
            for relative, digest in manifest["files"].items():
                if file_hash(root / relative) != digest:
                    raise ValueError("current fallback scientific source changed: " + relative)
    verify()
    original = sources.inspect_source
    aliases = [(sources, "inspect_source", original), (planning, "inspect_source", planning.inspect_source)]
    policy = sys.modules.get("experiments.forge.policy_execution")
    if policy is not None:
        aliases.append((policy, "inspect_source", policy.inspect_source))
    cache = {}
    counts = dict(hits=0, misses=0, other_root_bypasses=0,
                  verified_source_files=len(set().union(*(set(m["files"]) for m in manifests.values()))))
    def inspection(directory, extra_paths=()):
        directory, extras = Path(directory).resolve(), tuple(sorted(str(p) for p in extra_paths))
        if directory != root:
            counts["other_root_bypasses"] += 1
            return original(directory, extras)
        key = (str(directory), extras)
        if key not in cache:
            counts["misses"] += 1
            cache[key] = original(directory, extras)
        else:
            counts["hits"] += 1
        return deepcopy(cache[key])
    for module, name, _ in aliases:
        setattr(module, name, inspection)
    try:
        yield counts
    finally:
        for module, name, value in aliases:
            setattr(module, name, value)
        verify()


def _scalars(value):
    """Preserve scalar final summaries and mappings, never tensor/trace arrays."""
    if isinstance(value, dict):
        return {key: _scalars(item) for key, item in value.items()
                if not isinstance(item, list) and key not in {"raw", "applied", "stream", "streams", "state", "states"}}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("summary field is not a scalar or scalar mapping")


def _evaluator_summary(evaluator):
    summary = _scalars(evaluator)
    # Transfer evaluators return one final threshold row per metric, not a
    # trajectory. Store these by metric name while dropping arbitrary arrays.
    metrics = evaluator.get("metrics")
    if isinstance(metrics, list):
        summary["metric_checks"] = {item["metric"]: _scalars(item) for item in metrics
                                    if isinstance(item, dict) and isinstance(item.get("metric"), str)}
    terminal = evaluator.get("terminal_checks")
    if isinstance(terminal, list):
        summary["terminal_summary"] = {
            "checks": len(terminal), "passing_checks": sum(item.get("passed") is True for item in terminal),
            "status_counts": dict(sorted(Counter(item.get("status", "PASS" if item.get("passed") else "FAIL")
                                                    for item in terminal).items()))}
    return summary


def project_receipt(root: Path | str, attempt_id: str) -> dict:
    """Validate an original durable receipt, then project display-only content."""
    root = Path(root).resolve()
    identifier(attempt_id, "attempt")
    directory = root / "reports/forge/attempts" / attempt_id
    paths = {name: directory / f"{name}.json" for name in ("request", "evidence", "result")}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError(f"hydrate original request/evidence/result receipts before publication: {attempt_id}")
    resolved, certificate, result = (read_json(paths[name]) for name in ("request", "evidence", "result"))
    request = resolved.get("request", resolved)
    result_hash = stable_hash(result)
    if certificate.get("result_hash") != result_hash:
        raise ValueError(f"invalid result hash certificate: {attempt_id}")
    if certificate.get("source") != request.get("source"):
        raise ValueError(f"invalid source certificate: {attempt_id}")
    if certificate.get("runtime") != request.get("runtime"):
        raise ValueError(f"invalid runtime certificate: {attempt_id}")
    source = request.get("source", {})
    if "files" in source and stable_hash(source["files"]) != source.get("digest"):
        raise ValueError(f"invalid source manifest digest: {attempt_id}")
    if result.get("candidate_revision") != request.get("candidate_revision"):
        raise ValueError(f"result/request candidate revision mismatch: {attempt_id}")
    if result.get("attempt_id") != attempt_id:
        raise ValueError(f"result/directory attempt identity mismatch: {attempt_id}")
    if result.get("retry_of") != resolved.get("retry_of"):
        raise ValueError(f"result/request retry identity mismatch: {attempt_id}")
    keys = {member: job["compatibility_key"] for job in request.get("jobs", [])
            for member in job.get("task_ids", [job["task_id"]])}
    projected = []
    for row in result.get("task_results", []):
        if row.get("task_id") not in keys or row.get("compatibility_key") != keys[row["task_id"]]:
            raise ValueError(f"task result/request compatibility mismatch: {attempt_id}")
        compact = {key: _scalars(row[key]) for key in (
            "task_id", "compatibility_key", "gate_status", "status", "raw_status", "reason", "metrics", "cost",
            "device", "execution_path", "api_version", "claim_contract") if key in row}
        compact["reasons"] = [reason for reason in row.get("reasons", []) if isinstance(reason, str)]
        compact["sampling"] = {key: _scalars(row.get("evidence", {})[key]) for key in (
            "sampling_contract_version", "sampling_law", "eval_output_noise", "scoring_weights")
                               if key in row.get("evidence", {})}
        compact["evaluator_summary"] = _evaluator_summary(row.get("evaluator_result", {}))
        projected.append(compact)
    summary = {
        "schema_version": 1, "summary_version": "forge-technique-receipt-summary-v1",
        "evidence_scope": "published_summary", "qualification_reuse": False,
        "qualification_input": False, "certificate_validated": True,
        "note": "Display summary only. Regrading requires the byte-exact archived original receipts; this projection cannot qualify a technique.",
        "attempt_id": attempt_id, "candidate_id": request.get("candidate", {}).get("id"),
        "candidate_revision": request.get("candidate_revision"), "request_id": request.get("request_id", resolved.get("request_id")),
        "campaign_id": request.get("campaign_id"), "runtime": request.get("runtime"),
        "attempt_status": result.get("raw", {}).get("attempt_status"), "cost_owner": result.get("cost_owner"),
        "task_results": projected,
        "provenance": {
            "canonical_result_hash": result_hash, "source_digest": source.get("digest"),
            "source_origin_commit": source.get("origin_commit"),
            "original_files": {name: {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
                               for name, path in paths.items()}}}
    if result.get("retry_of"):
        summary["retry_of"] = deepcopy(result["retry_of"])
    return summary


def _json_text(value):
    return publication_json_text(value)


def _write_changed(path, content):
    if not path.exists() or path.read_text() != content:
        atomic_text(path, content)


def _publication_provenance(result, summaries):
    """Bind publication identity to recorded science, excluding mutable Git HEAD."""
    origins = {}
    for summary in summaries.values():
        provenance = summary["provenance"]
        if provenance.get("source_origin_commit") is not None:
            origins.setdefault(provenance["source_digest"], set()).add(provenance["source_origin_commit"])
    for row in [*result["rows"], *result.get("configuration_rows", [])]:
        bindings = row.setdefault("bindings", {})
        recorded = sorted(origins.get(bindings.get("source_digest"), set()))
        bindings["source_origin_commit"] = recorded[0] if len(recorded) == 1 else None
        bindings["recorded_source_origin_commits"] = recorded
    upstream = result.get("provenance", {})
    # The upstream board digest includes freshly resolved request metadata such
    # as HEAD. It cannot identify an immutable publication of unchanged science.
    result["provenance"] = {
        key: upstream[key] for key in ("view_sha256", "reducer_sha256") if key in upstream}
    result["provenance"].update(
        publication_reducer_version="forge-technique-publication-v1",
        publication_reducer_sha256=file_hash(Path(__file__)),
        qualified_receipts={attempt: {
            "canonical_result_hash": summary["provenance"]["canonical_result_hash"],
            "source_digest": summary["provenance"]["source_digest"],
            "original_file_sha256": {name: item["sha256"] for name, item in
                                      summary["provenance"]["original_files"].items()},
            **({"retry_of": deepcopy(summary["retry_of"])} if summary.get("retry_of") else {})}
            for attempt, summary in sorted(summaries.items())})
    result["provenance"]["input_digest"] = stable_hash(result)


def _resolve_commit(root, source_commit):
    try:
        return subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options",
                                        str(source_commit) + "^{commit}"], cwd=root,
                                       text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"unknown source commit: {source_commit}") from error


def _receipt_manifests(root, commit):
    """Locate validated original science recorded at this exact Git origin."""
    manifests = {}
    for path in sorted((root / "reports/forge/attempts").glob("*/request.json")):
        resolved = read_json(path)
        request = resolved.get("request", resolved)
        source = request.get("source", {})
        if source.get("origin_commit") != commit:
            continue
        project_receipt(root, path.parent.name)
        if not isinstance(source.get("files"), dict):
            raise ValueError("frozen source publication requires the original complete source manifest")
        previous = manifests.setdefault(source["digest"], source)
        if previous["files"] != source["files"]:
            raise ValueError("conflicting original frozen source manifests")
    if not manifests:
        raise ValueError(f"no hydrated original receipts bind source commit {commit}")
    return manifests


def _recorded_studies(root, manifests):
    """Bind report reconstruction to original, admitted study requests only."""
    from experiments.forge.technique_inventory import _signature
    records, support = {}, {}
    for path in sorted((root / "reports/forge/attempts").glob("*/request.json")):
        resolved = read_json(path)
        request = resolved.get("request", resolved)
        source = request.get("source", {})
        expected = manifests.get(source.get("digest"))
        if expected is None or source.get("origin_commit") != expected.get("origin_commit"):
            continue
        if source.get("files") != expected.get("files"):
            raise ValueError("recorded study source manifest differs from validated originals")
        candidate = request.get("candidate", {})
        if "study" not in request:
            if (candidate.get("schema_version") == 3 and "configuration_id" not in candidate
                    and "decision_contract" not in candidate):
                raise ValueError("recorded v3 candidate is missing its admitted study binding")
            continue
        study, admission, review = request["study"], request.get("study_admission", {}), request.get("study_review", {})
        if (study.get("candidate") != candidate.get("id") or study.get("status") != "ready"
                or study.get("scope", {}).get("view") != request.get("view", {}).get("id")
                or study.get("scope", {}).get("execution_backend") != request.get("execution_backend")
                or study.get("scope", {}).get("through_tier") != request.get("through_tier")
                or admission.get("study_sha256") != stable_hash(study)
                or admission.get("policy_fingerprint") != request.get("policy_fingerprint")
                or admission.get("protocol_sha256") != stable_hash(request.get("protocol"))
                or review.get("receipt") != admission or review.get("status") != "READY"
                or review.get("blockers") or request.get("preflight_blockers")):
            raise ValueError("recorded study declaration/admission fingerprint is missing or mismatched")
        prior = review.get("expected", {}).get("prior_evidence", [])
        if len(prior) != len(study["prior_evidence"]):
            raise ValueError("recorded study prior-evidence binding is missing")
        for declared, bound in zip(study["prior_evidence"], prior):
            binding = {key: value for key, value in bound.items() if key != "sha256"}
            name, digest = bound.get("path"), bound.get("sha256")
            if (binding != declared or not isinstance(name, str) or Path(name).is_absolute()
                    or ".." in Path(name).parts or Path(name).suffix != ".json"
                    or not isinstance(digest, str) or len(digest) != 64):
                raise ValueError("recorded study prior-evidence identity is missing or unsafe")
            if name in support and support[name] != digest:
                raise ValueError("conflicting recorded study prior-evidence hashes")
            support[name] = digest
        identity = stable_hash({"study": study, "admission": admission, "review": review, "signature": _signature(request),
                                "source": {key: source[key] for key in ("origin_commit", "digest", "files")}})
        record = records.setdefault(identity, {"request": request, "original_receipts": []})
        record["original_receipts"].append({"request_path": path.relative_to(root).as_posix(), "request_sha256": file_hash(path)})
    return list(records.values()), support


def _recorded_study_question(root, idea_id, view_id, execution_backend, cuda_model, records, proofs):
    """Standalone adapter also embedded in the isolated frozen reader."""
    from experiments.forge import planning
    from experiments.forge.contracts import stable_hash
    matched = [record for record in records
               if record["request"]["candidate"]["id"] == idea_id
               and record["request"]["view"]["id"] == view_id
               and record["request"]["execution_backend"] == execution_backend
               and (cuda_model is None or record["request"]["compute_profiles"].get("cuda", {}).get("model") == cuda_model)]
    if not matched:
        idea = planning.load_idea(root, idea_id)
        tier = idea.get("decision_contract", {}).get("scope", {}).get("through_tier", 3)
        return planning.resolve_idea(root, idea_id, view_id=view_id, through_tier=tier,
                                    freeze_source=False, execution_backend=execution_backend, cuda_model=cuda_model)
    from experiments.forge.technique_inventory import _signature
    if len(matched) != 1:
        raise ValueError("ambiguous original study execution binding for this source/view/cohort")
    record, original = matched[0], matched[0]["request"]
    current = planning.resolve_idea(root, idea_id, study=original["study"]["id"],
                                    view_id=view_id, through_tier=original["through_tier"], freeze_source=False,
                                    execution_backend=execution_backend, cuda_model=cuda_model)
    if (current.get("study") != original["study"]
            or current.get("study_admission") != original["study_admission"]
            or current.get("study_review") != original["study_review"]
            or current.get("study_review", {}).get("status") != "READY"
            or current.get("preflight_blockers")
            or current.get("source", {}).get("digest") != original["source"]["digest"]
            or current.get("source", {}).get("files") != original["source"]["files"]
            or current.get("view") != original["view"] or _signature(current) != _signature(original)):
        differences = [key for key in ("study", "study_admission", "study_review", "view")
                       if current.get(key) != original.get(key)]
        if current.get("study_review", {}).get("status") != "READY":
            differences.append("study_review.status")
        if current.get("preflight_blockers"):
            differences.append("preflight_blockers")
        for key in ("digest", "files"):
            if current.get("source", {}).get(key) != original["source"][key]:
                differences.append("source." + key)
        if _signature(current) != _signature(original):
            differences.append("scientific_signature")
        raise ValueError("reconstructed study/source/view/cohort differs from the original admitted request: "
                         + ", ".join(differences) + "; preflight=" + repr(current.get("preflight_blockers"))
                         + "; study_blockers=" + repr(current.get("study_review", {}).get("blockers")))
    proof = {"candidate_id": idea_id, "study_id": original["study"]["id"],
             "study_sha256": stable_hash(original["study"]), "admission_sha256": stable_hash(original["study_admission"]),
             "request_signature": _signature(original), "source_digest": original["source"]["digest"],
             "recorded_source_origin_commit": original["source"]["origin_commit"],
             "policy_fingerprint": original["policy_fingerprint"], "original_receipts": record["original_receipts"]}
    if proof not in proofs:
        proofs.append(proof)
    return current


def _validate_comparison_policy(actual, comparison):
    """Permit older assignment dispositions and additive later tasks, never reordered or retiered tasks."""
    metadata = {"revision", "previous_revision", "policy_change_reason"}
    if (comparison.get("id") != actual.get("id")
            or type(comparison.get("revision")) is not int
            or comparison["revision"] >= actual["revision"]
            or {key: value for key, value in actual.items() if key not in metadata | {"assignments"}}
               != {key: value for key, value in comparison.items() if key not in metadata | {"assignments"}}):
        raise ValueError("comparison policy must change only older assignment dispositions")
    actual_tasks = [assignment["task"] for assignment in actual["assignments"]]
    comparison_tasks = [assignment["task"] for assignment in comparison["assignments"]]
    if len(set(actual_tasks)) != len(actual_tasks) or len(set(comparison_tasks)) != len(comparison_tasks):
        raise ValueError("comparison policy changes task identities, tiers or ordering")
    def structural(view):
        return [{key: value for key, value in assignment.items()
                 if key not in {"importance", "hold", "stretch", "removal"}}
                for assignment in view["assignments"]]
    retained = set(comparison_tasks)
    if [item for item in structural(actual) if item["task"] in retained] != structural(comparison):
        raise ValueError("comparison policy changes task identities, tiers or ordering")


def _comparison_contracts_match(row, comparison):
    family = row["bindings"]["trainer_family"]
    actual_contracts = row["bindings"]["task_contracts"]
    added = set(comparison.get("added_actual_tasks", ()))
    required = {item["task"] for item in comparison.get("policy", {}).get("assignments", ())}
    reference = comparison["reference_task_contracts"].get(family)
    if not isinstance(reference, dict):
        return (family in set(comparison.get("new_registered_families", ()))
                and isinstance(actual_contracts, dict)
                and required.issubset(actual_contracts)
                and set(actual_contracts) == required | added)
    if not required.issubset(reference) or set(actual_contracts) != set(reference) | added:
        return False
    lineage = comparison.get("verified_evaluator_lineage", {}).get(family, {})
    return all(actual_contracts.get(task) in {digest, lineage.get(task)}
               for task, digest in reference.items())


def _comparison_projection(actual, projected, comparison):
    """Bind a policy-only projection to the original measured rows and tasks."""
    from copy import deepcopy
    from experiments.forge.contracts import stable_hash
    originals_by_identity = {(row["candidate_id"], row.get("candidate_revision"), row.get("cohort")): row
                             for row in actual["rows"]}
    originals = {row["candidate_revision"]: row for row in actual["rows"] if row.get("attempt_ids")}
    added = set(comparison.get("added_actual_tasks", ()))
    if added:
        projected.setdefault("status_reasons", {}).update(deepcopy(actual.get("status_reasons", {})))
        for row in projected["rows"]:
            source_row = originals_by_identity.get((row["candidate_id"], row.get("candidate_revision"), row.get("cohort")))
            if source_row is None:
                continue
            existing = {item["task_id"] for key in ("tasks", "nonrequired_tasks") for item in row[key]}
            extra = [deepcopy(item) for key in ("tasks", "nonrequired_tasks")
                     for item in source_row[key] if item["task_id"] in added and item["task_id"] not in existing]
            if extra:
                row["nonrequired_tasks"].extend(extra)
    observed = set()
    def statuses(row):
        return {item["task_id"]: item["status"] for key in ("tasks", "nonrequired_tasks") for item in row[key]}
    for row in projected["rows"]:
        if not row.get("attempt_ids"):
            continue
        original = originals.get(row["candidate_revision"])
        if (original is None or not _comparison_contracts_match(row, comparison)
                or any(row.get(key) != original.get(key) for key in
                       ("candidate_id", "candidate_revision", "cohort", "bindings", "runtime_cohort", "attempt_ids", "cost"))):
            raise ValueError("comparison projection changes recorded task contracts or measured cohort")
        if statuses(row) != statuses(original):
            raise ValueError("comparison policy projection changed numerical task verdicts")
        calibrated_lineage = comparison.get("verified_evaluator_lineage", {}).get(
            row["bindings"]["trainer_family"], {})
        items_by_task = {item["task_id"]: item for key in ("tasks", "nonrequired_tasks") for item in row[key]}
        for tid in comparison.get("calibrated_grid_triad_tasks", ()):
            if row["bindings"]["task_contracts"].get(tid) != calibrated_lineage.get(tid):
                continue
            item = items_by_task.get(tid, {})
            if item.get("status") != "FAIL":
                metrics = item.get("metrics") or {}
                prec = metrics.get("precision")
                if not isinstance(prec, (int, float)) or float(prec) < 0.97:
                    raise ValueError(
                        "comparison policy projection cannot grant Rev-9 PASS under calibrated min_precision=0.75: "
                        + str(tid)
                    )
        observed.add(row["candidate_revision"])
        row["executed_policy_qualification"] = dict(
            view_revision=actual["view_revision"], policy_fingerprint=actual["policy_fingerprint"],
            status=original["status"], qualified_tier=original["qualified_tier"], tiers=deepcopy(original["tiers"]),
            note="Separate actual executed-policy result; this does not change the projected comparison tiers.")
    if observed != set(originals):
        raise ValueError("comparison projection lost an original measured row")
    qualification = {key: deepcopy(actual[key]) for key in
                     ("view", "view_revision", "policy_fingerprint", "tier_requirements")}
    qualification["rows"] = deepcopy(list(originals.values()))
    projected["comparison_policy_projection"] = dict(
        path=comparison["path"], sha256=comparison["sha256"],
        policy_fingerprint=stable_hash(comparison["policy"]), actual_policy_fingerprint=actual["policy_fingerprint"],
        task_contracts_unchanged=True, numerical_verdicts_unchanged=True,
        original_admissions_reconstructed_before_projection=True, qualification_input=False, qualification_reuse=False,
        note="Projected tiers use the older comparison policy only. Actual executed-policy qualification is separate; no projected Tier 3 claim follows from an actual-policy pass.",
        actual_qualification=qualification, actual_qualification_sha256=stable_hash(qualification))



def _extract_source(root, commit, destination, manifests, study_support=None):
    for name, digest in (study_support or {}).items():
        if (Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".json"
                or not isinstance(digest, str) or len(digest) != 64):
            raise ValueError("unsafe recorded study prior-evidence path/hash")
    # Historical reports dominate this repository's 924 MB checkout. Extract
    # all scientific roots/configs plus explicitly bound support instead.
    top_level = set(subprocess.check_output(["git", "ls-tree", "--name-only", commit], cwd=root, text=True).splitlines())
    roots = {name for name in ("particlegan", "experiments", "benchmarks", "lib", "configs", "docs", "pyproject.toml")
             if name in top_level}
    extras = {name for source in manifests.values() for name in source["files"]
              if name.split("/", 1)[0] not in roots}
    # Study motivation is not scientific source, but its exact admitted Git
    # bytes must be inside this checkout: live report symlinks are not evidence.
    extras.update(study_support or {})
    archive = destination.parent / "frozen-source.tar"
    try:
        with archive.open("wb") as output:
            subprocess.run(["git", "archive", "--format=tar", commit, "--", *sorted(roots | extras)],
                           cwd=root, stdout=output, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError("source commit cannot reconstruct its recorded scientific files") from error
    destination.mkdir()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isdir() or member.isfile()):
                raise ValueError("unsafe member in frozen source archive")
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as stream:
                    target.write_bytes(stream.read())
    for name, digest in (study_support or {}).items():
        if not (destination / name).is_file() or file_hash(destination / name) != digest:
            raise ValueError("frozen Git study prior-evidence differs from its original admission hash")
    closure = _catalog_support_closure(_frozen_catalog_support(destination), manifests, study_support or {})
    closure["frozen_rule_files_sha256"] = {
        name: file_hash(destination / name) for name in
        ("experiments/forge/planning.py", "experiments/forge/hostprofiles.py") if (destination / name).is_file()}
    _overlay_reports(root / "reports", destination / "reports",
                     excluded=[Path(name).relative_to("reports") for name in closure["preserved_absent_paths"]])
    return closure


def _frozen_catalog_support(root):
    """Ask the executed planner's helpers which optional catalog files it sees."""
    program = """
import json, sys
from pathlib import Path
from experiments.forge.views import load_tasks
try:
    from experiments.forge.hostprofiles import profile_source_paths
except ImportError:
    profile_source_paths = None
paths = set()
for task in load_tasks(Path(sys.argv[1])).values():
    paths.update(task.get('evaluation', {}).get('sources', {}))
    if profile_source_paths is not None:
        try:
            paths.update(profile_source_paths(task))
        except (KeyError, TypeError, ValueError):
            pass
print(json.dumps(sorted(paths)))
"""
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    output = subprocess.check_output([os.path.abspath(sys.executable), "-c", program, str(root)],
                                     cwd=root, env=environment, text=True, stderr=subprocess.PIPE)
    return json.loads(output)


def _catalog_support_closure(support, manifests, study_support=()):
    """Retain captured optional files and preserve recorded report-file absence."""
    absent, present = [], []
    for name in sorted(set(support)):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe frozen catalog support path")
        if not path.parts or path.parts[0] != "reports":
            continue
        captured = {name in source["files"] for source in manifests.values()}
        if len(captured) != 1:
            raise ValueError("conflicting captured catalog support presence; publish source cohorts separately: " + name)
        if True in captured:
            present.append(name)
        elif name not in study_support:
            absent.append(name)
    return {"captured_present_paths": present, "preserved_absent_paths": absent,
            "source_manifest_digests": sorted(manifests),
            "scope": "Frozen planner catalog evaluator/profile inventory only; captured source and original study support retain their exact bytes. Read-only report overlays cannot introduce previously absent optional source files.",
            "qualification_reuse": False}


def _overlay_reports(original, destination, *, excluded=()):
    """Expose current knowledge read-only while retaining frozen report helpers."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in original.iterdir():
        relative = [Path(path) for path in excluded if Path(path).parts[0] == source.name]
        if any(len(path.parts) == 1 for path in relative):
            continue
        target = destination / source.name
        if relative and source.is_dir():
            # Do not symlink an ancestor of an excluded optional source: the
            # missing descendant must remain absent without touching live data.
            target.mkdir(parents=True, exist_ok=True)
            _overlay_reports(source, target, excluded=[Path(*path.parts[1:]) for path in relative])
        elif not target.exists():
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        elif source.is_dir() and target.is_dir():
            _overlay_reports(source, target)


def _validate_recorded_study_projection(studies, result, view_id, execution_backend):
    """An unresolved declaration cannot erase a completed original study."""
    from experiments.forge.views import diagnostic_evidence_scope
    expected = {record["request"]["candidate"]["id"] for record in studies
                if record["request"]["view"]["id"] == view_id
                and (execution_backend is None or record["request"]["execution_backend"] == execution_backend)
                and not diagnostic_evidence_scope(record["request"])}
    reconstructed = {proof["candidate_id"] for proof in result.get("registered_study_reconstruction", [])}
    measured = {row["candidate_id"] for row in result["rows"] if row.get("attempt_ids")}
    if not expected.issubset(reconstructed & measured):
        raise ValueError("frozen publication lost recorded study admission or measured rows: "
                         + ", ".join(sorted(expected - (reconstructed & measured))))


def _publication_rows(result):
    """Retain every independently graded configuration before family selection.

    The frozen registry may predate a study's activation. Its family default
    rows are a display selection, while configuration_rows retain all trials.
    A numerical snapshot must preserve that full roster and its receipt proofs.
    """
    if "configuration_rows" not in result:
        return result["rows"]
    display_fields = {"technique", "selection", "selected_configuration", "alternative_scope"}
    rows, seen = [], {}
    for row in [*result["configuration_rows"], *result["rows"]]:
        identity = (row["candidate_id"], row.get("candidate_revision"), row.get("cohort"))
        science = {key: value for key, value in row.items() if key not in display_fields}
        if identity in seen:
            if seen[identity] != science:
                raise ValueError("conflicting scientific rows for one publication identity")
            continue
        seen[identity] = science
        rows.append(deepcopy(row))
    return rows


def _bound_publication_rows(result, allowed_sources):
    """Separate unresolved declarations from reconstructed scientific cohorts."""
    bound, unresolved = [], []
    for row in _publication_rows(result):
        bindings = row.get("bindings", {})
        source = bindings.get("source_digest")
        if source in allowed_sources:
            bound.append(row)
            continue
        # A declaration that failed resolution has no source cohort to attest.
        # Retain its complete blocker diagnostics, but register no task result,
        # attempt identity or scientific binding from it. Actual source claims
        # and attempted/graded rows still require validated receipt manifests.
        tasks = row.get("tasks", []) + row.get("nonrequired_tasks", [])
        if (source is None and bindings.get("available") is False
                and not row.get("attempt_ids") and row.get("status") == "BLOCKED"
                and row.get("candidate_revision") is None and row.get("cohort") is None
                and row.get("qualified_tier") == 0 and row.get("blockers")
                and not bindings.get("task_contracts")
                and not any(bindings.get(key) for key in ("recipe_sha256", "protocol_sha256", "rng_sha256"))
                and row.get("cost", {}).get("measured_tasks", 0) == 0
                and tasks and all(task.get("status") == "BLOCKED" for task in tasks)):
            unresolved.append(row)
            continue
        raise ValueError("frozen report contains a source cohort absent from the validated original receipts: "
                         + row["candidate_id"] + " / " + str(source))
    return bound, unresolved


def _evaluator_revision_reaches(root, task, target_evaluation_sha256):
    """Verify source-only evaluator_revision ancestry without changing gates, sampling or budgets."""
    from experiments.forge.views import task_evaluation_fingerprint
    seen = set()
    current = deepcopy(task)
    while True:
        fingerprint = task_evaluation_fingerprint(current)
        if fingerprint == target_evaluation_sha256:
            return True
        if fingerprint in seen:
            return False
        seen.add(fingerprint)
        evaluation = current.get("evaluation", {})
        integration = evaluation.get("integration_source_binding")
        if (isinstance(integration, dict)
                and integration.get("id") == "reviewed-model-prs-integration-20261109"):
            # Combined integration adds a source binding beside (rather than
            # replacing) evaluator_revision. Verify its exact archived bytes
            # before following that older lineage. This is comparison ancestry,
            # never permission to reuse a measurement at the integrated source.
            commit = integration.get("previous_task_commit")
            relative = integration.get("previous_task_path")
            receipt_path = integration.get("integration_receipt")
            paths = (relative, receipt_path)
            if (not isinstance(commit, str) or len(commit) != 40
                    or any(char not in "0123456789abcdef" for char in commit)
                    or any(not isinstance(path, str) or Path(path).is_absolute()
                           or ".." in Path(path).parts or Path(path).suffix != ".json"
                           for path in paths)
                    or integration.get("qualification_input") is not False):
                return False
            try:
                previous_raw = subprocess.check_output(
                    ["git", "show", f"{commit}:{relative}"], cwd=root)
                previous = json.loads(previous_raw)
                current_raw = (Path(root) / relative).read_bytes()
                receipt = read_json(Path(root) / receipt_path)
                rows = [row for row in receipt["tasks"]
                        if row.get("previous_task_path") == relative]
                if (len(rows) != 1
                        or json.loads(current_raw) != current
                        or receipt.get("schema") != "pr_combined_integration_source_binding_v1"
                        or receipt.get("pr") != "combined"
                        or any(receipt.get(key) is not False for key in (
                            "training_rerun", "qualification_results_changed",
                            "automatic_cross_source_reuse", "qualification_input"))):
                    return False
                row = rows[0]
                if (any(row.get(key) != integration.get(key) for key in (
                        "previous_task_commit", "previous_task_path",
                        "previous_task_sha256", "previous_task_blob"))
                        or hashlib.sha256(previous_raw).hexdigest() != integration.get("previous_task_sha256")
                        or hashlib.sha1(f"blob {len(previous_raw)}\0".encode() + previous_raw).hexdigest()
                        != integration.get("previous_task_blob")
                        or hashlib.sha256(current_raw).hexdigest() != row.get("current_task_sha256")
                        or row.get("sources_before") != previous.get("evaluation", {}).get("sources")
                        or row.get("sources_after") != evaluation.get("sources")
                        or any(file_hash(Path(root) / path) != digest
                               for path, digest in row["sources_after"].items())):
                    return False
            except (OSError, subprocess.CalledProcessError, json.JSONDecodeError,
                    KeyError, TypeError, AttributeError):
                return False
            restored = deepcopy(current)
            restored["evaluation"].pop("integration_source_binding")
            if "integration_source_binding" in previous.get("evaluation", {}):
                restored["evaluation"]["integration_source_binding"] = deepcopy(
                    previous["evaluation"]["integration_source_binding"])
            restored["evaluation"]["sources"] = deepcopy(previous.get("evaluation", {}).get("sources"))
            # No execution, gate, ownership, sampler or earlier revision change
            # can be justified by this source-only binding, even if receipted.
            if restored != previous:
                return False
            current = previous
            continue
        revision = evaluation.get("evaluator_revision")
        if not isinstance(revision, dict):
            return False
        commit = revision.get("previous_task_commit")
        rel = revision.get("previous_task_path")
        if isinstance(commit, str) and isinstance(rel, str):
            rel_path = Path(rel)
            if (len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit)
                    or rel_path.is_absolute() or ".." in rel_path.parts or rel_path.suffix != ".json"):
                return False
            try:
                previous = json.loads(subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root, text=True))
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                return False
            if ({key: value for key, value in previous.items() if key != "evaluation"}
                    != {key: value for key, value in current.items() if key != "evaluation"}):
                return False
            prev_eval = {key: value for key, value in previous.get("evaluation", {}).items()
                         if key not in {"sources", "evaluator_revision"}}
            curr_eval = {key: value for key, value in current.get("evaluation", {}).items()
                         if key not in {"sources", "evaluator_revision"}}
            if prev_eval != curr_eval:
                return False
            current = previous
            continue
        prev_source = revision.get("previous_source_sha256")
        sources = deepcopy(current.get("evaluation", {}).get("sources", {}))
        added_members = revision.get("added_source_members", [])
        if isinstance(prev_source, str) and isinstance(sources, dict) and isinstance(added_members, list):
            for member in added_members:
                sources.pop(member, None)
            if len(sources) != 1:
                return False
            only = next(iter(sources))
            next_task = deepcopy(current)
            next_task["evaluation"]["sources"] = {only: prev_source}
            if "previous_revision" in revision and revision["previous_revision"] is not None:
                next_task["evaluation"]["evaluator_revision"] = deepcopy(revision["previous_revision"])
            else:
                next_task["evaluation"].pop("evaluator_revision", None)
            current = next_task
            continue
        if isinstance(prev_source, dict) and isinstance(sources, dict) and set(prev_source).issubset(sources):
            next_task = deepcopy(current)
            for member in added_members if isinstance(added_members, list) else ():
                next_task["evaluation"]["sources"].pop(member, None)
            for member, digest in prev_source.items():
                next_task["evaluation"]["sources"][member] = digest
            if "previous_revision" in revision and revision["previous_revision"] is not None:
                next_task["evaluation"]["evaluator_revision"] = deepcopy(revision["previous_revision"])
            else:
                next_task["evaluation"].pop("evaluator_revision", None)
            current = next_task
            continue
        return False


def _measured_candidates_at_source(root, manifests, studies, view_id, execution_backend):
    """Find every non-diagnostic candidate with original attempts at this source."""
    from experiments.forge.views import diagnostic_evidence_scope
    measured = {record["request"]["candidate"]["id"] for record in studies
                if record["request"]["view"]["id"] == view_id
                and (execution_backend is None or record["request"]["execution_backend"] == execution_backend)
                and not diagnostic_evidence_scope(record["request"])}
    for path in sorted((root / "reports/forge/attempts").glob("*/request.json")):
        resolved = read_json(path)
        request = resolved.get("request", resolved)
        source = request.get("source", {})
        expected = manifests.get(source.get("digest"))
        if expected is None or source.get("origin_commit") != expected.get("origin_commit"):
            continue
        if (request.get("view", {}).get("id") == view_id
                and (execution_backend is None or request.get("execution_backend") == execution_backend)
                and not diagnostic_evidence_scope(request)):
            candidate_id = request.get("candidate", {}).get("id")
            if isinstance(candidate_id, str):
                measured.add(candidate_id)
    return measured


def _frozen_report(root, source_commit, *, view_id, execution_backend, temporary,
                   declaration_registration=None, candidate_roster=None, comparison_policy=None):
    if declaration_registration is not None and candidate_roster is not None:
        raise ValueError("cannot combine --declaration-registration and --candidate")
    commit = _resolve_commit(root, source_commit)
    manifests = _receipt_manifests(root, commit)
    studies, support = _recorded_studies(root, manifests)
    roster = None
    if declaration_registration is not None:
        registration_path = Path(declaration_registration)
        if not registration_path.is_absolute():
            registration_path = root / registration_path
        relative = registration_path.resolve().relative_to(root.resolve()).as_posix()
        recorded_bytes = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
        if registration_path.read_bytes() != recorded_bytes:
            raise ValueError("declaration roster registration differs from the executed Git source")
        registration = read_json(registration_path)
        if (registration.get("scope") != "recipe_prior_refactor_ordinary_measurement"
                or execution_backend != "cuda" or len(registration.get("families", {})) != 7):
            raise ValueError("declaration roster requires the exact seven-leader ordinary CUDA registration")
        ids = [plan["candidate_id"] for plan in registration["families"].values()]
        measured = {record["request"]["candidate"]["id"] for record in studies}
        if len(set(ids)) != 7 or not measured.issubset(ids):
            raise ValueError("declaration roster omits an original measured candidate")
        if any(plan["source_digest"] not in manifests for plan in registration["families"].values()):
            raise ValueError("declaration roster scientific source differs from original receipts")
        roster = {"registration_path": relative, "registration_sha256": file_hash(registration_path),
                  "candidate_ids": sorted(ids), "measured_candidate_ids": sorted(measured),
                  "execution_backend": "cuda", "qualification_transfer": False}
    elif candidate_roster is not None:
        selected_ids = sorted({identifier(cid, "candidate") for cid in candidate_roster})
        measured = _measured_candidates_at_source(root, manifests, studies, view_id, execution_backend)
        if not selected_ids or not measured.issubset(selected_ids):
            raise ValueError("candidate roster omits an original measured candidate")
        roster = {"scope": "explicit_candidate_roster", "candidate_ids": selected_ids,
                  "measured_candidate_ids": sorted(measured),
                  "execution_backend": execution_backend, "qualification_transfer": False}
    frozen_root = temporary / "source"
    catalog_closure = _extract_source(root, commit, frozen_root, manifests, study_support=support)
    contract = temporary / "manifests.json"
    atomic_text(contract, _json_text(manifests))
    study_contract = temporary / "recorded-studies.json"
    atomic_text(study_contract, _json_text(studies))
    roster_contract = temporary / "declaration-roster.json"
    atomic_text(roster_contract, _json_text(roster))
    comparison = None
    if comparison_policy is not None:
        comparison_path = Path(comparison_policy)
        if not comparison_path.is_absolute():
            comparison_path = root / comparison_path
        policy = read_json(comparison_path)
        actual_view = read_json(frozen_root / "configs/forge/views" / f"{view_id}.json")
        _validate_comparison_policy(actual_view, policy)
        recorded = read_json(root / CURRENT_PREFIX.with_suffix(".json"))
        manifest = read_json(root / EVIDENCE_MANIFEST)
        if stable_hash(policy) != manifest["policy_fingerprint"] or recorded["policy_fingerprint"] != stable_hash(policy):
            raise ValueError("comparison policy must match the registered inventory policy")
        # Validate compact original evidence, without reconstructing or grading
        # any historical implementation. Only the new source is regraded.
        verified = set()
        for entry in manifest["cohorts"]:
            _, rows = _snapshot(root, entry, manifest)
            verified.update(row["candidate_revision"] for row in rows.values())
        from experiments.forge.trainer_families import load_families
        from experiments.forge.views import load_tasks, task_evaluation_fingerprint
        frozen_tasks = load_tasks(frozen_root)
        comparison_task_ids = {assignment["task"] for assignment in policy["assignments"]}
        added_actual_tasks = sorted({assignment["task"] for assignment in actual_view["assignments"]} - comparison_task_ids)
        reference, verified_lineage = {}, {}
        recorded_catalog = recorded.get("task_contracts", {})
        for row in recorded["rows"]:
            contracts = row.get("bindings", {}).get("task_contracts")
            if row.get("candidate_revision") in verified and isinstance(contracts, dict) and contracts:
                family = row["trainer_family"]
                if family in reference and not row.get("attempt_ids"):
                    continue
                reference[family] = contracts
                verified_lineage.pop(family, None)
                family_lineage = {}
                for task_id, ref_digest in contracts.items():
                    ref_contract = recorded_catalog.get(ref_digest)
                    frozen_task = frozen_tasks.get(task_id)
                    if not isinstance(ref_contract, dict) or frozen_task is None:
                        continue
                    frozen_eval = task_evaluation_fingerprint(frozen_task)
                    ref_eval = ref_contract.get("evaluation_sha256")
                    if frozen_eval != ref_eval and _evaluator_revision_reaches(root, frozen_task, ref_eval):
                        family_lineage[task_id] = stable_hash({**ref_contract, "evaluation_sha256": frozen_eval})
                    elif (
                        task_id in {"grid100", "rotated100", "staggered100"}
                        and ref_eval == "f659a437977477685011168eb644108a0ce5b4dc21d145bea709cc307d8e3105"
                        and frozen_eval == "035a006309308989d60bc90a9b77ea63fa042648a65c155bcbf09bed0e9b165e"
                        and (frozen_task.get("evaluation", {}).get("evaluator_revision") or {}).get("id")
                        == "tier2-grid-triad-clean-sampling-calibration-v1"
                    ):
                        family_lineage[task_id] = stable_hash({**ref_contract, "evaluation_sha256": frozen_eval})
                if family_lineage:
                    verified_lineage[family] = family_lineage
        from experiments.forge import planning, trainer_families
        for record in studies:
            req = record.get("request", {})
            cand_id = req.get("candidate", {}).get("id")
            control = req.get("study", {}).get("control", {})
            ctrl_id = control.get("candidate_id")
            if not cand_id or not ctrl_id or control.get("task_map"):
                continue
            try:
                fam = trainer_families.family_for_candidate(
                    frozen_root, cand_id, planning.load_idea(frozen_root, cand_id), current_presentation=True
                )["id"]
                ctrl_fam = trainer_families.family_for_candidate(
                    frozen_root, ctrl_id, planning.load_idea(frozen_root, ctrl_id), current_presentation=True
                )["id"]
            except (KeyError, ValueError, FileNotFoundError):
                continue
            if fam not in reference and ctrl_fam in reference:
                reference[fam] = reference[ctrl_fam]
                if ctrl_fam in verified_lineage:
                    verified_lineage[fam] = verified_lineage[ctrl_fam]
        new_registered_families = sorted(set(load_families(frozen_root)) - set(reference))
        comparison = {"policy": policy, "path": os.path.relpath(comparison_path, root),
                      "sha256": file_hash(comparison_path), "reference_task_contracts": reference,
                      "added_actual_tasks": added_actual_tasks, "verified_evaluator_lineage": verified_lineage,
                      "new_registered_families": new_registered_families,
                      "calibrated_grid_triad_tasks": ["grid100", "rotated100", "staggered100"]}
    comparison_contract = temporary / "comparison-policy.json"
    atomic_text(comparison_contract, _json_text(comparison))
    prefix = temporary / "inventory"
    # A fresh process is essential: already imported live model/resolver
    # modules cannot grade a reconstructed older checkout faithfully.
    program = (inspect.getsource(_recorded_study_question) + inspect.getsource(_validate_comparison_policy)
               + inspect.getsource(_comparison_contracts_match) + inspect.getsource(_comparison_projection)) + """
import json, sys
import torch
torch.set_num_threads(1)
from pathlib import Path
from experiments.forge.sources import inspect_source
from experiments.forge.technique_board import write_report
root, contract, study_contract, prefix, goal, backend, roster_contract, comparison_contract = sys.argv[1:]
root = Path(root)
manifests = json.loads(Path(contract).read_text())
recorded_studies = json.loads(Path(study_contract).read_text())
declaration_roster = json.loads(Path(roster_contract).read_text())
comparison = json.loads(Path(comparison_contract).read_text())
study_proofs = []
admitted_requests = {}
for expected in manifests.values():
    actual = inspect_source(root, list(expected['files']))
    if actual['digest'] != expected['digest'] or actual['files'] != expected['files']:
        raise ValueError('reconstructed scientific source differs from original receipt manifest')
# This checkout is immutable and its manifest was checked above without a
# cache. Repeated declaration resolution reads the same source/support set.
# Memoize only that original inspection, never requests, grades or reducers.
from copy import deepcopy
from experiments.forge import sources as source_module
original_inspect_source = source_module.inspect_source
source_inspection_cache = {}
source_inspection_counts = {'hits': 0, 'misses': 0, 'other_root_bypasses': 0}
def immutable_source_inspection(candidate_root, extra_paths=()):
    candidate_root = Path(candidate_root).resolve()
    if candidate_root != root.resolve():
        source_inspection_counts['other_root_bypasses'] += 1
        return original_inspect_source(candidate_root, extra_paths)
    extras = tuple(sorted(str(path) for path in extra_paths))
    key = (str(candidate_root), extras)
    if key not in source_inspection_cache:
        source_inspection_counts['misses'] += 1
        source_inspection_cache[key] = original_inspect_source(candidate_root, extras)
    else:
        source_inspection_counts['hits'] += 1
    return deepcopy(source_inspection_cache[key])
source_module.inspect_source = immutable_source_inspection
# A read-only board must reconstruct the declared question, not request a new
# Tier 3 admission for a contract authorized only through Tier 1. The normal
# frozen resolver still checks every binding and returns the full task view;
# no scope, scientific source, original receipt or numerical gate is edited.
if (root / 'experiments/forge/knowledge.py').is_file():
    from experiments.forge import knowledge, planning
    # Planning imports the source reader by value. Bind its alias explicitly
    # even if another frozen helper happened to import it before this point.
    planning.inspect_source = immutable_source_inspection
    if planning.inspect_source is not source_module.inspect_source:
        raise ValueError('frozen planning source inspection cache alias mismatch')
    if hasattr(planning, 'declaration_paths'):
        # Scientific revisions exclude labels. The board emits the first
        # declaration for an equivalent cohort, so prefer the exact candidate
        # IDs in originals already validated against these source manifests.
        # Strict publication checks still reject mixed candidate identities.
        recorded_candidates = set()
        sources = {(digest, source.get('origin_commit')) for digest, source in manifests.items()}
        for path in (root / 'reports/forge/attempts').glob('*/request.json'):
            resolved = json.loads(path.read_text())
            request = resolved.get('request', resolved)
            source = request.get('source', {})
            if (source.get('digest'), source.get('origin_commit')) in sources:
                recorded_candidates.add(request['candidate']['id'])
        original_declaration_paths = planning.declaration_paths
        canonical_candidates = set()
        if (root / 'experiments/forge/trainer_families.py').is_file():
            from experiments.forge.trainer_families import load_families
            canonical_candidates = {family['canonical_candidate'] for family in load_families(root).values()}
        def receipt_first_declarations(root):
            paths = original_declaration_paths(root)
            if declaration_roster is not None:
                selected = set(declaration_roster['candidate_ids'])
                paths = [path for path in paths if json.loads(path.read_text()).get('id', path.stem) in selected]
                if {json.loads(path.read_text()).get('id', path.stem) for path in paths} != selected:
                    raise ValueError('frozen declaration roster is incomplete')
            return sorted(paths,
                          key=lambda path: (path.stem not in recorded_candidates,
                                            path.stem not in canonical_candidates, path.stem))
        planning.declaration_paths = receipt_first_declarations
        if declaration_roster is not None:
            # The roster limits board enumeration. Exact study controls and
            # parent declarations must still resolve from the full catalog.
            original_load_idea = planning.load_idea
            def full_catalog_load_idea(checkout, idea_id):
                previous_paths = planning.declaration_paths
                planning.declaration_paths = original_declaration_paths
                try:
                    return original_load_idea(checkout, idea_id)
                finally:
                    planning.declaration_paths = previous_paths
            planning.load_idea = full_catalog_load_idea
    def question_request(root, idea_id, view_id, execution_backend='cuda', cuda_model=None):
        request = _recorded_study_question(root, idea_id, view_id, execution_backend, cuda_model,
                                          recorded_studies, study_proofs)
        admitted_requests[(idea_id, view_id, execution_backend, cuda_model)] = deepcopy(request)
        return request
    knowledge._current_request = question_request
if (root / 'reports/forge/word_checkpoint_publication.py').is_file():
    from reports.forge.word_checkpoint_publication import install
    install()
result = write_report(root, goal, execution_backend=None if backend == 'all' else backend, output_prefix=prefix)
report_path = Path(result['json'])
report = json.loads(report_path.read_text())
if comparison is not None:
    from experiments.forge import views
    from experiments.forge.contracts import stable_hash
    actual = deepcopy(report)
    actual_policy = views.load_view(root, goal)
    _validate_comparison_policy(actual_policy, comparison['policy'])
    for row in actual['rows']:
        if not row.get('attempt_ids'):
            continue
        family = row['bindings']['trainer_family']
        if not _comparison_contracts_match(row, comparison):
            raise ValueError('comparison projection changes recorded task contracts: ' + str(family))
    # All admission/source reconstruction above used the original view. The
    # projection may only reuse those verified requests, never reconstruct a
    # study with its view overridden.
    original_view_reader = views.load_view
    original_question = knowledge._current_request
    def comparison_view(directory, name):
        if Path(directory).resolve() == root.resolve() and name == goal:
            return deepcopy(comparison['policy'])
        return original_view_reader(directory, name)
    def admitted_question(directory, idea_id, view_id, execution_backend='cuda', cuda_model=None):
        key = (idea_id, view_id, execution_backend, cuda_model)
        if key not in admitted_requests:
            raise ValueError('comparison projection requested an unverified admission')
        return deepcopy(admitted_requests[key])
    try:
        views.load_view = comparison_view
        knowledge._current_request = admitted_question
        write_report(root, goal, execution_backend=None if backend == 'all' else backend, output_prefix=prefix)
        report = json.loads(report_path.read_text())
    finally:
        views.load_view = original_view_reader
        knowledge._current_request = original_question
    _comparison_projection(actual, report, comparison)
report['registered_study_reconstruction'] = study_proofs
if declaration_roster is not None:
    report['declaration_roster'] = declaration_roster
report['frozen_source_inspection_cache'] = {
    'scope': 'Original inspect_source, exact immutable frozen root and sorted extra_paths only; original manifest validation precedes cache installation. Requests, receipts, gates and reducers are uncached.',
    'counts': source_inspection_counts}
report_path.write_text(json.dumps(report, sort_keys=True))
print(json.dumps(result, sort_keys=True))
"""
    environment = {**os.environ, "PYTHONPATH": str(frozen_root), "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    try:
        output = subprocess.check_output([os.path.abspath(sys.executable), "-c", program, str(frozen_root),
                                          str(contract), str(study_contract), str(prefix), view_id, execution_backend or "all",
                                          str(roster_contract), str(comparison_contract)],
                                         cwd=frozen_root, env=environment, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError("frozen independent report failed: " + error.stderr.strip()) from error
    metadata = json.loads(output)
    result = read_json(metadata["json"])
    result["rows"], unresolved = _bound_publication_rows(result, set(manifests))
    _validate_recorded_study_projection(studies, result, view_id, execution_backend)
    if catalog_closure is not None:
        result["frozen_catalog_support_closure"] = catalog_closure
    if roster is not None:
        if {row["candidate_id"] for row in result["rows"] + unresolved} != set(roster["candidate_ids"]):
            raise ValueError("frozen publication omits a registered leader")
        measured = set(roster["measured_candidate_ids"])
        actual = {row["candidate_id"] for row in result["rows"] if row.get("attempt_ids")}
        proofs = {proof["candidate_id"] for proof in result.get("registered_study_reconstruction", [])}
        if actual != measured or proofs != measured or unresolved:
            raise ValueError("frozen registered leader measurement/study reconstruction is incomplete")
    if unresolved:
        result["unresolved_configuration_rows"] = unresolved
    if "configuration_rows" in result:
        result["snapshot_row_scope"] = "all_configuration_variants"
    return metadata, result, commit, sorted(manifests)


def regenerate(root: Path | str = REPOSITORY_ROOT, *, view_id="discriminator_stability", execution_backend="cuda",
               output_prefix: Path | str = "reports/forge/technique-inventory", source_commit=None,
               declaration_registration=None, candidate_roster=None, comparison_policy=None) -> dict:
    """Regrade original evidence and publish deterministic compact artifacts."""
    root = Path(root).resolve()
    if (declaration_registration is not None or candidate_roster is not None) and source_commit is None:
        raise ValueError("declaration roster requires exact frozen source reconstruction")
    if declaration_registration is not None and candidate_roster is not None:
        raise ValueError("cannot combine --declaration-registration and --candidate")
    if comparison_policy is not None and source_commit is None:
        raise ValueError("independent comparison projection requires an exact frozen source")
    prefix = Path(output_prefix)
    if not prefix.is_absolute():
        prefix = root / prefix
    json_path, markdown_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
    published_attempts = set()
    if json_path.exists():
        previous = read_json(json_path)
        published_attempts = {attempt for row in previous.get("rows", []) for attempt in row.get("attempt_ids", [])}
        for attempt in sorted(published_attempts):
            identifier(attempt, "attempt")
            original = root / "reports/forge/attempts" / attempt
            if not all((original / f"{name}.json").is_file() for name in ("request", "evidence", "result")):
                raise ValueError(f"hydrate original receipts referenced by the published report before regeneration: {attempt}")
    # Stage the existing independent reducer before publishing anything. Invalid
    # originals cannot overwrite a previously measured report with a projection.
    with tempfile.TemporaryDirectory(prefix="forge-technique-publication-") as temporary:
        if source_commit is None:
            metadata = write_report(root, view_id, execution_backend=execution_backend,
                                    output_prefix=Path(temporary) / "inventory")
            result = read_json(metadata["json"])
            result["rows"] = _publication_rows(result)
            if "configuration_rows" in result:
                result["snapshot_row_scope"] = "all_configuration_variants"
            commit, frozen_digests = None, []
        else:
            comparison_options = {"comparison_policy": comparison_policy} if comparison_policy is not None else {}
            if candidate_roster is not None:
                comparison_options["candidate_roster"] = candidate_roster
            metadata, result, commit, frozen_digests = _frozen_report(
                root, source_commit, view_id=view_id, execution_backend=execution_backend, temporary=Path(temporary),
                declaration_registration=declaration_registration, **comparison_options)
    attempt_ids = sorted({attempt for row in result["rows"] for attempt in row.get("attempt_ids", [])})
    if published_attempts - set(attempt_ids):
        if source_commit is None:
            raise ValueError("live source no longer retains the published measured cohort; use --source-commit or a different output prefix")
        raise ValueError("frozen regrade no longer retains the published measured cohort; restore the recorded runtime/hardware or use a different output prefix")
    summaries = {attempt: project_receipt(root, attempt) for attempt in attempt_ids}
    result["publication_scope"] = "frozen_source" if commit else "live_current"
    if commit:
        result["frozen_source"] = {"commit": commit, "source_digests": frozen_digests,
                                   "qualifies_latest_checkout": False}
        adapter = root / "reports/forge/word_checkpoint_publication.py"
        if adapter.is_file():
            result.setdefault("reporting_adapters", {})["word_checkpoint_publication"] = {
                "path": "reports/forge/word_checkpoint_publication.py",
                "sha256": file_hash(adapter),
                "scope": "Bind the existing word grader's own-checkpoint proof to its selected smoke parent; numerical grading and original task declarations remain frozen."}
        result.setdefault("reporting_adapters", {})["registered_study_reconstruction"] = {
            "path": "reports/forge/regenerate_technique_inventory.py", "sha256": file_hash(Path(__file__)),
            "scope": "Resolve only the exact original admitted study ID and Git-pinned motivation; require unchanged study/admission/source/view/full scientific cohort. No training or numerical gate changes."}
        result.setdefault("reporting_adapters", {})["immutable_source_inspection_cache"] = {
            "path": "reports/forge/regenerate_technique_inventory.py", "sha256": file_hash(Path(__file__)),
            **result.get("frozen_source_inspection_cache", {})}
    _publication_provenance(result, summaries)
    repo_prefix = os.path.relpath(root, markdown_path.parent.resolve())
    markdown = render_markdown(result, json_link=json_path.name, repo_link_prefix=repo_prefix)
    if commit:
        markdown = markdown.replace("exact current cohort", "exact recorded cohort").replace(
            "Current rows bind", "Recorded rows bind")
        first, rest = markdown.split("\n", 1)
        markdown = first + f"\n\nThis report grades the **frozen source cohort `{commit}`** reconstructed from Git and " \
            "verified against original receipt manifests. Its outcomes do not qualify the latest checkout.\n" + rest
    for attempt in attempt_ids:
        markdown = markdown.replace(f"{repo_prefix}/reports/forge/attempts/{attempt}/result.json",
                                    f"{repo_prefix}/reports/forge/technique-receipts/{attempt}.json")
    prefix_argument = os.path.relpath(prefix, root)
    command = ("python reports/forge/regenerate_technique_inventory.py --root . --goal " + shlex.quote(view_id) +
               (" --device " + execution_backend if execution_backend else " --device all") +
               " --output-prefix " + shlex.quote(prefix_argument) +
               (" --source-commit " + commit if commit else ""))
    if declaration_registration is not None:
        command += " --declaration-registration " + shlex.quote(str(declaration_registration))
    if candidate_roster is not None:
        for cid in sorted({identifier(item, "candidate") for item in candidate_roster}):
            command += " --candidate " + shlex.quote(cid)
    if comparison_policy is not None:
        command += " --comparison-policy " + shlex.quote(str(comparison_policy))
    lines = markdown.splitlines()
    lines = [command if line.startswith("python -m experiments.forge techniques ") else line for line in lines]
    markdown = "\n".join(lines) + "\n\nPublished receipt summaries retain final metrics, gate outcomes and original file hashes. " \
        "They are display artifacts and supply no qualification input. Hydrate byte-exact original " \
        "request/evidence/result receipts from the artifact archive before a full independent regrade.\n"
    for attempt, summary in summaries.items():
        _write_changed(root / "reports/forge/technique-receipts" / f"{attempt}.json", _json_text(summary))
    _write_changed(json_path, _json_text(result))
    _write_changed(markdown_path, markdown)
    return {**metadata, "report": str(markdown_path), "json": str(json_path),
            "input_digest": result["provenance"]["input_digest"],
            "publication_scope": result["publication_scope"], "source_commit": commit,
            "summary_receipts": len(summaries), "qualification_reuse": False}


def _published_report(root, path):
    """Read an intact display publication; composing never regrades evidence."""
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    report = read_json(path)
    if report.get("publication_scope") not in {"live_current", "frozen_source", "recorded_cohort_composition"}:
        raise ValueError("composition requires an independently published current or frozen report")
    copy = deepcopy(report)
    claimed = copy.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(copy):
        raise ValueError("published report input digest mismatch")
    markdown = path.with_suffix(".md")
    if not markdown.is_file():
        raise ValueError("published report Markdown companion is missing")
    return report, {"json": path, "markdown": markdown,
                    "json_sha256": file_hash(path), "markdown_sha256": file_hash(markdown)}


def _validate_published_row(root, report, row, visited=None):
    """Bind each measured display row to its already-published receipt proofs."""
    if report.get("publication_scope") == "recorded_cohort_composition":
        pointer = report.get("source_publications", {}).get(row.get("publication_key"))
        if not isinstance(pointer, dict):
            raise ValueError("composed row has no source publication")
        path = (root / pointer["json"]).resolve()
        visited = set(visited or ())
        if path in visited:
            raise ValueError("cyclic source publication composition")
        visited.add(path)
        source, paths = _published_report(root, path)
        if any(pointer.get(name + "_sha256") != paths[name + "_sha256"]
               for name in ("json", "markdown")) or pointer.get("input_digest") != source["provenance"]["input_digest"]:
            raise ValueError("composed source publication hash mismatch")
        identities = ("candidate_id", "candidate_revision", "cohort")
        matches = [item for item in source.get("rows", [])
                   if all(item.get(key) == row.get(key) for key in identities)]
        if len(matches) != 1:
            raise ValueError("composed row has no unique original publication row")
        # Labels and display flags can change; scientific row contents cannot.
        display_fields = {"publication_key", "qualification_reuse", "qualification_input", "technique"}
        science = lambda item: {key: value for key, value in item.items() if key not in display_fields}
        if science(row) != science(matches[0]):
            raise ValueError("composed row differs from its original scientific publication")
        _validate_published_row(root, source, matches[0], visited)
        return
    proofs = report.get("provenance", {}).get("qualified_receipts", {})
    summaries = {}
    for attempt in row.get("attempt_ids", []):
        identifier(attempt, "attempt")
        proof = proofs.get(attempt)
        path = root / "reports/forge/technique-receipts" / f"{attempt}.json"
        if not isinstance(proof, dict) or not path.is_file():
            raise ValueError("measured publication row lacks its compact receipt proof")
        summary = read_json(path)
        summaries[attempt] = summary
        provenance = summary.get("provenance", {})
        actual = {"canonical_result_hash": provenance.get("canonical_result_hash"),
                  "source_digest": provenance.get("source_digest"),
                  "original_file_sha256": {name: item.get("sha256") for name, item in
                                            provenance.get("original_files", {}).items()}}
        if summary.get("retry_of"):
            actual["retry_of"] = deepcopy(summary["retry_of"])
        if actual != proof or summary.get("certificate_validated") is not True:
            raise ValueError("compact receipt proof differs from its publication")
        if (summary.get("qualification_input") is not False or summary.get("qualification_reuse") is not False
                or summary.get("candidate_id") != row.get("candidate_id")
                or summary.get("candidate_revision") != row.get("candidate_revision")
                or provenance.get("source_digest") != row.get("bindings", {}).get("source_digest")):
            raise ValueError("compact receipt candidate/source cohort differs from its publication row: "
                             + _json_text({"attempt_id": attempt,
                                           "receipt": {"candidate_id": summary.get("candidate_id"),
                                                       "candidate_revision": summary.get("candidate_revision"),
                                                       "source_digest": provenance.get("source_digest"),
                                                       "qualification_input": summary.get("qualification_input"),
                                                       "qualification_reuse": summary.get("qualification_reuse")},
                                           "row": {"candidate_id": row.get("candidate_id"),
                                                   "candidate_revision": row.get("candidate_revision"),
                                                   "source_digest": row.get("bindings", {}).get("source_digest")}}).strip())
    if any(summary.get("retry_of") for summary in summaries.values()):
        from experiments.forge.family_reports import certified_retry_successors
        certified_retry_successors(summaries)
    for tier, required in report.get("tier_requirements", {}).items():
        cell = row.get("tiers", {}).get(tier, {})
        if (cell.get("total") != len(required) or type(cell.get("passed")) is not int
                or not 0 <= cell["passed"] <= cell["total"]):
            raise ValueError("published row changed a declared tier denominator")


def _compose_markdown(result, paths, root, markdown_path, original_path, current_path, candidate_id, prefix):
    def link(path):
        return os.path.relpath(path, markdown_path.parent)

    def cell(value):
        return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")

    tiers = list(result["tier_requirements"])
    lines = ["# Forge technique inventory: recorded source cohorts", "",
             "Each cell is **passes / full required total** in its row's recorded source and runtime. "
             "The retained inventory and appended technique keep their separate evidence identities.", "",
             "| Technique | Recorded source | Exact revision / cohort | Compute | " +
             " | ".join(f"Tier {tier}" for tier in tiers) + " | Recorded tier | Other outcomes | Paid seconds |",
             "| --- | --- | --- | --- | " + " | ".join("---:" for _ in tiers) + " | ---: | --- | ---: |"]
    for row in result["rows"]:
        source = row["bindings"]["source_digest"]
        pointer = paths[row["publication_key"]]
        source_link = f"[`{source[:12]}`]({link(pointer['markdown'])})"
        runtime = row.get("runtime_cohort", {})
        models = sorted({profile.get("model") for profile in runtime.get("compute_profiles", {}).values()
                         if profile and profile.get("model")})
        compute = runtime.get("execution_backend", "unrecorded") + (" / " + ", ".join(models) if models else "")
        counts = Counter(task["status"] for task in row.get("tasks", []) if task["status"] != "PASS")
        others = ", ".join(f"{status} {count}" for status, count in sorted(counts.items())) or "all required tasks PASS"
        seconds = row.get("cost", {}).get("wall_seconds")
        values = [row["technique"], source_link,
                  f"{row['candidate_revision'][:12]} / {row['cohort'][:12]}", compute,
                  *[f"{row['tiers'][tier]['passed']}/{row['tiers'][tier]['total']}" for tier in tiers],
                  row["qualified_tier"], others, round(seconds, 3) if seconds is not None else "unknown"]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", "Recorded tiers describe each independently graded publication. This display does not pool passes, "
              "rank across source cohorts, or qualify the latest checkout. Recipes, priors, initialization, budgets, "
              "clean/noisy sampling and hardware remain bound to their original rows.", "",
              "UNKNOWN means unmeasured or unrun evidence. Failed or blocked prerequisite gates stop further work; "
              "every declared task remains in its tier denominator.", "",
              f"[Previous recorded inventory]({link(paths['original']['markdown'])}) · "
              f"[Full current inventory, including all technique denominator rows]({link(paths['current']['markdown'])}) · "
              f"[Exact row bindings and publication hashes]({markdown_path.with_suffix('.json').name})", "",
              "Regenerate this display from the committed publications without launching training or requiring raw receipt hydration:",
              "", "```sh",
              "python reports/forge/regenerate_technique_inventory.py --root . --compose-original " +
              shlex.quote(os.path.relpath(original_path, root)) + " --compose-current " +
              shlex.quote(os.path.relpath(current_path, root)) + " --append-candidate " + shlex.quote(candidate_id) +
              " --output-prefix " + shlex.quote(os.path.relpath(prefix, root)), "```", "",
              f"Publication input digest `{result['provenance']['input_digest']}`. "
              "Full independent regrading uses each linked publication's own regeneration command and byte-exact original receipts.", ""]
    return "\n".join(lines)


def compose(root: Path | str = REPOSITORY_ROOT, *, original_report="reports/forge/technique-inventory.json",
            current_report="reports/forge/r3gan-technique-inventory.json",
            candidate_id="r3gan-stacked-training-toy-v1",
            output_prefix="reports/forge/technique-inventory-expanded") -> dict:
    """Append one new technique to recorded rows without pooling qualification.

    Source publications are immutable inputs. This display can regenerate in a
    fresh checkout using compact receipt proofs; it supplies no gate evidence.
    """
    root = Path(root).resolve()
    identifier(candidate_id, "candidate")
    original, original_paths = _published_report(root, original_report)
    current, current_paths = _published_report(root, current_report)
    if original.get("publication_scope") not in {"frozen_source", "recorded_cohort_composition"}:
        raise ValueError("original publication must preserve a frozen source cohort")
    # Earlier compositions recorded runtime on every row but omitted this field.
    def backend(report):
        if report.get("execution_backend") is not None:
            return report["execution_backend"]
        cohorts = {row.get("runtime_cohort", {}).get("execution_backend") for row in report.get("rows", [])}
        return next(iter(cohorts)) if len(cohorts) == 1 else None
    for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements", "execution_backend"):
        left, right = (backend(original), backend(current)) if key == "execution_backend" else (original.get(key), current.get(key))
        if left != right:
            raise ValueError(f"source publications have incompatible {key}; keep them in separate reports")
    if any(row.get("candidate_id") == candidate_id for row in original.get("rows", [])):
        raise ValueError("appended technique is already present in the original publication")
    additions = [row for row in current.get("rows", []) if row.get("candidate_id") == candidate_id]
    if len(additions) != 1:
        raise ValueError("current publication must contain exactly one requested technique/runtime cohort")
    current_candidates = {row.get("candidate_id") for row in current.get("rows", [])}
    if any(row.get("candidate_id") not in current_candidates for row in original.get("rows", [])):
        raise ValueError("current publication must retain every original technique denominator row")
    prefix = Path(output_prefix)
    if not prefix.is_absolute():
        prefix = root / prefix
    json_path, markdown_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
    paths = {"original": original_paths, "current": current_paths}
    protected = {pointer[name] for pointer in paths.values() for name in ("json", "markdown")}
    if json_path.resolve() in protected or markdown_path.resolve() in protected:
        raise ValueError("composition output cannot overwrite an input publication")
    rows = []
    for key, report, selected in (("original", original, original["rows"]), ("current", current, additions)):
        for original_row in selected:
            _validate_published_row(root, report, original_row)
            if not original_row.get("bindings", {}).get("source_digest"):
                raise ValueError("composition row is missing its recorded source identity")
            row = deepcopy(original_row)
            row.update(publication_key=key, qualification_reuse=False, qualification_input=False)
            if key == "current" and candidate_id == "r3gan-stacked-training-toy-v1":
                row["technique"] = "R3GAN Stacked-MNIST recipe (toy-host adaptation)"
            rows.append(row)
    result = {"schema_version": 1, "reducer_version": "forge-technique-composition-v1",
              "publication_scope": "recorded_cohort_composition", "qualification_reuse": False,
              "qualification_input": False, "view": original["view"], "view_revision": original["view_revision"],
              "policy_fingerprint": original["policy_fingerprint"],
              "execution_backend": backend(original),
              "tier_requirements": deepcopy(original["tier_requirements"]), "rows": rows,
              "source_publications": {key: {
                  "json": os.path.relpath(pointer["json"], root), "markdown": os.path.relpath(pointer["markdown"], root),
                  "json_sha256": pointer["json_sha256"], "markdown_sha256": pointer["markdown_sha256"],
                  "publication_scope": report["publication_scope"],
                  "input_digest": report["provenance"]["input_digest"], "frozen_source": report.get("frozen_source")}
                  for key, report, pointer in (("original", original, original_paths), ("current", current, current_paths))}}
    for catalog in ("recipe_contracts", "protocol_contracts", "task_contracts", "status_reasons"):
        combined = {}
        for report in (original, current):
            for digest, contract in report.get(catalog, {}).items():
                if digest != stable_hash(contract):
                    raise ValueError(f"invalid {catalog} identity in input publication")
                if digest in combined and combined[digest] != contract:
                    raise ValueError("conflicting scientific contract identities between publications")
                combined[digest] = deepcopy(contract)
        result[catalog] = dict(sorted(combined.items()))
    result["provenance"] = {"publication_reducer_sha256": file_hash(Path(__file__)),
                            "input_publication_sha256": {key: pointer["json_sha256"] for key, pointer in paths.items()},
                            "selected_rows_sha256": stable_hash(rows)}
    result["provenance"]["input_digest"] = stable_hash(result)
    markdown = _compose_markdown(result, paths, root, markdown_path, original_paths["json"],
                                 current_paths["json"], candidate_id, prefix)
    _write_changed(json_path, _json_text(result))
    _write_changed(markdown_path, markdown)
    return {"report": str(markdown_path), "json": str(json_path), "rows": len(rows),
            "input_digest": result["provenance"]["input_digest"], "publication_scope": result["publication_scope"],
            "qualification_reuse": False}


EVIDENCE_MANIFEST = Path("reports/forge/technique-evidence/manifest.json")
CURRENT_PREFIX = Path("reports/forge/technique-inventory")
CONTRACT_CATALOGS = ("recipe_contracts", "protocol_contracts", "task_contracts", "status_reasons")
POLICY_FIELDS = ("view", "view_revision", "policy_fingerprint", "tier_requirements")


def _snapshot(root, entry, manifest):
    """Numerical evidence needs its own digest/proofs, not another Markdown board."""
    path = root / entry["snapshot"]
    if file_hash(path) != entry["json_sha256"]:
        raise ValueError("technique evidence snapshot hash mismatch")
    report = read_json(path)
    copy = deepcopy(report)
    claimed = copy.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(copy):
        raise ValueError("technique evidence input digest mismatch")
    if report.get("publication_scope") != "frozen_source":
        raise ValueError("technique evidence must retain a frozen source cohort")
    if report.get("frozen_source", {}).get("commit") != entry["source_commit"]:
        raise ValueError("technique evidence source commit mismatch")
    for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements"):
        if manifest[key] != report.get(key):
            raise ValueError(f"technique evidence has incompatible {key}")
    selected = {}
    for name, recorded_at in entry["candidates"].items():
        matches = [row for row in report["rows"] if row.get("candidate_id") == name
                   and (bool(row.get("attempt_ids")) if isinstance(recorded_at, str) else True)]
        if len(matches) != 1:
            raise ValueError("technique evidence needs one exact candidate/runtime row")
        row = matches[0]
        if row.get("attempt_ids") and not isinstance(recorded_at, str):
            raise ValueError("measured technique evidence needs its recorded completion time")
        _validate_published_row(root, report, row)
        selected[name] = row
    return report, selected


def _archived_reports(root, manifest):
    """Validate historical policies without admitting their rows to selection."""
    reports, fingerprints = [], {manifest["policy_fingerprint"]}
    for policy in manifest.get("archived_policies", []):
        if (policy.get("view") != manifest["view"] or
                not isinstance(policy.get("view_revision"), int) or
                policy["view_revision"] >= manifest["view_revision"] or
                policy.get("policy_fingerprint") in fingerprints):
            raise ValueError("invalid archived technique evidence policy")
        fingerprints.add(policy["policy_fingerprint"])
        selection = policy.get("family_selection")
        if selection is not None and file_hash(root / selection["path"]) != selection["sha256"]:
            raise ValueError("archived family selection hash mismatch")
        reports.extend((policy, entry, *_snapshot(root, entry, policy)) for entry in policy["cohorts"])
    return reports


def _ownership_migration_helper():
    path = Path(__file__).resolve().parent / "recipe-prior-refactor/publication_migration.py"
    spec = importlib.util.spec_from_file_location("forge_prior_ownership_publication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _advanced_family_selection(root, manifest, report, registered_rows, prior_reports, *, ownership_migration=None):
    """Stage exact new pins for the existing choices; never rank new recipes."""
    if ownership_migration is not None:
        return _ownership_migration_helper().advance_selection(
            root, manifest, report, registered_rows, prior_reports, ownership_migration,
            validate_row=_validate_published_row)
    from experiments.forge.trainer_families import (CURRENT_SELECTION, family_for_candidate,
                                                   family_row_pin, load_current_selection)
    from experiments.forge.planning import declaration_paths
    from experiments.forge.views import load_view
    path = root / CURRENT_SELECTION
    if not path.is_file():
        return None, None
    card = read_json(path)
    # Preserve compatibility with an explicitly prepared selection card.
    if card.get("policy_fingerprint") == report["policy_fingerprint"]:
        return None, None
    pins = load_current_selection(root, view_id=manifest["view"],
                                  policy_fingerprint=manifest["policy_fingerprint"])
    declarations = {path.stem: read_json(path) for path in declaration_paths(root)}
    selections = []
    for family, pin in pins.items():
        original_matches = []
        for _, _, rows in prior_reports:
            for original in rows.values():
                row = {**original, "trainer_family": family}
                if family_row_pin(row, selection_kind=pin["selection_kind"], reason=pin["reason"],
                                  measurement_views=pin.get("measurement_views"),
                                  measurement_tasks=pin.get("measurement_tasks")) == pin:
                    original_matches.append(row)
        if not original_matches:
            raise ValueError(f"policy advance requires verified previous family selection: {family}")
        matches = [row for row in registered_rows.values() if row["candidate_id"] == pin["candidate_id"]
                   and row.get("runtime_cohort", {}).get("execution_backend") == pin["execution_backend"]]
        if len(matches) != 1:
            raise ValueError(f"policy advance requires one new verified row for selected family: {family}")
        row = {**matches[0], "trainer_family": family}
        if family_for_candidate(root, row["candidate_id"], declarations.get(row["candidate_id"]),
                                current_presentation=True)["id"] != family:
            raise ValueError(f"policy advance cannot change selected configuration family: {family}")
        views = pin.get("measurement_views", [manifest["view"]])
        tasks = pin.get("measurement_tasks")
        required = {assignment["task"] for view in views for assignment in load_view(root, view)["assignments"]
                    if assignment["importance"] == "required" and assignment["qualification_tier"] == 1}
        required.update(tasks or [])
        observations = row.get("tasks", []) + row.get("nonrequired_tasks", [])
        observed = {task["task_id"]: task["status"] for task in observations}
        complete = (bool(row.get("attempt_ids")) and bool(required) and len(observed) == len(observations)
                    and all(observed.get(task) in {"PASS", "FAIL"} for task in required))
        reason = ("Retain the previously selected configuration in its verified new-policy source cohort; "
                  "no outcome-based recipe reselection, qualification transfer or default adoption.")
        selections.append(family_row_pin(row, selection_kind="current_measurement" if complete else "historical_incumbent",
                                         reason=reason, measurement_views=views if complete else None,
                                         measurement_tasks=tasks if complete else None))
    advanced = {**deepcopy(card), "policy_fingerprint": report["policy_fingerprint"], "selections": selections}
    original = path.read_bytes().decode("utf-8")
    digest = file_hash(path)
    relative = CURRENT_SELECTION.parent / "history" / f"family-current-{digest}.json"
    archive = root / relative
    if archive.exists() and archive.read_bytes() != original.encode("utf-8"):
        raise ValueError("archived family selection conflicts with original bytes")
    return advanced, (relative, original, digest)


def _current_tier_cell(row, tier, required, json_link):
    """Display recorded states without turning missing/blocked work into FAIL."""
    summary = row["tiers"][tier]
    passed, total = summary["passed"], summary["total"]
    counts = summary.get("counts")
    if counts is None:
        # Some older snapshots retain only the aggregate and task statuses.
        # Unknown tasks stay in the denominator; the aggregate is not regraded.
        required = set(required)
        counts = Counter(task["status"] for task in row.get("tasks", [])
                         if task["task_id"] in required and task["status"] != "PASS")
        counts["PASS"] = passed
        remaining = total - sum(counts.values())
        if remaining >= 0:
            counts["UNKNOWN"] += remaining
    counts = {status: count for status, count in counts.items() if count}
    if sum(counts.values()) != total or counts.get("PASS", 0) != passed:
        # An incomplete legacy breakdown cannot establish an all-failed state.
        return f"{passed}/{total}<br>[{total} required; task statuses]({json_link})"
    if len(counts) == 1:
        status = next(iter(counts))
        if status == "PASS":
            return f"{passed}/{total} PASS"
        if status != "FAIL":
            return f"{status.replace('_', ' ')} ({total} required)"
    if not passed and set(counts) <= {"UNKNOWN", "NOT_RUN"}:
        return f"NOT RUN/UNKNOWN ({total} required)"
    other = " · ".join(f"{status.replace('_', ' ')} {count}"
                       for status, count in sorted(counts.items()) if status != "PASS")
    return f"{passed}/{total}" + (f"<br>{other}" if other else "")


ATLAS_PROGRESS_INPUTS = {
    "baseline": ("atlas-current-gpu-diagnostics-native-v2-20261003/summary.json",
                 "9976a462fa3c8fbbb33bcdb4dd6b7697c65167f7c5a9c2bdf2c8ed6f6f5cd488",
                 "particlegan_atlas_current_gpu_native_v2_publication_v1"),
    "adaptations_v3": ("atlas-named-gpu-diagnostics-native-v3-20261003/results.json",
                       "8ba7230a71041e3a0185528c9e76c5f39bff6eeadfb98b59d715ef72ff76b9e0",
                       "particlegan_atlas_named_gpu_diagnostics_publication_v3"),
    "adaptations_v4b": ("atlas-named-gpu-diagnostics-native-v4b-20261004/results.json",
                        "0a175519b59b5180078d9669519060b79dc30d596daca3c8be1aca2dd9ce1d5f",
                        "particlegan_atlas_named_gpu_diagnostics_publication_v4b"),
    "word_context": ("atlas-word-retained-context-20261004/context.json",
                     "54d61041c72befd8b4196f78d368374ac1c98470fd5742d4772f9cb6da5bcbdd",
                     "pg_atlas_word_retained_context_v1"),
}

# A separate, accepted numerical diagnostic, bound to its immutable passive
# export. The earlier C6 INVALID remains a different row.
WORD_HALF_BASE_INPUT = ("word-half-base-20261004/results.json",
                       "d2bc198d65c935ca31160505510b8efc7aa2cf9e5f62ac6429f758173e7ca8b7",
                       "pg_word_half_base_publication_v1")
WORD_HALF_BASE_READOUT_SHA256 = "08db1485b912d00558af0f3496dff6332c9b27d2cb13dfca68496f8a90d54bbe"
WORD_HALF_BASE_RECIPE_SHA256 = "ea470cca5c55a31e6f726945402f0dfad945fea2ec95b654afba922e93c8ac69"
WORD_HALF_BASE_GIF_SHA256 = "2d94247b902d6384ef172faa99db0ae4d655bd0171ae6a4ef4d77eaadbc1012b"


def _word_half_base_diagnostic(root):
    """Read one sealed completed export; never grade, sample or select a winner."""
    root = Path(root)
    relative, digest, schema = WORD_HALF_BASE_INPUT
    report_path = root / "reports/forge" / relative
    if not report_path.exists():
        if report_path.parent.exists():
            raise ValueError("half_base retained report is unavailable")
        return {}
    report = read_json(report_path)
    if file_hash(report_path) != digest or report.get("schema") != schema:
        raise ValueError("half_base display requires the exact retained report")
    cohort = "word_joint_policy_min11_rates_v1"
    task = "five_word_joint_acquisition_" + cohort
    flags = ("qualification_input", "ordinary_tier_credit", "historical_credit",
             "default_adoption", "speed_ranking", "cross_tuple_pooling")
    if any(report.get(flag) is not False for flag in flags):
        raise ValueError("half_base display cannot confer qualification, default or speed credit")
    if (report.get("candidate_id") != "word-min11-half_base-rates-v1"
            or report.get("profile") != "half_base"
            or report.get("family") != "atlas_word_joint_min11_rates"
            or report.get("task_cohort") != cohort or report.get("task_id") != task
            or report.get("parent_task_id") != "five_word_joint_acquisition"
            or report.get("status") != "COMPLETE" or report.get("accepted_numeric") != "FAIL"):
        raise ValueError("half_base configuration or accepted numerical status changed")
    parents = ("two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition",
               "five_word_joint_acquisition", "trajectory", "residual_student", "unipolar",
               "cover_leftover", "mid_scale_identity", "mode_hold", "vector_two_broad",
               "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic", "vector_overlap",
               "vector_spiral", "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2",
               "grid100", "rotated100", "staggered100", "ring_hold", "ring_extension")
    slots = {task if parent == "five_word_joint_acquisition" else parent:
             {"status": "FAIL" if parent == "five_word_joint_acquisition" else "NOT_RUN"}
             for parent in parents}
    if (report.get("slots") != slots or report.get("counts") != {"FAIL": 1, "NOT_RUN": 25}
            or report.get("required_slots") != 26 or report.get("scheduled_slots") != 1
            or report.get("not_run_slots") != 25 or report.get("tiers") != {"1": 5, "2": 19, "3": 2}):
        raise ValueError("half_base separate 26-slot denominator changed")
    source = report.get("source", {})
    if source != dict(origin_commit="f9f7ed9d7a06c48d4ec56999107658983d7e8efc",
                      digest="5995590d3c303207c664fe3b0ba7dc1e09dd3da7a90abf87086634c789175026",
                      files=1444,
                      supervisor_sha256="dae842935a1ea3c4c7aae1694cb91e2f6c8b55815095f7230e64c7dc0ec368a1"):
        raise ValueError("half_base executed source identity changed")
    recipe = report.get("resolved_recipe", {})
    overrides = dict(lr=.00265625, prior_lr_mult=1.5, d_lr_mult=1.)
    if (report.get("recipe_overrides") != overrides
            or report.get("resolved_recipe_sha256") != WORD_HALF_BASE_RECIPE_SHA256
            or stable_hash(recipe) != WORD_HALF_BASE_RECIPE_SHA256
            or any(recipe.get(key) != value for key, value in overrides.items())
            or recipe.get("prior_kind") != "particles" or recipe.get("sigma_rel") != 0.0
            or recipe.get("standardize") is not False or recipe.get("encoder_mode") != "none"):
        raise ValueError("half_base actual Recipe or raw particle representation changed")
    steps = [(i * 20001 + 23) // 24 for i in range(1, 25)]
    protocol = report.get("protocol", {})
    if (protocol.get("resources") != dict(num_particles=11, z_dim=2, batch_size=256)
            or protocol.get("seed") != 0 or protocol.get("updates") != 20001
            or protocol.get("schedule_horizon") != 20000 or protocol.get("eval_samples") != 1024
            or protocol.get("metric_steps") != steps or protocol.get("terminal_steps") != steps[-5:]
            or protocol.get("thresholds") != [["sample_count", ">=", 1024], ["quality_fraction", ">=", .95],
                ["modes", "==", 5], ["mass_tv", "<=", .1], ["reconstruction_exact", "==", 1],
                ["minimum_reconstruction_token_probability", ">=", .9]]
            or protocol.get("serving") != "actual policy-selected G/E/raw ParticlePrior; DV12 retained, output noise off"
            or protocol.get("objective") != "original joint RpGAN plus original regularizers; free continuous E; no reconstruction training loss"):
        raise ValueError("half_base original word protocol or selected serving law changed")
    result = report.get("result", {})
    owners = sorted(("continuous_controller", "stationarity_lr", "row_evidence", "birth_death",
                     "learned_output_noise", "selected_averaging", "optimizer_surprise", "reopen_guard"))
    hooks = ("begin_step", "after_critic_step", "after_generator_backward", "after_generator_step", "finish_step")
    summary = result.get("original_grader_summary", {})
    if (result.get("original_gate") != "FAIL" or result.get("completed_steps") != 20001
            or result.get("policy_owner") != "particlegan.UpdatePolicy"
            or result.get("optimizer_updates") != {key: 20001 for key in ("generator", "encoder", "prior", "discriminator")}
            or result.get("lifecycle_calls") != {key: 20001 for key in hooks}
            or result.get("requested_owners") != owners or result.get("enabled_owners") != owners
            or result.get("actual_birth_death") != dict(rows=11, neighbours=5, reference_half=6, isolation=True)
            or result.get("metric_steps") != steps or result.get("terminal_steps") != steps[-5:]
            or result.get("selected_source") not in {"fast", "averaged"}
            or summary.get("complete") is not True or summary.get("observations") != 24
            or summary.get("minimum_stable_checks") != 5 or summary.get("passing_observations") != 0
            or summary.get("passing_suffix") != 0 or any(result.get(flag) is not False for flag in flags)):
        raise ValueError("half_base full public ownership, clocks or numerical decision changed")
    metrics = result.get("final_metrics", {})
    if (not {"sample_count", "step", "quality_fraction", "modes", "mass_tv", "reconstruction_exact",
             "minimum_reconstruction_token_probability"}.issubset(metrics)
            or metrics.get("step") != 20001 or metrics.get("sample_count") != 1024
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values())
            or result.get("independent_atlas_qualification") is not False
            or result.get("synthetic_mechanism_probes_are_learning_credit") is not False):
        raise ValueError("half_base endpoint metrics must retain scalar numbers")
    historical = report.get("historical_word", {})
    if (historical.get("status") != "INVALID" or historical.get("accepted_numeric") != "UNAVAILABLE"
            or historical.get("paid_seconds") != 558.5739127129782
            or historical.get("already_in_prior_total") is not True or historical.get("recertified") is not False):
        raise ValueError("half_base display cannot replace the original C6 INVALID")
    cost = report.get("cost", {})
    expected_cost = dict(paid_seconds=556.4301753160544, reserved_seconds=0., charged_seconds=556.4301753160544,
                         overrun_seconds=0., prior_charged_seconds=910.2391431590077,
                         inclusive_charged_seconds=1466.669318475062, original_cap_seconds=10500,
                         allowance_seconds=900, export_grace_seconds=0)
    lanes = {"0": dict(prior_charged_seconds=234.82608077581972, current_charged_seconds=0.,
                       inclusive_charged_seconds=234.82608077581972, cap_seconds=7500),
             "1": dict(prior_charged_seconds=675.4130623831879, current_charged_seconds=556.4301753160544,
                       inclusive_charged_seconds=1231.8432376992423, cap_seconds=3000)}
    if (any(cost.get(key) != value for key, value in expected_cost.items()) or cost.get("lanes") != lanes
            or cost.get("paid_vs_reserve_separate") is not True or cost.get("speed_comparison_available") is not False):
        raise ValueError("half_base inclusive cost or unchanged lane ceilings changed")
    media = result.get("actual_goal_gif", {})
    media_steps = [steps[i] for i in (0, 3, 6, 9, 12, 14, 17, 20, 23)]
    if (media.get("path") != "media/goal.gif" or media.get("frames") != 9
            or media.get("actual_steps") != media_steps or media.get("sha256") != WORD_HALF_BASE_GIF_SHA256):
        raise ValueError("half_base accepted goal media identity changed")
    gif_path = report_path.parent / media["path"]
    readout = report_path.with_name("README.md")
    if (not gif_path.is_file() or gif_path.stat().st_size != media.get("bytes")
            or file_hash(gif_path) != media["sha256"] or not readout.is_file()
            or file_hash(readout) != WORD_HALF_BASE_READOUT_SHA256):
        raise ValueError("half_base retained goal media or readout bytes changed")
    return dict(schema_version=1, scope="separate_named_configuration_diagnostic",
                candidate_id=report["candidate_id"], profile=report["profile"], family=report["family"],
                task_cohort=cohort, task_id=task, status="COMPLETE", numerical_gate="FAIL",
                required=26, passed=0, counts=report["counts"], source=source,
                recipe_overrides=overrides, resolved_recipe=recipe, resolved_recipe_sha256=WORD_HALF_BASE_RECIPE_SHA256,
                protocol=protocol, result=result, historical_word=historical, cost=cost,
                representation=dict(label="Particles (N11 joint cloud; free encoder)",
                    prior_kind=recipe["prior_kind"], sigma_rel=recipe["sigma_rel"], standardize=recipe["standardize"],
                    actual_prior_rows=11, canonical_target_words=5, encoder_mode=recipe["encoder_mode"]),
                media=dict(media, path=gif_path.relative_to(root).as_posix()),
                readout=readout.relative_to(root).as_posix(),
                input=dict(report=report_path.relative_to(root).as_posix(), sha256=digest,
                           readout_sha256=WORD_HALF_BASE_READOUT_SHA256),
                accounting=dict(inclusive_charged_seconds=cost["inclusive_charged_seconds"],
                    gpu0_charged_seconds=lanes["0"]["inclusive_charged_seconds"],
                    gpu1_charged_seconds=lanes["1"]["inclusive_charged_seconds"]),
                qualification_input=False, qualification_reuse=False, ordinary_tier_credit=False,
                cross_cohort_pooling=False, default_adoption=False, speed_ranking=False)


def _representation_kind(prior):
    """Read the declared law; family names are not representation evidence."""
    kind = (prior or {}).get("kind")
    return {"mog": "MoG", "particle_cloud": "Particles", "particles": "Particles"}.get(kind, "UNKNOWN")


def _ordinary_representation(result, row):
    reference = _representation_kind(row.get("bindings", {}).get("prior"))
    if reference != "MoG":
        return reference
    contracts = result.get("task_contracts", {})
    task_kinds = {_representation_kind(contracts.get(digest, {}).get("prior"))
                  for digest in row.get("bindings", {}).get("task_contracts", {}).values()}
    return "MoG + particles (per task)" if "Particles" in task_kinds else reference


def _atlas_unblocking_progress(root):
    """Project separate source-scoped model scores, never qualification rows."""
    directory = Path(root) / "reports/forge"
    paths = {key: directory / values[0] for key, values in ATLAS_PROGRESS_INPUTS.items()}
    if not any(path.exists() for path in paths.values()):
        return {}
    reports, inputs = {}, {}
    for key, (relative, digest, schema) in ATLAS_PROGRESS_INPUTS.items():
        report = read_json(paths[key])
        if file_hash(paths[key]) != digest or report.get("schema") != schema:
            raise ValueError("Atlas progress requires the exact retained report: " + key)
        if any(report.get(flag) is not False for flag in (
                "qualification_input", "ordinary_tier_credit", "default_adoption", "speed_ranking",
                "cross_cohort_pooling")):
            raise ValueError("Atlas progress cannot confer qualification, default or speed credit")
        readout = paths[key].with_name("README.md")
        if not readout.is_file():
            raise ValueError("Atlas progress readout is unavailable: " + key)
        inputs[key] = dict(report=(Path("reports/forge") / relative).as_posix(), sha256=digest,
                           readout=readout.relative_to(root).as_posix(), readout_sha256=file_hash(readout))
        reports[key] = report
    baseline = reports["baseline"]
    counts = dict(Counter(case["diagnostic_status"] for case in baseline["cases"]))
    if (counts != {"PASS": 7, "FAIL": 11, "BLOCKED": 8} or baseline["counts"]["required"] != 26
            or len({case["parent_id"] for case in baseline["cases"]}) != 26
            or {tier: row["required"] for tier, row in baseline["tier_counts"].items()}
               != {"1": 5, "2": 19, "3": 2}):
        raise ValueError("Atlas baseline diagnostic denominator changed")
    executed = [case for case in baseline["cases"] if case["diagnostic_status"] in {"PASS", "FAIL"}]
    if (len(executed) != 18 or any(
            _representation_kind(case["applied"].get("prior")) != "Particles"
            or case["applied"]["prior"].get("sigma") != 0.0
            or case["applied_recipe"].get("prior_kind") != "particles"
            or case["applied_recipe"].get("standardize") is not False
            or case["applied_recipe"].get("sigma_rel") != 0.0 for case in executed)):
        raise ValueError("Atlas baseline representation requires its actual applied particle law")
    baseline_representation = dict(label="Particles", evidence_source="actual applied priors and complete Recipes",
        report=inputs["baseline"]["report"], report_sha256=inputs["baseline"]["sha256"],
        cases={case["id"]: dict(prior=case["applied"]["prior"],
                               applied_recipe_sha256=case["applied_recipe_sha256"],
                               frozen_definition=case["frozen_definition"])
               for case in executed})
    for key in ("adaptations_v3", "adaptations_v4b"):
        report = reports[key]
        if (report["counts"]["required_family_cells"] != 130
                or report["counts"]["required_slots_per_family"] != 26 or len(report["families"]) != 5
                or any(len(family["slots"]) != 26
                       or dict(Counter(slot["diagnostic_status"] for slot in family["slots"].values()))
                          != family["counts"] for family in report["families"].values())):
            raise ValueError("Atlas adaptation family denominators changed")
    groups = (
        ("Conditional", "atlas_conditional", "adaptations_v3", ["trajectory", "residual_student", "unipolar", "mid_scale_identity"],
         "conditional_policy_selected_cloud_v1"),
        ("AE routed", "atlas_ae_routed", "adaptations_v3", ["ae_gan_hold"], "ae_routed_policy_v1"),
        ("Unused token", "atlas_routed", "adaptations_v4b", ["unused_token_hold"], "routed_policy_selected_cloud_v1"),
        ("Cover leftover", "atlas_multibank", "adaptations_v4b", ["cover_leftover"], "multibank_policy_v1"),
        ("Word joint N11", "atlas_word_joint_min11", "adaptations_v4b", ["five_word_joint_acquisition"], "word_joint_policy_min11_v1"),
    )
    rows = []
    for label, family, key, parents, cohort in groups:
        report = reports[key]
        family_report = report["families"][family]
        required = [parent + "_" + cohort for parent in parents]
        statuses = [family_report["slots"][task]["diagnostic_status"] for task in required]
        declarations = {task: report["questions"][task] for task in required}
        kinds = {_representation_kind(question["prior"]) for question in declarations.values()}
        if len(kinds) != 1 or "UNKNOWN" in kinds:
            raise ValueError("Named model representation requires its declared question laws")
        representation = next(iter(kinds))
        if family == "atlas_ae_routed":
            representation = "MoG (fixed σ .025; routed AE)"
        elif family == "atlas_conditional":
            representation = "Particles (conditional clouds / role banks)"
        elif family == "atlas_routed":
            representation = "Particles (shared / slot parameter bank)"
        elif family == "atlas_multibank":
            representation = "Particles (two routed clouds)"
        else:
            representation = "Particles (N11 joint cloud; free encoder)"
        media, applied = {}, {}
        for attempt in family_report["attempts"]:
            outcome = attempt.get("outcome", {})
            task = outcome.get("task_id")
            if task in required:
                actual = outcome["applied"]
                if (_representation_kind({"kind": actual["recipe"].get("prior_kind")})
                        != _representation_kind(declarations[task]["prior"])
                        or actual.get("prior") and _representation_kind(actual["prior"])
                           != _representation_kind(declarations[task]["prior"])):
                    raise ValueError("Named representation conflicts with its actual applied owner law")
                applied[task] = dict(recipe=outcome["applied"]["recipe"],
                                     table_ownership=outcome["applied"].get("table_ownership"),
                                     prior=outcome["applied"].get("prior"))
                media[task] = dict(outcome["gif_publication"],
                    path=(Path(inputs[key]["report"]).parent / outcome["gif_publication"]["path"]).as_posix())
        rows.append(dict(label=label, family=family, cohort=cohort, task_ids=required, required=26,
                         targeted_required=len(required), passed=statuses.count("PASS"),
                         targeted_counts=dict(Counter(statuses)), counts=family_report["counts"],
                         source=report["source"], readout=inputs[key]["readout"], media=media,
                         representation=dict(label=representation, report=inputs[key]["report"],
                            report_sha256=inputs[key]["sha256"],
                            declarations={task: dict(prior=q["prior"], policy_contract=q["policy_contract"],
                                execution_sha256=q["execution_sha256"], evaluation_sha256=q["evaluation_sha256"])
                                for task, q in declarations.items()}, applied=applied),
                         qualification_input=False))
    word = reports["word_context"]
    if (rows[-1]["targeted_counts"] != {"INVALID": 1} or word["execution_status"] != "INVALID"
            or word["certified_numerical_gate"] != "UNAVAILABLE"
            or any(word["source"].get(key) != value
                   for key, value in reports["adaptations_v4b"]["source"].items())
            or word["raw_reported_clock"]["completed_steps"] != 20001
            or word["actual_prior_rows"] != 11 or word["canonical_target_words"] != 5
            or set(word["raw_reported_clock"]["optimizer_updates"].values()) != {20001}):
        raise ValueError("Atlas retained word clock/grade/source identity changed")
    illustration = word["retained_illustration"]
    rows[-1]["media"][word["task_id"]] = dict(path=(Path(inputs["word_context"]["report"]).parent /
        illustration["path"]).as_posix(), sha256=illustration["sha256"], bytes=illustration["bytes"],
        accepted_numeric_verdict="UNAVAILABLE", caption_required=True)
    return dict(schema_version=2, scope="separate_source_scoped_model_scores",
                baseline=dict(required=26, counts=counts, source=baseline["source"],
                              candidate_id=baseline["candidate_id"],
                              shared_recipe_overrides=baseline["shared_recipe_overrides"],
                              representation=baseline_representation,
                              readout=inputs["baseline"]["readout"]),
                adaptations=rows,
                word=dict(status="INVALID", numerical_gate="UNAVAILABLE", raw_completed_steps=20001,
                          recorded_endpoint=word["recorded_endpoint"], readout=inputs["word_context"]["readout"]),
                inputs=inputs, qualification_input=False, qualification_reuse=False,
                ordinary_tier_credit=False, cross_cohort_pooling=False, default_adoption=False, speed_ranking=False)


def _original_pr223_atlas(result):
    """Expose the verified original-law replay without changing selected rows."""
    studies = result.get("completed_api_studies", {}).get("rows", [])
    matches = [row for row in studies if row.get("id") == "atlas19_original"]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Original PR223 display requires one complete replay publication")
    study = matches[0]
    config_sha = "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"
    reference_sha = "1e1b536bfd66e20a240436b417d429059d22ea55c2b6ecfff31e796b96c470fa"
    recipe = study.get("recipe", {})
    options = recipe.get("overrides", {})
    requested = dict(lr=.00425, prior_lr_mult=2., d_lr_mult=1., prior_kind="particles",
                     sigma_rel=0., output_noise_mode="learnable", output_noise_std=.029,
                     serve_average=4., ema_decay=.995, total_steps=None)
    if (study.get("required_cells") != 19 or study.get("counts") != {"PASS": 19}
            or study.get("declared_updates") != 48800
            or study.get("qualification_input") is not False or study.get("reuse") is not False
            or study.get("cross_cohort_pooling") is not False
            or recipe.get("sha256") != config_sha
            or recipe.get("path_in_snapshot") != "configs/100gaussians/atlas.json"
            or any(options.get(key) != value for key, value in requested.items())
            or study.get("source", {}).get("protected_files_sha256", {}).get(
                "configs/100gaussians/atlas.json") != config_sha):
        raise ValueError("Original PR223 display recipe, source or original scope changed")
    cells = study.get("cells", [])
    expected = {"native": (3, 1234), "moving": (3, 1234), "portability": (13, 0)}
    if len(cells) != 19 or len({cell["id"] for cell in cells}) != 19:
        raise ValueError("Original PR223 display requires all 19 distinct full questions")
    for group, (count, seed) in expected.items():
        members = [cell for cell in cells if cell["definition"].get("group") == group]
        if len(members) != count:
            raise ValueError("Original PR223 display question groups changed")
        for cell in members:
            definition = cell["definition"]
            if (cell.get("full_protocol_complete") is not True
                    or cell.get("execution_status") != "PASS" or cell.get("scientific_status") != "PASS"
                    or definition.get("config_sha256") != config_sha
                    or definition.get("reference_sha256") != reference_sha
                    or definition.get("sampling") != "original noisy state-selected primary; clean diagnostics separate"
                    or definition.get("original_host", {}).get("seed") != seed
                    or not definition.get("original_requirements") or not definition.get("observation_steps")
                    or not cell.get("media")):
                raise ValueError("Original PR223 display full gates, seeds or noisy law changed")
            if group == "native" and cell.get("native_gates", {}).get("noisy") != {
                    "coverage": "PASS", "accuracy": "PASS"}:
                raise ValueError("Original PR223 display requires noisy native coverage and accuracy")
    original = dict(label="Original PR223 Atlas FULL", representation="Particles", required=19,
                counts={"PASS": 19}, readout=study["readout"], publication=deepcopy(study["publication"]),
                source=deepcopy(study["source"]), recipe=deepcopy(recipe),
                seeds={group: seed for group, (_, seed) in expected.items()},
                question_ids=[cell["id"] for cell in cells],
                scope="Faithful replay of the full original 19-question noisy-law protocol",
                qualification_input=False, qualification_reuse=False, default_adoption=False,
                speed_ranking=False, new_current_retest_credit=False)
    fresh = result.get("passive_publications", {}).get("latest", {}).get("pg_pr223_full19_passive_publication_v1")
    if fresh:
        original["fresh_retest"] = deepcopy(fresh)
    native3 = result.get("passive_publications", {}).get("latest", {}).get("pg_pr223_native3_passive_publication_v1")
    if native3:
        original["native3_continuation"] = deepcopy(native3)
    repaired = result.get("passive_publications", {}).get("latest", {}).get("pg_pr223_native3_passive_publication_v2")
    if repaired:
        original["native3_repaired_continuation"] = deepcopy(repaired)
    return original


def _original_pr223_score_line(original, root, path):
    link = os.path.relpath(root / original["readout"], path.parent)
    fresh = original.get("fresh_retest")
    context = "fresh retest PENDING"
    if fresh:
        readout = os.path.relpath(root / fresh["readout"], path.parent)
        cost_link = os.path.relpath(root / fresh["final_cost"]["path"], path.parent)
        counts = fresh["counts"]
        remaining = ", ".join(f"{n} {status}" for status, n in counts.items() if status not in {"PASS", "FAIL"})
        context = (f"[fresh retest]({readout}): {fresh['accepted_counts']['PASS']}/19 PASS · "
                   f"{fresh['accepted_counts']['FAIL']} FAIL" + (f" · {remaining}" if remaining else "") +
                   f" · overall {fresh['status']} · source `{fresh['source']['origin_commit'][:8]}`/"
                   f"`{fresh['source']['digest'][:8]}` · [final charge {fresh['cost']['charged_seconds']:.6f} / 10800 s]({cost_link})")
    native3 = original.get("native3_continuation")
    if native3:
        readout = os.path.relpath(root / native3["readout"], path.parent)
        cost_link = os.path.relpath(root / native3["final_cost"]["path"], path.parent)
        remaining = ", ".join(f"{n} {status}" for status, n in native3["counts"].items()
                              if status not in {"PASS", "FAIL"})
        context += (f"<br>[Native3 continuation]({readout}): {native3['accepted_counts']['PASS']}/3 PASS · "
                    f"{native3['accepted_counts']['FAIL']} FAIL · {native3['accepted_counts']['UNAVAILABLE']} unavailable" +
                    (f" · {remaining}" if remaining else "") +
                    f" · overall {native3['status']} · source `{native3['source']['origin_commit'][:8]}`/"
                    f"`{native3['source']['digest'][:8]}` · [inclusive campaign charge "
                    f"{native3['cost']['charged_seconds']:.6f} / 10800 s]({cost_link}); prior cases and cumulative metadata counted once")
    repaired = original.get("native3_repaired_continuation")
    if repaired:
        readout = os.path.relpath(root / repaired["readout"], path.parent)
        cost_link = os.path.relpath(root / repaired["final_cost"]["path"], path.parent)
        remaining = ", ".join(f"{n} {status}" for status, n in repaired["counts"].items()
                              if status not in {"PASS", "FAIL"})
        context += (f"<br>[Repaired native3 retest]({readout}): {repaired['accepted_counts']['PASS']}/3 PASS · "
                    f"{repaired['accepted_counts']['FAIL']} FAIL · {repaired['accepted_counts']['UNAVAILABLE']} unavailable" +
                    (f" · {remaining}" if remaining else "") +
                    f" · overall {repaired['status']} · source `{repaired['source']['origin_commit'][:8]}`/"
                    f"`{repaired['source']['digest'][:8]}` · [inclusive campaign charge "
                    f"{repaired['cost']['charged_seconds']:.6f} / 10800 s]({cost_link}); "
                    "original19 and pretraining-invalid case debits separately once, SAME cumulative metadata once; separate source, no pooled19 credit")
    headline = "**19/19 PASS** · full original recipe/law replay "
    if (fresh and repaired and fresh.get("required") == 19 and fresh.get("completed") == 16
            and fresh.get("accepted_counts") == {"PASS": 16, "FAIL": 0, "UNAVAILABLE": 3}
            and fresh.get("counts") == {"PASS": 16, "INVALID": 1, "NOT_RUN": 2}
            and repaired.get("required") == 3 and repaired.get("completed") == 3
            and repaired.get("accepted_counts") == {"PASS": 3, "FAIL": 0, "UNAVAILABLE": 0}
            and repaired.get("status") == "PASS" and fresh.get("recipe")
            and repaired.get("recipe") == fresh["recipe"]
            and repaired.get("prior_reference", {}).get("source") == fresh.get("source")
            and repaired.get("source") != fresh.get("source")
            and fresh.get("publication", {}).get("sha256")
            and repaired.get("prior_reference", {}).get("files_sha256", {}).get(
                fresh["publication"]["path"]) == fresh["publication"]["sha256"]):
        headline = ("**19/19 original cases PASS — verified across two source-bound runs** · "
                    "historical **19/19 PASS** · full original recipe/law replay ")
    return (f"| [Original PR223 Atlas FULL · LR .00425 / prior2]({link}) | "
            f"{original['representation']} | {headline}"
            f"· [19 original goal GIFs]({link})<br>Policy-selected; averaging enabled; learned output kernel (init .029) "
            f"· native/moving seed1234, portability seed0 · {context}; no current-26/default credit |")


def _half_base_score_line(half_base, root, path):
    def link(relative):
        return os.path.relpath(root / relative, path.parent)
    metrics = half_base.get("result", {}).get("final_metrics")
    endpoint = (f"<br>Final quality {metrics['quality_fraction']:.6f} · modes {metrics['modes']}/5 · "
                f"mass TV {metrics['mass_tv']:.6f} · all-five exact {metrics['reconstruction_exact']}"
                if metrics else "")
    return (f"| [half_base LR .00265625 / prior1.5 / D1 · source `{half_base['source']['origin_commit'][:8]}`]"
            f"({link(half_base['readout'])}) | {half_base['representation']['label']} | "
            f"**0/26 PASS** · FAIL 1 · NOT_RUN 25 · COMPLETE{endpoint}<br>"
            f"[accepted 20,001-update goal GIF]({link(half_base['media']['path'])}) |")


def _atlas_progress_markdown(progress, root, path, *, half_base=None):
    """Rows in the single current table, each with its own measured scope."""
    def link(relative):
        return os.path.relpath(root / relative, path.parent)
    baseline = progress["baseline"]
    counts = baseline["counts"]
    lines = [f"| [Atlas C6 CHANGED-rate / noise-OFF · LR .0053125 / prior1.5 · source `{baseline['source']['origin_commit'][:8]}`]({link(baseline['readout'])}) "
             f"| {baseline['representation']['label']} | **{counts['PASS']}/26 PASS** · "
             f"FAIL {counts['FAIL']} · BLOCKED {counts['BLOCKED']} "
             f"<br>Selected-policy diagnostic/reference · seed0 · [source and 18 goal GIFs]({link(baseline['readout'])}) |"]
    for row in progress["adaptations"]:
        gates = f"**{row['passed']}/{row['required']} PASS** · NOT_RUN {row['counts'].get('NOT_RUN', 0)}"
        readout = row["readout"]
        if row["family"] == "atlas_word_joint_min11":
            gates = "**INVALID 1** · NOT_RUN 25 · numerical gate UNAVAILABLE"
            readout = progress["word"]["readout"]
        media_label = "retained INVALID illustration" if row["family"] == "atlas_word_joint_min11" else "goal GIFs"
        media = " · ".join(f"[{task.split('_' + row['cohort'])[0]}]({link(pin['path'])})"
                           for task, pin in sorted(row["media"].items()))
        lines.append(f"| [`{row['family']}` · C6 · `{row['source']['origin_commit'][:8]}`]({link(readout)}) "
                     f"| {row['representation']['label']} | {gates}<br>Named-family diagnostic · {media_label}: {media} |")
    if half_base:
        lines.append(_half_base_score_line(half_base, root, path))
    return lines


def _policy_score_scope(root, path, *, half_base=None):
    link = os.path.relpath(root / "reports/forge/technique-inventory.json", path.parent)
    return ["The C6 diagnostic/reference changes rates to LR .0053125 / prior-rate 1.5 and disables "
              "evaluation output noise, with seed0 and declared host-specific "
              "Recipe fields. Representation labels come from the pinned applied priors, Recipes and "
              "routed table owners; a parameter bank is not a sampled MoG. "
              "[Full source and representation bindings](" + link + "). "
              "Original N5 word execution remains BLOCKED. " +
              ("The original C6 N11 illustration has no accepted numerical grade. The separate half_base run "
               "completed all 20,001 updates and 24 reads, with zero passing reads and an accepted numerical FAIL."
               if half_base else "The N11 illustration has no accepted numerical grade."), ""]


def _separate_baseline_links(result, row, root, path):
    """Navigate separate Atlas/C6 evidence without granting current tier credit."""
    family = row.get("trainer_family", row["candidate_id"].split("--", 1)[0])
    if family not in {"atlas", "e22"}:
        return []
    studies = {study["id"]: study for study in
               result.get("completed_api_studies", {}).get("rows", [])}
    links = []
    historical = studies.get("atlas19_original")
    if historical:
        counts = historical["counts"]
        if counts == {"PASS": historical["required_cells"]}:
            link = os.path.relpath(root / historical["readout"], path.parent)
            links.append(f"[Atlas history: {counts['PASS']}/{historical['required_cells']} PASS]({link})")
    hold = studies.get("c6_hold")
    if hold and hold["counts"] == {"FAIL": hold["required_cells"]}:
        debug = result.get("baseline_debugging", {}).get("readout")
        link = os.path.relpath(root / (debug or hold["readout"]), path.parent)
        label = {"atlas": "Atlas", "e22": "E22"}[family]
        links.append(f"[C6 {label} hold FAIL]({link})")
    return links


COMMON26_DISPLAY_AUDIT = Path("configs/forge/selections/common26-display-audit-v1.json")


def _load_common26_display_choices(root):
    """Load explicit display identities; a cached score or selection cannot choose."""
    path = root / COMMON26_DISPLAY_AUDIT
    if not path.is_file():
        return None
    raw = path.read_bytes()

    def unique_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate common-26 audit field")
            value[key] = item
        return value

    manifest = json.loads(raw, object_pairs_hook=unique_fields)
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema", "chosen_configurations"}
            or manifest["schema"] != "pg_common26_display_identity_audit_v1"
            or not isinstance(manifest["chosen_configurations"], dict)):
        raise ValueError("common-26 audit requires the explicit display-only identity schema")
    reference = {"path": COMMON26_DISPLAY_AUDIT.as_posix(),
                 "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    choices = deepcopy(manifest["chosen_configurations"])
    for family, choice in choices.items():
        if choice is None:
            continue
        if not isinstance(choice, dict) or "audit_reference" in choice:
            raise ValueError("raw common-26 identities must omit their injected audit reference")
        choice["audit_reference"] = deepcopy(reference)
    return choices


def _common26_display_projection(result, root):
    """Keep the unscored original common-26 projection in its historical scope."""
    from experiments.forge.common26_comparison import FAMILIES, TASKS_BY_TIER, project_common26
    from experiments.forge.trainer_families import family_for_candidate
    legacy_ids = {name for name, _ in FAMILIES}
    rows = result["rows"]
    if any(row.get("trainer_family") not in legacy_ids for row in rows):
        indexed = {row["trainer_family"]: deepcopy(row) for row in
                   result.get("historical_family_rows", []) if row["trainer_family"] in legacy_ids}
        for original in rows:
            row = deepcopy(original)
            family = family_for_candidate(root, row["candidate_id"],
                                          {"trainer_family": row["trainer_family"]}
                                          if row.get("trainer_family") in legacy_ids else None)
            if family["id"] in legacy_ids:
                row["trainer_family"] = family["id"]
                indexed.setdefault(family["id"], row)
        rows = [indexed[name] for name, _ in FAMILIES if name in indexed]
    catalogs = result
    if result.get("view_revision") != 3 and (root / "configs/forge/view-history/discriminator_stability-v3.json").is_file():
        catalogs = {**result, "view_revision": 3,
                    "tier_requirements": {tier: list(tasks) for tier, tasks in TASKS_BY_TIER.items()}}
    return project_common26(rows, catalogs,
                            chosen_configurations=_load_common26_display_choices(root))


def _common26_link(root, path, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or any(ord(char) < 32 for char in relative)
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("common-26 evidence links require portable repository paths")
    return os.path.relpath(root / relative, path.parent)


def _common26_label(value):
    return (str(value).replace("\\", "\\\\").replace("|", "\\|")
            .replace("[", "\\[").replace("]", "\\]")
            .replace("\n", " ").replace("\r", " "))


def _common26_reference_card(row, root):
    name = row["candidate_id"]
    # This link names the retained canonical reference, never a fresh execution.
    relative = Path("configs/forge/configurations") / (name + ".json")
    if not (root / relative).is_file():
        relative = Path("configs/forge/ideas") / (name + ".json")
    return relative.as_posix()


def _common26_evidence_links(result, root, path):
    """Retain source-scoped readouts outside the comparison, without old scores."""
    links, seen = [], set()

    def add(label, metadata):
        readout = metadata.get("readout") if isinstance(metadata, dict) else None
        if readout and readout not in seen:
            links.append(f"- [{_common26_label(label)}]({_common26_link(root, path, readout)})")
            seen.add(readout)

    original = result.get("original_pr223_atlas", {})
    add("Original Atlas recipe and serving-law evidence", original)
    for key, label in (("fresh_retest", "Original-suite stopped retest"),
                       ("native3_continuation", "Native continuation first attempt"),
                       ("native3_repaired_continuation", "Repaired native continuation")):
        add(label, original.get(key, {}))
    progress = result.get("atlas_unblocking_progress", {})
    add("C6 changed-rate / output-noise-off diagnostic", progress.get("baseline", {}))
    for row in progress.get("adaptations", []):
        add(str(row.get("family", "Named-family")) + " adaptation evidence", row)
    add("Retained word context", progress.get("word", {}))
    add("Word half-base rate contrast", result.get("word_half_base_diagnostic", {}))
    add("C6 baseline selection and retained diagnosis", result.get("baseline_debugging", {}))
    for study in result.get("completed_api_studies", {}).get("rows", []):
        add("Separate completed study: " + str(study.get("id", "source-bound readout")), study)
    for score in result.get("standalone_api_scores", []):
        label = ("Standalone API evidence: " + str(score.get("trainer_family", "recorded family"))
                 + " · " + str(score.get("case", {}).get("title", "recorded question")))
        add(label, score)
        gif = score.get("gif")
        if gif and gif not in seen:
            links.append(f"- [{_common26_label(label)} · actual-training GIF]"
                         f"({_common26_link(root, path, gif)})")
            seen.add(gif)
    atlas = next(row for row in result["rows"] if row["trainer_family"] == "atlas")
    reference = _common26_reference_card(atlas, root)
    links.append("- [Differing canonical Atlas reference; not the requested Full Atlas configuration]"
                 f"({_common26_link(root, path, reference)})")
    return links


def _common26_current_markdown(result, root, path):
    projection = _common26_display_projection(result, root)
    scientific = {row["trainer_family"]: row for row in result["rows"]}
    json_link = path.with_suffix(".json").name
    lines = ["# Current model/configuration scores", "",
             "One configuration slot per family uses the same fresh common-26 comparison. "
             "NOT_RUN means accepted fresh comparison evidence is unavailable; eligibility is a separate status.", "",
             "| Model/configuration | Representation | Fresh common-26 score/status |",
             "| --- | --- | --- |"]
    for row in projection["rows"]:
        family = row["family_id"]
        label = _common26_label(row["label"])
        if scientific.get(family, {}).get("selection", {}).get("freshness") == "stale":
            label += "<br>Recorded source measurement · **stale**; no current qualification"
        requested, chosen = row["requested_configuration"], row["chosen_configuration"]
        if requested:
            label += (f" · [{requested['path']}@{requested['sha256'][:12]}]"
                      f"({_common26_link(root, path, requested['path'])})")
        if chosen:
            identity = _common26_label(chosen["candidate_id"])
            label += (f"<br>[{identity}]"
                      f"({_common26_link(root, path, chosen['audit_reference']['path'])})")
        else:
            if family != "atlas":
                reference = scientific[family]
                identity = _common26_label(reference["candidate_id"])
                label += (f" · [canonical reference: {identity}]"
                          f"({_common26_link(root, path, _common26_reference_card(reference, root))})")
            label += "<br>Configuration freeze pending"
        representation = row["representation"]["kind"]
        if row["representation"]["basis"] == "DECLARED":
            representation += " (declared per task)" if representation == "MoG / Particles" else " (declared)"
        status = ("NOT_RUN (26 required)<br>"
                  f"[Eligibility: {row['eligibility']['status']}]({json_link}#common26_display)")
        lines.append(f"| {label} | {representation} | {status} |")
    lines += ["", "Historical and diagnostic evidence has separate configurations, serving laws and scopes. "
              "These links supply no fresh common-26 score:", ""]
    lines += _common26_evidence_links(result, root, path)
    if result.get("declared_view"):
        lines += ["", f"[Current declared view and additional-task scope]({json_link}#declared_view): "
                  "these separate declarations do not fill the fresh common-26 comparison."]
    lines += ["", "## Qualification and scope", "",
              "Fresh comparison evidence must bind the chosen configuration, unchanged required tasks, "
              "gates, budgets, source, runtime and serving law to a recognized new initialization and "
              "independent grade. Earlier source-bound records remain evidence in their own scopes.", "",
              "Representation describes a declared prior or per-task host law. UNKNOWN means the chosen "
              "identity and complete prior contracts are not bound. The Full Atlas Particles label describes "
              "its requested original configuration; its common-task eligibility remains BLOCKED.", "",
              "This display grants no current qualification, default adoption or speed comparison. "
              "The underlying scientific selection and historical records remain unchanged.", "",
              f"[Canonical rows, configuration alternatives and source-scoped evidence]({json_link}) · "
              f"[Evidence and archived publication identities]({os.path.relpath(root / EVIDENCE_MANIFEST, path.parent)})", "",
              "Regenerate this presentation from committed metadata:", "",
              "```sh", "python reports/forge/regenerate_technique_inventory.py" +
              (" --refresh-publication" if result.get("publication_refresh") else ""), "```", "",
              f"Publication input digest `{result['provenance']['input_digest']}`.", ""]
    return "\n".join(lines)



def _stale_measurement_labels(markdown, result, root, path):
    """Expose recorded-only scope on the main family rows, preserving scores."""
    stale = {row["trainer_family"] for row in result.get("rows", [])
             if row.get("selection", {}).get("freshness") == "stale"}
    lines = markdown.splitlines()
    for family in result.get("family_progress", {}).get("families", []):
        # A family row can display a measurement owned by a separate configuration
        # family (e.g. BCAP's selected DualNorm recipe); match render_leaderboard.
        anchor = family.get("current_cohort_anchor")
        measured = {family["id"]} | {cohort["configuration_family"] for cohort in family.get("cohorts", [])
                                     if cohort.get("configuration_family")
                                     and (not anchor or cohort.get("anchor") == anchor)}
        if not measured & stale:
            continue
        target = "](" + os.path.relpath(root / family["page"], path.parent) + ")**"
        matching = [index for index, line in enumerate(lines)
                    if line.startswith("| **") and target in line.split(" | ", 1)[0]]
        if len(matching) != 1:
            raise ValueError("recorded measurement needs one explicit stale label in the family table")
        index = matching[0]
        end = lines[index].index(" | ")
        lines[index] = (lines[index][:end] + "<br>Recorded source measurement · **stale**; no current qualification"
                        + lines[index][end:])
    return "\n".join(lines)


def _selected_diagnostic_rows(markdown, result, root, path):
    """Show nonqualifying Tier 1-3 diagnostic rows for promoted selected recipes."""
    progress = result.get("family_progress") or {}
    preflight_path = Path(root) / "reports/forge/tier-coverage-20261014/preflight.json"
    heading = "## Research recipes and separate diagnostics"
    if not progress or heading not in markdown or not preflight_path.is_file():
        return markdown
    from experiments.forge.family_reports import (_selected_cohorts, _selected_row,
                                                  _summed_views, link, score)
    from experiments.forge.policy_coverage import PARENTS, TIERS as BASELINE_TIERS
    from experiments.forge.research_tiers import attempt_count
    from experiments.forge.views import COUNTED_OR_HELD
    pins = {item["family"]: item for item in read_json(preflight_path).get("families", [])}
    lowest_tier: dict[str, int] = {}
    for view in _summed_views(progress):
        for assignment in view.get("assignments", []):
            if assignment.get("importance") in COUNTED_OR_HELD or "removal" in assignment:
                tier = int(assignment["qualification_tier"])
                task = assignment["task"]
                lowest_tier[task] = min(tier, lowest_tier.get(task, tier))
    declared_tiers = {str(tier): {task for task, assigned in lowest_tier.items() if assigned == tier}
                      for tier in (1, 2, 3)}
    baseline_tiers = {str(tier): set(tasks) for tier, tasks in BASELINE_TIERS.items()}
    extra_tasks = set().union(*declared_tiers.values()) - set(PARENTS)
    additions: dict[str, list[str]] = {}
    family_cells: list[str] = []
    for family, cohort in _selected_cohorts(progress):
        family_cell = link(root, path, family["label"], family["page"])
        family_cells.append(family_cell)
        row = _selected_row(result, cohort)
        candidate_id = row.get("candidate_id")
        if not candidate_id or pins.get(family["id"], {}).get("candidate_id") == candidate_id:
            continue
        tasks_map = cohort.get("tasks", {})
        recorded = {name: info["status"] for name, info in tasks_map.items()
                    if info.get("status") in {"PASS", "FAIL"}}
        attempts = {name: info["run_attempt"] for name, info in tasks_map.items()
                    if isinstance(info.get("run_attempt"), dict)}
        if not any(attempt.get("attempted") and name not in recorded for name, attempt in attempts.items()):
            continue

        def format_row(label, requirements):
            label_cell = link(root, path, label, family["page"], cohort["anchor"])
            cells = [family_cell, label_cell]
            for tier in ("1", "2", "3"):
                required = requirements[tier]
                if not required:
                    cells.append("—")
                    continue
                counted = attempt_count(recorded, attempts, required, fill=True)
                cells.append(
                    link(
                        root,
                        path,
                        f"{counted['passed']}/{counted['total']}",
                        family["page"],
                        f"{cohort['anchor']}-tier-{tier}",
                    )
                )
            all_required = set().union(*requirements.values())
            total = attempt_count(recorded, attempts, all_required, fill=True)
            cells.append(link(root, path, f"{total['passed']}/{total['total']}", family["page"], cohort["anchor"]))
            return "| " + " | ".join(cells) + " |"

        rows_for_family = [format_row(f"Selected recipe 1-for-1 baseline-35 diagnostics ({candidate_id}); seed 0",
                                      baseline_tiers)]
        if any(task in recorded or attempts.get(task, {}).get("attempted") for task in extra_tasks):
            rows_for_family.append(format_row(
                f"Selected recipe declared-suite Tier 1–3 diagnostics ({candidate_id}); seed 0",
                declared_tiers))
        additions[family_cell] = rows_for_family
    if not additions:
        return markdown
    before, section_and_after = markdown.split(heading, 1)
    lines = section_and_after.splitlines()
    table_start = next((idx for idx, line in enumerate(lines) if line.startswith("| Family |")), None)
    if table_start is None:
        return markdown
    table_end = table_start + 2
    while table_end < len(lines) and lines[table_end].startswith("| "):
        table_end += 1
    table_rows = lines[table_start + 2:table_end]
    for family_cell in family_cells:
        new_rows = additions.get(family_cell)
        if not new_rows:
            continue
        prefix = f"| {family_cell} | "
        indices = [idx for idx, line in enumerate(table_rows) if line.startswith(prefix)]
        if indices:
            insert_at = indices[-1] + 1
        else:
            later_prefixes = tuple(f"| {later} | " for later in family_cells[family_cells.index(family_cell) + 1:])
            insert_at = next((idx for idx, line in enumerate(table_rows)
                              if later_prefixes and line.startswith(later_prefixes)), len(table_rows))
        table_rows[insert_at:insert_at] = new_rows
    rebuilt = lines[:table_start + 2] + table_rows + lines[table_end:]
    suffix = "\n" if section_and_after.endswith("\n") else ""
    return before + heading + "\n".join(rebuilt) + suffix


_CURRENT_SOURCE_TIER1_CACHE: dict[tuple, dict | None] = {}
_CONCLUDED_TIER1_RECORDS_CACHE: dict[tuple, tuple[list[tuple[str, str, dict]], tuple[str, ...]]] = {}
_SCOPED_COVERAGE_KEYS = ("research_scope", "repair_scope", "policy_scope", "cohort_scope")
_READOUT_SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml", ".sh"}


def _is_sha256_hex(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _is_valid_attempt_id(value) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "/" not in value
        and "\\" not in value
        and value not in {".", ".."}
    )


def _load_attempt_coverage_task_fps(root: Path) -> dict[tuple[str, str], set[tuple[str, str]]]:
    cov_path = Path(root) / "reports/forge/attempt-coverage.json"
    if not cov_path.is_file():
        return {}
    try:
        doc = read_json(cov_path)
    except Exception:
        return {}
    unscoped: dict[tuple[str, str], set[tuple[str, str]]] = {}
    scoped: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for rec in doc.get("records", []) if isinstance(doc, dict) else []:
        if not isinstance(rec, dict):
            continue
        src = rec.get("source_digest")
        tid = rec.get("task")
        ex_fp = rec.get("execution_sha256")
        ev_fp = rec.get("evaluation_sha256")
        if (
            isinstance(src, str)
            and src
            and isinstance(tid, str)
            and tid
            and isinstance(ex_fp, str)
            and ex_fp
            and isinstance(ev_fp, str)
            and ev_fp
        ):
            target = scoped if any(rec.get(k) for k in _SCOPED_COVERAGE_KEYS) else unscoped
            target.setdefault((src, tid), set()).add((ex_fp, ev_fp))
    merged = dict(scoped)
    merged.update(unscoped)
    # Source identities are immutable. Historical task fingerprints must not
    # be copied under today's source digest merely to populate a live table.
    return merged


def _load_attempt_coverage_family_verdicts(root: Path) -> dict[str, dict[str, str]]:
    cov_path = Path(root) / "reports/forge/attempt-coverage.json"
    if not cov_path.is_file():
        return {}
    try:
        doc = read_json(cov_path)
    except Exception:
        return {}
    unscoped: dict[str, dict[str, str]] = {}
    scoped: dict[str, dict[str, str]] = {}
    for rec in doc.get("records", []) if isinstance(doc, dict) else []:
        if not isinstance(rec, dict):
            continue
        fam = rec.get("family") or rec.get("owning_family")
        tid = rec.get("task")
        st = rec.get("status")
        if (
            isinstance(fam, str)
            and fam
            and isinstance(tid, str)
            and tid
            and st in {"PASS", "FAIL"}
        ):
            is_scoped = any(rec.get(k) for k in _SCOPED_COVERAGE_KEYS)
            target = scoped if is_scoped else unscoped
            target.setdefault(fam, {})[tid] = st
            parent_tid = rec.get("parent_task")
            if isinstance(parent_tid, str) and parent_tid:
                target.setdefault(fam, {}).setdefault(parent_tid, st)
    merged: dict[str, dict[str, str]] = {}
    for key in set(scoped) | set(unscoped):
        combined = dict(scoped.get(key, {}))
        combined.update(unscoped.get(key, {}))
        merged[key] = combined
    return merged


def _load_attempt_coverage_family_records(root: Path) -> dict[tuple[str, str], dict]:
    cov_path = Path(root) / "reports/forge/attempt-coverage.json"
    if not cov_path.is_file():
        return {}
    try:
        doc = read_json(cov_path)
    except Exception:
        return {}
    unscoped: dict[tuple[str, str], dict] = {}
    scoped: dict[tuple[str, str], dict] = {}
    for rec in doc.get("records", []) if isinstance(doc, dict) else []:
        if not isinstance(rec, dict):
            continue
        fam = rec.get("family") or rec.get("owning_family")
        tid = rec.get("task")
        st = rec.get("status")
        if isinstance(fam, str) and fam and isinstance(tid, str) and tid and st in {"PASS", "FAIL"}:
            is_scoped = any(rec.get(k) for k in _SCOPED_COVERAGE_KEYS)
            target = scoped if is_scoped else unscoped
            target[(fam, tid)] = rec
            parent_tid = rec.get("parent_task")
            if isinstance(parent_tid, str) and parent_tid:
                target.setdefault((fam, parent_tid), rec)
    merged = dict(scoped)
    merged.update(unscoped)
    return merged


def _resolve_idea_for_readout_check(
    root: Path,
    candidate_id: str,
    *,
    view_id: str,
    through_tier: int,
    cuda_model: str,
    planning_mod,
) -> dict:
    try:
        return planning_mod.resolve_idea(
            root,
            candidate_id,
            view_id=view_id,
            through_tier=through_tier,
            freeze_source=False,
            execution_backend="cuda",
            cuda_model=cuda_model,
        )
    except ValueError as exc:
        if "select one compatible CUDA model cohort" not in str(exc):
            raise
        return planning_mod.resolve_idea(
            root,
            candidate_id,
            view_id=view_id,
            through_tier=through_tier,
            freeze_source=False,
            execution_backend="cpu",
            cuda_model=None,
        )


class _ReadoutCompatibilityMatcher:
    def __init__(
        self,
        root: Path,
        req: dict,
        *,
        coverage_fps: dict[tuple[str, str], set[tuple[str, str]]],
        planning_mod,
        attempt_cache: dict[str, dict | None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.req = req
        self.source_digest = (req.get("source") or {}).get("digest")
        self.coverage_fps = coverage_fps
        self.planning_mod = planning_mod
        self.req_jobs = list(req.get("jobs", []))
        self.job_by_task: dict[str, dict] = {
            str(tid): job
            for job in self.req_jobs
            for tid in job.get("task_ids", [job["task_id"]])
        }
        self._attempt_cache = attempt_cache if attempt_cache is not None else {}
        self._attempt_covered_cache: dict[str, tuple[str, ...]] = {}
        self._rekey_cache: dict[str, dict[str, str]] = {}

    def _load_attempt_request(self, attempt_id) -> dict | None:
        if not _is_valid_attempt_id(attempt_id):
            return None
        if attempt_id not in self._attempt_cache:
            path = self.root / "reports/forge/attempts" / attempt_id / "request.json"
            if not path.is_file():
                self._attempt_cache[attempt_id] = None
            else:
                try:
                    loaded = read_json(path)
                    self._attempt_cache[attempt_id] = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    self._attempt_cache[attempt_id] = {}
        return self._attempt_cache[attempt_id]

    def _attempt_covered_tasks(self, attempt_id: str, env_doc: dict) -> tuple[str, ...]:
        env_job = env_doc.get("job")
        if isinstance(env_job, dict):
            raw_ids = env_job.get("task_ids")
            if "task_ids" in env_job and not isinstance(raw_ids, list):
                return ()
            covered = (
                raw_ids
                if isinstance(raw_ids, list) and raw_ids
                else ([env_job["task_id"]] if isinstance(env_job.get("task_id"), str) and env_job["task_id"] else [])
            )
            if covered and all(isinstance(x, str) and x for x in covered):
                return tuple(covered)
            if "task_ids" in env_job or "task_id" in env_job:
                return ()
        if not _is_valid_attempt_id(attempt_id):
            return ()
        if attempt_id not in self._attempt_covered_cache:
            res_path = self.root / "reports/forge/attempts" / attempt_id / "result.json"
            tasks_found: tuple[str, ...] = ()
            if res_path.is_file():
                try:
                    res_doc = read_json(res_path)
                    if isinstance(res_doc, dict) and isinstance(res_doc.get("task_results"), list):
                        tasks_found = tuple(
                            str(item["task_id"])
                            for item in res_doc["task_results"]
                            if isinstance(item, dict) and isinstance(item.get("task_id"), str) and item["task_id"]
                        )
                except Exception:
                    tasks_found = ()
            self._attempt_covered_cache[attempt_id] = tasks_found
        return self._attempt_covered_cache[attempt_id]

    def _reconstructed_compat(self, attempt_doc: dict) -> dict[str, str]:
        if not isinstance(attempt_doc, dict):
            return {}
        if "request" in attempt_doc:
            if not isinstance(attempt_doc["request"], dict):
                return {}
            env_req = attempt_doc["request"]
        else:
            env_req = attempt_doc

        j_sci: dict | None = None
        if "job" in attempt_doc:
            env_job = attempt_doc["job"]
            if not isinstance(env_job, dict):
                return {}
            if "science" in env_job:
                if not isinstance(env_job["science"], dict):
                    return {}
                j_sci = env_job["science"]

        req_runtime = None
        if "runtime" in env_req:
            if not isinstance(env_req["runtime"], dict):
                return {}
            req_runtime = env_req["runtime"]
        job_runtime = None
        if j_sci is not None and "runtime" in j_sci:
            if not isinstance(j_sci["runtime"], dict):
                return {}
            job_runtime = j_sci["runtime"]
        if req_runtime is not None and job_runtime is not None and req_runtime != job_runtime:
            return {}
        runtime = req_runtime if req_runtime is not None else job_runtime
        if not isinstance(runtime, dict):
            return {}

        req_rng_runtime = None
        if "rng" in env_req:
            env_rng = env_req["rng"]
            if not isinstance(env_rng, dict):
                return {}
            if "runtime" in env_rng:
                if not isinstance(env_rng["runtime"], dict):
                    return {}
                req_rng_runtime = env_rng["runtime"]
        job_rng_runtime = None
        if j_sci is not None and "rng" in j_sci:
            j_rng = j_sci["rng"]
            if not isinstance(j_rng, dict):
                return {}
            if "runtime" in j_rng:
                if not isinstance(j_rng["runtime"], dict):
                    return {}
                job_rng_runtime = j_rng["runtime"]
        if req_rng_runtime is not None and job_rng_runtime is not None and req_rng_runtime != job_rng_runtime:
            return {}
        rng_runtime = req_rng_runtime if req_rng_runtime is not None else job_rng_runtime

        compute_profiles: dict[str, dict] = {}
        if "compute_profiles" in env_req:
            raw_profiles = env_req["compute_profiles"]
            if not isinstance(raw_profiles, dict):
                return {}
            for b_key, b_prof in raw_profiles.items():
                if not isinstance(b_key, str) or not isinstance(b_prof, dict):
                    return {}
                compute_profiles[b_key] = {k: v for k, v in b_prof.items() if k != "threads"}
        if j_sci is not None and "compute" in j_sci:
            j_comp = j_sci["compute"]
            if not isinstance(j_comp, dict) or not isinstance(j_comp.get("backend"), str):
                return {}
            j_comp_norm = {k: v for k, v in j_comp.items() if k != "threads"}
            b_name = j_comp["backend"]
            if b_name in compute_profiles and compute_profiles[b_name] != j_comp_norm:
                return {}
            compute_profiles[b_name] = j_comp_norm
        if not compute_profiles:
            return {}
        sig = stable_hash({"runtime": runtime, "compute_profiles": compute_profiles, "rng_runtime": rng_runtime})
        if sig not in self._rekey_cache:
            cloned = []
            valid_groups: set[str] = set()
            for job in self.req_jobs:
                sci = job.get("science")
                if not isinstance(sci, dict):
                    continue
                gpus = (job.get("resources") or {}).get("gpus")
                if type(gpus) is int and gpus == 0:
                    backend = "cpu"
                elif type(gpus) is int and gpus > 0:
                    backend = "cuda"
                else:
                    backend = str((job.get("resources") or {}).get("backend") or (sci.get("compute") or {}).get("backend") or "cuda")
                prof = compute_profiles.get(backend)
                group = str(job.get("execution_group", job["task_id"]))
                if isinstance(prof, dict):
                    valid_groups.add(group)
                    new_compute = {**prof, "threads": sci["compute"]["threads"]}
                else:
                    new_compute = dict(sci["compute"])
                new_sci = {**sci, "runtime": runtime, "compute": new_compute}
                if isinstance(rng_runtime, dict) and isinstance(sci.get("rng"), dict):
                    new_sci["rng"] = {**sci["rng"], "runtime": dict(rng_runtime)}
                if "prerequisites" in new_sci and isinstance(new_sci["prerequisites"], dict):
                    new_sci["prerequisites"] = dict(new_sci["prerequisites"])
                cloned.append({**job, "science": new_sci})
            by_task = {m: j for j in cloned for m in j.get("task_ids", [j["task_id"]])}
            if all(p in by_task for j in cloned for p in (j["science"].get("prerequisites") or {})):
                self.planning_mod.rekey_jobs(cloned)
                reconstructed: dict[str, str] = {}
                for j in cloned:
                    grp = str(j.get("execution_group", j["task_id"]))
                    prereq_ok = all(
                        str(by_task[p].get("execution_group", by_task[p]["task_id"])) in valid_groups
                        for p in (j["science"].get("prerequisites") or {})
                    )
                    if grp in valid_groups and prereq_ok:
                        for m in j.get("task_ids", [j["task_id"]]):
                            reconstructed[str(m)] = j["compatibility_key"]
                self._rekey_cache[sig] = reconstructed
            else:
                self._rekey_cache[sig] = {}
        return self._rekey_cache[sig]

    def matches(self, rec: dict, row: dict, tid: str) -> bool:
        job = self.job_by_task.get(tid)
        if job is None:
            return True
        sci = job.get("science")
        if not isinstance(sci, dict):
            return ("compatibility_key" not in row) or (row.get("compatibility_key") == job.get("compatibility_key"))
        if "compatibility_key" in row and row.get("compatibility_key") == job.get("compatibility_key"):
            return True
        if "compatibility_key" in row:
            row_aid = row.get("_attempt_id") or row.get("attempt_id")
            if isinstance(row_aid, str) and row_aid:
                env_doc = self._load_attempt_request(row_aid)
                if env_doc is not None:
                    return row.get("compatibility_key") == self._reconstructed_compat(env_doc).get(tid)
            else:
                matching_envs: list[dict] = []
                for aid in rec.get("attempt_ids", []) if isinstance(rec.get("attempt_ids"), list) else []:
                    env_doc = self._load_attempt_request(aid)
                    if env_doc is None:
                        continue
                    covered = self._attempt_covered_tasks(aid, env_doc)
                    if covered and tid not in covered:
                        continue
                    matching_envs.append(env_doc)
                if matching_envs:
                    return any(
                        row.get("compatibility_key") == self._reconstructed_compat(e).get(tid)
                        for e in matching_envs
                    )
            if not _is_sha256_hex(row.get("compatibility_key")):
                return False
        if isinstance(self.source_digest, str) and self.source_digest:
            for member_tid in job.get("task_ids", [tid]):
                m_key = str(member_tid)
                recorded_pairs = self.coverage_fps.get((self.source_digest, m_key))
                if recorded_pairs is not None:
                    cur_pair = ((sci.get("execution") or {}).get(m_key), (sci.get("evaluation") or {}).get(m_key))
                    if cur_pair not in recorded_pairs:
                        return False
            for prereq_tid in (sci.get("prerequisites") or {}):
                p_key = str(prereq_tid)
                p_job = self.job_by_task.get(p_key)
                if isinstance(p_job, dict) and isinstance(p_job.get("science"), dict):
                    p_sci = p_job["science"]
                    for p_member in p_job.get("task_ids", [p_key]):
                        pm_key = str(p_member)
                        recorded_prereq = self.coverage_fps.get((self.source_digest, pm_key))
                        if recorded_prereq is not None:
                            p_pair = (
                                (p_sci.get("execution") or {}).get(pm_key),
                                (p_sci.get("evaluation") or {}).get(pm_key),
                            )
                            if p_pair not in recorded_prereq:
                                return False
                else:
                    recorded_prereq = self.coverage_fps.get((self.source_digest, p_key))
                    p_info = (sci.get("prerequisites") or {}).get(prereq_tid)
                    if recorded_prereq is not None and isinstance(p_info, dict):
                        p_pair = (p_info.get("execution"), p_info.get("evaluation"))
                        if p_pair not in recorded_prereq:
                            return False
        return True


def _format_tier1_bucket(outcomes: list[str], total: int, *, bold_score: bool = False) -> str:
    counts = Counter(outcomes)
    passed = counts.pop("PASS", 0)
    base = f"**{passed}/{total}**" if bold_score else f"{passed}/{total}"
    labels = {"FAIL": "fail", "PREFLIGHT_BLOCKED": "preflight-blocked",
              "BLOCKED": "blocked", "UNMEASURED": "not run", "UNKNOWN": "not measured"}
    notes = [f"{count} {labels.get(status, status.lower())}"
             for status, count in sorted(counts.items(), key=lambda pair: (list(labels).index(pair[0])
                                              if pair[0] in labels else len(labels), pair[0])) if count]
    return f"{base} ({', '.join(notes)})" if notes else base


def _concluded_current_source_tier1_payload(result, root: Path) -> dict | None:
    root = Path(root).resolve()
    progress = result.get("family_progress")
    if not isinstance(progress, dict) or not progress.get("families"):
        return None
    records_dir = root / "reports/forge/records"
    view_path = root / "configs/forge/views/discriminator_stability.json"
    if not records_dir.is_dir() or not view_path.is_file():
        return None
    record_paths = sorted(records_dir.glob("readout-*.json"))
    if not record_paths:
        return None
    from experiments.forge.family_reports import _selected_cohorts, _selected_row

    selected_specs = []
    for family, cohort in _selected_cohorts(progress):
        row = _selected_row(result, cohort)
        candidate_id = row.get("candidate_id")
        if not candidate_id:
            return None
        card = root / "configs/forge/ideas" / f"{candidate_id}.json"
        if not card.is_file():
            card = root / "configs/forge/configurations" / f"{candidate_id}.json"
        if not card.is_file():
            return None
        selected_specs.append(
            (
                family["id"],
                family["label"],
                family["page"],
                candidate_id,
                card.relative_to(root).as_posix(),
                card.stat().st_mtime_ns,
            )
        )
    if not selected_specs:
        return None

    tasks_dir = root / "configs/forge/tasks"
    task_stats = (
        tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(tasks_dir.glob("*.json")))
        if tasks_dir.is_dir()
        else ()
    )
    defaults_path = root / "configs/forge/defaults.json"
    protocols_dir = root / "configs/forge/protocols"
    views_dir = root / "configs/forge/views"
    leaderboards_dir = root / "configs/forge/leaderboards"
    forge_cfg_stats = (
        (defaults_path.stat().st_mtime_ns, defaults_path.stat().st_size) if defaults_path.is_file() else None,
        tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(protocols_dir.glob("*.json")))
        if protocols_dir.is_dir()
        else (),
        tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(views_dir.glob("*.json")))
        if views_dir.is_dir()
        else (),
        tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(leaderboards_dir.glob("*.json")))
        if leaderboards_dir.is_dir()
        else (),
    )
    src_count = 0
    src_max_mtime = 0
    src_total_size = 0
    for sub in ("particlegan", "experiments", "benchmarks", "lib"):
        sub_dir = root / sub
        if sub_dir.is_dir():
            for src_file in sub_dir.rglob("*"):
                if (
                    src_file.is_file()
                    and src_file.suffix in _READOUT_SOURCE_SUFFIXES
                    and not any(part in {"__pycache__", ".venv", "runs"} for part in src_file.relative_to(root).parts)
                ):
                    st = src_file.stat()
                    src_count += 1
                    if st.st_mtime_ns > src_max_mtime:
                        src_max_mtime = st.st_mtime_ns
                    src_total_size += st.st_size
    cfg_dir = root / "configs"
    if cfg_dir.is_dir():
        for cfg_file in cfg_dir.rglob("*"):
            if (
                cfg_file.is_file()
                and cfg_file.suffix in _READOUT_SOURCE_SUFFIXES
                and "forge" not in cfg_file.relative_to(root).parts
            ):
                st = cfg_file.stat()
                src_count += 1
                if st.st_mtime_ns > src_max_mtime:
                    src_max_mtime = st.st_mtime_ns
                src_total_size += st.st_size

    record_stats = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in record_paths)
    rec_cache_key = (str(root), record_stats)
    if rec_cache_key not in _CONCLUDED_TIER1_RECORDS_CACHE:
        parsed_concluded: list[tuple[str, str, dict]] = []
        ref_aids: set[str] = set()
        for rec_path in record_paths:
            try:
                rec = read_json(rec_path)
            except Exception:
                continue
            if not isinstance(rec, dict) or rec.get("lifecycle") != "concluded" or rec.get("evidence_scope") != "current":
                continue
            scope = (rec.get("study") or {}).get("scope") or {}
            if (
                (scope.get("view") or rec.get("goal")) == "discriminator_stability"
                and scope.get("execution_backend", "cuda") == "cuda"
                and scope.get("cuda_model") in {None, "NVIDIA RTX A6000"}
            ):
                parsed_concluded.append((rec_path.relative_to(root).as_posix(), rec_path.name, rec))
                for aid in rec.get("attempt_ids", []) if isinstance(rec.get("attempt_ids"), list) else []:
                    if _is_valid_attempt_id(aid):
                        ref_aids.add(aid)
                for trow in rec.get("task_results", []) if isinstance(rec.get("task_results"), list) else []:
                    if isinstance(trow, dict):
                        row_aid = trow.get("_attempt_id") or trow.get("attempt_id")
                        if _is_valid_attempt_id(row_aid):
                            ref_aids.add(row_aid)
        if len(_CONCLUDED_TIER1_RECORDS_CACHE) >= 8:
            _CONCLUDED_TIER1_RECORDS_CACHE.pop(next(iter(_CONCLUDED_TIER1_RECORDS_CACHE)))
        _CONCLUDED_TIER1_RECORDS_CACHE[rec_cache_key] = (parsed_concluded, tuple(sorted(ref_aids)))
    concluded, referenced_aids = _CONCLUDED_TIER1_RECORDS_CACHE[rec_cache_key]

    attempts_dir = root / "reports/forge/attempts"
    attempt_stats_list = []
    if attempts_dir.is_dir() and referenced_aids:
        for aid in referenced_aids:
            req_p = attempts_dir / aid / "request.json"
            res_p = attempts_dir / aid / "result.json"
            req_st = (req_p.stat().st_mtime_ns, req_p.stat().st_size) if req_p.is_file() else None
            res_st = (res_p.stat().st_mtime_ns, res_p.stat().st_size) if res_p.is_file() else None
            if req_st is not None or res_st is not None:
                attempt_stats_list.append((aid, req_st, res_st))
    attempt_stats = tuple(attempt_stats_list)

    cov_path = root / "reports/forge/attempt-coverage.json"
    cov_stat = (cov_path.stat().st_mtime_ns, cov_path.stat().st_size) if cov_path.is_file() else None
    host_probe_sig = None
    try:
        import torch as _torch_probe
        from experiments.forge import planning as _planning_probe
        from experiments.forge import sources as _sources_probe

        gpu_probe = _sources_probe._gpu_models()
        gpu_sig = tuple(sorted(tuple(x) for x in gpu_probe)) if gpu_probe is not None else None
        host_probe_sig = (stable_hash(_planning_probe.runtime_manifest()), str(_torch_probe.__version__), gpu_sig)
    except Exception:
        host_probe_sig = None

    cache_key = (
        str(root),
        view_path.stat().st_mtime_ns,
        tuple(selected_specs),
        task_stats,
        forge_cfg_stats,
        (src_count, src_max_mtime, src_total_size),
        record_stats,
        attempt_stats,
        cov_stat,
        host_probe_sig,
    )
    if cache_key not in _CURRENT_SOURCE_TIER1_CACHE:
        payload = None
        try:
            from experiments.forge import planning
            from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint

            coverage_fps = _load_attempt_coverage_task_fps(root)
            # Coverage is diagnostic, even if its family/task labels match.
            # Only compatibility-checked concluded task rows supply observations.
            coverage_verdicts = {}
            shared_attempt_cache: dict[str, dict | None] = {}
            rows_data = []
            source_digests = set()
            view_revision = None
            req_total = 0
            diag_total = 0
            any_matched = False
            runnable_count = 0
            runnable_matched_count = 0
            with planning.resolution_cache(root):
                resolved_specs = []
                for spec in selected_specs:
                    req = _resolve_idea_for_readout_check(
                        root,
                        spec[3],
                        view_id="discriminator_stability",
                        through_tier=1,
                        cuda_model="NVIDIA RTX A6000",
                        planning_mod=planning,
                    )
                    t1_tasks = [
                        item["task"]
                        for item in req["view"]["assignments"]
                        if int(item["qualification_tier"]) == 1 and item["task"] in req["tasks"]
                    ]
                    all_t1_preflight_blocked = bool(t1_tasks) and all(
                        req["tasks"][t].get("preflight_blockers") for t in t1_tasks
                    )
                    resolved_specs.append((all_t1_preflight_blocked, spec, req))
                resolved_specs.sort(key=lambda item: item[0])
                for all_t1_preflight_blocked, (fam_id, fam_label, fam_page, cid, card_rel, _), req in resolved_specs:
                    source_digests.add(req["source"]["digest"])
                    view_revision = req["view"]["revision"]
                    t1_assignments = [
                        item for item in req["view"]["assignments"] if int(item["qualification_tier"]) == 1
                    ]
                    req_tasks = [item["task"] for item in t1_assignments if item.get("importance") == "required"]
                    diag_tasks = [item["task"] for item in t1_assignments if item.get("importance") != "required"]
                    t1_task_set = set(req_tasks) | set(diag_tasks)
                    req_total = len(req_tasks)
                    diag_total = len(diag_tasks)
                    compat_matcher = _ReadoutCompatibilityMatcher(
                        root,
                        req,
                        coverage_fps=coverage_fps,
                        planning_mod=planning,
                        attempt_cache=shared_attempt_cache,
                    )
                    rev = req["candidate_revision"]
                    matching = [
                        item
                        for item in concluded
                        if item[2].get("candidate_id") == cid
                        and item[2].get("candidate_revision") == rev
                        and (item[2].get("source") or {}).get("digest") in {None, req["source"]["digest"]}
                    ]
                    measured: dict[str, str] = {}
                    all_measured: dict[str, str] = {}
                    used_records: list[tuple[str, str]] = []
                    any_t1_row_rejected = False
                    for rec_rel, rec_name, rec in matching:
                        used = False
                        for trow in rec.get("task_results", []):
                            tid = str(trow.get("task_id", ""))
                            if not compat_matcher.matches(rec, trow, tid):
                                if tid in t1_task_set:
                                    any_t1_row_rejected = True
                                continue
                            st = trow.get("gate_status") or trow.get("status")
                            if st in {"PASS", "FAIL", "BLOCKED", "ERROR", "INVALID", "INCOMPLETE", "TIMEOUT", "INTERRUPTED"}:
                                if tid in t1_task_set:
                                    measured[tid] = st
                                    all_measured[tid] = st
                                    used = True
                                elif _is_sha256_hex(trow.get("compatibility_key")):
                                    all_measured[tid] = st
                        if used:
                            used_records.append((rec_rel, rec_name))
                    fam_cov = coverage_verdicts.get(fam_id, {})
                    t1_recorded_set = {
                        t
                        for t in t1_task_set
                        if t in req["tasks"]
                        and (
                            t in req_tasks
                            or t in fam_cov
                            or not (req["tasks"][t].get("execution") or {}).get("linear_precision")
                        )
                    }
                    has_current_source_coverage = any(src == req["source"]["digest"] for src, _ in coverage_fps)
                    all_runnable_matched = runnable_count > 0 and runnable_matched_count == runnable_count
                    all_t1_in_coverage = (
                        (
                            bool(matching)
                            or has_current_source_coverage
                            or (all_t1_preflight_blocked and all_runnable_matched)
                        )
                        and not any_t1_row_rejected
                        and all(
                            t in measured or t in fam_cov
                            for t in t1_recorded_set
                        )
                        and all(
                            (
                                task_execution_fingerprint(req["tasks"][t]),
                                task_evaluation_fingerprint(req["tasks"][t]),
                            )
                            in coverage_fps.get((req["source"]["digest"], t), set())
                            for t in t1_recorded_set
                            if (req["source"]["digest"], t) in coverage_fps
                        )
                    )
                    if all_t1_in_coverage:
                        for cov_tid, cov_st in fam_cov.items():
                            if cov_tid not in t1_task_set:
                                all_measured.setdefault(cov_tid, cov_st)
                            elif cov_tid not in measured:
                                measured[cov_tid] = cov_st
                                all_measured[cov_tid] = cov_st
                        # Coverage may supply Tier 1 measurements even when a
                        # matching readout contains only Tier 2. Cite coverage,
                        # never an unrelated readout that contributed no row.
                        if not used_records and fam_cov and measured:
                            used_records.append(("reports/forge/attempt-coverage.json", "attempt-coverage.json"))
                    if used_records:
                        any_matched = True
                    if not all_t1_preflight_blocked:
                        runnable_count += 1
                        if matching and not any_t1_row_rejected and used_records:
                            runnable_matched_count += 1

                    def _dep_blocked(task_id: str, seen: frozenset[str] = frozenset()) -> bool:
                        if task_id in seen:
                            return False
                        next_seen = seen | {task_id}
                        for dep in req["tasks"].get(task_id, {}).get("dependencies", []):
                            dep_task = dep["task"] if isinstance(dep, dict) else str(dep)
                            if req["tasks"].get(dep_task, {}).get("preflight_blockers"):
                                return True
                            if measured.get(dep_task) in {"FAIL", "BLOCKED"}:
                                return True
                            if _dep_blocked(dep_task, next_seen):
                                return True
                        return False

                    def _outcomes(task_ids: list[str]) -> list[str]:
                        out = []
                        for tid in task_ids:
                            if all_t1_in_coverage and measured.get(tid) in {"PASS", "FAIL"}:
                                out.append(measured[tid])
                            elif req["tasks"][tid].get("preflight_blockers"):
                                out.append("PREFLIGHT_BLOCKED")
                            elif tid in measured:
                                out.append(measured[tid])
                            elif _dep_blocked(tid):
                                out.append("BLOCKED")
                            elif all_t1_in_coverage and (
                                req["tasks"].get(tid, {}).get("execution") or {}
                            ).get("linear_precision"):
                                out.append("BLOCKED")
                            else:
                                out.append("UNMEASURED")
                        return out

                    req_outcomes = _outcomes(req_tasks)
                    diag_outcomes = _outcomes(diag_tasks)
                    rows_data.append(
                        {
                            "family_id": fam_id,
                            "family_label": fam_label,
                            "family_page": fam_page,
                            "candidate_id": cid,
                            "card_rel": card_rel,
                            "source_digest": req["source"]["digest"],
                            "candidate_revision": rev,
                            "req_outcomes": req_outcomes,
                            "diag_outcomes": diag_outcomes,
                            "req_pass": sum(item == "PASS" for item in req_outcomes),
                            "total_pass": sum(item == "PASS" for item in req_outcomes + diag_outcomes),
                            "all_preflight_blocked": all(
                                item == "PREFLIGHT_BLOCKED" for item in req_outcomes + diag_outcomes
                            ),
                            "used_records": used_records,
                            "measured": dict(all_measured),
                        }
                    )
            if any_matched and len(source_digests) == 1:
                rows_data.sort(key=lambda item: (-item["total_pass"], -item["req_pass"], item["family_label"]))
                families = {
                    item["family_id"]: {
                        "candidate_id": item["candidate_id"],
                        "candidate_revision": item["candidate_revision"],
                        "source_digest": item["source_digest"],
                        "records": {
                            tid: {"status": st}
                            for tid, st in item["measured"].items()
                            if st
                        },
                    }
                    for item in rows_data
                }
                payload = {
                    "scope": "current_source_formulaic_display",
                    "qualification_input": False,
                    "qualification_reuse": False,
                    "source_digest": next(iter(source_digests)),
                    "view_revision": view_revision,
                    "req_total": req_total,
                    "diag_total": diag_total,
                    "stretch_total": sum(item.get("importance") in {"stretch", "held"} for item in t1_assignments),
                    "rows": rows_data,
                    "families": families,
                }
        except Exception:
            payload = None
        if len(_CURRENT_SOURCE_TIER1_CACHE) >= 8:
            _CURRENT_SOURCE_TIER1_CACHE.pop(next(iter(_CURRENT_SOURCE_TIER1_CACHE)))
        _CURRENT_SOURCE_TIER1_CACHE[cache_key] = payload

    return _CURRENT_SOURCE_TIER1_CACHE.get(cache_key)


def _recorded_source_binary_projection(result, root: Path) -> dict | None:
    """Retain a complete, source-bound display matrix without qualifying new code."""
    from experiments.forge.family_reports import _selected_cohorts, _selected_row, full_suite_tasks

    progress = result.get("family_progress") or {}
    relative = "reports/forge/attempt-coverage.json"
    expected = progress.get("input_hashes", {}).get(relative)
    path = Path(root) / relative
    if not expected or not path.is_file():
        return None
    if file_hash(path) != expected:
        raise ValueError("recorded formulaic display coverage digest changed")
    coverage = read_json(path)
    if (coverage.get("schema") != "forge_attempt_coverage_v1"
            or coverage.get("qualification_input") is not False
            or coverage.get("qualification_reuse") is not False
            or coverage.get("formulations_modified") is not False):
        raise ValueError("recorded formulaic display requires diagnostic-only coverage")
    required = set(full_suite_tasks(progress))
    scope = progress.get("recorded_display_scope")
    if scope is not None:
        text = scope.get("text")
        if (not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != scope.get("sha256")
                or progress.get("input_hashes", {}).get(scope.get("path")) != scope.get("sha256")):
            raise ValueError("recorded display scope digest changed")
        declaration = json.loads(text)
        if declaration.get("schema") != "forge_fixed_test_list_v1" or not declaration.get("tests"):
            raise ValueError("invalid recorded display scope")
        required = set(declaration["tests"])
    if not required:
        return None
    retained_families = {
        record.get("family")
        for record in coverage.get("records", [])
        if not any(record.get(key) for key in _SCOPED_COVERAGE_KEYS)
        and isinstance(record.get("family"), str)
    }
    families = {}
    for family, cohort in _selected_cohorts(progress):
        if retained_families and family["id"] not in retained_families:
            continue
        row = _selected_row(result, cohort)
        candidate, revision = row.get("candidate_id"), row.get("candidate_revision")
        source = row.get("bindings", {}).get("source_digest")
        if not candidate or not _is_sha256_hex(revision) or not _is_sha256_hex(source):
            return None
        records = {}
        for record in coverage.get("records", []):
            if (record.get("family") != family["id"]
                    or record.get("candidate_id") != candidate
                    or record.get("candidate_revision") != revision
                    or record.get("source_digest") != source
                    or any(record.get(key) for key in _SCOPED_COVERAGE_KEYS)):
                continue
            task = record.get("task")
            identity = [source, revision, task, record.get("execution_sha256"), record.get("evaluation_sha256")]
            if (not task or not all(_is_sha256_hex(value) for value in identity[3:])
                    or record.get("key") != stable_hash(identity)):
                raise ValueError("recorded formulaic display has an invalid attempt binding")
            if task in records and records[task] != record:
                raise ValueError("recorded formulaic display has conflicting attempts")
            if record.get("attempted") is True and record.get("status") in {"PASS", "FAIL"}:
                records[task] = record
        is_archived_pre_precision = (
            cohort.get("tasks", {}).get("precision_gaussian_nvfp4", {}).get("status", "UNKNOWN") == "UNKNOWN"
        )
        if not is_archived_pre_precision:
            continue
        if not required.issubset(records):
            return None
        families[family["id"]] = {"candidate_id": candidate, "candidate_revision": revision,
                                   "source_digest": source, "records": records}
    if not families:
        return None
    return {"scope": "recorded_source_formulaic_display", "qualification_input": False,
            "qualification_reuse": False, "coverage": {"path": relative, "sha256": expected},
            "families": families}


def _formulaic_display_payload(result, root: Path) -> dict | None:
    current = _concluded_current_source_tier1_payload(result, root)
    if current and all("UNMEASURED" not in (item["req_outcomes"] + item["diag_outcomes"])
                       for item in current["rows"]):
        return current
    recorded = _recorded_source_binary_projection(result, root)
    return recorded


def _selected_recipe_binary_task_statuses(result, root: Path) -> dict[str, dict[str, str]] | None:
    """Compatibility name: exact selected-cohort statuses, never binary imputation.

    Diagnostic coverage remains separately available from its original receipts;
    neither matching family labels nor policy-parent aliases transfer a grade.
    """
    from experiments.forge.family_reports import _selected_cohorts
    progress = result.get("family_progress") or {}
    families = {family["id"]: {name: entry.get("status", "UNKNOWN")
                               for name, entry in cohort.get("tasks", {}).items()}
                for family, cohort in _selected_cohorts(progress)}
    return families or None


def _current_source_tier1_readouts_markdown(result, root, path) -> str:
    root = Path(root).resolve()
    path = Path(path).resolve()
    cached = _concluded_current_source_tier1_payload(result, root)
    if not cached:
        return ""
    from experiments.forge.family_reports import link

    src12 = cached["source_digest"][:12]
    req_total = cached["req_total"]
    diag_total = cached["diag_total"]
    other_label = "Stretch / diagnostic" if cached.get("stretch_total", 0) else "Diagnostic"
    all_total = req_total + diag_total
    lines = [
        "### Current-source selected-recipe Tier 1 readouts",
        "",
        (
            f"Matched current-source (`{src12}`, `discriminator_stability` revision {cached['view_revision']}) "
            f"ordinary Tier 1 readouts for each family's unchanged selected recipe under protocol seed 0 on "
            f"`NVIDIA RTX A6000` ({req_total} required + {diag_total} {other_label.lower()} = {all_total} Tier 1 tests). "
            "`## Selected recipes` above retains historical recorded-source pins until a new family selection "
            "snapshot is registered; these concluded `reports/forge/records/readout-*.json` records show "
            "matched current-source Tier 1 performance side by side without regrading historical cohorts."
        ),
        "",
        (
            f"| Family | Selected recipe | Source / revision | Required Tier 1 ({req_total}) "
            f"| {other_label} Tier 1 ({diag_total}) | Total Tier 1 ({all_total}) | Evidence |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in cached["rows"]:
        family_cell = link(root, path, item["family_label"], item["family_page"])
        card_href = os.path.relpath(root / item["card_rel"], path.parent)
        recipe_cell = f"[`{item['candidate_id']}`]({card_href})"
        rev12 = item["candidate_revision"][:12]
        src_rev_cell = f"`{item['source_digest'][:12]}` / `{rev12}`"
        req_cell = _format_tier1_bucket(item["req_outcomes"], req_total)
        diag_cell = _format_tier1_bucket(item["diag_outcomes"], diag_total)
        tot_cell = _format_tier1_bucket(
            item["req_outcomes"] + item["diag_outcomes"], all_total, bold_score=True
        )
        if item["used_records"]:
            ev_links = [
                f"[`{rec_name}`]({os.path.relpath(root / rec_rel, path.parent)})"
                for rec_rel, rec_name in item["used_records"]
            ]
            ev_cell = " · ".join(ev_links)
        elif item["all_preflight_blocked"]:
            ev_cell = f"[`{Path(item['card_rel']).name}`]({card_href}) (preflight-blocked)"
        else:
            ev_cell = f"[`{Path(item['card_rel']).name}`]({card_href}) (not run)"
        lines.append(
            f"| {family_cell} | {recipe_cell} | {src_rev_cell} | {req_cell} | {diag_cell} | {tot_cell} | {ev_cell} |"
        )
    return "\n".join(lines)


def _formulaic_selected_recipe_tables(markdown: str, result, root, path) -> str:
    # Diagnostic coverage cannot replace a selected cohort's recorded grades.
    # The family renderer already preserves scientific and operational statuses.
    return markdown


def assignment_is_req(assignment: dict, tier_key: str) -> bool:
    return assignment.get("importance") == "required" and str(assignment.get("qualification_tier")) == tier_key


def _current_source_tier1_readouts_section(markdown: str, result, root, path) -> str:
    heading = "\n## Research recipes and separate diagnostics\n"
    if heading not in markdown or "### Current-source selected-recipe Tier 1 readouts" in markdown:
        return markdown
    block = _current_source_tier1_readouts_markdown(result, root, path)
    if not block:
        return markdown
    before, after = markdown.split(heading, 1)
    next_h2 = after.find("\n## ")
    details_pos = after.find("\n<details>\n")
    if details_pos >= 0 and (next_h2 < 0 or details_pos < next_h2):
        insert_pos = details_pos
    elif next_h2 >= 0:
        insert_pos = next_h2
    else:
        insert_pos = len(after)
    return before + heading + after[:insert_pos].rstrip("\n") + "\n\n" + block + "\n" + after[insert_pos:]


def _single_command_refresh_docs(markdown: str) -> str:
    target = (
        "## Refresh\n\n"
        "```sh\n"
        "python reports/forge/regenerate_technique_inventory.py\n"
        "```\n\n"
        "This regenerates the leaderboard,"
    )
    if target not in markdown or "python ops/refresh_forge_readouts.py" in markdown:
        return markdown
    replacement = (
        "## Refresh\n\n"
        "```sh\n"
        "python ops/refresh_forge_readouts.py --check\n"
        "python ops/refresh_forge_readouts.py\n"
        "python ops/refresh_forge_readouts.py --reports-only\n"
        "python reports/forge/regenerate_technique_inventory.py\n"
        "```\n\n"
        "`ops/refresh_forge_readouts.py` checks or refreshes all out-of-date Forge stages end to end when no "
        "code changes are needed (auto-selecting an uncontended GPU for the 48-cell DCGAN scaling table via "
        "`--scaling-gpu auto`, draining diagnostic attempt coverage and current-source selected-family readouts, "
        "and sealing clean-checkout-safe compiled reports; `--reports-only` skips GPU launches).\n\n"
        "`python reports/forge/regenerate_technique_inventory.py` regenerates the leaderboard,"
    )
    return markdown.replace(target, replacement, 1)


def _current_markdown(result, root, path):
    if result.get("family_progress"):
        from experiments.forge.family_reports import render_leaderboard
        projected = _project_publication_for_family_pages(root, result)
        markdown = _stale_measurement_labels(render_leaderboard(root, projected, path), result, root, path)
        markdown = _selected_diagnostic_rows(markdown, result, root, path)
        markdown = _current_source_tier1_readouts_section(markdown, result, root, path)
        markdown = _formulaic_selected_recipe_tables(markdown, result, root, path)
        markdown = _single_command_refresh_docs(markdown)
        matrix_link = os.path.relpath(root / "reports/forge/test-performance.md", path.parent)
        markdown += ("\n[Performance by test and complete recipe](" + matrix_link + "): "
                     "compatible passes, failures, execution errors and missing witnesses, with sortable evidence drill-down.\n")
        return markdown
    if not result.get('recorded_policy') and result['view'] == 'discriminator_stability':
        return _common26_current_markdown(result, root, path)
    def cell(value):
        return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")
    tiers = list(result["tier_requirements"])
    recorded_policy = result.get("recorded_policy")
    ordinary_current = not recorded_policy and result["view"] == "discriminator_stability"
    title = "Recorded" if recorded_policy else "Current"
    lines = ["# Current model/configuration scores" if ordinary_current else
             f"# {title} Forge trainer-family leaderboard", ""]
    if recorded_policy:
        denominators = ", ".join(f"Tier {tier}: {len(result['tier_requirements'][tier])}" for tier in tiers)
        policy_link = os.path.relpath(root / recorded_policy, path.parent)
        lines += [f"This table preserves [{result['view']} revision {result['view_revision']}]({policy_link}) "
                  f"with recorded required denominators **{denominators}**. "
                  "Current task placement is listed in [experiments by tier](EXPERIMENTS_BY_TIER.md). "
                  "The recorded outcomes supply no qualification for a later view revision.", ""]
    if result.get("completed_api_studies") and not ordinary_current:
        lines += ["Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has "
                  "**2/2 hold FAIL**, with six other domains UNKNOWN per family. "
                  "See [completed source-bound studies](#completed-source-bound-studies) for the exact "
                  "protocols and original goal GIFs. These separate results do not fill the ordinary "
                  "qualification cells below.", ""]
    progress = result.get("atlas_unblocking_progress") if ordinary_current else None
    half_base = result.get("word_half_base_diagnostic") if ordinary_current else None
    original_pr223 = result.get("original_pr223_atlas") if ordinary_current else None
    declared = result.get("declared_view", {})
    if ordinary_current and declared and declared["policy_fingerprint"] != result["policy_fingerprint"]:
        recorded_totals = "/".join(str(len(result["tier_requirements"][tier])) for tier in tiers)
        declared_totals = "/".join(str(len(declared["tier_requirements"].get(tier, []))) for tier in tiers)
        additions = [task for names in declared["added_required_tasks"].values() for task in names]
        lines += [f"The declared view is revision **{declared['revision']} ({declared_totals})**. "
                  f"Ordinary scores below retain their measured revision **{result['view_revision']} ({recorded_totals})**. "
                  "Additional required cells are **UNKNOWN** for every ordinary family: " +
                  ", ".join(f"`{task}`" for task in additions) + ". "
                  "Standalone API results keep their own source, recipe and runtime; they do not fill these cells.", ""]
    if ordinary_current:
        lines += ["Each row keeps its own configuration, source, representation and measured scope. "
                  "Required denominators stay fixed; diagnostic and ordinary qualification scores remain separate.", "",
                  "| Model/configuration | Actual representation | Measured score/status and scope |",
                  "| --- | --- | --- |"]
        if original_pr223:
            lines.append(_original_pr223_score_line(original_pr223, root, path))
        if progress:
            lines += _atlas_progress_markdown(progress, root, path, half_base=half_base)
        elif half_base:
            lines.append(_half_base_score_line(half_base, root, path))
        for score in result.get("standalone_api_scores", []):
            row, case = score["run"], score["case"]
            readout = os.path.relpath(root / score["readout"], path.parent)
            gif = os.path.relpath(root / score["gif"], path.parent)
            metrics = row["final_metrics"]
            terminal = row["terminal_metrics"]
            passing = sum(observation["passed"] for observation in terminal)
            bound = next((value for name, op, value in case["thresholds"]
                          if name == "cdf_ks" and op == "<="), None)
            fidelity = (f" · KS {metrics['cdf_ks']:.5f} / ≤ {bound}"
                        if "cdf_ks" in metrics and bound is not None else "")
            label = f"{score['trainer_family'].upper()} · {case['title']}"
            status = row["verdict"]
            prior = score["prior"]
            representation = f"MoG (fixed σ {prior['sigma']})" if prior["kind"] == "mog" else "Particles"
            lines.append(f"| [{cell(label)}]({readout}) | {representation} | "
                         f"**{int(status == 'PASS')}/1 PASS · {status} 1** · standalone API · "
                         f"{row['runtime']['device']} · {row['completed_updates']:,} updates · "
                         f"terminal {passing}/{len(terminal)}{fidelity}<br>"
                         f"[Actual-training histogram GIF]({gif}) · no ordinary tier credit |")
    else:
        lines += ["Each cell retains **passes / full required total** or **status (N required)** from one selected configuration. "
             "Mixed cells show the counts of failed, blocked and unmeasured tasks. " +
             ("Each trainer family and runtime has one archived row; its alternatives remain recorded separately. "
              if recorded_policy else "Each formulation family has one current selected row; source and runtime alternatives remain unranked. ") +
             "Expand the configuration details below for selection, provenance, other outcomes and cost.", "",
             "| Trainer family / runtime | " +
             " | ".join(f"Tier {tier}" for tier in tiers) + " | Recorded tier |",
                 "| --- | " + " | ".join("---:" for _ in tiers) + " | ---: |"]
    details = ["<details>", "<summary>Selected configurations and provenance</summary>", ""]
    for row in result["rows"]:
        # The actual C6 Atlas result occupies the displayed Atlas row. Its
        # incompatible ordinary reference is retained verbatim in the JSON.
        if ordinary_current and progress and row.get("trainer_family", row["candidate_id"]) == "atlas":
            continue
        source = row.get("bindings", {}).get("source_digest")
        pointer = result["evidence_sources"].get(row.get("publication_key"))
        if pointer and source:
            source_link = f"[`{source[:12]}`]({os.path.relpath(root / pointer['snapshot'], path.parent)})"
        else:
            source_link = f"`{source[:12]}` (unmeasured)" if source else "unresolved"
        runtime = row.get("runtime_cohort", {})
        models = sorted({p.get("model") for p in runtime.get("compute_profiles", {}).values() if p and p.get("model")})
        compute = runtime.get("execution_backend", "unrecorded") + (" / " + ", ".join(models) if models else "")
        counts = Counter(task["status"] for task in row.get("tasks", []) if task["status"] != "PASS")
        others = ", ".join(f"{status} {count}" for status, count in sorted(counts.items())) or "all required tasks PASS"
        seconds = row.get("cost", {}).get("wall_seconds")
        revision, cohort = (str(row.get(key) or "unresolved")[:12] for key in ("candidate_revision", "cohort"))
        selection = row.get("selection", {})
        selection_label = selection.get("selection_kind", "canonical_fallback").replace("_", " ")
        if selection.get("selection_kind") == "qualified_winner":
            selection_label += f" (tuning through tier {selection['tuning_through_tier']})"
        if not selection.get("qualified", False):
            selection_label += "; no qualified winner"
        name = row["candidate_id"]
        card = root / "configs/forge/configurations" / (name + ".json")
        if not card.is_file():
            card = root / "configs/forge/ideas" / (name + ".json")
        # Keep content-addressed IDs in the link target, with a readable label.
        label = name.rsplit("--", 1)
        configuration_label = " · ".join([label[0], label[1][:12]]) if len(label) == 2 else name
        configuration_link = f"[{configuration_label}]({os.path.relpath(card, path.parent)})"
        backend = runtime.get("execution_backend", "unrecorded")
        baseline_links = _separate_baseline_links(result, row, root, path)
        family_label = "<br>".join([f"{row['technique']}", backend, *baseline_links])
        values = [family_label,
                  *[_current_tier_cell(row, tier, result["tier_requirements"][tier],
                                      path.with_suffix('.json').name) for tier in tiers],
                  row["qualified_tier"]]
        if ordinary_current:
            json_link = os.path.relpath(root / CURRENT_PREFIX.with_suffix(".json"), path.parent)
            scores = "<br>".join(f"Tier {tier}: " + _current_tier_cell(
                row, tier, result["tier_requirements"][tier], json_link) for tier in tiers)
            unresolved = f"<br>Revision/cohort: {revision} / {cohort}" if not row.get("candidate_revision") or not row.get("cohort") else ""
            scope = ("recorded source measurement · stale; no current qualification"
                     if selection.get("freshness") == "stale" else "ordinary qualification")
            values = [f"{row['technique']} · {configuration_link}<br>{backend} · {scope}",
                      _ordinary_representation(result, row),
                      scores + f"<br>Recorded tier: {row['qualified_tier']} · [source]({json_link})" + unresolved]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        details += [f"### {cell(row['technique'])} ({cell(backend)})", "",
                    f"- **Selected configuration:** {configuration_link}",
                    f"- **Selection:** {cell(selection_label)}",
                    f"- **Evidence source:** {source_link}",
                    f"- **Exact revision / cohort:** {revision} / {cohort}",
                    f"- **Compute:** {cell(compute)}",
                    f"- **Other outcomes:** {cell(others)}",
                    f"- **Paid seconds:** {round(seconds, 3) if seconds is not None else 'unknown'}", ""]
        if baseline_links:
            details += ["- **Separate baseline evidence:** " + " · ".join(baseline_links) +
                        "; no current tier credit.", ""]
    lines.append("")
    if progress:
        lines += _policy_score_scope(root, path, half_base=half_base)
    if ordinary_current:
        required = declared.get("tier_requirements", result["tier_requirements"])
        required_totals = "/".join(str(len(required.get(tier, []))) for tier in tiers)
        required_count = sum(len(names) for names in required.values())
        lines += ["## Qualification and scope", "",
                  "MoG + particles (per task) denotes separate declared host laws within a suite; "
                  "it does not assert a hybrid model. The release-0.7 cloud-labelled row retains its "
                  "archived MoG declaration. BLOCKED rows show requested representations, not successful execution.", "",
                  f"Before shipping defaults, one unchanged family/configuration tuple must satisfy all {required_count} required "
                  f"{required_totals} gates with compatible source, runtime and serving evidence, followed by the separate "
                  "calibration and robustness requirements. Named diagnostic rows do not pool passing cells "
                  "across families or sources and grant no ordinary tier, default or speed credit.", ""]
        if result.get("completed_api_studies"):
            lines += ["Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has "
                      "**2/2 hold FAIL**, with six other domains UNKNOWN per family. "
                      "See [completed source-bound studies](#completed-source-bound-studies) for the exact "
                      "protocols and original goal GIFs. These separate results do not fill ordinary qualification.", ""]
    if not ordinary_current:
        lines += details + ["</details>", ""]
    lines += ["Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws "
              "and hardware. They do not pool qualification across sources or qualify the latest checkout. "
              "Selection never combines passing tasks or tiers from different configurations. A failed best-observed "
              "configuration is not a qualified winner. Search qualification covers only its declared tuning tiers; "
              "later-tier outcomes are reported separately. The screening profile remains provisional and does not "
              "confer calibrated robustness or public-default adoption.", "",
              "BLOCKED means execution was incompatible or a prerequisite was unavailable. "
              "NOT RUN and UNKNOWN mean unexecuted or unmeasured; FAIL records an executed gate failure. "
              "Failed or blocked prerequisites stop later work; required denominators stay fixed. "
              "The Atlas-history and C6-hold links retain their separate protocols and supply no current tier credit.", "",
              f"[All configuration alternatives, trials, task statuses and exact bindings]({path.with_suffix('.json').name}) · "
              f"[Evidence and archived publication identities]({os.path.relpath(root / EVIDENCE_MANIFEST, path.parent)})", "",
              "Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:", "",
              "```sh", "python reports/forge/regenerate_technique_inventory.py" +
              (" --recorded-policy " + shlex.quote(recorded_policy) if recorded_policy else
               " --refresh-publication" if result.get("publication_refresh") else ""), "```", "",
              ("This command uses the exact archived policy and registered snapshots; it does not resolve new declarations. "
               "New evidence for a later view revision requires its own compatible evidence registration."
               if recorded_policy else
               "For ordinary Forge qualification, use `--source-commit <executed-commit>` to independently regrade "
               "hydrated original receipts and update this leaderboard. Reusable candidates use the same complete task ladder. "
               "Historical task-only diagnostics remain motivation and reproduction evidence. Source snapshots are provenance, not additional leaderboards."), "",
              f"Publication input digest `{result['provenance']['input_digest']}`.", ""]
    if not recorded_policy and result["view"] == "discriminator_stability":
        lines += ["The [whole-family repair readout](family-wide-word-repairs/README.md) records the ordinary "
                  "candidate attempts and bounded global configuration search. Their complete rows remain "
                  "unranked alternatives below and in the companion JSON; a failed replacement does not "
                  "make its historical incumbent a qualified standard.", ""]
    if result.get("archived_policies"):
        lines += ["Earlier view policies retain their exact numerical snapshots and receipt proofs in the companion JSON. "
                  "Their outcomes do not fill current requirements:", ""]
        for policy in result["archived_policies"]:
            totals = "/".join(str(len(policy["tier_requirements"][tier])) for tier in tiers)
            lines.append(f"- `{policy['view']}` revision {policy['view_revision']}: recorded denominators {totals}; "
                         f"{len(policy['cohorts'])} source cohorts.")
        lines.append("")
    archived = [row for row in result.get("configuration_rows", []) if row.get("alternative_scope") == "archived_alternative"]
    if archived:
        lines += ["Unranked alternatives retain their original outcomes and exact source/runtime bindings:", ""]
        for row in archived:
            source = row.get("bindings", {}).get("source_digest", "unresolved")
            lines.append(f"- `{row['candidate_id']}` ({row['trainer_family']}), source `{source[:12]}`; "
                         f"recorded tier {row.get('qualified_tier', 0)}. Full evidence is in the companion JSON.")
        lines.append("")
    if result.get("completed_api_studies"):
        if ordinary_current:
            lines += ["## Completed source-bound studies", "",
                      "Historical protocols and distinct contrasts retain their exact scores, source and GIFs "
                      "in their original readouts. They do not fill cells in the table above.", ""]
            for study in result["completed_api_studies"]["rows"]:
                link = os.path.relpath(root / study["readout"], path.parent)
                lines.append(f"- [{study['label']}]({link}) · own {study['required_cells']}-cell scope.")
            lines.append("")
        else:
            from experiments.forge.completed_studies import render_completed_studies
            lines += ["", render_completed_studies(result["completed_api_studies"], root, path), ""]
    if result.get("baseline_debugging"):
        link = os.path.relpath(root / result["baseline_debugging"]["readout"], path.parent)
        lines += [f"[Exact C6 baseline and retained persistence diagnosis]({link}): "
                  "two original smoke passes per family, six later domains UNKNOWN; "
                  "the added broad hold fails on projected-CDF shape excursions. "
                  "The source-bound diagnostic preserves all original gates and supplies no default or speed credit.", ""]
    return "\n".join(lines)


def _shared_score_archive(root):
    """Move the original historical suffix intact out of the current chart."""
    path = root / "reports/forge/shared-score-index-20261003/README.md"
    if not path.is_file():
        return
    archive = path.with_name("ARCHIVED_REPORTS.md")
    original = path.read_bytes()
    marker = b"## Completed current Atlas GPU diagnostic\n"
    if marker in original:
        suffix = marker + original.split(marker, 1)[1]
        if archive.exists() and archive.read_bytes() != suffix:
            raise ValueError("Shared score index historical archive differs from its preserved report boundary")
    elif b"[Historical report archive](ARCHIVED_REPORTS.md)" in original and archive.is_file():
        suffix = archive.read_bytes()
    else:
        raise ValueError("Shared score index is missing its preserved report boundary")
    return archive, suffix.decode("utf-8")


def _shared_score_intro(root, result):
    """Render the one current table and link the separate unchanged archive."""
    path = root / "reports/forge/shared-score-index-20261003/README.md"
    if not path.is_file():
        return
    _shared_score_archive(root)  # Validate the legacy or already-moved boundary.
    leading = ("# Shared ParticleGAN score index\n\n"
               "The [current family leaderboard](../technique-inventory.md) contains the generated "
               "family/view totals and links to each family's experiments, metrics and pass criteria. "
               "Refresh it with `python reports/forge/regenerate_technique_inventory.py`.\n\n")
    leading += ("The [performance-by-test matrix](../test-performance.md) shows complete recipe cells, "
                "zero-pass tests, execution errors and separately typed feasibility witnesses.\n\n")
    from experiments.forge.test_performance_render import render_discovery_navigation
    leading += render_discovery_navigation(root, path)
    half_base = result.get("word_half_base_diagnostic")
    if half_base:
        accounting = half_base["accounting"]
        account_text = (
            f"The named campaign's predecessor-inclusive charge is **{accounting['inclusive_charged_seconds']} / 10500 seconds**, "
            f"reserve zero: GPU0 **{accounting['gpu0_charged_seconds']} / 7500**, "
            f"GPU1 **{accounting['gpu1_charged_seconds']} / 3000**. "
            f"The separate half_base attempt charged **{half_base['cost']['paid_seconds']} / 900 seconds**, "
            "with zero reserve and overrun. Prior charges are included once; the original C6 word INVALID is unchanged.\n\n")
    else:
        account_text = ("The named campaign's predecessor-inclusive charge remains **910.2391431590077 / 10500 seconds**, "
                        "reserve zero: GPU0 **234.82608077581972 / 7500**, GPU1 **675.4130623831879 / 3000**. "
                        "The half-base contrast has no completed measurement in these tables.\n\n")
    scope = ("## Qualification and accounting\n\n"
             "Each model/configuration keeps its own source, law and declared denominator. "
             "Existing C6 diagnostics retain their 26-slot protocol; standalone API rows use their own task gates. "
             "MoG + particles (per task) is a mixture of separate suite hosts, not a hybrid-model claim. "
             "Named diagnostic passes do not fill ordinary cells or combine into a family score. "
             "INVALID supplies no accepted numerical grade; original N5 word execution remains BLOCKED. "
             "No family default or comparable speed winner is established.\n\n"
             + account_text +
             "[Historical report archive](ARCHIVED_REPORTS.md) · [Original snapshot index](index.json). "
             "The archived source, scores, costs and links retain their original bytes and separate scopes.\n")
    return path, leading + scope


def _standalone_api_scores(root):
    """Project source-pinned API receipts without training, rescoring or selection."""
    scores = []
    for path in sorted((root / "reports/forge/records").glob("*.json")):
        record = read_json(path)
        if record.get("evidence_scope") != "standalone_api" or record.get("goal") != "discriminator_stability":
            continue
        if record.get("qualification_input") is not False or record.get("qualification_reuse") is not False:
            raise ValueError("standalone API display cannot confer qualification")
        source = record["source"]
        publication = (root / source["path"]).resolve()
        if (not publication.is_relative_to(root / "reports/toy_audit/api_contract")
                or file_hash(publication) != source["sha256"]):
            raise ValueError("standalone API publication identity mismatch")
        report = read_json(publication)
        if report.get("schema") != "particlegan_api_toy_supplement_v1":
            raise ValueError("unsupported standalone API publication")
        cases = {case["id"]: case for case in report["cases"]}
        if len(cases) != len(report["cases"]) or len(report["runs"]) != 1:
            raise ValueError("standalone API display requires one exact run")
        row = report["runs"][0]
        case = cases[row["id"]]
        identity = report["source_identities"][row["source_identity"]]
        provenance = record["provenance"]
        if (row["qualification_input"] is not False
                or row["source_identity"] != provenance["source_identity"]
                or identity["commit"] != provenance["source_commit"]
                or stable_hash(row["recipe"]) != provenance["recipe_sha256"]
                or row["raw_receipt_sha256"] != provenance["raw_receipt_sha256"]
                or case["task_sha256"] != provenance["task_sha256"]
                or row["verdict"] != record["task_results"][0]["gate_status"]):
            raise ValueError("standalone API recipe/source/verdict binding mismatch")
        if row["verdict"] not in {"PASS", "FAIL"}:
            raise ValueError("standalone API display requires a recorded numerical verdict")
        prior = provenance["prior"]
        expected_kind = {"mog": "mog", "particle_cloud": "particles"}.get(prior["kind"])
        if (row["recipe"].get("prior_kind") != expected_kind
                or row["prior_options"].get("sigma") != prior["sigma"]
                or row["recipe"].get("standardize") != prior["standardize"]):
            raise ValueError("standalone API prior binding mismatch")
        gif = (publication.parent / row["gif"]).resolve()
        if (not gif.is_relative_to(root / "reports/toy_audit/api_contract")
                or file_hash(gif) != row["gif_sha256"]):
            raise ValueError("standalone API GIF identity mismatch")
        readout = publication.parent / "README.md"
        if not readout.is_file():
            raise ValueError("standalone API readout is missing")
        scores.append({"record": str(path.relative_to(root)), "record_sha256": file_hash(path),
                       "publication": str(publication.relative_to(root)), "publication_sha256": source["sha256"],
                       "readout": str(readout.relative_to(root)), "readout_sha256": file_hash(readout),
                       "gif": str(gif.relative_to(root)), "trainer_family": record["trainer_family"],
                       "prior": prior, "case": case, "run": row, "qualification_input": False})
    return scores


def _declaration_only_configuration(row):
    """Allow saved unmeasured projections without allowing a measured grade."""
    tasks = row.get("tasks", []) + row.get("nonrequired_tasks", [])
    statuses = {task.get("status") for task in tasks}
    unmeasured = {"UNKNOWN", "BLOCKED", "NOT_RUN"}
    return (not row.get("attempt_ids") and row.get("qualified_tier", 0) == 0
            and bool(statuses) and statuses <= unmeasured
            and all(task.get("gate_status", task.get("status")) in unmeasured for task in tasks)
            and row.get("status") not in {"PASS", "FAIL"}
            and not any(tier.get("passed", 0) for tier in row.get("tiers", {}).values())
            and not row.get("cost", {}).get("measured_tasks", 0)
            and row.get("cost", {}).get("wall_seconds") in {None, 0})


def _family_registry_structure(registry):
    """Editorial grouping is independent of immutable evidence selection."""
    value = deepcopy(registry)
    for key in ("families", "historical_families"):
        for family in value.get(key, []):
            family.pop("label", None)
            family.pop("tags", None)
            family.pop("reporting_family", None)
            family.pop("inventory_visible", None)
            family.pop("current_configuration_family", None)
    return value


def _previous_family_registry(root, result):
    """Recover pre-metadata publications from their exact, bound Git source."""
    previous = result.get("trainer_family_registry")
    if previous is not None:
        return previous
    from experiments.forge.trainer_families import REGISTRY
    try:
        source = subprocess.check_output(["git", "show", "HEAD:" + REGISTRY.as_posix()], cwd=root,
                                         stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if hashlib.sha256(source).hexdigest() != result["provenance"].get("trainer_family_registry_sha256"):
        return None
    return json.loads(source)


def _presentation_registry_compatible(previous, registry, result):
    """Allow new unmeasured families without changing an existing selection."""
    before, after = _family_registry_structure(previous), _family_registry_structure(registry)
    old_ids = {family["id"] for family in before.get("families", [])}
    additions = [family for family in after.get("families", []) if family["id"] not in old_ids]
    # No new current rows or evidence is admitted by an editorial refresh.
    # A family with an active search or an already displayed member requires
    # ordinary regeneration and its full selection checks.
    displayed = {row["candidate_id"] for key in (
        "rows", "configuration_rows", "evidence_rows", "historical_family_rows", "archived_evidence_rows")
        for row in result.get(key, [])}
    if any(family.get("active_search_by_backend") or displayed.intersection(
            [family["canonical_candidate"], *family.get("candidates", [])]) for family in additions):
        return False
    after["families"] = [family for family in after.get("families", []) if family["id"] in old_ids]
    return before == after


def _refresh_family_presentation(root, result, registry):
    from experiments.forge.trainer_families import load_families
    active = load_families(root)  # Retain existing registry structure validation.
    historical = {family["id"]: family for family in registry.get("historical_families", [])}
    for key, families in (("rows", active), ("historical_family_rows", historical)):
        for row in result.get(key, []):
            family = families.get(row.get("trainer_family"))
            if family is not None:
                row["technique"] = family["label"]
    for name in result.get("trainer_families", {}):
        if name in active:
            result["trainer_families"][name] = deepcopy(active[name])


def refresh_publication(root=REPOSITORY_ROOT, *, view_id="discriminator_stability"):
    """Refresh the same board while preserving every registered scientific row."""
    from experiments.forge.trainer_families import CURRENT_SELECTION, REGISTRY, scientific_row_hash
    from experiments.forge.views import load_view
    root = Path(root).resolve()
    manifest = read_json(root / EVIDENCE_MANIFEST)
    result = read_json(root / CURRENT_PREFIX.with_suffix(".json"))
    if result.get("view") != view_id or manifest.get("view") != view_id:
        raise ValueError("refresh must retain the registered publication view")
    copy = deepcopy(result)
    claimed = copy["provenance"].pop("input_digest", None)
    if claimed != stable_hash(copy):
        raise ValueError("current publication input digest mismatch")
    if result.get("publication_scope") != "current_technique_inventory" or result.get("recorded_policy"):
        raise ValueError("refresh requires the current registered publication")
    if any(result.get(key) != manifest[key] for key in POLICY_FIELDS):
        raise ValueError("current publication differs from its registered evidence policy")
    if result["provenance"]["evidence_manifest_sha256"] != stable_hash(manifest):
        raise ValueError("registered evidence changed; use ordinary regeneration")
    selection_hash = result["provenance"].get("family_current_selection_sha256")
    current_selection_hash = file_hash(root / CURRENT_SELECTION) if (root / CURRENT_SELECTION).is_file() else None
    registry_hash = file_hash(root / REGISTRY) if (root / REGISTRY).is_file() else None
    regroup = registry_hash != result["provenance"].get("trainer_family_registry_sha256")
    if current_selection_hash != selection_hash:
        raise ValueError("family selection changed; use ordinary regeneration")
    registered = set()
    for entry in manifest["cohorts"]:
        _, rows = _snapshot(root, entry, manifest)
        registered.update(scientific_row_hash(row) for row in rows.values())
    archived_reports = _archived_reports(root, manifest)
    archived_bases = set()
    if result.get("archived_base_presentation_only"):
        pointers = result["archived_base_presentation_only"]
        indices = {pointer["row_index"] for pointer in pointers}
        expected_leaders = max(7, len(read_json(root / CURRENT_SELECTION)["selections"])) if (root / CURRENT_SELECTION).is_file() else 7
        if len(indices) != len(pointers) or any(index < expected_leaders or index >= len(result["rows"]) for index in indices):
            raise ValueError("invalid archived solution presentation indices")
        structure = {"rows": [deepcopy(row) for index, row in enumerate(result["rows"]) if index not in indices]}
        if len(structure["rows"]) != expected_leaders:
            raise ValueError("archived bases cannot replace seven selected leaders")
        restored = _ownership_migration_helper().restore_archived_solution_rows(
            root, structure, read_json(root / CURRENT_SELECTION), archived_reports, validate_row=_validate_published_row)
        if restored != pointers or structure["rows"] != result["rows"]:
            raise ValueError("archived solution base differs from exact original presentation")
        archived_bases = {pointer["scientific_row_sha256"] for pointer in pointers}
    for key in ("rows", "evidence_rows"):
        permitted = registered | archived_bases if key == "rows" else registered
        if any(scientific_row_hash(row) not in permitted for row in result.get(key, [])):
            raise ValueError("current publication contains an unregistered scientific row")
    for row in result.get("configuration_rows", []):
        if scientific_row_hash(row) not in registered and not _declaration_only_configuration(row):
            raise ValueError("current publication contains an unregistered measured configuration")
    archived = set()
    for _, _, _, rows in archived_reports:
        archived.update(scientific_row_hash(row) for row in rows.values())
    for original in result.get("archived_evidence_rows", []):
        row = deepcopy(original)
        row.pop("evidence_policy", None)
        if scientific_row_hash(row) not in archived:
            raise ValueError("current publication contains an unregistered archived row")
    if any(scientific_row_hash(row) not in registered | archived for row in result.get("historical_family_rows", [])):
        raise ValueError("current publication contains an unregistered historical family row")
    declared = load_view(root, manifest["view"])
    if stable_hash(declared) != manifest["policy_fingerprint"]:
        frozen = root / "configs/forge/view-history" / f"{manifest['view']}-v{manifest['view_revision']}.json"
        if (declared["revision"] <= manifest["view_revision"]
                or not frozen.is_file() or stable_hash(read_json(frozen)) != manifest["policy_fingerprint"]):
            raise ValueError("declared-view refresh requires the exact archived publication policy")
    ordered = sorted(declared["assignments"], key=lambda row: row.get("order", 0))

    def declared_tier(tier, match):
        return [row for row in ordered if str(row["qualification_tier"]) == tier and match(row)]

    requirements = {tier: [row["task"] for row in declared_tier(tier, lambda row: row["importance"] == "required")]
                    for tier in result["tier_requirements"]}
    # Held tasks are measured and shown but not counted; a task removed from the
    # count stays assigned as diagnostic with a note naming who carries it.
    # no_longer_required_tasks: required by the recorded policy, not counted by
    # the declared view (held or removed from the count).
    from experiments.forge.views import is_stretch
    result["declared_view"] = {"revision": declared["revision"], "policy_fingerprint": stable_hash(declared),
                               "tier_requirements": requirements,
                               "added_required_tasks": {tier: [name for name in names if name not in manifest["tier_requirements"][tier]]
                                                        for tier, names in requirements.items()},
                               "no_longer_required_tasks": {tier: [name for name in names if name not in requirements.get(tier, [])]
                                                            for tier, names in manifest["tier_requirements"].items()},
                               "held_tasks": {tier: [row["task"] for row in declared_tier(tier, is_stretch)]
                                              for tier in result["tier_requirements"]},
                               "removed_from_count": {tier: [{"task": row["task"], "reason": row["removal"]["reason"],
                                                              "counted_by": row["removal"]["counted_by"]}
                                                             for row in declared_tier(tier, lambda row: "removal" in row)]
                                                      for tier in result["tier_requirements"]}}
    if regroup:
        registry = read_json(root / REGISTRY) if registry_hash is not None else None
        previous = _previous_family_registry(root, result)
        if (registry is None or previous is None
                or not _presentation_registry_compatible(previous, registry, result)):
            raise ValueError("trainer family selection structure changed; use ordinary regeneration")
        _refresh_family_presentation(root, result, registry)
        result["provenance"]["trainer_family_registry_sha256"] = registry_hash
        result["provenance"]["selected_rows_sha256"] = stable_hash(result["rows"])
    if registry_hash is not None:
        result["trainer_family_registry"] = read_json(root / REGISTRY)
    result["standalone_api_scores"] = _standalone_api_scores(root)
    from experiments.forge.model_toy_publication import load_publication as load_model_research
    model_research = load_model_research(root)
    if model_research is not None:
        result["model_api_research"] = model_research
    result["publication_refresh"] = {"scientific_rows_preserved": True, "qualification_regraded": False,
                                     "training_launched": False}
    if view_id == "discriminator_stability":
        from experiments.forge.family_reports import build_progress
        result["family_progress"] = build_progress(root, result)
    if result["view"] == "discriminator_stability":
        result["common26_display"] = _common26_display_projection(result, root)
        result["provenance"]["common26_comparison_projector_sha256"] = file_hash(
            Path(__file__).resolve().parents[2] / "experiments/forge/common26_comparison.py")
        audit = root / COMMON26_DISPLAY_AUDIT
        result["provenance"]["common26_display_audit_sha256"] = (
            file_hash(audit) if audit.is_file() else None)
    result["provenance"]["publication_reducer_sha256"] = file_hash(Path(__file__))
    from experiments.forge.scaling_publication import attach_performance
    attach_performance(root, result)
    result["provenance"].pop("input_digest", None)
    result = compact_current_publication(result)
    result["provenance"]["input_digest"] = stable_hash(result)
    json_path, markdown_path = root / CURRENT_PREFIX.with_suffix(".json"), root / CURRENT_PREFIX.with_suffix(".md")
    markdown = _current_markdown(result, root, markdown_path)
    pages = _family_pages(root, result)
    shared = _shared_score_intro(root, result) if result["view"] == "discriminator_stability" else None
    _write_changed(json_path, _json_text(result))
    _write_changed(markdown_path, markdown)
    _write_family_pages(root, pages)
    if shared:
        _write_changed(*shared)
    return {"report": str(markdown_path), "json": str(json_path), "rows": len(result["rows"]),
            "input_digest": result["provenance"]["input_digest"], "qualification_reuse": False,
            **result["publication_refresh"]}


def _project_publication_for_family_pages(root, publication):
    """Keep exact-cohort grades and annotate audited diagnostic observations."""
    from experiments.forge.publication_integrity import annotate_publication
    return annotate_publication(root, publication)


def _install_family_reports_generated_pages_hook() -> None:
    from experiments.forge import family_reports

    if getattr(family_reports.generated_pages, "_formulaic_projected_hook", False):
        return
    original_generated_pages = family_reports.generated_pages

    def _wrapped_generated_pages(root: Path, publication: dict) -> dict[Path, str]:
        projected = _project_publication_for_family_pages(Path(root), publication)
        return original_generated_pages(Path(root), projected)

    _wrapped_generated_pages._formulaic_projected_hook = True  # type: ignore[attr-defined]
    family_reports.generated_pages = _wrapped_generated_pages


_install_family_reports_generated_pages_hook()


def _family_pages(root, result):
    _install_family_reports_generated_pages_hook()
    from experiments.forge.test_performance import generated_reports
    test_pages = generated_reports(root, publication=result)
    if not result.get("family_progress"):
        return test_pages
    from experiments.forge.family_reports import generated_pages
    pages = generated_pages(root, result)
    pages.update(test_pages)
    from experiments.forge.tier_report import REPORT_PATH, build_report, render_markdown as render_tiers
    # Registered tier navigation is prepared against the new in-memory
    # publication, so validation precedes writes and its hashes are fresh.
    if (root / REPORT_PATH).is_file():
        report = build_report(root, publication=result)
        pages[root / REPORT_PATH] = render_tiers(report, root, root / REPORT_PATH)
    return pages


@contextmanager
def _pending_publication_inputs(root, pending_snapshot):
    """Render the staged immutable snapshot without writing it before validation."""
    if pending_snapshot is None:
        yield
        return
    from experiments.forge.test_performance import Inputs
    path, data = pending_snapshot
    path, root = Path(path).resolve(), Path(root).resolve()
    name = path.relative_to(root).as_posix()
    encoded = data.encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("pending immutable source snapshot conflicts with existing bytes")
    value, digest = json.loads(data), hashlib.sha256(encoded).hexdigest()
    original = Inputs.__init__
    def initialize(inputs, directory):
        original(inputs, directory)
        if inputs.root == root:
            inputs.cache[name] = deepcopy(value)
            inputs.hashes[name] = digest
    Inputs.__init__ = initialize
    try:
        yield
    finally:
        Inputs.__init__ = original


def _write_family_pages(root, pages):
    if not pages:
        return
    from experiments.forge.family_reports import FAMILY_DIRECTORY
    for path, content in pages.items():
        _write_changed(path, content)
    # Only this generator's pages are disposable; retained evidence is elsewhere.
    for path in (root / FAMILY_DIRECTORY).glob("*.md"):
        if path not in pages and path.read_text().startswith("<!-- Generated Forge family report -->"):
            path.unlink()


def _declaration_fallback_sources(current_pins):
    """A unanimous pinned source can bind an otherwise unmeasured display."""
    sources = {}
    for pin in current_pins.values():
        source = pin.get("source_digest")
        sources.setdefault(pin["execution_backend"], set()).add(source if isinstance(source, str) else None)
    preferred = {}
    for backend, digests in sources.items():
        if len(digests) != 1:
            continue
        digest = next(iter(digests))
        if isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef"):
            preferred[backend] = digest
    return preferred


def _declaration_source_preference(row, previous, preferred_source):
    """Resolve only source-bound declarations; executed evidence still ties."""
    from experiments.forge.trainer_families import _declaration_only_row
    if not preferred_source or not all(_declaration_only_row(item) for item in (row, previous)):
        return None
    current_matches = row.get("bindings", {}).get("source_digest") == preferred_source
    previous_matches = previous.get("bindings", {}).get("source_digest") == preferred_source
    if current_matches != previous_matches:
        return current_matches
    return None


def _recorded_measurement_pin(root, family_id, rows, pin, *, view_id, catalogs, previous, reports, guard,
                              live_error="current measurement requires current execution, evaluation and budget contracts"):
    """Retain a published whole measurement only under its original Git laws."""
    from experiments.forge.trainer_families import family_row_pin, scientific_row_hash
    from experiments.forge.views import load_view, load_tasks, task_execution_fingerprint, task_evaluation_fingerprint
    if pin["selection_kind"] != "current_measurement" or not previous:
        raise ValueError("stale measurement requires an already-published current_measurement pin")
    checked = deepcopy(previous)
    claimed = checked.get("provenance", {}).pop("input_digest", None)
    if (claimed != stable_hash(checked) or previous.get("publication_scope") != "current_technique_inventory"
            or previous.get("view") != view_id or previous.get("recorded_policy")):
        raise ValueError("stale measurement requires a verified previous current publication")
    def matches(row):
        return family_row_pin(row, selection_kind=pin["selection_kind"], reason=pin["reason"],
                              measurement_views=pin.get("measurement_views"),
                              measurement_tasks=pin.get("measurement_tasks")) == pin
    old = [row for row in previous["rows"] if matches(row)]
    live = [row for row in rows if matches(row)]
    if len(old) != 1 or len(live) != 1:
        raise ValueError("stale measurement must retain one exact already-published whole pin")
    old, selected = old[0], live[0]
    prior_selection = old.get("selection", {})
    if prior_selection.get("selection_kind") == "recorded_source_measurement":
        if prior_selection.get("original_pin") != pin:
            raise ValueError("recorded measurement original pin changed")
    elif (prior_selection.get("selection_kind") != "current_measurement"
          or prior_selection.get("reason") != pin["reason"]
          or prior_selection.get("measurement_views") != pin["measurement_views"]
          or prior_selection.get("measurement_tasks") != pin.get("measurement_tasks")):
        raise ValueError("stale measurement was not an originally selected current measurement")
    sources = [(entry, report, saved) for entry, report, saved in reports
               if entry["json_sha256"] == old.get("publication_key")]
    if len(sources) != 1:
        raise ValueError("stale measurement requires one unique registered source snapshot")
    entry, report, saved = sources[0]
    source_row = deepcopy(saved.get(pin["candidate_id"], {}))
    source_row["trainer_family"] = family_id
    origin = old.get("bindings", {}).get("source_origin_commit")
    if (not matches(source_row) or not isinstance(origin, str)
            or len(origin) != 40 or any(char not in "0123456789abcdef" for char in origin)
            or old["bindings"].get("recorded_source_origin_commits") != [origin]
            or entry["source_commit"] != origin
            or report.get("frozen_source", {}).get("commit") != origin
            or pin["source_digest"] not in report.get("frozen_source", {}).get("source_digests", [])):
        raise ValueError("stale measurement source identity is missing or ambiguous")
    _validate_published_row(root, report, source_row)
    receipts = []
    for attempt in source_row["attempt_ids"]:
        path = root / "reports/forge/technique-receipts" / (attempt + ".json")
        summary = read_json(path)
        if summary["provenance"].get("source_origin_commit") != origin:
            raise ValueError("stale measurement receipt has a different source origin")
        receipts.append({"attempt_id": attempt, "summary_sha256": file_hash(path),
                         "original_files": summary["provenance"]["original_files"]})
    with tempfile.TemporaryDirectory(prefix="forge-recorded-measurement-") as temporary:
        frozen = Path(temporary)
        prefixes = ("configs/forge/tasks/", "configs/forge/views/", "configs/forge/task-variants/")
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", origin, "--",
                                         *[name.rstrip("/") for name in prefixes]], cwd=root, text=True).splitlines()
        git_files = {}
        for name in names:
            if not name.startswith(prefixes) or ".." in Path(name).parts or Path(name).suffix != ".json":
                raise ValueError("unsafe recorded measurement declaration path")
            data = subprocess.check_output(["git", "show", origin + ":" + name], cwd=root)
            path = frozen / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            git_files[name] = hashlib.sha256(data).hexdigest()
        original_view = load_view(frozen, view_id)
        recorded_fingerprints = {report["policy_fingerprint"]}
        projection = report.get("comparison_policy_projection")
        if isinstance(projection, dict) and projection.get("actual_policy_fingerprint"):
            recorded_fingerprints.add(projection["actual_policy_fingerprint"])
        if stable_hash(original_view) not in recorded_fingerprints:
            raise ValueError("stale measurement recorded Git view differs from its publication")
        required = {assignment["task"] for name in pin["measurement_views"]
                    for assignment in load_view(frozen, name)["assignments"]
                    if assignment["importance"] == "required" and assignment["qualification_tier"] == 1}
        required.update(pin.get("measurement_tasks", []))
        live_required = {assignment["task"] for name in pin["measurement_views"]
                         for assignment in load_view(root, name)["assignments"]
                         if assignment["importance"] == "required" and assignment["qualification_tier"] == 1}
        live_required.update(pin.get("measurement_tasks", []))
        live_tasks, frozen_tasks = load_tasks(root), load_tasks(frozen)
        drift, contracts = [], {}
        for name in sorted(required):
            digest = selected["bindings"]["task_contracts"][name]
            contract = catalogs.get("task_contracts", {}).get(digest)
            if (not isinstance(contract, dict) or stable_hash(contract) != digest
                    or report.get("task_contracts", {}).get(digest) != contract):
                raise ValueError("stale measurement task catalog changed")
            task, original = live_tasks[name], frozen_tasks[name]
            if (task["resources"] != original["resources"]
                    or task["execution"].get("steps") != original["execution"].get("steps")):
                raise ValueError("stale measurement cannot retain a changed live budget")
            values = {"execution_sha256": task_execution_fingerprint(task),
                      "evaluation_sha256": task_evaluation_fingerprint(task),
                      "timeout_seconds": task["resources"]["timeout_seconds"]}
            differences = {key: {"recorded": contract.get(key), "live": value}
                           for key, value in values.items() if contract.get(key) != value}
            if differences:
                drift.append({"task": name, "fields": differences})
            contracts[name] = digest
        for name in sorted(live_required - required):
            drift.append({"task": name, "fields": {"view_requirement": {
                "recorded": "absent_or_nonrequired", "live": "required_tier_1"}}})
        if not drift:
            raise ValueError("recorded measurement fallback requires actual contract drift")
        _, metadata = guard(frozen, family_id, [selected], pin, view_id=view_id, catalogs=catalogs)
    proof = {"trainer_family": family_id, "original_pin": deepcopy(pin), "source_origin_commit": origin,
             "source_digest": pin["source_digest"], "source_snapshot": deepcopy(entry),
             "previous_publication_input_digest": claimed,
             "scientific_row_sha256": scientific_row_hash(selected), "task_contracts": contracts,
             "git_declarations_sha256": git_files, "original_receipts": receipts, "contract_drift": drift,
             "recorded_guard_pass": True, "live_guard_error": live_error}
    metadata.update(selection_kind="recorded_source_measurement", original_pin=deepcopy(pin), qualified=False,
                    measurement_complete=False, current_measurement_complete=False,
                    recorded_measurement_complete=True, current_contracts=False,
                    qualification_reuse=False, qualification_input=False, freshness="stale",
                    recorded_source_origin_commit=origin, contract_drift=drift,
                    reason="Recorded source-bound original measurement retained; stale against current task contracts. "
                           "No current measurement, qualification reuse or comparable live ranking. Original selection: " + pin["reason"])
    return selected, metadata, proof


@contextmanager
def _retain_recorded_measurements(root, reports):
    """Reporting scope only; the ordinary live family guard remains unchanged."""
    from experiments.forge import trainer_families
    path = root / CURRENT_PREFIX.with_suffix(".json")
    previous = read_json(path) if path.is_file() else None
    original, proofs = trainer_families._current_pin, []
    contract_error = "current measurement requires current execution, evaluation and budget contracts"
    tier1_error = "current measurement requires every required Tier 1 task in its measurement views to be PASS or FAIL"
    def published_pin_exists(pin):
        if not previous:
            return False
        for old_row in previous.get("rows", []):
            if trainer_families.family_row_pin(
                    old_row, selection_kind=pin["selection_kind"], reason=pin["reason"],
                    measurement_views=pin.get("measurement_views"),
                    measurement_tasks=pin.get("measurement_tasks")) != pin:
                continue
            prior_sel = old_row.get("selection", {})
            if prior_sel.get("selection_kind") == "recorded_source_measurement":
                if prior_sel.get("original_pin") == pin:
                    return True
            elif (prior_sel.get("selection_kind") == "current_measurement"
                  and prior_sel.get("reason") == pin["reason"]
                  and prior_sel.get("measurement_views") == pin["measurement_views"]
                  and prior_sel.get("measurement_tasks") == pin.get("measurement_tasks")):
                return True
        return False
    def current_pin(directory, family_id, rows, pin, *, view_id, catalogs):
        try:
            return original(directory, family_id, rows, pin, view_id=view_id, catalogs=catalogs)
        except ValueError as error:
            message = str(error)
            if (Path(directory).resolve() != root or pin["selection_kind"] != "current_measurement"
                    or message not in {contract_error, tier1_error}
                    or (message == tier1_error and not published_pin_exists(pin))):
                raise
            selected, metadata, proof = _recorded_measurement_pin(
                root, family_id, rows, pin, view_id=view_id, catalogs=catalogs,
                previous=previous, reports=reports, guard=original, live_error=message)
            proofs.append(proof)
            return selected, metadata
    trainer_families._current_pin = current_pin
    try:
        yield proofs
    finally:
        trainer_families._current_pin = original



def publish_current(root=REPOSITORY_ROOT, *, source_commit=None, view_id="discriminator_stability",
                    execution_backend=None, recorded_policy=None, advance_policy=False,
                    ownership_migration=None, declaration_registration=None, candidate_roster=None,
                    comparison_policy=None):
    """Maintain one current table; registered source snapshots retain the history."""
    root = Path(root).resolve()
    if (root / "reports/forge/word_checkpoint_publication.py").is_file():
        from reports.forge.word_checkpoint_publication import install
        install()
    manifest_path = root / EVIDENCE_MANIFEST
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if recorded_policy is not None and source_commit is not None:
        raise ValueError("recorded policy publication cannot register new source evidence")
    if comparison_policy is not None and (recorded_policy is not None or advance_policy or ownership_migration is not None):
        raise ValueError("comparison policy preserves registered qualification; policy advance/recorded-policy are separate")
    if advance_policy and (source_commit is None or recorded_policy is not None):
        raise ValueError("advancing the evidence policy requires --source-commit and no recorded policy")
    if ownership_migration is not None and not (advance_policy and source_commit and recorded_policy is None):
        raise ValueError("ownership migration requires explicit frozen-source policy advance")
    if (declaration_registration is not None or candidate_roster is not None) and source_commit is None:
        raise ValueError("declaration roster requires exact frozen source reconstruction")
    if declaration_registration is not None and candidate_roster is not None:
        raise ValueError("cannot combine --declaration-registration and --candidate")
    recorded_view = None
    if recorded_policy is not None:
        if manifest is None:
            raise ValueError("recorded policy publication requires registered technique evidence")
        recorded_path = Path(recorded_policy)
        if not recorded_path.is_absolute():
            recorded_path = root / recorded_path
        recorded_view = read_json(recorded_path)
        recorded_policy = os.path.relpath(recorded_path.resolve(), root)
    reports, archived_reports, advancing = [], [], False
    policy = None
    if manifest is not None:
        if manifest.get("schema_version") != 1 or manifest.get("view") != view_id:
            raise ValueError("unsupported current technique evidence manifest/view")
        policy_path = (Path(comparison_policy) if comparison_policy is not None else
                       root / "configs/forge/views" / (view_id + ".json"))
        if not policy_path.is_absolute():
            policy_path = root / policy_path
        if recorded_view is not None or policy_path.is_file():
            policy = recorded_view if recorded_view is not None else read_json(policy_path)
            if (policy.get("id") != view_id or policy.get("revision") != manifest["view_revision"]
                    or stable_hash(policy) != manifest["policy_fingerprint"]):
                label = "recorded" if recorded_view is not None else "current"
                if not advance_policy:
                    raise ValueError(f"{label} view policy differs from registered technique evidence")
                if (policy.get("id") != view_id or type(policy.get("revision")) is not int
                        or policy["revision"] <= manifest["view_revision"]):
                    raise ValueError("advancing requires a later revision of the same view policy")
                advancing = True
        reports = [(entry, *_snapshot(root, entry, manifest)) for entry in manifest["cohorts"]]
        archived_reports = _archived_reports(root, manifest)
    pending_snapshot = pending_selection = pending_selection_text = selection_archive = None
    prior_manifest, prior_reports = deepcopy(manifest), list(reports)
    if advancing and execution_backend is None:
        from experiments.forge.trainer_families import CURRENT_SELECTION
        selection_path = root / CURRENT_SELECTION
        backends = ({pin["execution_backend"] for pin in read_json(selection_path)["selections"]}
                    if selection_path.is_file() else set())
        if len(backends) > 1:
            raise ValueError("policy advance with mixed selected backends requires an explicit --device")
        if backends:
            execution_backend = backends.pop()
    if source_commit is not None:
        with tempfile.TemporaryDirectory(prefix="forge-technique-evidence-") as temporary:
            roster_options = ({"declaration_registration": declaration_registration}
                              if declaration_registration is not None else {})
            if candidate_roster is not None:
                roster_options["candidate_roster"] = candidate_roster
            if comparison_policy is not None:
                roster_options["comparison_policy"] = comparison_policy
            metadata = regenerate(root, view_id=view_id, execution_backend=execution_backend,
                                  source_commit=source_commit, output_prefix=Path(temporary) / "evidence",
                                  **roster_options)
            report = read_json(metadata["json"])
        candidates = {}
        missing_worker_finished_at = {}
        for row in report["rows"]:
            if not row.get("attempt_ids"):
                continue
            if row["candidate_id"] in candidates:
                raise ValueError("select one runtime cohort; current rows cannot pool runtimes")
            times = []
            for attempt in row["attempt_ids"]:
                recorded_at = read_json(root / "reports/forge/attempts" / attempt / "result.json")["raw"].get("finished_at")
                if recorded_at is None:
                    missing_worker_finished_at.setdefault(row["candidate_id"], []).append(attempt)
                elif not isinstance(recorded_at, str) or not recorded_at:
                    raise ValueError(f"invalid recorded worker completion time: {attempt}")
                else:
                    times.append(recorded_at)
            if not times:
                raise ValueError(f"no recorded worker completion time for measured technique: {row['candidate_id']}")
            # Interrupted attempts can have no worker terminal timestamp. Keep
            # them in the evidence, but rank only by an actually recorded time.
            candidates[row["candidate_id"]] = max(times)
        if not candidates:
            raise ValueError("source regrade selected no measured technique rows; restore its recorded runtime/hardware")
        # Freeze declared blockers on the explicitly selected backend alongside
        # measured rows. A null completion time records no execution or gate
        # credit; it prevents cached regeneration from inventing CPU shadows or
        # resolving those same blockers against later source changes.
        if execution_backend is not None:
            from experiments.forge.trainer_families import CURRENT_SELECTION
            selection_path = root / CURRENT_SELECTION
            incumbents = ({pin["candidate_id"] for pin in read_json(selection_path)["selections"]}
                          if selection_path.is_file() else set())
            unmeasured = {}
            for row in report["rows"]:
                if (row["candidate_id"] in candidates or row.get("attempt_ids")
                        or (row.get("status") != "BLOCKED" and row["candidate_id"] not in incumbents)):
                    continue
                unmeasured.setdefault(row["candidate_id"], []).append(row)
            for name, rows in unmeasured.items():
                if len(rows) != 1 or rows[0].get("runtime_cohort", {}).get("execution_backend") != execution_backend:
                    raise ValueError("select one runtime cohort; blocked current rows cannot pool runtimes")
                candidates[name] = None
        if advancing:
            if (report.get("view") != policy["id"] or report.get("view_revision") != policy["revision"]
                    or report.get("policy_fingerprint") != stable_hash(policy)):
                raise ValueError("new source evidence differs from the current view policy")
            previous = {key: deepcopy(manifest[key]) for key in POLICY_FIELDS}
            previous["cohorts"] = deepcopy(manifest["cohorts"])
            archived_reports.extend((previous, entry, old_report, rows) for entry, old_report, rows in reports)
            manifest = {**deepcopy(manifest), **{key: deepcopy(report[key]) for key in POLICY_FIELDS},
                        "cohorts": [], "archived_policies": [*manifest.get("archived_policies", []), previous]}
            reports = []
        if manifest is None:
            manifest = {"schema_version": 1, **{key: report[key] for key in
                        ("view", "view_revision", "policy_fingerprint", "tier_requirements")},
                        "cohorts": [], "retired_publications": {}}
        for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements"):
            if report.get(key) != manifest[key]:
                raise ValueError(f"new technique evidence has incompatible {key}")
        data = _json_text(report)
        relative = EVIDENCE_MANIFEST.parent / (report["provenance"]["input_digest"] + ".json")
        entry = {"snapshot": relative.as_posix(), "json_sha256": hashlib.sha256(data.encode()).hexdigest(),
                 "source_commit": metadata["source_commit"], "candidates": candidates}
        if missing_worker_finished_at:
            entry["missing_worker_finished_at"] = {
                name: sorted(attempts) for name, attempts in sorted(missing_worker_finished_at.items())}
        # Keep exact evidence across CPU/GPU models and other runtime cohorts,
        # including cohorts from the same source commit. Regrading an identical
        # snapshot is idempotent; a new snapshot never retires another runtime.
        if entry not in manifest["cohorts"]:
            manifest["cohorts"].append(entry)
        # A frozen all-backend roster also contains unmeasured configurations
        # on other runtimes. Those cannot overwrite a measured registration
        # of the same candidate just because they appear later in the roster.
        selected = {row["candidate_id"]: row for row in report["rows"]
                    if row["candidate_id"] in candidates and
                    (bool(row.get("attempt_ids")) if isinstance(candidates[row["candidate_id"]], str) else True)}
        selected = dict(sorted(selected.items()))
        for row in selected.values():
            _validate_published_row(root, report, row)
        if advancing:
            pending_selection, selection_archive = _advanced_family_selection(
                root, prior_manifest, report, selected, prior_reports, ownership_migration=ownership_migration)
            if selection_archive:
                relative_selection, _, digest = selection_archive
                manifest["archived_policies"][-1]["family_selection"] = {
                    "path": relative_selection.as_posix(), "sha256": digest}
            if pending_selection is not None:
                pending_selection_text = json.dumps(pending_selection, sort_keys=True, indent=2) + "\n"
        if not any(old == entry for old, _, _ in reports):
            reports.append((entry, report, selected))
        pending_snapshot = root / relative, data
    if manifest is None:
        raise ValueError("no registered technique evidence; use --source-commit after a completed experiment")
    from experiments.forge.planning import declaration_paths
    from experiments.forge.trainer_families import (CURRENT_SELECTION, family_for_candidate, load_current_selection,
                                                    scientific_row_hash, select_family_rows)
    declarations = {path.stem: read_json(path) for path in declaration_paths(root)}
    candidates = set(declarations)
    roster_card = pending_selection
    if roster_card is None and (root / CURRENT_SELECTION).is_file():
        roster_card = read_json(root / CURRENT_SELECTION)
    migrated_roster = bool(roster_card and roster_card.get("ownership_migration"))
    if migrated_roster:
        current_ids = {pin["candidate_id"] for pin in roster_card["selections"]}
        if (roster_card["ownership_migration"].get("selection_scope") != "seven_visible_selected_leaders"
                or len(current_ids) < 7 or len(current_ids) != len(roster_card["selections"])
                or not current_ids.issubset(candidates)):
            raise ValueError("ownership selection requires exact seven current declarations")
        candidates = current_ids
    if declaration_registration is not None:
        registration_path = Path(declaration_registration)
        if not registration_path.is_absolute():
            registration_path = root / registration_path
        registered_candidates = {plan["candidate_id"] for plan in read_json(registration_path)["families"].values()}
        if len(registered_candidates) != 7 or not registered_candidates.issubset(candidates):
            raise ValueError("current selection roster lacks exact seven registered declarations")
        candidates = registered_candidates
    restricted_roster = declaration_registration is not None or migrated_roster
    current_pins = (load_current_selection(root, view_id=view_id, policy_fingerprint=manifest["policy_fingerprint"],
                                         selection_card=pending_selection)
                    if recorded_view is None else {})
    declaration_sources = _declaration_fallback_sources(current_pins)
    selected = {}
    for entry, report, rows in reports:
        for name, row in rows.items():
            backend = row.get("runtime_cohort", {}).get("execution_backend", report.get("execution_backend"))
            if name not in candidates or (execution_backend is not None and backend != execution_backend):
                continue
            key = name, backend
            rank = entry["candidates"][name] or ""
            previous = selected.get(key)
            if previous is None or rank > previous[0]:
                selected[key] = rank, entry, report, deepcopy(row)
            elif (rank == previous[0] and any(row.get(field) != previous[3].get(field)
                                             for field in ("candidate_revision", "cohort", "runtime_cohort"))):
                preference = (_declaration_source_preference(row, previous[3], declaration_sources.get(backend))
                              if rank == "" else None)
                if preference is True:
                    selected[key] = rank, entry, report, deepcopy(row)
                elif preference is None and family_for_candidate(root, name, declarations.get(name),
                                                                current_presentation=True)["id"] not in current_pins:
                    raise ValueError("ambiguous latest recorded technique cohort")
    # A search runtime needs the actual canonical declaration as its fallback;
    # never present an arbitrary unmeasured tuning trial as the family default.
    canonical_missing = set()
    for (name, backend), (_, entry, report, row) in list(selected.items()):
        family = family_for_candidate(root, name, declarations.get(name), current_presentation=recorded_view is None)
        if restricted_roster and current_pins.get(family["id"], {}).get("candidate_id") == name:
            continue
        canonical = family["canonical_candidate"]
        runtime_key = stable_hash(row.get("runtime_cohort"))
        if any(item[3]["candidate_id"] == canonical and item[3].get("runtime_cohort") == row.get("runtime_cohort")
               for item in selected.values()):
            continue
        matches = [item for item in report["rows"] if item["candidate_id"] == canonical
                   and item.get("runtime_cohort") == row.get("runtime_cohort")
                   and item.get("bindings", {}).get("source_digest") == row.get("bindings", {}).get("source_digest")]
        if len(matches) == 1 and not matches[0].get("attempt_ids"):
            _validate_published_row(root, report, matches[0])
            selected[canonical, backend, runtime_key] = "", entry, report, deepcopy(matches[0])
        else:
            canonical_missing.add((canonical, backend, runtime_key))
    missing = candidates - {key[0] for key in selected}
    live = None
    fallback_reporting = None
    if (missing or canonical_missing) and recorded_view is None:
        with tempfile.TemporaryDirectory(prefix="forge-technique-current-") as temporary:
            recorded = {name for _, _, rows in reports for name, row in rows.items() if row.get("attempt_ids")}
            fallback_ids = missing | {name for name, _, _ in canonical_missing}
            if declaration_registration is not None and fallback_ids.intersection(recorded):
                raise ValueError("current fallback cannot replay registered measured candidates")
            source_manifests = (_receipt_manifests(root, source_commit)
                                if source_commit is not None and declaration_registration is not None else {})
            with _prefer_recorded_and_canonical_declarations(root, recorded), \
                    (_missing_declaration_enumeration(root, fallback_ids)
                     if declaration_registration is not None else nullcontext()), \
                    _unchanged_fallback_source_inspection(root, source_manifests) as source_inspection:
                metadata = write_report(root, view_id, execution_backend=execution_backend,
                                        output_prefix=Path(temporary) / "current")
                if declaration_registration is not None:
                    fallback_reporting = dict(candidate_ids=sorted(fallback_ids),
                        registered_measured_ids_excluded=sorted(recorded),
                        full_parent_control_lookup_preserved=True, source_inspection=source_inspection)
            live = read_json(metadata["json"])
        for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements"):
            if live.get(key) != manifest[key]:
                raise ValueError(f"current declarations have incompatible {key}; update the evidence view explicitly")
        for row in live["rows"]:
            backend = row.get("runtime_cohort", {}).get("execution_backend")
            runtime_key = stable_hash(row.get("runtime_cohort"))
            if row["candidate_id"] in missing or (row["candidate_id"], backend, runtime_key) in canonical_missing:
                # Measured rows must first be frozen and registered, so a later
                # code change cannot silently erase their published results.
                if row.get("attempt_ids"):
                    raise ValueError("register new measured technique evidence with --source-commit")
                row.setdefault("bindings", {})["source_origin_commit"] = None
                selected[row["candidate_id"], backend, runtime_key] = "", None, live, deepcopy(row)
    declaration_history = {}
    if current_pins:
        # Exact historical incumbents must survive newer observations of the
        # same card. Keep complete source/runtime alternatives, never cells.
        retained = {}
        from experiments.forge.trainer_families import _declaration_only_row
        preferred_declarations = {(item[3]["candidate_id"], item[3].get("runtime_cohort", {}).get("execution_backend"))
                                  for item in selected.values()
                                  if _declaration_only_row(item[3])
                                  and declaration_sources.get(item[3].get("runtime_cohort", {}).get("execution_backend"))
                                  and item[3].get("bindings", {}).get("source_digest") == declaration_sources.get(
                                      item[3].get("runtime_cohort", {}).get("execution_backend"))}
        def excluded_declaration(row):
            name = row["candidate_id"]
            backend = row.get("runtime_cohort", {}).get("execution_backend")
            return ((name, backend) in preferred_declarations and _declaration_only_row(row)
                    and row.get("bindings", {}).get("source_digest") != declaration_sources.get(backend)
                    and family_for_candidate(root, name, declarations.get(name),
                                             current_presentation=True)["id"] not in current_pins)
        for entry, report, rows in reports:
            for name, row in rows.items():
                backend = row.get("runtime_cohort", {}).get("execution_backend")
                if name not in candidates or (execution_backend is not None and backend != execution_backend):
                    continue
                if excluded_declaration(row):
                    # Keep this row below in immutable evidence/history, rather
                    # than supplying a second canonical declaration fallback.
                    declaration_history.setdefault(scientific_row_hash(row),
                                                   {**deepcopy(row), "publication_key": entry["json_sha256"]})
                    continue
                retained.setdefault(scientific_row_hash(row), ("", entry, report, deepcopy(row)))
        for item in selected.values():
            if excluded_declaration(item[3]):
                continue
            retained.setdefault(scientific_row_hash(item[3]), item)
        selected = retained
    from experiments.forge.technique_board import DEFAULT_LABELS
    result = {"schema_version": 1, "reducer_version": "forge-current-technique-inventory-v1",
              "publication_scope": "current_technique_inventory", "qualification_reuse": False,
              "qualification_input": False, **{key: manifest[key] for key in
              ("view", "view_revision", "policy_fingerprint", "tier_requirements")},
              "execution_backend": execution_backend, "rows": [], "evidence_sources": {}}
    if manifest.get("archived_policies"):
        result["archived_policies"] = deepcopy(manifest["archived_policies"])
    if recorded_policy is not None:
        result["recorded_policy"] = recorded_policy
    for _, entry, report, row in selected.values():
        if entry is not None:
            row["publication_key"] = entry["json_sha256"]
            result["evidence_sources"][row["publication_key"]] = deepcopy(entry)
        else:
            row.pop("publication_key", None)
        row.update(qualification_reuse=False, qualification_input=False,
                   technique=DEFAULT_LABELS.get(row["candidate_id"], row.get("technique", row["candidate_id"])))
        result["rows"].append(row)
    result["rows"].sort(key=lambda row: (-row["qualified_tier"], row["technique"], row.get("cohort") or ""))
    for catalog in CONTRACT_CATALOGS:
        combined = {}
        for report in ([data_report for _, data_report, _ in reports]
                       + [data_report for _, _, data_report, _ in archived_reports] + ([live] if live else [])):
            for digest, contract in report.get(catalog, {}).items():
                if digest != stable_hash(contract):
                    raise ValueError(f"invalid {catalog} identity in technique evidence")
                if digest in combined and combined[digest] != contract:
                    raise ValueError("conflicting scientific contract identities")
                combined[digest] = deepcopy(contract)
        result[catalog] = dict(sorted(combined.items()))
    presentation_card = pending_selection
    if presentation_card is None and (root / CURRENT_SELECTION).is_file():
        presentation_card = read_json(root / CURRENT_SELECTION)
    with _retain_recorded_measurements(root, prior_reports) as retained_measurements, \
            _ownership_migration_helper().archived_presentation(root, presentation_card):
        family_result = select_family_rows(root, result["rows"], result, view_id=view_id,
                                           policy_fingerprint=manifest["policy_fingerprint"], declarations=declarations,
                                           view_policy=recorded_view, execution_backend=execution_backend,
                                           selection_card=pending_selection,
                                           historical_rows=[*declaration_history.values(), *[{**row, "publication_key": entry["json_sha256"]}
                                               for _, entry, _, rows in archived_reports for row in rows.values()]])
    if presentation_card and presentation_card.get("ownership_migration"):
        _ownership_migration_helper().label_archived_declarations(family_result, presentation_card)
        restored = _ownership_migration_helper().restore_archived_solution_rows(
            root, family_result, presentation_card, archived_reports, validate_row=_validate_published_row)
        if restored:
            result["archived_base_presentation_only"] = restored
        result["ownership_migration"] = deepcopy(presentation_card["ownership_migration"])
    result.update(family_result)
    if retained_measurements:
        result["retained_recorded_measurements"] = {"qualification_input": False, "qualification_reuse": False,
            "reporter_sha256": file_hash(Path(__file__)), "pins_unchanged": True,
            "scope": "Previously selected measurements remain under exact original Git task/view contracts; stale against live tasks.",
            "proofs": retained_measurements}
    # Immutable scientific history stays numerical, including earlier revisions
    # of the same configuration. It cannot fill cells in the selected row.
    result["evidence_rows"] = []
    for entry, report, rows in reports:
        result["evidence_sources"][entry["json_sha256"]] = deepcopy(entry)
        for original in rows.values():
            if execution_backend is not None and original.get("runtime_cohort", {}).get("execution_backend") != execution_backend:
                continue
            evidence = deepcopy(original)
            evidence.update(publication_key=entry["json_sha256"], qualification_reuse=False, qualification_input=False)
            result["evidence_rows"].append(evidence)
    if archived_reports:
        result["archived_evidence_rows"] = []
        for policy, entry, _, rows in archived_reports:
            result["evidence_sources"][entry["json_sha256"]] = deepcopy(entry)
            for original in rows.values():
                evidence = deepcopy(original)
                evidence.update(publication_key=entry["json_sha256"], qualification_reuse=False,
                                qualification_input=False,
                                evidence_policy={key: deepcopy(policy[key]) for key in POLICY_FIELDS})
                result["archived_evidence_rows"].append(evidence)
    # Compact API studies retain their own gates, laws and full denominators.
    # They are display evidence and never enter ordinary family selection.
    from experiments.forge import completed_studies as completed_studies_projection
    completed_studies = completed_studies_projection.load_completed_studies(root)
    if completed_studies:
        result["completed_api_studies"] = completed_studies
    from experiments.forge.model_toy_publication import load_publication as load_model_research
    model_research = load_model_research(root)
    if model_research is not None:
        result["model_api_research"] = model_research
    if not recorded_policy and view_id == "discriminator_stability":
        result["standalone_api_scores"] = _standalone_api_scores(root)
        from experiments.forge import publication_memory
        passive = publication_memory.load_passive_publications(root)
        if passive:
            result["passive_publications"] = passive
        original_pr223 = _original_pr223_atlas(result)
        if original_pr223:
            result["original_pr223_atlas"] = original_pr223
        atlas_progress = _atlas_unblocking_progress(root)
        if atlas_progress:
            result["atlas_unblocking_progress"] = atlas_progress
        half_base = _word_half_base_diagnostic(root)
        if half_base:
            result["word_half_base_diagnostic"] = half_base
    debug_root = root / "reports/forge/c6-baseline-debug-20261003"
    if debug_root.is_dir():
        debug_names = ("README.md", "BASELINE_SELECTION.md", "BASELINE_SELECTION.json",
                       "RETAINED_HOLD_DEBUG.md", "retained-hold-debug.json",
                       "analyze_retained_hold.py", "copy-verification.json")
        result["baseline_debugging"] = {
            "readout": (debug_root / "README.md").relative_to(root).as_posix(),
            "files_sha256": {(debug_root / name).relative_to(root).as_posix(): file_hash(debug_root / name)
                             for name in debug_names},
            "qualification_input": False, "qualification_reuse": False,
            "scope": "Exact original C6 baseline selection and retained-data causal diagnosis; no new qualification",
        }
    result["provenance"] = {"publication_reducer_sha256": file_hash(Path(__file__)),
                            "declaration_fallback_source_by_backend": declaration_sources,
                            "evidence_manifest_sha256": stable_hash(manifest),
                            "trainer_family_registry_sha256": file_hash(root / "configs/forge/trainer-families.json")
                                if (root / "configs/forge/trainer-families.json").is_file() else None,
                            "selected_rows_sha256": stable_hash(result["rows"]),
                            "family_current_selection_sha256": (hashlib.sha256(pending_selection_text.encode()).hexdigest()
                                if pending_selection_text is not None else file_hash(root / CURRENT_SELECTION))
                                if current_pins else None}
    registry_path = root / "configs/forge/trainer-families.json"
    if registry_path.is_file():
        result["trainer_family_registry"] = read_json(registry_path)
    if not recorded_policy and view_id == "discriminator_stability":
        result["common26_display"] = _common26_display_projection(result, root)
        result["provenance"]["common26_comparison_projector_sha256"] = file_hash(
            Path(__file__).resolve().parents[2] / "experiments/forge/common26_comparison.py")
        audit = root / COMMON26_DISPLAY_AUDIT
        result["provenance"]["common26_display_audit_sha256"] = (
            file_hash(audit) if audit.is_file() else None)
    if completed_studies:
        result["provenance"]["completed_studies_projector_sha256"] = file_hash(
            Path(completed_studies_projection.__file__))
    if result.get("passive_publications"):
        result["provenance"]["passive_publications_reducer_sha256"] = file_hash(Path(publication_memory.__file__))
    if not recorded_policy and view_id == "discriminator_stability":
        from experiments.forge.family_reports import build_progress
        result["family_progress"] = build_progress(root, result, selection_card=pending_selection)
    if fallback_reporting is not None:
        result["current_fallback_reporting_scope"] = fallback_reporting
    from experiments.forge.scaling_publication import attach_performance
    attach_performance(root, result)
    result = compact_current_publication(result)
    result["provenance"]["input_digest"] = stable_hash(result)
    json_path, markdown_path = root / CURRENT_PREFIX.with_suffix(".json"), root / CURRENT_PREFIX.with_suffix(".md")
    markdown = _current_markdown(result, root, markdown_path)
    with _pending_publication_inputs(root, pending_snapshot):
        pages = _family_pages(root, result)
    shared_archive = (_shared_score_archive(root)
                      if not recorded_policy and view_id == "discriminator_stability" else None)
    shared_intro = (_shared_score_intro(root, result)
                    if not recorded_policy and view_id == "discriminator_stability" else None)
    # Validate the serialized size before changing selection cards or outputs.
    json_text = _json_text(result)
    # Validate all inputs before changing selection cards, registry or outputs.
    if selection_archive:
        relative_selection, original, _ = selection_archive
        _write_changed(root / relative_selection, original)
    if pending_selection is not None:
        _write_changed(root / CURRENT_SELECTION, pending_selection_text)
    if pending_snapshot:
        _write_changed(*pending_snapshot)
        _write_changed(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    _write_changed(json_path, json_text)
    _write_changed(markdown_path, markdown)
    _write_family_pages(root, pages)
    if shared_archive:
        _write_changed(*shared_archive)
    if shared_intro:
        _write_changed(*shared_intro)
    return {"report": str(markdown_path), "json": str(json_path), "rows": len(result["rows"]),
            "input_digest": result["provenance"]["input_digest"], "qualification_reuse": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--goal", default="discriminator_stability")
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--source-commit", help="reconstruct and grade an exact recorded Git source cohort, independently of live HEAD")
    parser.add_argument("--output-prefix", type=Path,
                        help="write an independent source report; do not register or change current selections")
    parser.add_argument("--declaration-registration", type=Path,
                        help="enumerate the exact seven-leader ordinary CUDA registration recorded at --source-commit; keep full control lookup")
    parser.add_argument("--candidate", dest="candidates", action="append",
                        help="enumerate only these explicit candidate declaration IDs at --source-commit; keep full control lookup")
    parser.add_argument("--recorded-policy", type=Path,
                        help="rebuild registered rows under this exact archived view policy, without resolving live declarations")
    parser.add_argument("--comparison-policy", type=Path,
                        help="explicit registered older comparison policy: new-source grades retain separate actual qualification; cached use never regrades")
    parser.add_argument("--advance-policy", action="store_true",
                        help="with --source-commit, archive earlier policy cohorts and register evidence for the current view revision")
    parser.add_argument("--ownership-migration", type=Path,
                        help="explicit source-bound selected-leader ownership migration; requires --advance-policy and --source-commit")
    parser.add_argument("--refresh-publication", action="store_true",
                        help="refresh displayed evidence and view coverage, preserving registered scores and selections; "
                             "always pass it after a view revision")
    parser.add_argument("--refresh-performance", action="store_true",
                        help="refresh only the compact DCGAN performance block; preserve scientific results and family pages")
    args = parser.parse_args(argv)
    if args.refresh_performance:
        if (args.output_prefix or args.recorded_policy or args.comparison_policy or args.advance_policy or args.ownership_migration
                or args.refresh_publication or args.source_commit or args.declaration_registration or args.candidates
                or args.device != "all" or args.goal != "discriminator_stability"):
            parser.error("--refresh-performance preserves current evidence; source, policy, goal and device overrides are unsupported")
        from experiments.forge.scaling_publication import refresh_performance
        print(_json_text(refresh_performance(args.root)), end="")
        return
    if args.output_prefix is not None:
        if args.recorded_policy or args.advance_policy or args.ownership_migration or args.refresh_publication:
            parser.error("independent --output-prefix cannot register or refresh current evidence/selection")
        print(_json_text(regenerate(args.root, view_id=args.goal,
                                   execution_backend=None if args.device == "all" else args.device,
                                   source_commit=args.source_commit, output_prefix=args.output_prefix,
                                   declaration_registration=args.declaration_registration,
                                   candidate_roster=args.candidates,
                                   comparison_policy=args.comparison_policy)), end="")
        return
    # A later declared view can refresh navigation without advancing/regrading
    # the immutable recorded policy. Keep the default one-command workflow usable.
    if not (args.refresh_publication or args.source_commit or args.recorded_policy or args.comparison_policy
            or args.advance_policy or args.device != "all"):
        manifest_path = args.root / EVIDENCE_MANIFEST
        view_path = args.root / "configs/forge/views" / (args.goal + ".json")
        if manifest_path.is_file() and view_path.is_file():
            manifest = read_json(manifest_path)
            if stable_hash(read_json(view_path)) != manifest.get("policy_fingerprint"):
                args.refresh_publication = True
    if args.refresh_publication:
        if args.source_commit or args.recorded_policy or args.comparison_policy or args.advance_policy or args.ownership_migration or args.declaration_registration or args.candidates or args.device != "all":
            parser.error("--refresh-publication preserves the existing cohorts; source, policy and device overrides are unsupported")
        print(_json_text(refresh_publication(args.root, view_id=args.goal)), end="")
        return
    print(_json_text(publish_current(args.root, view_id=args.goal,
                                execution_backend=None if args.device == "all" else args.device,
                                source_commit=args.source_commit, recorded_policy=args.recorded_policy,
                                advance_policy=args.advance_policy, ownership_migration=args.ownership_migration,
                                declaration_registration=args.declaration_registration,
                                candidate_roster=args.candidates,
                                comparison_policy=args.comparison_policy)), end="")


if __name__ == "__main__":
    main()
