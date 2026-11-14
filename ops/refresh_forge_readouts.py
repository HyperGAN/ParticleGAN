"""Single-command Forge readout checker and refresher when no code changes are needed.

Usage:
  python ops/refresh_forge_readouts.py --check
  python ops/refresh_forge_readouts.py --check --reports-only
  python ops/refresh_forge_readouts.py --gpus 0,1 --scaling-gpu auto
  python ops/refresh_forge_readouts.py --reports-only

Exit codes:
  0  All Forge readouts are current (or were refreshed cleanly to current).
  1  (--check only) Out-of-date Forge readouts remain and can be refreshed automatically.
  2  Code or declaration changes are required (preflight error, broken task/adapter,
     unrecoverable stale queue state, or non-gate task execution failure such as ERROR/OOM).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.forge import attempt_coverage, history, knowledge, planning, queue as forge_queue
from experiments.forge import scaling, scaling_export, scaling_publication, studies
from experiments.forge.contracts import (
    atomic_json,
    atomic_text,
    file_hash,
    file_lock,
    identifier,
    read_json,
    stable_hash,
)
from experiments.forge.sources import inspect_source, snapshot_source
from experiments.forge.leaderboard_contract import FIXED_TEST_LIST
from experiments.forge.trainer_families import CURRENT_SELECTION, current_family_candidates
from experiments.forge.views import (
    COUNTED_OR_HELD,
    diagnostic_evidence_scope,
    load_tasks,
    load_view,
    task_evaluation_fingerprint,
    task_execution_fingerprint,
)

CURRENT_EXIT = 0
NEEDS_REFRESH_EXIT = 1
CODE_CHANGE_EXIT = 2

DEFAULT_CUDA_MODEL = "NVIDIA RTX A6000"
ORDINARY_LEADERBOARD_VIEWS = (
    "adaptation",
    "clockfree_continuous",
    "formulation_comparison",
    "host_profile_transfer",
    "quality_coverage",
)
SCOPED_ATTEMPT_KEYS = ("research_scope", "repair_scope", "policy_scope", "cohort_scope")
OP_COMPLEMENT = {">=": "<", "<=": ">", ">": "<=", "<": ">=", "==": "<"}


class CodeChangeRequiredError(RuntimeError):
    """Raised when refreshing readouts cannot proceed without code or card changes."""


def _log(message: str, *, json_mode: bool = False) -> None:
    stream = sys.stderr if json_mode else sys.stdout
    print(f"[refresh-forge] {message}", file=stream, flush=True)


def _is_scoped_attempt(record: dict[str, Any]) -> bool:
    return any(record.get(key) for key in SCOPED_ATTEMPT_KEYS)


def catalog_source_manifest(root: Path, idea: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the exact source manifest (including task evaluation/hostprofile extras) used by planning.resolve_idea."""
    from experiments.forge.hostprofiles import profile_source_paths

    root = Path(root).resolve()
    extra_sources = set((idea or {}).get("source_files", []))
    for task in load_tasks(root).values():
        extra_sources.update(
            relative
            for relative in task.get("evaluation", {}).get("sources", {})
            if (root / relative).is_file()
        )
        try:
            profile_sources = profile_source_paths(task)
        except (KeyError, TypeError, ValueError):
            profile_sources = {}
        extra_sources.update(
            relative for relative in profile_sources if (root / relative).is_file()
        )
    return inspect_source(root, sorted(extra_sources))


def _request_matches_worktree(root: Path, req: dict[str, Any]) -> bool:
    expected = catalog_source_manifest(root, req.get("candidate"))["digest"]
    return req.get("source", {}).get("digest") == expected


def scaling_source_files_sha256(root: Path) -> dict[str, str]:
    """Compute the exact code-file digest map used by scaling_benchmark._source()."""
    root = Path(root).resolve()
    paths = sorted(
        set((root / "particlegan").rglob("*.py"))
        | {
            root / "experiments/forge/scaling_benchmark.py",
            root / "experiments/forge/api.py",
            root / "experiments/forge/rng.py",
            root / "experiments/forge/priors.py",
            root / "experiments/forge/state.py",
            root / "experiments/forge/contracts.py",
            root / "benchmarks/toy_audit/reproducibility.py",
        }
    )
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise CodeChangeRequiredError(f"missing scaling benchmark source files: {missing}")
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def _git_task_names_at_commit(root: Path, commit: str) -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, "configs/forge/tasks"],
            cwd=root,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise CodeChangeRequiredError(f"cannot inspect frozen commit {commit}: {exc}") from exc
    names = set()
    for line in output.splitlines():
        path = Path(line.strip())
        if path.suffix == ".json" and path.parent.as_posix() == "configs/forge/tasks":
            names.add(path.stem)
    return names


def _selected_family_source_commit(root: Path, publication: dict[str, Any], family: str, selection: dict[str, Any]) -> str:
    sources = publication.get("evidence_sources", {})
    pub_key = selection.get("publication_key")
    commit = sources.get(pub_key, {}).get("source_commit") if pub_key else None
    if not commit:
        commit = selection.get("bindings", {}).get("source_origin_commit")
    if not commit:
        preflight = root / "reports/forge/tier-coverage-20261014/preflight.json"
        if preflight.is_file():
            pins = {item["family"]: item for item in read_json(preflight).get("families", [])}
            commit = pins.get(family, {}).get("source_commit") or pins.get(family, {}).get("origin_commit")
    if not commit:
        raise CodeChangeRequiredError(f"selected family {family} has no recorded frozen source commit")
    return commit


def _selected_family_preserved_absent_paths(
    root: Path, publication: dict[str, Any], selection: dict[str, Any]
) -> list[str]:
    pub_key = selection.get("publication_key")
    snap_rel = publication.get("evidence_sources", {}).get(pub_key, {}).get("snapshot") if pub_key else None
    if not snap_rel:
        return []
    snap_path = root / snap_rel
    if not snap_path.is_file():
        return []
    closure = read_json(snap_path).get("frozen_catalog_support_closure") or {}
    return sorted(set(closure.get("preserved_absent_paths") or []))


def check_queue_status(root: Path, queue_root: Path) -> dict[str, Any]:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return {"current": True, "active_submissions": [], "awaiting_readout_submissions": []}
    state = read_json(state_path)
    active, awaiting = [], []
    for req_id, entry in sorted(state.get("submissions", {}).items()):
        status = entry.get("status")
        lifecycle = entry.get("lifecycle")
        cand = entry.get("request", {}).get("candidate", {}).get("id", "unknown")
        study_id = studies.selected_study_id(entry.get("request", {}))
        if status in {"queued", "running", "paused"}:
            active.append({"request_id": req_id, "candidate_id": cand, "study_id": study_id, "status": status})
        elif lifecycle == "awaiting_readout":
            req = entry.get("request", {})
            keys = [job["compatibility_key"] for job in req.get("jobs", [])]
            has_results = any(state.get("jobs", {}).get(k, {}).get("result") is not None for k in keys)
            if not has_results and not _request_matches_worktree(root, req):
                raise CodeChangeRequiredError(
                    f"queue submission {req_id} ({cand}) is awaiting_readout on superseded source "
                    f"with zero durable attempts; supersede or clean its lifecycle explicitly"
                )
            awaiting.append({"request_id": req_id, "candidate_id": cand, "study_id": study_id, "status": status})
    return {
        "current": not active and not awaiting,
        "active_submissions": active,
        "awaiting_readout_submissions": awaiting,
    }


def _scaling_contention_status(result: dict[str, Any]) -> str:
    hw = result.get("hardware")
    if isinstance(hw, dict):
        return str(hw.get("contention_status", "unknown"))
    id_hw = (result.get("identity") or {}).get("hardware")
    if isinstance(id_hw, dict) and "contention_status" in id_hw:
        return str(id_hw["contention_status"])
    return "clear"


def _safe_float(raw: str, default: float = 0.0) -> float:
    token = raw.strip()
    if not token:
        return default
    try:
        val = float(token)
    except ValueError:
        return default
    return val if val == val else default


def _lookup_gpu_token(
    token: str,
    by_phys_index: dict[str, tuple[str, float, float]],
    by_uuid: dict[str, tuple[str, float, float]],
) -> tuple[str, float, float] | None:
    tok = token.strip()
    if not tok or tok.startswith("MIG-"):
        return None
    if tok.startswith("GPU-"):
        if tok in by_uuid:
            return by_uuid[tok]
        matches = [info for uuid_key, info in by_uuid.items() if uuid_key.startswith(tok)]
        return matches[0] if len(matches) == 1 else None
    if tok.isdigit():
        return by_phys_index.get(tok)
    return None


def _resolve_scaling_gpu(
    scaling_gpu: str,
    gpus: list[str] | tuple[str, ...] = ("0", "1"),
    *,
    json_mode: bool = False,
) -> str:
    raw = str(scaling_gpu).strip()
    if not raw:
        raise CodeChangeRequiredError("a non-empty --scaling-gpu index is required when scaling_performance is stale")
    if raw.lower() != "auto":
        return raw

    candidates = [str(g).strip() for g in gpus if str(g).strip()]
    if not candidates:
        raise CodeChangeRequiredError("cannot auto-select --scaling-gpu from empty --gpus list")

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_tokens = (
        [tok.strip() for tok in visible_env.split(",") if tok.strip()]
        if visible_env is not None and visible_env.strip()
        else None
    )
    if visible_tokens is not None:
        valid_candidates = [
            cand for cand in candidates if not cand.isdigit() or int(cand) < len(visible_tokens)
        ]
        if not valid_candidates:
            raise CodeChangeRequiredError(
                f"no --gpus candidate {candidates} is within visible CUDA_VISIBLE_DEVICES ({visible_env!r})"
            )
    else:
        valid_candidates = candidates

    try:
        gpu_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        app_proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if gpu_proc.returncode == 0 and app_proc.returncode == 0:
            by_phys_index: dict[str, tuple[str, float, float]] = {}
            by_uuid: dict[str, tuple[str, float, float]] = {}
            for line in gpu_proc.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 4 or not parts[0].isdigit() or not parts[1].startswith("GPU-"):
                    continue
                idx_str, uuid_str = parts[0], parts[1]
                mem_used = _safe_float(parts[2], 0.0)
                util = _safe_float(parts[3], 0.0)
                info = (uuid_str, mem_used, util)
                by_phys_index[idx_str] = info
                by_uuid[uuid_str] = info

            own_pid = str(os.getpid())
            ext_procs_by_uuid: Counter[str] = Counter()
            for line in app_proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 3 and parts[0].startswith("GPU-") and parts[1].isdigit() and parts[1] != own_pid:
                    ext_procs_by_uuid[parts[0]] += 1

            visible_uuids: set[str] | None = None
            if visible_tokens is not None:
                visible_uuids = {
                    info[0]
                    for tok in visible_tokens
                    if (info := _lookup_gpu_token(tok, by_phys_index, by_uuid)) is not None
                }

            scored: list[tuple[tuple[bool, int, float, float, int], str, str, int, float]] = []
            for pos, cand in enumerate(valid_candidates):
                gpu_info: tuple[str, float, float] | None = None
                if visible_tokens is not None:
                    if cand.isdigit():
                        gpu_info = _lookup_gpu_token(visible_tokens[int(cand)], by_phys_index, by_uuid)
                    else:
                        cand_info = _lookup_gpu_token(cand, by_phys_index, by_uuid)
                        if cand_info is not None and visible_uuids is not None and cand_info[0] in visible_uuids:
                            gpu_info = cand_info
                else:
                    gpu_info = _lookup_gpu_token(cand, by_phys_index, by_uuid)
                if gpu_info is None:
                    continue
                uuid_str, mem_used, util = gpu_info
                proc_count = ext_procs_by_uuid.get(uuid_str, 0)
                score = (proc_count > 0, proc_count, mem_used, util, pos)
                scored.append((score, cand, uuid_str, proc_count, mem_used))

            if scored:
                scored.sort(key=lambda item: item[0])
                _, best_cand, best_uuid, best_procs, best_mem = scored[0]
                if best_procs == 0:
                    _log(
                        f"Auto-selected uncontended scaling GPU cuda:{best_cand} "
                        f"({best_uuid}, 0 external compute processes, {best_mem:.0f} MiB used)",
                        json_mode=json_mode,
                    )
                else:
                    _log(
                        f"Warning: all candidate GPUs have external compute processes; auto-selected "
                        f"least-contended cuda:{best_cand} ({best_uuid}, {best_procs} external compute processes)",
                        json_mode=json_mode,
                    )
                return best_cand
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    _log(f"nvidia-smi unavailable for --scaling-gpu auto; falling back to cuda:{valid_candidates[0]}", json_mode=json_mode)
    return valid_candidates[0]


def check_scaling_performance(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    try:
        evidence = scaling_publication.load_performance(root)
    except ValueError as exc:
        raise CodeChangeRequiredError(f"invalid scaling-performance.json: {exc}") from exc
    expected_choices = current_family_candidates(root)
    expected_perf_pairs = {
        (item.get("trainer_family") or item["family"], item["candidate_id"])
        for item in expected_choices
    }
    expected_reg_pairs = {(item["family"], item["candidate_id"]) for item in expected_choices}
    expected_tasks = [item["task"] for item in load_view(root, "dcgan_performance")["assignments"]]
    expected_cells = {(fam, cid, tid) for fam, cid in expected_perf_pairs for tid in expected_tasks}
    if evidence is None:
        return {"current": False, "reason": "missing_manifest", "missing_cells": len(expected_cells)}

    unavailable = [
        row for row in evidence.get("unavailable", [])
        if (row["trainer_family"], row["candidate_id"]) in expected_perf_pairs
    ]
    records = [
        row for row in evidence.get("records", [])
        if (row["trainer_family"], row["candidate_id"]) in expected_perf_pairs
    ]
    complete_cells = set()
    non_complete = []
    cohorts = set()
    expected_files_sha = scaling_source_files_sha256(root)
    stale_source_cells = []
    contended_cells = []
    comp_keys_by_task: dict[str, set[str]] = {tid: set() for tid in expected_tasks}
    complete_pairs_by_task: dict[str, set[tuple[str, str]]] = {tid: set() for tid in expected_tasks}

    for row in records:
        cell = (row["trainer_family"], row["candidate_id"], row["task_id"])
        result = row["result"]
        status = scaling_publication.performance_status(result)
        contention = _scaling_contention_status(result)
        if contention != "clear":
            contended_cells.append({"cell": cell, "contention": contention})
        hw_hash = stable_hash(scaling_publication.hardware_key(result))
        if status == "COMPLETE":
            complete_cells.add(cell)
            if row["task_id"] in complete_pairs_by_task:
                complete_pairs_by_task[row["task_id"]].add((row["trainer_family"], row["candidate_id"]))
                if "comparison_key" in row:
                    comp_key = str(row["comparison_key"])
                elif isinstance(result.get("hardware"), dict):
                    comp_key = scaling_publication.comparison_key(
                        result, receipt_sha256=row.get("receipt_sha256")
                    )
                else:
                    comp_key = stable_hash(
                        (
                            row["task_id"],
                            result["identity"]["source"]["sha256"],
                            result["identity"]["runtime_sha256"],
                            hw_hash,
                            contention,
                            stable_hash(row) if contention != "clear" else None,
                        )
                    )
                comp_keys_by_task[row["task_id"]].add(comp_key)
        else:
            non_complete.append({"cell": cell, "status": status})
        cohorts.add(
            (
                result["identity"]["source"]["sha256"],
                result["identity"]["runtime_sha256"],
                hw_hash,
            )
        )
        if result["identity"]["source"].get("files_sha256") != expected_files_sha:
            stale_source_cells.append(cell)

    unpaired_tasks = []
    for tid in expected_tasks:
        unavailable_for_task = {
            (u["trainer_family"], u["candidate_id"])
            for u in unavailable
            if u.get("task_id") == tid
        }
        runnable_for_task = expected_perf_pairs - unavailable_for_task
        if runnable_for_task and (
            complete_pairs_by_task.get(tid, set()) != runnable_for_task
            or len(comp_keys_by_task.get(tid, set())) != 1
        ):
            unpaired_tasks.append(tid)

    declaration_mismatches = []
    reg_id = (evidence.get("export") or {}).get("registration")
    if reg_id:
        reg_path = scaling.manifest_path(root, reg_id)
        if reg_path.is_file():
            reg_doc = read_json(reg_path)
            by_pair = {
                (item["family"], item["candidate_id"]): item["candidate_declaration_sha256"]
                for item in reg_doc.get("family_selections", [])
            }
            for fam, cid in sorted(expected_reg_pairs):
                idea_path = root / "configs/forge/ideas" / f"{cid}.json"
                if by_pair.get((fam, cid)) != file_hash(idea_path):
                    declaration_mismatches.append((fam, cid))
        else:
            declaration_mismatches.append(("registration_missing", reg_id))

    missing = sorted(expected_cells - complete_cells)
    published_source_current = (
        not unavailable
        and not non_complete
        and not missing
        and len(cohorts) == 1
        and not contended_cells
        and not unpaired_tasks
        and not declaration_mismatches
    )
    return {
        "current": published_source_current and not stale_source_cells,
        "published_source_current": published_source_current,
        "expected_cells": len(expected_cells),
        "complete_cells": len(complete_cells),
        "missing_cells": len(missing),
        "unavailable_cells": len(unavailable),
        "non_complete_cells": non_complete,
        "table_cohorts": len(cohorts),
        "contended_cells": len(contended_cells),
        "unpaired_tasks": sorted(unpaired_tasks),
        "stale_source_cells": len(stale_source_cells),
        "declaration_mismatches": declaration_mismatches,
    }


def check_retained_diagnostic_integrity(root: Path) -> dict[str, Any]:
    """Check retained observation bookkeeping without asserting formula faithfulness."""
    root = Path(root).resolve()
    pub_path = root / "reports/forge/technique-inventory.json"
    cov_path = root / "reports/forge/attempt-coverage.json"
    if not pub_path.is_file() or not cov_path.is_file():
        return {"current": False, "reason": "missing_report"}
    publication = read_json(pub_path)
    coverage = read_json(cov_path)
    recorded_hash = publication.get("family_progress", {}).get("input_hashes", {}).get(
        "reports/forge/attempt-coverage.json")
    if recorded_hash is not None and file_hash(cov_path) != recorded_hash:
        return {"current": False, "scope": "retained_diagnostic_integrity", "reason": "coverage_digest_changed"}
    summary = coverage.get("summary", {})
    pending = int(summary.get("pending", 0))
    running = int(summary.get("running", 0))

    full_suite_card = root / FIXED_TEST_LIST
    full_suite_names = (
        sorted({*read_json(full_suite_card)["tests"], "ring_hold"})
        if full_suite_card.is_file()
        else None
    )
    current_scope_names = set(full_suite_names or [])
    # This integrity stage checks the source-bound published coverage, not
    # experiments introduced after that source was frozen. Surface new cells
    # separately; completing them requires new compatible evidence, never a
    # relabel or rerun of the archived source merely to refresh a report.
    unmeasured_current_tasks = []
    scope = publication.get("family_progress", {}).get("recorded_display_scope")
    if scope is not None:
        import hashlib
        text = scope.get("text")
        if (not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != scope.get("sha256")
                or publication["family_progress"].get("input_hashes", {}).get(scope.get("path")) != scope.get("sha256")):
            raise ValueError("recorded display scope digest changed")
        recorded_names = sorted({*json.loads(text)["tests"], "ring_hold"})
        unmeasured_current_tasks = sorted(set(full_suite_names or []) - set(recorded_names))
        full_suite_names = recorded_names
    valid_statuses = attempt_coverage.DONE

    selected = attempt_coverage.selected_tasks(publication)
    expected_selected: dict[str, tuple[str, str, str, list[str]]] = {}
    commit_tasks_cache: dict[str, set[str]] = {}
    for family, selection, tasks in selected:
        if full_suite_names is not None:
            available = full_suite_names if scope is not None else sorted({*full_suite_names, *tasks})
        else:
            commit = _selected_family_source_commit(root, publication, family, selection)
            if commit not in commit_tasks_cache:
                commit_tasks_cache[commit] = _git_task_names_at_commit(root, commit)
            available = [task for task in tasks if task in commit_tasks_cache[commit]]
        expected_selected[family] = (
            selection["candidate_id"],
            selection["candidate_revision"],
            selection["bindings"]["source_digest"],
            available,
        )

    retained_families = {
        record.get("family")
        for record in coverage.get("records", [])
        if not _is_scoped_attempt(record) and isinstance(record.get("family"), str)
    }
    unmeasured_new_families = (
        sorted(set(expected_selected) - retained_families)
        if retained_families
        else []
    )

    superseded_records = []
    covered_by_family: dict[str, set[str]] = {family: set() for family in expected_selected}
    for record in coverage.get("records", []):
        if _is_scoped_attempt(record):
            continue
        family = record.get("family")
        spec = expected_selected.get(family)
        if spec is None or (
            record.get("candidate_id"),
            record.get("candidate_revision"),
            record.get("source_digest"),
        ) != spec[:3]:
            superseded_records.append(
                {
                    "family": family,
                    "candidate_id": record.get("candidate_id"),
                    "task": record.get("task"),
                }
            )
            continue
        if record.get("attempted") and record.get("status") in valid_statuses:
            covered_by_family[family].add(record["task"])
        elif full_suite_names is not None and record.get("task") in full_suite_names:
            superseded_records.append(
                {
                    "family": family,
                    "candidate_id": record.get("candidate_id"),
                    "task": record.get("task"),
                }
            )

    missing_by_family = {}
    for family, (cid, _, _, available) in sorted(expected_selected.items()):
        if family in unmeasured_new_families:
            continue
        missing = sorted(set(available) - covered_by_family.get(family, set()))
        if missing:
            missing_by_family[family] = {"candidate_id": cid, "missing_tasks": missing}
    if scope is not None:
        retained_covered = [
            covered
            for fam, covered in covered_by_family.items()
            if fam not in unmeasured_new_families
        ]
        unmeasured_current_tasks = sorted(
            name
            for name in current_scope_names
            if any(name not in covered for covered in retained_covered)
        )

    current = pending == 0 and running == 0 and not superseded_records and not missing_by_family
    return {
        "current": current,
        "scope": "retained_diagnostic_integrity",
        "qualification_input": False,
        "qualification_reuse": False,
        "pending": pending,
        "running": running,
        "superseded_record_count": len(superseded_records),
        "missing_by_family": missing_by_family,
        "unmeasured_current_tasks": unmeasured_current_tasks,
        "unmeasured_new_families": unmeasured_new_families,
    }


def _coverage_producer_errors(root: Path, record: dict, request: dict, task: dict, job: dict) -> list[str]:
    """Verify the witness against its durable producer, never a display label.

    Producer receipts retain their original bytes. Reading and independently
    grading current-source raw evidence does not rewrite a historical result.
    Task graders own full-budget/cadence/stopping requirements; a generic step
    comparison would incorrectly accept partial traces or reject valid stops.
    """
    from experiments.forge.views import grade_result

    producer = record.get("producer", {})
    if not isinstance(producer, dict):
        return ["missing_or_invalid_producer_receipt"]
    try:
        attempt_id = identifier(producer.get("attempt_id"), "producer attempt")
        directory = root / "reports/forge/attempts" / attempt_id
        resolved = read_json(directory / "request.json")
        result = read_json(directory / "result.json")
        certificate = read_json(directory / "evidence.json")
    except (OSError, ValueError, TypeError):
        return ["missing_or_invalid_producer_receipt"]
    if not all(isinstance(value, dict) for value in (resolved, result, certificate)):
        return ["missing_or_invalid_producer_receipt"]
    frozen = resolved.get("request", resolved)
    digest = stable_hash(result)
    if (producer.get("result_hash") != digest or certificate.get("result_hash") != digest
            or certificate.get("source") != frozen.get("source")
            or certificate.get("runtime") != frozen.get("runtime")
            or result.get("attempt_id") != attempt_id
            or result.get("candidate_revision") != request["candidate_revision"]
            or result.get("retry_of") != resolved.get("retry_of")):
        return ["invalid_producer_certificate"]
    if (any(frozen.get(key) != request.get(key) for key in
            ("candidate_revision", "runtime", "protocol"))
            or frozen.get("source", {}).get("digest") != request["source"]["digest"]
            or frozen.get("candidate", {}).get("id") != request["candidate"]["id"]):
        return ["stale_producer_contract"]
    name = task["id"]
    frozen_task = frozen.get("tasks", {}).get(name)
    if (not frozen_task or task_execution_fingerprint(frozen_task) != task_execution_fingerprint(task)
            or task_evaluation_fingerprint(frozen_task) != task_evaluation_fingerprint(task)):
        return ["stale_producer_task"]
    producer_jobs = [item for item in frozen.get("jobs", [])
                     if name in item.get("task_ids", [item.get("task_id")])]
    if (len(producer_jobs) != 1 or not job.get("science")
            or job.get("compatibility_key") != stable_hash(job["science"])
            or producer_jobs[0].get("science") != job["science"]
            or producer_jobs[0].get("compatibility_key") != job["compatibility_key"]):
        return ["producer_job_mismatch"]
    rows = [row for row in result.get("task_results", []) if row.get("task_id") == name]
    if (len(rows) != 1 or rows[0].get("compatibility_key") != job["compatibility_key"]
            or rows[0].get("observed_contract") != record["observed_contract"]
            or rows[0].get("gate_status") != record["status"]
            or rows[0].get("raw_status") != "completed"):
        return ["producer_observation_mismatch"]
    raw = result.get("raw", {})
    raw_result = raw.get("result", {})
    if raw.get("attempt_status") != "completed" or not isinstance(raw_result, dict) or not raw_result:
        return ["missing_completed_producer_evidence"]
    member_raw = raw_result.get("task_results", {}).get(name, raw_result)
    applied = member_raw.get("applied", member_raw)
    recipe = applied.get("recipe_serialized", applied.get("recipe"))
    if (not recipe or stable_hash(recipe) != stable_hash(record["observed_contract"]["resolved_recipe"])
            or applied.get("initializer") != record["observed_contract"]["initializer"]
            or applied.get("rng", {}).get("seed") != request["protocol"]["seed"]):
        return ["producer_applied_contract_mismatch_or_missing"]
    if frozen.get("requires_independent_grading"):
        grading = raw.get("grading", {})
        if (grading.get("raw_hash") != stable_hash(raw_result)
                or grading.get("source_digest") != request["source"]["digest"]):
            return ["invalid_producer_grading_certificate"]
        grade = grading.get("grades", {}).get(name, {})
    else:
        continuation = member_raw.get("evidence", {}).get("continuity", {}).get("diagnostic_continuation") is True
        grade = grade_result(task, member_raw, diagnostic=continuation)
    if not isinstance(grade, dict) or grade.get("gate_status", grade.get("status")) != record["status"]:
        return ["producer_grade_mismatch_or_incomplete"]
    return []


def _coverage_observation_errors(root: Path, record: dict, request: dict, task: dict, job: dict) -> list[str]:
    """Compare observed execution witnesses to the current frozen contract.

    Compact historical rows lacking a witness stay useful diagnostics. Expected
    declarations cannot be copied into a row to reconstruct missing observations.
    """
    from experiments.forge.api import task_formulation_context
    from experiments.forge.initialization import task_initializer
    from experiments.forge.publication_integrity import coverage_disposition, IDENTITY_FIELDS
    from experiments.forge.sampling import validate_declaration

    errors = []
    try:
        if coverage_disposition(root, record):
            errors.append("corrected_archived_diagnostic")
    except ValueError:
        errors.append("altered_corrected_observation")
    if record.get("key") != stable_hash([record.get(field) for field in IDENTITY_FIELDS]):
        errors.append("invalid_scientific_key")
    for field, expected in (
        ("candidate_id", request["candidate"]["id"]),
        ("candidate_revision", request["candidate_revision"]),
        ("source_digest", request["source"]["digest"]),
        ("execution_sha256", task_execution_fingerprint(task)),
        ("evaluation_sha256", task_evaluation_fingerprint(task)),
    ):
        if record.get(field) != expected:
            errors.append(field + "_mismatch")
    if not record.get("attempted") or record.get("status") not in {"PASS", "FAIL"}:
        errors.append("no_completed_numerical_measurement")
    witness = record.get("observed_contract")
    if not isinstance(witness, dict) or witness.get("schema_version") != 1:
        return errors + ["missing_observed_contract"]
    if not job:
        return errors + ["missing_current_job_binding"]
    science = job.get("science", {})
    protocol = request.get("protocol")
    if (not protocol or type(protocol.get("seed")) is not int
            or witness.get("protocol") != protocol
            or science.get("protocol") != protocol or science.get("seed") != protocol["seed"]):
        errors.append("protocol_seed_mismatch_or_missing")
    try:
        initializer = task_initializer(task, request["candidate"])
    except (ValueError, KeyError, TypeError):
        errors.append("unresolved_declared_initializer")
    else:
        if (witness.get("initializer") != initializer
                or science.get("task_initializers", {}).get(task["id"]) != initializer):
            errors.append("initializer_mismatch_or_missing")
    for field in ("execution", "evaluation"):
        if not task.get(field) or witness.get(field) != task[field]:
            errors.append(field + "_contract_mismatch_or_missing")
    runtime = science.get("runtime", request.get("runtime"))
    compute = science.get("compute")
    if not runtime or witness.get("runtime") != runtime or record.get("runtime") != runtime:
        errors.append("runtime_mismatch_or_missing")
    if not compute or witness.get("compute") != compute:
        errors.append("compute_mismatch_or_missing")
    try:
        expected_sampling = validate_declaration(task, required=True)
    except (ValueError, KeyError, TypeError):
        return errors + ["unresolved_declared_sampling"]
    expected_sampling["scoring_weights"] = task.get("evaluation", {}).get("scoring_weights", "live")
    if witness.get("sampling") != expected_sampling:
        errors.append("sampling_mismatch_or_missing")
    if not isinstance(witness.get("resolved_recipe"), dict) or not witness["resolved_recipe"]:
        errors.append("missing_observed_recipe")
    elif not errors:
        # Resolve only for an otherwise compatible observation. This constructs
        # no networks, optimizers or training run, and never repairs a recipe.
        try:
            context = task_formulation_context(request["candidate"], task, request["protocol"],
                                               device="cpu", root=root)
            if stable_hash(witness["resolved_recipe"]) != stable_hash(context.recipe.to_dict()):
                errors.append("resolved_recipe_mismatch")
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            errors.append("unresolved_recipe:" + type(exc).__name__)
    if not errors:
        errors.extend(_coverage_producer_errors(root, record, request, task, job))
    return errors


def check_attempt_coverage(root: Path) -> dict[str, Any]:
    """Check faithful current-source measurements, separately from archive integrity."""
    root = Path(root).resolve()
    path = root / "reports/forge/attempt-coverage.json"
    if not path.is_file():
        return {"current": False, "scope": "current_source_measurements", "reason": "missing_report"}
    coverage = read_json(path)
    names_path = root / FIXED_TEST_LIST
    names = set(read_json(names_path)["tests"]) | {"ring_hold"} if names_path.is_file() else set()
    records = coverage.get("records", [])
    missing_by_family, invalid_counts, examples, binding_errors = {}, Counter(), [], []
    accepted_count = 0
    optional_precision_tasks = set()
    with planning.resolution_cache(root):
        for selection in current_family_candidates(root):
            family, cid = selection["family"], selection["candidate_id"]
            try:
                from reports.forge.regenerate_technique_inventory import _resolve_idea_for_readout_check
                request = _resolve_idea_for_readout_check(
                    root, cid, view_id="discriminator_stability", through_tier=_discriminator_stability_max_tier(root),
                    cuda_model=DEFAULT_CUDA_MODEL, planning_mod=planning)
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                binding_errors.append({"family": family, "candidate_id": cid, "error": str(exc)})
                missing_by_family[family] = {"candidate_id": cid, "missing_tasks": sorted(names)}
                continue
            optional_precision = {assignment["task"] for assignment in request.get("view", {}).get("assignments", [])
                                  if assignment.get("importance") != "required"
                                  and request["tasks"].get(assignment["task"], {}).get("execution", {}).get("linear_precision")}
            required = (names or set(request["tasks"])) - optional_precision
            optional_precision_tasks.update(optional_precision)
            jobs = {tid: job for job in request.get("jobs", [])
                    for tid in job.get("task_ids", [job["task_id"]])}
            accepted, observations = set(), {}
            for record in records:
                if _is_scoped_attempt(record) or record.get("family") != family or record.get("task") not in required:
                    continue
                task_name = record["task"]
                task = request["tasks"].get(task_name)
                errors = (["missing_current_task_binding"] if task is None else
                          _coverage_observation_errors(root, record, request, task, jobs.get(task_name, {})))
                if errors:
                    invalid_counts.update(errors)
                    if len(examples) < 12:
                        examples.append({"family": family, "task": task_name, "reasons": errors})
                    continue
                digest = stable_hash(record)
                if task_name in observations and observations[task_name] != digest:
                    accepted.discard(task_name)
                    invalid_counts["conflicting_observations"] += 1
                    observations[task_name] = None
                elif task_name not in observations:
                    observations[task_name] = digest
                    accepted.add(task_name)
            accepted_count += len(accepted)
            missing = sorted(required - accepted)
            if missing:
                missing_by_family[family] = {"candidate_id": cid, "missing_tasks": missing}
    summary = coverage.get("summary", {})
    pending, running = int(summary.get("pending", 0)), int(summary.get("running", 0))
    return {"current": not missing_by_family and not binding_errors and pending == 0 and running == 0,
            "scope": "current_source_measurements", "qualification_reuse": False,
            "pending": pending, "running": running, "faithful_measured_cells": accepted_count,
            "missing_by_family": missing_by_family, "binding_errors": binding_errors,
            "optional_precision_tasks": sorted(optional_precision_tasks),
            "invalid_record_counts": dict(sorted(invalid_counts.items())), "gap_examples": examples,
            "note": "Retained diagnostic integrity is checked separately. Missing observed execution witnesses "
                    "remain measurement gaps; declarations and archived verdicts cannot reconstruct them."}


def _concluded_ordinary_records(root: Path) -> list[dict[str, Any]]:
    records_dir = Path(root) / "reports/forge/records"
    if not records_dir.is_dir():
        return []
    result = []
    for path in sorted(records_dir.glob("readout-*.json")):
        doc = read_json(path)
        if doc.get("lifecycle") == "concluded" and doc.get("evidence_scope") == "current":
            result.append(doc)
    return result


def _discriminator_stability_max_tier(root: Path) -> int:
    view_path = Path(root) / "configs/forge/views/discriminator_stability.json"
    if not view_path.is_file():
        return 4
    assignments = read_json(view_path).get("assignments", [])
    return max((int(item["qualification_tier"]) for item in assignments), default=2)


def check_selected_family_readouts(root: Path, *, cuda_model: str = DEFAULT_CUDA_MODEL) -> dict[str, Any]:
    from reports.forge.regenerate_technique_inventory import (
        _ReadoutCompatibilityMatcher,
        _load_attempt_coverage_task_fps,
        _resolve_idea_for_readout_check,
    )

    root = Path(root).resolve()
    concluded = _concluded_ordinary_records(root)
    coverage_fps = _load_attempt_coverage_task_fps(root)
    coverage_verdicts = {}  # Diagnostic coverage never supplies ordinary task measurements.
    shared_attempt_cache: dict[str, dict[str, Any] | None] = {}
    leaderboard_texts = {}
    for view_id in ORDINARY_LEADERBOARD_VIEWS:
        lb_path = root / "reports/forge/leaderboards" / f"{view_id}.md"
        leaderboard_texts[view_id] = lb_path.read_text() if lb_path.is_file() else ""

    requested_max_tier = _discriminator_stability_max_tier(root)
    missing_families = []
    stale_leaderboards = []
    local_attempts_missing_families = []
    with planning.resolution_cache(root):
        for choice in current_family_candidates(root):
            family, cid = choice["family"], choice["candidate_id"]
            try:
                req = _resolve_idea_for_readout_check(
                    root,
                    cid,
                    view_id="discriminator_stability",
                    through_tier=requested_max_tier,
                    cuda_model=cuda_model,
                    planning_mod=planning,
                )
            except Exception as exc:
                raise CodeChangeRequiredError(f"cannot resolve selected candidate {cid}: {exc}") from exc

            assignments = req["view"]["assignments"]
            fam_cov = coverage_verdicts.get(family, {})

            def _is_optional_precision_tasks(item: dict[str, Any]) -> bool:
                tid = item["task"]
                return (
                    item.get("importance") != "required"
                    and tid not in fam_cov
                    and bool((req["tasks"].get(tid, {}).get("execution") or {}).get("linear_precision"))
                )

            t1_assignments = [item for item in assignments if int(item["qualification_tier"]) == 1]
            t1_runnable = [
                item["task"]
                for item in t1_assignments
                if not req["tasks"][item["task"]].get("preflight_blockers")
                and not _is_optional_precision_tasks(item)
            ]
            if not t1_runnable:
                continue

            compat_matcher = _ReadoutCompatibilityMatcher(
                root,
                req,
                coverage_fps=coverage_fps,
                planning_mod=planning,
                attempt_cache=shared_attempt_cache,
            )
            rev = req["candidate_revision"]
            req_source_digest = (req.get("source") or {}).get("digest")
            matching = []
            for rec in concluded:
                scope = (rec.get("study") or {}).get("scope") or {}
                if (
                    rec.get("candidate_id") == cid
                    and rec.get("candidate_revision") == rev
                    and (
                        not req_source_digest
                        or (rec.get("source") or {}).get("digest") in {None, req_source_digest}
                    )
                    and (scope.get("view") or rec.get("goal")) == "discriminator_stability"
                    and scope.get("execution_backend", "cuda") == "cuda"
                    and scope.get("cuda_model") in {None, cuda_model}
                ):
                    matching.append(rec)
            measured_statuses: dict[str, str] = {}
            unresolved_errors: dict[str, str] = {}
            matched_attempt_ids: set[str] = set()
            for rec in matching:
                rec_used = False
                rec_row_attempts: set[str] = set()
                for row in rec.get("task_results", []):
                    tid = str(row.get("task_id", ""))
                    if not compat_matcher.matches(rec, row, tid):
                        continue
                    st = row.get("gate_status") or row.get("status")
                    if st in {"PASS", "FAIL", "BLOCKED"}:
                        measured_statuses[tid] = st
                        unresolved_errors.pop(tid, None)
                        rec_used = True
                        row_aid = row.get("_attempt_id") or row.get("attempt_id")
                        if row_aid:
                            rec_row_attempts.add(str(row_aid))
                    elif st in {"ERROR", "INVALID", "INCOMPLETE"} and tid not in measured_statuses:
                        unresolved_errors[tid] = st
                if rec_used:
                    if rec_row_attempts:
                        matched_attempt_ids.update(rec_row_attempts)
                    else:
                        matched_attempt_ids.update(str(aid) for aid in rec.get("attempt_ids", []) if aid)

            if unresolved_errors:
                bad_tid, bad_st = sorted(unresolved_errors.items())[0]
                raise CodeChangeRequiredError(
                    f"selected candidate {cid} has {bad_st} on {bad_tid} "
                    f"in current-source readout"
                )

            def _dep_blocked(task_id: str, seen: frozenset[str] = frozenset()) -> bool:
                if task_id in seen:
                    return False
                next_seen = seen | {task_id}
                for dep in req["tasks"].get(task_id, {}).get("dependencies", []):
                    dep_task = dep["task"] if isinstance(dep, dict) else str(dep)
                    if req["tasks"].get(dep_task, {}).get("preflight_blockers"):
                        return True
                    if measured_statuses.get(dep_task) in {"FAIL", "BLOCKED"}:
                        return True
                    if _dep_blocked(dep_task, next_seen):
                        return True
                return False

            view_max_tier = max((int(item["qualification_tier"]) for item in assignments), default=1)
            unmeasured_by_tier: dict[int, list[str]] = {}
            tier_gate_open = True
            for tier in range(1, view_max_tier + 1):
                if not tier_gate_open:
                    break
                tier_assignments = [
                    item for item in assignments if int(item["qualification_tier"]) == tier
                ]
                tier_runnable = [
                    item["task"]
                    for item in tier_assignments
                    if not req["tasks"][item["task"]].get("preflight_blockers")
                    and not _dep_blocked(item["task"])
                    and not _is_optional_precision_tasks(item)
                ]
                unmeasured = [task for task in tier_runnable if task not in measured_statuses]
                if unmeasured:
                    unmeasured_by_tier[tier] = unmeasured
                tier_required = [
                    item["task"] for item in tier_assignments if item.get("importance") == "required"
                ]
                tier_gate_open = (not unmeasured) and all(
                    not req["tasks"][task].get("preflight_blockers")
                    and measured_statuses.get(task) == "PASS"
                    for task in tier_required
                )

            if (
                not matched_attempt_ids
                or not all(
                    (root / "reports/forge/attempts" / aid / "result.json").is_file()
                    and (compat_matcher._load_attempt_request(aid) or {}).get("candidate_revision") == rev
                    for aid in matched_attempt_ids
                )
            ):
                local_attempts_missing_families.append(
                    {"family": family, "candidate_id": cid, "candidate_revision": rev}
                )

            if not matching or unmeasured_by_tier:
                missing_families.append(
                    {
                        "family": family,
                        "candidate_id": cid,
                        "candidate_revision": rev,
                        "unmeasured_tier1": unmeasured_by_tier.get(1, []),
                        "unmeasured_tier2": unmeasured_by_tier.get(2, []),
                        "unmeasured_by_tier": {
                            str(tier): tasks for tier, tasks in sorted(unmeasured_by_tier.items())
                        },
                    }
                )
                continue

            rev_short = rev[:12]
            needle = f"/ {rev_short} / cuda ({cuda_model}) | current |"
            for view_id, text in leaderboard_texts.items():
                matched_lines = [line for line in text.splitlines() if needle in line]
                if not matched_lines or any("| unknown |" in line for line in matched_lines):
                    stale_leaderboards.append({"family": family, "candidate_id": cid, "view": view_id})

    return {
        "current": not missing_families and not stale_leaderboards,
        "missing_families": missing_families,
        "stale_leaderboards": stale_leaderboards,
        "local_attempts_missing_families": local_attempts_missing_families,
    }


def _expected_selected_diagnostic_phrases(root: Path, inv_json: dict[str, Any]) -> list[str]:
    progress = inv_json.get("family_progress") or {}
    preflight_path = Path(root) / "reports/forge/tier-coverage-20261014/preflight.json"
    if not progress or not preflight_path.is_file():
        return []
    from experiments.forge.family_reports import _selected_cohorts, _selected_row, _summed_views
    from experiments.forge.policy_coverage import PARENTS

    pins = {item["family"]: item for item in read_json(preflight_path).get("families", [])}
    lowest_tier: dict[str, int] = {}
    for view in _summed_views(progress):
        for assignment in view.get("assignments", []):
            if assignment.get("importance") in COUNTED_OR_HELD or "removal" in assignment:
                tier = int(assignment["qualification_tier"])
                task = assignment["task"]
                lowest_tier[task] = min(tier, lowest_tier.get(task, tier))
    declared_tasks = {task for task, assigned in lowest_tier.items() if assigned in (1, 2, 3)}
    extra_tasks = declared_tasks - set(PARENTS)

    phrases: list[str] = []
    for family, cohort in _selected_cohorts(progress):
        row = _selected_row(inv_json, cohort)
        candidate_id = row.get("candidate_id")
        if not candidate_id or pins.get(family["id"], {}).get("candidate_id") == candidate_id:
            continue
        tasks_map = cohort.get("tasks", {})
        recorded = {
            name: info["status"]
            for name, info in tasks_map.items()
            if info.get("status") in {"PASS", "FAIL"}
        }
        attempts = {
            name: info["run_attempt"]
            for name, info in tasks_map.items()
            if isinstance(info.get("run_attempt"), dict)
        }
        if not any(attempt.get("attempted") and name not in recorded for name, attempt in attempts.items()):
            continue
        phrases.append(f"Selected recipe 1-for-1 baseline-35 diagnostics ({candidate_id}); seed 0")
        if any(task in recorded or attempts.get(task, {}).get("attempted") for task in extra_tasks):
            phrases.append(f"Selected recipe declared-suite Tier 1–3 diagnostics ({candidate_id}); seed 0")
    return phrases


def _attempt_coverage_markdown_synced(root: Path, cov: dict[str, Any]) -> bool:
    md_path = Path(root) / "reports/forge/ATTEMPT_COVERAGE.md"
    if not md_path.is_file():
        return False
    md_text = md_path.read_text()
    records = cov.get("records", [])
    for family in sorted({rec["family"] for rec in records if "family" in rec}):
        fam_records = [rec for rec in records if rec.get("family") == family]
        counts = Counter(rec["status"] for rec in fam_records)
        label = fam_records[0].get("label", family)
        expected_row = (
            f"| {label} | {sum(bool(rec.get('attempted')) for rec in fam_records)}/{len(fam_records)} | "
            f"{counts.get('PENDING', 0)} | "
            + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
            + " |"
        )
        if expected_row not in md_text:
            return False
    return True


def _verified_current_publication(doc: Any, *, view_id: str | None = None) -> bool:
    if not isinstance(doc, dict):
        return False
    if doc.get("publication_scope") != "current_technique_inventory" or doc.get("recorded_policy"):
        return False
    if view_id is not None and doc.get("view") != view_id:
        return False
    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        return False
    checked = deepcopy(doc)
    claimed = checked["provenance"].pop("input_digest", None)
    return isinstance(claimed, str) and claimed == stable_hash(checked)


def _git_head_verified_publication_bytes(root: Path, *, view_id: str | None = None) -> bytes | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "show", "HEAD:reports/forge/technique-inventory.json"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        doc = json.loads(proc.stdout.decode("utf-8"))
        if _verified_current_publication(doc, view_id=view_id):
            return proc.stdout
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError, ValueError):
        return None
    return None


def _registered_evidence_satisfies_selection(root: Path) -> bool:
    root = Path(root).resolve()
    sel_path = root / CURRENT_SELECTION
    manifest_path = root / "reports/forge/technique-evidence/manifest.json"
    if not sel_path.is_file() or not manifest_path.is_file():
        return False
    try:
        from experiments.forge.trainer_families import load_current_selection, scientific_row_hash
        from reports.forge import regenerate_technique_inventory as rti

        manifest = read_json(manifest_path)
        cohorts = manifest.get("cohorts") or []
        view_id = manifest.get("view")
        policy_fp = manifest.get("policy_fingerprint")
        view_rev = manifest.get("view_revision")
        if (
            manifest.get("schema_version") != 1
            or not cohorts
            or not isinstance(view_id, str)
            or not isinstance(policy_fp, str)
        ):
            return False
        live_view_path = root / "configs/forge/views" / f"{view_id}.json"
        if not live_view_path.is_file():
            return False
        live_view = read_json(live_view_path)
        if stable_hash(live_view) != policy_fp:
            frozen_view_path = root / "configs/forge/view-history" / f"{view_id}-v{view_rev}.json"
            if not frozen_view_path.is_file():
                return False
            frozen_view = read_json(frozen_view_path)
            if (
                frozen_view.get("id") != view_id
                or frozen_view.get("revision") != view_rev
                or stable_hash(frozen_view) != policy_fp
            ):
                return False
        registered_hashes: set[str] = set()
        for entry in cohorts:
            _, rows = rti._snapshot(root, entry, manifest)
            registered_hashes.update(scientific_row_hash(row) for row in rows.values())
        archived_hashes: set[str] = set()
        for _, _, _, rows in rti._archived_reports(root, manifest):
            archived_hashes.update(scientific_row_hash(row) for row in rows.values())
        pins = load_current_selection(
            root,
            view_id=view_id,
            policy_fingerprint=policy_fp,
            selection_card=read_json(sel_path),
        )
        if not pins:
            return False
        for pin in pins.values():
            row_sha = pin.get("scientific_row_sha256")
            permitted = registered_hashes | archived_hashes if pin.get("historical_row") else registered_hashes
            if not row_sha or row_sha not in permitted:
                return False
        has_verified_baseline = False
        inv_json_path = root / "reports/forge/technique-inventory.json"
        if inv_json_path.is_file():
            try:
                has_verified_baseline = _verified_current_publication(read_json(inv_json_path), view_id=view_id)
            except Exception:
                has_verified_baseline = False
        if not has_verified_baseline:
            has_verified_baseline = _git_head_verified_publication_bytes(root, view_id=view_id) is not None
        if not has_verified_baseline and any(pin.get("selection_kind") == "current_measurement" for pin in pins.values()):
            live_src = inspect_source(root)["digest"]
            if any(
                pin.get("selection_kind") == "current_measurement" and pin.get("source_digest") != live_src
                for pin in pins.values()
            ):
                return False
        return True
    except Exception:
        return False


def _validate_publication_registration_sync(root: Path, inv_json: dict[str, Any]) -> bool:
    provenance = inv_json.get("provenance") or {}
    sel_path = Path(root) / CURRENT_SELECTION
    sel_mismatch = (
        sel_path.is_file()
        and provenance.get("family_current_selection_sha256") is not None
        and file_hash(sel_path) != provenance.get("family_current_selection_sha256")
    )
    manifest_path = Path(root) / "reports/forge/technique-evidence/manifest.json"
    manifest_mismatch = (
        manifest_path.is_file()
        and provenance.get("evidence_manifest_sha256") is not None
        and stable_hash(read_json(manifest_path)) != provenance.get("evidence_manifest_sha256")
    )
    if not sel_mismatch and not manifest_mismatch:
        if _verified_current_publication(inv_json):
            return True
        if _registered_evidence_satisfies_selection(Path(root)):
            return False
        raise CodeChangeRequiredError(
            "reports/forge/technique-inventory.json failed publication input digest verification and "
            "cannot be regenerated from registered snapshots"
        )
    if _registered_evidence_satisfies_selection(Path(root)):
        return False
    if sel_mismatch:
        raise CodeChangeRequiredError(
            f"{CURRENT_SELECTION.as_posix()} differs from reports/forge/technique-inventory.json; "
            "register the promoted family selection snapshot with "
            "`python reports/forge/regenerate_technique_inventory.py --source-commit <executed-commit>`"
        )
    raise CodeChangeRequiredError(
        "reports/forge/technique-evidence/manifest.json differs from "
        "reports/forge/technique-inventory.json; regenerate the registered inventory first"
    )


def _republish_registered_inventory(root: Path, *, json_mode: bool = False) -> dict[str, Any]:
    from reports.forge import regenerate_technique_inventory as rti

    root = Path(root).resolve()
    inv_json_path = root / "reports/forge/technique-inventory.json"
    manifest_path = root / "reports/forge/technique-evidence/manifest.json"
    cur_doc: dict[str, Any] | None = None
    if inv_json_path.is_file():
        try:
            loaded = read_json(inv_json_path)
            if isinstance(loaded, dict):
                cur_doc = loaded
        except ValueError:
            cur_doc = None

    if cur_doc is None or not _verified_current_publication(cur_doc):
        expected_view = None
        if manifest_path.is_file():
            try:
                expected_view = str(read_json(manifest_path).get("view") or "") or None
            except ValueError:
                expected_view = None
        head_bytes = _git_head_verified_publication_bytes(root, view_id=expected_view)
        if head_bytes is not None:
            _log(
                "Restoring verified baseline technique-inventory.json from HEAD before publication sync",
                json_mode=json_mode,
            )
            inv_json_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_text(inv_json_path, head_bytes.decode("utf-8"))
            cur_doc = read_json(inv_json_path)

    needs_republish = False
    if cur_doc is not None:
        needs_republish = not _validate_publication_registration_sync(root, cur_doc)
    elif manifest_path.is_file():
        if not _registered_evidence_satisfies_selection(root):
            raise CodeChangeRequiredError(
                "reports/forge/technique-evidence/manifest.json cannot satisfy current family selection without "
                "new snapshot registration"
            )
        needs_republish = True

    inv_md_path = root / "reports/forge/technique-inventory.md"
    if not inv_md_path.is_file():
        inv_md_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(inv_md_path, "# Forge technique inventory\n")

    try:
        if needs_republish:
            manifest = read_json(manifest_path)
            view_id = str(manifest["view"])
            live_view_path = root / "configs/forge/views" / f"{view_id}.json"
            live_differs = stable_hash(read_json(live_view_path)) != manifest.get("policy_fingerprint")
            comparison_policy = (
                root / "configs/forge/view-history" / f"{view_id}-v{manifest['view_revision']}.json"
                if live_differs
                else None
            )
            _log(
                f"Republishing technique-inventory.json/.md from registered evidence snapshots (view={view_id})",
                json_mode=json_mode,
            )
            summary = rti.publish_current(root, view_id=view_id, comparison_policy=comparison_policy)
            if live_differs:
                summary = rti.refresh_publication(root, view_id=view_id)
            return summary
        return rti.refresh_publication(root)
    except ValueError as exc:
        raise CodeChangeRequiredError(
            f"cannot refresh technique-inventory publication without snapshot registration: {exc}"
        ) from exc


def _generated_pages_synced(root: Path, inv_json: dict[str, Any], inv_md: str) -> bool:
    root = Path(root).resolve()
    prov = inv_json.get("provenance") or {}
    input_digest = prov.get("input_digest")
    if not input_digest or f"Publication input digest `{input_digest}`." not in inv_md:
        return False
    reducer_path = root / "reports/forge/regenerate_technique_inventory.py"
    if (
        reducer_path.is_file()
        and prov.get("publication_reducer_sha256") is not None
        and prov.get("publication_reducer_sha256") != file_hash(reducer_path)
    ):
        return False
    registry_path = root / "configs/forge/trainer-families.json"
    if (
        registry_path.is_file()
        and prov.get("trainer_family_registry_sha256") is not None
        and prov.get("trainer_family_registry_sha256") != file_hash(registry_path)
    ):
        return False
    tier_page = root / "reports/forge/EXPERIMENTS_BY_TIER.md"
    if not tier_page.is_file():
        return False
    tier_text = tier_page.read_text()
    if (
        not tier_text.startswith("# Forge experiments by tier\n")
        or "Declaration input digest: `" not in tier_text
        or "Published artifact input digest: `" not in tier_text
    ):
        return False
    tp_json_path = root / "reports/forge/test-performance.json"
    tp_md_path = root / "reports/forge/test-performance.md"
    tp_html_path = root / "reports/forge/test-performance.html"
    if not tp_md_path.is_file() or not tp_md_path.read_text().startswith("# Which recipes pass each test?\n"):
        return False
    # test-performance.json and test-performance.html are gitignored offline artifacts;
    # validate them when present locally without failing clean git checkouts or CI runners.
    if tp_json_path.is_file():
        try:
            tp_json = read_json(tp_json_path)
        except Exception:
            return False
        if tp_json.get("source_publication_digest") != input_digest:
            return False
    if tp_html_path.is_file() and not tp_html_path.read_text().strip():
        return False
    if inv_json.get("family_progress"):
        from experiments.forge.family_reports import generated_pages
        from reports.forge import regenerate_technique_inventory as rti

        # Archived display scores still need an integrity check when the latest
        # source has no Tier 1 payload. A digest footer alone cannot detect a
        # changed overview score or a false source-qualification claim.
        if inv_md != rti._current_markdown(inv_json, root, root / "reports/forge/technique-inventory.md"):
            return False

        for page_path, expected_text in generated_pages(root, inv_json).items():
            if not page_path.is_file() or page_path.read_text() != expected_text:
                return False
    if inv_json.get("view") == "discriminator_stability":
        from reports.forge.regenerate_technique_inventory import _shared_score_intro

        shared = _shared_score_intro(root, inv_json)
        if shared is not None:
            shared_path, shared_text = shared
            if not Path(shared_path).is_file() or Path(shared_path).read_text() != shared_text:
                return False
    return True


def check_compiled_reports(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    mem_status = knowledge.freshness(root)
    inv_cov = history.validate_inventory(root)
    comp_path = root / "reports/forge/compilation.json"
    comp = read_json(comp_path) if comp_path.is_file() else {}
    clean_checkout_safe = not any(
        key.startswith("reports/forge/attempts/") for key in comp.get("input_hashes", {})
    )
    selected_cids = {choice["candidate_id"] for choice in current_family_candidates(root)}
    raw_pending = comp.get("pending_readout") if "pending_readout" in comp else None
    selected_pending_readout = (
        ["missing"]
        if raw_pending is None
        else [cid for cid in raw_pending if cid in selected_cids]
    )

    inv_json_path = root / "reports/forge/technique-inventory.json"
    inv_md_path = root / "reports/forge/technique-inventory.md"
    manifest_path = root / "reports/forge/technique-evidence/manifest.json"
    perf_path = root / "reports/forge/scaling-performance.json"
    cov_path = root / "reports/forge/attempt-coverage.json"

    publication_registration_synced = False
    perf_synced = False
    unknown_perf_rows = -1
    diagnostic_rows_present = False
    current_source_tier1_synced = False
    refresh_docs_synced = False
    attempt_summary_synced = False
    attempt_markdown_synced = False
    generated_pages_synced = False

    inv_json: dict[str, Any] | None = None
    if inv_json_path.is_file():
        try:
            loaded_inv = read_json(inv_json_path)
            if isinstance(loaded_inv, dict):
                inv_json = loaded_inv
            elif not _registered_evidence_satisfies_selection(root):
                raise CodeChangeRequiredError("invalid technique-inventory.json: top-level JSON must be an object")
        except ValueError as exc:
            if not _registered_evidence_satisfies_selection(root):
                raise CodeChangeRequiredError(f"invalid technique-inventory.json: {exc}") from exc
    elif manifest_path.is_file() and not _registered_evidence_satisfies_selection(root):
        raise CodeChangeRequiredError(
            "reports/forge/technique-evidence/manifest.json cannot satisfy current family selection without "
            "new snapshot registration"
        )

    if inv_json is not None:
        publication_registration_synced = _validate_publication_registration_sync(root, inv_json)
        if inv_md_path.is_file():
            inv_md = inv_md_path.read_text()
            if perf_path.is_file():
                perf_synced = (
                    inv_json.get("scaling_performance", {}).get("manifest_sha256")
                    == file_hash(perf_path)
                )
            if scaling_publication.START in inv_md and scaling_publication.END in inv_md:
                block = inv_md.split(scaling_publication.START, 1)[1].split(
                    scaling_publication.END, 1
                )[0]
                unknown_perf_rows = sum(1 for line in block.splitlines() if "| UNKNOWN |" in line)
            expected_diag_phrases = _expected_selected_diagnostic_phrases(root, inv_json)
            diagnostic_rows_present = all(phrase in inv_md for phrase in expected_diag_phrases)
            if inv_json.get("family_progress"):
                from reports.forge import regenerate_technique_inventory as rti

                current_source_tier1_synced = inv_md == rti._current_markdown(inv_json, root, inv_md_path)
                refresh_docs_synced = (
                    "python ops/refresh_forge_readouts.py --check" in inv_md
                    and "python ops/refresh_forge_readouts.py --reports-only" in inv_md
                )
            else:
                current_source_tier1_synced = True
                refresh_docs_synced = True
            if cov_path.is_file():
                cov_doc = read_json(cov_path)
                cov_sum = cov_doc.get("summary", {})
                expected_phrase = (
                    f"**{cov_sum.get('attempted')}/{cov_sum.get('total')} distinct recipe/test pairs attempted "
                    "(selected and research)**"
                )
                attempt_summary_synced = expected_phrase in inv_md
                attempt_markdown_synced = _attempt_coverage_markdown_synced(root, cov_doc)
            generated_pages_synced = _generated_pages_synced(root, inv_json, inv_md)

    current = (
        bool(mem_status.get("fresh"))
        and bool(inv_cov.get("valid"))
        and clean_checkout_safe
        and selected_pending_readout == []
        and publication_registration_synced
        and perf_synced
        and unknown_perf_rows == 0
        and diagnostic_rows_present
        and current_source_tier1_synced
        and refresh_docs_synced
        and attempt_summary_synced
        and attempt_markdown_synced
        and generated_pages_synced
    )
    return {
        "current": current,
        "memory_fresh": bool(mem_status.get("fresh")),
        "inventory_valid": bool(inv_cov.get("valid")),
        "clean_checkout_safe": clean_checkout_safe,
        "selected_pending_readout": selected_pending_readout,
        "publication_registration_synced": publication_registration_synced,
        "scaling_performance_synced": perf_synced,
        "unknown_performance_rows": unknown_perf_rows,
        "selected_diagnostic_rows_present": diagnostic_rows_present,
        "current_source_tier1_synced": current_source_tier1_synced,
        "refresh_docs_synced": refresh_docs_synced,
        "attempt_summary_synced": attempt_summary_synced,
        "attempt_markdown_synced": attempt_markdown_synced,
        "generated_pages_synced": generated_pages_synced,
    }


def check_status(
    root: Path,
    *,
    queue_root: Path | None = None,
    cuda_model: str = DEFAULT_CUDA_MODEL,
) -> dict[str, Any]:
    root = Path(root).resolve()
    queue_root = Path(queue_root or (root / "runs/forge")).resolve()
    stages = {
        "queue": check_queue_status(root, queue_root),
        "scaling_performance": check_scaling_performance(root),
        "attempt_coverage": check_attempt_coverage(root),
        "retained_diagnostic_integrity": check_retained_diagnostic_integrity(root),
        "selected_family_readouts": check_selected_family_readouts(root, cuda_model=cuda_model),
        "compiled_reports": check_compiled_reports(root),
    }
    stale = [name for name, info in stages.items() if not info.get("current")]
    return {
        "status": "CURRENT" if not stale else "NEEDS_REFRESH",
        "scope": "current_source_measurements",
        "source_digest": catalog_source_manifest(root)["digest"],
        "stale_stages": stale,
        "stages": stages,
    }


def check_published_status(
    root: Path,
    *,
    queue_root: Path | None = None,
    cuda_model: str = DEFAULT_CUDA_MODEL,
) -> dict[str, Any]:
    """Check retained publications while reporting latest-source gaps separately.

    Original receipts and registered selection snapshots must still validate.
    This scope never transfers their measurements onto different source bytes.
    """
    strict = check_status(root, queue_root=queue_root, cuda_model=cuda_model)
    stages = {name: deepcopy(strict["stages"][name])
              for name in ("queue", "compiled_reports")}
    stages["retained_diagnostic_integrity"] = deepcopy(
        strict["stages"]["retained_diagnostic_integrity"] if "retained_diagnostic_integrity" in strict["stages"]
        else check_retained_diagnostic_integrity(root))
    performance = deepcopy(strict["stages"]["scaling_performance"])
    performance["current"] = performance.get("published_source_current", False)
    performance["scope"] = "recorded_source_cohort"
    stages["published_scaling_performance"] = performance
    stages["registered_selected_publications"] = {
        "current": _registered_evidence_satisfies_selection(Path(root)),
        "scope": "registered_original_snapshots",
        "qualification_reuse": False,
    }
    stale = [name for name, info in stages.items() if not info.get("current")]
    return {
        "status": "CURRENT" if not stale else "NEEDS_REFRESH",
        "scope": "published_reports",
        "source_digest": strict["source_digest"],
        "stale_stages": stale,
        "stages": stages,
        "current_source_measurements": strict,
        "qualification_reuse": False,
        "note": "Publication integrity does not establish measurements for the latest source. "
                "Original source cohorts and qualifications remain unchanged; source drift does not authorize reruns.",
    }


def _conclude_queue_submissions(
    root: Path,
    queue_root: Path,
    *,
    campaign_id: str | None = None,
    json_mode: bool = False,
) -> list[str]:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return []
    state = read_json(state_path)
    concluded_records = []
    for req_id, entry in sorted(state.get("submissions", {}).items()):
        if entry.get("lifecycle") != "awaiting_readout":
            continue
        req = entry["request"]
        if campaign_id is not None and req.get("campaign_id") != campaign_id:
            continue
        if not _request_matches_worktree(root, req):
            raise CodeChangeRequiredError(
                f"submission {req_id} has stale source digest {req.get('source', {}).get('digest')}; "
                "cannot auto-conclude"
            )
        req_digest = req["source"]["digest"]
        cid = req["candidate"]["id"]
        cand_rev = str(req.get("candidate_revision", ""))
        study_id = studies.selected_study_id(req)
        view_id = req.get("view", {}).get("id", "unknown")
        is_diag = bool(diagnostic_evidence_scope(req))
        statuses = Counter()
        for job_def in req.get("jobs", []):
            job = state.get("jobs", {}).get(job_def["compatibility_key"], {})
            res = job.get("result")
            if res:
                for row in res.get("task_results", []):
                    st = row.get("gate_status", "UNKNOWN")
                    statuses[st] += 1
                    if st in {"ERROR", "INVALID", "INCOMPLETE"}:
                        raise CodeChangeRequiredError(
                            f"task {row.get('task_id')} for {cid} ({study_id or req_id}) ended with {st}: "
                            f"{row.get('reason', 'see run log')}"
                        )
        summary_counts = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())) or "no runnable tasks"
        scope_note = "Diagnostic performance/research scope only; no qualification credit." if is_diag else (
            "Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact."
        )
        candidate_ref = cid
        if cand_rev:
            full_ref = f"{cid}@{cand_rev}"
            candidate_ref = full_ref if len(full_ref) <= 160 else f"{cid}@{cand_rev[:16]}"
        _log(f"Recording readout for {cid} (study={study_id or 'none'}, {summary_counts})", json_mode=json_mode)
        try:
            rec = knowledge.readout(
                root,
                candidate_ref,
                conclusion=f"Completed {view_id} seed-0 CUDA execution for {cid} ({summary_counts}). {scope_note}",
                comparison=f"Measured unchanged declaration {cid} at source {req_digest[:12]} under protocol seed 0.",
                next_action="Retain source-bound readout and compiled Forge reports.",
                study_id=study_id,
            )
        except ValueError as exc:
            raise CodeChangeRequiredError(
                f"cannot conclude readout for {cid} (study={study_id or 'none'}): {exc}"
            ) from exc
        concluded_records.append(rec["record_id"])
    return concluded_records


def _scaling_registration_already_concluded_or_stale(
    root: Path,
    manifest: dict[str, Any],
    *,
    has_live_queue_submissions: bool,
    expected_files_sha: dict[str, str],
) -> bool:
    receipt_dir = Path(root) / "reports/forge/scaling-receipts" / str(manifest.get("id", ""))
    if receipt_dir.is_dir():
        receipt_files = sorted(receipt_dir.glob("*.json"))
        if receipt_files and not has_live_queue_submissions:
            return True
        for receipt_path in receipt_files:
            doc = read_json(receipt_path)
            res = doc.get("result", {})
            files_sha = res.get("identity", {}).get("source", {}).get("files_sha256")
            if files_sha is not None and files_sha != expected_files_sha:
                return True
            if _scaling_contention_status(res) != "clear":
                return True

    manifest_studies = set(manifest.get("studies", []))
    records_dir = Path(root) / "reports/forge/records"
    if manifest_studies and records_dir.is_dir():
        for rec_path in sorted(records_dir.glob("readout-*.json")):
            rec = read_json(rec_path)
            if rec.get("lifecycle") == "concluded" and rec.get("study_id") in manifest_studies:
                if not has_live_queue_submissions:
                    return True
    return False


def _select_or_register_scaling_campaign(
    root: Path,
    queue_root: Path,
    *,
    cuda_model: str,
    json_mode: bool = False,
) -> dict[str, Any]:
    expected_pairs = {(c["family"], c["candidate_id"]) for c in current_family_candidates(root)}
    expected_task_ids = [item["task"] for item in load_view(root, "dcgan_performance")["assignments"]]
    expected_files_sha = scaling_source_files_sha256(root)
    perf_dir = root / "configs/forge/performance"
    state_path = Path(queue_root) / "queue/state.json"
    q_state = read_json(state_path) if state_path.is_file() else {"submissions": {}}
    current_digest = catalog_source_manifest(root)["digest"]

    if perf_dir.is_dir():
        for path in sorted(perf_dir.glob("*.json"), reverse=True):
            try:
                manifest = scaling._load(root, path.stem)
            except ValueError:
                continue
            pairs = {(s["family"], s["candidate_id"]) for s in manifest.get("family_selections", [])}
            if (
                manifest.get("view") != "dcgan_performance"
                or manifest.get("execution_backend") != "cuda"
                or manifest.get("cuda_model") != cuda_model
                or pairs != expected_pairs
                or manifest.get("task_ids") != expected_task_ids
            ):
                continue
            existing_subs = [
                item
                for item in q_state.get("submissions", {}).values()
                if item["request"].get("campaign_id") == manifest["campaign"]["id"]
            ]
            if any(not _request_matches_worktree(root, item["request"]) for item in existing_subs):
                continue
            if _scaling_registration_already_concluded_or_stale(
                root,
                manifest,
                has_live_queue_submissions=bool(existing_subs),
                expected_files_sha=expected_files_sha,
            ):
                continue
            _log(f"Using registered scaling campaign {manifest['id']}", json_mode=json_mode)
            return manifest

    src12 = current_digest[:12]
    reg_id = f"dcgan-performance-cloud-{src12}"
    idx = 2
    while scaling.manifest_path(root, reg_id).exists():
        reg_id = f"dcgan-performance-cloud-{src12}-r{idx}"
        idx += 1
    _log(f"Registering new unified scaling campaign {reg_id}", json_mode=json_mode)
    scaling.register(root, reg_id, view_id="dcgan_performance", execution_backend="cuda", cuda_model=cuda_model)
    return scaling._load(root, reg_id)


def _job_matches_resolved_gpu(job: dict[str, Any], hw: dict[str, Any], resolved_gpu: str) -> bool:
    target = str(resolved_gpu).strip()
    if not target:
        return True
    phys_uuid = hw.get("physical_uuid")
    uuid_matches = (
        target.startswith("GPU-")
        and isinstance(phys_uuid, str)
        and (phys_uuid == target or phys_uuid.startswith(target))
    )
    worker_dev = str((job.get("worker") or {}).get("device") or "").strip()
    if worker_dev:
        return worker_dev == target or uuid_matches
    cvd = str(hw.get("cuda_visible_devices") or "").strip()
    if cvd:
        return cvd == target or uuid_matches
    if target.startswith("GPU-") and isinstance(phys_uuid, str) and phys_uuid:
        return uuid_matches
    dev = str(hw.get("device") or "").strip()
    if dev:
        expected_dev = f"cuda:{target}" if target.isdigit() else target
        return dev == expected_dev
    return True


def _evict_superseded_scaling_queue_jobs(
    root: Path,
    queue_root: Path,
    manifest: dict[str, Any],
    resolved_gpu: str,
    *,
    json_mode: bool = False,
) -> int:
    queue_root = Path(queue_root).resolve()
    state_path = queue_root / "queue/state.json"
    if not state_path.is_file():
        return 0
    expected_keys: set[str] = set()
    view_id = str(manifest.get("view") or "dcgan_performance")
    for sel in manifest.get("family_selections", []):
        cid = sel.get("candidate_id")
        if not cid:
            continue
        study_id = sel.get("study") or sel.get("study_id")
        try:
            if study_id and scaling.study_path(root, str(study_id)).is_file():
                req = planning.resolve_idea(root, cid, study=str(study_id))
            else:
                req = planning.resolve_idea(
                    root,
                    cid,
                    view_id=view_id,
                    freeze_source=False,
                    execution_backend=manifest.get("execution_backend", "cuda"),
                    cuda_model=manifest.get("cuda_model"),
                )
            for job_def in req.get("jobs", []):
                expected_keys.add(job_def["compatibility_key"])
        except Exception:
            continue
    if not expected_keys:
        return 0

    expected_device = f"cuda:{resolved_gpu}" if str(resolved_gpu).isdigit() else str(resolved_gpu)
    with file_lock(queue_root / "queue.lock"):
        if not state_path.is_file():
            return 0
        state = read_json(state_path)
        jobs = state.get("jobs", {})
        subs = state.get("submissions", {})
        evict_keys: set[str] = set()
        evict_sub_ids: set[str] = set()
        evict_attempts: list[str] = []

        for ck in sorted(expected_keys):
            job = jobs.get(ck)
            if not job or (job.get("status") != "terminal" and job.get("state") not in {"completed", "failed"}):
                continue
            res_obj = job.get("result") or {}
            raw_res = ((res_obj.get("raw") or {}).get("result") or {})
            hw = raw_res.get("hardware") if isinstance(raw_res.get("hardware"), dict) else None
            if hw is None:
                trows = res_obj.get("task_results") or []
                if trows and isinstance(trows[0], dict) and isinstance(trows[0].get("hardware"), dict):
                    hw = trows[0]["hardware"]
            hw_dict = hw if isinstance(hw, dict) else {}
            bad_contention = hw_dict.get("contention_status") != "clear"
            bad_device = not _job_matches_resolved_gpu(job, hw_dict, resolved_gpu)
            if not (bad_contention or bad_device):
                continue
            subscriber_ids = list(job.get("subscribers") or [])
            blocking_subscribers = [
                sid
                for sid in subscriber_ids
                if sid in subs
                and (
                    subs[sid].get("request", {}).get("view", {}).get("id") != "dcgan_performance"
                    or subs[sid].get("status") in {"queued", "running", "paused"}
                )
            ]
            if blocking_subscribers:
                continue
            evict_keys.add(ck)
            evict_sub_ids.update(sid for sid in subscriber_ids if sid in subs)
            aid = res_obj.get("attempt_id") or job.get("attempt_id")
            if isinstance(aid, str) and aid:
                evict_attempts.append(aid)

        if not evict_keys:
            return 0

        archive_dir = queue_root / "superseded-scaling" / str(manifest.get("id", "scaling")) / "attempts"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for aid in evict_attempts:
            src = Path(root) / "reports/forge/attempts" / aid
            dst = archive_dir / aid
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
        for ck in evict_keys:
            jobs.pop(ck, None)
        for sid in evict_sub_ids:
            subs.pop(sid, None)
        if evict_sub_ids:
            for other_job in jobs.values():
                if isinstance(other_job.get("subscribers"), list):
                    other_job["subscribers"] = [
                        sid for sid in other_job["subscribers"] if sid not in evict_sub_ids
                    ]
        atomic_json(state_path, state)

    _log(
        f"Evicted {len(evict_keys)} superseded/contended scaling job(s) before draining on {expected_device}",
        json_mode=json_mode,
    )
    return len(evict_keys)


def refresh_scaling_performance(
    root: Path,
    queue_root: Path,
    *,
    scaling_gpu: str,
    gpus: list[str] | tuple[str, ...] = ("0", "1"),
    cuda_model: str,
    json_mode: bool = False,
) -> dict[str, Any]:
    resolved_gpu = _resolve_scaling_gpu(scaling_gpu, gpus, json_mode=json_mode)
    manifest = _select_or_register_scaling_campaign(root, queue_root, cuda_model=cuda_model, json_mode=json_mode)
    reg_id = manifest["id"]
    campaign_id = manifest["campaign"]["id"]
    _evict_superseded_scaling_queue_jobs(root, queue_root, manifest, resolved_gpu, json_mode=json_mode)
    queue = forge_queue.Queue(queue_root, report_root=root / "reports/forge")

    enqueued = scaling.enqueue(root, queue_root, reg_id, queue=queue)
    blocked = [item for item in enqueued["submissions"] if item["status"] == "BLOCKED"]
    if blocked:
        raise CodeChangeRequiredError(f"scaling campaign {reg_id} has preflight blockers: {blocked}")

    _log(
        f"Draining scaling campaign {campaign_id} on single GPU cuda:{resolved_gpu} (workers_per_gpu=1). "
        f"Tail logs: tail -f {queue_root / 'events.jsonl'}",
        json_mode=json_mode,
    )
    forge_queue.drain(
        queue,
        devices=[resolved_gpu],
        workers_per_gpu=1,
        allow_sharing=False,
        campaign=campaign_id,
    )
    _conclude_queue_submissions(root, queue_root, campaign_id=campaign_id, json_mode=json_mode)
    published = scaling_export.publish(root, queue_root, reg_id, allow_partial=False, refresh=False)
    verify = check_scaling_performance(root)
    if not verify["current"]:
        raise CodeChangeRequiredError(f"scaling performance remained incomplete after execution: {verify}")
    return published


def _ensure_frozen_checkout(
    root: Path,
    commit: str,
    checkout_root: Path,
    *,
    preserved_absent_paths: list[str] | tuple[str, ...] = (),
) -> Path:
    absent_list = sorted(set(preserved_absent_paths))
    dir_name = f"{commit}-{stable_hash(absent_list)[:8]}" if absent_list else commit
    target = checkout_root / dir_name
    expected_stamp = json.dumps({"commit": commit, "preserved_absent_paths": absent_list}, sort_keys=True)
    stamp = target / ".git-archive-commit"
    if stamp.is_file() and stamp.read_text().strip() == expected_stamp:
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        archive = subprocess.Popen(
            ["git", "archive", commit],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None and archive.stderr is not None
        tar_proc = subprocess.run(
            ["tar", "-x", "-C", str(target)],
            stdin=archive.stdout,
            capture_output=True,
            text=True,
        )
        archive.stdout.close()
        git_err = archive.stderr.read().decode("utf-8", errors="replace").strip()
        archive.stderr.close()
        git_rc = archive.wait()
        if git_rc != 0:
            raise CodeChangeRequiredError(
                f"git archive failed for frozen commit {commit} (rc={git_rc}): {git_err or 'missing commit'}"
            )
        if tar_proc.returncode != 0:
            raise CodeChangeRequiredError(
                f"tar extraction failed for frozen commit {commit} (rc={tar_proc.returncode}): "
                f"{tar_proc.stderr.strip() or 'invalid archive'}"
            )
        for rel in absent_list:
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise CodeChangeRequiredError(f"unsafe preserved_absent_path: {rel}")
            (target / rel_path).unlink(missing_ok=True)
        stamp.write_text(f"{expected_stamp}\n")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def _prepare_combined_attempt_request(
    frozen_checkout: Path,
    candidate_id: str,
    available_tasks: list[str],
    state_dir: Path,
    family: str,
) -> Path:
    req_dir = state_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    output_path = req_dir / f"{family}.json"
    all_ordinary_views = (
        "clockfree_continuous",
        "discriminator_stability",
        "adaptation",
        "formulation_comparison",
        "host_profile_transfer",
        "quality_coverage",
    )
    view_names = [
        name
        for name in all_ordinary_views
        if (frozen_checkout / "configs/forge/views" / f"{name}.json").is_file()
    ]
    if not view_names:
        raise CodeChangeRequiredError(
            f"frozen checkout {frozen_checkout} has no ordinary view declarations for {family}"
        )
    partial_specs = []
    partial_outputs = []
    for view_name in view_names:
        part_out = req_dir / f"_{family}-{view_name}.json"
        partial_outputs.append(part_out)
        partial_specs.append(
            {
                "candidate_id": candidate_id,
                "view_id": view_name,
                "tasks": available_tasks,
                "output": str(part_out),
            }
        )
    spec_path = req_dir / f"_{family}-prepare.json"
    atomic_json(spec_path, {"source_root": str(frozen_checkout), "requests": partial_specs})
    try:
        subprocess.run(
            [sys.executable, str(attempt_coverage.WORKER), "prepare", str(spec_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        raise CodeChangeRequiredError(
            f"failed to prepare frozen attempt request for {family} ({candidate_id}): "
            f"{(exc.stderr or '').strip() or exc}"
        ) from exc
    merged = read_json(partial_outputs[0])
    seen_job_keys = {job["compatibility_key"] for job in merged.get("jobs", [])}
    for part_out in partial_outputs[1:]:
        part = read_json(part_out)
        merged["tasks"].update(part.get("tasks", {}))
        for job in part.get("jobs", []):
            if job["compatibility_key"] not in seen_job_keys:
                merged["jobs"].append(job)
                seen_job_keys.add(job["compatibility_key"])
    for part_out in [*partial_outputs, spec_path]:
        part_out.unlink(missing_ok=True)
    atomic_json(output_path, merged)
    return output_path


def _record_to_state_job(record: dict[str, Any]) -> dict[str, Any]:
    """Invert coverage_value()'s new_attempt evidence rewrite for lossless round-tripping."""
    job = deepcopy(record)
    if job.get("origin") == "new_attempt" and "original_attempt" in job:
        job["evidence"] = job.pop("original_attempt")
    return job


def _certified_parent_from_queue(
    queue_root: Path,
    candidate_revision: str,
    task_id: str,
    compatibility_key: str,
) -> dict[str, Any] | None:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return None
    q_state = read_json(state_path)
    job = q_state.get("jobs", {}).get(compatibility_key)
    if not job or job.get("status") != "terminal" or not job.get("result"):
        return None
    res = job["result"]
    if res.get("candidate_revision") != candidate_revision:
        return None
    row = next((item for item in res.get("task_results", []) if item.get("task_id") == task_id), None)
    if not row or row.get("gate_status") != "PASS":
        return None
    ev = row.get("evidence", {})
    cp, manifest = ev.get("checkpoint"), ev.get("artifact_manifest")
    if not cp or not manifest or not Path(cp).is_file() or not Path(manifest).is_file():
        return None
    return {
        "attempt_id": res["attempt_id"],
        "result_hash": stable_hash(res),
        "result": row,
        "candidate_revision": candidate_revision,
        "compatibility_key": compatibility_key,
    }


def _is_cross_gpu_rng_mismatch(job: dict[str, Any]) -> bool:
    return (
        job.get("status") == "ERROR"
        and "trainer/named RNG checkpoint mismatch" in str(job.get("reason", ""))
    )


def _frozen_preflight_blockers(frozen_checkout: Path, candidate_id: str) -> dict[str, list[str]]:
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from experiments.forge import planning\n"
        "co = Path(sys.argv[1])\n"
        "cid = sys.argv[2]\n"
        "views = ('clockfree_continuous', 'discriminator_stability', 'adaptation', "
        "'formulation_comparison', 'host_profile_transfer', 'quality_coverage')\n"
        "out = {}\n"
        "for name in views:\n"
        "    vp = co / 'configs/forge/views' / f'{name}.json'\n"
        "    if not vp.is_file():\n"
        "        continue\n"
        "    v = json.loads(vp.read_text())\n"
        "    max_tier = max(item['qualification_tier'] for item in v['assignments'])\n"
        "    req = planning.resolve_idea(co, cid, view_id=name, through_tier=max_tier, "
        "freeze_source=False, execution_backend='cuda')\n"
        "    for tid, card in req.get('tasks', {}).items():\n"
        "        blockers = card.get('preflight_blockers') or []\n"
        "        if blockers:\n"
        "            out.setdefault(tid, []).extend(blockers)\n"
        "print(json.dumps({k: sorted(set(v)) for k, v in out.items()}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(frozen_checkout)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(frozen_checkout), candidate_id],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        raise CodeChangeRequiredError(
            f"failed to inspect frozen preflight blockers for {candidate_id} in {frozen_checkout}: "
            f"{(exc.stderr or '').strip() or exc}"
        ) from exc
    return json.loads(proc.stdout)


def _is_expected_preflight_refusal(
    job: dict[str, Any],
    preflight_by_family_task: dict[tuple[str, str], list[str]],
) -> bool:
    if job.get("status") != "ERROR":
        return False
    reason = str(job.get("reason", ""))
    blockers = preflight_by_family_task.get((str(job.get("family", "")), str(job.get("task", ""))), [])
    for blocker in blockers:
        raw = str(blocker).strip()
        suffix = raw.split(": ", 1)[-1].strip()
        if (raw and raw in reason) or (suffix and suffix in reason):
            return True
    return False


def _pin_checkpoint_pair_gpus(
    jobs: dict[str, dict[str, Any]],
    family: str,
    pair_gpus: list[str],
) -> None:
    family_jobs = {
        job["task"]: job
        for job in jobs.values()
        if not _is_scoped_attempt(job) and job.get("family") == family
    }
    allowed_gpus = {str(g) for g in pair_gpus}
    default_gpu = str(pair_gpus[0])
    changed = True
    while changed:
        changed = False
        for _, child in sorted(family_jobs.items()):
            for dep in child.get("dependencies", []):
                if not (isinstance(dep, dict) and dep.get("kind") == "checkpoint"):
                    continue
                parent = family_jobs.get(dep["task"])
                if parent is None:
                    continue
                target_gpu: str | None = None
                if parent.get("required_gpu") is not None and str(parent["required_gpu"]) in allowed_gpus:
                    target_gpu = str(parent["required_gpu"])
                elif parent.get("gpu") is not None and str(parent["gpu"]) in allowed_gpus:
                    target_gpu = str(parent["gpu"])
                elif parent.get("certified_parent"):
                    device = str(parent["certified_parent"].get("result", {}).get("device", ""))
                    if device.startswith("cuda:"):
                        cand_gpu = device.split(":", 1)[1].strip()
                        if cand_gpu in allowed_gpus:
                            target_gpu = cand_gpu
                if target_gpu is None:
                    target_gpu = default_gpu
                if parent.get("status") == "PENDING" and parent.get("required_gpu") != target_gpu:
                    parent["required_gpu"] = target_gpu
                    changed = True
                if child.get("status") == "PENDING" and child.get("required_gpu") != target_gpu:
                    child["required_gpu"] = target_gpu
                    changed = True


def refresh_attempt_coverage(
    root: Path,
    queue_root: Path,
    attempt_state_root: Path,
    *,
    gpus: list[str],
    json_mode: bool = False,
) -> dict[str, Any]:
    if len(gpus) < 2 or len(set(gpus[:2])) != 2:
        raise CodeChangeRequiredError(
            "selected-family attempt_coverage.drain requires two distinct GPUs (--gpus 0,1)"
        )
    pair_gpus = [str(gpus[0]), str(gpus[1])]
    root = Path(root).resolve()
    attempt_state_root = Path(attempt_state_root).resolve()
    attempt_state_root.mkdir(parents=True, exist_ok=True)
    if (root / FIXED_TEST_LIST).is_file():
        from ops.formulaic_coverage import (
            RetiredCoverageExecutionError, ensure_formulaic_attempt_coverage,
        )

        try:
            return ensure_formulaic_attempt_coverage(
                root, attempt_state_root, pair_gpus, json_mode=json_mode
            )
        except RetiredCoverageExecutionError as error:
            raise CodeChangeRequiredError(str(error)) from error

    publication = read_json(root / "reports/forge/technique-inventory.json")
    existing_cov = (
        read_json(root / "reports/forge/attempt-coverage.json")
        if (root / "reports/forge/attempt-coverage.json").is_file()
        else {"records": []}
    )
    previous_state = (
        read_json(attempt_state_root / "state.json")
        if (attempt_state_root / "state.json").is_file()
        else {"jobs": {}}
    )

    selected = attempt_coverage.selected_tasks(publication)
    expected_selected: dict[str, tuple[dict[str, Any], str, list[str]]] = {}
    commit_tasks_cache: dict[str, set[str]] = {}
    for family, selection, tasks in selected:
        commit = _selected_family_source_commit(root, publication, family, selection)
        if commit not in commit_tasks_cache:
            commit_tasks_cache[commit] = _git_task_names_at_commit(root, commit)
        available = [task for task in tasks if task in commit_tasks_cache[commit]]
        expected_selected[family] = (selection, commit, available)

    jobs: dict[str, dict[str, Any]] = {}
    # 1. Preserve all scoped records and all still-current selected-family records.
    for record in existing_cov.get("records", []):
        if _is_scoped_attempt(record):
            job = _record_to_state_job(record)
            jobs[job["key"]] = job
            continue
        family = record.get("family")
        if family not in expected_selected:
            continue
        selection = expected_selected[family][0]
        if (
            record.get("candidate_id") == selection["candidate_id"]
            and record.get("candidate_revision") == selection["candidate_revision"]
            and record.get("source_digest") == selection["bindings"]["source_digest"]
        ):
            job = _record_to_state_job(record)
            jobs[job["key"]] = job

    # 2. Merge any in-progress or completed jobs from attempt_state_root/state.json that match current targets.
    for key, prev_job in previous_state.get("jobs", {}).items():
        if _is_scoped_attempt(prev_job):
            jobs.setdefault(key, deepcopy(prev_job))
            continue
        if _is_cross_gpu_rng_mismatch(prev_job):
            continue
        family = prev_job.get("family")
        if family in expected_selected:
            selection = expected_selected[family][0]
            if (
                prev_job.get("candidate_id") == selection["candidate_id"]
                and prev_job.get("candidate_revision") == selection["candidate_revision"]
                and prev_job.get("source_digest") == selection["bindings"]["source_digest"]
            ):
                if key not in jobs or jobs[key].get("status") == "PENDING":
                    jobs[key] = deepcopy(prev_job)

    retained = attempt_coverage.retained_attempts(root)
    checkout_root = Path(queue_root) / "frozen-checkouts"
    newly_scheduled_keys: set[str] = set()
    preflight_by_family_task: dict[tuple[str, str], list[str]] = {}

    for family, (selection, commit, available) in expected_selected.items():
        covered_tasks = {
            job["task"]
            for job in jobs.values()
            if not _is_scoped_attempt(job)
            and job.get("family") == family
            and job.get("status") in attempt_coverage.DONE
        }
        if set(available).issubset(covered_tasks):
            continue

        _log(
            f"Preparing frozen attempt-coverage request for {family} ({selection['candidate_id']} @ {commit[:12]})",
            json_mode=json_mode,
        )
        absent_paths = _selected_family_preserved_absent_paths(root, publication, selection)
        frozen_dir = _ensure_frozen_checkout(
            root, commit, checkout_root, preserved_absent_paths=absent_paths
        )
        for tid, blockers in _frozen_preflight_blockers(frozen_dir, selection["candidate_id"]).items():
            preflight_by_family_task[(family, tid)] = blockers
        request_path = _prepare_combined_attempt_request(
            frozen_dir, selection["candidate_id"], available, attempt_state_root, family
        )
        request = read_json(request_path)
        if (
            request["source"]["digest"] != selection["bindings"]["source_digest"]
            or request["candidate_revision"] != selection["candidate_revision"]
        ):
            raise CodeChangeRequiredError(
                f"{family}: frozen request mismatch with selected row bindings"
            )

        recorded_statuses = {
            item["task_id"]: item["status"]
            for item in selection.get("tasks", []) + selection.get("nonrequired_tasks", [])
            if item.get("status") in {"PASS", "FAIL"}
        }
        job_compat_by_task = {
            tid: job_def["compatibility_key"]
            for job_def in request.get("jobs", [])
            for tid in job_def.get("task_ids", [job_def["task_id"]])
        }

        # Identify checkpoint parents required by unrecorded children so passing parents
        # without local checkpoint files run once to materialize certified state.
        checkpoint_parents_needed: set[str] = set()
        for task_name in available:
            if task_name in covered_tasks or task_name in recorded_statuses:
                continue
            card = request["tasks"][task_name]
            for dep in card.get("dependencies", []):
                if isinstance(dep, dict) and dep.get("kind") == "checkpoint":
                    checkpoint_parents_needed.add(dep["task"])

        req_sha = file_hash(request_path)
        for task_name in available:
            card = request["tasks"][task_name]
            execution = task_execution_fingerprint(card)
            evaluation = task_evaluation_fingerprint(card)
            binding = selection["bindings"]["task_contracts"].get(task_name)
            pin = publication.get("task_contracts", {}).get(binding) if binding else None
            if pin and (execution != pin["execution_sha256"] or evaluation != pin["evaluation_sha256"]):
                raise CodeChangeRequiredError(
                    f"{family}/{task_name}: frozen task contract differs from published binding"
                )
            key = attempt_coverage.identity(
                request["source"]["digest"], request["candidate_revision"], task_name, execution, evaluation
            )
            existing = jobs.get(key)
            if existing and existing.get("status") in attempt_coverage.DONE:
                continue

            job: dict[str, Any] = {
                "key": key,
                "family": family,
                "candidate_id": selection["candidate_id"],
                "candidate_revision": request["candidate_revision"],
                "source_digest": request["source"]["digest"],
                "task": task_name,
                "execution_sha256": execution,
                "evaluation_sha256": evaluation,
                "request_path": str(request_path),
                "source_root": request["source"]["snapshot_path"],
                "request_sha256": req_sha,
                "timeout_seconds": card["resources"]["timeout_seconds"],
                "resources": card["resources"],
                "dependencies": card.get("dependencies", []),
                "status": "PENDING",
            }
            if key in retained:
                job.update(retained[key])
            elif task_name in recorded_statuses:
                rec_status = recorded_statuses[task_name]
                if rec_status == "PASS" and task_name in checkpoint_parents_needed:
                    cert = _certified_parent_from_queue(
                        queue_root,
                        request["candidate_revision"],
                        task_name,
                        job_compat_by_task.get(task_name, ""),
                    )
                    if cert is not None:
                        job.update(
                            status="PASS",
                            attempted=True,
                            origin="retained_selected_row",
                            diagnostic=False,
                            evidence="reports/forge/technique-inventory.json",
                            certified_parent=cert,
                        )
                    else:
                        newly_scheduled_keys.add(key)
                else:
                    job.update(
                        status=rec_status,
                        attempted=True,
                        origin="retained_selected_row",
                        diagnostic=False,
                        evidence="reports/forge/technique-inventory.json",
                    )
            else:
                newly_scheduled_keys.add(key)
            jobs[key] = job

        _pin_checkpoint_pair_gpus(jobs, family, pair_gpus)

    state = {
        "schema_version": 1,
        "jobs": jobs,
        "maximum_tests": 4,
        "maximum_per_gpu": 2,
        "include_research": True,
    }
    atomic_json(attempt_state_root / "state.json", state)
    pending_count = sum(1 for item in jobs.values() if item.get("status") in {"PENDING", "RUNNING"})
    _log(
        f"Draining attempt coverage ({pending_count} pending/running of {len(jobs)} records) on GPUs "
        f"{','.join(pair_gpus)}. Tail logs: tail -f {attempt_state_root / 'events.jsonl'}",
        json_mode=json_mode,
    )
    result_summary = attempt_coverage.drain(root, attempt_state_root, gpus=pair_gpus)
    final_state = read_json(attempt_state_root / "state.json")
    bad_new = [
        {
            "family": item["family"],
            "candidate_id": item["candidate_id"],
            "task": item["task"],
            "status": item["status"],
            "reason": item.get("reason", ""),
        }
        for key, item in final_state.get("jobs", {}).items()
        if key in newly_scheduled_keys
        and item.get("status") in {"ERROR", "INVALID", "INCOMPLETE", "INTERRUPTED"}
        and not _is_expected_preflight_refusal(item, preflight_by_family_task)
    ]
    if bad_new:
        raise CodeChangeRequiredError(f"newly executed diagnostic attempts failed with non-gate errors: {bad_new}")
    if (root / FIXED_TEST_LIST).is_file():
        from ops.formulaic_coverage import ensure_formulaic_attempt_coverage

        result_summary = ensure_formulaic_attempt_coverage(
            root, attempt_state_root, pair_gpus, json_mode=json_mode
        )
    return result_summary


def _signature_for_request(req: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    t1_tasks = [
        item["task"]
        for item in req["view"]["assignments"]
        if item["qualification_tier"] == 1 and not req["tasks"][item["task"]].get("preflight_blockers")
    ]
    search_order = t1_tasks + [name for name in sorted(req["tasks"]) if name not in t1_tasks]
    for task_id in search_order:
        evaluation = req["tasks"][task_id].get("evaluation", {})
        thresholds = evaluation.get("thresholds") or []
        for entry in thresholds:
            if (
                isinstance(entry, list)
                and len(entry) == 3
                and isinstance(entry[0], str)
                and entry[1] in OP_COMPLEMENT
                and isinstance(entry[2], (int, float))
            ):
                metric, op, threshold = entry[0], entry[1], float(entry[2])
                base = {"task_id": task_id, "metric": metric, "threshold": threshold, "phase": "final"}
                return {**base, "op": op}, {**base, "op": OP_COMPLEMENT[op]}
    raise CodeChangeRequiredError(f"no numerical final-metric threshold found for view {req['view']['id']}")


def _requal_binding_fingerprint(
    candidate_id: str,
    req: dict[str, Any],
    *,
    cuda_model: str,
) -> str:
    return stable_hash(
        {
            "candidate_id": candidate_id,
            "candidate_revision": req.get("candidate_revision"),
            "policy_fingerprint": req.get("policy_fingerprint"),
            "jobs": sorted(job["compatibility_key"] for job in req.get("jobs", [])),
            "cuda_model": cuda_model,
        }
    )[:8]


def _study_reusable_for_requal(
    root: Path,
    q_state: dict[str, Any],
    concluded_records: list[dict[str, Any]],
    study: dict[str, Any],
    *,
    candidate_id: str,
    req: dict[str, Any],
    cuda_model: str,
    required_through_tier: int,
) -> bool:
    if study.get("candidate") != candidate_id:
        return False
    scope = study.get("scope") or {}
    if (
        scope.get("view") != "discriminator_stability"
        or scope.get("execution_backend") != "cuda"
        or scope.get("cuda_model") != cuda_model
        or int(scope.get("through_tier", 1)) < required_through_tier
    ):
        return False
    study_id = study["id"]
    for sub in q_state.get("submissions", {}).values():
        if studies.selected_study_id(sub.get("request", {})) == study_id and sub.get("status") == "concluded":
            return False
    study_hash = stable_hash(study)
    for rec in concluded_records:
        if rec.get("study_id") == study_id:
            if rec.get("candidate_revision") != req.get("candidate_revision"):
                return False
            if rec.get("study_sha256") not in {None, study_hash}:
                return False
    try:
        resolved = planning.resolve_idea(root, candidate_id, study=study_id, freeze_source=False)
    except Exception:
        return False
    if resolved.get("preflight_blockers") or resolved.get("study_review", {}).get("status") != "READY":
        return False
    locked = (q_state.get("studies") or {}).get(study_id)
    if locked is not None and locked != {
        "declaration": resolved["study"],
        "binding": resolved["study_admission"],
    }:
        return False
    return True


def _select_or_create_requal_campaign(
    root: Path,
    q_state: dict[str, Any],
    *,
    src12: str,
    batch_sig: str,
    max_allowance: int,
    family_count: int,
) -> dict[str, Any]:
    req_cand_budget = max(max_allowance, 100000)
    req_total_budget = max(max_allowance * max(family_count, 1) * 2, 600000)
    campaigns_dir = Path(root) / "configs/forge/campaigns"
    q_campaigns = q_state.get("campaigns") or {}

    base_id = identifier(f"stale-family-requal-{src12}", "campaign")
    base_path = campaigns_dir / f"{base_id}.json"
    if base_path.is_file() and base_id not in q_campaigns:
        existing = read_json(base_path)
        if (
            existing.get("id") == base_id
            and float(existing.get("candidate_budget_seconds", 0)) >= req_cand_budget
            and float(existing.get("budget_seconds", 0)) >= req_total_budget
        ):
            return existing

    if not base_path.exists() and base_id not in q_campaigns:
        campaign_id = base_id
    else:
        campaign_id = identifier(f"stale-family-requal-{src12}-{batch_sig}", "campaign")
        idx = 2
        while (campaigns_dir / f"{campaign_id}.json").exists() or campaign_id in q_campaigns:
            campaign_id = identifier(f"stale-family-requal-{src12}-{batch_sig}-r{idx}", "campaign")
            idx += 1

    campaign = {
        "id": campaign_id,
        "candidate_budget_seconds": req_cand_budget,
        "budget_seconds": req_total_budget,
    }
    atomic_json(campaigns_dir / f"{campaign_id}.json", campaign)
    return campaign


def _ensure_ordinary_requal_study(
    root: Path,
    family: str,
    candidate_id: str,
    req: dict[str, Any],
    campaign: dict[str, Any],
    *,
    cuda_model: str,
    through_tier: int | None = None,
    q_state: dict[str, Any] | None = None,
    concluded_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    q_state = q_state or {}
    concluded_records = concluded_records if concluded_records is not None else _concluded_ordinary_records(root)
    target_tier = through_tier if through_tier is not None else _discriminator_stability_max_tier(root)
    src12 = req["source"]["digest"][:12]
    bind8 = _requal_binding_fingerprint(candidate_id, req, cuda_model=cuda_model)

    legacy_id = identifier(f"requal-{src12}-{family}-ordinary", "study")
    bound_id = identifier(f"requal-{src12}-{family}-{bind8}-ordinary", "study")
    for candidate_study_id in (bound_id, legacy_id):
        existing_path = studies.study_path(root, candidate_study_id)
        if existing_path.is_file():
            existing_study = studies.load_study(root, candidate_study_id)
            if _study_reusable_for_requal(
                root,
                q_state,
                concluded_records,
                existing_study,
                candidate_id=candidate_id,
                req=req,
                cuda_model=cuda_model,
                required_through_tier=min(2, target_tier),
            ):
                return existing_study

    q_studies = q_state.get("studies") or {}
    if not studies.study_path(root, legacy_id).exists() and legacy_id not in q_studies:
        study_id = legacy_id
    else:
        study_id = bound_id
        idx = 2
        while studies.study_path(root, study_id).exists() or study_id in q_studies:
            study_id = identifier(f"requal-{src12}-{family}-{bind8}-r{idx}-ordinary", "study")
            idx += 1

    control_id = (
        "bispf-optimized-quintic-v1"
        if candidate_id != "bispf-optimized-quintic-v1"
        else "atlas-particle-normal-v4"
    )
    prediction, falsifier = _signature_for_request(req)
    study = studies.scaffold(
        study_id,
        candidate_id,
        control_id,
        "discriminator_stability",
        hypothesis=(
            f"Ordinary re-qualification of the unchanged selected {family} recipe ({candidate_id}) on "
            f"current develop source {src12} under protocol seed 0 on {cuda_model} records current-source "
            "discriminator_stability and shared goal-view verdicts."
        ),
        execution_backend="cuda",
    )
    study["status"] = "ready"
    study["scope"]["through_tier"] = target_tier
    study["scope"]["cuda_model"] = cuda_model
    study["campaign"] = deepcopy(campaign)
    study["prior_evidence"] = [
        {
            "path": CURRENT_SELECTION.as_posix(),
            "selector": [],
            "identity": {"scope": "whole_candidate_family_current", "default_adoption": False},
            "use": "motivation_only",
        }
    ]
    study["prediction"] = prediction
    study["falsifier"] = falsifier
    study["competing_explanation"] = (
        "Source updates since the prior family pin could alter unguarded numerical paths or runtime "
        "bindings; seed-0 public-API execution on fixed hardware isolates current-source behavior."
    )
    studies.validate_study(study)
    atomic_json(studies.study_path(root, study_id), study)
    return study


def refresh_selected_family_readouts(
    root: Path,
    queue_root: Path,
    *,
    gpus: list[str],
    cuda_model: str,
    json_mode: bool = False,
) -> list[str]:
    status = check_selected_family_readouts(root, cuda_model=cuda_model)
    if status["current"]:
        return []

    targets_by_family: dict[str, str] = {}
    for item in status["missing_families"]:
        targets_by_family[item["family"]] = item["candidate_id"]
    for item in status.get("local_attempts_missing_families", []):
        targets_by_family.setdefault(item["family"], item["candidate_id"])
    if not targets_by_family:
        return []

    max_tier = _discriminator_stability_max_tier(root)
    src12 = catalog_source_manifest(root)["digest"][:12]
    prepared: list[tuple[str, str, dict[str, Any]]] = []
    max_allowance = 0
    for family, cid in sorted(targets_by_family.items()):
        base_req = planning.resolve_idea(
            root,
            cid,
            view_id="discriminator_stability",
            through_tier=max_tier,
            freeze_source=False,
            execution_backend="cuda",
            cuda_model=cuda_model,
        )
        allowance = sum(job["budget_seconds"] for job in base_req["jobs"])
        max_allowance = max(max_allowance, allowance)
        prepared.append((family, cid, base_req))

    state_path = Path(queue_root) / "queue/state.json"
    q_state = read_json(state_path) if state_path.is_file() else {}
    concluded_records = _concluded_ordinary_records(root)
    batch_sig = stable_hash(
        [
            (family, _requal_binding_fingerprint(cid, base_req, cuda_model=cuda_model))
            for family, cid, base_req in prepared
        ]
    )[:8]
    default_campaign = _select_or_create_requal_campaign(
        root,
        q_state,
        src12=src12,
        batch_sig=batch_sig,
        max_allowance=max_allowance,
        family_count=len(prepared),
    )

    queue = forge_queue.Queue(queue_root, report_root=root / "reports/forge")
    submitted_campaigns: list[str] = []
    for family, cid, base_req in prepared:
        study = _ensure_ordinary_requal_study(
            root,
            family,
            cid,
            base_req,
            default_campaign,
            cuda_model=cuda_model,
            through_tier=max_tier,
            q_state=q_state,
            concluded_records=concluded_records,
        )
        request = planning.resolve_idea(root, cid, study=study["id"])
        if request["preflight_blockers"]:
            raise CodeChangeRequiredError(
                f"ordinary requalification study {study['id']} is blocked: {request['preflight_blockers']}"
            )
        request["source"]["snapshot_path"] = str(
            snapshot_source(Path(root), Path(queue_root), request["source"])
        )
        study_campaign = deepcopy(study["campaign"])
        entry = queue.submit(request, study_campaign)
        if study_campaign["id"] not in submitted_campaigns:
            submitted_campaigns.append(study_campaign["id"])
        _log(
            f"Submitted {study['id']} ({cid}) -> request {entry['request']['request_id']} ({entry['status']})",
            json_mode=json_mode,
        )

    drain_devices = [str(g) for g in gpus]
    concluded_ids: list[str] = []
    for campaign_id in submitted_campaigns:
        _log(
            f"Draining ordinary requalification campaign {campaign_id} across GPUs {','.join(drain_devices)} "
            f"(workers_per_gpu=1). Tail logs: tail -f {queue_root / 'events.jsonl'}",
            json_mode=json_mode,
        )
        forge_queue.drain(
            queue,
            devices=drain_devices,
            workers_per_gpu=1,
            allow_sharing=False,
            campaign=campaign_id,
        )
        concluded_ids.extend(
            _conclude_queue_submissions(root, queue_root, campaign_id=campaign_id, json_mode=json_mode)
        )
    return concluded_ids


def _retry_shared_gpu_timeouts(
    queue: forge_queue.Queue,
    queue_root: Path,
    root: Path,
    *,
    json_mode: bool = False,
) -> list[str]:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return []
    state = read_json(state_path)
    retried_keys: list[str] = []
    seen_keys: set[str] = set()
    for _, entry in sorted(state.get("submissions", {}).items()):
        if entry.get("lifecycle") not in {"ready", "running", "awaiting_readout"}:
            continue
        req = entry["request"]
        if not _request_matches_worktree(root, req):
            continue
        for job_def in req.get("jobs", []):
            key = job_def["compatibility_key"]
            if key in seen_keys:
                continue
            job = state.get("jobs", {}).get(key, {})
            res = job.get("result") or {}
            raw = res.get("raw") or {}
            if (
                job.get("status") == "terminal"
                and raw.get("attempt_status") == "timeout"
                and len(job.get("attempts", [])) == 1
            ):
                seen_keys.add(key)
                _log(
                    f"Retrying single-attempt timeout job {job_def.get('task_id')} for "
                    f"{req['candidate']['id']} on exclusive GPU slot (workers_per_gpu=1)",
                    json_mode=json_mode,
                )
                queue.retry(
                    key,
                    reason=(
                        "rerun single-attempt wall-time timeout on exclusive GPU slot "
                        "(workers_per_gpu=1) without GPU-sharing contention"
                    ),
                )
                retried_keys.append(key)
    return retried_keys


def _ensure_attempt_coverage_markdown(root: Path, *, json_mode: bool = False) -> None:
    cov_path = Path(root) / "reports/forge/attempt-coverage.json"
    if not cov_path.is_file():
        return
    cov_doc = read_json(cov_path)
    if _attempt_coverage_markdown_synced(root, cov_doc):
        return
    _log("Regenerating reports/forge/ATTEMPT_COVERAGE.md from attempt-coverage.json", json_mode=json_mode)
    state = {
        "schema_version": 1,
        "jobs": {rec["key"]: _record_to_state_job(rec) for rec in cov_doc.get("records", [])},
        "maximum_tests": cov_doc.get("maximum_tests", 4),
        "maximum_per_gpu": cov_doc.get("maximum_per_gpu", 2),
        "include_research": True,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        attempt_coverage.publish(tmp_root, state)
        rendered_md = (tmp_root / "reports/forge/ATTEMPT_COVERAGE.md").read_text()
    atomic_text(Path(root) / "reports/forge/ATTEMPT_COVERAGE.md", rendered_md)


def _ensure_tier_report_markdown(root: Path) -> None:
    from experiments.forge.tier_report import REPORT_PATH, build_report, render_markdown as render_tiers

    target = Path(root) / REPORT_PATH
    inv_path = Path(root) / "reports/forge/technique-inventory.json"
    if target.is_file() or not inv_path.is_file():
        return
    publication = read_json(inv_path)
    if not publication.get("family_progress"):
        return
    report = build_report(root, publication=publication)
    atomic_text(target, render_tiers(report, root, target))


def refresh_compiled_reports(
    root: Path,
    *,
    cuda_model: str = DEFAULT_CUDA_MODEL,
    json_mode: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    _log("Refreshing technique-inventory.json/.md, family pages, problems.md, test-performance.md", json_mode=json_mode)
    pub_summary = _republish_registered_inventory(root, json_mode=json_mode)

    _ensure_tier_report_markdown(root)
    _ensure_attempt_coverage_markdown(root, json_mode=json_mode)

    if not history.validate_inventory(root).get("valid"):
        _log("Updating Forge history catalog coverage", json_mode=json_mode)
        history.import_history(root)

    readout_status = check_selected_family_readouts(root, cuda_model=cuda_model)
    missing_local_attempts = readout_status.get("local_attempts_missing_families", [])
    if not missing_local_attempts:
        _log("Compiling live goal leaderboards and automation telemetry", json_mode=json_mode)
        knowledge.compile_memory(root, summaries_only=False)
    else:
        _log(
            "Skipping full goal-leaderboard re-reduction (raw attempt envelopes absent on clean checkout; "
            "preserving committed goal leaderboards)",
            json_mode=json_mode,
        )

    _log("Sealing clean-checkout-safe EXPERIMENT_MEMORY.md and compilation.json", json_mode=json_mode)
    mem_summary = knowledge.compile_memory(root, summaries_only=True)
    return {
        "publication": pub_summary,
        "memory": mem_summary,
        "leaderboards_recompiled": not missing_local_attempts,
    }


def refresh_all(
    root: Path,
    *,
    queue_root: Path | None = None,
    attempt_state_root: Path | None = None,
    gpus: list[str] | None = None,
    scaling_gpu: str = "auto",
    cuda_model: str = DEFAULT_CUDA_MODEL,
    reports_only: bool = False,
    json_mode: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    queue_root = Path(queue_root or (root / "runs/forge")).resolve()
    attempt_state_root = Path(
        attempt_state_root or (queue_root / "attempt-coverage-refresh")
    ).resolve()
    raw_gpus = ["0", "1"] if gpus is None else gpus
    gpu_list = [str(g).strip() for g in raw_gpus if str(g).strip()]

    initial = check_status(root, queue_root=queue_root, cuda_model=cuda_model)
    if initial["status"] == "CURRENT":
        _log("All Forge readouts are already CURRENT; nothing to run.", json_mode=json_mode)
        return {"action": "NOOP", **initial}

    actions_executed: list[str] = []
    if not reports_only:
        if not initial["stages"]["compiled_reports"].get("publication_registration_synced", True):
            _republish_registered_inventory(root, json_mode=json_mode)
            actions_executed.append("publication_registration_sync")

        gpu_stages_stale = any(
            stage in initial["stale_stages"]
            for stage in ("queue", "scaling_performance", "attempt_coverage", "selected_family_readouts")
        )
        if gpu_stages_stale and not gpu_list:
            raise CodeChangeRequiredError(
                "at least one GPU index is required in --gpus when GPU refresh stages are stale"
            )

        q_info = initial["stages"]["queue"]
        if not q_info["current"]:
            queue = forge_queue.Queue(queue_root, report_root=root / "reports/forge")
            retried = _retry_shared_gpu_timeouts(queue, queue_root, root, json_mode=json_mode)
            if q_info["active_submissions"] or retried:
                _log(
                    f"Draining active queue submissions across GPUs {','.join(gpu_list)} (workers_per_gpu=1). "
                    f"Tail logs: tail -f {queue_root / 'events.jsonl'}",
                    json_mode=json_mode,
                )
                forge_queue.drain(queue, devices=gpu_list, workers_per_gpu=1, allow_sharing=False)
            _conclude_queue_submissions(root, queue_root, json_mode=json_mode)
            actions_executed.append("queue")

        if not check_scaling_performance(root)["current"]:
            refresh_scaling_performance(
                root,
                queue_root,
                scaling_gpu=str(scaling_gpu),
                gpus=gpu_list,
                cuda_model=cuda_model,
                json_mode=json_mode,
            )
            actions_executed.append("scaling_performance")

        if not check_attempt_coverage(root)["current"]:
            refresh_attempt_coverage(
                root,
                queue_root,
                attempt_state_root,
                gpus=gpu_list,
                json_mode=json_mode,
            )
            actions_executed.append("attempt_coverage")

        if not check_selected_family_readouts(root, cuda_model=cuda_model)["current"]:
            records = refresh_selected_family_readouts(
                root,
                queue_root,
                gpus=gpu_list,
                cuda_model=cuda_model,
                json_mode=json_mode,
            )
            if records:
                actions_executed.append("selected_family_readouts")

    if (
        actions_executed
        or not initial["stages"]["compiled_reports"]["current"]
        or not initial["stages"]["selected_family_readouts"]["current"]
    ):
        refresh_compiled_reports(root, cuda_model=cuda_model, json_mode=json_mode)
        actions_executed.append("compiled_reports")

    final = check_status(root, queue_root=queue_root, cuda_model=cuda_model)
    if final["status"] != "CURRENT":
        if reports_only:
            return {"action": "REPORTS_ONLY_PARTIAL", "executed_stages": actions_executed, **final}
        raise CodeChangeRequiredError(
            f"Forge readouts still report stale stages after full refresh: {final['stale_stages']}"
        )
    return {"action": "REFRESHED", "executed_stages": actions_executed, **final}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT, help="Repository root path")
    parser.add_argument("--queue-root", type=Path, default=None, help="Forge queue root (default: runs/forge)")
    parser.add_argument(
        "--attempt-state-root",
        type=Path,
        default=None,
        help="Attempt-coverage state directory (default: runs/forge/attempt-coverage-refresh)",
    )
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU indices for parallel stages")
    parser.add_argument(
        "--scaling-gpu",
        default="auto",
        help="Single GPU index (or 'auto' for an uncontended GPU from --gpus) for DCGAN scaling table",
    )
    parser.add_argument("--cuda-model", default=DEFAULT_CUDA_MODEL, help="Named CUDA hardware cohort")
    parser.add_argument("--check", action="store_true", help="Read-only check; exit 0 if current, 1 if stale, 2 if broken")
    parser.add_argument("--reports-only", action="store_true", help="Regenerate reports without GPU runs; with --check, inspect published integrity read-only and report current-source gaps separately")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON summary on stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gpus = [token.strip() for token in args.gpus.split(",") if token.strip()]
    try:
        if args.check:
            checker = check_published_status if args.reports_only else check_status
            report = checker(args.root, queue_root=args.queue_root, cuda_model=args.cuda_model)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                _log(f"Status: {report['status']} (stale stages: {report['stale_stages'] or 'none'})")
                if args.reports_only:
                    measured = report["current_source_measurements"]
                    _log(f"Scope: published reports; current-source measurements: {measured['status']} "
                         f"(stale stages: {measured['stale_stages'] or 'none'}). No qualification reuse.")
                if report["stale_stages"]:
                    print(json.dumps(report, indent=2, sort_keys=True))
            return CURRENT_EXIT if report["status"] == "CURRENT" else NEEDS_REFRESH_EXIT

        result = refresh_all(
            args.root,
            queue_root=args.queue_root,
            attempt_state_root=args.attempt_state_root,
            gpus=gpus,
            scaling_gpu=args.scaling_gpu,
            cuda_model=args.cuda_model,
            reports_only=args.reports_only,
            json_mode=args.json,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            _log(
                f"Completed ({result['action']}): executed stages={result.get('executed_stages', [])}, "
                f"status={result['status']}"
            )
        if args.reports_only and result["status"] != "CURRENT":
            return NEEDS_REFRESH_EXIT
        return CURRENT_EXIT
    except CodeChangeRequiredError as exc:
        payload = {"status": "CODE_CHANGE_REQUIRED", "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"[refresh-forge] CODE_CHANGE_REQUIRED: {exc}", file=sys.stderr, flush=True)
        return CODE_CHANGE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
