"""Export Forge inventory tables/JSON and publish atomically to Hugging Face.

Derived publication tables (Parquet/JSON) are read-only views for browsing and
filtering in the Hugging Face Dataset Viewer and Leaderboard Space. They never
modify qualification, launch training, or substitute for byte-exact archives.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

from .artifact_resolver import inspect_archive
from .artifacts import HF_REPO_PATTERN
from .contracts import atomic_json, read_json
from .family_reports import (
    TIERS,
    _cohort_tier_statuses,
    _count,
    _selected_cohorts,
    _selected_row,
    _sum,
    _summed_views,
    full_suite_tasks,
    is_stretch,
)
from .leaderboard_contract import (
    budget_text,
    family_search_budget,
    fixed_outcomes,
    holdout_claim,
    load_fixed_test_list,
    selection_lineage,
)

DEFAULT_HF_REPO = "HyperGAN/particlegan-forge-evidence"
DEFAULT_MEDIA_PREFIX = "media/forge/"
DEFAULT_RECEIPT_RELPATH = Path("reports/forge/publications/inventory-latest.json")
HF_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

SCIENTIFIC_CAVEATS = [
    (
        "Counts describe screening and selection evidence. Scientific superiority or "
        "public-default adoption requires report-bound accepted calibration and "
        "independently frozen confirmation evidence."
    ),
    "Software repairs grant neither; unmeasured confirmation remains unmeasured.",
    (
        "Not measured, blocked, incompatible, and other non-numerical outcomes are "
        "not scientific failures and earn no pass credit."
    ),
    "Rows with different tuning budgets are not equally tuned comparisons.",
]


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _git_head_commit(root: Path) -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha if sha else None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_file_last_commit(root: Path, rel_path: str) -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--", rel_path],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha if sha else None
    except (OSError, subprocess.CalledProcessError):
        return None


def _opt_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _opt_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _opt_str(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _opt_bool(value) -> bool | None:
    return value if isinstance(value, bool) else None


def _load_idea_meta(root: Path, cache: dict[str, dict | None], candidate_id: str) -> dict | None:
    if not candidate_id:
        return None
    if candidate_id in cache:
        return cache[candidate_id]
    path = root / "configs/forge/ideas" / f"{candidate_id}.json"
    if path.is_file():
        try:
            cache[candidate_id] = read_json(path)
        except Exception:
            cache[candidate_id] = None
    else:
        cache[candidate_id] = None
    return cache[candidate_id]


def _resolve_candidate_family(
    root: Path,
    candidate_id: str,
    cand_to_family: dict[str, str],
    idea_cache: dict[str, dict | None],
) -> str | None:
    if candidate_id in cand_to_family:
        return cand_to_family[candidate_id]
    idea = _load_idea_meta(root, idea_cache, candidate_id)
    if isinstance(idea, dict):
        return _opt_str(idea.get("trainer_family"))
    return None


def _resolve_candidate_optimizer_family(
    root: Path,
    candidate_id: str,
    idea_cache: dict[str, dict | None],
) -> str | None:
    idea = _load_idea_meta(root, idea_cache, candidate_id)
    if isinstance(idea, dict):
        overrides = idea.get("recipe_overrides")
        if isinstance(overrides, dict):
            return _opt_str(overrides.get("optimizer_family"))
    return None


def _extract_receipt_energy(cost: dict, evaluator_summary: dict | None = None) -> dict:
    """Extract nullable hardware energy telemetry from a receipt's ``cost`` dict.

    Historical receipts without ``cost["energy"]`` remain strictly ``None`` across
    all energy fields and are never fabricated.
    """
    empty = {
        "joules_total": None,
        "joules_per_step": None,
        "avg_power_watts": None,
        "first_pass_step": None,
        "sustained_pass_step": None,
        "joules_to_first_pass": None,
        "joules_to_sustained_pass": None,
        "energy_backend": None,
        "gpu_name": None,
        "contended": None,
    }
    energy = cost.get("energy") if isinstance(cost, dict) else None
    if not isinstance(energy, dict):
        return empty

    joules_total = _opt_float(energy.get("joules_total"))
    joules_per_step = _opt_float(energy.get("joules_per_step"))
    avg_power_watts = _opt_float(energy.get("avg_power_watts"))
    if joules_total is None and joules_per_step is None and avg_power_watts is None:
        return empty

    steps = _opt_int(cost.get("completed_steps"))
    if joules_per_step is None and joules_total is not None and steps is not None and steps > 0:
        joules_per_step = joules_total / float(steps)

    fp_step = _opt_int(energy.get("first_pass_step"))
    sp_step = _opt_int(energy.get("sustained_pass_step"))
    if isinstance(evaluator_summary, dict):
        conv = evaluator_summary.get("convergence")
        if isinstance(conv, dict):
            if fp_step is None:
                fp_step = _opt_int(conv.get("first_pass_step"))
            if sp_step is None:
                sp_step = _opt_int(
                    conv.get("confirmed_step")
                    if conv.get("confirmed_step") is not None
                    else conv.get("stable_from_step")
                )

    j_fp = _opt_float(energy.get("joules_to_first_pass"))
    if j_fp is None and joules_per_step is not None and fp_step is not None:
        j_fp = joules_per_step * float(fp_step)

    j_sp = _opt_float(energy.get("joules_to_sustained_pass"))
    if j_sp is None and joules_per_step is not None and sp_step is not None:
        j_sp = joules_per_step * float(sp_step)

    return {
        "joules_total": joules_total,
        "joules_per_step": joules_per_step,
        "avg_power_watts": avg_power_watts,
        "first_pass_step": fp_step,
        "sustained_pass_step": sp_step,
        "joules_to_first_pass": j_fp,
        "joules_to_sustained_pass": j_sp,
        "energy_backend": _opt_str(energy.get("energy_backend")),
        "gpu_name": _opt_str(energy.get("gpu_name")),
        "contended": _opt_bool(energy.get("contended")),
    }


def _readout_occam_counts(root: Path, readout_name: str | None) -> tuple[int | None, int | None, str | None]:
    if not isinstance(readout_name, str) or not readout_name.strip():
        return None, None, None
    path = root / "reports/forge/records" / readout_name.strip()
    if not path.is_file():
        return None, None, None
    try:
        doc = read_json(path)
    except Exception:
        return None, None, None
    prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    src_commit = _opt_str(prov.get("source_origin_commit")) or _opt_str(doc.get("source_commit"))
    task_results = doc.get("task_results")
    if not isinstance(task_results, list) or not task_results:
        return None, None, src_commit
    executed_statuses = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "ERROR", "TIMEOUT"}
    passed = sum(
        1 for tr in task_results
        if isinstance(tr, dict) and (tr.get("gate_status") or tr.get("status")) == "PASS"
    )
    executed = sum(
        1 for tr in task_results
        if isinstance(tr, dict) and (tr.get("gate_status") or tr.get("status")) in executed_statuses
    )
    return (passed if executed > 0 else None), (executed if executed > 0 else None), src_commit


def _infer_contended(entry: dict, doc: dict, r256: dict) -> bool | None:
    for container in (entry, r256, doc):
        if isinstance(container, dict) and isinstance(container.get("contended"), bool):
            return container["contended"]
    util_before = r256.get("gpu_util_before") if isinstance(r256, dict) else None
    if (
        isinstance(util_before, (list, tuple))
        and len(util_before) >= 1
        and isinstance(util_before[0], (int, float))
        and not isinstance(util_before[0], bool)
    ):
        return bool(util_before[0] > 5)
    return None


def _collect_study_energy_rows(
    root: Path,
    cand_to_family: dict[str, str],
    idea_cache: dict[str, dict | None],
) -> list[dict]:
    rows: list[dict] = []
    studies_dir = root / "reports/forge/studies"
    if not studies_dir.is_dir():
        return rows

    for study_file in sorted(studies_dir.glob("*energy-benchmark*.json")):
        try:
            doc = read_json(study_file)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        rel_source = f"reports/forge/studies/{study_file.name}"
        file_commit = (
            _opt_str(doc.get("source_commit"))
            or _opt_str(doc.get("git_commit"))
            or _git_file_last_commit(root, rel_source)
        )
        default_gpu = _opt_str(doc.get("gpu_name"))
        default_scope = _opt_str(doc.get("evidence_scope")) or "study_energy_benchmark"

        bench_map = doc.get("energy_benchmark")
        if isinstance(bench_map, dict):
            for candidate_id in sorted(bench_map):
                entry = bench_map[candidate_id]
                if not isinstance(entry, dict):
                    continue
                r256 = entry.get("dcgan_r256") if isinstance(entry.get("dcgan_r256"), dict) else {}
                ring16 = entry.get("ring16_acquisition") if isinstance(entry.get("ring16_acquisition"), dict) else {}
                grid100 = entry.get("grid100") if isinstance(entry.get("grid100"), dict) else {}
                staggered100 = entry.get("staggered100") if isinstance(entry.get("staggered100"), dict) else {}

                sustained = (
                    r256.get("sustained_uninstrumented")
                    if isinstance(r256.get("sustained_uninstrumented"), dict)
                    else {}
                )
                step_ms_obj = sustained.get("step_ms") if isinstance(sustained.get("step_ms"), dict) else {}
                r256_ms = _opt_float(step_ms_obj.get("median"))
                if r256_ms is None:
                    r256_ms = _opt_float(step_ms_obj.get("mean"))
                if r256_ms is None:
                    r256_ms = _opt_float(r256.get("ms_per_step"))

                power_poll = (
                    sustained.get("power_poll_watts")
                    if isinstance(sustained.get("power_poll_watts"), dict)
                    else {}
                )
                r256_watts = _opt_float(sustained.get("avg_power_watts_from_energy"))
                if r256_watts is None:
                    r256_watts = _opt_float(power_poll.get("median"))
                if r256_watts is None:
                    r256_watts = _opt_float(r256.get("avg_power_watts"))

                r256_j = _opt_float(sustained.get("joules_per_step"))
                if r256_j is None:
                    r256_j = _opt_float(r256.get("joules_per_step"))

                opt_state = r256.get("optimizer_state") if isinstance(r256.get("optimizer_state"), dict) else {}
                state_bpp = _opt_float(opt_state.get("state_bytes_per_param"))
                if state_bpp is None:
                    state_bpp = _opt_float(grid100.get("state_bytes_per_param"))
                if state_bpp is None:
                    state_bpp = _opt_float(entry.get("state_bytes_per_param"))

                opt_family = (
                    _opt_str(r256.get("optimizer_family"))
                    or _opt_str(grid100.get("optimizer_family"))
                    or _opt_str(ring16.get("optimizer_family"))
                    or _opt_str(entry.get("optimizer_family"))
                    or _resolve_candidate_optimizer_family(root, candidate_id, idea_cache)
                )

                ro_pass, ro_exec, ro_commit = _readout_occam_counts(root, grid100.get("official_forge_readout"))
                occam_pass = _opt_int(entry.get("occam_pass"))
                occam_executed = _opt_int(entry.get("occam_executed"))
                if occam_pass is None:
                    occam_pass = ro_pass
                if occam_executed is None:
                    occam_executed = ro_exec

                src_commit = _opt_str(entry.get("source_commit")) or ro_commit or file_commit
                gpu = _opt_str(r256.get("gpu_name")) or _opt_str(entry.get("hardware_gpu")) or default_gpu
                contended = _infer_contended(entry, doc, r256)
                scope = _opt_str(entry.get("evidence_scope")) or default_scope

                rows.append(
                    {
                        "candidate_id": str(candidate_id),
                        "family_id": _opt_str(entry.get("family_id"))
                        or _resolve_candidate_family(root, candidate_id, cand_to_family, idea_cache),
                        "optimizer_family": opt_family,
                        "hardware_gpu": gpu,
                        "contended": contended,
                        "source_commit": src_commit,
                        "source_file": rel_source,
                        "r256_ms_per_step": r256_ms,
                        "r256_watts": r256_watts,
                        "r256_joules_per_step": r256_j,
                        "state_bytes_per_param": state_bpp,
                        "ring16_first_pass_step": _opt_int(ring16.get("first_pass_step")),
                        "ring16_j_to_pass": _opt_float(ring16.get("joules_to_first_pass")),
                        "grid100_first_pass_step": _opt_int(grid100.get("first_pass_step")),
                        "grid100_sustained_pass_step": _opt_int(
                            grid100.get("first_sustained_5check_step")
                            if grid100.get("first_sustained_5check_step") is not None
                            else grid100.get("sustained_pass_step")
                        ),
                        "grid100_j_to_pass": _opt_float(grid100.get("joules_to_first_pass")),
                        "grid100_j_to_sustained_pass": _opt_float(grid100.get("joules_to_sustained_pass")),
                        "staggered100_first_pass_step": _opt_int(staggered100.get("first_pass_step")),
                        "staggered100_j_to_pass": _opt_float(staggered100.get("joules_to_first_pass")),
                        "occam_pass": occam_pass,
                        "occam_executed": occam_executed,
                        "evidence_scope": scope,
                    }
                )
        elif isinstance(doc.get("benchmarks"), list):
            for entry in doc["benchmarks"]:
                if not isinstance(entry, dict) or not _opt_str(entry.get("candidate_id")):
                    continue
                candidate_id = str(entry["candidate_id"]).strip()
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "family_id": _opt_str(entry.get("family_id"))
                        or _resolve_candidate_family(root, candidate_id, cand_to_family, idea_cache),
                        "optimizer_family": _opt_str(entry.get("optimizer_family"))
                        or _resolve_candidate_optimizer_family(root, candidate_id, idea_cache),
                        "hardware_gpu": _opt_str(entry.get("hardware_gpu")) or default_gpu,
                        "contended": _opt_bool(entry.get("contended")),
                        "source_commit": _opt_str(entry.get("source_commit")) or file_commit,
                        "source_file": rel_source,
                        "r256_ms_per_step": _opt_float(entry.get("r256_ms_per_step")),
                        "r256_watts": _opt_float(entry.get("r256_watts")),
                        "r256_joules_per_step": _opt_float(entry.get("r256_joules_per_step")),
                        "state_bytes_per_param": _opt_float(entry.get("state_bytes_per_param")),
                        "ring16_first_pass_step": _opt_int(entry.get("ring16_first_pass_step")),
                        "ring16_j_to_pass": _opt_float(entry.get("ring16_j_to_pass")),
                        "grid100_first_pass_step": _opt_int(entry.get("grid100_first_pass_step")),
                        "grid100_sustained_pass_step": _opt_int(entry.get("grid100_sustained_pass_step")),
                        "grid100_j_to_pass": _opt_float(entry.get("grid100_j_to_pass")),
                        "grid100_j_to_sustained_pass": _opt_float(entry.get("grid100_j_to_sustained_pass")),
                        "staggered100_first_pass_step": _opt_int(entry.get("staggered100_first_pass_step")),
                        "staggered100_j_to_pass": _opt_float(entry.get("staggered100_j_to_pass")),
                        "occam_pass": _opt_int(entry.get("occam_pass")),
                        "occam_executed": _opt_int(entry.get("occam_executed")),
                        "evidence_scope": _opt_str(entry.get("evidence_scope")) or default_scope,
                    }
                )
    return rows


def _collect_receipt_energy_rows(
    root: Path,
    receipt_energy_entries: list[dict],
    cand_to_family: dict[str, str],
    idea_cache: dict[str, dict | None],
) -> list[dict]:
    """Aggregate Forge technique receipts carrying ``cost.energy`` without mixing cohorts."""
    grouped: dict[tuple, dict] = {}
    for item in receipt_energy_entries:
        candidate_id = item["candidate_id"]
        key = (
            candidate_id,
            item["source_commit"],
            item["source_digest"],
            item["gpu_name"],
            item["contended"],
            item["evidence_scope"],
        )
        row = grouped.get(key)
        if row is None:
            row = {
                "candidate_id": candidate_id,
                "family_id": _resolve_candidate_family(root, candidate_id, cand_to_family, idea_cache),
                "optimizer_family": _resolve_candidate_optimizer_family(root, candidate_id, idea_cache),
                "hardware_gpu": item["gpu_name"],
                "contended": item["contended"],
                "source_commit": item["source_commit"],
                "source_file": item["receipt_file"],
                "r256_ms_per_step": None,
                "r256_watts": None,
                "r256_joules_per_step": None,
                "state_bytes_per_param": None,
                "ring16_first_pass_step": None,
                "ring16_j_to_pass": None,
                "grid100_first_pass_step": None,
                "grid100_sustained_pass_step": None,
                "grid100_j_to_pass": None,
                "grid100_j_to_sustained_pass": None,
                "staggered100_first_pass_step": None,
                "staggered100_j_to_pass": None,
                "occam_pass": None,
                "occam_executed": None,
                "evidence_scope": item["evidence_scope"] or "forge_receipt_energy",
                "_source_digest": item["source_digest"],
                "_attempt_ids": set(),
            }
            grouped[key] = row
        row["_attempt_ids"].add(item["attempt_id"])
        task_id = item["task_id"]
        if task_id == "dcgan_performance_r256":
            if row["r256_joules_per_step"] is None:
                row["r256_joules_per_step"] = item["joules_per_step"]
            if row["r256_watts"] is None:
                row["r256_watts"] = item["avg_power_watts"]
            if row["r256_ms_per_step"] is None and item["wall_seconds"] is not None and item["completed_steps"]:
                row["r256_ms_per_step"] = 1000.0 * item["wall_seconds"] / float(item["completed_steps"])
        elif task_id == "ring16_acquisition":
            if row["ring16_first_pass_step"] is None:
                row["ring16_first_pass_step"] = item["first_pass_step"]
            if row["ring16_j_to_pass"] is None:
                row["ring16_j_to_pass"] = item["joules_to_first_pass"]
        elif task_id == "grid100":
            if row["grid100_first_pass_step"] is None:
                row["grid100_first_pass_step"] = item["first_pass_step"]
            if row["grid100_sustained_pass_step"] is None:
                row["grid100_sustained_pass_step"] = item["sustained_pass_step"]
            if row["grid100_j_to_pass"] is None:
                row["grid100_j_to_pass"] = item["joules_to_first_pass"]
            if row["grid100_j_to_sustained_pass"] is None:
                row["grid100_j_to_sustained_pass"] = item["joules_to_sustained_pass"]
        elif task_id == "staggered100":
            if row["staggered100_first_pass_step"] is None:
                row["staggered100_first_pass_step"] = item["first_pass_step"]
            if row["staggered100_j_to_pass"] is None:
                row["staggered100_j_to_pass"] = item["joules_to_first_pass"]
    return [grouped[k] for k in sorted(grouped, key=lambda item: tuple("" if x is None else str(x) for x in item))]


def _gif_to_media_relpath(gif_path: str | None) -> str | None:
    if not isinstance(gif_path, str) or not gif_path:
        return None
    normalized = gif_path.replace("\\", "/").lstrip("./")
    if normalized.startswith("reports/forge/"):
        return DEFAULT_MEDIA_PREFIX + normalized[len("reports/forge/"):]
    if normalized.startswith("media/"):
        return normalized
    return DEFAULT_MEDIA_PREFIX + normalized


def _collect_media_index(progress: dict) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Index training GIFs by (family_id, task_id) and (attempt_id, task_id)."""
    by_family_task: dict[tuple[str, str], str] = {}
    by_attempt_task: dict[tuple[str, str], str] = {}
    families = list(progress.get("families", [])) + list(progress.get("configuration_families", []))
    for family in families:
        fam_id = family.get("id", "")
        for cohort in family.get("cohorts", []):
            for task_id, cell in cohort.get("tasks", {}).items():
                for artifact in cell.get("training_media", []) or []:
                    rel = _gif_to_media_relpath(artifact.get("gif"))
                    if not rel:
                        continue
                    by_family_task.setdefault((fam_id, task_id), rel)
                    attempt_id = artifact.get("attempt_id") or cell.get("attempt_id")
                    if attempt_id:
                        by_attempt_task.setdefault((attempt_id, task_id), rel)
    return by_family_task, by_attempt_task


def _tier_bucket(tier_data: dict) -> dict:
    counts = tier_data.get("counts", {})
    coverage = tier_data.get("attempt_coverage", {})
    not_run_cov = coverage.get("not_run")
    blocked = int(coverage.get("blocked", 0))
    diagnostic = int(coverage.get("diagnostic_attempts", 0))
    incompatible = int(coverage.get("incompatible", tier_data.get("incompatible", 0)))
    not_run = int(not_run_cov) if not_run_cov is not None else int(tier_data.get("not_run", 0))
    return {
        "pass": int(tier_data.get("passed", counts.get("PASS", 0))),
        "total": int(tier_data.get("total", 0)),
        "fail": int(counts.get("FAIL", 0)),
        "not_run": not_run,
        "blocked": blocked,
        "diagnostic": diagnostic,
        "incompatible": incompatible,
    }


def _stale_family_ids(publication: dict) -> set[str]:
    proofs = publication.get("retained_recorded_measurements", {}).get("proofs", [])
    return {p["trainer_family"] for p in proofs if isinstance(p, dict) and p.get("trainer_family")}


def _collect_task_definitions(progress: dict, full_suite_names: list[str]) -> list[dict]:
    summed = _summed_views(progress)
    full_suite_set = set(full_suite_names)
    problem_intents = {}
    for entry in progress.get("problem_definitions", {}).get("problems", []):
        if isinstance(entry, dict):
            # Some problem entries list tasks or an anchor matching the task id.
            for evidence in entry.get("evidence", []) or []:
                anchor = evidence.get("anchor")
                if anchor and entry.get("intent"):
                    problem_intents.setdefault(anchor, entry["intent"])
            if entry.get("task") and entry.get("intent"):
                problem_intents.setdefault(entry["task"], entry["intent"])

    by_task: dict[str, dict] = {}
    role_priority = {"required": 0, "stretch": 1, "held": 2, "removed": 3}

    for view in summed:
        view_id = view["id"]
        for assignment in view.get("assignments", []):
            task_id = assignment["task"]
            tier = int(assignment["qualification_tier"])
            if "removal" in assignment:
                role = "removed"
                note = assignment["removal"]
            elif assignment.get("importance") == "held":
                role = "held"
                note = assignment.get("hold", {})
            elif assignment.get("importance") == "stretch":
                role = "stretch"
                note = assignment.get("stretch", {})
            else:
                role = "required"
                note = {}
            rec = by_task.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "tier": tier,
                    "role": role,
                    "views": [],
                    "summary": note.get("summary") or problem_intents.get(task_id) or "",
                    "re_entry_rule": note.get("reentry") or note.get("graduation"),
                    "in_full_suite": task_id in full_suite_set,
                },
            )
            rec["tier"] = min(rec["tier"], tier)
            if role_priority[role] < role_priority[rec["role"]]:
                rec["role"] = role
            view_label = f"{view_id} (Tier {tier})"
            if view_label not in rec["views"]:
                rec["views"].append(view_label)
            if not rec["summary"] and note.get("summary"):
                rec["summary"] = note["summary"]
            if not rec["re_entry_rule"] and (note.get("reentry") or note.get("graduation")):
                rec["re_entry_rule"] = note.get("reentry") or note.get("graduation")

    for task_id in full_suite_names:
        by_task.setdefault(
            task_id,
            {
                "task_id": task_id,
                "tier": 0,
                "role": "required",
                "views": [],
                "summary": problem_intents.get(task_id) or "",
                "re_entry_rule": None,
                "in_full_suite": True,
            },
        )
    return [by_task[k] for k in sorted(by_task)]


def _compact_metrics_summary(cell: dict) -> dict:
    raw_metrics = cell.get("metrics")
    if not isinstance(raw_metrics, dict):
        return {}
    # Technique-inventory task cells sometimes nest the task_result dict under "metrics".
    inner = raw_metrics.get("metrics") if isinstance(raw_metrics.get("metrics"), dict) else raw_metrics
    summary = {}
    for key, val in sorted(inner.items()):
        if isinstance(val, (int, float, bool, str)) or val is None:
            summary[key] = val
    return summary


def build_inventory_exports(
    root: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    generated_at: str | None = None,
) -> dict:
    """Build structured leaderboard, receipt, and attempt datasets from committed Forge reports."""
    root = Path(root).resolve()
    inv_path = root / "reports/forge/technique-inventory.json"
    if not inv_path.is_file():
        raise FileNotFoundError(f"missing Forge inventory: {inv_path}")
    publication = read_json(inv_path)
    progress = publication["family_progress"]
    report_tiers = tuple(str(t) for t in progress.get("tiers", TIERS))
    summed = _summed_views(progress)
    selected = list(_selected_cohorts(progress))
    stale_families = _stale_family_ids(publication)
    media_by_family_task, media_by_attempt_task = _collect_media_index(progress)

    fixed = load_fixed_test_list(root)
    full_suite_names = list(fixed["tests"]) if fixed else full_suite_tasks(progress)
    counted_task_set = {
        a["task"] for view in summed for a in view["assignments"] if a["importance"] == "required"
    }
    not_counted_names = [name for name in full_suite_names if name not in counted_task_set]

    assignments = [a for view in summed for a in view["assignments"]]
    counted_reqs = sum(a["importance"] == "required" for a in assignments)
    distinct_required = len(counted_task_set)
    stretch_assignments = [a for a in assignments if is_stretch(a)]
    stretch_distinct = len({a["task"] for a in stretch_assignments})
    held_assignments = [a for a in assignments if a["importance"] == "held"]
    held_distinct = len({a["task"] for a in held_assignments})
    removed_assignments = [a for a in assignments if "removal" in a]
    removed_distinct = len({a["task"] for a in removed_assignments})

    task_defs = _collect_task_definitions(progress, full_suite_names)
    task_def_by_id = {d["task_id"]: d for d in task_defs}

    cand_to_family: dict[str, str] = {}
    for family, cohort in selected:
        sel_row = _selected_row(publication, cohort)
        cid = sel_row.get("candidate_id") or family.get("benchmark_configuration", {}).get("candidate_id") or ""
        if cid:
            cand_to_family[str(cid)] = family["id"]
    idea_cache: dict[str, dict | None] = {}

    # Receipts table from reports/forge/technique-receipts/*.json
    receipt_rows: list[dict] = []
    receipt_energy_by_attempt_task: dict[tuple[str, str], dict] = {}
    receipt_energy_entries: list[dict] = []
    receipts_dir = root / "reports/forge/technique-receipts"
    if receipts_dir.is_dir():
        for rec_file in sorted(receipts_dir.glob("*.json")):
            doc = read_json(rec_file)
            attempt_id = str(doc.get("attempt_id") or rec_file.stem)
            provenance = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
            canonical_hash = str(provenance.get("canonical_result_hash") or "")
            source_commit = provenance.get("source_origin_commit")
            source_digest = provenance.get("source_digest")
            receipt_rel = f"reports/forge/technique-receipts/{rec_file.name}"
            task_results = doc.get("task_results") or []
            if not task_results:
                receipt_rows.append(
                    {
                        "attempt_id": attempt_id,
                        "request_id": str(doc.get("request_id") or ""),
                        "campaign_id": str(doc.get("campaign_id") or ""),
                        "candidate_id": str(doc.get("candidate_id") or ""),
                        "candidate_revision": str(doc.get("candidate_revision") or ""),
                        "evidence_scope": str(doc.get("evidence_scope") or ""),
                        "attempt_status": str(doc.get("attempt_status") or ""),
                        "task_id": "",
                        "status": str(doc.get("attempt_status") or "UNKNOWN"),
                        "gate_status": "",
                        "raw_status": "",
                        "reason": str(doc.get("note") or ""),
                        "compatibility_key": "",
                        "execution_path": "",
                        "device": "",
                        "wall_seconds": None,
                        "completed_steps": None,
                        "joules_total": None,
                        "joules_per_step": None,
                        "avg_power_watts": None,
                        "first_pass_step": None,
                        "sustained_pass_step": None,
                        "joules_to_first_pass": None,
                        "joules_to_sustained_pass": None,
                        "energy_backend": None,
                        "canonical_result_hash": canonical_hash,
                        "source_commit": str(source_commit) if source_commit is not None else None,
                        "source_digest": str(source_digest) if source_digest is not None else None,
                        "sampling_law": None,
                        "eval_output_noise": None,
                        "metrics_json": "{}",
                        "media_gif_relpath": None,
                        "receipt_file": receipt_rel,
                    }
                )
            for tr in task_results:
                if not isinstance(tr, dict):
                    continue
                task_id = str(tr.get("task_id") or "")
                cost = tr.get("cost") if isinstance(tr.get("cost"), dict) else {}
                sampling = tr.get("sampling") if isinstance(tr.get("sampling"), dict) else {}
                wall_sec = cost.get("wall_seconds")
                steps = cost.get("completed_steps")
                metrics_obj = tr.get("metrics") if isinstance(tr.get("metrics"), dict) else {}
                eval_summary = tr.get("evaluator_summary") if isinstance(tr.get("evaluator_summary"), dict) else None
                energy_info = _extract_receipt_energy(cost, eval_summary)
                has_energy = (
                    energy_info["joules_total"] is not None
                    or energy_info["joules_per_step"] is not None
                    or energy_info["avg_power_watts"] is not None
                )
                wall_float = float(wall_sec) if isinstance(wall_sec, (int, float)) and not isinstance(wall_sec, bool) else None
                steps_int = int(steps) if isinstance(steps, int) and not isinstance(steps, bool) else None
                if has_energy:
                    receipt_energy_by_attempt_task[(attempt_id, task_id)] = dict(energy_info)
                    receipt_energy_entries.append(
                        {
                            "attempt_id": attempt_id,
                            "candidate_id": str(doc.get("candidate_id") or ""),
                            "task_id": task_id,
                            "evidence_scope": str(doc.get("evidence_scope") or ""),
                            "source_commit": str(source_commit) if source_commit is not None else None,
                            "source_digest": str(source_digest) if source_digest is not None else None,
                            "wall_seconds": wall_float,
                            "completed_steps": steps_int,
                            "receipt_file": receipt_rel,
                            **energy_info,
                        }
                    )
                receipt_rows.append(
                    {
                        "attempt_id": attempt_id,
                        "request_id": str(doc.get("request_id") or ""),
                        "campaign_id": str(doc.get("campaign_id") or ""),
                        "candidate_id": str(doc.get("candidate_id") or ""),
                        "candidate_revision": str(doc.get("candidate_revision") or ""),
                        "evidence_scope": str(doc.get("evidence_scope") or ""),
                        "attempt_status": str(doc.get("attempt_status") or ""),
                        "task_id": task_id,
                        "status": str(tr.get("status") or ""),
                        "gate_status": str(tr.get("gate_status") or ""),
                        "raw_status": str(tr.get("raw_status") or ""),
                        "reason": str(tr.get("reason") or ""),
                        "compatibility_key": str(tr.get("compatibility_key") or ""),
                        "execution_path": str(tr.get("execution_path") or ""),
                        "device": str(cost.get("device") or tr.get("device") or ""),
                        "wall_seconds": wall_float,
                        "completed_steps": steps_int,
                        "joules_total": energy_info["joules_total"],
                        "joules_per_step": energy_info["joules_per_step"],
                        "avg_power_watts": energy_info["avg_power_watts"],
                        "first_pass_step": energy_info["first_pass_step"],
                        "sustained_pass_step": energy_info["sustained_pass_step"],
                        "joules_to_first_pass": energy_info["joules_to_first_pass"],
                        "joules_to_sustained_pass": energy_info["joules_to_sustained_pass"],
                        "energy_backend": energy_info["energy_backend"],
                        "canonical_result_hash": canonical_hash,
                        "source_commit": str(source_commit) if source_commit is not None else None,
                        "source_digest": str(source_digest) if source_digest is not None else None,
                        "sampling_law": str(sampling.get("sampling_law")) if sampling.get("sampling_law") is not None else None,
                        "eval_output_noise": str(sampling.get("eval_output_noise")) if sampling.get("eval_output_noise") is not None else None,
                        "metrics_json": json.dumps(metrics_obj, sort_keys=True, separators=(",", ":")),
                        "media_gif_relpath": media_by_attempt_task.get((attempt_id, task_id)),
                        "receipt_file": receipt_rel,
                    }
                )

    study_energy_rows = _collect_study_energy_rows(root, cand_to_family, idea_cache)
    receipt_energy_rows_raw = _collect_receipt_energy_rows(root, receipt_energy_entries, cand_to_family, idea_cache)
    energy_public_keys = (
        "candidate_id",
        "family_id",
        "optimizer_family",
        "hardware_gpu",
        "contended",
        "source_commit",
        "source_file",
        "r256_ms_per_step",
        "r256_watts",
        "r256_joules_per_step",
        "state_bytes_per_param",
        "ring16_first_pass_step",
        "ring16_j_to_pass",
        "grid100_first_pass_step",
        "grid100_sustained_pass_step",
        "grid100_j_to_pass",
        "grid100_j_to_sustained_pass",
        "staggered100_first_pass_step",
        "staggered100_j_to_pass",
        "occam_pass",
        "occam_executed",
        "evidence_scope",
    )
    energy_rows: list[dict] = [
        {k: row.get(k) for k in energy_public_keys}
        for row in (*study_energy_rows, *receipt_energy_rows_raw)
    ]

    def _match_family_energy(
        candidate_id: str,
        backend: str,
        source_digest: str,
        source_origin_commit: str,
        cohort_attempt_ids: set[str],
    ) -> tuple[float | None, float | None, float | None]:
        """Match energy summary strictly within the same candidate/backend/source cohort."""
        if not candidate_id or backend != "cuda":
            return None, None, None
        matched_r256: float | None = None
        matched_ring16: float | None = None
        matched_grid100: float | None = None
        for row in (*receipt_energy_rows_raw, *study_energy_rows):
            if row.get("candidate_id") != candidate_id:
                continue
            if row.get("contended") is True:
                continue
            row_attempts = row.get("_attempt_ids") or set()
            row_digest = row.get("_source_digest")
            row_commit = row.get("source_commit")
            provenance_match = bool(
                (cohort_attempt_ids and row_attempts and (cohort_attempt_ids & row_attempts))
                or (source_digest and row_digest and row_digest == source_digest)
                or (source_origin_commit and row_commit and row_commit == source_origin_commit)
            )
            if not provenance_match:
                continue
            if matched_r256 is None and row.get("r256_joules_per_step") is not None:
                matched_r256 = row["r256_joules_per_step"]
            if matched_ring16 is None and row.get("ring16_j_to_pass") is not None:
                matched_ring16 = row["ring16_j_to_pass"]
            if matched_grid100 is None and row.get("grid100_j_to_pass") is not None:
                matched_grid100 = row["grid100_j_to_pass"]
        return matched_r256, matched_ring16, matched_grid100

    leaderboard_rows: list[dict] = []
    families_json: list[dict] = []
    family_tasks_json: list[dict] = []

    config_families = {
        item["id"]: item for item in progress.get("configuration_families", []) if isinstance(item, dict)
    }

    for family, cohort in selected:
        fam_id = family["id"]
        fam_label = family["label"]
        row = _selected_row(publication, cohort)
        bindings = row.get("bindings", {}) if isinstance(row.get("bindings"), dict) else {}
        cfg_fam_id = cohort.get("configuration_family")
        cfg_fam = config_families.get(cfg_fam_id) if cfg_fam_id else None
        selected_recipe_label = cfg_fam["label"] if cfg_fam else fam_label

        is_stale = fam_id in stale_families or (cfg_fam_id in stale_families if cfg_fam_id else False)
        stale_note = "Recorded source measurement · stale; no current qualification" if is_stale else None

        actual_qual = row.get("executed_policy_qualification")
        qual_note = None
        if actual_qual:
            qual_note = (
                f"comparison policy revision {publication.get('view_revision')}, "
                f"tier {row.get('qualified_tier', 0)}; actual executed-policy qualification "
                f"tier {actual_qual.get('qualified_tier')}, revision {actual_qual.get('view_revision')} "
                f"({actual_qual.get('status')})."
            )

        tiers_dict = {t: _tier_bucket(cohort.get("tiers", {}).get(t, {})) for t in ("1", "2", "3", "4")}
        total_dict = _tier_bucket(cohort.get("total", {}))

        statuses = {
            test: cohort.get("tasks", {}).get(test, {}).get("status", "UNKNOWN") for test in full_suite_names
        }
        fs_counts = fixed_outcomes(statuses, full_suite_names, not_counted_names)
        fs_other_total = sum(fs_counts["other"].values())
        fs_held = fs_counts["held"]
        fs_held_other = fs_held["total"] - fs_held["passed"] - fs_held["failed"] - fs_held["not_measured"]

        candidate_id = row.get("candidate_id") or family.get("benchmark_configuration", {}).get("candidate_id") or ""
        lineage = selection_lineage(root, candidate_id) if candidate_id else {"search_id": None}
        tuning_budget = budget_text(family_search_budget(root, fam_id), lineage)
        holdout_text = None
        if lineage.get("search_id"):
            claim = holdout_claim(_cohort_tier_statuses(progress, cohort), lineage.get("tuning_through_tier"))
            holdout_text = claim.get("claim")

        stretch_tier_buckets = {}
        stretch_tier_raw = {}
        for t in ("1", "2", "3", "4"):
            raw = _count(
                [
                    cohort["tasks"].get(a["task"], {"status": "UNKNOWN"})
                    for view in summed
                    for a in view["assignments"]
                    if is_stretch(a) and str(a["qualification_tier"]) == t
                ]
            )
            stretch_tier_raw[t] = raw
            stretch_tier_buckets[t] = _tier_bucket(raw)
        stretch_total_bucket = _tier_bucket(_sum([stretch_tier_raw[t] for t in report_tiers if t in stretch_tier_raw]))

        recipe_sha = bindings.get("recipe_sha256") or ""
        source_digest = bindings.get("source_digest") or ""
        source_origin_commit = bindings.get("source_origin_commit") or ""
        cand_rev = row.get("candidate_revision") or ""
        backend = cohort.get("backend") or "cuda"
        anchor = cohort.get("anchor") or ""
        family_page = family.get("page") or ""

        cohort_tasks = cohort.get("tasks", {})
        cohort_attempt_ids = {
            str(aid)
            for cell in cohort_tasks.values()
            if isinstance(cell, dict)
            for aid in (cell.get("attempt_id"), (cell.get("run_attempt") or {}).get("attempt_id"))
            if aid
        }
        fam_r256_j, fam_ring16_j, fam_grid100_j = _match_family_energy(
            candidate_id, backend, source_digest, source_origin_commit, cohort_attempt_ids
        )

        families_json.append(
            {
                "id": fam_id,
                "label": fam_label,
                "selected_recipe": selected_recipe_label,
                "configuration_family": cfg_fam_id,
                "candidate_id": candidate_id,
                "candidate_revision": cand_rev,
                "recipe_sha256": recipe_sha,
                "source_digest": source_digest,
                "source_origin_commit": source_origin_commit,
                "backend": backend,
                "cohort_anchor": anchor,
                "family_page": family_page,
                "stale": is_stale,
                "stale_note": stale_note,
                "qualified_tier": int(row.get("qualified_tier") or 0),
                "qualification_note": qual_note,
                "tuning_budget": tuning_budget,
                "holdout_claim": holdout_text,
                "r256_joules_per_step": fam_r256_j,
                "ring16_j_to_pass": fam_ring16_j,
                "grid100_j_to_pass": fam_grid100_j,
                "tiers": tiers_dict,
                "total_view_requirements": total_dict,
                "full_suite": {
                    "pass": int(fs_counts["passed"]),
                    "fail": int(fs_counts["failed"]),
                    "not_measured": int(fs_counts["not_measured"]),
                    "other": int(fs_other_total),
                    "other_breakdown": {k: int(v) for k, v in fs_counts["other"].items()},
                    "total": int(fs_counts["total"]),
                    "held": {
                        "pass": int(fs_held["passed"]),
                        "fail": int(fs_held["failed"]),
                        "not_measured": int(fs_held["not_measured"]),
                        "other": int(fs_held_other),
                        "total": int(fs_held["total"]),
                    },
                },
                "stretch": {
                    "tiers": stretch_tier_buckets,
                    "total": stretch_total_bucket,
                },
            }
        )

        leaderboard_rows.append(
            {
                "family_id": fam_id,
                "family_label": fam_label,
                "selected_recipe_label": selected_recipe_label,
                "candidate_id": candidate_id,
                "candidate_revision": cand_rev,
                "recipe_sha256": recipe_sha,
                "source_digest": source_digest,
                "source_origin_commit": source_origin_commit,
                "backend": backend,
                "cohort_anchor": anchor,
                "family_page": family_page,
                "evidence_scope": str(row.get("evidence_scope") or ""),
                "stale": bool(is_stale),
                "qualified_tier": int(row.get("qualified_tier") or 0),
                "executed_policy_tier": int(actual_qual["qualified_tier"]) if actual_qual and actual_qual.get("qualified_tier") is not None else None,
                "executed_policy_revision": int(actual_qual["view_revision"]) if actual_qual and actual_qual.get("view_revision") is not None else None,
                "executed_policy_status": str(actual_qual["status"]) if actual_qual and actual_qual.get("status") is not None else None,
                "tier1_pass": tiers_dict["1"]["pass"],
                "tier1_total": tiers_dict["1"]["total"],
                "tier1_not_run": tiers_dict["1"]["not_run"],
                "tier1_blocked": tiers_dict["1"]["blocked"],
                "tier1_diagnostic": tiers_dict["1"]["diagnostic"],
                "tier2_pass": tiers_dict["2"]["pass"],
                "tier2_total": tiers_dict["2"]["total"],
                "tier2_not_run": tiers_dict["2"]["not_run"],
                "tier2_blocked": tiers_dict["2"]["blocked"],
                "tier2_diagnostic": tiers_dict["2"]["diagnostic"],
                "tier3_pass": tiers_dict["3"]["pass"],
                "tier3_total": tiers_dict["3"]["total"],
                "tier3_not_run": tiers_dict["3"]["not_run"],
                "tier3_blocked": tiers_dict["3"]["blocked"],
                "tier3_diagnostic": tiers_dict["3"]["diagnostic"],
                "tier4_pass": tiers_dict["4"]["pass"],
                "tier4_total": tiers_dict["4"]["total"],
                "tier4_not_run": tiers_dict["4"]["not_run"],
                "tier4_blocked": tiers_dict["4"]["blocked"],
                "tier4_diagnostic": tiers_dict["4"]["diagnostic"],
                "total_pass": total_dict["pass"],
                "total_required": total_dict["total"],
                "total_fail": total_dict["fail"],
                "total_not_run": total_dict["not_run"],
                "total_blocked": total_dict["blocked"],
                "total_diagnostic": total_dict["diagnostic"],
                "total_incompatible": total_dict["incompatible"],
                "full_suite_pass": int(fs_counts["passed"]),
                "full_suite_fail": int(fs_counts["failed"]),
                "full_suite_not_measured": int(fs_counts["not_measured"]),
                "full_suite_other": int(fs_other_total),
                "full_suite_total": int(fs_counts["total"]),
                "full_suite_held_pass": int(fs_held["passed"]),
                "full_suite_held_fail": int(fs_held["failed"]),
                "full_suite_held_not_measured": int(fs_held["not_measured"]),
                "full_suite_held_other": int(fs_held_other),
                "full_suite_held_total": int(fs_held["total"]),
                "stretch_tier1_pass": stretch_tier_buckets["1"]["pass"],
                "stretch_tier1_total": stretch_tier_buckets["1"]["total"],
                "stretch_tier2_pass": stretch_tier_buckets["2"]["pass"],
                "stretch_tier2_total": stretch_tier_buckets["2"]["total"],
                "stretch_tier3_pass": stretch_tier_buckets["3"]["pass"],
                "stretch_tier3_total": stretch_tier_buckets["3"]["total"],
                "stretch_tier4_pass": stretch_tier_buckets["4"]["pass"],
                "stretch_tier4_total": stretch_tier_buckets["4"]["total"],
                "stretch_total_pass": stretch_total_bucket["pass"],
                "stretch_total": stretch_total_bucket["total"],
                "r256_joules_per_step": fam_r256_j,
                "ring16_j_to_pass": fam_ring16_j,
                "grid100_j_to_pass": fam_grid100_j,
                "tuning_budget": tuning_budget,
                "holdout_claim": holdout_text,
            }
        )

        all_task_ids = sorted(set(task_def_by_id) | set(cohort_tasks))
        for task_id in all_task_ids:
            cell = cohort_tasks.get(task_id, {})
            tdef = task_def_by_id.get(task_id, {"tier": 0, "role": "diagnostic", "views": []})
            raw_status = str(cell.get("status") or "UNKNOWN")
            coverage_state = cell.get("coverage_status") or cell.get("attempt_coverage_status")
            if raw_status == "UNKNOWN":
                run_attempt = cell.get("run_attempt") or {}
                if cell.get("evidence_correction"):
                    display_status = "diagnostic"
                elif run_attempt.get("status"):
                    display_status = str(run_attempt["status"])
                elif coverage_state:
                    display_status = str(coverage_state)
                else:
                    display_status = "not_run"
            else:
                display_status = raw_status
            attempt_id = cell.get("attempt_id") or (cell.get("run_attempt") or {}).get("attempt_id")
            gif_relpath = None
            for artifact in cell.get("training_media", []) or []:
                gif_relpath = _gif_to_media_relpath(artifact.get("gif"))
                if gif_relpath:
                    break
            if gif_relpath is None and attempt_id:
                gif_relpath = media_by_attempt_task.get((attempt_id, task_id))
            if gif_relpath is None:
                gif_relpath = media_by_family_task.get((fam_id, task_id))
            cost_dict = (cell.get("metrics") or {}).get("cost") if isinstance(cell.get("metrics"), dict) else None
            wall_seconds = None
            if isinstance(cost_dict, dict) and isinstance(cost_dict.get("wall_seconds"), (int, float)):
                wall_seconds = float(cost_dict["wall_seconds"])
            cell_energy = None
            if isinstance(cost_dict, dict):
                extracted = _extract_receipt_energy(cost_dict)
                if (
                    extracted["joules_total"] is not None
                    or extracted["joules_per_step"] is not None
                    or extracted["avg_power_watts"] is not None
                ):
                    cell_energy = extracted
            if cell_energy is None and attempt_id:
                rec_energy = receipt_energy_by_attempt_task.get((str(attempt_id), task_id))
                if rec_energy is not None:
                    cell_energy = dict(rec_energy)
            family_tasks_json.append(
                {
                    "family_id": fam_id,
                    "family_label": fam_label,
                    "candidate_id": candidate_id,
                    "task_id": task_id,
                    "tier": int(tdef["tier"]),
                    "role": tdef["role"],
                    "views": list(tdef["views"]),
                    "status": display_status,
                    "recorded_status": raw_status,
                    "reason": cell.get("reason") or cell.get("coverage_reason") or "",
                    "attempt_id": attempt_id,
                    "recipe_hash": recipe_sha,
                    "source_hash": source_digest,
                    "wall_seconds": wall_seconds,
                    "energy": cell_energy,
                    "metrics_summary": _compact_metrics_summary(cell),
                    "gif_relpath": gif_relpath,
                }
            )

    # Attempts table from reports/forge/attempt-coverage.json
    attempt_rows: list[dict] = []
    attempt_summary = progress.get("run_attempt_coverage", {})
    ac_path = root / "reports/forge/attempt-coverage.json"
    if ac_path.is_file():
        ac_doc = read_json(ac_path)
        if isinstance(ac_doc.get("summary"), dict):
            attempt_summary = ac_doc["summary"]
        for rec in ac_doc.get("records", []):
            if not isinstance(rec, dict):
                continue
            timeout_val = rec.get("timeout_seconds")
            attempt_rows.append(
                {
                    "key": str(rec.get("key") or ""),
                    "family": str(rec.get("family") or ""),
                    "candidate_id": str(rec.get("candidate_id") or ""),
                    "candidate_revision": str(rec.get("candidate_revision") or ""),
                    "task": str(rec.get("task") or ""),
                    "status": str(rec.get("status") or ""),
                    "attempted": bool(rec.get("attempted", False)),
                    "diagnostic": bool(rec.get("diagnostic", False)),
                    "origin": str(rec.get("origin") or ""),
                    "source_digest": str(rec.get("source_digest") or ""),
                    "execution_sha256": str(rec.get("execution_sha256") or ""),
                    "evaluation_sha256": str(rec.get("evaluation_sha256") or ""),
                    "evidence": str(rec.get("evidence") or ""),
                    "timeout_seconds": int(timeout_val) if isinstance(timeout_val, int) and not isinstance(timeout_val, bool) else None,
                }
            )

    git_commit = _git_head_commit(root)
    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    declared_view_revisions = {view["id"]: int(view["revision"]) for view in summed}

    leaderboard_payload = {
        "schema_version": 1,
        "schema": "particlegan_forge_leaderboard_v1",
        "generated_at": timestamp,
        "git_commit": git_commit,
        "dataset_repo": repo,
        "media_prefix": DEFAULT_MEDIA_PREFIX,
        "view": publication.get("view", "discriminator_stability"),
        "view_revision": int(publication.get("view_revision") or 0),
        "declared_view_revisions": declared_view_revisions,
        "full_suite_list": (
            {
                "id": fixed["id"],
                "dated": fixed["dated"],
                "sha256": fixed["sha256"],
                "count": len(full_suite_names),
            }
            if fixed
            else {"id": "current_views", "dated": None, "sha256": None, "count": len(full_suite_names)}
        ),
        "summary": {
            "selected_families": len(families_json),
            "counted_view_requirements": int(counted_reqs),
            "distinct_required_tests": int(distinct_required),
            "full_suite_tests": len(full_suite_names),
            "stretch_assignments": len(stretch_assignments),
            "stretch_distinct_tests": int(stretch_distinct),
            "held_assignments": len(held_assignments),
            "held_distinct_tests": int(held_distinct),
            "removed_assignments": len(removed_assignments),
            "removed_distinct_tests": int(removed_distinct),
            "receipt_rows": len(receipt_rows),
            "receipts_with_energy": len(receipt_energy_entries),
            "energy_benchmark_rows": len(energy_rows),
            "attempt_coverage": attempt_summary,
        },
        "caveats": list(SCIENTIFIC_CAVEATS),
        "families": families_json,
        "tasks": family_tasks_json,
        "task_definitions": task_defs,
        "energy_benchmarks": energy_rows,
    }

    return {
        "leaderboard_rows": leaderboard_rows,
        "receipt_rows": receipt_rows,
        "attempt_rows": attempt_rows,
        "energy_rows": energy_rows,
        "leaderboard_json": leaderboard_payload,
        "git_commit": git_commit,
        "generated_at": timestamp,
    }


def _write_parquet_tables(staging_dir: Path, exports: dict) -> dict[str, tuple[Path, int]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to export Parquet tables; install with `pip install 'particlegan[hf]'`"
        ) from error

    leaderboard_schema = pa.schema(
        [
            ("family_id", pa.string()),
            ("family_label", pa.string()),
            ("selected_recipe_label", pa.string()),
            ("candidate_id", pa.string()),
            ("candidate_revision", pa.string()),
            ("recipe_sha256", pa.string()),
            ("source_digest", pa.string()),
            ("source_origin_commit", pa.string()),
            ("backend", pa.string()),
            ("cohort_anchor", pa.string()),
            ("family_page", pa.string()),
            ("evidence_scope", pa.string()),
            ("stale", pa.bool_()),
            ("qualified_tier", pa.int64()),
            ("executed_policy_tier", pa.int64()),
            ("executed_policy_revision", pa.int64()),
            ("executed_policy_status", pa.string()),
            ("tier1_pass", pa.int64()),
            ("tier1_total", pa.int64()),
            ("tier1_not_run", pa.int64()),
            ("tier1_blocked", pa.int64()),
            ("tier1_diagnostic", pa.int64()),
            ("tier2_pass", pa.int64()),
            ("tier2_total", pa.int64()),
            ("tier2_not_run", pa.int64()),
            ("tier2_blocked", pa.int64()),
            ("tier2_diagnostic", pa.int64()),
            ("tier3_pass", pa.int64()),
            ("tier3_total", pa.int64()),
            ("tier3_not_run", pa.int64()),
            ("tier3_blocked", pa.int64()),
            ("tier3_diagnostic", pa.int64()),
            ("tier4_pass", pa.int64()),
            ("tier4_total", pa.int64()),
            ("tier4_not_run", pa.int64()),
            ("tier4_blocked", pa.int64()),
            ("tier4_diagnostic", pa.int64()),
            ("total_pass", pa.int64()),
            ("total_required", pa.int64()),
            ("total_fail", pa.int64()),
            ("total_not_run", pa.int64()),
            ("total_blocked", pa.int64()),
            ("total_diagnostic", pa.int64()),
            ("total_incompatible", pa.int64()),
            ("full_suite_pass", pa.int64()),
            ("full_suite_fail", pa.int64()),
            ("full_suite_not_measured", pa.int64()),
            ("full_suite_other", pa.int64()),
            ("full_suite_total", pa.int64()),
            ("full_suite_held_pass", pa.int64()),
            ("full_suite_held_fail", pa.int64()),
            ("full_suite_held_not_measured", pa.int64()),
            ("full_suite_held_other", pa.int64()),
            ("full_suite_held_total", pa.int64()),
            ("stretch_tier1_pass", pa.int64()),
            ("stretch_tier1_total", pa.int64()),
            ("stretch_tier2_pass", pa.int64()),
            ("stretch_tier2_total", pa.int64()),
            ("stretch_tier3_pass", pa.int64()),
            ("stretch_tier3_total", pa.int64()),
            ("stretch_tier4_pass", pa.int64()),
            ("stretch_tier4_total", pa.int64()),
            ("stretch_total_pass", pa.int64()),
            ("stretch_total", pa.int64()),
            ("r256_joules_per_step", pa.float64()),
            ("ring16_j_to_pass", pa.float64()),
            ("grid100_j_to_pass", pa.float64()),
            ("tuning_budget", pa.string()),
            ("holdout_claim", pa.string()),
        ]
    )

    receipts_schema = pa.schema(
        [
            ("attempt_id", pa.string()),
            ("request_id", pa.string()),
            ("campaign_id", pa.string()),
            ("candidate_id", pa.string()),
            ("candidate_revision", pa.string()),
            ("evidence_scope", pa.string()),
            ("attempt_status", pa.string()),
            ("task_id", pa.string()),
            ("status", pa.string()),
            ("gate_status", pa.string()),
            ("raw_status", pa.string()),
            ("reason", pa.string()),
            ("compatibility_key", pa.string()),
            ("execution_path", pa.string()),
            ("device", pa.string()),
            ("wall_seconds", pa.float64()),
            ("completed_steps", pa.int64()),
            ("joules_total", pa.float64()),
            ("joules_per_step", pa.float64()),
            ("avg_power_watts", pa.float64()),
            ("first_pass_step", pa.int64()),
            ("sustained_pass_step", pa.int64()),
            ("joules_to_first_pass", pa.float64()),
            ("joules_to_sustained_pass", pa.float64()),
            ("energy_backend", pa.string()),
            ("canonical_result_hash", pa.string()),
            ("source_commit", pa.string()),
            ("source_digest", pa.string()),
            ("sampling_law", pa.string()),
            ("eval_output_noise", pa.string()),
            ("metrics_json", pa.string()),
            ("media_gif_relpath", pa.string()),
            ("receipt_file", pa.string()),
        ]
    )

    attempts_schema = pa.schema(
        [
            ("key", pa.string()),
            ("family", pa.string()),
            ("candidate_id", pa.string()),
            ("candidate_revision", pa.string()),
            ("task", pa.string()),
            ("status", pa.string()),
            ("attempted", pa.bool_()),
            ("diagnostic", pa.bool_()),
            ("origin", pa.string()),
            ("source_digest", pa.string()),
            ("execution_sha256", pa.string()),
            ("evaluation_sha256", pa.string()),
            ("evidence", pa.string()),
            ("timeout_seconds", pa.int64()),
        ]
    )

    energy_schema = pa.schema(
        [
            ("candidate_id", pa.string()),
            ("family_id", pa.string()),
            ("optimizer_family", pa.string()),
            ("hardware_gpu", pa.string()),
            ("contended", pa.bool_()),
            ("source_commit", pa.string()),
            ("source_file", pa.string()),
            ("r256_ms_per_step", pa.float64()),
            ("r256_watts", pa.float64()),
            ("r256_joules_per_step", pa.float64()),
            ("state_bytes_per_param", pa.float64()),
            ("ring16_first_pass_step", pa.int64()),
            ("ring16_j_to_pass", pa.float64()),
            ("grid100_first_pass_step", pa.int64()),
            ("grid100_sustained_pass_step", pa.int64()),
            ("grid100_j_to_pass", pa.float64()),
            ("grid100_j_to_sustained_pass", pa.float64()),
            ("staggered100_first_pass_step", pa.int64()),
            ("staggered100_j_to_pass", pa.float64()),
            ("occam_pass", pa.int64()),
            ("occam_executed", pa.int64()),
            ("evidence_scope", pa.string()),
        ]
    )

    data_dir = staging_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, tuple[Path, int]] = {}
    for hf_path, rows, schema in (
        ("data/leaderboard.parquet", exports["leaderboard_rows"], leaderboard_schema),
        ("data/receipts.parquet", exports["receipt_rows"], receipts_schema),
        ("data/attempts.parquet", exports["attempt_rows"], attempts_schema),
        ("data/energy.parquet", exports.get("energy_rows", []), energy_schema),
    ):
        cols = {field.name: [row.get(field.name) for row in rows] for field in schema}
        table = pa.Table.from_pydict(cols, schema=schema)
        out_path = staging_dir / hf_path
        pq.write_table(table, out_path, compression="zstd")
        written[hf_path] = (out_path, len(rows))
    return written


def render_dataset_card(exports: dict, *, repo: str = DEFAULT_HF_REPO) -> str:
    payload = exports["leaderboard_json"]
    summary = payload["summary"]
    git_commit = payload.get("git_commit") or "unknown"
    short_commit = git_commit[:12]
    lines = [
        "---",
        "license: mit",
        "pretty_name: ParticleGAN Forge Evidence & Benchmarks",
        "tags:",
        "  - particlegan",
        "  - gan",
        "  - benchmarks",
        "  - forge",
        "  - pytorch",
        "configs:",
        "  - config_name: leaderboard",
        "    default: true",
        "    data_files:",
        "      - split: train",
        "        path: data/leaderboard.parquet",
        "  - config_name: receipts",
        "    data_files:",
        "      - split: train",
        "        path: data/receipts.parquet",
        "  - config_name: attempts",
        "    data_files:",
        "      - split: train",
        "        path: data/attempts.parquet",
        "  - config_name: energy",
        "    data_files:",
        "      - split: train",
        "        path: data/energy.parquet",
        "---",
        "",
        "# ParticleGAN Forge Evidence & Benchmarks",
        "",
        "Public benchmark tables, hardware GPU energy & `J-to-PASS` telemetry, training media, and content-addressed execution archives for "
        "[HyperGAN/ParticleGAN](https://github.com/HyperGAN/ParticleGAN).",
        "",
        f"- **Source Git commit:** `{short_commit}` (`{git_commit}`)",
        f"- **Generated (UTC):** `{payload['generated_at']}`",
        f"- **Selected families:** {summary['selected_families']}",
        f"- **Counted view requirements:** {summary['counted_view_requirements']} "
        f"({summary['distinct_required_tests']} distinct required tests)",
        f"- **Full-suite tests:** {summary['full_suite_tests']} "
        f"(`{payload['full_suite_list']['id']}`)",
        f"- **Stretch view assignments:** {summary['stretch_assignments']} "
        f"({summary['stretch_distinct_tests']} distinct stretch tests)",
        f"- **Published technique receipts:** {summary['receipt_rows']}",
        f"- **Hardware NVML energy & `J-to-PASS` rows:** {summary.get('energy_benchmark_rows', 0)}",
        "",
        "## Dataset Tables",
        "",
        "| Config | File | Description |",
        "| --- | --- | --- |",
        "| `leaderboard` | `data/leaderboard.parquet` | Selected recipes per family with Tier 1–4 counts, full-suite 69-test counts, stretch counts, matched nullable energy summaries (`r256_joules_per_step`, `ring16_j_to_pass`, `grid100_j_to_pass`), and tuning-budget disclosures. |",
        "| `receipts` | `data/receipts.parquet` | Flattened per-task Forge technique receipts with status, gate metrics, runtime cost, nullable hardware NVML energy telemetry (`joules_total`, `joules_per_step`, `avg_power_watts`, `joules_to_first_pass`, `joules_to_sustained_pass`), and source provenance. |",
        "| `attempts` | `data/attempts.parquet` | Attempt coverage records across distinct recipe/test pairs. |",
        "| `energy` | `data/energy.parquet` | Hardware NVML energy & `J-to-PASS` (Joules-to-PASS) benchmarks across matched GPU cohorts (committed study benchmarks and energy-instrumented Forge receipts), with GPU model, contention flag, optimizer state footprint, and source provenance. |",
        "| *(Space payload)* | `data/leaderboard.json` | Structured JSON (including `energy_benchmarks`) consumed by the [HyperGAN/particlegan-leaderboard](https://huggingface.co/spaces/HyperGAN/particlegan-leaderboard) Static Space. |",
        "",
        "## Hardware Energy & `J-to-PASS` Telemetry",
        "",
        "- **`J-to-PASS` (Joules-to-PASS)** measures cumulative hardware GPU energy (via NVML `nvmlDeviceGetTotalEnergyConsumption` millijoule counters, with power-polling fallback) required for a candidate to reach its first passing evaluation (`joules_to_first_pass`) and sustained passing criterion (`joules_to_sustained_pass`).",
        "- Historical receipts collected before NVML telemetry retain `null` across all energy columns; missing hardware measurements are never fabricated or back-filled across incompatible source commits or GPU cohorts.",
        "",
        "## Scientific Scope & Caveats",
        "",
    ]
    for caveat in SCIENTIFIC_CAVEATS:
        lines.append(f"- {caveat}")
    lines += [
        "",
        "Derived Parquet/JSON tables never substitute for original byte-exact evidence. "
        "Restore original archived envelopes with explicit opt-in verification:",
        "",
        "```bash",
        "python -m experiments.forge artifacts hydrate <archive-card.json> \\",
        f"  --allow-hf --hf-repo {repo} --hf-rev <40-char-hf-revision> \\",
        "  --destination runs/forge/hydrated/<target>",
        "```",
        "",
    ]
    return "\n".join(lines)


def _extract_commit_sha(commit_info, api, repo: str) -> str:
    oid = getattr(commit_info, "oid", None)
    if isinstance(oid, str) and HF_SHA_RE.fullmatch(oid):
        return oid
    commit_url = getattr(commit_info, "commit_url", None)
    if isinstance(commit_url, str):
        tail = commit_url.rstrip("/").rsplit("/", 1)[-1]
        if HF_SHA_RE.fullmatch(tail):
            return tail
    info = api.repo_info(repo, repo_type="dataset")
    if isinstance(info.sha, str) and HF_SHA_RE.fullmatch(info.sha):
        return info.sha
    raise RuntimeError(f"unable to determine immutable 40-char HF revision SHA for {repo}")


def publish_to_hf(
    root: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    inventory: bool = False,
    archive_cards: tuple[Path, ...] | list[Path] = (),
    mirrors: tuple[Path, ...] | list[Path] = (),
    locations: Path | None = None,
    dry_run: bool = False,
    output_dir: Path | None = None,
    receipt_path: Path | None = None,
    commit_message: str | None = None,
    generated_at: str | None = None,
) -> dict:
    """Export requested Forge artifacts and optionally publish atomically to Hugging Face."""
    root = Path(root).resolve()
    if not HF_REPO_PATTERN.fullmatch(repo) or ".." in repo:
        raise ValueError(f"invalid HF dataset repo id: {repo!r}")
    if not inventory and not archive_cards:
        raise ValueError("specify at least one of --inventory or --archive-card")

    temp_holder = None
    if output_dir is not None:
        staging_dir = Path(output_dir).resolve()
        staging_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_holder = tempfile.TemporaryDirectory(prefix="forge-hf-publish-")
        staging_dir = Path(temp_holder.name)

    try:
        staged_files: dict[str, dict] = {}
        upload_map: dict[str, Path] = {}
        published_cards: list[dict] = []
        git_commit = _git_head_commit(root)
        timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if inventory:
            exports = build_inventory_exports(root, repo=repo, generated_at=timestamp)
            parquet_written = _write_parquet_tables(staging_dir, exports)
            for hf_path, (local_path, row_count) in parquet_written.items():
                digest, size = _sha256_file(local_path)
                staged_files[hf_path] = {"sha256": digest, "bytes": size, "rows": row_count}
                upload_map[hf_path] = local_path

            json_path = staging_dir / "data/leaderboard.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(exports["leaderboard_json"], indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            digest, size = _sha256_file(json_path)
            staged_files["data/leaderboard.json"] = {
                "sha256": digest,
                "bytes": size,
                "families": len(exports["leaderboard_json"]["families"]),
                "tasks": len(exports["leaderboard_json"]["tasks"]),
            }
            upload_map["data/leaderboard.json"] = json_path

            readme_path = staging_dir / "README.md"
            readme_path.write_text(render_dataset_card(exports, repo=repo))
            digest, size = _sha256_file(readme_path)
            staged_files["README.md"] = {"sha256": digest, "bytes": size}
            upload_map["README.md"] = readme_path

        for card_arg in archive_cards:
            card_path = Path(card_arg)
            resolved_card = card_path if card_path.is_absolute() else (root / card_path)
            card = read_json(resolved_card)
            inspection = inspect_archive(root, card, mirrors=mirrors, locations=locations)
            archive_path = Path(inspection["archive_path"])
            sha256 = inspection["archive_sha256"]
            hf_path = f"archives/sha256/{sha256}.tar.gz"
            size = archive_path.stat().st_size
            staged_files[hf_path] = {"sha256": sha256, "bytes": size}
            upload_map[hf_path] = archive_path
            try:
                rel_card = resolved_card.resolve().relative_to(root).as_posix()
            except ValueError:
                rel_card = str(resolved_card)
            published_cards.append(
                {
                    "card_path": rel_card,
                    "archive_sha256": sha256,
                    "bytes": size,
                    "hf_path": hf_path,
                    "member_count": inspection["archive_member_count"],
                }
            )

        hf_revision = None
        if not dry_run:
            try:
                from huggingface_hub import CommitOperationAdd, HfApi
            except ImportError as error:
                raise RuntimeError(
                    "huggingface_hub is required to publish; install with `pip install 'particlegan[hf]'`"
                ) from error
            api = HfApi()
            operations = [
                CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=str(local_path))
                for hf_path, local_path in sorted(upload_map.items())
            ]
            msg = commit_message or (
                f"Publish Forge inventory ({git_commit[:10] if git_commit else 'working-tree'})"
                if inventory
                else f"Publish Forge archives ({len(published_cards)} card(s))"
            )
            commit_info = api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                operations=operations,
                commit_message=msg,
            )
            hf_revision = _extract_commit_sha(commit_info, api, repo)

        receipt = {
            "schema_version": 1,
            "schema": "forge_hf_publication_receipt_v1",
            "published_at": timestamp,
            "git_source_commit": git_commit,
            "repo": repo,
            "repo_type": "dataset",
            "hf_revision": hf_revision,
            "dry_run": bool(dry_run),
            "inventory": bool(inventory),
            "files": dict(sorted(staged_files.items())),
            "archive_cards": published_cards,
            "qualification_input": False,
            "qualification_reuse": False,
            "training_launched": False,
        }

        target_receipt = None
        if receipt_path is not None:
            target_receipt = receipt_path if Path(receipt_path).is_absolute() else (root / receipt_path)
        elif not dry_run and inventory:
            target_receipt = root / DEFAULT_RECEIPT_RELPATH

        if target_receipt is not None:
            atomic_json(target_receipt, receipt)
            try:
                receipt["receipt_path"] = target_receipt.resolve().relative_to(root).as_posix()
            except ValueError:
                receipt["receipt_path"] = str(target_receipt)

        if output_dir is not None:
            receipt["staging_dir"] = str(staging_dir)
        return receipt
    finally:
        if temp_holder is not None:
            temp_holder.cleanup()
