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
from typing import Any

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
from .leaderboard_media import index_media_lookup

DEFAULT_HF_REPO = "HyperGAN/particlegan-forge-evidence"
DEFAULT_MEDIA_PREFIX = "media/forge/"
DEFAULT_PAGES_BASE_URL = "https://hypergan.github.io/ParticleGAN"
DEFAULT_RECEIPT_RELPATH = Path("reports/forge/publications/inventory-latest.json")
HF_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_INLINE_SVG_BYTES = 20480

DEFAULT_LAWS_EPIGRAPH = (
    "Our goal is not to preserve any fixed set of laws. "
    "Our goal is to discover the laws of learning as evidence advances."
)
DEFAULT_LAWS_EPISTEMIC_NOTE = (
    "This law taxonomy is a living scientific framework derived from current "
    "mathematical and empirical audits. Individual laws may be refined, split, "
    "merged, or replaced as new counterexamples and evidence emerge."
)

_ACRONYM_MAP = {
    "ae": "AE",
    "vae": "VAE",
    "gan": "GAN",
    "bigan": "BiGAN",
    "cgan": "CGAN",
    "dcgan": "DCGAN",
    "ddgan": "DDGAN",
    "pacgan": "PacGAN",
    "cyclegan": "CycleGAN",
    "misgan": "MisGAN",
    "bytegan": "ByteGAN",
    "mlp": "MLP",
    "resbn": "ResBN",
    "rgb": "RGB",
    "knn": "k-NN",
    "w1": "W1",
    "mxfp8": "MXFP8",
    "nvfp4": "NVFP4",
    "mog": "MoG",
    "1d": "1D",
    "2d": "2D",
    "3d": "3D",
    "8d": "8D",
    "32d": "32D",
    "cpu": "CPU",
    "gpu": "GPU",
    "lr": "LR",
    "ema": "EMA",
    "bn": "BN",
    "fl": "FL",
    "cov": "Cov",
    "img": "Image",
    "interp": "Interpolation",
    "aniso10": "Anisotropic 10:1",
    "res16": "ResNet-16",
}

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
        fam = _opt_str(idea.get("trainer_family"))
        if fam:
            return fam
    if candidate_id.startswith("axiom"):
        return "axiom"
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


def _extract_receipt_energy(
    cost: dict,
    evaluator_summary: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    """Extract nullable hardware energy telemetry from a receipt's ``cost`` dict.

    Historical receipts without ``cost["energy"]`` remain strictly ``None`` across
    all energy fields and are never fabricated. Prefers measured per-checkpoint
    cumulative NVML Joules when recorded, falling back to linear extrapolation
    ``joules_per_step * pass_step`` only for legacy receipts lacking checkpoint Joules.
    """
    empty = {
        "joules_total": None,
        "joules_per_step": None,
        "avg_power_watts": None,
        "first_pass_step": None,
        "sustained_pass_step": None,
        "joules_to_first_pass": None,
        "joules_to_sustained_pass": None,
        "cumulative_joules_to_first_pass": None,
        "cumulative_joules_to_sustained_pass": None,
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
    if steps is None:
        steps = _opt_int(cost.get("completed_updates"))
    if steps is None and isinstance(metrics, dict):
        steps = _opt_int(metrics.get("completed_steps")) or _opt_int(metrics.get("completed_updates"))
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

    ckpts = energy.get("checkpoint_joules")
    cum_j_fp = _opt_float(energy.get("cumulative_joules_to_first_pass"))
    if cum_j_fp is None and isinstance(ckpts, dict) and fp_step is not None:
        cum_j_fp = _opt_float(ckpts.get(str(fp_step)) if str(fp_step) in ckpts else ckpts.get(fp_step))
    cum_j_sp = _opt_float(energy.get("cumulative_joules_to_sustained_pass"))
    if cum_j_sp is None and isinstance(ckpts, dict) and sp_step is not None:
        cum_j_sp = _opt_float(ckpts.get(str(sp_step)) if str(sp_step) in ckpts else ckpts.get(sp_step))

    j_fp = cum_j_fp if cum_j_fp is not None else _opt_float(energy.get("joules_to_first_pass"))
    if j_fp is None and joules_per_step is not None and fp_step is not None:
        j_fp = joules_per_step * float(fp_step)

    j_sp = cum_j_sp if cum_j_sp is not None else _opt_float(energy.get("joules_to_sustained_pass"))
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
        "cumulative_joules_to_first_pass": cum_j_fp,
        "cumulative_joules_to_sustained_pass": cum_j_sp,
        "energy_backend": _opt_str(energy.get("energy_backend")),
        "gpu_name": _opt_str(energy.get("gpu_name")),
        "contended": _opt_bool(energy.get("contended")),
    }


def _is_incomplete_timeout_result(tr: dict, attempt_status: str = "") -> bool:
    """Identify INCOMPLETE wall-time timeout rows excluded from Pareto accuracy denominators."""
    gate = str(tr.get("gate_status") or tr.get("status") or "").upper()
    raw = str(tr.get("raw_status") or "").lower()
    att = str(attempt_status or "").lower()
    if gate in {"PASS", "FAIL"}:
        return False
    return gate in {"INCOMPLETE", "TIMEOUT"} or raw == "timeout" or att == "timeout"


def _readout_occam_counts(root: Path, readout_name: str | None) -> tuple[int | None, int | None, int | None, str | None]:
    """Count Occam passes, scored completed executions, and excluded INCOMPLETE timeouts."""
    if not isinstance(readout_name, str) or not readout_name.strip():
        return None, None, None, None
    path = root / "reports/forge/records" / readout_name.strip()
    if not path.is_file():
        return None, None, None, None
    try:
        doc = read_json(path)
    except Exception:
        return None, None, None, None
    prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    src_commit = _opt_str(prov.get("source_origin_commit")) or _opt_str(doc.get("source_commit"))
    task_results = doc.get("task_results")
    if not isinstance(task_results, list) or not task_results:
        return None, None, None, src_commit
    scored_statuses = {"PASS", "FAIL", "INVALID", "ERROR"}
    passed = 0
    executed = 0
    timeouts = 0
    for tr in task_results:
        if not isinstance(tr, dict):
            continue
        if _is_incomplete_timeout_result(tr):
            timeouts += 1
            continue
        gate = str(tr.get("gate_status") or tr.get("status") or "").upper()
        if gate == "PASS":
            passed += 1
        if gate in scored_statuses:
            executed += 1
    if executed > 0 or timeouts > 0:
        return passed, executed, timeouts, src_commit
    return None, None, None, src_commit


def _receipt_ids_occam_counts(root: Path, receipt_ids: Any) -> tuple[int | None, int | None, int | None]:
    """Count Occam passes, scored completed executions, and excluded timeouts from receipt IDs."""
    if not isinstance(receipt_ids, list) or not receipt_ids:
        return None, None, None
    receipts_dir = root / "reports/forge/technique-receipts"
    if not receipts_dir.is_dir():
        return None, None, None
    scored_statuses = {"PASS", "FAIL", "INVALID", "ERROR"}
    passed = 0
    executed = 0
    timeouts = 0
    seen_any = False
    for raw_id in receipt_ids:
        if not isinstance(raw_id, str) or not raw_id.strip():
            continue
        rec_path = receipts_dir / f"{raw_id.strip()}.json"
        if not rec_path.is_file():
            continue
        try:
            doc = read_json(rec_path)
        except Exception:
            continue
        att_status = str(doc.get("attempt_status") or "")
        task_results = doc.get("task_results")
        if not isinstance(task_results, list) or not task_results:
            if att_status.lower() == "timeout":
                timeouts += 1
                seen_any = True
            continue
        for tr in task_results:
            if not isinstance(tr, dict):
                continue
            seen_any = True
            if _is_incomplete_timeout_result(tr, attempt_status=att_status):
                timeouts += 1
                continue
            gate = str(tr.get("gate_status") or tr.get("status") or "").upper()
            if gate == "PASS":
                passed += 1
            if gate in scored_statuses:
                executed += 1
    if not seen_any:
        return None, None, None
    return passed, executed, timeouts


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

                ro_pass, ro_exec, ro_timeouts, ro_commit = _readout_occam_counts(
                    root, grid100.get("official_forge_readout")
                )
                rec_pass, rec_exec, rec_timeouts = _receipt_ids_occam_counts(root, entry.get("occam_receipt_ids"))
                occam_pass = _opt_int(entry.get("occam_pass"))
                occam_executed = _opt_int(entry.get("occam_executed"))
                occam_timeouts = _opt_int(entry.get("occam_timeouts"))
                if rec_exec is not None:
                    # Exclude INCOMPLETE wall-time timeouts from Pareto accuracy denominator
                    # while preserving transparent timeout count metadata.
                    occam_pass = rec_pass
                    occam_executed = rec_exec
                    occam_timeouts = rec_timeouts
                else:
                    if occam_pass is None:
                        occam_pass = ro_pass
                    if occam_executed is None:
                        occam_executed = ro_exec
                    if occam_timeouts is None:
                        occam_timeouts = ro_timeouts if ro_timeouts is not None else (0 if occam_executed is not None else None)

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
                        "occam_timeouts": occam_timeouts,
                        "evidence_scope": scope,
                    }
                )
        elif isinstance(doc.get("benchmarks"), list):
            for entry in doc["benchmarks"]:
                if not isinstance(entry, dict) or not _opt_str(entry.get("candidate_id")):
                    continue
                candidate_id = str(entry["candidate_id"]).strip()
                occam_exec_b = _opt_int(entry.get("occam_executed"))
                occam_to_b = _opt_int(entry.get("occam_timeouts"))
                if occam_to_b is None and occam_exec_b is not None:
                    occam_to_b = 0
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
                        "occam_executed": occam_exec_b,
                        "occam_timeouts": occam_to_b,
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
                "occam_timeouts": None,
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
    metric_keys = (
        "r256_ms_per_step",
        "r256_joules_per_step",
        "ring16_j_to_pass",
        "grid100_j_to_pass",
        "grid100_j_to_sustained_pass",
        "staggered100_j_to_pass",
    )
    return [
        grouped[k]
        for k in sorted(grouped, key=lambda item: tuple("" if x is None else str(x) for x in item))
        if any(grouped[k].get(mk) is not None for mk in metric_keys)
    ]


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


def _collect_media_with_manifest(
    root: Path, progress: dict
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], dict[tuple[str, str], dict]]:
    """Merge inventory training_media with committed `reports/forge/leaderboard-media-index.json`."""
    by_family_task, by_attempt_task = _collect_media_index(progress)
    meta_by_family_task: dict[tuple[str, str], dict] = {}
    idx_fam, idx_att = index_media_lookup(root)
    for key, entry in idx_fam.items():
        rel = _gif_to_media_relpath(entry.get("gif_relpath"))
        if not rel:
            continue
        # Prefer committed manifest entries when selected_cohort or when progress had no GIF
        if key not in by_family_task or entry.get("cohort_match") == "selected_cohort":
            by_family_task[key] = rel
        meta_by_family_task[key] = {
            "cohort_match": entry.get("cohort_match"),
            "recorded_grade": entry.get("recorded_grade"),
            "grade_matches_selected": entry.get("grade_matches_selected"),
            "has_visual_samples": bool(entry.get("has_visual_samples")),
            "attempt_id": entry.get("attempt_id"),
            "gpu_name": entry.get("gpu_name"),
            "bytes": entry.get("bytes"),
            "observation_count": entry.get("observation_count"),
        }
    for akey, entry in idx_att.items():
        rel = _gif_to_media_relpath(entry.get("gif_relpath"))
        if rel:
            by_attempt_task.setdefault(akey, rel)
    return by_family_task, by_attempt_task, meta_by_family_task


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


def _clean_str_list(values) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _opt_str(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _humanize_task_id(task_id: str) -> str:
    parts = [p for p in re.split(r"[_\-]+", str(task_id or "").strip()) if p]
    if not parts:
        return str(task_id or "")
    rendered = []
    for part in parts:
        low = part.lower()
        if low in _ACRONYM_MAP:
            rendered.append(_ACRONYM_MAP[low])
        elif re.fullmatch(r"v\d+", low):
            rendered.append(low)
        else:
            rendered.append(part.capitalize())
    return " ".join(rendered)


def _load_laws_registry_payload(root: Path, *, pages_base_url: str) -> tuple[dict, list[dict]]:
    """Load the evolving laws registry from ``configs/forge/laws.json`` with safe fallback."""
    base = pages_base_url.rstrip("/")
    doc: dict | None = None
    laws_path = root / "configs/forge/laws.json"
    if laws_path.is_file():
        try:
            from . import laws_registry as _laws_registry  # type: ignore

            loaded = _laws_registry.load_laws(root)
            if isinstance(loaded, dict):
                doc = loaded
        except Exception:
            doc = None
        if doc is None:
            try:
                loaded = read_json(laws_path)
                if isinstance(loaded, dict):
                    doc = loaded
            except Exception:
                doc = None

    if not isinstance(doc, dict) or not isinstance(doc.get("laws"), list):
        return (
            {
                "schema_version": 1,
                "epistemic_status": "provisional_evolving",
                "epigraph": DEFAULT_LAWS_EPIGRAPH,
                "epistemic_note": DEFAULT_LAWS_EPISTEMIC_NOTE,
                "doc_path": "docs/four-laws.md",
                "pages_url": f"{base}/laws/",
                "source": "unavailable",
            },
            [],
        )

    doc_path = _opt_str(doc.get("doc_path")) or "docs/four-laws.md"
    laws_meta = {
        "schema_version": _opt_int(doc.get("schema_version")) or 1,
        "epistemic_status": _opt_str(doc.get("epistemic_status")) or "provisional_evolving",
        "epigraph": _opt_str(doc.get("epigraph")) or DEFAULT_LAWS_EPIGRAPH,
        "epistemic_note": _opt_str(doc.get("epistemic_note")) or DEFAULT_LAWS_EPISTEMIC_NOTE,
        "doc_path": doc_path,
        "pages_url": f"{base}/laws/",
        "source": "configs/forge/laws.json",
    }

    laws_out: list[dict] = []
    for raw_law in doc.get("laws", []):
        if not isinstance(raw_law, dict):
            continue
        law_id = _opt_str(raw_law.get("id"))
        if not law_id:
            continue
        sub_out: list[dict] = []
        for raw_sub in raw_law.get("sub_laws") or []:
            if not isinstance(raw_sub, dict):
                continue
            sub_id = _opt_str(raw_sub.get("id"))
            if not sub_id:
                continue
            sub_out.append(
                {
                    "id": sub_id,
                    "short_label": _opt_str(raw_sub.get("short_label")) or sub_id,
                    "title": _opt_str(raw_sub.get("title")) or sub_id,
                    "doc_anchor": _opt_str(raw_sub.get("doc_anchor")) or _opt_str(raw_law.get("doc_anchor")) or "",
                }
            )
        laws_out.append(
            {
                "id": law_id,
                "short_label": _opt_str(raw_law.get("short_label")) or law_id,
                "title": _opt_str(raw_law.get("title")) or law_id,
                "domain_level": _opt_str(raw_law.get("domain_level")) or _opt_str(raw_law.get("title")) or law_id,
                "question": _opt_str(raw_law.get("question")) or "",
                "doc_anchor": _opt_str(raw_law.get("doc_anchor")) or "",
                "status": _opt_str(raw_law.get("status")) or "active",
                "sub_laws": sub_out,
            }
        )
    laws_meta["active_law_count"] = sum(1 for law in laws_out if law.get("status") == "active")
    laws_meta["provisional_law_count"] = sum(1 for law in laws_out if law.get("status") == "provisional")
    return laws_meta, laws_out


def _load_family_doc_enrichment(
    root: Path,
    fam_id: str,
    base_tags: list[str] | None,
    *,
    pages_base_url: str,
) -> dict:
    """Load optional researcher documentation enrichment for one trainer family."""
    base = pages_base_url.rstrip("/")
    doc_path = root / "configs/forge/family-documentation" / f"{fam_id}.json"
    doc: dict = {}
    if doc_path.is_file():
        try:
            loaded = read_json(doc_path)
            if isinstance(loaded, dict):
                doc = loaded
        except Exception:
            doc = {}

    law_mechanisms: dict[str, dict] = {}
    raw_mechs = doc.get("law_mechanisms")
    if isinstance(raw_mechs, dict):
        for law_id, entry in raw_mechs.items():
            key = _opt_str(law_id)
            if not key or not isinstance(entry, dict):
                continue
            law_mechanisms[key] = {
                "status": _opt_str(entry.get("status")) or "unaddressed",
                "mechanism": _opt_str(entry.get("mechanism")) or "",
                "how_it_works": _opt_str(entry.get("how_it_works")) or "",
                "known_limits": _opt_str(entry.get("known_limits")) or "",
            }

    equations: list[dict] = []
    if isinstance(doc.get("equations"), list):
        for eq in doc["equations"]:
            if not isinstance(eq, dict):
                continue
            latex = _opt_str(eq.get("latex"))
            if not latex:
                continue
            equations.append(
                {
                    "label": _opt_str(eq.get("label")) or "Equation",
                    "latex": latex,
                    "explanation": _opt_str(eq.get("explanation")) or "",
                }
            )

    quickstart: dict | None = None
    raw_qs = doc.get("quickstart")
    if isinstance(raw_qs, dict):
        preset = _opt_str(raw_qs.get("preset"))
        raw_snippet = raw_qs.get("snippet")
        if isinstance(raw_snippet, list):
            snippet_lines = [str(line) for line in raw_snippet]
        elif isinstance(raw_snippet, str):
            snippet_lines = raw_snippet.splitlines()
        else:
            snippet_lines = []
        if preset or snippet_lines:
            quickstart = {
                "preset": preset or fam_id,
                "snippet": snippet_lines,
            }

    tags = _clean_str_list(list(base_tags or []) + list(doc.get("tags") or []))
    return {
        "overview": _opt_str(doc.get("overview")),
        "law_mechanisms": law_mechanisms,
        "equations": equations,
        "mermaid_diagram": _opt_str(doc.get("mermaid_diagram")),
        "quickstart": quickstart,
        "tags": tags,
        "pages_url": f"{base}/families/{fam_id}/",
    }


def _build_four_laws_adherence_l4_payload(
    root: Path,
    *,
    pages_base_url: str = DEFAULT_PAGES_BASE_URL,
) -> dict | None:
    """Build matched NVIDIA L4 diagnostic Four Laws Adherence summary from committed receipts."""
    try:
        from . import experiment_docs as _experiment_docs  # type: ignore

        catalog = _experiment_docs.build_experiment_catalog(root)
    except Exception:
        return None
    exps = catalog.get("experiments") if isinstance(catalog, dict) else None
    if not isinstance(exps, dict):
        return None

    fl_probes = [
        e
        for e in exps.values()
        if isinstance(e, dict)
        and isinstance(e.get("four_laws_adherence"), dict)
        and e["four_laws_adherence"].get("role") == "four_laws_probe"
    ]
    if not fl_probes:
        return None
    fl_probes.sort(key=lambda e: (str(e["four_laws_adherence"].get("law") or ""), str(e["task_id"])))
    probe_ids = [str(e["task_id"]) for e in fl_probes]
    law_of = {str(e["task_id"]): str(e["four_laws_adherence"].get("law") or "") for e in fl_probes}

    short_labels = {
        "fl_law1_aniso_unequal_modes": "L1 Aniso",
        "fl_law1_noisy_atomic_codebook": "L1 Noisy",
        "fl_law2_pure_adversarial_codim1": "L2 Codim1",
        "fl_law2_groupnorm_and_vit_sprites": "L2 GN+ViT",
        "fl_law3_highd_orphan_d32": "L3 R32",
        "fl_law41_trust_rare1pct_tanh": "L4.1 Rare1%",
        "fl_law42_swap_deep_critic_16l": "L4.2 Swap16L",
        "fl_law42_deep_resbn_and_conv1x1": "L4.2 ResBN",
        "fl_law44_highd_sparse_prior_and_shock": "L4.4 R512",
    }
    base = pages_base_url.rstrip("/")

    by_cand: dict[str, dict[str, dict]] = {}
    cand_family: dict[str, str] = {}
    for tid in [*probe_ids, "bytegan_hypercube8", "bytegan_bigan_dictionary"]:
        entry = exps.get(tid)
        if not isinstance(entry, dict):
            continue
        for dr in entry.get("four_laws_readouts") or []:
            if not isinstance(dr, dict) or dr.get("gpu_name") != "NVIDIA L4":
                continue
            cid = str(dr.get("candidate_id") or "")
            if not cid:
                continue
            by_cand.setdefault(cid, {})[tid] = dr
            if dr.get("family_id"):
                cand_family[cid] = str(dr["family_id"])

    l4_probe_cands = [cid for cid, m in by_cand.items() if any(pid in m for pid in probe_ids)]
    if not l4_probe_cands:
        return None

    law_order = tuple(dict.fromkeys([*("law1", "law2", "law3", "law4"), *(law_of[pid] for pid in probe_ids if law_of[pid])]))
    probes_by_law = {lid: [pid for pid in probe_ids if law_of[pid] == lid] for lid in law_order}

    def _sort_key(cid: str) -> tuple:
        m = by_cand[cid]
        passes = sum(1 for pid in probe_ids if (m.get(pid) or {}).get("status") == "PASS")
        bg_passes = sum(
            1 for tid in ("bytegan_hypercube8", "bytegan_bigan_dictionary")
            if (m.get(tid) or {}).get("status") == "PASS"
        )
        margins = [
            m[pid]["min_margin"]
            for pid in probe_ids
            if pid in m and isinstance(m[pid].get("min_margin"), (int, float))
        ]
        mean_m = sum(margins) / len(margins) if margins else -1e9
        return (-passes, -bg_passes, -mean_m, cid)

    rows: list[dict] = []
    for cid in sorted(l4_probe_cands, key=_sort_key):
        m = by_cand[cid]
        laws_summary: dict[str, dict] = {}
        total_pass = 0
        total_measured = 0
        for lid in law_order:
            lpids = probes_by_law[lid]
            lp = sum(1 for pid in lpids if (m.get(pid) or {}).get("status") == "PASS")
            lm = sum(1 for pid in lpids if pid in m)
            total_pass += lp
            total_measured += lm
            laws_summary[lid] = {"pass": lp, "measured": lm, "total": len(lpids)}

        total_j = 0.0
        has_j = False
        probe_cells: dict[str, dict] = {}
        for pid in probe_ids:
            dr = m.get(pid)
            if not dr:
                continue
            jt = _opt_float(dr.get("joules_total"))
            if jt is not None:
                total_j += jt
                has_j = True
            probe_cells[pid] = {
                "status": str(dr.get("status") or "UNKNOWN"),
                "min_margin": _opt_float(dr.get("min_margin")),
                "joules_to_pass": _opt_float(dr.get("joules_to_pass")),
                "joules_total": jt,
                "receipt": _opt_str(dr.get("readout")),
            }

        bg_measured = [tid for tid in ("bytegan_hypercube8", "bytegan_bigan_dictionary") if tid in m]
        bg_pass = sum(1 for tid in bg_measured if m[tid].get("status") == "PASS")
        bg_j_to_pass = {
            tid: _opt_float(m[tid].get("joules_to_pass"))
            for tid in bg_measured
            if m[tid].get("status") == "PASS" and _opt_float(m[tid].get("joules_to_pass")) is not None
        }
        rows.append(
            {
                "candidate_id": cid,
                "family_id": cand_family.get(cid) or None,
                "blocked": False,
                "blocked_reason": None,
                "laws": laws_summary,
                "probes_pass": total_pass,
                "probes_measured": total_measured,
                "probes_total": len(probe_ids),
                "bytegan_pass": bg_pass if bg_measured else None,
                "bytegan_measured": len(bg_measured),
                "bytegan_j_to_pass": bg_j_to_pass,
                "nine_probe_joules_total": round(total_j, 2) if has_j else None,
                "probes": probe_cells,
            }
        )

    for blocked_cid, blocked_fam, reason in (
        ("atlas-particle-normal-v4", "atlas", "requires MoGParticlePrior"),
        ("e22", "e22", "clean/live policy contract"),
    ):
        if blocked_cid not in by_cand or not any(pid in by_cand[blocked_cid] for pid in probe_ids):
            bg_m = [
                tid
                for tid in ("bytegan_hypercube8", "bytegan_bigan_dictionary")
                if tid in by_cand.get(blocked_cid, {})
            ]
            bg_p = sum(1 for tid in bg_m if by_cand[blocked_cid][tid].get("status") == "PASS") if bg_m else None
            rows.append(
                {
                    "candidate_id": blocked_cid,
                    "family_id": blocked_fam,
                    "blocked": True,
                    "blocked_reason": reason,
                    "laws": {lid: {"pass": 0, "measured": 0, "total": len(probes_by_law[lid])} for lid in law_order},
                    "probes_pass": 0,
                    "probes_measured": 0,
                    "probes_total": len(probe_ids),
                    "bytegan_pass": bg_p,
                    "bytegan_measured": len(bg_m),
                    "bytegan_j_to_pass": {},
                    "nine_probe_joules_total": None,
                    "probes": {},
                }
            )

    return {
        "view_id": "four-laws-adherence-v1",
        "hardware_gpu": "NVIDIA L4",
        "evidence_scope": "research_diagnostic",
        "seed": 0,
        "probes": [
            {
                "task_id": pid,
                "short_label": short_labels.get(pid, pid),
                "law": law_of[pid],
                "pages_url": f"{base}/experiments/{pid}/",
            }
            for pid in probe_ids
        ],
        "rows": rows,
    }


def _load_experiment_docs_catalog(root: Path) -> dict[str, dict]:
    """Read optional experiment catalog from ``experiments.forge.experiment_docs`` if present."""
    try:
        from . import experiment_docs as _experiment_docs  # type: ignore

        builder = getattr(_experiment_docs, "build_experiment_catalog", None)
        if not callable(builder):
            return {}
        catalog = builder(root)
    except Exception:
        return {}

    by_id: dict[str, dict] = {}
    if isinstance(catalog, dict):
        if "experiments" in catalog:
            tasks_obj = catalog.get("experiments")
        elif "tasks" in catalog:
            tasks_obj = catalog.get("tasks")
        else:
            tasks_obj = catalog
        if isinstance(tasks_obj, list):
            for entry in tasks_obj:
                if isinstance(entry, dict):
                    tid = _opt_str(entry.get("task_id")) or _opt_str(entry.get("id"))
                    if tid:
                        doc = entry.get("documentation") if isinstance(entry.get("documentation"), dict) else {}
                        by_id[tid] = {**doc, **entry}
        elif isinstance(tasks_obj, dict):
            for tid, entry in tasks_obj.items():
                if isinstance(entry, dict):
                    key = _opt_str(entry.get("task_id")) or _opt_str(tid)
                    if key:
                        doc = entry.get("documentation") if isinstance(entry.get("documentation"), dict) else {}
                        by_id[key] = {**doc, **entry}
    return by_id


def _synthesize_prior_summary(execution: dict, host_def: dict) -> str | None:
    prior = execution.get("prior") if isinstance(execution.get("prior"), dict) else {}
    prior_contract = execution.get("prior_contract") if isinstance(execution.get("prior_contract"), dict) else {}
    parts: list[str] = []
    kind = _opt_str(prior.get("kind")) or _opt_str(prior_contract.get("kind"))
    if kind:
        sub = []
        sigma = _opt_float(prior.get("sigma"))
        if sigma is not None:
            sub.append(f"sigma={sigma:g}")
        std = _opt_bool(prior.get("standardize"))
        if std is not None:
            sub.append(f"standardize={str(std).lower()}")
        parts.append(f"{kind} ({', '.join(sub)})" if sub else kind)
    particles = _opt_int(host_def.get("particles")) or _opt_int(prior.get("num_particles"))
    if particles is not None:
        parts.append(f"{particles} particles")
    z_dim = _opt_int(host_def.get("z_dim")) or _opt_int(prior.get("z_dim"))
    if z_dim is not None:
        parts.append(f"z_dim={z_dim}")
    return " · ".join(parts) if parts else None


def _synthesize_model_summary(task_json: dict, execution: dict, host_def: dict) -> str | None:
    parts: list[str] = []
    adapter = _opt_str(task_json.get("adapter"))
    host = _opt_str(execution.get("host"))
    kind = _opt_str(host_def.get("kind"))
    if adapter:
        parts.append(adapter)
    if host and host != task_json.get("id"):
        parts.append(f"host={host}")
    if kind:
        parts.append(kind)
    arch_bits = []
    layers = _opt_int(host_def.get("layers"))
    hidden = _opt_int(host_def.get("hidden"))
    if layers is not None and hidden is not None:
        arch_bits.append(f"{layers}×{hidden} MLP")
    elif hidden is not None:
        arch_bits.append(f"hidden={hidden}")
    fourier = _opt_int(host_def.get("fourier"))
    if fourier is not None:
        arch_bits.append(f"fourier={fourier}")
    batch = _opt_int(host_def.get("batch"))
    if batch is not None:
        arch_bits.append(f"batch={batch}")
    perf = execution.get("performance") if isinstance(execution.get("performance"), dict) else {}
    if perf:
        res = _opt_int(perf.get("resolution"))
        width = _opt_int(perf.get("width"))
        pbatch = _opt_int(perf.get("batch_size"))
        if res is not None:
            arch_bits.append(f"r={res}")
        if width is not None:
            arch_bits.append(f"w={width}")
        if pbatch is not None:
            arch_bits.append(f"batch={pbatch}")
    if arch_bits:
        parts.append(", ".join(arch_bits))
    return " · ".join(parts) if parts else None


def _synthesize_protocol_notes(execution: dict, evaluation: dict) -> str | None:
    parts: list[str] = []
    protocol = _opt_str(execution.get("protocol"))
    if protocol:
        parts.append(f"Protocol: {protocol}")
    init_name = _opt_str(execution.get("initializer"))
    if init_name:
        parts.append(f"Initializer: {init_name}")
    sampling = _opt_str(evaluation.get("sampling_law"))
    if sampling:
        parts.append(f"Sampling: {sampling}")
    scoring = _opt_str(evaluation.get("scoring_weights"))
    if scoring:
        parts.append(f"Weights: {scoring}")
    eval_noise = _opt_str(evaluation.get("eval_output_noise"))
    if eval_noise:
        parts.append(f"Eval noise: {eval_noise}")
    stable = _opt_int(evaluation.get("minimum_stable_checks"))
    if stable is not None:
        parts.append(f"Sustained checks >= {stable}")
    return " · ".join(parts) if parts else None


def _extract_thresholds(evaluation: dict, cat_entry: dict) -> dict:
    raw = cat_entry.get("thresholds")
    if isinstance(raw, dict) and raw:
        return {
            str(k): v
            for k, v in raw.items()
            if isinstance(v, (int, float, bool, str)) or (isinstance(v, list) and len(v) <= 8)
        }
    out: dict = {}
    raw_thresh = evaluation.get("thresholds")
    if isinstance(raw_thresh, list):
        for entry in raw_thresh:
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                metric_name = _opt_str(entry[0])
                op = _opt_str(entry[1])
                val = entry[2]
                if metric_name and op and isinstance(val, (int, float, bool, str)):
                    out[metric_name] = f"{op} {val}"
    for bucket_name, prefix in (
        ("thresholds", ""),
        ("coverage_thresholds", "coverage."),
        ("accuracy_limits", "accuracy."),
    ):
        bucket = evaluation.get(bucket_name)
        if not isinstance(bucket, dict):
            continue
        for k, v in sorted(bucket.items()):
            if isinstance(v, (int, float, bool, str)):
                out[f"{prefix}{k}"] = v
            elif isinstance(v, list) and len(v) <= 6 and all(isinstance(x, (int, float, bool, str)) for x in v):
                out[f"{prefix}{k}"] = v
    return out


def _mermaid_safe(text: str, max_len: int = 44) -> str:
    cleaned = re.sub(r'["`<>]', "", str(text or "")).replace("\n", " ").strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned or "Component"


def _synthesize_task_mermaid(
    task_id: str,
    task_json: dict,
    prior_summary: str | None,
    model_summary: str | None,
    steps: int | None,
    evaluation: dict,
) -> str | None:
    if not task_json:
        return None
    prior_lbl = _mermaid_safe(prior_summary or "Task Prior P")
    model_lbl = _mermaid_safe(model_summary or _opt_str(task_json.get("adapter")) or "Generator & Critic")
    step_lbl = f"{steps} steps" if steps is not None else "Task budget"
    eval_lbl = _mermaid_safe(
        " · ".join(
            bit
            for bit in (
                _opt_str(evaluation.get("scoring_weights")),
                _opt_str(evaluation.get("sampling_law")),
                _opt_str(evaluation.get("kind")),
            )
            if bit
        )
        or "Numerical Gate"
    )
    target_lbl = _mermaid_safe(_humanize_task_id(task_id))
    return "\n".join(
        [
            "flowchart TD",
            f'  P["Prior: {prior_lbl}"] --> G["Model: {model_lbl}"]',
            f'  G -->|"Train ({step_lbl})"| D["Critic & Target: {target_lbl}"]',
            f'  D --> E["Gate: {eval_lbl}"]',
        ]
    )


def _load_task_svg(root: Path, task_id: str, cat_entry: dict) -> tuple[str | None, str | None]:
    rel_candidate = _opt_str(cat_entry.get("svg_relpath")) or f"reports/forge/experiments/assets/{task_id}.svg"
    svg_path = root / rel_candidate
    svg_relpath: str | None = None
    svg_inline: str | None = None
    if svg_path.is_file():
        svg_relpath = rel_candidate.replace("\\", "/")
        try:
            raw_bytes = svg_path.read_bytes()
            if len(raw_bytes) <= MAX_INLINE_SVG_BYTES:
                text = raw_bytes.decode("utf-8", errors="strict").strip()
                text = re.sub(r"^<\?xml[^>]*\?>\s*", "", text)
                if text.startswith("<svg"):
                    svg_inline = text
        except Exception:
            svg_inline = None
    elif isinstance(cat_entry.get("svg_inline"), str):
        text = re.sub(r"^<\?xml[^>]*\?>\s*", "", cat_entry["svg_inline"].strip())
        if text.startswith("<svg") and len(text.encode("utf-8")) <= MAX_INLINE_SVG_BYTES:
            svg_inline = text
    return svg_relpath, svg_inline


def _enrich_task_definitions(
    root: Path,
    task_defs: list[dict],
    *,
    pages_base_url: str,
    default_candidate: str = "unified-v11",
) -> list[dict]:
    """Attach optional documentation, contract, diagram, and portal metadata to task definitions.

    Reads ``experiments.forge.experiment_docs.build_experiment_catalog`` first when available,
    then falls back cleanly to ``configs/forge/task-documentation/<task_id>.json`` and inline
    ``configs/forge/tasks/<task_id>.json`` metadata.
    """
    base = pages_base_url.rstrip("/")
    catalog = _load_experiment_docs_catalog(root)
    tasks_dir = root / "configs/forge/tasks"
    task_docs_dir = root / "configs/forge/task-documentation"
    enriched: list[dict] = []

    seen_ids = {str(rec["task_id"]) for rec in task_defs}
    extended_defs = list(task_defs)
    # Also include Four-Laws adherence probe tasks so they have full schematics & gates in the Space.
    for cat_tid in sorted(catalog):
        if cat_tid in seen_ids:
            continue
        cat_item = catalog[cat_tid]
        fl_meta = cat_item.get("four_laws_adherence")
        if (isinstance(fl_meta, dict) and fl_meta.get("role") == "four_laws_probe") or cat_tid.startswith("fl_law"):
            seen_ids.add(cat_tid)
            extended_defs.append(
                {
                    "task_id": cat_tid,
                    "tier": 1,
                    "role": "diagnostic",
                    "views": ["four-laws-adherence-v1 (Tier 1)"],
                    "in_full_suite": False,
                    "summary": _opt_str(cat_item.get("intuition")) or "",
                    "reason": "four_laws_adherence_probe",
                }
            )

    for rec in extended_defs:
        item = dict(rec)
        task_id = str(item["task_id"])
        cat_entry = catalog.get(task_id) or {}
        task_file = tasks_dir / f"{task_id}.json"
        task_json: dict = {}
        if task_file.is_file():
            try:
                loaded = read_json(task_file)
                if isinstance(loaded, dict):
                    task_json = loaded
            except Exception:
                task_json = {}

        sidecar_doc: dict = {}
        sidecar_file = task_docs_dir / f"{task_id}.json"
        if sidecar_file.is_file():
            try:
                loaded_doc = read_json(sidecar_file)
                if isinstance(loaded_doc, dict):
                    sidecar_doc = loaded_doc
            except Exception:
                sidecar_doc = {}

        inline_doc = task_json.get("documentation") if isinstance(task_json.get("documentation"), dict) else {}
        # Priority: experiment_docs catalog -> sidecar task-documentation -> inline task["documentation"]
        doc_block = {**inline_doc, **sidecar_doc}
        execution = task_json.get("execution") if isinstance(task_json.get("execution"), dict) else {}
        evaluation = task_json.get("evaluation") if isinstance(task_json.get("evaluation"), dict) else {}
        host_def = execution.get("host_definition") if isinstance(execution.get("host_definition"), dict) else {}

        title = _opt_str(cat_entry.get("title")) or _opt_str(doc_block.get("title")) or _humanize_task_id(task_id)
        laws = _clean_str_list(cat_entry.get("laws") or doc_block.get("laws"))
        edge_cases = _clean_str_list(cat_entry.get("edge_cases") or doc_block.get("edge_cases"))
        intuition = (
            _opt_str(cat_entry.get("intuition"))
            or _opt_str(doc_block.get("intuition"))
            or _opt_str(task_json.get("description"))
            or _opt_str(item.get("summary"))
        )
        failure_mode = _opt_str(cat_entry.get("failure_mode")) or _opt_str(doc_block.get("failure_mode"))
        protocol_notes = (
            _opt_str(cat_entry.get("protocol_notes"))
            or _opt_str(doc_block.get("protocol_notes"))
            or _synthesize_protocol_notes(execution, evaluation)
        )
        prior_summary = _opt_str(cat_entry.get("prior_summary")) or _synthesize_prior_summary(execution, host_def)
        model_summary = _opt_str(cat_entry.get("model_summary")) or _synthesize_model_summary(
            task_json, execution, host_def
        )
        steps = _opt_int(cat_entry.get("steps")) or _opt_int(execution.get("steps")) or _opt_int(host_def.get("steps"))
        thresholds = _extract_thresholds(evaluation, cat_entry)
        mermaid = (
            _opt_str(cat_entry.get("mermaid"))
            or _opt_str(doc_block.get("custom_mermaid"))
            or _opt_str(doc_block.get("mermaid"))
            or _synthesize_task_mermaid(task_id, task_json, prior_summary, model_summary, steps, evaluation)
        )
        svg_relpath, svg_inline = _load_task_svg(root, task_id, cat_entry)
        reproduction_cmd = (
            _opt_str(cat_entry.get("reproduction_cmd"))
            or f"python -m experiments.forge probe {default_candidate} --task {task_id}"
        )

        item.update(
            {
                "title": title,
                "laws": laws,
                "edge_cases": edge_cases,
                "intuition": intuition,
                "failure_mode": failure_mode,
                "protocol_notes": protocol_notes,
                "prior_summary": prior_summary,
                "model_summary": model_summary,
                "steps": steps,
                "thresholds": thresholds,
                "mermaid": mermaid,
                "svg_relpath": svg_relpath,
                "svg_inline": svg_inline,
                "reproduction_cmd": reproduction_cmd,
                "pages_url": f"{base}/experiments/{task_id}/",
                "task_config_path": f"configs/forge/tasks/{task_id}.json" if task_file.is_file() else None,
                "adapter": _opt_str(task_json.get("adapter")),
                "sampling_law": _opt_str(evaluation.get("sampling_law")),
                "scoring_weights": _opt_str(evaluation.get("scoring_weights")),
            }
        )
        enriched.append(item)
    return enriched


def build_inventory_exports(
    root: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    generated_at: str | None = None,
    pages_base_url: str = DEFAULT_PAGES_BASE_URL,
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
    media_by_family_task, media_by_attempt_task, media_meta_by_family_task = _collect_media_with_manifest(root, progress)
    laws_meta, laws_list = _load_laws_registry_payload(root, pages_base_url=pages_base_url)

    fixed = load_fixed_test_list(root)
    full_suite_names = list(fixed["tests"]) if fixed else full_suite_tasks(progress)
    unimplemented_backlog = list(fixed.get("unimplemented_backlog") or []) if fixed else []
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

    cand_to_family: dict[str, str] = {}
    default_probe_candidate = "unified-v11"
    for family, cohort in selected:
        sel_row = _selected_row(publication, cohort)
        cid = sel_row.get("candidate_id") or family.get("benchmark_configuration", {}).get("candidate_id") or ""
        if cid:
            cand_to_family[str(cid)] = family["id"]
            if family["id"] == "unified":
                default_probe_candidate = str(cid)

    task_defs = _enrich_task_definitions(
        root,
        _collect_task_definitions(progress, full_suite_names),
        pages_base_url=pages_base_url,
        default_candidate=default_probe_candidate,
    )
    task_def_by_id = {d["task_id"]: d for d in task_defs}
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
                if steps is None:
                    steps = cost.get("completed_updates")
                metrics_obj = tr.get("metrics") if isinstance(tr.get("metrics"), dict) else {}
                if steps is None:
                    steps = metrics_obj.get("completed_steps") or metrics_obj.get("completed_updates")
                eval_summary = tr.get("evaluator_summary") if isinstance(tr.get("evaluator_summary"), dict) else None
                energy_info = _extract_receipt_energy(cost, eval_summary, metrics=metrics_obj)
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
        "occam_timeouts",
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

    # Index uncontended DCGAN r256 median latency by (family_id, candidate_id) from both
    # receipt-backed records and provenance-bound solo measurements in scaling-performance.json.
    scaling_r256_by_fam_cand: dict[tuple[str, str], float] = {}
    scaling_r256_by_fam: dict[str, float] = {}
    sp_pub = publication.get("scaling_performance") if isinstance(publication.get("scaling_performance"), dict) else {}
    for rec in sp_pub.get("records") or []:
        if not isinstance(rec, dict) or rec.get("task_id") != "dcgan_performance_r256":
            continue
        fid_s = _opt_str(rec.get("trainer_family"))
        cid_s = _opt_str(rec.get("candidate_id"))
        res_s = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        med = _opt_float((res_s.get("metrics") or {}).get("update_ms_median"))
        if fid_s and med is not None:
            if cid_s:
                scaling_r256_by_fam_cand.setdefault((fid_s, cid_s), med)
            scaling_r256_by_fam.setdefault(fid_s, med)
    sp_manifest_path = root / "reports/forge/scaling-performance.json"
    if sp_manifest_path.is_file():
        try:
            sp_doc = read_json(sp_manifest_path)
            for item in sp_doc.get("measurements") or []:
                if not isinstance(item, dict) or item.get("task_id") != "dcgan_performance_r256":
                    continue
                if item.get("contended") is True or str(item.get("status") or "COMPLETE").upper() != "COMPLETE":
                    continue
                fid_s = _opt_str(item.get("trainer_family")) or _opt_str(item.get("family_id"))
                cid_s = _opt_str(item.get("candidate_id"))
                med = _opt_float(item.get("median_ms"))
                if med is None and isinstance(item.get("metrics"), dict):
                    med = _opt_float(item["metrics"].get("update_ms_median"))
                if fid_s and med is not None:
                    if cid_s:
                        scaling_r256_by_fam_cand.setdefault((fid_s, cid_s), med)
                    scaling_r256_by_fam.setdefault(fid_s, med)
        except Exception:
            pass

    from .views import compatible_diagnostic_passes

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
        fam_r256_ms = (
            scaling_r256_by_fam_cand.get((fam_id, candidate_id))
            if candidate_id
            else None
        )
        if fam_r256_ms is None:
            fam_r256_ms = scaling_r256_by_fam.get(fam_id)
        if fam_r256_ms is None and fam_id == "bcap-pure":
            fam_r256_ms = scaling_r256_by_fam.get("bcap-dualnorm")
        fam_diag_passes = compatible_diagnostic_passes(
            root,
            candidate_id=candidate_id,
            candidate_revision=cand_rev,
            source_digest=source_digest,
        )
        fam_doc = _load_family_doc_enrichment(
            root,
            fam_id,
            family.get("tags"),
            pages_base_url=pages_base_url,
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
                "pages_url": fam_doc["pages_url"],
                "overview": fam_doc["overview"],
                "law_mechanisms": fam_doc["law_mechanisms"],
                "equations": fam_doc["equations"],
                "mermaid_diagram": fam_doc["mermaid_diagram"],
                "quickstart": fam_doc["quickstart"],
                "tags": fam_doc["tags"],
                "stale": is_stale,
                "stale_note": stale_note,
                "qualified_tier": int(row.get("qualified_tier") or 0),
                "qualification_note": qual_note,
                "tuning_budget": tuning_budget,
                "holdout_claim": holdout_text,
                "scaling_r256_ms": fam_r256_ms,
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
            diag_pass = fam_diag_passes.get(task_id) if raw_status in {"UNKNOWN", "NOT_RUN"} else None
            if raw_status == "UNKNOWN":
                run_attempt = cell.get("run_attempt") or {}
                if cell.get("evidence_correction"):
                    display_status = "diagnostic"
                elif diag_pass is not None:
                    display_status = "DIAG_PASS"
                elif run_attempt.get("status"):
                    display_status = str(run_attempt["status"])
                elif coverage_state:
                    display_status = str(coverage_state)
                else:
                    display_status = "not_run"
            elif raw_status == "NOT_RUN" and diag_pass is not None:
                display_status = "DIAG_PASS"
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
            gif_meta = media_meta_by_family_task.get((fam_id, task_id)) if gif_relpath else None
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
                    **({"gif_meta": gif_meta} if gif_meta else {}),
                    **({"diagnostic_pass": diag_pass} if diag_pass else {}),
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
    four_laws_l4 = _build_four_laws_adherence_l4_payload(root, pages_base_url=pages_base_url)

    leaderboard_payload = {
        "schema_version": 1,
        "schema": "particlegan_forge_leaderboard_v1",
        "generated_at": timestamp,
        "git_commit": git_commit,
        "dataset_repo": repo,
        "media_prefix": DEFAULT_MEDIA_PREFIX,
        "pages_base_url": pages_base_url.rstrip("/"),
        "view": publication.get("view", "discriminator_stability"),
        "view_revision": int(publication.get("view_revision") or 0),
        "declared_view_revisions": declared_view_revisions,
        "laws_meta": laws_meta,
        "laws": laws_list,
        "four_laws_adherence_l4": four_laws_l4,
        "full_suite_list": (
            {
                "id": fixed["id"],
                "dated": fixed["dated"],
                "sha256": fixed["sha256"],
                "count": len(full_suite_names),
                "unimplemented_backlog_count": len(unimplemented_backlog),
                "unimplemented_backlog": unimplemented_backlog,
            }
            if fixed
            else {
                "id": "current_views",
                "dated": None,
                "sha256": None,
                "count": len(full_suite_names),
                "unimplemented_backlog_count": 0,
                "unimplemented_backlog": [],
            }
        ),
        "summary": {
            "selected_families": len(families_json),
            "counted_view_requirements": int(counted_reqs),
            "distinct_required_tests": int(distinct_required),
            "full_suite_tests": len(full_suite_names),
            "unimplemented_backlog_tests": len(unimplemented_backlog),
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


def export_leaderboard_json(
    root: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    generated_at: str | None = None,
    pages_base_url: str = DEFAULT_PAGES_BASE_URL,
) -> dict:
    """Export the enriched, v1-compatible ``leaderboard.json`` payload offline."""
    exports = build_inventory_exports(
        root,
        repo=repo,
        generated_at=generated_at,
        pages_base_url=pages_base_url,
    )
    return exports["leaderboard_json"]


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
            ("occam_timeouts", pa.int64()),
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
        f"| `leaderboard` | `data/leaderboard.parquet` | Selected recipes per family with Tier 1–4 counts, full-suite {summary['full_suite_tests']}-test counts, stretch counts, matched nullable energy summaries (`r256_joules_per_step`, `ring16_j_to_pass`, `grid100_j_to_pass`), and tuning-budget disclosures. |",
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
