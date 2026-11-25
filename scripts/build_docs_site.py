#!/usr/bin/env python3
"""Deterministic staging and build tool for the ParticleGAN MkDocs Material Research Portal.

Assembles a clean temporary MkDocs ``docs_dir`` from the repository's real source
files (never duplicating committed Markdown under ``reports/``), rewrites relative
repository links and GitHub Alerts for MkDocs Material, exports a compact
``assets/data/leaderboard.json`` snapshot (offline, zero network calls, zero
PyTorch dependency required), generates dynamic navigation in ``<build-dir>/mkdocs.yml``,
and optionally runs ``mkdocs build --strict`` or ``mkdocs serve``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPO_SLUG = "HyperGAN/ParticleGAN"
DEFAULT_GIT_REF = "develop"
DEFAULT_HF_REPO = "HyperGAN/particlegan-forge-evidence"
DEFAULT_MEDIA_PREFIX = "media/forge/"
FIXED_TEST_LIST_REL = Path("configs/forge/leaderboards/technique-inventory-full-suite-2026-11-23.json")

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

FALLBACK_LAWS_REGISTRY = {
    "schema_version": 1,
    "epistemic_status": "provisional_evolving",
    "epigraph": (
        "Our goal is not to preserve the four laws. Our goal is to discover the laws "
        "of learning, even if doing so ultimately requires abandoning the current four."
    ),
    "epistemic_note": (
        "This law taxonomy is a living scientific framework derived from current "
        "mathematical and empirical audits. Individual laws may be refined, split, "
        "merged, or replaced as new counterexamples and evidence emerge."
    ),
    "doc_path": "docs/four-laws.md",
    "laws": [
        {
            "id": "law1",
            "short_label": "Law 1",
            "title": "Representational Geometry",
            "domain_level": "Representational Geometry",
            "question": "Can the system represent the required probability measure efficiently and faithfully?",
            "doc_anchor": "law-1-representational-geometry",
            "status": "active",
            "sub_laws": [],
        },
        {
            "id": "law2",
            "short_label": "Law 2",
            "title": "Potential Geometry",
            "domain_level": "Potential Geometry",
            "question": (
                "Does the critic communicate useful learning forces with appropriate "
                "geometric control and computational efficiency?"
            ),
            "doc_anchor": "law-2-potential-geometry",
            "status": "active",
            "sub_laws": [],
        },
        {
            "id": "law3",
            "short_label": "Law 3",
            "title": "Transport Identifiability",
            "domain_level": "Transport Identifiability",
            "question": "Does the learning system possess enough information to determine appropriate transport?",
            "doc_anchor": "law-3-transport-identifiability",
            "status": "active",
            "sub_laws": [
                {
                    "id": "law3a",
                    "short_label": "Law 3A",
                    "title": "Unconditional W1 Ray-Length & Screening Identifiability",
                    "doc_anchor": "law-3-transport-identifiability",
                },
                {
                    "id": "law3b",
                    "short_label": "Law 3B",
                    "title": "Conditional Fiber & Stratum Identifiability",
                    "doc_anchor": "law-3-transport-identifiability",
                },
            ],
        },
        {
            "id": "law4",
            "short_label": "Law 4",
            "title": "Functional and Game Dynamics",
            "domain_level": "Functional and Game Dynamics",
            "question": (
                "Can the learning system realize useful functional changes and exhibit "
                "appropriate coupled learning dynamics?"
            ),
            "doc_anchor": "law-4-functional-and-game-dynamics",
            "status": "active",
            "sub_laws": [
                {
                    "id": "law4_1",
                    "short_label": "Law 4.1",
                    "title": "Functional Realization",
                    "doc_anchor": "law-4-functional-and-game-dynamics",
                },
                {
                    "id": "law4_2",
                    "short_label": "Law 4.2",
                    "title": "Pathology-Driven Spectral Conditioning",
                    "doc_anchor": "law-4-functional-and-game-dynamics",
                },
                {
                    "id": "law4_3",
                    "short_label": "Law 4.3",
                    "title": "Displacement & Kreiss Control",
                    "doc_anchor": "law-4-functional-and-game-dynamics",
                },
                {
                    "id": "law4_4",
                    "short_label": "Law 4.4",
                    "title": "Trajectory Scale vs. Settling",
                    "doc_anchor": "law-4-functional-and-game-dynamics",
                },
                {
                    "id": "law4_5",
                    "short_label": "Law 4.5",
                    "title": "Coupled Minimax Contraction",
                    "doc_anchor": "law-4-functional-and-game-dynamics",
                },
            ],
        },
    ],
}

FENCE_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
GH_ALERT_RE = re.compile(r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*)$", re.IGNORECASE)
REF_DEF_RE = re.compile(r"^([ \t]*\[[^\]]+\]:[ \t]*)(\S+)(.*)$")
LINE_ANCHOR_RE = re.compile(r"^L\d+(?:[-,]L?\d+)*$")
LINE_TOKEN_RE = re.compile(
    r"(?P<md_label>!?\[(?:`+[^`]*`+|\[[^\]]*\]|[^\]\[`])*\])\((?P<md_target>[^)\n]+)\)"
    r"|(?P<inline_code>`+[^`]*`+)"
    r"|(?P<anchor_name><a\s+name=(?P<anchor_q>[\"'])(?P<anchor_val>[^\"']+)(?P=anchor_q)\s*>)"
    r"|(?P<html_prefix><(?:a|img)\b[^>]*?\b(?:href|src)\s*=\s*)(?P<html_q>[\"'])(?P<html_target>[^\"']+)(?P=html_q)",
    re.IGNORECASE,
)
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
AUDIT_P_RE = re.compile(r"^p(\d+)-")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _git_head_commit(root: Path) -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha or None
    except (OSError, subprocess.CalledProcessError):
        return None


def load_laws_registry(root: Path) -> dict:
    """Load evolving laws registry from configs/forge/laws.json with clean fallback."""
    laws_path = root / "configs/forge/laws.json"
    if laws_path.is_file():
        try:
            doc = _read_json(laws_path)
            if isinstance(doc, dict) and isinstance(doc.get("laws"), list) and doc["laws"]:
                return doc
        except Exception:
            pass
    return json.loads(json.dumps(FALLBACK_LAWS_REGISTRY))


def _load_hf_revision(root: Path) -> str:
    receipt = root / "reports/forge/publications/inventory-latest.json"
    if receipt.is_file():
        try:
            doc = _read_json(receipt)
            rev = _opt_str(doc.get("hf_revision"))
            if rev:
                return rev
        except Exception:
            pass
    return "main"


def _load_scaling_r256_ms(root: Path) -> dict[str, float]:
    """Extract matched idle r256 median update latency (ms/step) by family id if present."""
    path = root / "reports/forge/scaling-performance.json"
    if not path.is_file():
        return {}
    try:
        doc = _read_json(path)
    except Exception:
        return {}
    out: dict[str, float] = {}
    records = doc.get("records") or doc.get("measurements") or doc.get("entries") or []
    if isinstance(records, dict):
        records = list(records.values())
    for rec in records:
        if not isinstance(rec, dict):
            continue
        task = rec.get("task") or rec.get("task_id") or ""
        if task != "dcgan_performance_r256":
            continue
        fam = _opt_str(rec.get("family")) or _opt_str(rec.get("family_id")) or _opt_str(rec.get("reporting_family"))
        ms = (
            _opt_float(rec.get("median_ms"))
            or _opt_float(rec.get("median_update_ms"))
            or _opt_float((rec.get("timing") or {}).get("median_ms"))
        )
        if fam and ms is not None:
            out.setdefault(fam, ms)
            if fam == "bcap-dualnorm":
                out.setdefault("bcap-pure", ms)
    return out


# ---------------------------------------------------------------------------
# Zero-torch stdlib fallback exporter for assets/data/leaderboard.json
# ---------------------------------------------------------------------------


def _gif_to_media_relpath(gif_path: str | None) -> str | None:
    if not isinstance(gif_path, str) or not gif_path:
        return None
    normalized = gif_path.replace("\\", "/").lstrip("./")
    if normalized.startswith("reports/forge/"):
        return DEFAULT_MEDIA_PREFIX + normalized[len("reports/forge/"):]
    if normalized.startswith("media/"):
        return normalized
    return DEFAULT_MEDIA_PREFIX + normalized


def _tier_bucket_stdlib(tier_data: dict) -> dict:
    counts = tier_data.get("counts", {}) if isinstance(tier_data, dict) else {}
    coverage = tier_data.get("attempt_coverage", {}) if isinstance(tier_data, dict) else {}
    not_run_cov = coverage.get("not_run")
    return {
        "pass": int(tier_data.get("passed", counts.get("PASS", 0))) if isinstance(tier_data, dict) else 0,
        "total": int(tier_data.get("total", 0)) if isinstance(tier_data, dict) else 0,
        "fail": int(counts.get("FAIL", 0)),
        "not_run": int(not_run_cov) if not_run_cov is not None else int(tier_data.get("not_run", 0) if isinstance(tier_data, dict) else 0),
        "blocked": int(coverage.get("blocked", 0)),
        "diagnostic": int(coverage.get("diagnostic_attempts", 0)),
        "incompatible": int(coverage.get("incompatible", tier_data.get("incompatible", 0) if isinstance(tier_data, dict) else 0)),
    }


def _collect_study_energy_stdlib(root: Path, cand_to_family: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    studies_dir = root / "reports/forge/studies"
    if not studies_dir.is_dir():
        return rows
    idea_cache: dict[str, dict | None] = {}

    def _idea(cid: str) -> dict | None:
        if cid not in idea_cache:
            p = root / "configs/forge/ideas" / f"{cid}.json"
            try:
                idea_cache[cid] = _read_json(p) if p.is_file() else None
            except Exception:
                idea_cache[cid] = None
        return idea_cache[cid]

    def _readout_occam(readout_name: str | None) -> tuple[int | None, int | None, str | None]:
        if not isinstance(readout_name, str) or not readout_name.strip():
            return None, None, None
        p = root / "reports/forge/records" / readout_name.strip()
        if not p.is_file():
            return None, None, None
        try:
            doc = _read_json(p)
        except Exception:
            return None, None, None
        prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
        src_commit = _opt_str(prov.get("source_origin_commit")) or _opt_str(doc.get("source_commit"))
        trs = doc.get("task_results") or []
        executed_statuses = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "ERROR", "TIMEOUT"}
        passed = sum(1 for tr in trs if isinstance(tr, dict) and (tr.get("gate_status") or tr.get("status")) == "PASS")
        executed = sum(
            1 for tr in trs if isinstance(tr, dict) and (tr.get("gate_status") or tr.get("status")) in executed_statuses
        )
        return (passed if executed > 0 else None), (executed if executed > 0 else None), src_commit

    for study_file in sorted(studies_dir.glob("*energy-benchmark*.json")):
        try:
            doc = _read_json(study_file)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        rel_source = f"reports/forge/studies/{study_file.name}"
        default_gpu = _opt_str(doc.get("gpu_name"))
        file_commit = _opt_str(doc.get("source_commit")) or _opt_str(doc.get("git_commit"))
        bench_map = doc.get("energy_benchmark")
        if isinstance(bench_map, dict):
            for cid in sorted(bench_map):
                entry = bench_map[cid]
                if not isinstance(entry, dict):
                    continue
                r256 = entry.get("dcgan_r256") if isinstance(entry.get("dcgan_r256"), dict) else {}
                ring16 = entry.get("ring16_acquisition") if isinstance(entry.get("ring16_acquisition"), dict) else {}
                grid100 = entry.get("grid100") if isinstance(entry.get("grid100"), dict) else {}
                staggered100 = entry.get("staggered100") if isinstance(entry.get("staggered100"), dict) else {}
                sustained = r256.get("sustained_uninstrumented") if isinstance(r256.get("sustained_uninstrumented"), dict) else {}
                step_ms_obj = sustained.get("step_ms") if isinstance(sustained.get("step_ms"), dict) else {}
                power_poll = sustained.get("power_poll_watts") if isinstance(sustained.get("power_poll_watts"), dict) else {}
                opt_state = r256.get("optimizer_state") if isinstance(r256.get("optimizer_state"), dict) else {}
                idea = _idea(cid) or {}
                overrides = idea.get("recipe_overrides") if isinstance(idea.get("recipe_overrides"), dict) else {}
                ro_pass, ro_exec, ro_commit = _readout_occam(grid100.get("official_forge_readout"))
                util_before = r256.get("gpu_util_before")
                contended = _opt_bool(entry.get("contended"))
                if contended is None:
                    contended = _opt_bool(r256.get("contended"))
                if contended is None and isinstance(util_before, list) and util_before and isinstance(util_before[0], (int, float)):
                    contended = bool(util_before[0] > 5)
                rows.append(
                    {
                        "candidate_id": str(cid),
                        "family_id": _opt_str(entry.get("family_id")) or cand_to_family.get(cid) or _opt_str(idea.get("trainer_family")),
                        "optimizer_family": (
                            _opt_str(r256.get("optimizer_family"))
                            or _opt_str(grid100.get("optimizer_family"))
                            or _opt_str(entry.get("optimizer_family"))
                            or _opt_str(overrides.get("optimizer_family"))
                        ),
                        "hardware_gpu": _opt_str(r256.get("gpu_name")) or _opt_str(entry.get("hardware_gpu")) or default_gpu,
                        "contended": contended,
                        "source_commit": _opt_str(entry.get("source_commit")) or ro_commit or file_commit,
                        "source_file": rel_source,
                        "r256_ms_per_step": _opt_float(step_ms_obj.get("median")) or _opt_float(step_ms_obj.get("mean")) or _opt_float(r256.get("ms_per_step")),
                        "r256_watts": _opt_float(sustained.get("avg_power_watts_from_energy")) or _opt_float(power_poll.get("median")) or _opt_float(r256.get("avg_power_watts")),
                        "r256_joules_per_step": _opt_float(sustained.get("joules_per_step")) or _opt_float(r256.get("joules_per_step")),
                        "state_bytes_per_param": _opt_float(opt_state.get("state_bytes_per_param")) or _opt_float(grid100.get("state_bytes_per_param")),
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
                        "occam_pass": _opt_int(entry.get("occam_pass")) if _opt_int(entry.get("occam_pass")) is not None else ro_pass,
                        "occam_executed": _opt_int(entry.get("occam_executed")) if _opt_int(entry.get("occam_executed")) is not None else ro_exec,
                        "evidence_scope": _opt_str(entry.get("evidence_scope")) or "study_energy_benchmark",
                    }
                )
    return rows


def build_leaderboard_json_stdlib(root: Path, *, generated_at: str | None = None) -> dict:
    """Build a schema-compatible leaderboard.json snapshot using Python stdlib only."""
    root = Path(root).resolve()
    inv_path = root / "reports/forge/technique-inventory.json"
    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = _git_head_commit(root)
    if not inv_path.is_file():
        return {
            "schema_version": 1,
            "schema": "particlegan_forge_leaderboard_v1",
            "generated_at": timestamp,
            "git_commit": git_commit,
            "dataset_repo": DEFAULT_HF_REPO,
            "media_prefix": DEFAULT_MEDIA_PREFIX,
            "summary": {"selected_families": 0, "full_suite_tests": 0, "energy_benchmark_rows": 0},
            "caveats": list(SCIENTIFIC_CAVEATS),
            "families": [],
            "tasks": [],
            "task_definitions": [],
            "energy_benchmarks": [],
        }

    publication = _read_json(inv_path)
    progress = publication.get("family_progress", {})
    excluded_views = set(progress.get("diagnostic_views", [])) | set(progress.get("scoped_views", []))
    summed = [v for v in progress.get("views", []) if v.get("id") not in excluded_views]
    config_families = {
        item["id"]: item for item in progress.get("configuration_families", []) if isinstance(item, dict)
    }
    stale_families = {
        p["trainer_family"]
        for p in publication.get("retained_recorded_measurements", {}).get("proofs", [])
        if isinstance(p, dict) and p.get("trainer_family")
    }

    fixed_path = root / FIXED_TEST_LIST_REL
    fixed_doc = _read_json(fixed_path) if fixed_path.is_file() else None
    fixed_sha = hashlib.sha256(fixed_path.read_bytes()).hexdigest() if fixed_path.is_file() else None
    counted_task_set = {
        a["task"] for view in summed for a in view.get("assignments", []) if a.get("importance") == "required"
    }
    if fixed_doc and isinstance(fixed_doc.get("tests"), list):
        full_suite_names = list(fixed_doc["tests"])
    else:
        full_suite_names = sorted({a["task"] for view in summed for a in view.get("assignments", [])})
    not_counted_set = set(full_suite_names) - counted_task_set

    assignments = [a for view in summed for a in view.get("assignments", [])]
    counted_reqs = sum(1 for a in assignments if a.get("importance") == "required")
    stretch_assignments = [
        a for a in assignments if a.get("importance") == "stretch" or "stretch" in a
    ]
    held_assignments = [a for a in assignments if a.get("importance") == "held"]
    removed_assignments = [a for a in assignments if "removal" in a]

    # Index task definitions
    task_def_map: dict[str, dict] = {}
    role_priority = {"required": 0, "stretch": 1, "held": 2, "removed": 3}
    for view in summed:
        vid = view.get("id", "")
        for a in view.get("assignments", []):
            tid = a["task"]
            tier = int(a.get("qualification_tier", 0))
            if "removal" in a:
                role, note = "removed", a["removal"]
            elif a.get("importance") == "held":
                role, note = "held", a.get("hold", {})
            elif a.get("importance") == "stretch" or "stretch" in a:
                role, note = "stretch", a.get("stretch", {})
            else:
                role, note = "required", {}
            rec = task_def_map.setdefault(
                tid,
                {
                    "task_id": tid,
                    "tier": tier,
                    "role": role,
                    "views": [],
                    "summary": note.get("summary") or "",
                    "re_entry_rule": note.get("reentry") or note.get("graduation"),
                    "in_full_suite": tid in set(full_suite_names),
                },
            )
            rec["tier"] = min(rec["tier"], tier)
            if role_priority[role] < role_priority[rec["role"]]:
                rec["role"] = role
            vlabel = f"{vid} (Tier {tier})"
            if vlabel not in rec["views"]:
                rec["views"].append(vlabel)
    for tid in full_suite_names:
        task_def_map.setdefault(
            tid,
            {
                "task_id": tid,
                "tier": 0,
                "role": "required",
                "views": [],
                "summary": "",
                "re_entry_rule": None,
                "in_full_suite": True,
            },
        )
    task_defs = [task_def_map[k] for k in sorted(task_def_map)]

    # Index media GIFs
    media_by_family_task: dict[tuple[str, str], str] = {}
    for fam in list(progress.get("families", [])) + list(progress.get("configuration_families", [])):
        fid = fam.get("id", "")
        for cohort in fam.get("cohorts", []):
            for tid, cell in (cohort.get("tasks") or {}).items():
                for art in cell.get("training_media") or []:
                    rel = _gif_to_media_relpath(art.get("gif"))
                    if rel:
                        media_by_family_task.setdefault((fid, tid), rel)

    selected_rows_by_key = {}
    for row in publication.get("rows", []):
        if isinstance(row, dict):
            selected_rows_by_key[(row.get("trainer_family"), row.get("backend"), row.get("candidate_id"))] = row
            selected_rows_by_key[(row.get("trainer_family"), row.get("backend"))] = row

    families_json: list[dict] = []
    family_tasks_json: list[dict] = []
    cand_to_family: dict[str, str] = {}

    for family in progress.get("families", []):
        cohorts = family.get("cohorts", [])
        if not cohorts:
            continue
        anchor_pin = family.get("current_cohort_anchor")
        selected_cohorts = [c for c in cohorts if not anchor_pin or c.get("anchor") == anchor_pin]
        if not selected_cohorts:
            continue
        selected_cohort = selected_cohorts[0]
        fam_id = family["id"]
        fam_label = family.get("label", fam_id)
        cfg_fam_id = selected_cohort.get("configuration_family")
        cfg_fam = config_families.get(cfg_fam_id) if cfg_fam_id else None
        selected_recipe_label = cfg_fam["label"] if cfg_fam else fam_label
        backend = selected_cohort.get("backend") or "cuda"
        cand_id = (
            selected_cohort.get("candidate_id")
            or family.get("benchmark_configuration", {}).get("candidate_id")
            or ""
        )
        row = (
            selected_rows_by_key.get((fam_id, backend, cand_id))
            or selected_rows_by_key.get((fam_id, backend))
            or {}
        )
        cand_id = row.get("candidate_id") or cand_id
        if cand_id:
            cand_to_family[str(cand_id)] = fam_id
        bindings = row.get("bindings", {}) if isinstance(row.get("bindings"), dict) else {}
        is_stale = fam_id in stale_families or (cfg_fam_id in stale_families if cfg_fam_id else False)
        tiers_dict = {t: _tier_bucket_stdlib(selected_cohort.get("tiers", {}).get(t, {})) for t in ("1", "2", "3", "4")}
        total_dict = _tier_bucket_stdlib(selected_cohort.get("total", {}))

        cohort_tasks = selected_cohort.get("tasks", {})
        statuses = {tid: (cohort_tasks.get(tid) or {}).get("status", "UNKNOWN") for tid in full_suite_names}
        counts = Counter(statuses.values())
        not_measured_keys = {"UNKNOWN", "NOT_RUN"}
        other_breakdown = {
            k: int(v) for k, v in sorted(counts.items()) if k not in {"PASS", "FAIL"} and k not in not_measured_keys
        }
        held_names = [tid for tid in full_suite_names if tid in not_counted_set]
        held_counts = Counter(statuses[tid] for tid in held_names)
        held_pass = int(held_counts.get("PASS", 0))
        held_fail = int(held_counts.get("FAIL", 0))
        held_nm = int(sum(held_counts.get(k, 0) for k in not_measured_keys))

        # Stretch totals
        st_tiers = {}
        st_pass_all = 0
        st_tot_all = 0
        for t in ("1", "2", "3", "4"):
            t_assigns = [a for a in stretch_assignments if str(a.get("qualification_tier")) == t]
            tp = sum(1 for a in t_assigns if (cohort_tasks.get(a["task"]) or {}).get("status") == "PASS")
            st_tiers[t] = {"pass": tp, "total": len(t_assigns)}
            st_pass_all += tp
            st_tot_all += len(t_assigns)

        families_json.append(
            {
                "id": fam_id,
                "label": fam_label,
                "selected_recipe": selected_recipe_label,
                "configuration_family": cfg_fam_id,
                "candidate_id": cand_id,
                "candidate_revision": row.get("candidate_revision") or "",
                "recipe_sha256": bindings.get("recipe_sha256") or "",
                "source_digest": bindings.get("source_digest") or "",
                "source_origin_commit": bindings.get("source_origin_commit") or "",
                "backend": backend,
                "cohort_anchor": selected_cohort.get("anchor") or "",
                "family_page": family.get("page") or f"reports/forge/families/{fam_id}.md",
                "stale": is_stale,
                "stale_note": "Recorded source measurement · stale; no current qualification" if is_stale else None,
                "qualified_tier": int(row.get("qualified_tier") or 0),
                "qualification_note": None,
                "tuning_budget": None,
                "holdout_claim": None,
                "r256_joules_per_step": None,
                "ring16_j_to_pass": None,
                "grid100_j_to_pass": None,
                "tiers": tiers_dict,
                "total_view_requirements": total_dict,
                "full_suite": {
                    "pass": int(counts.get("PASS", 0)),
                    "fail": int(counts.get("FAIL", 0)),
                    "not_measured": int(sum(counts.get(k, 0) for k in not_measured_keys)),
                    "other": int(sum(other_breakdown.values())),
                    "other_breakdown": other_breakdown,
                    "total": len(full_suite_names),
                    "held": {
                        "pass": held_pass,
                        "fail": held_fail,
                        "not_measured": held_nm,
                        "other": max(0, len(held_names) - held_pass - held_fail - held_nm),
                        "total": len(held_names),
                    },
                },
                "stretch": {
                    "tiers": st_tiers,
                    "total": {"pass": st_pass_all, "total": st_tot_all},
                },
            }
        )

        for tid in sorted(set(task_def_map) | set(cohort_tasks)):
            cell = cohort_tasks.get(tid, {})
            tdef = task_def_map.get(tid, {"tier": 0, "role": "diagnostic", "views": []})
            raw_status = str(cell.get("status") or "UNKNOWN")
            display_status = raw_status if raw_status != "UNKNOWN" else (
                str((cell.get("run_attempt") or {}).get("status") or cell.get("coverage_status") or "not_run")
            )
            gif_rel = None
            for art in cell.get("training_media") or []:
                gif_rel = _gif_to_media_relpath(art.get("gif"))
                if gif_rel:
                    break
            if gif_rel is None:
                gif_rel = media_by_family_task.get((fam_id, tid))
            raw_metrics = cell.get("metrics") if isinstance(cell.get("metrics"), dict) else {}
            cost = raw_metrics.get("cost") if isinstance(raw_metrics.get("cost"), dict) else {}
            inner = raw_metrics.get("metrics") if isinstance(raw_metrics.get("metrics"), dict) else raw_metrics
            summary_m = {
                k: v for k, v in sorted(inner.items()) if isinstance(v, (int, float, bool, str)) or v is None
            }
            family_tasks_json.append(
                {
                    "family_id": fam_id,
                    "family_label": fam_label,
                    "candidate_id": cand_id,
                    "task_id": tid,
                    "tier": int(tdef["tier"]),
                    "role": tdef["role"],
                    "views": list(tdef["views"]),
                    "status": display_status,
                    "recorded_status": raw_status,
                    "reason": cell.get("reason") or cell.get("coverage_reason") or "",
                    "wall_seconds": _opt_float(cost.get("wall_seconds")),
                    "metrics_summary": summary_m,
                    "gif_relpath": gif_rel,
                }
            )

    energy_rows = _collect_study_energy_stdlib(root, cand_to_family)

    return {
        "schema_version": 1,
        "schema": "particlegan_forge_leaderboard_v1",
        "generated_at": timestamp,
        "git_commit": git_commit,
        "dataset_repo": DEFAULT_HF_REPO,
        "media_prefix": DEFAULT_MEDIA_PREFIX,
        "view": publication.get("view", "discriminator_stability"),
        "view_revision": int(publication.get("view_revision") or 0),
        "declared_view_revisions": {v["id"]: int(v.get("revision", 0)) for v in summed if "id" in v},
        "full_suite_list": (
            {
                "id": fixed_doc["id"],
                "dated": fixed_doc["dated"],
                "sha256": fixed_sha,
                "count": len(full_suite_names),
            }
            if fixed_doc
            else {"id": "current_views", "dated": None, "sha256": None, "count": len(full_suite_names)}
        ),
        "summary": {
            "selected_families": len(families_json),
            "counted_view_requirements": int(counted_reqs),
            "distinct_required_tests": len(counted_task_set),
            "full_suite_tests": len(full_suite_names),
            "stretch_assignments": len(stretch_assignments),
            "stretch_distinct_tests": len({a["task"] for a in stretch_assignments}),
            "held_assignments": len(held_assignments),
            "held_distinct_tests": len({a["task"] for a in held_assignments}),
            "removed_assignments": len(removed_assignments),
            "removed_distinct_tests": len({a["task"] for a in removed_assignments}),
            "energy_benchmark_rows": len(energy_rows),
        },
        "caveats": list(SCIENTIFIC_CAVEATS),
        "families": families_json,
        "tasks": family_tasks_json,
        "task_definitions": task_defs,
        "energy_benchmarks": energy_rows,
    }


def build_portal_leaderboard_payload(root: Path, *, prefer_stdlib: bool = False) -> dict:
    """Build enriched offline leaderboard.json for the portal widget.

    Uses ``experiments.forge.hf_publish`` when available (e.g., local research env),
    and seamlessly falls back to the zero-torch stdlib exporter (e.g., lightweight
    GitHub Pages CI without PyTorch).
    """
    root = Path(root).resolve()
    payload: dict | None = None

    if not prefer_stdlib:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            from experiments.forge import hf_publish

            if hasattr(hf_publish, "export_leaderboard_json"):
                payload = hf_publish.export_leaderboard_json(root)
            else:
                payload = hf_publish.build_inventory_exports(root)["leaderboard_json"]
        except Exception:
            payload = None

    if payload is None:
        payload = build_leaderboard_json_stdlib(root)

    # Additive portal enrichment (idempotent when hf_publish already enriched)
    payload.setdefault("hf_revision", _load_hf_revision(root))
    laws_reg = load_laws_registry(root)
    if not payload.get("laws"):
        payload["laws"] = laws_reg.get("laws", [])
    if not payload.get("laws_meta"):
        payload["laws_meta"] = {
            "schema_version": laws_reg.get("schema_version", 1),
            "epistemic_status": laws_reg.get("epistemic_status", "provisional_evolving"),
            "epigraph": laws_reg.get("epigraph", ""),
            "epistemic_note": laws_reg.get("epistemic_note", ""),
            "doc_path": laws_reg.get("doc_path", "docs/four-laws.md"),
        }

    scaling_ms = _load_scaling_r256_ms(root)
    for fam in payload.get("families", []):
        fid = fam.get("id", "")
        if fid in scaling_ms and fam.get("scaling_r256_ms") is None:
            fam["scaling_r256_ms"] = scaling_ms[fid]
        doc_path = root / "configs/forge/family-documentation" / f"{fid}.json"
        if doc_path.is_file():
            try:
                fdoc = _read_json(doc_path)
            except Exception:
                fdoc = {}
            if isinstance(fdoc, dict):
                if not fam.get("overview") and fdoc.get("overview"):
                    fam["overview"] = fdoc["overview"]
                if not fam.get("law_mechanisms") and isinstance(fdoc.get("law_mechanisms"), dict):
                    fam["law_mechanisms"] = fdoc["law_mechanisms"]
                if not fam.get("quickstart") and isinstance(fdoc.get("quickstart"), dict):
                    fam["quickstart"] = fdoc["quickstart"]

    task_doc_dir = root / "configs/forge/task-documentation"
    exp_assets_dir = root / "reports/forge/experiments/assets"
    if task_doc_dir.is_dir() or exp_assets_dir.is_dir():
        for td in payload.get("task_definitions", []):
            tid = td.get("task_id", "")
            tdoc_path = task_doc_dir / f"{tid}.json"
            if tdoc_path.is_file():
                try:
                    tdoc = _read_json(tdoc_path)
                except Exception:
                    tdoc = {}
                if isinstance(tdoc, dict):
                    for key in ("title", "laws", "edge_cases", "intuition", "failure_mode", "protocol_notes"):
                        if key not in td and key in tdoc:
                            td[key] = tdoc[key]
            svg_path = exp_assets_dir / f"{tid}.svg"
            if svg_path.is_file() and not td.get("svg_relpath"):
                td["svg_relpath"] = f"experiments/assets/{tid}.svg"

    return payload


# ---------------------------------------------------------------------------
# Markdown Transform Pipeline (GitHub Alerts, Anchors, Smart Link Rewriting)
# ---------------------------------------------------------------------------


def convert_github_alerts(text: str) -> str:
    """Convert GitHub blockquote alerts (> [!IMPORTANT]) to MkDocs Material admonitions."""
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m_fence = FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(2)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and not m_fence.group(3).strip():
                in_fence = False
            out.append(line)
            i += 1
            continue

        if not in_fence:
            m_alert = GH_ALERT_RE.match(line)
            if m_alert:
                kind = m_alert.group(1).lower()
                title = m_alert.group(2).strip()
                body: list[str] = []
                i += 1
                while i < n and lines[i].lstrip().startswith(">"):
                    stripped = re.sub(r"^\s*>\s?", "", lines[i], count=1)
                    body.append(stripped)
                    i += 1
                if out and out[-1].strip():
                    out.append("")
                header = f'!!! {kind} "{title}"' if title else f"!!! {kind}"
                out.append(header)
                if not body:
                    out.append("    ")
                else:
                    for bline in body:
                        out.append(f"    {bline}" if bline else "")
                if i < n and lines[i].strip():
                    out.append("")
                continue

        out.append(line)
        i += 1

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + suffix


def _split_md_target(raw: str) -> tuple[str, str]:
    """Split a Markdown link target into (url_part, optional_title_suffix)."""
    s = raw.strip()
    if s.startswith("<"):
        end = s.find(">")
        if end != -1:
            return s[1:end], s[end + 1 :]
    m = re.match(r'^(\S+)(\s+["\'].*["\'])$', s)
    if m:
        return m.group(1), m.group(2)
    return s, ""


def _resolve_target(
    raw_url: str,
    *,
    is_image: bool,
    src_repo_rel: str,
    dst_docs_rel: str,
    repo_to_staged: dict[str, str],
    staged_paths: set[str],
    root: Path,
    git_ref: str,
    repo_slug: str,
) -> str:
    url = raw_url.strip()
    if not url or url.startswith("#") or SCHEME_RE.match(url):
        return raw_url

    path_part, has_hash, fragment = url.partition("#")
    frag_suffix = f"#{fragment}" if has_hash else ""
    if not path_part:
        return raw_url

    # For files authored natively inside docs/portal/, keep clean staged relative links intact
    # (supporting both Markdown source-dir paths and MkDocs directory-URL paths such as
    # ../laws/ from leaderboard.md -> /leaderboard/index.html).
    if src_repo_rel.startswith("docs/portal/"):
        base_dirs = [posixpath.dirname(dst_docs_rel)]
        if dst_docs_rel.endswith(".md") and not (
            dst_docs_rel == "index.md" or dst_docs_rel.endswith("/index.md")
        ):
            base_dirs.append(dst_docs_rel.removesuffix(".md"))
        for base_dir in base_dirs:
            cand = posixpath.normpath(posixpath.join(base_dir, path_part))
            if (
                cand in staged_paths
                or f"{cand.rstrip('/')}/index.md" in staged_paths
                or f"{cand.rstrip('/')}.md" in staged_paths
            ):
                return raw_url

    src_dir = posixpath.dirname(src_repo_rel)
    resolved_repo = re.sub(
        r"^(?:\./)+", "", posixpath.normpath(posixpath.join(src_dir, path_part))
    ).lstrip("/")

    if resolved_repo.startswith("../"):
        return raw_url

    # GitHub source-line anchors (#L60-L77) only resolve on GitHub's blob viewer.
    if fragment and LINE_ANCHOR_RE.fullmatch(fragment):
        return f"https://github.com/{repo_slug}/blob/{git_ref}/{resolved_repo}{frag_suffix}"

    # Direct match to a staged file (Markdown, SVG, PNG, etc.)
    staged_target = repo_to_staged.get(resolved_repo)
    if staged_target is None:
        staged_target = repo_to_staged.get(posixpath.join(resolved_repo, "README.md"))

    if staged_target is not None:
        dst_dir = posixpath.dirname(dst_docs_rel) or "."
        rel = posixpath.relpath(staged_target, start=dst_dir)
        return f"{rel}{frag_suffix}"

    # Fallback: rewrite to canonical GitHub blob/tree/raw URL so MkDocs --strict has zero broken links
    fs_path = root / resolved_repo
    if is_image and fs_path.is_file():
        return f"https://raw.githubusercontent.com/{repo_slug}/{git_ref}/{resolved_repo}"
    if fs_path.is_dir():
        return f"https://github.com/{repo_slug}/tree/{git_ref}/{resolved_repo}{frag_suffix}"
    return f"https://github.com/{repo_slug}/blob/{git_ref}/{resolved_repo}{frag_suffix}"


def _rewrite_line(
    line: str,
    *,
    src_repo_rel: str,
    dst_docs_rel: str,
    repo_to_staged: dict[str, str],
    staged_paths: set[str],
    root: Path,
    git_ref: str,
    repo_slug: str,
) -> str:
    m_ref = REF_DEF_RE.match(line)
    if m_ref:
        prefix, raw_target, tail = m_ref.group(1), m_ref.group(2), m_ref.group(3)
        url_part, title_suffix = _split_md_target(raw_target)
        new_url = _resolve_target(
            url_part,
            is_image=False,
            src_repo_rel=src_repo_rel,
            dst_docs_rel=dst_docs_rel,
            repo_to_staged=repo_to_staged,
            staged_paths=staged_paths,
            root=root,
            git_ref=git_ref,
            repo_slug=repo_slug,
        )
        return f"{prefix}{new_url}{title_suffix}{tail}"

    def _repl(match: re.Match) -> str:
        if match.group("inline_code") is not None:
            return match.group("inline_code")
        if match.group("anchor_name") is not None:
            return f'<a id="{match.group("anchor_val")}">'
        if match.group("md_label") is not None:
            label = match.group("md_label")
            raw_inner = match.group("md_target")
            is_image = label.startswith("!")
            url_part, title_suffix = _split_md_target(raw_inner)
            new_url = _resolve_target(
                url_part,
                is_image=is_image,
                src_repo_rel=src_repo_rel,
                dst_docs_rel=dst_docs_rel,
                repo_to_staged=repo_to_staged,
                staged_paths=staged_paths,
                root=root,
                git_ref=git_ref,
                repo_slug=repo_slug,
            )
            return f"{label}({new_url}{title_suffix})"
        if match.group("html_prefix") is not None:
            prefix = match.group("html_prefix")
            quote = match.group("html_q")
            target = match.group("html_target")
            is_image = prefix.lower().startswith("<img")
            new_url = _resolve_target(
                target,
                is_image=is_image,
                src_repo_rel=src_repo_rel,
                dst_docs_rel=dst_docs_rel,
                repo_to_staged=repo_to_staged,
                staged_paths=staged_paths,
                root=root,
                git_ref=git_ref,
                repo_slug=repo_slug,
            )
            return f"{prefix}{quote}{new_url}{quote}"
        return match.group(0)

    return LINE_TOKEN_RE.sub(_repl, line)


def _convert_gh_backtick_math_line(line: str) -> str:
    """Convert GitHub inline math $`...`$ to standard $...$ for MkDocs arithmatex.

    Preserves regular inline code spans (including literal ``$`x`$`` inside double backticks).
    """
    if "$`" not in line:
        return line
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "`":
            j = i
            while j < n and line[j] == "`":
                j += 1
            ticks = j - i
            k = j
            found = -1
            while k < n:
                if line[k] == "`":
                    m = k
                    while m < n and line[m] == "`":
                        m += 1
                    if m - k == ticks:
                        found = m
                        break
                    k = m
                else:
                    k += 1
            if found != -1:
                # Check whether this backtick span is GitHub $`...`$ inline math
                # (single backtick pair wrapped by unescaped single '$' on both sides).
                bs = 0
                idx_bs = i - 2
                while idx_bs >= 0 and line[idx_bs] == "\\":
                    bs += 1
                    idx_bs -= 1
                is_gh_math = (
                    ticks == 1
                    and i >= 1
                    and line[i - 1] == "$"
                    and not (i >= 2 and line[i - 2] == "$")
                    and (bs % 2 == 0)
                    and found < n
                    and line[found] == "$"
                    and not (found + 1 < n and line[found + 1] == "$")
                )
                if is_gh_math:
                    inner = line[j : found - ticks]
                    # Keep the already-appended leading '$', append inner + trailing '$'
                    out.append(inner)
                    out.append("$")
                    i = found + 1
                    continue
                out.append(line[i:found])
                i = found
                continue
            out.append(line[i:j])
            i = j
            continue
        out.append(line[i])
        i += 1
    return "".join(out)


def convert_github_backtick_math(text: str) -> str:
    """Convert GitHub's $`...`$ inline math spans to $...$ outside fenced code blocks."""
    if "$`" not in text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in lines:
        m_fence = FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(2)
            if not in_fence:
                if not (marker[0] == "`" and "`" in m_fence.group(3)):
                    in_fence = True
                    fence_char = marker[0]
                    fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and not m_fence.group(3).strip():
                in_fence = False
            out.append(line)
            continue
        out.append(line if in_fence else _convert_gh_backtick_math_line(line))
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + suffix


def transform_markdown(
    text: str,
    *,
    src_repo_rel: str,
    dst_docs_rel: str,
    repo_to_staged: dict[str, str],
    staged_paths: set[str],
    root: Path,
    git_ref: str = DEFAULT_GIT_REF,
    repo_slug: str = REPO_SLUG,
) -> str:
    """Apply GitHub Alert conversion, $`...`$ -> $...$, <a name> -> <a id>, and smart relative link rewriting."""
    converted = convert_github_backtick_math(convert_github_alerts(text))
    lines = converted.splitlines()
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        m_fence = FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(2)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and not m_fence.group(3).strip():
                in_fence = False
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        out.append(
            _rewrite_line(
                line,
                src_repo_rel=src_repo_rel,
                dst_docs_rel=dst_docs_rel,
                repo_to_staged=repo_to_staged,
                staged_paths=staged_paths,
                root=root,
                git_ref=git_ref,
                repo_slug=repo_slug,
            )
        )

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + suffix


# ---------------------------------------------------------------------------
# Generated Fallback / Index Pages
# ---------------------------------------------------------------------------


def _extract_h1(path: Path, fallback: str) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                title = re.sub(r"`([^`]+)`", r"\1", title)
                return title
    except OSError:
        pass
    return fallback


def _render_laws_hub_fallback(laws_reg: dict, audit_files: list[tuple[str, str]]) -> str:
    epigraph = laws_reg.get("epigraph") or FALLBACK_LAWS_REGISTRY["epigraph"]
    note = laws_reg.get("epistemic_note") or FALLBACK_LAWS_REGISTRY["epistemic_note"]
    lines = [
        "# Explore by Laws",
        "",
        f"> *\"{epigraph}\"*",
        "",
        '!!! info "Provisional, Evolving Scientific Framework"',
        f"    {note}",
        "",
        "The canonical theoretical formalization lives in **[The Four Laws of Adversarial Generative Learning](four-laws.md)**, supported by ten adversarial mathematical audit tracks (`P0`–`P10`).",
        "",
        "## Current Law Taxonomy",
        "",
        "```mermaid",
        "flowchart LR",
        '    L1["Law 1: Representational Geometry<br/>(Prior & Pushforward Topology)"] --> L2["Law 2: Potential Geometry<br/>(Bounded Informative Critic Forces)"]',
        '    L2 --> L3["Law 3: Transport Identifiability<br/>(3A Ray-Length & 3B Conditional Fibers)"]',
        '    L3 --> L4["Law 4: Functional & Game Dynamics<br/>(4.1-4.5 Conditioning, Trust & Contraction)"]',
        '    L4 --> EO["Energy-Optimal Learning<br/>(min Joules-to-PASS)"]',
        "```",
        "",
        "At equilibrium, energy-optimal generative learning balances generator Jacobian conditioning $\\lVert J_G \\rVert_\\infty$, spatial critic slope regularity $\\lVert \\nabla_x D_\\psi(x) \\rVert_2 \\le \\kappa$, transport identifiability across modes and strata, and coupled minimax contraction under finite hardware energy:",
        "",
        "$$",
        "\\mathcal{E}_{\\text{pass}} = \\sum_{t=1}^{T_{\\text{pass}}} E_{\\text{step}}(t), \\qquad \\min_{\\mathcal{A}} \\; \\mathcal{E}_{\\text{pass}}(\\mathcal{A}) \\quad \\text{subject to faithful measure recovery } P_G \\approx P_Y.",
        "$$",
        "",
        "| Law | Domain Level | Fundamental Question | Formalization |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for law in laws_reg.get("laws", []):
        anchor = law.get("doc_anchor") or ""
        link = f"[Read {law.get('short_label', law['id'])}](four-laws.md#{anchor})" if anchor else "[Read](four-laws.md)"
        lines.append(
            f"| **{law.get('short_label', law['id'])}** | {law.get('title', '')} | {law.get('question', '')} | {link} |"
        )
        for sub in law.get("sub_laws", []) or []:
            sub_anchor = sub.get("doc_anchor") or anchor
            sub_link = f"[Section](four-laws.md#{sub_anchor})" if sub_anchor else "[Section](four-laws.md)"
            lines.append(
                f"| ↳ `{sub.get('short_label', sub['id'])}` | {sub.get('title', '')} | *Sub-law of {law.get('short_label', law['id'])}* | {sub_link} |"
            )
    if audit_files:
        lines += [
            "",
            "## Mathematical Audits (`P0`–`P10`)",
            "",
            "Browse the [Audit Overview](audit/index.md) or jump directly to any track:",
            "",
        ]
        for title, rel in audit_files:
            if rel == "laws/audit/index.md":
                continue
            sub_rel = posixpath.relpath(rel, start="laws")
            lines.append(f"- [{title}]({sub_rel})")
    lines.append("")
    return "\n".join(lines)


def _render_families_index(
    active_entries: list[tuple[str, str, dict | None]],
    historical_entries: list[tuple[str, str]],
) -> str:
    lines = [
        "# Recipe Families",
        "",
        "Each family report records the selected recipe's mathematical formulation, update loop, law mechanisms, training details, and full per-task Forge receipts.",
        "",
        "## Active Selected Families",
        "",
        "| Family | Candidate | Full-Suite PASS | Full-Suite FAIL | View Total | Status |",
        "| :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for title, rel, meta in active_entries:
        sub_rel = posixpath.relpath(rel, start="families")
        if meta:
            fs = meta.get("full_suite") or {}
            tot = meta.get("total_view_requirements") or {}
            status = "Stale" if meta.get("stale") else "Current"
            lines.append(
                f"| **[{title}]({sub_rel})** | `{meta.get('candidate_id') or '—'}` | "
                f"{fs.get('pass', 0)}/{fs.get('total', 0)} | {fs.get('fail', 0)} | "
                f"{tot.get('pass', 0)}/{tot.get('total', 0)} | {status} |"
            )
        else:
            lines.append(f"| **[{title}]({sub_rel})** | — | — | — | — | Active |")

    if historical_entries:
        lines += [
            "",
            "## Historical & Ablation Families",
            "",
            "Archived configurations and structural ablations preserved under their exact original evidence identities:",
            "",
        ]
        for title, rel in historical_entries:
            sub_rel = posixpath.relpath(rel, start="families")
            lines.append(f"- [{title}]({sub_rel})")
    lines.append("")
    return "\n".join(lines)


def _render_experiments_fallback(leaderboard_payload: dict, root: Path) -> str:
    task_defs = leaderboard_payload.get("task_definitions") or []
    problems_path = root / "reports/forge/problems.md"
    problem_anchors: set[str] = set()
    if problems_path.is_file():
        problem_anchors = set(
            re.findall(r'<a\s+(?:name|id)=["\']([^"\']+)["\']', problems_path.read_text(encoding="utf-8", errors="replace"))
        )
    lines = [
        "# Forge Experiment Atlas",
        "",
        "Every Forge experiment executes through the public `particlegan` API under deterministic initialization, protocol seed `0`, and explicit numerical pass/fail gates.",
        "",
        "See also the [Unresolved Training Problems](../studies/problems.md) catalogue and the [Interactive Leaderboard](../leaderboard.md).",
        "",
        "| Task | Tier | Role | Full Suite | Summary |",
        "| :--- | :---: | :--- | :---: | :--- |",
    ]
    for td in sorted(task_defs, key=lambda d: (d.get("tier", 99), d.get("task_id", ""))):
        tid = td.get("task_id", "")
        tier = td.get("tier", 0)
        role = td.get("role", "required")
        in_fs = "Yes" if td.get("in_full_suite") else "—"
        summary = (td.get("summary") or "Declared Forge benchmark task.").replace("|", "\\|")
        task_cell = f"[`{tid}`](../studies/problems.md#{tid})" if tid in problem_anchors else f"`{tid}`"
        lines.append(f"| {task_cell} | T{tier} | `{role}` | {in_fs} | {summary} |")
    lines.append("")
    return "\n".join(lines)


def _render_studies_index(
    study_entries: list[tuple[str, str]],
    axiom_entries: list[tuple[str, str]] | None = None,
) -> str:
    lines = [
        "# Studies & Unresolved Problems",
        "",
        "- **[Unresolved Training Problems](problems.md)** — Outstanding Forge tasks, held failures, diagnoses, and cross-family recorded outcomes.",
        "",
    ]
    if axiom_entries:
        lines += [
            "## Axiom Family (`axiom-v4`)",
            "",
        ]
        for title, rel in axiom_entries:
            sub_rel = posixpath.relpath(rel, start="studies")
            lines.append(f"- [{title}]({sub_rel})")
        lines.append("")
    lines += [
        "## Research Studies",
        "",
    ]
    for title, rel in study_entries:
        sub_rel = posixpath.relpath(rel, start="studies")
        lines.append(f"- [{title}]({sub_rel})")
    lines.append("")
    return "\n".join(lines)


def _render_reference_index(ref_entries: list[tuple[str, str]]) -> str:
    lines = [
        "# Reference Documentation",
        "",
        "Core API specifications, initialization contracts, input-domain walkthroughs, and Forge experimentation methodology.",
        "",
    ]
    for title, rel in ref_entries:
        sub_rel = posixpath.relpath(rel, start="reference")
        lines.append(f"- [{title}]({sub_rel})")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic YAML Emitter (zero third-party deps, preserves !!python/name)
# ---------------------------------------------------------------------------


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _emit_nav_yaml(items: list, indent: int = 2) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for item in items:
        if isinstance(item, str):
            lines.append(f"{prefix}- {_yaml_quote(item)}")
        elif isinstance(item, dict):
            for key, val in item.items():
                if isinstance(val, str):
                    lines.append(f"{prefix}- {_yaml_quote(key)}: {_yaml_quote(val)}")
                elif isinstance(val, list):
                    lines.append(f"{prefix}- {_yaml_quote(key)}:")
                    lines.extend(_emit_nav_yaml(val, indent=indent + 4))
    return lines


def _render_resolved_mkdocs_yml(
    base_mkdocs_yml: Path,
    *,
    docs_dir: Path,
    site_dir: Path,
    nav: list,
) -> str:
    raw = base_mkdocs_yml.read_text(encoding="utf-8")
    # Strip top-level docs_dir, site_dir, and nav sections cleanly while preserving !!python/name tags
    out_lines: list[str] = []
    skip_nav = False
    for line in raw.splitlines():
        if skip_nav:
            if line and not line[0].isspace():
                skip_nav = False
            else:
                continue
        if re.match(r"^(docs_dir|site_dir)\s*:", line):
            continue
        if re.match(r"^nav\s*:", line):
            skip_nav = True
            continue
        out_lines.append(line)

    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    out_lines += [
        "",
        f"docs_dir: {_yaml_quote(str(docs_dir.resolve()))}",
        f"site_dir: {_yaml_quote(str(site_dir.resolve()))}",
        "",
        "nav:",
        *_emit_nav_yaml(nav, indent=2),
        "",
    ]
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Main Staging Orchestrator
# ---------------------------------------------------------------------------


def stage_docs_site(
    root: Path,
    *,
    build_dir: Path,
    site_dir: Path,
    git_ref: str = DEFAULT_GIT_REF,
    prefer_stdlib_leaderboard: bool = False,
) -> dict:
    """Stage all portal sources into ``build_dir`` without mutating the repository."""
    root = Path(root).resolve()
    build_dir = Path(build_dir).resolve()
    site_dir = Path(site_dir).resolve()
    docs_dir = build_dir / "docs"

    if build_dir.exists():
        shutil.rmtree(build_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build leaderboard.json snapshot first (offline, deterministic)
    leaderboard_payload = build_portal_leaderboard_payload(
        root, prefer_stdlib=prefer_stdlib_leaderboard
    )
    lb_out = docs_dir / "assets/data/leaderboard.json"
    lb_out.parent.mkdir(parents=True, exist_ok=True)
    lb_out.write_text(
        json.dumps(leaderboard_payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # 2. Discover all files to stage: map repo_rel -> staged_rel
    markdown_map: dict[str, str] = {}
    binary_map: dict[str, str] = {}

    def _add_md(repo_rel: str, staged_rel: str) -> None:
        if (root / repo_rel).is_file():
            markdown_map[repo_rel] = staged_rel

    def _add_bin(repo_rel: str, staged_rel: str) -> None:
        if (root / repo_rel).is_file():
            binary_map[repo_rel] = staged_rel

    # Portal sources + hero
    _add_md("docs/portal/index.md", "index.md")
    _add_md("docs/portal/leaderboard.md", "leaderboard.md")
    portal_assets = root / "docs/portal/assets"
    if portal_assets.is_dir():
        for asset in sorted(portal_assets.rglob("*")):
            if asset.is_file():
                rel = asset.relative_to(root).as_posix()
                dst = asset.relative_to(root / "docs/portal").as_posix()
                binary_map[rel] = dst
    _add_bin("docs/media/readme/particlegan-hero.png", "assets/images/particlegan-hero.png")

    # Canonical Leaderboard Markdown
    _add_md("reports/forge/technique-inventory.md", "technique-inventory.md")

    # Laws Hub + Four Laws + Four Laws Audit
    has_laws_hub = (root / "reports/forge/LAWS_HUB.md").is_file()
    if has_laws_hub:
        _add_md("reports/forge/LAWS_HUB.md", "laws/index.md")
    _add_md("docs/four-laws.md", "laws/four-laws.md")

    audit_dir = root / "reports/four-laws-audit"
    audit_entries: list[tuple[str, str]] = []
    if audit_dir.is_dir():
        audit_mds = sorted(audit_dir.glob("*.md"))
        for md in audit_mds:
            rel = md.relative_to(root).as_posix()
            dst = "laws/audit/index.md" if md.name == "README.md" else f"laws/audit/{md.name}"
            markdown_map[rel] = dst

        def _audit_sort_key(path: Path) -> tuple[int, int, str]:
            if path.name == "README.md":
                return (0, 0, path.name)
            m = AUDIT_P_RE.match(path.name)
            if m:
                return (2, int(m.group(1)), path.name)
            return (1, 0, path.name)

        for md in sorted(audit_mds, key=_audit_sort_key):
            dst = "laws/audit/index.md" if md.name == "README.md" else f"laws/audit/{md.name}"
            audit_entries.append((_extract_h1(md, md.stem), dst))

    # Families + family assets
    fam_dir = root / "reports/forge/families"
    fam_assets_dir = fam_dir / "assets"
    if fam_assets_dir.is_dir():
        for asset in sorted(fam_assets_dir.rglob("*")):
            if asset.is_file():
                rel = asset.relative_to(root).as_posix()
                dst = f"families/assets/{asset.relative_to(fam_assets_dir).as_posix()}"
                binary_map[rel] = dst

    family_meta_by_id = {f["id"]: f for f in leaderboard_payload.get("families", []) if isinstance(f, dict)}
    priority_active_ids = [
        "unified",
        "axiom",
        "minimal",
        "rhbi",
        "empf",
        "bispf",
        "bcap-pure",
        "bcap-dualnorm",
        "atlas",
        "e22",
        "release07-gan-v3",
        "k3p",
        "ka2",
        "r1r2",
    ]
    selected_id_set = set(family_meta_by_id) | {"minimal", "bcap-dualnorm"}
    active_family_entries: list[tuple[str, str, dict | None]] = []
    historical_family_entries: list[tuple[str, str]] = []

    if fam_dir.is_dir():
        fam_files = {p.stem: p for p in sorted(fam_dir.glob("*.md"))}
        for stem, p in fam_files.items():
            markdown_map[p.relative_to(root).as_posix()] = f"families/{p.name}"

        ordered_active = [fid for fid in priority_active_ids if fid in fam_files]
        for fid in sorted(selected_id_set):
            if fid in fam_files and fid not in ordered_active:
                ordered_active.append(fid)

        for fid in ordered_active:
            p = fam_files[fid]
            meta = family_meta_by_id.get(fid)
            title = (meta.get("label") if meta else None) or _extract_h1(p, fid)
            active_family_entries.append((title, f"families/{p.name}", meta))

        for fid in sorted(fam_files):
            if fid in ordered_active:
                continue
            p = fam_files[fid]
            historical_family_entries.append((_extract_h1(p, fid), f"families/{p.name}"))

    # Experiments (real per-experiment pages when merged, graceful fallback otherwise)
    exp_dir = root / "reports/forge/experiments"
    has_exp_dir = exp_dir.is_dir() and (exp_dir / "README.md").is_file()
    experiment_entries: list[tuple[str, str]] = []
    if has_exp_dir:
        for md in sorted(exp_dir.glob("*.md")):
            rel = md.relative_to(root).as_posix()
            dst = "experiments/index.md" if md.name == "README.md" else f"experiments/{md.name}"
            markdown_map[rel] = dst
            if md.name != "README.md":
                experiment_entries.append((_extract_h1(md, md.stem), dst))
        exp_assets = exp_dir / "assets"
        if exp_assets.is_dir():
            for asset in sorted(exp_assets.rglob("*.svg")):
                if asset.is_file():
                    rel = asset.relative_to(root).as_posix()
                    dst = f"experiments/assets/{asset.relative_to(exp_assets).as_posix()}"
                    binary_map[rel] = dst

    # Studies, Axiom Family reports & Problems
    _add_md("reports/forge/problems.md", "studies/problems.md")
    axiom_dir = root / "reports/axiom"
    axiom_entries: list[tuple[str, str]] = []
    if axiom_dir.is_dir():
        axiom_order = ("axiom-study.md", "design-spec.md", "four-laws-benchmark.md", "gap-analysis.md")
        axiom_mds = sorted(
            axiom_dir.glob("*.md"),
            key=lambda p: (axiom_order.index(p.name) if p.name in axiom_order else 99, p.name),
        )
        for md in axiom_mds:
            rel = md.relative_to(root).as_posix()
            dst = f"studies/axiom/{md.name}"
            markdown_map[rel] = dst
            axiom_entries.append((_extract_h1(md, md.stem), dst))
    studies_dir = root / "reports/forge/studies"
    study_entries: list[tuple[str, str]] = []
    if studies_dir.is_dir():
        for md in sorted(studies_dir.glob("*.md")):
            rel = md.relative_to(root).as_posix()
            dst = f"studies/{md.name}"
            markdown_map[rel] = dst
            study_entries.append((_extract_h1(md, md.stem), dst))

    # Reference docs
    reference_specs = [
        ("docs/api.md", "reference/api.md", "API Reference"),
        ("docs/initialization.md", "reference/initialization.md", "Repeatable Initialization"),
        ("docs/bispf-input-domains.md", "reference/bispf-input-domains.md", "BISPF Input Domains"),
        ("docs/e22.md", "reference/e22.md", "E22 Policy"),
        ("docs/atlas.md", "reference/atlas.md", "Atlas Guide"),
        ("RESEARCH_GLOSSARY.md", "reference/glossary.md", "Research Glossary"),
        ("EXPERIMENTATION.md", "reference/experimentation.md", "Experimentation Guide"),
    ]
    ref_entries: list[tuple[str, str]] = []
    for src_rel, dst_rel, label in reference_specs:
        if (root / src_rel).is_file():
            markdown_map[src_rel] = dst_rel
            ref_entries.append((label, dst_rel))

    # Full lookup map for smart link rewriting
    repo_to_staged: dict[str, str] = {**markdown_map, **binary_map}
    staged_paths: set[str] = set(repo_to_staged.values()) | {
        "assets/data/leaderboard.json",
        "laws/index.md",
        "families/index.md",
        "experiments/index.md",
        "studies/index.md",
        "reference/index.md",
    }

    # 3. Copy binary/static assets
    for src_rel, dst_rel in sorted(binary_map.items()):
        dst = docs_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / src_rel, dst)

    # 4. Transform and write staged Markdown files
    for src_rel, dst_rel in sorted(markdown_map.items()):
        raw_text = (root / src_rel).read_text(encoding="utf-8", errors="replace")
        staged_text = transform_markdown(
            raw_text,
            src_repo_rel=src_rel,
            dst_docs_rel=dst_rel,
            repo_to_staged=repo_to_staged,
            staged_paths=staged_paths,
            root=root,
            git_ref=git_ref,
            repo_slug=REPO_SLUG,
        )
        dst = docs_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(staged_text, encoding="utf-8")

    # 5. Write generated index/fallback pages where needed
    if not has_laws_hub:
        laws_reg = load_laws_registry(root)
        (docs_dir / "laws").mkdir(parents=True, exist_ok=True)
        (docs_dir / "laws/index.md").write_text(
            _render_laws_hub_fallback(laws_reg, audit_entries), encoding="utf-8"
        )

    (docs_dir / "families").mkdir(parents=True, exist_ok=True)
    (docs_dir / "families/index.md").write_text(
        _render_families_index(active_family_entries, historical_family_entries), encoding="utf-8"
    )

    if not has_exp_dir:
        (docs_dir / "experiments").mkdir(parents=True, exist_ok=True)
        (docs_dir / "experiments/index.md").write_text(
            _render_experiments_fallback(leaderboard_payload, root), encoding="utf-8"
        )

    (docs_dir / "studies").mkdir(parents=True, exist_ok=True)
    (docs_dir / "studies/index.md").write_text(
        _render_studies_index(study_entries, axiom_entries), encoding="utf-8"
    )

    (docs_dir / "reference").mkdir(parents=True, exist_ok=True)
    (docs_dir / "reference/index.md").write_text(
        _render_reference_index(ref_entries), encoding="utf-8"
    )

    # 6. Build dynamic navigation tree
    laws_nav: list = ["laws/index.md"]
    if (docs_dir / "laws/four-laws.md").is_file():
        laws_nav.append({"Canonical Laws Formalization": "laws/four-laws.md"})
    if audit_entries:
        audit_subnav: list = []
        if (docs_dir / "laws/audit/index.md").is_file():
            audit_subnav.append("laws/audit/index.md")
        for title, rel in audit_entries:
            if rel != "laws/audit/index.md":
                audit_subnav.append({title: rel})
        laws_nav.append({"Mathematical Audits (P0–P10)": audit_subnav})

    families_nav: list = ["families/index.md"]
    if active_family_entries:
        families_nav.append(
            {
                "Active Selected Families": [
                    {title: rel} for title, rel, _ in active_family_entries
                ]
            }
        )
    if historical_family_entries:
        families_nav.append(
            {
                "Historical & Ablation Families": [
                    {title: rel} for title, rel in historical_family_entries
                ]
            }
        )

    experiments_nav: list = ["experiments/index.md"]
    if experiment_entries:
        experiments_nav.append(
            {"All Experiments": [{title: rel} for title, rel in experiment_entries]}
        )

    studies_nav: list = ["studies/index.md"]
    if (docs_dir / "studies/problems.md").is_file():
        studies_nav.append({"Unresolved Training Problems": "studies/problems.md"})
    if axiom_entries:
        studies_nav.append({"Axiom Family (v4)": [{title: rel} for title, rel in axiom_entries]})
    if study_entries:
        studies_nav.append({"Research Studies": [{title: rel} for title, rel in study_entries]})

    reference_nav: list = ["reference/index.md"] + [{label: rel} for label, rel in ref_entries]

    leaderboard_nav: list = [{"Interactive Leaderboard & Pareto": "leaderboard.md"}]
    if (docs_dir / "technique-inventory.md").is_file():
        leaderboard_nav.append({"Canonical Forge Leaderboard": "technique-inventory.md"})

    nav = [
        {"Home": "index.md"},
        {"Leaderboard": leaderboard_nav},
        {"Laws": laws_nav},
        {"Families": families_nav},
        {"Experiments": experiments_nav},
        {"Studies": studies_nav},
        {"Reference": reference_nav},
    ]

    # 7. Write resolved <build-dir>/mkdocs.yml
    resolved_yml = build_dir / "mkdocs.yml"
    resolved_yml.write_text(
        _render_resolved_mkdocs_yml(
            root / "mkdocs.yml", docs_dir=docs_dir, site_dir=site_dir, nav=nav
        ),
        encoding="utf-8",
    )

    return {
        "build_dir": str(build_dir),
        "docs_dir": str(docs_dir),
        "site_dir": str(site_dir),
        "mkdocs_yml": str(resolved_yml),
        "staged_markdown_count": len(markdown_map),
        "staged_asset_count": len(binary_map),
        "active_families": len(active_family_entries),
        "historical_families": len(historical_family_entries),
        "experiments_count": len(experiment_entries),
        "studies_count": len(study_entries),
        "leaderboard_json": str(lb_out),
    }


def _resolve_mkdocs_command(mkdocs_bin: str | None) -> list[str]:
    if mkdocs_bin:
        return [mkdocs_bin]
    found = shutil.which("mkdocs")
    if found:
        return [found]
    return [sys.executable, "-m", "mkdocs"]


def main(argv: list[str] | None = None) -> int:
    default_tmp = Path(tempfile.gettempdir())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root directory")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_tmp / "particlegan-docs-build",
        help="Temporary staging directory for assembled docs/ and resolved mkdocs.yml",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=default_tmp / "particlegan-docs-site",
        help="Output directory for compiled MkDocs HTML site",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --strict to mkdocs build (default: True)",
    )
    parser.add_argument("--serve", action="store_true", help="Launch mkdocs serve after staging")
    parser.add_argument(
        "--skip-mkdocs",
        action="store_true",
        help="Stage docs, leaderboard.json, and mkdocs.yml without invoking mkdocs",
    )
    parser.add_argument("--mkdocs-bin", type=str, default=None, help="Explicit mkdocs executable path")
    parser.add_argument("--git-ref", type=str, default=DEFAULT_GIT_REF, help="Git ref for external repo blob links")
    parser.add_argument(
        "--stdlib-leaderboard",
        action="store_true",
        help="Force zero-torch stdlib leaderboard snapshot builder",
    )
    args = parser.parse_args(argv)

    summary = stage_docs_site(
        args.root,
        build_dir=args.build_dir,
        site_dir=args.site_dir,
        git_ref=args.git_ref,
        prefer_stdlib_leaderboard=args.stdlib_leaderboard,
    )
    print(
        f"Staged ParticleGAN Research Portal: "
        f"{summary['staged_markdown_count']} markdown pages, "
        f"{summary['staged_asset_count']} assets, "
        f"{summary['active_families']} active + {summary['historical_families']} historical families, "
        f"{summary['studies_count']} studies -> {summary['build_dir']}"
    )

    if args.skip_mkdocs:
        return 0

    cmd = _resolve_mkdocs_command(args.mkdocs_bin)
    if args.serve:
        subprocess.run([*cmd, "serve", "-f", summary["mkdocs_yml"]], check=True)
    else:
        build_cmd = [*cmd, "build", "-f", summary["mkdocs_yml"], "--clean"]
        if args.strict:
            build_cmd.append("--strict")
        subprocess.run(build_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
