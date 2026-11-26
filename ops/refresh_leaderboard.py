#!/usr/bin/env python3
"""One-command orchestrator for refreshing ParticleGAN leaderboards everywhere.

Coordinates deterministic updates across all four publication surfaces:
  1. GitHub repository (`reports/forge/technique-inventory.md`, family pages,
     experiment docs, compiled memory, and the open `develop -> master` PR).
  2. Hugging Face Dataset (`HyperGAN/particlegan-forge-evidence`:
     `data/leaderboard.json`, Parquet tables, and actual-training GIFs).
  3. Hugging Face Static Space (`HyperGAN/particlegan-leaderboard`:
     `ops/hf_space/` static UI + Pareto frontier viewer).
  4. GitHub Pages Research Portal (`https://hypergan.github.io/ParticleGAN/`,
     validated locally via `scripts/build_docs_site.py` and deployed on merge
     to `develop` by `.github/workflows/pages.yml`).

Scientific invariants enforced by this tool:
  - Never bypasses Forge source-manifest, certificate, or compatibility checks.
  - Never fabricates calibration acceptance (`default_adoption` remains `false`).
  - Never promotes receipts recorded on a dirty working tree.
  - Never tags busy-GPU runs as uncontended energy benchmarks.
  - Never commits bulk logs, checkpoints, or binary GIFs into Git.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HF_DATASET = "HyperGAN/particlegan-forge-evidence"
DEFAULT_HF_SPACE = "HyperGAN/particlegan-leaderboard"
DEFAULT_PAGES_URL = "https://hypergan.github.io/ParticleGAN"
DEFAULT_GITHUB_REPO = "HyperGAN/ParticleGAN"
DEFAULT_COMPARISON_POLICY = "configs/forge/view-history/discriminator_stability-v9.json"
DEFAULT_PUBLICATION_RECEIPT = "reports/forge/publications/inventory-latest.json"
DEFAULT_MEDIA_STAGING = "runs/forge/leaderboard_media_staging"
INVENTORY_URL = (
    "https://github.com/HyperGAN/ParticleGAN/blob/develop/reports/forge/technique-inventory.md"
)

_VERSION_RE = re.compile(r"^(?P<prefix>.+)-v(?P<num>\d+)$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _parse_versioned_candidate(candidate_id: str) -> tuple[str, int] | None:
    m = _VERSION_RE.match(candidate_id)
    if not m:
        return None
    return m.group("prefix"), int(m.group("num"))


def _load_occam_task_ids(root: Path) -> list[str]:
    view_path = root / "configs/forge/views/r4-occam-ablation-v1.json"
    if not view_path.is_file():
        return []
    doc = _read_json(view_path)
    return sorted(
        {
            str(a["task"])
            for a in doc.get("assignments", [])
            if isinstance(a, dict) and isinstance(a.get("task"), str)
        }
    )


def _find_complete_occam_cohorts(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Locate committed 16/16 complete `r4-occam-ablation-v1` receipt cohorts by candidate."""
    occam_tasks = set(_load_occam_task_ids(root))
    if not occam_tasks:
        return {}
    receipts_dir = root / "reports/forge/technique-receipts"
    if not receipts_dir.is_dir():
        return {}

    grouped: dict[tuple[str, str, str, str], dict[str, tuple[str, str, bool]]] = {}
    for rec_path in sorted(receipts_dir.glob("*.json")):
        try:
            doc = _read_json(rec_path)
        except Exception:
            continue
        cid = str(doc.get("candidate_id") or "")
        if not cid:
            continue
        prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
        src_commit = str(prov.get("source_origin_commit") or "")
        src_digest = str(prov.get("source_digest") or "")
        attempt_id = str(doc.get("attempt_id") or rec_path.stem)
        for tr in doc.get("task_results") or []:
            if not isinstance(tr, dict):
                continue
            tid = str(tr.get("task_id") or "")
            if tid not in occam_tasks:
                continue
            cost = tr.get("cost") if isinstance(tr.get("cost"), dict) else {}
            energy = cost.get("energy") if isinstance(cost.get("energy"), dict) else {}
            gpu = str(energy.get("gpu_name") or "").strip() or "unknown"
            status = str(tr.get("gate_status") or tr.get("status") or "").upper()
            raw_status = str(tr.get("raw_status") or "").upper()
            reason = str(tr.get("reason") or "").lower()
            is_timeout = (
                status in {"TIMEOUT", "INCOMPLETE"}
                or raw_status in {"TIMEOUT", "INCOMPLETE"}
                or "timeout" in reason
            )
            key = (cid, src_commit, src_digest, gpu)
            bucket = grouped.setdefault(key, {})
            prev = bucket.get(tid)
            if prev is None or (prev[1] != "PASS" and status == "PASS"):
                bucket[tid] = (attempt_id, status, is_timeout)

    complete: dict[str, list[dict[str, Any]]] = {}
    for (cid, src_commit, src_digest, gpu), tasks in sorted(grouped.items()):
        if not occam_tasks.issubset(tasks):
            continue
        passed = sum(1 for tid in occam_tasks if tasks[tid][1] == "PASS")
        timeouts = sum(1 for tid in occam_tasks if tasks[tid][2])
        executed = sum(
            1 for tid in occam_tasks if tasks[tid][1] not in {"INCOMPLETE", "TIMEOUT"}
        )
        attempt_ids = sorted({tasks[tid][0] for tid in occam_tasks})
        complete.setdefault(cid, []).append(
            {
                "candidate_id": cid,
                "source_commit": src_commit or None,
                "source_digest": src_digest or None,
                "hardware_gpu": gpu,
                "occam_pass": passed,
                "occam_executed": executed,
                "occam_timeouts": timeouts,
                "occam_receipt_ids": attempt_ids,
            }
        )
    return complete


def _find_occam_readout_for_candidate(root: Path, candidate_id: str) -> dict[str, Any] | None:
    """Find a committed `r4-occam-ablation-v1` readout file for `candidate_id` if present."""
    records_dir = root / "reports/forge/records"
    if not records_dir.is_dir():
        return None
    matches: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("readout-*.json")):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        if doc.get("candidate_id") != candidate_id:
            continue
        if doc.get("view_id") != "r4-occam-ablation-v1":
            continue
        summary = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}
        passed = summary.get("passed")
        total = summary.get("total") or summary.get("executed")
        if isinstance(passed, int) and isinstance(total, int) and total >= 15:
            matches.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "occam_pass": passed,
                    "occam_executed": total,
                    "occam_timeouts": int(summary.get("incomplete", 0) or 0),
                    "source_commit": doc.get("source_origin_commit"),
                }
            )
    return matches[-1] if matches else None


def _collect_study_energy_index(root: Path) -> set[tuple[str, str]]:
    """Return `(candidate_id, hardware_gpu)` pairs covered by uncontended energy studies."""
    covered: set[tuple[str, str]] = set()
    studies_dir = root / "reports/forge/studies"
    if not studies_dir.is_dir():
        return covered
    for path in sorted(studies_dir.glob("*.json")):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("energy_benchmark"), dict):
            continue
        for cid, entry in doc["energy_benchmark"].items():
            if not isinstance(entry, dict) or entry.get("contended") is True:
                continue
            r256 = entry.get("dcgan_r256") if isinstance(entry.get("dcgan_r256"), dict) else {}
            gpu = str(
                r256.get("gpu_name")
                or entry.get("hardware_gpu")
                or doc.get("hardware_gpu")
                or ""
            ).strip()
            covered.add((str(cid), gpu))
    return covered


def _family_for_candidate(root: Path, candidate_id: str) -> str:
    reg_path = root / "configs/forge/trainer-families.json"
    if reg_path.is_file():
        reg = _read_json(reg_path)
        for fam in reg.get("families", []):
            fid = str(fam.get("id") or "")
            if candidate_id in (fam.get("candidates") or []):
                return fid
    parsed = _parse_versioned_candidate(candidate_id)
    if parsed is not None:
        return parsed[0]
    return candidate_id


def _query_gpu_status() -> list[dict[str, Any]]:
    """Query local NVIDIA GPUs via `nvidia-smi`; returns empty list if unavailable."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    out = subprocess.check_output(
        [
            binary,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.PIPE,
    )
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "util_pct": int(float(parts[2])),
                "mem_used_mib": int(float(parts[3])),
                "mem_total_mib": int(float(parts[4])),
            }
        )
    return gpus


# ---------------------------------------------------------------------------
# Subcommand 1: status
# ---------------------------------------------------------------------------


def collect_status(root: Path, *, check_pr: bool = False) -> dict[str, Any]:
    """Audit local repo, selected families, Pareto energy coverage, and HF receipt."""
    root = Path(root).resolve()
    branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    head_sha = _git_output(root, "rev-parse", "HEAD")
    dirty_lines = [
        line for line in _git_output(root, "status", "--porcelain").splitlines() if line.strip()
    ]

    sel_path = root / "configs/forge/selections/family-current-v1.json"
    selected: dict[str, str] = {}
    if sel_path.is_file():
        sel_doc = _read_json(sel_path)
        for entry in sel_doc.get("selections", []):
            selected[str(entry["trainer_family"])] = str(entry["candidate_id"])

    # Scan ideas & receipts for newer versioned candidates beyond pinned family leaders.
    known_candidates: set[str] = set()
    ideas_dir = root / "configs/forge/ideas"
    if ideas_dir.is_dir():
        known_candidates.update(p.stem for p in ideas_dir.glob("*.json"))
    receipts_dir = root / "reports/forge/technique-receipts"
    receipt_candidates: set[str] = set()
    if receipts_dir.is_dir():
        for rec in receipts_dir.glob("*.json"):
            try:
                cid = _read_json(rec).get("candidate_id")
                if isinstance(cid, str) and cid:
                    receipt_candidates.add(cid)
            except Exception:
                continue
    known_candidates.update(receipt_candidates)

    newer_candidates: list[dict[str, Any]] = []
    for fam_id, pinned_cid in sorted(selected.items()):
        parsed = _parse_versioned_candidate(pinned_cid)
        if parsed is None:
            continue
        prefix, pinned_num = parsed
        higher = []
        for cand in sorted(known_candidates):
            cp = _parse_versioned_candidate(cand)
            if cp is not None and cp[0] == prefix and cp[1] > pinned_num:
                higher.append((cp[1], cand))
        if higher:
            higher.sort()
            newer_candidates.append(
                {
                    "family_id": fam_id,
                    "pinned_candidate": pinned_cid,
                    "newer_candidates": [c for _, c in higher],
                    "newer_with_receipts": [c for _, c in higher if c in receipt_candidates],
                }
            )

    # Check which Occam-complete candidates lack an uncontended energy benchmark study.
    occam_cohorts = _find_complete_occam_cohorts(root)
    energy_studies = _collect_study_energy_index(root)
    missing_energy: list[dict[str, Any]] = []
    for cid, cohorts in sorted(occam_cohorts.items()):
        for cohort in cohorts:
            gpu = str(cohort["hardware_gpu"])
            if (cid, gpu) not in energy_studies and not any(
                ec == cid for ec, _ in energy_studies
            ):
                missing_energy.append(
                    {
                        "candidate_id": cid,
                        "hardware_gpu": gpu,
                        "occam_pass": cohort["occam_pass"],
                        "occam_executed": cohort["occam_executed"],
                    }
                )

    # Check HF publication receipt freshness.
    receipt_path = root / DEFAULT_PUBLICATION_RECEIPT
    hf_receipt: dict[str, Any] | None = None
    if receipt_path.is_file():
        rec = _read_json(receipt_path)
        hf_receipt = {
            "path": DEFAULT_PUBLICATION_RECEIPT,
            "published_at": rec.get("published_at"),
            "git_source_commit": rec.get("git_source_commit"),
            "hf_revision": rec.get("hf_revision"),
            "matches_head": rec.get("git_source_commit") == head_sha,
            "energy_rows": ((rec.get("files") or {}).get("data/energy.parquet") or {}).get("rows"),
            "families": ((rec.get("files") or {}).get("data/leaderboard.json") or {}).get("families"),
        }

    # Check staged media waiting for HF upload.
    staging_dir = root / DEFAULT_MEDIA_STAGING
    staged_gifs = sorted(staging_dir.rglob("*.gif")) if staging_dir.is_dir() else []

    status: dict[str, Any] = {
        "checked_at": _utc_now(),
        "git": {
            "branch": branch,
            "head": head_sha,
            "clean": len(dirty_lines) == 0,
            "dirty_count": len(dirty_lines),
        },
        "selected_families": selected,
        "newer_candidates_beyond_pins": newer_candidates,
        "occam_complete_missing_energy_study": missing_energy,
        "hf_publication_receipt": hf_receipt,
        "staged_media_gif_count": len(staged_gifs),
        "recommended_actions": [],
    }

    actions: list[str] = status["recommended_actions"]
    if not status["git"]["clean"]:
        actions.append(
            "Working tree has uncommitted changes; commit before running new promotion benchmarks."
        )
    if missing_energy:
        cids = " ".join(sorted({item["candidate_id"] for item in missing_energy}))
        actions.append(
            f"Run uncontended energy benchmark for Pareto coverage: "
            f"python ops/refresh_leaderboard.py energy --ideas {cids} --output <study-name>.json"
        )
    if newer_candidates:
        summary = ", ".join(
            f"{item['family_id']}: {item['pinned_candidate']} -> {item['newer_candidates'][-1]}"
            for item in newer_candidates
        )
        actions.append(
            f"Newer candidates exist beyond family-current-v1.json ({summary}); "
            "run clean-commit discriminator_stability screening and promote with "
            "`python ops/refresh_leaderboard.py promote`."
        )
    if hf_receipt is None or not hf_receipt.get("matches_head"):
        actions.append(
            "HF publication receipt is behind HEAD; run "
            "`python ops/refresh_leaderboard.py sync --publish` when ready."
        )
    if staged_gifs:
        actions.append(
            f"{len(staged_gifs)} staged GIF(s) in {DEFAULT_MEDIA_STAGING}; "
            "pass `--upload-media` to `sync --publish` if any are newly rendered."
        )

    if check_pr:
        status["develop_to_master_pr"] = _check_release_pr(root)

    return status


# ---------------------------------------------------------------------------
# Subcommand 2: energy
# ---------------------------------------------------------------------------


def build_curated_energy_study(
    root: Path,
    raw_output: dict[str, Any],
    *,
    contended: bool,
    source_commit: str,
    note_prefix: str | None = None,
) -> dict[str, Any]:
    """Attach provenance, family IDs, and committed 16-task Occam receipts to profiler output."""
    root = Path(root).resolve()
    occam_cohorts = _find_complete_occam_cohorts(root)
    enriched_bench: dict[str, Any] = {}
    raw_bench = raw_output.get("energy_benchmark") or {}
    if not isinstance(raw_bench, dict) or not raw_bench:
        raise ValueError("Profiler output is missing a non-empty 'energy_benchmark' mapping")

    for cid, entry in sorted(raw_bench.items()):
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        r256 = item.get("dcgan_r256") if isinstance(item.get("dcgan_r256"), dict) else {}
        gpu_name = str(r256.get("gpu_name") or "").strip()
        item["family_id"] = _family_for_candidate(root, cid)
        item["contended"] = bool(contended)
        item["source_commit"] = source_commit

        # Prefer exact GPU-matched complete 16-task Occam receipt cohort.
        cohorts = occam_cohorts.get(cid) or []
        gpu_matched = [c for c in cohorts if c["hardware_gpu"] == gpu_name]
        chosen = (gpu_matched[-1] if gpu_matched else None) or (cohorts[-1] if cohorts else None)
        if chosen is not None:
            item["occam_view"] = "r4-occam-ablation-v1"
            item["occam_pass"] = chosen["occam_pass"]
            item["occam_executed"] = chosen["occam_executed"]
            item["occam_timeouts"] = chosen["occam_timeouts"]
            item["occam_hardware_gpu"] = chosen["hardware_gpu"]
            item["occam_receipt_ids"] = chosen["occam_receipt_ids"]
            occam_desc = (
                f"matched {chosen['hardware_gpu']} r4-occam-ablation-v1 "
                f"({chosen['occam_pass']}/{chosen['occam_executed']} PASS, "
                f"{chosen['occam_timeouts']} timeouts)"
            )
        else:
            readout = _find_occam_readout_for_candidate(root, cid)
            if readout is not None:
                item["occam_view"] = "r4-occam-ablation-v1"
                item["occam_pass"] = readout["occam_pass"]
                item["occam_executed"] = readout["occam_executed"]
                item["occam_timeouts"] = readout["occam_timeouts"]
                item["official_forge_readout"] = readout["path"]
                occam_desc = (
                    f"matched r4-occam-ablation-v1 {readout['path']} "
                    f"({readout['occam_pass']}/{readout['occam_executed']} PASS)"
                )
            else:
                occam_desc = "no committed 16-task r4-occam-ablation-v1 cohort found yet"

        util_before = r256.get("gpu_util_before")
        cont_label = "Contended (diagnostic)" if contended else "Uncontended"
        prefix = f"{note_prefix}; " if note_prefix else ""
        item["cohort_note"] = (
            f"{prefix}{cont_label} {gpu_name or 'GPU'} (gpu_util_before={util_before}) "
            f"via experiments/law4_optimizer_profiler.py --mode energy-benchmark; {occam_desc}."
        )
        enriched_bench[cid] = item

    return {
        "device": raw_output.get("device"),
        "energy_benchmark": enriched_bench,
        "generated_at": _utc_now(),
        "mode": raw_output.get("mode", "energy-benchmark"),
        "pytorch_version": raw_output.get("pytorch_version"),
        "schema_version": 1,
        "source_commit": source_commit,
    }


def run_energy_benchmark(
    root: Path,
    *,
    ideas: list[str],
    output: Path,
    device: str = "cuda:0",
    allow_busy: bool = False,
    dry_run: bool = False,
    note: str | None = None,
    gpu_query_fn: Callable[[], list[dict[str, Any]]] = _query_gpu_status,
    runner_fn: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run an idle-GPU NVML energy benchmark and write a curated study JSON."""
    root = Path(root).resolve()
    if not ideas:
        raise ValueError("At least one idea ID must be passed to --ideas")

    gpus = gpu_query_fn()
    busy = [g for g in gpus if int(g.get("util_pct", 0)) > 5]
    if busy and not allow_busy:
        desc = ", ".join(f"GPU {g['index']} ({g['name']}): {g['util_pct']}%" for g in busy)
        raise RuntimeError(
            f"Refusing uncontended energy benchmark while GPU(s) are busy: {desc}. "
            "Wait for idle GPUs or pass --allow-busy for a contended diagnostic run."
        )

    out_path = output if output.is_absolute() else (root / "reports/forge/studies" / output)
    raw_path = root / "runs/forge" / f"{out_path.stem}-raw.json"
    cmd = [
        sys.executable,
        str(root / "experiments/law4_optimizer_profiler.py"),
        "--mode",
        "energy-benchmark",
        "--device",
        device,
        "--ideas",
        *ideas,
        "--output",
        str(raw_path),
    ]
    source_commit = _git_output(root, "rev-parse", "HEAD")

    if dry_run:
        return {
            "dry_run": True,
            "command": cmd,
            "raw_output": str(raw_path),
            "curated_study": str(out_path),
            "contended": bool(busy),
            "gpus": gpus,
            "source_commit": source_commit,
        }

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    runner_fn(cmd, cwd=root, check=True)
    raw_doc = _read_json(raw_path)
    curated = build_curated_energy_study(
        root,
        raw_doc,
        contended=bool(busy),
        source_commit=source_commit,
        note_prefix=note,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "dry_run": False,
        "curated_study": str(out_path),
        "raw_output": str(raw_path),
        "candidates": sorted(curated["energy_benchmark"].keys()),
        "contended": bool(busy),
        "source_commit": source_commit,
    }


# ---------------------------------------------------------------------------
# Subcommand 3: promote
# ---------------------------------------------------------------------------


def verify_source_commit_receipts(
    root: Path,
    source_commit: str,
    candidates: list[str],
    *,
    source_snapshot: Path | None = None,
    cuda_model: str | None = None,
) -> dict[str, Any]:
    """Verify hydrated receipts at `source_commit` match the Git tree or verified snapshot.

    Fails fast before any write or regeneration if receipts were recorded on a
    dirty working tree (without a cryptographically verified `--source-snapshot`)
    or if required candidates have no receipt at `source_commit`.
    """
    root = Path(root).resolve()
    resolved_commit = _git_output(root, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
    attempts_dir = root / "reports/forge/attempts"
    if not attempts_dir.is_dir():
        raise ValueError(f"Missing attempts directory: {attempts_dir}")

    matched_candidates: dict[str, list[str]] = {c: [] for c in candidates}
    manifests: dict[str, dict[str, str]] = {}
    for req_path in sorted(attempts_dir.glob("*/request.json")):
        try:
            envelope = _read_json(req_path)
        except Exception:
            continue
        req = envelope.get("request", envelope)
        if cuda_model is not None and req.get("execution_backend") == "cuda":
            if (req.get("compute_profiles") or {}).get("cuda", {}).get("model") != cuda_model:
                continue
        src = req.get("source") if isinstance(req.get("source"), dict) else {}
        if src.get("origin_commit") != resolved_commit:
            continue
        cid = str((req.get("candidate") or {}).get("id") or "")
        if cid in matched_candidates:
            matched_candidates[cid].append(req_path.parent.name)
            files = src.get("files")
            digest = str(src.get("digest") or "")
            if isinstance(files, dict) and digest:
                manifests[digest] = files

    missing = [cid for cid, aids in matched_candidates.items() if not aids]
    if missing:
        raise ValueError(
            f"No hydrated receipts at commit {resolved_commit[:12]} for candidate(s): "
            f"{', '.join(missing)}. Run a clean-commit Forge study from {resolved_commit[:12]} first."
        )

    if source_snapshot is not None:
        snap_dir = source_snapshot if source_snapshot.is_absolute() else (root / source_snapshot)
        snap_dir = snap_dir.resolve()
        if not snap_dir.is_dir():
            raise ValueError(f"Source snapshot directory does not exist: {snap_dir}")
        snap_mismatches: list[str] = []
        for digest, files in sorted(manifests.items()):
            for rel_path, expected_sha in sorted(files.items()):
                target = snap_dir / rel_path
                if not target.is_file():
                    snap_mismatches.append(f"{rel_path} (missing in snapshot)")
                    continue
                actual_sha = _sha256_file(target)
                if actual_sha != expected_sha:
                    snap_mismatches.append(
                        f"{rel_path} (receipt {expected_sha[:10]} != snapshot {actual_sha[:10]})"
                    )
        if snap_mismatches:
            preview = "; ".join(snap_mismatches[:5])
            extra = f" (+{len(snap_mismatches) - 5} more)" if len(snap_mismatches) > 5 else ""
            raise ValueError(
                f"Source snapshot {snap_dir} does not match recorded receipt manifest: {preview}{extra}"
            )
        return {
            "source_commit": resolved_commit,
            "source_snapshot": str(snap_dir),
            "candidates": {cid: sorted(aids) for cid, aids in matched_candidates.items()},
            "verified_manifests": len(manifests),
        }

    mismatches: list[str] = []
    for digest, files in sorted(manifests.items()):
        for rel_path, expected_sha in sorted(files.items()):
            try:
                blob = subprocess.check_output(
                    ["git", "show", f"{resolved_commit}:{rel_path}"],
                    cwd=root,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
                mismatches.append(f"{rel_path} (missing in {resolved_commit[:12]})")
                continue
            actual_sha = _sha256_bytes(blob)
            if actual_sha != expected_sha:
                mismatches.append(
                    f"{rel_path} (receipt {expected_sha[:10]} != git {actual_sha[:10]})"
                )

    if mismatches:
        preview = "; ".join(mismatches[:5])
        extra = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        raise ValueError(
            f"Receipts recorded at {resolved_commit[:12]} do not match the clean Git tree "
            f"(likely recorded on a dirty working tree): {preview}{extra}. "
            "Pass a cryptographically verified --source-snapshot (and --study-commit) or "
            "re-run from a clean committed checkout."
        )

    return {
        "source_commit": resolved_commit,
        "candidates": {cid: sorted(aids) for cid, aids in matched_candidates.items()},
        "verified_manifests": len(manifests),
    }


def _parse_family_assignments(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Expected FAMILY=CANDIDATE, got {raw!r}")
        fam, cid = raw.split("=", 1)
        fam, cid = fam.strip(), cid.strip()
        if not fam or not cid:
            raise ValueError(f"Empty family or candidate in {raw!r}")
        if fam in mapping:
            raise ValueError(f"Duplicate family assignment: {fam!r}")
        mapping[fam] = cid
    if not mapping:
        raise ValueError("Provide at least one --family FAMILY=CANDIDATE assignment")
    return mapping


def _auto_selection_reason(row: dict[str, Any], family_id: str, candidate_id: str) -> str:
    tasks = list(row.get("tasks") or []) + list(row.get("nonrequired_tasks") or [])
    executed = [t for t in tasks if t.get("status") not in {"UNKNOWN", "NOT_MEASURED"}]
    passed = [t for t in executed if t.get("status") == "PASS"]
    failed = sorted(t["task_id"] for t in executed if t.get("status") == "FAIL")
    incomplete = sorted(t["task_id"] for t in executed if t.get("status") in {"INCOMPLETE", "TIMEOUT"})
    tier1 = (row.get("tiers") or {}).get("1") or {}
    t1_pass = tier1.get("passed", 0)
    t1_tot = tier1.get("total", 0)
    gpu = ((row.get("runtime_cohort") or {}).get("compute_profile") or {}).get("model") or "CUDA"
    fail_clause = (
        f"; honest numerical failures: {', '.join(failed)}" if failed else "; 0 numerical failures"
    )
    inc_clause = f"; incomplete/timeout: {', '.join(incomplete)}" if incomplete else ""
    return (
        f"Promote {candidate_id} as the selected {family_id} configuration following ordinary "
        f"clean-commit discriminator_stability CUDA measurement on {gpu} "
        f"({t1_pass}/{t1_tot} projected Tier 1 PASS; {len(passed)}/{len(executed)} executed tasks "
        f"PASS{fail_clause}{inc_clause}). Actual executed-policy qualification and "
        "comparison-policy projection retain their separate source-bound proofs; screening "
        "calibration remains BLOCKED under criteria-v1.json and this pin carries no calibrated "
        "default_adoption claim."
    )


def promote_families(
    root: Path,
    *,
    source_commit: str,
    family_map: dict[str, str],
    comparison_policy: str = DEFAULT_COMPARISON_POLICY,
    reasons: dict[str, str] | None = None,
    keep_canonical: bool = False,
    dry_run: bool = False,
    source_snapshot: Path | None = None,
    study_commit: str | None = None,
    cuda_model: str | None = None,
) -> dict[str, Any]:
    """Verify receipts, register the frozen-source snapshot, and update `family-current-v1.json`."""
    from experiments.forge.trainer_families import family_row_pin

    root = Path(root).resolve()
    reasons = reasons or {}
    candidates = sorted(set(family_map.values()))
    preflight = verify_source_commit_receipts(
        root,
        source_commit,
        candidates,
        source_snapshot=source_snapshot,
        cuda_model=cuda_model,
    )
    resolved_commit = preflight["source_commit"]

    reg_path = root / "configs/forge/trainer-families.json"
    reg_doc = _read_json(reg_path)
    families_by_id = {f["id"]: f for f in reg_doc.get("families", [])}
    for fam_id in family_map:
        if fam_id not in families_by_id:
            raise ValueError(f"Unknown trainer family in configs/forge/trainer-families.json: {fam_id}")

    extra_regen_args: list[str] = []
    if source_snapshot is not None:
        extra_regen_args.extend(["--source-snapshot", str(source_snapshot)])
    if study_commit is not None:
        extra_regen_args.extend(["--study-commit", str(study_commit)])
    if cuda_model is not None:
        extra_regen_args.extend(["--cuda-model", str(cuda_model)])

    # Step 1: Preview candidate rows into a temporary prefix (read-only on repo files).
    with tempfile.TemporaryDirectory(prefix="pg-promote-") as tmp:
        preview_prefix = Path(tmp) / "preview"
        preview_cmd = [
            sys.executable,
            str(root / "reports/forge/regenerate_technique_inventory.py"),
            "--source-commit",
            resolved_commit,
            "--device",
            "cuda",
            "--comparison-policy",
            comparison_policy,
            "--output-prefix",
            str(preview_prefix),
            *extra_regen_args,
        ]
        for cid in candidates:
            preview_cmd.extend(["--candidate", cid])
        subprocess.run(preview_cmd, cwd=root, check=True)
        preview_doc = _read_json(preview_prefix.with_suffix(".json"))

    rows_by_cand: dict[str, list[dict[str, Any]]] = {}
    for row in preview_doc.get("rows", []):
        cid = str(row.get("candidate_id") or "")
        rows_by_cand.setdefault(cid, []).append(row)

    new_pins: dict[str, dict[str, Any]] = {}
    for fam_id, cid in sorted(family_map.items()):
        cand_rows = [
            r for r in rows_by_cand.get(cid, []) if r.get("attempt_ids")
        ]
        if len(cand_rows) != 1:
            raise ValueError(
                f"Expected exactly 1 measured CUDA row for {cid} at {resolved_commit[:12]}, "
                f"found {len(cand_rows)}"
            )
        row = dict(cand_rows[0])
        row["trainer_family"] = fam_id
        reason = reasons.get(fam_id) or _auto_selection_reason(row, fam_id, cid)
        pin = family_row_pin(
            row,
            selection_kind="current_measurement",
            reason=reason,
            measurement_views=["discriminator_stability"],
        )
        new_pins[fam_id] = pin

    sel_path = root / "configs/forge/selections/family-current-v1.json"
    old_sel_bytes = sel_path.read_bytes()
    old_sel_sha = _sha256_bytes(old_sel_bytes)
    archive_rel = f"configs/forge/selections/history/family-current-{old_sel_sha}.json"

    plan_summary = {
        "dry_run": dry_run,
        "source_commit": resolved_commit,
        "comparison_policy": comparison_policy,
        "archive_path": archive_rel,
        "archive_sha256": old_sel_sha,
        "promotions": {
            fam_id: {
                "candidate_id": pin["candidate_id"],
                "scientific_row_sha256": pin["scientific_row_sha256"],
                "cohort": pin["cohort"],
            }
            for fam_id, pin in sorted(new_pins.items())
        },
        "default_adoption": False,
    }
    if dry_run:
        return plan_summary

    # Step 2: Update configs/forge/trainer-families.json so candidate is registered in its family.
    reg_changed = False
    for fam_id, cid in sorted(family_map.items()):
        fam = families_by_id[fam_id]
        if cid not in fam["candidates"]:
            fam["candidates"].append(cid)
            reg_changed = True
        if not keep_canonical and fam.get("canonical_candidate") != cid:
            fam["canonical_candidate"] = cid
            reg_changed = True
    if reg_changed:
        reg_path.write_text(json.dumps(reg_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Step 3: Archive prior selection byte-for-byte and write updated family-current-v1.json.
    archive_path = root / archive_rel
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        archive_path.write_bytes(old_sel_bytes)

    sel_doc = json.loads(old_sel_bytes.decode("utf-8"))
    sel_doc["default_adoption"] = False
    sel_doc["previous_selection"] = {"path": archive_rel, "sha256": old_sel_sha}
    updated_selections: list[dict[str, Any]] = []
    replaced: set[str] = set()
    for entry in sel_doc.get("selections", []):
        fid = str(entry.get("trainer_family") or "")
        if fid in new_pins:
            updated_selections.append(new_pins[fid])
            replaced.add(fid)
        else:
            updated_selections.append(entry)
    for fid, pin in sorted(new_pins.items()):
        if fid not in replaced:
            updated_selections.append(pin)
    updated_selections.sort(key=lambda item: str(item.get("trainer_family") or ""))
    sel_doc["selections"] = updated_selections
    sel_path.write_text(json.dumps(sel_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Step 4: Register snapshot in reports/forge/technique-evidence/manifest.json & regenerate.
    register_cmd = [
        sys.executable,
        str(root / "reports/forge/regenerate_technique_inventory.py"),
        "--source-commit",
        resolved_commit,
        "--device",
        "cuda",
        "--comparison-policy",
        comparison_policy,
        *extra_regen_args,
    ]
    for cid in candidates:
        register_cmd.extend(["--candidate", cid])
    subprocess.run(register_cmd, cwd=root, check=True)

    # Step 5: Refresh publication view coverage so all derived inventory artifacts stay consistent.
    subprocess.run(
        [
            sys.executable,
            str(root / "reports/forge/regenerate_technique_inventory.py"),
            "--refresh-publication",
        ],
        cwd=root,
        check=True,
    )
    return plan_summary


# ---------------------------------------------------------------------------
# Subcommand 4: sync
# ---------------------------------------------------------------------------


def build_sync_steps(
    root: Path,
    *,
    publish: bool = False,
    skip_compile: bool = False,
    skip_docs: bool = False,
    skip_inventory: bool = False,
    skip_checks: bool = False,
    check_mkdocs: bool = False,
    collect_local_media: bool = False,
    ingest_media_dirs: list[Path] | None = None,
    upload_media: bool = False,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_space: str = DEFAULT_HF_SPACE,
    commit_message: str | None = None,
    staging_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build the ordered deterministic refresh pipeline commands."""
    root = Path(root).resolve()
    py = sys.executable
    media_staging = staging_dir or (root / DEFAULT_MEDIA_STAGING)
    steps: list[dict[str, Any]] = []

    if not skip_compile:
        steps.append(
            {
                "name": "forge_compile",
                "description": "Rebuild compiled experiment memory summaries (preserving published qualifications)",
                "cmd": [py, "-m", "experiments.forge", "compile", "--summaries-only"],
            }
        )

    if not skip_inventory:
        steps.append(
            {
                "name": "refresh_technique_inventory",
                "description": "Refresh canonical technique-inventory.md, JSON, and family pages",
                "cmd": [
                    py,
                    str(root / "reports/forge/regenerate_technique_inventory.py"),
                    "--refresh-publication",
                ],
            }
        )

    if not skip_docs:
        steps.append(
            {
                "name": "experiment_docs",
                "description": "Rebuild LAWS_HUB.md and per-experiment documentation pages",
                "cmd": [py, "-m", "experiments.forge.experiment_docs"],
            }
        )

    if collect_local_media:
        steps.append(
            {
                "name": "collect_local_media",
                "description": "Export local certified actual-training GIFs and update media index",
                "cmd": [
                    py,
                    "-m",
                    "experiments.forge.leaderboard_media",
                    "collect-local",
                    "--staging-dir",
                    str(media_staging),
                ],
            }
        )

    if ingest_media_dirs:
        cmd = [
            py,
            "-m",
            "experiments.forge.leaderboard_media",
            "ingest-rendered",
            "--staging-dir",
            str(media_staging),
        ]
        for src in ingest_media_dirs:
            cmd.extend(["--source-dir", str(src)])
        steps.append(
            {
                "name": "ingest_rendered_media",
                "description": "Ingest externally rendered actual-training GIFs into media index",
                "cmd": cmd,
            }
        )

    tmp_root = Path(tempfile.gettempdir())
    portal_cmd = [
        py,
        str(root / "scripts/build_docs_site.py"),
        "--build-dir",
        str(tmp_root / "particlegan-refresh-portal-build"),
        "--site-dir",
        str(tmp_root / "particlegan-refresh-portal-site"),
    ]
    if check_mkdocs:
        portal_cmd.append("--strict")
    else:
        portal_cmd.append("--skip-mkdocs")
    steps.append(
        {
            "name": "validate_github_pages_portal",
            "description": "Stage & validate GitHub Pages Research Portal (including stdlib parity)",
            "cmd": portal_cmd,
        }
    )

    if not skip_checks:
        steps.append(
            {
                "name": "check_repo_bloat",
                "description": "Verify no bulk logs, oversized JSONs, or binary GIFs in Git",
                "cmd": [py, str(root / "scripts/check_repo_bloat.py")],
            }
        )
        steps.append(
            {
                "name": "check_lint",
                "description": "Verify zero new Ruff findings",
                "cmd": [py, str(root / "scripts/check_lint.py")],
            }
        )

    if upload_media or collect_local_media or ingest_media_dirs:
        media_cmd = [
            py,
            "-m",
            "experiments.forge.leaderboard_media",
            "upload-hf",
            "--staging-dir",
            str(media_staging),
            "--repo",
            hf_dataset,
        ]
        if not publish:
            media_cmd.append("--dry-run")
        steps.append(
            {
                "name": "upload_hf_media",
                "description": (
                    "Upload staged actual-training GIFs to Hugging Face Dataset"
                    if publish
                    else "Dry-run validate staged leaderboard GIFs for Hugging Face Dataset"
                ),
                "cmd": media_cmd,
            }
        )

    pub_cmd = [
        py,
        "-m",
        "experiments.forge",
        "publish",
        "--inventory",
        "--repo",
        hf_dataset,
    ]
    if publish:
        pub_cmd.extend(["--receipt", str(root / DEFAULT_PUBLICATION_RECEIPT)])
        if commit_message:
            pub_cmd.extend(["--commit-message", commit_message])
    else:
        pub_cmd.append("--dry-run")
    steps.append(
        {
            "name": "publish_hf_dataset",
            "description": (
                f"Publish data/leaderboard.json & Parquet tables to HF Dataset {hf_dataset}"
                if publish
                else f"Dry-run build & validate HF Dataset exports for {hf_dataset}"
            ),
            "cmd": pub_cmd,
        }
    )

    space_cmd = [
        py,
        str(root / "ops/hf_space/deploy.py"),
        "--space-id",
        hf_space,
    ]
    if publish:
        if commit_message:
            space_cmd.extend(["--commit-message", commit_message])
    else:
        space_cmd.append("--dry-run")
    steps.append(
        {
            "name": "deploy_hf_space",
            "description": (
                f"Deploy static leaderboard UI to HF Space {hf_space}"
                if publish
                else f"Dry-run validate HF Static Space bundle for {hf_space}"
            ),
            "cmd": space_cmd,
        }
    )

    return steps


def run_sync(
    root: Path,
    *,
    publish: bool = False,
    dry_run: bool = False,
    skip_compile: bool = False,
    skip_docs: bool = False,
    skip_inventory: bool = False,
    skip_checks: bool = False,
    skip_verify: bool = False,
    check_mkdocs: bool = False,
    collect_local_media: bool = False,
    ingest_media_dirs: list[Path] | None = None,
    upload_media: bool = False,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_space: str = DEFAULT_HF_SPACE,
    commit_message: str | None = None,
    runner_fn: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Execute (or preview with `--dry-run`) the end-to-end leaderboard refresh pipeline."""
    root = Path(root).resolve()
    steps = build_sync_steps(
        root,
        publish=publish,
        skip_compile=skip_compile,
        skip_docs=skip_docs,
        skip_inventory=skip_inventory,
        skip_checks=skip_checks,
        check_mkdocs=check_mkdocs,
        collect_local_media=collect_local_media,
        ingest_media_dirs=ingest_media_dirs,
        upload_media=upload_media,
        hf_dataset=hf_dataset,
        hf_space=hf_space,
        commit_message=commit_message,
    )

    executed: list[dict[str, Any]] = []
    if not dry_run:
        for step in steps:
            print(f"==> [{step['name']}] {step['description']}", flush=True)
            runner_fn(step["cmd"], cwd=root, check=True)
            executed.append({"name": step["name"], "status": "ok"})

    report: dict[str, Any] = {
        "mode": "plan_only" if dry_run else ("publish" if publish else "local_dry_run"),
        "steps": steps,
        "executed": executed,
        "completed_at": _utc_now(),
    }
    if publish and not dry_run and not skip_verify:
        report["verification"] = verify_live_surfaces(
            root,
            hf_dataset=hf_dataset,
            hf_space=hf_space,
            pages_url=DEFAULT_PAGES_URL,
            check_pr=True,
        )
    return report


# ---------------------------------------------------------------------------
# Subcommand 5: verify
# ---------------------------------------------------------------------------


def _default_http_get(url: str, timeout: float = 20.0) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ParticleGAN-Leaderboard-Refresh/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as err:
        return int(err.code), err.read() if hasattr(err, "read") else b""


def _check_release_pr(
    root: Path,
    *,
    gh_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    gh = shutil.which("gh")
    if not gh:
        return {"ok": False, "error": "gh CLI not installed"}
    proc = gh_runner(
        [
            gh,
            "pr",
            "list",
            "--base",
            "master",
            "--head",
            "develop",
            "--state",
            "open",
            "--json",
            "number,url,title,body",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "gh pr list failed").strip()}
    prs = json.loads(proc.stdout or "[]")
    if not prs:
        return {"ok": False, "error": "No open develop -> master PR found"}
    pr = prs[0]
    has_link = INVENTORY_URL in str(pr.get("body") or "")
    return {
        "ok": has_link,
        "number": pr.get("number"),
        "url": pr.get("url"),
        "title": pr.get("title"),
        "links_technique_inventory": has_link,
    }


def _watch_pages_workflow(root: Path) -> dict[str, Any]:
    gh = shutil.which("gh")
    if not gh:
        return {"ok": False, "error": "gh CLI not installed"}
    proc = subprocess.run(
        [
            gh,
            "run",
            "list",
            "--workflow",
            "pages.yml",
            "--branch",
            "develop",
            "--limit",
            "1",
            "--json",
            "databaseId,status,conclusion,url,headSha",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()}
    runs = json.loads(proc.stdout or "[]")
    if not runs:
        return {"ok": False, "error": "No pages.yml workflow runs found on develop"}
    run = runs[0]
    run_id = str(run["databaseId"])
    if run.get("status") != "completed":
        subprocess.run([gh, "run", "watch", run_id, "--exit-status"], cwd=root, check=False)
        proc2 = subprocess.run(
            [
                gh,
                "run",
                "view",
                run_id,
                "--json",
                "databaseId,status,conclusion,url,headSha",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc2.returncode == 0 and proc2.stdout.strip():
            run = json.loads(proc2.stdout)
    return {
        "ok": run.get("conclusion") == "success",
        "run_id": run.get("databaseId"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "head_sha": run.get("headSha"),
        "url": run.get("url"),
    }


def verify_live_surfaces(
    root: Path,
    *,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_space: str = DEFAULT_HF_SPACE,
    pages_url: str = DEFAULT_PAGES_URL,
    check_pr: bool = True,
    watch_pages: bool = False,
    http_get: Callable[[str], tuple[int, bytes]] = _default_http_get,
    gh_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Verify live HF Dataset, HF Space, GitHub Pages, and develop->master PR."""
    root = Path(root).resolve()
    pages_base = pages_url.rstrip("/")

    # Expected selected families from local repo for parity check (mapped via reporting_family).
    reporting_map: dict[str, str] = {}
    reg_path = root / "configs/forge/trainer-families.json"
    if reg_path.is_file():
        for fam in _read_json(reg_path).get("families", []):
            fid = str(fam.get("id") or "")
            if fid:
                reporting_map[fid] = str(fam.get("reporting_family") or fid)
    expected_selected: dict[str, str] = {}
    sel_path = root / "configs/forge/selections/family-current-v1.json"
    if sel_path.is_file():
        for entry in _read_json(sel_path).get("selections", []):
            tf = str(entry["trainer_family"])
            expected_selected[reporting_map.get(tf, tf)] = str(entry["candidate_id"])

    checks: dict[str, Any] = {}

    if watch_pages:
        checks["github_pages_workflow"] = _watch_pages_workflow(root)

    # 1. HF Dataset data/leaderboard.json
    hf_lb_url = f"https://huggingface.co/datasets/{hf_dataset}/resolve/main/data/leaderboard.json"
    code, body = http_get(hf_lb_url)
    hf_info: dict[str, Any] = {"url": hf_lb_url, "http_status": code, "ok": False}
    if code == 200:
        try:
            payload = json.loads(body.decode("utf-8"))
            fams = {str(f["id"]): str(f.get("candidate_id") or "") for f in payload.get("families", [])}
            eb = payload.get("energy_benchmarks") or []
            study_cids = sorted(
                {
                    str(r.get("candidate_id") or "")
                    for r in eb
                    if (r.get("evidence_scope") or r.get("source_kind")) == "study_energy_benchmark"
                }
            )
            mismatches = {
                fid: {"local": cid, "hf": fams.get(fid)}
                for fid, cid in expected_selected.items()
                if fams.get(fid) != cid
            }
            hf_info.update(
                {
                    "ok": len(fams) >= 10 and len(eb) > 0 and not mismatches,
                    "generated_at": payload.get("generated_at"),
                    "families": len(fams),
                    "energy_rows": len(eb),
                    "study_energy_candidates": study_cids,
                    "selected_mismatches": mismatches,
                }
            )
        except Exception as exc:
            hf_info["error"] = str(exc)
    checks["hf_dataset"] = hf_info

    # 2. HF Static Space
    space_url = f"https://huggingface.co/spaces/{hf_space}"
    sc, _ = http_get(space_url)
    space_app_url = f"https://huggingface.co/spaces/{hf_space}/raw/main/app.js"
    ac, abody = http_get(space_app_url)
    local_app = root / "ops/hf_space/app.js"
    app_matches_local = (
        ac == 200 and local_app.is_file() and _sha256_bytes(abody) == _sha256_file(local_app)
    )
    checks["hf_space"] = {
        "url": space_url,
        "http_status": sc,
        "app_js_status": ac,
        "app_js_matches_local": app_matches_local,
        "ok": sc == 200 and ac == 200 and app_matches_local,
    }

    # 3. GitHub Pages Research Portal
    pages_urls = {
        "home": f"{pages_base}/",
        "leaderboard": f"{pages_base}/leaderboard/",
        "technique_inventory": f"{pages_base}/technique-inventory/",
        "laws": f"{pages_base}/laws/",
    }
    page_statuses: dict[str, int] = {}
    for label, url in pages_urls.items():
        pcode, _ = http_get(url)
        page_statuses[label] = pcode

    pages_lb_url = f"{pages_base}/assets/data/leaderboard.json"
    plcode, plbody = http_get(pages_lb_url)
    pages_info: dict[str, Any] = {
        "base_url": pages_base,
        "page_statuses": page_statuses,
        "leaderboard_json_status": plcode,
        "ok": False,
    }
    if plcode == 200:
        try:
            ppayload = json.loads(plbody.decode("utf-8"))
            pfams = {str(f["id"]): str(f.get("candidate_id") or "") for f in ppayload.get("families", [])}
            peb = ppayload.get("energy_benchmarks") or []
            pmismatches = {
                fid: {"local": cid, "pages": pfams.get(fid)}
                for fid, cid in expected_selected.items()
                if pfams.get(fid) != cid
            }
            pages_info.update(
                {
                    "ok": (
                        all(code == 200 for code in page_statuses.values())
                        and len(pfams) >= 10
                        and len(peb) > 0
                        and not pmismatches
                    ),
                    "generated_at": ppayload.get("generated_at"),
                    "families": len(pfams),
                    "energy_rows": len(peb),
                    "selected_mismatches": pmismatches,
                }
            )
        except Exception as exc:
            pages_info["error"] = str(exc)
    checks["github_pages"] = pages_info

    # 4. develop -> master PR check
    if check_pr:
        checks["develop_to_master_pr"] = _check_release_pr(root, gh_runner=gh_runner)

    overall_ok = all(bool(section.get("ok")) for section in checks.values())
    return {
        "checked_at": _utc_now(),
        "ok": overall_ok,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ops/refresh_leaderboard.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root directory")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit structured JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    st = sub.add_parser("status", help="Offline audit of pins, Pareto energy coverage, and HF freshness")
    st.add_argument("--check-pr", action="store_true", help="Also query GitHub CLI for develop->master PR")

    en = sub.add_parser("energy", help="Run idle-GPU NVML energy benchmark for Pareto Frontier updates")
    en.add_argument("--ideas", nargs="+", required=True, help="Candidate idea IDs to benchmark")
    en.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output study JSON filename (under reports/forge/studies/ if relative)",
    )
    en.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    en.add_argument(
        "--allow-busy",
        action="store_true",
        help="Allow running while GPUs are busy (explicitly marks contended=true)",
    )
    en.add_argument("--note", default=None, help="Optional prefix for cohort_note")
    en.add_argument("--dry-run", action="store_true", help="Check GPU state and print plan without running")

    pr = sub.add_parser("promote", help="Promote verified family cohorts in family-current-v1.json")
    pr.add_argument("--source-commit", required=True, help="Exact Git origin commit SHA recorded on executed receipts")
    pr.add_argument("--source-snapshot", type=Path, default=None, help="Optional verified content-addressed snapshot directory (runs/forge/snapshots/<digest>)")
    pr.add_argument("--study-commit", default=None, help="Optional Git commit holding committed study declarations (defaults to --source-commit)")
    pr.add_argument("--cuda-model", default=None, help="Optional exact CUDA GPU model filter (e.g. 'NVIDIA RTX A6000')")
    pr.add_argument(
        "--family",
        dest="families",
        action="append",
        required=True,
        metavar="FAMILY=CANDIDATE",
        help="Promote CANDIDATE as the selected row for FAMILY; repeat as needed",
    )
    pr.add_argument(
        "--reason",
        dest="reasons",
        action="append",
        default=[],
        metavar="FAMILY=TEXT",
        help="Optional custom selection reason for FAMILY",
    )
    pr.add_argument(
        "--comparison-policy",
        default=DEFAULT_COMPARISON_POLICY,
        help=f"Registered comparison policy (default: {DEFAULT_COMPARISON_POLICY})",
    )
    pr.add_argument(
        "--keep-canonical",
        action="store_true",
        help="Add candidate to trainer-families.json without changing canonical_candidate",
    )
    pr.add_argument("--dry-run", action="store_true", help="Validate receipts & preview pins without writing")

    sy = sub.add_parser("sync", help="Rebuild local reports/portal and optionally publish to Hugging Face")
    sy.add_argument(
        "--publish",
        action="store_true",
        help="Publish HF Dataset, upload staged GIFs, deploy HF Static Space, and verify live",
    )
    sy.add_argument("--dry-run", action="store_true", help="Print the ordered command plan without running")
    sy.add_argument("--skip-compile", action="store_true", help="Skip `experiments.forge compile`")
    sy.add_argument("--skip-docs", action="store_true", help="Skip `experiments.forge.experiment_docs`")
    sy.add_argument("--skip-inventory", action="store_true", help="Skip `regenerate_technique_inventory.py`")
    sy.add_argument("--skip-checks", action="store_true", help="Skip bloat and lint gates")
    sy.add_argument("--skip-verify", action="store_true", help="Skip post-publish live HTTP verification")
    sy.add_argument("--check-mkdocs", action="store_true", help="Run full `mkdocs build --strict` locally")
    sy.add_argument("--collect-local-media", action="store_true", help="Export local selected-cohort GIFs")
    sy.add_argument(
        "--ingest-media",
        dest="ingest_media",
        action="append",
        type=Path,
        default=[],
        help="Ingest externally rendered GIF directory; repeat as needed",
    )
    sy.add_argument("--upload-media", action="store_true", help="Include HF media upload step")
    sy.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET, help="Target HF Dataset repo")
    sy.add_argument("--hf-space", default=DEFAULT_HF_SPACE, help="Target HF Static Space repo")
    sy.add_argument("--commit-message", default=None, help="Optional HF commit message")

    vf = sub.add_parser("verify", help="Verify live HF Dataset, HF Space, GitHub Pages, and release PR")
    vf.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    vf.add_argument("--hf-space", default=DEFAULT_HF_SPACE)
    vf.add_argument("--pages-url", default=DEFAULT_PAGES_URL)
    vf.add_argument("--no-pr", action="store_true", help="Skip develop -> master PR check")
    vf.add_argument(
        "--watch-pages",
        action="store_true",
        help="Watch the latest develop Research Portal (GitHub Pages) Actions workflow run first",
    )

    return parser


def _print_human_status(data: dict[str, Any]) -> None:
    git_info = data["git"]
    clean_str = "clean" if git_info["clean"] else f"DIRTY ({git_info['dirty_count']} file(s))"
    print(f"Git: {git_info['branch']} @ {git_info['head'][:12]} [{clean_str}]")
    print(f"Selected families: {len(data['selected_families'])}")
    hf = data.get("hf_publication_receipt")
    if hf:
        match = "matches HEAD" if hf["matches_head"] else f"behind HEAD ({str(hf['git_source_commit'])[:12]})"
        print(
            f"HF Dataset receipt: {hf['published_at']} ({match}; "
            f"{hf['families']} families, {hf['energy_rows']} energy rows)"
        )
    else:
        print("HF Dataset receipt: missing")
    if data["newer_candidates_beyond_pins"]:
        print("Newer candidates beyond family-current-v1.json:")
        for item in data["newer_candidates_beyond_pins"]:
            print(
                f"  - {item['family_id']}: pinned {item['pinned_candidate']} "
                f"-> newer {item['newer_candidates']}"
            )
    if data["occam_complete_missing_energy_study"]:
        print("Occam-complete candidates missing an uncontended energy study:")
        for item in data["occam_complete_missing_energy_study"]:
            print(
                f"  - {item['candidate_id']} on {item['hardware_gpu']} "
                f"({item['occam_pass']}/{item['occam_executed']} Occam PASS)"
            )
    if data["recommended_actions"]:
        print("Recommended next steps:")
        for act in data["recommended_actions"]:
            print(f"  * {act}")


def _print_human_verify(data: dict[str, Any]) -> None:
    overall = "PASS" if data["ok"] else "NEEDS ATTENTION"
    print(f"Live Leaderboard Verification: [{overall}] ({data['checked_at']})")
    for name, info in data["checks"].items():
        badge = "OK" if info.get("ok") else "FAIL"
        print(f"  - {name}: [{badge}] {json.dumps(info, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()

    if args.command == "status":
        result = collect_status(root, check_pr=args.check_pr)
        if args.as_json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            _print_human_status(result)
        return 0

    if args.command == "energy":
        result = run_energy_benchmark(
            root,
            ideas=args.ideas,
            output=args.output,
            device=args.device,
            allow_busy=args.allow_busy,
            dry_run=args.dry_run,
            note=args.note,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "promote":
        family_map = _parse_family_assignments(args.families)
        reason_map: dict[str, str] = {}
        for raw in args.reasons:
            if "=" not in raw:
                raise ValueError(f"Expected --reason FAMILY=TEXT, got {raw!r}")
            fam, txt = raw.split("=", 1)
            reason_map[fam.strip()] = txt.strip()
        result = promote_families(
            root,
            source_commit=args.source_commit,
            family_map=family_map,
            comparison_policy=args.comparison_policy,
            reasons=reason_map,
            keep_canonical=args.keep_canonical,
            dry_run=args.dry_run,
            source_snapshot=args.source_snapshot,
            study_commit=args.study_commit,
            cuda_model=args.cuda_model,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "sync":
        result = run_sync(
            root,
            publish=args.publish,
            dry_run=args.dry_run,
            skip_compile=args.skip_compile,
            skip_docs=args.skip_docs,
            skip_inventory=args.skip_inventory,
            skip_checks=args.skip_checks,
            skip_verify=args.skip_verify,
            check_mkdocs=args.check_mkdocs,
            collect_local_media=args.collect_local_media,
            ingest_media_dirs=args.ingest_media,
            upload_media=args.upload_media,
            hf_dataset=args.hf_dataset,
            hf_space=args.hf_space,
            commit_message=args.commit_message,
        )
        if args.as_json or args.dry_run:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Sync [{result['mode']}] completed ({len(result['executed'])} steps).")
            if "verification" in result:
                _print_human_verify(result["verification"])
                return 0 if result["verification"]["ok"] else 1
        return 0

    if args.command == "verify":
        result = verify_live_surfaces(
            root,
            hf_dataset=args.hf_dataset,
            hf_space=args.hf_space,
            pages_url=args.pages_url,
            check_pr=not args.no_pr,
            watch_pages=args.watch_pages,
        )
        if args.as_json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            _print_human_verify(result)
        return 0 if result["ok"] else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
