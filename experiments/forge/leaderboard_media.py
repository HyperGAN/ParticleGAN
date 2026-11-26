"""Provenance-verified actual-training GIF exporter and leaderboard media index.

Renders training GIFs strictly from certified saved training observations and
retained scored-output arrays (`optimizer_updates_added=0`,
`sampling_draws_added=0`, `qualification_input=False`). Never modifies
qualification grades, never commits binary `.gif` files into Git, and keeps
manifest-loading helpers stdlib-only so lightweight CI (e.g., GitHub Pages
without PyTorch) can read `reports/forge/leaderboard-media-index.json`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

DEFAULT_MEDIA_INDEX_RELPATH = Path("reports/forge/leaderboard-media-index.json")
DEFAULT_HF_REPO = "HyperGAN/particlegan-forge-evidence"
DEFAULT_MEDIA_PREFIX = "media/forge/"
LEADERBOARD_MEDIA_SUBDIR = "media/forge/leaderboard"
MEDIA_INDEX_SCHEMA = "forge_leaderboard_media_index_v1"

# Certified actual-training GIF receipt kinds emitted by `tier1_media` and
# family media publishers (all strictly `optimizer_updates_added=0`,
# `sampling_draws_added=0`, `qualification_input=False`).
VALID_GIF_KINDS = frozenset({
    "actual_training_saved_observations_gif",
    "actual_training_native_saved_observations_gif",
    "actual_training_saved_images_gif",
    "actual_training_saved_spiral_gif",
})

# Priority order for choosing the best GIF when multiple media entries exist for
# the same (family_id, task_id): exact selected-cohort runs with visual sample
# panels always beat replay or curves-only media.
_COHORT_RANK = {
    "selected_cohort": 0,
    "colab_replay_media": 1,
    "matched_candidate_replay": 2,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_media_relpath(raw: str | None) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    normalized = raw.replace("\\", "/").lstrip("./")
    if normalized.startswith("reports/forge/"):
        return DEFAULT_MEDIA_PREFIX + normalized[len("reports/forge/"):]
    if normalized.startswith("media/"):
        return normalized
    return DEFAULT_MEDIA_PREFIX + normalized


def load_leaderboard_media_index(root: Path) -> dict[str, Any]:
    """Read `reports/forge/leaderboard-media-index.json` using stdlib only."""
    path = Path(root).resolve() / DEFAULT_MEDIA_INDEX_RELPATH
    if not path.is_file():
        return {
            "schema_version": 1,
            "schema": MEDIA_INDEX_SCHEMA,
            "qualification_input": False,
            "hf_repo": DEFAULT_HF_REPO,
            "media_prefix": DEFAULT_MEDIA_PREFIX,
            "items": [],
        }
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != MEDIA_INDEX_SCHEMA or not isinstance(doc.get("items"), list):
        raise ValueError(f"Unsupported leaderboard media index schema in {path}")
    return doc


def index_media_lookup(root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Return `(by_family_task, by_attempt_task)` dicts from the committed media index.

    Stdlib-only: safe to call in PyTorch-free CI builds (GitHub Pages).
    """
    doc = load_leaderboard_media_index(root)
    by_family_task: dict[tuple[str, str], dict[str, Any]] = {}
    by_attempt_task: dict[tuple[str, str], dict[str, Any]] = {}

    def _score(entry: dict[str, Any]) -> tuple[int, int, int, str]:
        cohort_pri = _COHORT_RANK.get(str(entry.get("cohort_match") or ""), 9)
        grade_match_pri = 0 if entry.get("grade_matches_selected", True) else 1
        visual_pri = 0 if entry.get("has_visual_samples") else 1
        return (cohort_pri, grade_match_pri, visual_pri, str(entry.get("gif_relpath") or ""))

    for item in doc.get("items", []):
        if not isinstance(item, dict):
            continue
        rel = _normalize_media_relpath(item.get("gif_relpath") or item.get("gif"))
        if not rel:
            continue
        normalized = {**item, "gif_relpath": rel}
        fid = str(item.get("family_id") or "")
        tid = str(item.get("task_id") or "")
        aid = str(item.get("attempt_id") or "")
        if fid and tid:
            key = (fid, tid)
            prev = by_family_task.get(key)
            if prev is None or _score(normalized) < _score(prev):
                by_family_task[key] = normalized
        if aid and tid:
            akey = (aid, tid)
            prev_a = by_attempt_task.get(akey)
            if prev_a is None or _score(normalized) < _score(prev_a):
                by_attempt_task[akey] = normalized
    return by_family_task, by_attempt_task


def _extract_gpu_name(request: dict[str, Any], row: dict[str, Any]) -> str | None:
    cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
    energy = cost.get("energy") if isinstance(cost.get("energy"), dict) else {}
    gpu = energy.get("gpu_name")
    if isinstance(gpu, str) and gpu.strip():
        return gpu.strip()
    cprofs = request.get("compute_profiles") if isinstance(request.get("compute_profiles"), dict) else {}
    cuda = cprofs.get("cuda") if isinstance(cprofs.get("cuda"), dict) else {}
    model = cuda.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _render_scalar_views_fallback(
    task: dict[str, Any],
    row: dict[str, Any],
    local: Path,
    gif_path: Path,
    prior_error: Exception,
) -> dict[str, Any]:
    """Render saved `views` records (e.g., ae_gan_hold / mid_scale_identity) without extra runs."""
    import torch

    from benchmarks.toy_audit.api_run import render_gif
    from experiments.forge.contracts import atomic_json, file_hash, stable_hash

    evidence = row["evidence"]
    descriptor = evidence.get("saved_observer_outputs")
    if not isinstance(descriptor, dict) or not descriptor.get("path"):
        raise prior_error
    source = (local / descriptor["path"]).resolve()
    if not source.is_relative_to(local.resolve()) or not source.is_file():
        raise prior_error
    if file_hash(source) != descriptor["sha256"]:
        raise ValueError("saved scored-output bytes differ from certified descriptor")
    records = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) and "views" in r for r in records):
        raise prior_error

    by_step = {r["step"]: r for r in records}
    observations = evidence.get("observations") or []
    if not observations or not all(p["step"] in by_step for p in observations):
        raise prior_error
    selected = [by_step[p["step"]] for p in observations]
    if not all(
        r.get("metrics") == {k: v for k, v in p.items() if k != "step"}
        for r, p in zip(selected, observations)
    ):
        raise ValueError("saved view metrics differ from recorded observations")

    case = {
        "id": task["id"],
        "goal": task.get("description", task["id"]),
        "default_steps": task["execution"]["steps"],
        "sampling": evidence["sampling_law"],
    }
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    display = [
        {
            **r,
            "metrics": {
                k: v
                for k, v in (r.get("metrics") or {}).items()
                if type(v) in (int, float, bool)
            },
        }
        for r in selected
    ]
    render_gif(
        case,
        display,
        gif_path,
        full_budget=True,
        requested_steps=task["execution"]["steps"],
        final_verdict=row["gate_status"],
    )
    receipt = {
        "schema_version": 1,
        "task_id": task["id"],
        "recorded_grade": row["gate_status"],
        "kind": "actual_training_saved_observations_gif",
        "observation_count": len(selected),
        "source_inputs": {str(source): file_hash(source)},
        "observations_sha256": stable_hash(observations),
        "gif_sha256": file_hash(gif_path),
        "optimizer_updates_added": 0,
        "sampling_draws_added": 0,
        "qualification_input": False,
        "selection": "All scored scheduled records from the original saved training views",
        "metric_labels": "Scalar display labels only; original saved metrics and views are unchanged",
        "retained_records": len(records),
        "adapter_note": f"Strict exporter rejected extra initial/display records or non-scalar labels: {prior_error}",
        "renderer": "benchmarks.toy_audit.api_run:render_gif",
    }
    atomic_json(gif_path.with_suffix(".json"), receipt)
    return receipt


def export_attempt_media(attempt_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Export actual-training GIFs + provenance sidecars for all tasks in one certified attempt."""
    from experiments.forge.contracts import atomic_json, read_json, stable_hash
    from experiments.forge.tier1_media import render

    attempt_dir = Path(attempt_dir).resolve()
    output_dir = Path(output_dir).resolve()
    envelope = read_json(attempt_dir / "request.json")
    result = read_json(attempt_dir / "result.json")
    certificate = read_json(attempt_dir / "evidence.json")
    request = envelope.get("request", envelope)

    if certificate["result_hash"] != stable_hash(result) or certificate["source"] != request["source"]:
        raise ValueError(f"{attempt_dir.name}: original source/result certificate differs")
    if result["candidate_revision"] != request["candidate_revision"]:
        raise ValueError(f"{attempt_dir.name}: candidate revision differs")

    local = Path(certificate["local_artifact_root"])
    if not local.is_dir() or not (local / "result.json").is_file():
        raise FileNotFoundError(f"{attempt_dir.name}: local_artifact_root unavailable: {local}")
    if read_json(local / "result.json") != result:
        raise ValueError(f"{attempt_dir.name}: local result differs from certified envelope")

    candidate_id = str((request.get("candidate") or {}).get("id") or result.get("candidate_id") or "")
    source_digest = str((request.get("source") or {}).get("digest") or "")
    attempt_id = str(result.get("attempt_id") or attempt_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    receipts: list[dict[str, Any]] = []
    for row in result.get("task_results", []):
        gate = row.get("gate_status")
        if gate not in {"PASS", "FAIL", "BLOCKED"} or not row.get("evidence"):
            continue
        tid = str(row["task_id"])
        task = request["tasks"][tid]
        gif_path = output_dir / f"{tid}.gif"

        display_row = row
        ev = row.get("evidence") or {}
        if ev.get("dense") and not ev.get("observations"):
            display_row = {**row, "evidence": {**ev, "observations": ev["dense"]}}

        receipt: dict[str, Any] | None = None
        try:
            receipt = render(task, display_row, local, gif_path)
        except (ValueError, TypeError) as exc:
            try:
                receipt = _render_scalar_views_fallback(task, display_row, local, gif_path, exc)
            except Exception:
                # Final curves-only fallback strictly from saved observations (same as attempt_worker.py).
                if display_row["evidence"].get("saved_observer_outputs") and display_row["evidence"].get("observations"):
                    stripped_ev = dict(display_row["evidence"])
                    stripped_ev.pop("saved_observer_outputs", None)
                    receipt = render(task, {**display_row, "evidence": stripped_ev}, local, gif_path)
                    receipt["media_note"] = f"Saved training metric curves; sample-panel fallback: {exc}"
                else:
                    raise

        if receipt is None or not gif_path.is_file():
            continue

        has_visual_samples = bool(receipt.get("source_inputs"))
        if task.get("adapter") in {"native100", "native100_continuation"}:
            has_visual_samples = True

        enriched = {
            **receipt,
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "source_digest": source_digest,
            "gpu_name": _extract_gpu_name(request, row),
            "bytes": gif_path.stat().st_size,
            "has_visual_samples": has_visual_samples,
        }
        atomic_json(gif_path.with_suffix(".json"), enriched)
        receipts.append(enriched)
    return receipts


def _selected_family_maps(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return `(by_family_id, candidate_to_family)` from `technique-inventory.json`."""
    from experiments.forge.contracts import read_json
    from experiments.forge.family_reports import _selected_cohorts, _selected_row

    inv = read_json(root / "reports/forge/technique-inventory.json")
    prog = inv["family_progress"]
    by_fam: dict[str, dict[str, Any]] = {}
    cand_to_fam: dict[str, str] = {}
    for fam, coh in _selected_cohorts(prog):
        sel = _selected_row(inv, coh)
        fid = str(fam["id"])
        cid = str(sel.get("candidate_id") or fam.get("benchmark_configuration", {}).get("candidate_id") or "")
        if cid:
            cand_to_fam[cid] = fid
        by_fam[fid] = {
            "family_id": fid,
            "label": str(fam.get("label") or fid),
            "candidate_id": cid,
            "attempt_ids": list(sel.get("attempt_ids", [])),
            "tasks": coh.get("tasks", {}),
            "source_digest": str((sel.get("bindings") or {}).get("source_digest") or ""),
        }
    return by_fam, cand_to_fam


def _manifest_entry(
    *,
    family_id: str,
    candidate_id: str,
    task_id: str,
    attempt_id: str,
    recorded_grade: str,
    selected_cohort_grade: str | None,
    cohort_match: str,
    gif_relpath: str,
    gif_sha256: str,
    size_bytes: int | None,
    observation_count: int | None,
    observations_sha256: str | None,
    has_visual_samples: bool,
    source_digest: str | None = None,
    gpu_name: str | None = None,
) -> dict[str, Any]:
    grade_matches = (
        True
        if selected_cohort_grade is None or selected_cohort_grade in {"UNKNOWN", "NOT_RUN", ""}
        else (str(recorded_grade).upper() == str(selected_cohort_grade).upper())
    )
    entry: dict[str, Any] = {
        "attempt_id": attempt_id,
        "bytes": int(size_bytes) if isinstance(size_bytes, int) else None,
        "candidate_id": candidate_id,
        "cohort_match": cohort_match,
        "family_id": family_id,
        "gif_relpath": _normalize_media_relpath(gif_relpath),
        "gif_sha256": gif_sha256,
        "grade_matches_selected": grade_matches,
        "has_visual_samples": bool(has_visual_samples),
        "kind": "actual_training_saved_observations_gif",
        "observation_count": int(observation_count) if isinstance(observation_count, int) else None,
        "observations_sha256": observations_sha256,
        "optimizer_updates_added": 0,
        "qualification_input": False,
        "recorded_grade": str(recorded_grade),
        "sampling_draws_added": 0,
        "selected_cohort_grade": selected_cohort_grade,
        "task_id": task_id,
    }
    if source_digest:
        entry["source_digest"] = source_digest
    if gpu_name:
        entry["gpu_name"] = gpu_name
    return entry


def _merge_items(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by `(family_id, task_id)`, preferring exact selected-cohort visual media."""
    def _priority(item: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            _COHORT_RANK.get(str(item.get("cohort_match") or ""), 9),
            0 if item.get("grade_matches_selected", True) else 1,
            0 if item.get("has_visual_samples") else 1,
            str(item.get("gif_relpath") or ""),
        )

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in [*existing, *incoming]:
        key = (str(item.get("family_id") or ""), str(item.get("task_id") or ""))
        if not key[0] or not key[1]:
            continue
        prev = merged.get(key)
        if prev is None or _priority(item) < _priority(prev):
            merged[key] = item
    return [merged[k] for k in sorted(merged)]


def collect_existing_and_local_media(
    root: Path,
    staging_media_dir: Path,
    *,
    write_index: bool = True,
) -> dict[str, Any]:
    """Collect already-verified HF media and render all available local selected-cohort GIFs."""
    from experiments.forge.contracts import atomic_json, read_json

    root = Path(root).resolve()
    staging_media_dir = Path(staging_media_dir).resolve()
    staging_media_dir.mkdir(parents=True, exist_ok=True)

    by_fam, _ = _selected_family_maps(root)
    hm_path = root / "reports/forge/hf-media-manifest-2026-11-21.json"
    hm_files = read_json(hm_path).get("files", {}) if hm_path.is_file() else {}

    collected: list[dict[str, Any]] = []
    stats: dict[str, int] = {"hf_existing": 0, "local_rendered": 0, "local_reused": 0, "local_errors": 0}
    errors: list[dict[str, str]] = []

    # 1. Verified HF-hosted BISPF selected-cohort media (reports/forge/bispf-optimized/quality-media/)
    bispf_info = by_fam.get("bispf")
    bispf_q_path = root / "reports/forge/bispf-optimized/quality.json"
    bispf_m_dir = root / "reports/forge/bispf-optimized/quality-media"
    if bispf_info and bispf_q_path.is_file() and bispf_m_dir.is_dir():
        bq = read_json(bispf_q_path)
        task_to_aid: dict[str, str] = {}
        for summ in bq.get("attempt_summaries", []):
            aid = str(summ.get("attempt_id") or "")
            for tr in summ.get("task_results", []):
                tid = str(tr.get("task_id") or "")
                if tid and aid:
                    task_to_aid[tid] = aid
        for jp in sorted(bispf_m_dir.glob("*.json")):
            if jp.stem.endswith("-transient"):
                continue
            gif_repo_rel = jp.with_suffix(".gif").relative_to(root).as_posix()
            hm_entry = hm_files.get(gif_repo_rel)
            if not hm_entry:
                continue
            doc = read_json(jp)
            if doc.get("kind") not in VALID_GIF_KINDS:
                continue
            tid = str(doc.get("task_id") or jp.stem)
            aid = task_to_aid.get(tid, "")
            if aid and aid not in set(bispf_info["attempt_ids"]):
                continue
            if hm_entry[1] != doc.get("gif_sha256"):
                continue
            coh_grade = (bispf_info["tasks"].get(tid) or {}).get("status")
            collected.append(
                _manifest_entry(
                    family_id="bispf",
                    candidate_id=bispf_info["candidate_id"],
                    task_id=tid,
                    attempt_id=aid or "bispf-optimized-quality",
                    recorded_grade=str(doc.get("recorded_grade") or coh_grade or ""),
                    selected_cohort_grade=coh_grade,
                    cohort_match="selected_cohort",
                    gif_relpath=gif_repo_rel,
                    gif_sha256=str(doc["gif_sha256"]),
                    size_bytes=hm_entry[0],
                    observation_count=doc.get("observation_count"),
                    observations_sha256=doc.get("observations_sha256"),
                    has_visual_samples=bool(doc.get("source_inputs")) or tid in {"grid100", "rotated100", "staggered100"},
                    source_digest=bispf_info["source_digest"],
                )
            )
            stats["hf_existing"] += 1

    # 2. Verified HF-hosted BCAP-pure selected-cohort media (reports/forge/recipe-prior-refactor/publication/media/)
    bcap_info = by_fam.get("bcap-pure")
    rpr_dir = root / "reports/forge/recipe-prior-refactor/publication/media"
    if bcap_info and rpr_dir.is_dir():
        bcap_aids = set(bcap_info["attempt_ids"])
        # Map task_id -> attempt_id from selected cohort tasks
        bcap_task_aid = {
            tid: str(cell.get("attempt_id") or "")
            for tid, cell in bcap_info["tasks"].items()
            if isinstance(cell, dict) and cell.get("attempt_id")
        }
        for jp in sorted(rpr_dir.glob("bcap-pure-ordinary-*.json")):
            gif_repo_rel = jp.with_suffix(".gif").relative_to(root).as_posix()
            hm_entry = hm_files.get(gif_repo_rel)
            if not hm_entry:
                continue
            doc = read_json(jp)
            if doc.get("kind") not in VALID_GIF_KINDS:
                continue
            if hm_entry[1] != doc.get("gif_sha256"):
                continue
            tid = str(doc.get("task_id") or "")
            src_str = " ".join((doc.get("source_inputs") or {}).keys())
            matched_aid = next((a for a in bcap_aids if a in src_str), bcap_task_aid.get(tid, ""))
            coh_grade = (bcap_info["tasks"].get(tid) or {}).get("status")
            if coh_grade and str(doc.get("recorded_grade") or "") != str(coh_grade):
                continue
            collected.append(
                _manifest_entry(
                    family_id="bcap-pure",
                    candidate_id=bcap_info["candidate_id"],
                    task_id=tid,
                    attempt_id=matched_aid or "bcap-pure-ordinary",
                    recorded_grade=str(doc.get("recorded_grade") or coh_grade or ""),
                    selected_cohort_grade=coh_grade,
                    cohort_match="selected_cohort",
                    gif_relpath=gif_repo_rel,
                    gif_sha256=str(doc["gif_sha256"]),
                    size_bytes=hm_entry[0],
                    observation_count=doc.get("observation_count"),
                    observations_sha256=doc.get("observations_sha256"),
                    has_visual_samples=bool(doc.get("source_inputs")) or tid in {"grid100", "rotated100", "staggered100"},
                    source_digest=bcap_info["source_digest"],
                )
            )
            stats["hf_existing"] += 1

    # 3. Preserve existing combined-media-index entries (e.g., e22 policy cloud & tier1-completion)
    combined_path = root / "reports/forge/atlas-compatible-coverage-20261013/combined-media-index.json"
    if combined_path.is_file():
        combined_doc = read_json(combined_path)
        for item in combined_doc.get("items", []):
            fid = str(item.get("family") or "")
            if fid not in by_fam:
                continue
            fam_info = by_fam[fid]
            tid = str(item.get("task_id") or "")
            aid = str(item.get("attempt_id") or "")
            gif_repo_rel = str(item.get("gif") or "")
            hm_entry = hm_files.get(gif_repo_rel)
            if not hm_entry or hm_entry[1] != item.get("gif_sha256"):
                continue
            coh_grade = (fam_info["tasks"].get(tid) or {}).get("status")
            if coh_grade and str(item.get("recorded_grade") or "") != str(coh_grade):
                continue
            collected.append(
                _manifest_entry(
                    family_id=fid,
                    candidate_id=fam_info["candidate_id"],
                    task_id=tid,
                    attempt_id=aid,
                    recorded_grade=str(item.get("recorded_grade") or coh_grade or ""),
                    selected_cohort_grade=coh_grade,
                    cohort_match="selected_cohort",
                    gif_relpath=gif_repo_rel,
                    gif_sha256=str(item["gif_sha256"]),
                    size_bytes=hm_entry[0],
                    observation_count=item.get("observation_count"),
                    observations_sha256=item.get("observations_sha256"),
                    has_visual_samples=bool(item.get("source_inputs")),
                    source_digest=fam_info["source_digest"],
                )
            )
            stats["hf_existing"] += 1

    # 4. Render all available local selected-cohort attempts (axiom, empf, atlas, release07-gan-v3, k3p, ka2, r1r2)
    att_root = root / "reports/forge/attempts"
    for fid, fam_info in sorted(by_fam.items()):
        cid = fam_info["candidate_id"]
        for aid in fam_info["attempt_ids"]:
            att_dir = att_root / aid
            if not (att_dir / "result.json").is_file() or not (att_dir / "evidence.json").is_file():
                continue
            try:
                cert = read_json(att_dir / "evidence.json")
                lar = Path(cert.get("local_artifact_root") or "")
                if not lar.is_dir() or not (lar / "result.json").is_file():
                    continue
                res = read_json(att_dir / "result.json")
                task_rows = [
                    r for r in res.get("task_results", [])
                    if r.get("gate_status") in {"PASS", "FAIL"} and r.get("evidence")
                ]
                if not task_rows:
                    continue
                dest_dir = staging_media_dir / LEADERBOARD_MEDIA_SUBDIR / fid / cid
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Check if already rendered cleanly in staging_media_dir
                already_ok = True
                cached_receipts: list[dict[str, Any]] = []
                for tr in task_rows:
                    tid = str(tr["task_id"])
                    gp = dest_dir / f"{tid}.gif"
                    sp = dest_dir / f"{tid}.json"
                    if not gp.is_file() or not sp.is_file():
                        already_ok = False
                        break
                    rec = read_json(sp)
                    if rec.get("attempt_id") != aid or rec.get("gif_sha256") != _file_sha256(gp):
                        already_ok = False
                        break
                    cached_receipts.append(rec)

                if already_ok and cached_receipts:
                    exported = cached_receipts
                    stats["local_reused"] += len(exported)
                else:
                    exported = export_attempt_media(att_dir, dest_dir)
                    stats["local_rendered"] += len(exported)

                for rec in exported:
                    tid = str(rec["task_id"])
                    gif_rel = f"{LEADERBOARD_MEDIA_SUBDIR}/{fid}/{cid}/{tid}.gif"
                    coh_grade = (fam_info["tasks"].get(tid) or {}).get("status")
                    collected.append(
                        _manifest_entry(
                            family_id=fid,
                            candidate_id=cid,
                            task_id=tid,
                            attempt_id=aid,
                            recorded_grade=str(rec.get("recorded_grade") or ""),
                            selected_cohort_grade=coh_grade,
                            cohort_match="selected_cohort",
                            gif_relpath=gif_rel,
                            gif_sha256=str(rec["gif_sha256"]),
                            size_bytes=rec.get("bytes"),
                            observation_count=rec.get("observation_count"),
                            observations_sha256=rec.get("observations_sha256"),
                            has_visual_samples=bool(rec.get("has_visual_samples")),
                            source_digest=rec.get("source_digest") or fam_info["source_digest"],
                            gpu_name=rec.get("gpu_name"),
                        )
                    )
            except Exception as exc:
                stats["local_errors"] += 1
                errors.append({"family_id": fid, "attempt_id": aid, "error": f"{type(exc).__name__}: {exc}"})

    existing_doc = load_leaderboard_media_index(root)
    merged_items = _merge_items(existing_doc.get("items", []), collected)
    payload = {
        "schema_version": 1,
        "schema": MEDIA_INDEX_SCHEMA,
        "generated_at": _utc_now(),
        "qualification_input": False,
        "hf_repo": DEFAULT_HF_REPO,
        "media_prefix": DEFAULT_MEDIA_PREFIX,
        "items": merged_items,
    }
    if write_index:
        atomic_json(root / DEFAULT_MEDIA_INDEX_RELPATH, payload)
    return {"total_items": len(merged_items), "stats": stats, "errors": errors}


def ingest_rendered_media_dirs(
    root: Path,
    source_dirs: list[Path],
    staging_media_dir: Path,
    *,
    cohort_match: str = "colab_replay_media",
    write_index: bool = True,
) -> dict[str, Any]:
    """Verify & merge rendered `.gif` + `.json` sidecars (e.g. from Colab) into staging & index."""
    from experiments.forge.contracts import atomic_json, read_json

    root = Path(root).resolve()
    staging_media_dir = Path(staging_media_dir).resolve()
    by_fam, cand_to_fam = _selected_family_maps(root)

    incoming: list[dict[str, Any]] = []
    copied = 0
    skipped = 0
    for src_root in source_dirs:
        src_root = Path(src_root).resolve()
        if not src_root.is_dir():
            continue
        for sp in sorted(src_root.rglob("*.json")):
            gp = sp.with_suffix(".gif")
            if not gp.is_file():
                continue
            try:
                rec = read_json(sp)
            except Exception:
                skipped += 1
                continue
            if rec.get("kind") not in VALID_GIF_KINDS:
                skipped += 1
                continue
            actual_sha = _file_sha256(gp)
            if rec.get("gif_sha256") != actual_sha:
                skipped += 1
                continue
            cid = str(rec.get("candidate_id") or "")
            fid = cand_to_fam.get(cid)
            if not fid or fid not in by_fam:
                skipped += 1
                continue
            fam_info = by_fam[fid]
            tid = str(rec.get("task_id") or sp.stem)
            aid = str(rec.get("attempt_id") or sp.parent.name)
            dest = staging_media_dir / LEADERBOARD_MEDIA_SUBDIR / fid / cid / f"{tid}.gif"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gp, dest)
            shutil.copy2(sp, dest.with_suffix(".json"))
            copied += 1

            coh_grade = (fam_info["tasks"].get(tid) or {}).get("status")
            effective_match = "selected_cohort" if aid in set(fam_info["attempt_ids"]) else cohort_match
            incoming.append(
                _manifest_entry(
                    family_id=fid,
                    candidate_id=cid,
                    task_id=tid,
                    attempt_id=aid,
                    recorded_grade=str(rec.get("recorded_grade") or ""),
                    selected_cohort_grade=coh_grade,
                    cohort_match=effective_match,
                    gif_relpath=f"{LEADERBOARD_MEDIA_SUBDIR}/{fid}/{cid}/{tid}.gif",
                    gif_sha256=actual_sha,
                    size_bytes=dest.stat().st_size,
                    observation_count=rec.get("observation_count"),
                    observations_sha256=rec.get("observations_sha256"),
                    has_visual_samples=bool(rec.get("has_visual_samples") or rec.get("source_inputs")),
                    source_digest=rec.get("source_digest"),
                    gpu_name=rec.get("gpu_name"),
                )
            )

    staged_relpaths = sorted({str(it["gif_relpath"]) for it in incoming if it.get("gif_relpath")})
    existing_doc = load_leaderboard_media_index(root)
    merged_items = _merge_items(existing_doc.get("items", []), incoming)
    payload = {
        "schema_version": 1,
        "schema": MEDIA_INDEX_SCHEMA,
        "generated_at": _utc_now(),
        "qualification_input": False,
        "hf_repo": DEFAULT_HF_REPO,
        "media_prefix": DEFAULT_MEDIA_PREFIX,
        "items": merged_items,
    }
    if write_index and copied > 0:
        atomic_json(root / DEFAULT_MEDIA_INDEX_RELPATH, payload)
    return {
        "copied": copied,
        "skipped": skipped,
        "total_items": len(merged_items),
        "staged_relpaths": staged_relpaths,
    }


def upload_staged_media_to_hf(
    root: Path,
    staging_media_dir: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    dry_run: bool = False,
    commit_message: str | None = None,
    only_relpaths: set[str] | None = None,
) -> dict[str, Any]:
    """Upload staged `media/forge/leaderboard/**/*.gif` files to the HF dataset repo."""
    root = Path(root).resolve()
    staging_media_dir = Path(staging_media_dir).resolve()
    doc = load_leaderboard_media_index(root)

    upload_files: list[tuple[str, Path, int]] = []
    for item in doc.get("items", []):
        rel = str(item.get("gif_relpath") or "")
        if not rel.startswith(LEADERBOARD_MEDIA_SUBDIR + "/"):
            continue
        if only_relpaths is not None and rel not in only_relpaths:
            continue
        local = staging_media_dir / rel
        if not local.is_file():
            continue
        if _file_sha256(local) != item.get("gif_sha256"):
            raise ValueError(f"Staged GIF SHA-256 mismatch for {rel}")
        upload_files.append((rel, local, local.stat().st_size))

    total_bytes = sum(sz for _, _, sz in upload_files)
    result: dict[str, Any] = {
        "repo": repo,
        "dry_run": dry_run,
        "file_count": len(upload_files),
        "total_bytes": total_bytes,
        "files": [rel for rel, _, _ in upload_files],
    }
    if dry_run or not upload_files:
        return result

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    existing_remote = set(api.list_repo_files(repo, repo_type="dataset"))
    # Always upload all staged files (overwrite ensures exact SHA-256 match with manifest).
    operations = [
        CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(local))
        for rel, local, _ in sorted(upload_files)
    ]
    msg = commit_message or f"Upload {len(operations)} actual-training leaderboard GIFs"
    info = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=operations,
        commit_message=msg,
    )
    result["commit_oid"] = getattr(info, "oid", None)
    result["previously_present"] = sum(1 for rel, _, _ in upload_files if rel in existing_remote)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)

    ea = sub.add_parser("export-attempt", help="Export actual-training GIFs for one certified attempt directory")
    ea.add_argument("attempt_dir", type=Path)
    ea.add_argument("--output", type=Path, required=True)

    cl = sub.add_parser("collect-local", help="Collect existing HF media and export all local selected-cohort GIFs")
    cl.add_argument("--staging-dir", type=Path, default=Path("runs/forge/leaderboard_media_staging"))
    cl.add_argument("--no-write-index", action="store_true")

    ing = sub.add_parser("ingest-rendered", help="Ingest rendered GIF directories (e.g., Colab downloads) into index")
    ing.add_argument("--source-dir", type=Path, action="append", required=True)
    ing.add_argument("--staging-dir", type=Path, default=Path("runs/forge/leaderboard_media_staging"))
    ing.add_argument("--cohort-match", default="colab_replay_media")

    up = sub.add_parser("upload-hf", help="Upload staged leaderboard GIFs to HF dataset")
    up.add_argument("--staging-dir", type=Path, default=Path("runs/forge/leaderboard_media_staging"))
    up.add_argument("--repo", default=DEFAULT_HF_REPO)
    up.add_argument("--dry-run", action="store_true")
    up.add_argument("--commit-message", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "export-attempt":
        out = export_attempt_media(args.attempt_dir, args.output)
        print(json.dumps({"attempt": args.attempt_dir.name, "exported": len(out)}, indent=2))
        return 0
    if args.command == "collect-local":
        staging = args.staging_dir if args.staging_dir.is_absolute() else (root / args.staging_dir)
        out = collect_existing_and_local_media(root, staging, write_index=not args.no_write_index)
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "ingest-rendered":
        staging = args.staging_dir if args.staging_dir.is_absolute() else (root / args.staging_dir)
        sources = [p if p.is_absolute() else (root / p) for p in args.source_dir]
        out = ingest_rendered_media_dirs(root, sources, staging, cohort_match=args.cohort_match)
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "upload-hf":
        staging = args.staging_dir if args.staging_dir.is_absolute() else (root / args.staging_dir)
        out = upload_staged_media_to_hf(
            root, staging, repo=args.repo, dry_run=args.dry_run, commit_message=args.commit_message
        )
        print(json.dumps(out, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
