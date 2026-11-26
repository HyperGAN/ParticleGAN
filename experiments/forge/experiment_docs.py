"""Hybrid Forge experiment annotation extractor, catalog builder, and Markdown renderer.

Combines four strictly read-only metadata layers without ever mutating pinned
``configs/forge/tasks/*.json`` contracts or SHA-256-pinned Python benchmark files:

1. Live task contract from ``configs/forge/tasks/<task_id>.json``.
2. Editorial documentation from inline ``task["documentation"]`` (for new unpinned tasks)
   and/or sidecar ``configs/forge/task-documentation/<task_id>.json`` (keeping all
   existing SHA-256 / stable-hash-pinned task files byte-identical).
3. Read-only Python AST inspection (``ast.parse``) of ``execution.host_source`` and
   ``evaluation.sources`` for module docstrings, ``EXPERIMENT_DOC``, ``SPECS``,
   ``HOST_DEFINITIONS``, and ``THRESHOLDS``.
4. Recorded family outcomes from ``reports/forge/technique-inventory.json``,
   historical problem diagnoses from ``reports/forge/problem-definitions.json``,
   Four-Laws diagnostic readouts, and published Hugging Face media URLs.

Public API:
- ``load_laws_registry(root) -> dict``
- ``extract_python_annotations(path) -> dict``
- ``build_experiment_catalog(root) -> dict``
- ``write_experiment_docs(root) -> dict``

CLI:
- ``python -m experiments.forge.experiment_docs``
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import re
from typing import Any

from .experiment_svg import BADGE_TEXT, infer_diagram_kind, write_task_svg
from .family_reports import _selected_cohorts, _selected_row
from .hf_publish import DEFAULT_HF_REPO, _collect_media_index, _gif_to_media_relpath
from .leaderboard_contract import FIXED_TEST_LIST

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_PORTAL_URL = "https://hypergan.github.io/ParticleGAN/"
FULL_SUITE_LIST_REL = FIXED_TEST_LIST
LAWS_REGISTRY_REL = Path("configs/forge/laws.json")
TASK_DOCS_DIR_REL = Path("configs/forge/task-documentation")
FAMILY_DOCS_DIR_REL = Path("configs/forge/family-documentation")
EXPERIMENTS_DIR_REL = Path("reports/forge/experiments")
LAWS_HUB_REL = Path("reports/forge/LAWS_HUB.md")
INVENTORY_JSON_REL = Path("reports/forge/technique-inventory.json")
PROBLEM_DEFS_REL = Path("reports/forge/problem-definitions.json")
HF_RECEIPT_REL = Path("reports/forge/publications/inventory-latest.json")

# Fallback schema matching configs/forge/laws.json (schema v1) when not yet merged.
DEFAULT_LAWS_REGISTRY: dict[str, Any] = {
    "schema_version": 1,
    "epistemic_status": "provisional_evolving",
    "epigraph": (
        "Our goal is not to preserve the four laws. Our goal is to discover the laws of "
        "learning, even if doing so ultimately requires abandoning the current four."
    ),
    "epistemic_note": (
        "This law taxonomy is a living scientific framework derived from current mathematical "
        "and empirical audits. Individual laws may be refined, split, merged, or replaced as "
        "new counterexamples and evidence emerge."
    ),
    "doc_path": "docs/four-laws.md",
    "laws": [
        {
            "id": "law1",
            "short_label": "Law 1",
            "title": "Representational Geometry",
            "domain_level": "Representational Geometry",
            "question": "Can the system represent the required probability measure efficiently and faithfully?",
            "doc_anchor": "law-1--representational-geometry",
            "status": "active",
            "sub_laws": [
                {
                    "id": "law1a",
                    "short_label": "Law 1A",
                    "title": "Anisotropic Local Metric & Unbiased Sketching",
                    "doc_anchor": "law-1--representational-geometry",
                },
                {
                    "id": "law1b",
                    "short_label": "Law 1B",
                    "title": "Discrete-Continuous Phase Continuity",
                    "doc_anchor": "law-1--representational-geometry",
                },
            ],
        },
        {
            "id": "law2",
            "short_label": "Law 2",
            "title": "Potential Geometry",
            "domain_level": "Potential Geometry",
            "question": "Does the critic communicate useful learning forces with appropriate geometric control and computational efficiency?",
            "doc_anchor": "law-2--potential-geometry",
            "status": "active",
            "sub_laws": [],
        },
        {
            "id": "law3",
            "short_label": "Law 3",
            "title": "Transport Identifiability",
            "domain_level": "Transport Identifiability",
            "question": "Does the learning system possess enough information to determine appropriate transport?",
            "doc_anchor": "law-3--transport-identifiability",
            "status": "active",
            "sub_laws": [
                {
                    "id": "law3a",
                    "short_label": "Law 3A",
                    "title": "Unconditional W1 Ray-Length & Screening Identifiability",
                    "doc_anchor": "law-3--transport-identifiability",
                },
                {
                    "id": "law3b",
                    "short_label": "Law 3B",
                    "title": "Conditional Fiber & Stratum Identifiability",
                    "doc_anchor": "law-3--transport-identifiability",
                },
            ],
        },
        {
            "id": "law4",
            "short_label": "Law 4",
            "title": "Functional and Game Dynamics",
            "domain_level": "Functional and Game Dynamics",
            "question": "Can the learning system realize useful functional changes and exhibit appropriate coupled learning dynamics?",
            "doc_anchor": "law-4--functional-and-game-dynamics",
            "status": "active",
            "sub_laws": [
                {
                    "id": "law4_1",
                    "short_label": "Law 4.1",
                    "title": "Functional Realization",
                    "doc_anchor": "law-4--functional-and-game-dynamics",
                },
                {
                    "id": "law4_2",
                    "short_label": "Law 4.2",
                    "title": "Pathology-Driven Spectral Conditioning",
                    "doc_anchor": "law-4--functional-and-game-dynamics",
                },
                {
                    "id": "law4_3",
                    "short_label": "Law 4.3",
                    "title": "Displacement & Kreiss Control",
                    "doc_anchor": "law-4--functional-and-game-dynamics",
                },
                {
                    "id": "law4_4",
                    "short_label": "Law 4.4",
                    "title": "Trajectory Scale vs. Settling",
                    "doc_anchor": "law-4--functional-and-game-dynamics",
                },
                {
                    "id": "law4_5",
                    "short_label": "Law 4.5",
                    "title": "Coupled Minimax Contraction",
                    "doc_anchor": "law-4--functional-and-game-dynamics",
                },
            ],
        },
        {
            "id": "law5",
            "short_label": "Law 5",
            "title": "Observation & Statistical-Metric Identifiability",
            "domain_level": "Observation & Statistical-Metric Identifiability",
            "question": "Is the target probability measure identifiable through the observation channel and statistical metric from finite batches?",
            "doc_anchor": "law-5--observation--statistical-metric-identifiability",
            "status": "provisional",
            "sub_laws": [
                {
                    "id": "law5a",
                    "short_label": "Law 5A",
                    "title": "Partial & Corrupted Observation Injectivity",
                    "doc_anchor": "law-5--observation--statistical-metric-identifiability",
                },
                {
                    "id": "law5b",
                    "short_label": "Law 5B",
                    "title": "Intrinsic & Feature-Space Metric Geometry",
                    "doc_anchor": "law-5--observation--statistical-metric-identifiability",
                },
                {
                    "id": "law5c",
                    "short_label": "Law 5C",
                    "title": "Conditional Fiber Deconfounding (Unifies Law 3B)",
                    "doc_anchor": "law-5--observation--statistical-metric-identifiability",
                },
            ],
        },
    ],
}

EDITORIAL_FIELDS = frozenset({
    "title",
    "laws",
    "edge_cases",
    "intuition",
    "failure_mode",
    "protocol_notes",
    "diagram_kind",
    "custom_mermaid",
})


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_laws_registry(root: Path | str = ROOT) -> dict[str, Any]:
    """Load the evolving laws taxonomy dynamically from ``configs/forge/laws.json``."""
    root = Path(root)
    path = root / LAWS_REGISTRY_REL
    if path.is_file():
        data = _read_json(path)
        if isinstance(data, dict) and isinstance(data.get("laws"), list) and data["laws"]:
            return data
    return json.loads(json.dumps(DEFAULT_LAWS_REGISTRY))


def _law_lookup(registry: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return (id -> law_or_sublaw_meta, sublaw_id -> parent_law_id)."""
    by_id: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str] = {}
    doc_path = registry.get("doc_path", "docs/four-laws.md")
    for law in registry.get("laws", []):
        lid = str(law["id"])
        by_id[lid] = {
            "id": lid,
            "short_label": law.get("short_label", lid),
            "title": law.get("title", lid),
            "question": law.get("question", ""),
            "status": law.get("status", "active"),
            "doc_path": doc_path,
            "doc_anchor": law.get("doc_anchor", ""),
            "parent_id": None,
            "is_sublaw": False,
        }
        parent_of[lid] = lid
        for sub in law.get("sub_laws", []) or []:
            sid = str(sub["id"])
            by_id[sid] = {
                "id": sid,
                "short_label": sub.get("short_label", sid),
                "title": sub.get("title", sid),
                "question": sub.get("question", law.get("question", "")),
                "status": law.get("status", "active"),
                "doc_path": doc_path,
                "doc_anchor": sub.get("doc_anchor", law.get("doc_anchor", "")),
                "parent_id": lid,
                "is_sublaw": True,
            }
            parent_of[sid] = lid
    return by_id, parent_of


def _safe_eval_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    """Safely evaluate literal expressions and simple constant references without executing code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"unresolved name {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval_ast(node.operand, env)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _safe_eval_ast(node.left, env)
        right = _safe_eval_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div) and right != 0:
            return left / right
        raise ValueError("unsupported binary op")
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_safe_eval_ast(elt, env) for elt in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Set):
        return [_safe_eval_ast(elt, env) for elt in node.elts]
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                spread = _safe_eval_ast(v, env)
                if isinstance(spread, dict):
                    out.update(spread)
            else:
                out[_safe_eval_ast(k, env)] = _safe_eval_ast(v, env)
        return out
    raise ValueError(f"unsupported AST node {type(node).__name__}")


def extract_python_annotations(path: Path | str) -> dict[str, Any]:
    """Read-only AST extraction of module docstring, ``EXPERIMENT_DOC``, ``SPECS``,
    ``HOST_DEFINITIONS``, and ``THRESHOLDS`` from a Python benchmark file.

    Never imports or executes the file, preserving all SHA-256 source pins.
    """
    file_path = Path(path)
    result: dict[str, Any] = {
        "path": file_path.as_posix(),
        "docstring": None,
        "experiment_doc": None,
        "specs": {},
        "host_definitions": {},
        "thresholds": {},
        "extracted_symbols": [],
    }
    if not file_path.is_file():
        return result
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except (OSError, SyntaxError):
        return result

    doc = ast.get_docstring(tree)
    if doc:
        result["docstring"] = doc.strip()
        result["extracted_symbols"].append("__doc__")

    # Collect simple top-level scalar/tuple constants so dicts referencing e.g.
    # _RARE1PCT_OUT_BOUND or SADDLE_SHIFT_STEP evaluate cleanly.
    const_env: dict[str, Any] = {}
    targets_of_interest = {"EXPERIMENT_DOC", "SPECS", "HOST_DEFINITIONS", "THRESHOLDS"}
    for stmt in tree.body:
        name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name = stmt.target.id
            value_node = stmt.value
        if not name or value_node is None:
            continue
        try:
            evaluated = _safe_eval_ast(value_node, const_env)
        except Exception:
            # For partial dict extraction (e.g., if one key has a call expression)
            if name in targets_of_interest and isinstance(value_node, ast.Dict):
                partial: dict[Any, Any] = {}
                for k_node, v_node in zip(value_node.keys, value_node.values):
                    if k_node is None:
                        continue
                    try:
                        partial[_safe_eval_ast(k_node, const_env)] = _safe_eval_ast(v_node, const_env)
                    except Exception:
                        continue
                evaluated = partial if partial else None
            else:
                evaluated = None
        if evaluated is None:
            continue
        if isinstance(evaluated, (int, float, str, bool, tuple, list)):
            const_env[name] = evaluated
        if name == "EXPERIMENT_DOC" and isinstance(evaluated, dict):
            result["experiment_doc"] = evaluated
            result["extracted_symbols"].append("EXPERIMENT_DOC")
        elif name == "SPECS" and isinstance(evaluated, dict):
            result["specs"] = evaluated
            result["extracted_symbols"].append("SPECS")
        elif name == "HOST_DEFINITIONS" and isinstance(evaluated, dict):
            result["host_definitions"] = evaluated
            result["extracted_symbols"].append("HOST_DEFINITIONS")
        elif name == "THRESHOLDS" and isinstance(evaluated, dict):
            result["thresholds"] = evaluated
            result["extracted_symbols"].append("THRESHOLDS")
    return result


def validate_task_documentation(doc: dict[str, Any], valid_law_ids: set[str] | None = None) -> None:
    """Validate an editorial task documentation block."""
    if not isinstance(doc, dict):
        raise ValueError("task documentation must be a JSON object")
    unknown = set(doc) - EDITORIAL_FIELDS - {"task_id", "schema_version"}
    if unknown:
        raise ValueError(f"unknown documentation fields: {sorted(unknown)}")
    for field in ("title", "intuition", "failure_mode"):
        val = doc.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"documentation.{field} must be a non-empty string")
    laws = doc.get("laws")
    if not isinstance(laws, list) or not all(isinstance(x, str) and x for x in laws):
        raise ValueError("documentation.laws must be a list of non-empty strings")
    if valid_law_ids is not None:
        bad = [x for x in laws if x not in valid_law_ids]
        if bad:
            raise ValueError(f"unknown law IDs in documentation.laws: {bad}")
    edge_cases = doc.get("edge_cases")
    if not isinstance(edge_cases, list) or not all(isinstance(x, str) and x for x in edge_cases):
        raise ValueError("documentation.edge_cases must be a list of non-empty strings")
    notes = doc.get("protocol_notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("documentation.protocol_notes must be a string or null")
    cm = doc.get("custom_mermaid")
    if cm is not None and (not isinstance(cm, str) or not cm.strip()):
        raise ValueError("documentation.custom_mermaid must be a non-empty string or null")


def _load_sidecar_documentation(root: Path) -> dict[str, tuple[dict[str, Any], str]]:
    docs: dict[str, tuple[dict[str, Any], str]] = {}
    doc_dir = root / TASK_DOCS_DIR_REL
    if not doc_dir.is_dir():
        return docs
    for path in sorted(doc_dir.glob("*.json")):
        payload = _read_json(path)
        tid = str(payload.get("task_id") or path.stem)
        rel = path.relative_to(root).as_posix()
        docs[tid] = (payload, rel)
    return docs


def _load_views(root: Path) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    views_dir = root / "configs/forge/views"
    if not views_dir.is_dir():
        return views
    for path in sorted(views_dir.glob("*.json")):
        view = _read_json(path)
        vid = str(view.get("id") or path.stem)
        views[vid] = view
    return views


def _load_family_results(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], str]:
    """Load per-task selected family outcomes & DCGAN throughput rows from technique-inventory.json."""
    per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scaling_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

    hf_repo = DEFAULT_HF_REPO
    receipt_path = root / HF_RECEIPT_REL
    if receipt_path.is_file():
        try:
            receipt = _read_json(receipt_path)
            if isinstance(receipt.get("repo"), str) and receipt["repo"]:
                hf_repo = receipt["repo"]
        except Exception:
            pass

    inv_path = root / INVENTORY_JSON_REL
    if not inv_path.is_file():
        return per_task, scaling_by_task, hf_repo
    publication = _read_json(inv_path)
    progress = publication.get("family_progress")
    if not isinstance(progress, dict):
        return per_task, scaling_by_task, hf_repo

    media_by_family_task, media_by_attempt_task = _collect_media_index(progress)
    try:
        from .leaderboard_media import index_media_lookup

        idx_fam, idx_att = index_media_lookup(root)
        for key, entry in idx_fam.items():
            rel = _gif_to_media_relpath(entry.get("gif_relpath"))
            if rel and (key not in media_by_family_task or entry.get("cohort_match") == "selected_cohort"):
                media_by_family_task[key] = rel
        for akey, entry in idx_att.items():
            rel = _gif_to_media_relpath(entry.get("gif_relpath"))
            if rel:
                media_by_attempt_task.setdefault(akey, rel)
    except Exception:
        pass
    from .views import build_diagnostic_pass_index, compatible_diagnostic_passes

    diag_index = build_diagnostic_pass_index(root)
    # Build family metadata list (deduplicated by family_id, preferring CUDA primary cohort)
    seen_family: set[str] = set()
    selected_candidate_by_family: dict[str, str] = {}
    for fam, cohort in _selected_cohorts(progress):
        fam_id = str(fam["id"])
        if fam_id in seen_family:
            continue
        seen_family.add(fam_id)
        row = _selected_row(publication, cohort)
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            selected_candidate_by_family[fam_id] = candidate_id
        cand_rev = str(row.get("candidate_revision") or "")
        bindings = row.get("bindings") if isinstance(row.get("bindings"), dict) else {}
        source_digest = str(bindings.get("source_digest") or "")
        diag_passes = compatible_diagnostic_passes(
            root,
            candidate_id=candidate_id,
            candidate_revision=cand_rev,
            source_digest=source_digest,
            _index=diag_index,
        )
        anchor = str(cohort.get("anchor") or fam.get("current_cohort_anchor") or "")
        fam_label = str(fam.get("label") or fam_id)
        page = str(fam.get("page") or f"reports/forge/families/{fam_id}.md")
        cohort_tasks = cohort.get("tasks") or {}
        for task_id, cell in cohort_tasks.items():
            raw_status = str(cell.get("status") or "UNKNOWN")
            coverage_state = cell.get("coverage_status") or cell.get("attempt_coverage_status")
            diag_pass = diag_passes.get(task_id) if raw_status in {"UNKNOWN", "NOT_RUN"} else None
            if raw_status == "UNKNOWN":
                run_attempt = cell.get("run_attempt") or {}
                if cell.get("evidence_correction"):
                    display_status = "DIAGNOSTIC"
                elif diag_pass is not None:
                    display_status = "DIAG_PASS"
                elif run_attempt.get("status"):
                    display_status = str(run_attempt["status"]).upper()
                elif coverage_state:
                    display_status = str(coverage_state).upper().replace("_", " ")
                else:
                    display_status = "NOT MEASURED"
            elif raw_status == "NOT_RUN" and diag_pass is not None:
                display_status = "DIAG_PASS"
            else:
                display_status = raw_status.upper()

            attempt_id = cell.get("attempt_id") or (cell.get("run_attempt") or {}).get("attempt_id")
            gif_relpath = None
            for artifact in cell.get("training_media") or []:
                gif_relpath = _gif_to_media_relpath(artifact.get("gif"))
                if gif_relpath:
                    break
            if gif_relpath is None and attempt_id:
                gif_relpath = media_by_attempt_task.get((str(attempt_id), task_id))
            if gif_relpath is None:
                gif_relpath = media_by_family_task.get((fam_id, task_id))
            gif_url = (
                f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{gif_relpath}"
                if gif_relpath
                else None
            )

            # Extract compact metric summary
            metrics_str = _summarize_cell_metrics(cell)
            per_task[task_id].append({
                "family_id": fam_id,
                "family_label": fam_label,
                "candidate_id": candidate_id,
                "status": display_status,
                "recorded_status": raw_status,
                "anchor": anchor,
                "page": page,
                "metrics": metrics_str,
                "gif_relpath": gif_relpath,
                "gif_url": gif_url,
                **({"diagnostic_pass": diag_pass} if diag_pass else {}),
            })

    # Scaling performance records for dcgan_performance_* tasks
    fam_labels = {str(f["id"]): str(f.get("label") or f["id"]) for f in progress.get("families", []) if isinstance(f, dict)}
    fam_pages = {str(f["id"]): str(f.get("page") or f"reports/forge/families/{f['id']}.md") for f in progress.get("families", []) if isinstance(f, dict)}
    sp = publication.get("scaling_performance") or {}
    seen_scaling_keys: set[tuple[str, str, str]] = set()
    for rec in sp.get("records") or []:
        if not isinstance(rec, dict):
            continue
        tid = str(rec.get("task_id") or "")
        if not tid:
            continue
        res = rec.get("result") if isinstance(rec.get("result"), dict) else rec
        metrics = res.get("metrics") if isinstance(res.get("metrics"), dict) else {}
        hw = res.get("hardware") if isinstance(res.get("hardware"), dict) else {}
        step_ms = metrics.get("update_ms_median")
        if step_ms is None and isinstance(metrics.get("step_ms"), dict):
            step_ms = metrics["step_ms"].get("median")
        img_s = metrics.get("images_per_second")
        peak_mib = metrics.get("peak_allocated_mib")
        if peak_mib is None and isinstance(metrics.get("gpu_memory_mib"), dict):
            peak_mib = metrics["gpu_memory_mib"].get("peak_allocated")
        fid = str(rec.get("trainer_family") or "")
        cid = str(rec.get("candidate_id") or res.get("candidate_id") or "")
        seen_scaling_keys.add((tid, fid, cid))
        scaling_by_task[tid].append({
            "family_id": fid,
            "family_label": fam_labels.get(fid, fid),
            "page": fam_pages.get(fid, f"reports/forge/families/{fid}.md"),
            "candidate_id": cid,
            "status": str(res.get("benchmark_status") or res.get("status") or "COMPLETE").upper(),
            "device": str(hw.get("name") or hw.get("type") or "CUDA"),
            "step_ms_median": step_ms,
            "images_per_second": img_s,
            "peak_allocated_mib": peak_mib,
        })

    # Include provenance-bound solo measurements stored in reports/forge/scaling-performance.json
    perf_manifest_path = root / "reports/forge/scaling-performance.json"
    if perf_manifest_path.is_file():
        try:
            perf_doc = _read_json(perf_manifest_path)
            for item in perf_doc.get("measurements") or []:
                if not isinstance(item, dict) or item.get("contended") is True:
                    continue
                tid = str(item.get("task_id") or "")
                fid = str(item.get("trainer_family") or item.get("family_id") or "")
                cid = str(item.get("candidate_id") or "")
                if not tid or not fid or (tid, fid, cid) in seen_scaling_keys:
                    continue
                seen_scaling_keys.add((tid, fid, cid))
                metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
                hw = item.get("hardware") if isinstance(item.get("hardware"), dict) else {}
                scaling_by_task[tid].append({
                    "family_id": fid,
                    "family_label": fam_labels.get(fid, fid),
                    "page": fam_pages.get(fid, f"reports/forge/families/{fid}.md"),
                    "candidate_id": cid,
                    "status": str(item.get("status") or "COMPLETE").upper(),
                    "device": str(hw.get("name") or item.get("hardware_gpu") or "CUDA"),
                    "step_ms_median": item.get("median_ms") if item.get("median_ms") is not None else metrics.get("update_ms_median"),
                    "images_per_second": item.get("images_per_second") if item.get("images_per_second") is not None else metrics.get("images_per_second"),
                    "peak_allocated_mib": item.get("peak_allocated_mib") if item.get("peak_allocated_mib") is not None else metrics.get("peak_allocated_mib"),
                })
        except Exception:
            pass

    # Deduplicate scaling records per (task_id, family_id), preferring the currently selected candidate.
    for tid, rows in list(scaling_by_task.items()):
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_family[str(r.get("family_id") or "")].append(r)
        filtered: list[dict[str, Any]] = []
        for fid, fam_rows in by_family.items():
            sel_cid = selected_candidate_by_family.get(fid)
            if sel_cid:
                exact = [r for r in fam_rows if str(r.get("candidate_id") or "") == sel_cid]
                if exact:
                    filtered.append(exact[0])
                    continue
            filtered.extend(fam_rows)
        scaling_by_task[tid] = filtered

    return per_task, scaling_by_task, hf_repo


def _resolve_candidate_family_map(root: Path) -> dict[str, str]:
    """Map candidate_id -> trainer_family from configs/forge/ideas/*.json (read-only)."""
    mapping: dict[str, str] = {}
    ideas_dir = root / "configs/forge/ideas"
    if not ideas_dir.is_dir():
        return mapping
    for path in sorted(ideas_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        cid = str(doc.get("id") or path.stem)
        fam = str(doc.get("trainer_family") or "").strip()
        if not fam:
            if cid.startswith("minimal"):
                fam = "minimal"
            elif cid.startswith("unified"):
                fam = "unified"
            elif cid.startswith("rhbi"):
                fam = "rhbi"
            elif cid.startswith("empf"):
                fam = "empf"
            elif cid.startswith("bispf"):
                fam = "bispf"
            elif cid.startswith("atlas"):
                fam = "atlas"
            elif cid.startswith("axiom"):
                fam = "axiom"
        if fam:
            mapping[cid] = fam
    return mapping


def _format_energy_cell(joules_total: float | None, joules_to_pass: float | None, status: str) -> str:
    if status == "PASS" and isinstance(joules_to_pass, (int, float)) and joules_to_pass >= 0:
        if joules_to_pass >= 1000.0:
            return f"{joules_to_pass / 1000.0:.2f} kJ (J-to-PASS)"
        return f"{joules_to_pass:.0f} J (J-to-PASS)"
    if isinstance(joules_total, (int, float)) and joules_total > 0:
        if joules_total >= 1000.0:
            return f"{joules_total / 1000.0:.2f} kJ"
        return f"{joules_total:.0f} J"
    return "—"


def _load_four_laws_readouts(
    root: Path,
    fl_task_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Scan compact Forge readouts and technique-receipts for four-laws-adherence-v1 diagnostic results."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fam_map = _resolve_candidate_family_map(root)
    best_by_key: dict[tuple[str, str, str], tuple[tuple[int, int, str], dict[str, Any]]] = {}

    def _consider(entry: dict[str, Any], tie_key: str) -> None:
        tid = entry["task_id"]
        cand = entry["candidate_id"]
        gpu = entry.get("gpu_name") or ""
        status = entry["status"]
        # Prefer definitive PASS/FAIL over INCOMPLETE/TIMEOUT, then hardware NVML energy, then newest path.
        status_rank = 2 if status in {"PASS", "FAIL"} else (1 if status == "BLOCKED" else 0)
        energy_rank = 1 if isinstance(entry.get("joules_total"), (int, float)) else 0
        rank = (status_rank, energy_rank, tie_key)
        key = (tid, cand, gpu)
        prev = best_by_key.get(key)
        if prev is None or rank > prev[0]:
            best_by_key[key] = (rank, entry)

    records_dir = root / "reports/forge/records"
    if records_dir.is_dir():
        for path in sorted(records_dir.glob("readout-*.json"), reverse=True):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "four-laws-adherence" not in text:
                continue
            try:
                doc = json.loads(text)
            except Exception:
                continue
            scope = (doc.get("study") or {}).get("scope") if isinstance(doc.get("study"), dict) else {}
            view_id = doc.get("view") or (scope or {}).get("view")
            if view_id not in {"four-laws-adherence-v1", "four-laws-adherence-v2"}:
                continue
            cand = str(doc.get("candidate_id") or (doc.get("candidate") or {}).get("id") or path.stem)
            fam = str(doc.get("trainer_family") or (doc.get("candidate") or {}).get("trainer_family") or fam_map.get(cand, ""))
            default_gpu = str((scope or {}).get("cuda_model") or "")
            for tr in doc.get("task_results") or []:
                if not isinstance(tr, dict):
                    continue
                tid = str(tr.get("task_id") or "")
                if not tid:
                    continue
                status = str(tr.get("gate_status") or tr.get("status") or "UNKNOWN").upper()
                metrics = tr.get("metrics") if isinstance(tr.get("metrics"), dict) else {}
                cost = tr.get("cost") if isinstance(tr.get("cost"), dict) else {}
                en = cost.get("energy") if isinstance(cost.get("energy"), dict) else {}
                gpu = str(en.get("gpu_name") or default_gpu or "")
                mm = metrics.get("min_margin")
                mm_val = float(mm) if isinstance(mm, (int, float)) and not isinstance(mm, bool) else None
                jt = en.get("joules_total")
                jt_val = float(jt) if isinstance(jt, (int, float)) and not isinstance(jt, bool) else None
                jtp = en.get("joules_to_sustained_pass")
                if jtp is None:
                    jtp = en.get("joules_to_first_pass")
                jtp_val = float(jtp) if isinstance(jtp, (int, float)) and not isinstance(jtp, bool) else None
                _consider(
                    {
                        "task_id": tid,
                        "family_id": fam,
                        "candidate_id": cand,
                        "gpu_name": gpu,
                        "status": status,
                        "min_margin": mm_val,
                        "joules_total": jt_val,
                        "joules_to_pass": jtp_val,
                        "metrics": _summarize_metrics_dict(metrics),
                        "readout": path.relative_to(root).as_posix(),
                    },
                    f"1:{path.name}",
                )

    receipts_dir = root / "reports/forge/technique-receipts"
    if receipts_dir.is_dir():
        allowed = fl_task_ids or set()
        for path in sorted(receipts_dir.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            cand = str(doc.get("candidate_id") or "")
            if not cand:
                continue
            campaign_id = str(doc.get("campaign_id") or "")
            is_colab = campaign_id.startswith("colab-catchup-")
            fam = fam_map.get(cand, "")
            for tr in doc.get("task_results") or []:
                if not isinstance(tr, dict):
                    continue
                tid = str(tr.get("task_id") or "")
                if not tid:
                    continue
                is_fl_probe = tid.startswith("fl_law")
                if not is_fl_probe and not (tid in allowed and is_colab):
                    continue
                status = str(tr.get("gate_status") or tr.get("status") or "UNKNOWN").upper()
                metrics = tr.get("metrics") if isinstance(tr.get("metrics"), dict) else {}
                cost = tr.get("cost") if isinstance(tr.get("cost"), dict) else {}
                en = cost.get("energy") if isinstance(cost.get("energy"), dict) else {}
                gpu = str(en.get("gpu_name") or ("NVIDIA L4" if "-l4-" in campaign_id else ""))
                mm = metrics.get("min_margin")
                mm_val = float(mm) if isinstance(mm, (int, float)) and not isinstance(mm, bool) else None
                jt = en.get("joules_total")
                jt_val = float(jt) if isinstance(jt, (int, float)) and not isinstance(jt, bool) else None
                jtp = en.get("joules_to_sustained_pass")
                if jtp is None:
                    jtp = en.get("joules_to_first_pass")
                jtp_val = float(jtp) if isinstance(jtp, (int, float)) and not isinstance(jtp, bool) else None
                _consider(
                    {
                        "task_id": tid,
                        "family_id": fam,
                        "candidate_id": cand,
                        "gpu_name": gpu,
                        "status": status,
                        "min_margin": mm_val,
                        "joules_total": jt_val,
                        "joules_to_pass": jtp_val,
                        "metrics": _summarize_metrics_dict(metrics),
                        "readout": path.relative_to(root).as_posix(),
                    },
                    f"0:{path.name}",
                )

    for (_, _, _), (_, entry) in sorted(best_by_key.items()):
        by_task[entry["task_id"]].append(entry)
    for tid in by_task:
        by_task[tid].sort(
            key=lambda r: (
                0 if r["status"] == "PASS" else (1 if r["status"] == "FAIL" else 2),
                -(r["min_margin"] if isinstance(r["min_margin"], (int, float)) else -1e9),
                r["candidate_id"],
                r.get("gpu_name", ""),
            )
        )
    return by_task


def _summarize_metrics_dict(metrics: dict[str, Any]) -> str:
    if not isinstance(metrics, dict):
        return ""
    skip = {"cost", "step", "seconds", "sample_count", "ema", "component_mass", "target_mass",
            "component_counts", "component_covariance_errors", "component_core_covariance_errors",
            "component_core_eigen_ratios", "component_spill", "component_resolved",
            "quality_mode_fractions", "mode_fractions"}
    parts = []
    for key, val in metrics.items():
        if key in skip or isinstance(val, (dict, list, bool)):
            continue
        if isinstance(val, int):
            parts.append(f"{key}={val}")
        elif isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        if len(parts) >= 4:
            break
    return ", ".join(parts)


def _summarize_cell_metrics(cell: dict[str, Any]) -> str:
    metrics = cell.get("metrics")
    if isinstance(metrics, dict):
        summary = _summarize_metrics_dict(metrics)
        if summary:
            return summary
    reason = cell.get("reason") or cell.get("coverage_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()[:90]
    return ""


def _load_problem_definitions(root: Path) -> dict[str, dict[str, Any]]:
    path = root / PROBLEM_DEFS_REL
    if not path.is_file():
        return {}
    data = _read_json(path)
    out: dict[str, dict[str, Any]] = {}
    for entry in data.get("problems", []) or []:
        if isinstance(entry, dict) and isinstance(entry.get("task"), str):
            out[entry["task"]] = entry
    return out


def _load_api_contract_gifs(root: Path) -> dict[str, str]:
    """Map task_id -> repo-relative GIF path when an api_contract goal.gif exists."""
    gifs: dict[str, str] = {}
    base = root / "reports/toy_audit/api_contract"
    if not base.is_dir():
        return gifs
    for gif_path in sorted(base.glob("api-*/goal.gif")):
        slug = gif_path.parent.name[len("api-"):].replace("-", "_")
        gifs[slug] = gif_path.relative_to(root).as_posix()
    return gifs


def _load_family_law_mechanisms(root: Path) -> dict[str, dict[str, Any]]:
    """Read optional ``law_mechanisms`` from ``configs/forge/family-documentation/*.json``."""
    out: dict[str, dict[str, Any]] = {}
    fam_dir = root / FAMILY_DOCS_DIR_REL
    if not fam_dir.is_dir():
        return out
    registry_meta: dict[str, dict[str, Any]] = {}
    reg_path = root / "configs/forge/trainer-families.json"
    if reg_path.is_file():
        try:
            reg = _read_json(reg_path)
            for item in (reg.get("families") or []) + (reg.get("historical_families") or []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    registry_meta[item["id"]] = item
        except Exception:
            registry_meta = {}
    for path in sorted(fam_dir.glob("*.json")):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        fid = str(doc.get("family") or path.stem)
        mech = doc.get("law_mechanisms")
        if isinstance(mech, dict) and mech:
            rmeta = registry_meta.get(fid, {})
            label = str(doc.get("title") or doc.get("label") or rmeta.get("label") or fid)
            candidate_page = f"reports/forge/families/{fid}.md"
            if (root / candidate_page).is_file():
                page = candidate_page
                has_report_page = True
            else:
                parent_id = rmeta.get("reporting_family")
                parent_page = f"reports/forge/families/{parent_id}.md" if isinstance(parent_id, str) else ""
                if parent_page and (root / parent_page).is_file():
                    page = parent_page
                else:
                    page = path.relative_to(root).as_posix()
                has_report_page = False
            out[fid] = {
                "family_id": fid,
                "label": label,
                "page": page,
                "has_report_page": has_report_page,
                "law_mechanisms": mech,
            }
    return out


def _mermaid_quote(text: str) -> str:
    """Escape text for safe Mermaid double-quoted node labels."""
    clean = str(text).replace('"', "'").replace("\n", " ").replace("[", "(").replace("]", ")")
    return clean.strip()


def generate_mermaid_diagram(task: dict[str, Any], doc: dict[str, Any]) -> str:
    """Generate a deterministic, syntax-safe Mermaid flowchart for a Forge task."""
    if isinstance(doc.get("custom_mermaid"), str) and doc["custom_mermaid"].strip():
        return doc["custom_mermaid"].strip()

    tid = task["id"]
    adapter = task.get("adapter", "")
    ex = task.get("execution", {})
    ev = task.get("evaluation", {})
    hd = ex.get("host_definition", {}) if isinstance(ex.get("host_definition"), dict) else {}
    prior = ex.get("prior", {})
    res = ex.get("resources") or hd.get("resources") or {}
    model = ex.get("model") or hd.get("model") or {}
    if not isinstance(model, dict):
        model = {}

    prior_kind = prior.get("kind", "mog")
    prior_sigma = prior.get("sigma", "default")
    n_p = res.get("num_particles") or hd.get("particles")
    z_dim = res.get("z_dim") or hd.get("z_dim")
    prior_lbl = f"Prior: {prior_kind} (sigma={prior_sigma}"
    if n_p:
        prior_lbl += f", P={n_p}"
    if z_dim:
        prior_lbl += f", z={z_dim}"
    prior_lbl += ")"

    steps = ex.get("steps", hd.get("steps", "?"))
    eval_noise = ev.get("eval_output_noise", "clean")
    min_stable = ev.get("minimum_stable_checks")
    gate_lbl = f"Gate: {ev.get('kind', 'eval')} ({steps} steps, {eval_noise})"
    if min_stable:
        gate_lbl = f"Gate: {ev.get('kind', 'eval')} ({steps} steps, {min_stable} sustained)"
    if ex.get("continuation_of"):
        gate_lbl += f" <- continues {ex['continuation_of']}"

    # Specialized topologies
    if adapter == "clockfree_audit" or tid.startswith("clockfree_"):
        return "\n".join([
            "flowchart LR",
            f'    P["{_mermaid_quote(prior_lbl)}"] --> R["Reference Run (24 steps)"]',
            '    R --> C1["Perturb: Step Label (+8192)"]',
            '    R --> C2["Perturb: Horizon (4096)"]',
            '    R --> C3["Perturb: Eval Cadence"]',
            '    R --> C4["Perturb: Checkpoint Restart"]',
            '    C1 --> H["Exact State-Hash Parity Gate"]',
            '    C2 --> H',
            '    C3 --> H',
            '    C4 --> H',
        ])

    if adapter == "dcgan_performance_v1" or tid.startswith("dcgan_performance"):
        perf = ex.get("performance", {})
        sz = perf.get("image_size", 64)
        bs = perf.get("batch_size", 16)
        ch = perf.get("base_channels", 64)
        warm = perf.get("warmup_updates", 3)
        meas = perf.get("measured_updates", 10)
        return "\n".join([
            "flowchart LR",
            f'    P["{_mermaid_quote(prior_lbl)}"] --> G["{_mermaid_quote(f"DCGAN Generator ({sz}x{sz} RGB, W={ch})")}"]',
            f'    M["{_mermaid_quote(f"Mock Uniform -1..1 RGB Batch (B={bs})")}"] --> D["DCGAN Critic (4x4 Linear Head)"]',
            "    G --> D",
            f'    D --> T["{_mermaid_quote(f"Timing & Memory Telemetry ({warm}w + {meas}m updates)")}"]',
        ])

    if "ddgan" in tid:
        has_ae = "ae_ddgan" in tid
        lines = ["flowchart LR"]
        if has_ae:
            lines.append('    X["Real Data x0"] --> E["Particle AE Encoder E(x)"]')
            lines.append(f'    E --> P["{_mermaid_quote(prior_lbl)}"]')
        else:
            lines.append(f'    P["{_mermaid_quote(prior_lbl)}"] --> G["4-Step Reverse Chain G(x_t, z, t)"]')
        if has_ae:
            lines.append('    P --> G["4-Step Latent DDGAN Chain G(x_t, z, t)"]')
            lines.append('    G --> Dec["AE Decoder"]')
            lines.append('    Dec --> D["Conditional Critic D(x_{t-1}, x_t, t)"]')
        else:
            lines.append('    G --> D["Conditional Critic D(x_{t-1}, x_t, t)"]')
        lines.append(f'    D --> Gate["{_mermaid_quote(gate_lbl)}"]')
        return "\n".join(lines)

    if "misgan" in tid:
        has_imp = "imputer" in tid or "p80" in tid or "four_mode" in tid
        lines = [
            "flowchart LR",
            '    Obs["Incomplete 8D Data (x * m, m)"] --> Dx["Data Critic D_x (detached m)"]',
            '    Obs --> Dm["Mask Critic D_m"]',
            f'    P["{_mermaid_quote(prior_lbl)}"] --> Gx["Data Generator G_x"]',
            '    Gm["Mask Generator G_m"] --> Dm',
            '    Gx --> Dx',
        ]
        if has_imp:
            lines.append('    Obs --> Gi["Neural Imputer G_i(x*m, m, z_i)"]')
            lines.append('    Gi --> Di["Imputer Critic D_i"]')
            lines.append(f'    Di --> Gate["{_mermaid_quote(gate_lbl)}"]')
        else:
            lines.append(f'    Dx --> Gate["{_mermaid_quote(gate_lbl)}"]')
        return "\n".join(lines)

    if tid == "cyclegan_unpaired_constellation":
        return "\n".join([
            "flowchart LR",
            '    X["Domain X: 8-Mode Ring"] --> Gxy["Translator G_XY"]',
            '    Gxy --> Dy["Critic D_Y"]',
            '    Gxy --> Gyx["Translator G_YX"]',
            '    Y["Domain Y: 8-Mode Square"] --> Gyx',
            '    Gyx --> Dx["Critic D_X"]',
            f'    Dx --> Gate["{_mermaid_quote(gate_lbl)}"]',
            '    Dy --> Gate',
        ])

    if "bigan" in tid or "five_word" in tid:
        return "\n".join([
            "flowchart LR",
            f'    P["{_mermaid_quote(prior_lbl)}"] --> G["Generator G(z)"]',
            '    X["Real Data x"] --> E["Encoder E(x)"]',
            '    G --> D["Joint Critic D(x, z)"]',
            '    E --> D',
            '    X --> D',
            f'    D --> Gate["{_mermaid_quote(gate_lbl)}"]',
        ])

    if "vae" in tid or "ae_gan" in tid or "ae_reconstruction" in tid:
        return "\n".join([
            "flowchart LR",
            '    X["Real Data x"] --> E["Encoder E(x)"]',
            f'    E --> P["{_mermaid_quote(prior_lbl)}"]',
            '    P --> Dec["Decoder / Generator G(z)"]',
            '    Dec --> Rec["Reconstruction Loss"]',
            '    Dec --> D["Adversarial Critic D(x)"]',
            f'    D --> Gate["{_mermaid_quote(gate_lbl)}"]',
            '    Rec --> Gate',
        ])

    if tid == "fl_law42_deep_resbn_and_conv1x1":
        return "\n".join([
            "flowchart LR",
            f'    P["{_mermaid_quote(prior_lbl)}"] --> G1["Primary: 8-Block 2D Linear ResBN"]',
            '    P --> G2["Isomorphic Twin: 4D Conv2d 1x1"]',
            '    G1 --> D1["Critic D (Linear)"]',
            '    G2 --> D2["Critic D (Conv1x1)"]',
            f'    D1 --> Gate["{_mermaid_quote(gate_lbl)}"]',
            '    D2 --> Gate',
        ])

    # General generator -> target/shift/quantization -> critic -> gate pipeline
    gen_lbl = "Generator G(z)"
    if "nn.Identity" in str(model.get("generator", "")):
        gen_lbl = "Generator: nn.Identity (0 params)"
    elif tid == "grid100_mlp16":
        gen_lbl = "Generator: 16-Layer Skip-Free MLP"
    elif "res16" in tid:
        gen_lbl = "Generator: 16-Block ResNet + Tanh"
    elif tid == "fl_law2_groupnorm_and_vit_sprites":
        gen_lbl = "Generator: 32x32 RGB ResNet"
    elif tid == "fl_law41_trust_rare1pct_tanh":
        gen_lbl = "Generator: 4L MLP + 3.0*tanh"
    elif adapter == "transfer_image" or tid.startswith("img_") or tid == "interp_img_patch_shift":
        gen_lbl = "Generator: 8x8 Conv2d Image Net"
    elif "nash_overpowered_generator" in tid:
        gen_lbl = "Generator: 512x4 Overpowered (2.2x amp)"
    elif "nash_" in tid:
        gen_lbl = "Generator: 32x2 Compressed (0.08x)"
    elif model.get("hidden"):
        gen_lbl = f"Generator: {model.get('layers', 2)}L MLP (h={model.get('hidden')})"

    critic_lbl = "Critic D(x)"
    if tid == "fl_law2_groupnorm_and_vit_sprites":
        critic_lbl = "Critic: ResNet+GroupNorm + Patch-4 ViT"
    elif tid == "fl_law42_swap_deep_critic_16l":
        critic_lbl = "Critic: 16-Layer Skip-Free LeakyReLU (F0)"
    elif "pacgan8" in tid:
        critic_lbl = "Critic: PacGAN m=8 Packed D(x1..x8)"
    elif "pacgan" in tid:
        critic_lbl = "Critic: PacGAN m=4 Packed D(x1..x4)"
    elif tid == "cgan_overlapping_classes":
        critic_lbl = "Critic: Projection Discriminator D(x, y)"
    elif tid == "infogan_anisotropic_modes":
        critic_lbl = "Critic D(x) + Variational Q(c|x)"
    elif "nash_overpowered_generator" in tid:
        critic_lbl = "Critic: 32x2 Tiny Tilted D"
    elif "nash_" in tid:
        critic_lbl = "Critic: 512x4 Overpowered Tilted D"
    elif "smooth_silu" in str(model.get("discriminator_family", "")) or "smooth_silu" in str(model.get("family", "")):
        critic_lbl = "Critic: Smooth SiLU MLP"

    mid_node = None
    shift_step = ex.get("shift_step") or hd.get("shift_step")
    if shift_step:
        mid_node = f"Target Shift @ Step {shift_step}"
    elif "stability" in tid and "gaussian1d" in tid:
        mid_node = "Stationary Hold -> Target Shift -> Deadline"
    elif tid == "target_shift_recovery":
        mid_node = "Target Shift + Paired Frozen Control"
    elif tid.startswith("interp_"):
        mid_node = "Unseen Latent k-NN Chords / Barycenters"
    elif "mxfp8" in tid:
        mid_node = "MXFP8 Software-Ref Quantization"
    elif "nvfp4" in tid:
        mid_node = "NVFP4 Software-Ref Quantization"
    elif tid == "fl_law44_highd_sparse_prior_and_shock":
        mid_node = "10% Obs Mask + Step-200 1e4x Shock"

    lines = [
        "flowchart LR",
        f'    P["{_mermaid_quote(prior_lbl)}"] --> G["{_mermaid_quote(gen_lbl)}"]',
    ]
    if mid_node:
        lines.append(f'    G --> M["{_mermaid_quote(mid_node)}"]')
        lines.append(f'    M --> D["{_mermaid_quote(critic_lbl)}"]')
    else:
        lines.append(f'    G --> D["{_mermaid_quote(critic_lbl)}"]')
    lines.append(f'    D --> Gate["{_mermaid_quote(gate_lbl)}"]')
    return "\n".join(lines)


def _extract_thresholds(task: dict[str, Any], ast_thresholds: list[Any] | None = None) -> list[tuple[str, str, str]]:
    """Return normalized list of (metric, operator, bound) tuples for table rendering."""
    ev = task.get("evaluation", {})
    rows: list[tuple[str, str, str]] = []
    raw = ev.get("thresholds")
    if isinstance(raw, list) and raw:
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                rows.append((str(item[0]), str(item[1]), str(item[2])))
    elif isinstance(raw, dict):
        for k, v in raw.items():
            rows.append((str(k), "==", str(v)))

    if not rows and isinstance(ev.get("coverage_thresholds"), dict):
        for k, v in ev["coverage_thresholds"].items():
            op = ">=" if k.startswith("min_") else ("<=" if k.startswith("max_") else "==")
            rows.append((str(k), op, str(v)))
    if isinstance(ev.get("accuracy_limits"), dict):
        for k, v in ev["accuracy_limits"].items():
            rows.append((f"accuracy.{k}", "<=", str(v)))
    if "max_saturated_fraction" in ev:
        rows.append(("saturated_fraction", "<=", str(ev["max_saturated_fraction"])))
    if not rows and isinstance(ev.get("conditions"), list):
        for cond in ev["conditions"]:
            rows.append((f"parity.{cond}", "==", "exact state hash"))
    if not rows and ev.get("kind") == "image_training_performance":
        rows.append(("execution", "==", "COMPLETE (timing/memory telemetry; no quality gate)"))
    if not rows and isinstance(ast_thresholds, list):
        for item in ast_thresholds:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                rows.append((str(item[0]), str(item[1]), str(item[2])))
    return rows


def build_experiment_catalog(root: Path | str = ROOT) -> dict[str, Any]:
    """Build the unified metadata catalog for all base Forge tasks.

    Reusable by ``hf_publish.py``, ``regenerate_technique_inventory.py``,
    ``build_docs_site.py``, and the Hugging Face Space.
    """
    root = Path(root).resolve()
    laws_registry = load_laws_registry(root)
    law_by_id, parent_of_law = _law_lookup(laws_registry)
    valid_law_ids = set(law_by_id)

    tasks_dir = root / "configs/forge/tasks"
    sidecar_docs = _load_sidecar_documentation(root)
    views = _load_views(root)
    problem_defs = _load_problem_definitions(root)
    family_results, scaling_by_task, hf_repo = _load_family_results(root)
    fl_view_tasks = {
        a["task"]
        for vid in ("four-laws-adherence-v1", "four-laws-adherence-v2")
        for a in (views.get(vid) or {}).get("assignments", [])
        if isinstance(a, dict) and isinstance(a.get("task"), str)
    }
    fl_readouts = _load_four_laws_readouts(root, fl_task_ids=fl_view_tasks)
    api_gifs = _load_api_contract_gifs(root)
    family_law_mechs = _load_family_law_mechanisms(root)

    full_suite_set: set[str] = set()
    backlog_set: set[str] = set()
    fs_path = root / FULL_SUITE_LIST_REL
    if fs_path.is_file():
        fs_doc = _read_json(fs_path)
        full_suite_set = set(fs_doc.get("tests", []))
        backlog_set = set(fs_doc.get("unimplemented_backlog", fs_doc.get("blocked_placeholders", [])) or [])

    # Pre-index view memberships
    view_memberships: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vid, view in sorted(views.items()):
        for assign in view.get("assignments", []) or []:
            tid = assign.get("task")
            if isinstance(tid, str):
                view_memberships[tid].append({
                    "view": vid,
                    "tier": assign.get("qualification_tier"),
                    "importance": assign.get("importance"),
                    "role": assign.get("role"),
                    "law": assign.get("law"),
                })

    ast_cache: dict[str, dict[str, Any]] = {}

    def get_ast(rel_path: str) -> dict[str, Any]:
        if rel_path not in ast_cache:
            ast_cache[rel_path] = extract_python_annotations(root / rel_path)
        return ast_cache[rel_path]

    experiments: dict[str, dict[str, Any]] = {}
    for task_path in sorted(tasks_dir.glob("*.json")):
        task = _read_json(task_path)
        tid = str(task["id"])
        ex = task.get("execution", {})
        ev = task.get("evaluation", {})
        task_rel = task_path.relative_to(root).as_posix()

        # Gather Python sources for read-only AST extraction
        py_sources: list[str] = []
        hs = ex.get("host_source")
        if isinstance(hs, str) and hs.endswith(".py"):
            py_sources.append(hs)
        for src in sorted((ev.get("sources") or {}).keys()):
            if isinstance(src, str) and src.endswith(".py") and src not in py_sources:
                py_sources.append(src)

        ast_records: list[dict[str, Any]] = []
        ast_spec: dict[str, Any] = {}
        ast_thresh: list[Any] | None = None
        ast_exp_doc: dict[str, Any] | None = None
        ast_docstring: str | None = None
        for rel_py in py_sources:
            info = get_ast(rel_py)
            if info["extracted_symbols"]:
                ast_records.append({
                    "path": rel_py,
                    "symbols": list(info["extracted_symbols"]),
                })
            if ast_docstring is None and info["docstring"]:
                ast_docstring = info["docstring"]
            if isinstance(info["experiment_doc"], dict):
                if tid in info["experiment_doc"] and isinstance(info["experiment_doc"][tid], dict):
                    ast_exp_doc = info["experiment_doc"][tid]
                elif "title" in info["experiment_doc"] and ast_exp_doc is None:
                    ast_exp_doc = info["experiment_doc"]
            if tid in info["specs"] and isinstance(info["specs"][tid], dict):
                ast_spec = info["specs"][tid]
            elif tid in info["host_definitions"] and isinstance(info["host_definitions"][tid], dict):
                ast_spec = info["host_definitions"][tid]
            if tid in info["thresholds"] and isinstance(info["thresholds"][tid], list):
                ast_thresh = info["thresholds"][tid]

        # Merge editorial metadata: inline task["documentation"] > sidecar > AST EXPERIMENT_DOC > fallback
        doc_sources = [task_rel]
        editorial: dict[str, Any] = {}
        if isinstance(ast_exp_doc, dict):
            editorial.update(ast_exp_doc)
        if tid in sidecar_docs:
            sc_doc, sc_rel = sidecar_docs[tid]
            editorial.update({k: v for k, v in sc_doc.items() if k not in ("task_id", "schema_version")})
            doc_sources.append(sc_rel)
        if isinstance(task.get("documentation"), dict):
            editorial.update(task["documentation"])

        # Ensure safe defaults if an unannotated task appears in the future
        if not editorial.get("title"):
            editorial["title"] = str(ast_spec.get("title") or tid.replace("_", " ").title())
        if not editorial.get("intuition"):
            desc = str(task.get("description") or ast_spec.get("goal") or "").strip()
            editorial["intuition"] = desc or (ast_docstring.splitlines()[0] if ast_docstring else f"Forge benchmark task `{tid}`.")
        if not editorial.get("failure_mode"):
            prob = problem_defs.get(tid, {})
            editorial["failure_mode"] = str(
                prob.get("problem")
                or "Incomplete or uncalibrated GAN formulations fail to satisfy the declared numerical gates consistently."
            )
        if not isinstance(editorial.get("laws"), list):
            fl_law = next(
                (
                    m["law"]
                    for m in view_memberships[tid]
                    if m["view"] in {"four-laws-adherence-v2", "four-laws-adherence-v1"} and m.get("law") in valid_law_ids
                ),
                None,
            )
            editorial["laws"] = [fl_law] if fl_law else []
        if not isinstance(editorial.get("edge_cases"), list):
            editorial["edge_cases"] = []
        editorial.setdefault("protocol_notes", None)
        editorial["diagram_kind"] = infer_diagram_kind(task, editorial)
        editorial.setdefault("custom_mermaid", None)

        validate_task_documentation(editorial, valid_law_ids=valid_law_ids)

        # Compute top-level laws covered by this task (including parent laws of sub-laws)
        top_laws: list[str] = []
        for lid in editorial["laws"]:
            parent = parent_of_law.get(lid, lid)
            if parent not in top_laws:
                top_laws.append(parent)

        ds_assign = next((m for m in view_memberships[tid] if m["view"] == "discriminator_stability"), None)
        fl_assign = next((m for m in view_memberships[tid] if m["view"] == "four-laws-adherence-v1"), None)
        fl_v2_assign = next((m for m in view_memberships[tid] if m["view"] == "four-laws-adherence-v2"), None)

        thresholds = _extract_thresholds(task, ast_thresh)
        mermaid_src = generate_mermaid_diagram(task, editorial)

        experiments[tid] = {
            "task_id": tid,
            "task_path": task_rel,
            "adapter": task.get("adapter", ""),
            "evaluation_kind": ev.get("kind", ""),
            "in_full_suite": tid in full_suite_set,
            "in_unimplemented_backlog": tid in backlog_set,
            "discriminator_stability": ds_assign,
            "four_laws_adherence": fl_assign,
            "four_laws_adherence_v2": fl_v2_assign,
            "views": view_memberships[tid],
            "documentation": editorial,
            "top_laws": top_laws,
            "thresholds": thresholds,
            "prior": ex.get("prior", {}),
            "steps": ex.get("steps"),
            "dependencies": task.get("dependencies", []),
            "guards": ev.get("guards", {}),
            "sampling_law": ev.get("sampling_law", ""),
            "eval_output_noise": ev.get("eval_output_noise", ""),
            "scoring_weights": ev.get("scoring_weights", "live"),
            "mermaid": mermaid_src,
            "svg_relpath": f"{EXPERIMENTS_DIR_REL.as_posix()}/assets/{tid}.svg",
            "page_relpath": f"{EXPERIMENTS_DIR_REL.as_posix()}/{tid}.md",
            "family_results": family_results.get(tid, []),
            "scaling_results": scaling_by_task.get(tid, []),
            "four_laws_readouts": fl_readouts.get(tid, []),
            "problem_definition": problem_defs.get(tid),
            "api_contract_gif": api_gifs.get(tid),
            "ast_sources": ast_records,
            "provenance_files": doc_sources + [r["path"] for r in ast_records],
            "raw_task": task,
        }

    return {
        "schema_version": 1,
        "hf_repo": hf_repo,
        "laws_registry": laws_registry,
        "family_law_mechanisms": family_law_mechs,
        "experiment_count": len(experiments),
        "full_suite_count": sum(1 for e in experiments.values() if e["in_full_suite"]),
        "unimplemented_backlog_count": sum(1 for e in experiments.values() if e["in_unimplemented_backlog"]),
        "unimplemented_backlog": sorted(tid for tid, e in experiments.items() if e["in_unimplemented_backlog"]),
        "experiments": experiments,
    }


def _md_cell(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ").strip()


def _rel_link(root: Path, from_file: Path, target_rel: str, label: str, anchor: str | None = None) -> str:
    rel = os.path.relpath(root / target_rel, from_file.parent).replace("\\", "/")
    href = f"{rel}#{anchor}" if anchor else rel
    return f"[{_md_cell(label)}]({href})"


def _status_badge(status: str) -> str:
    up = status.upper()
    if up == "PASS":
        return "**PASS**"
    if up == "FAIL":
        return "**FAIL**"
    if up in ("BLOCKED", "INCOMPATIBLE"):
        return f"`{up}`"
    if up in ("COMPLETE",):
        return "**COMPLETE**"
    return f"`{up}`"


def render_experiment_page(root: Path, entry: dict[str, Any], catalog: dict[str, Any]) -> str:
    """Render a dedicated GFM+MkDocs-safe Markdown page for one Forge experiment."""
    tid = entry["task_id"]
    doc = entry["documentation"]
    task = entry["raw_task"]
    ex = task.get("execution", {})
    hd = ex.get("host_definition", {}) if isinstance(ex.get("host_definition"), dict) else {}
    res = ex.get("resources") or hd.get("resources") or {}
    model = ex.get("model") or hd.get("model") or {}

    laws_reg = catalog["laws_registry"]
    law_by_id, _ = _law_lookup(laws_reg)
    doc_path_laws = laws_reg.get("doc_path", "docs/four-laws.md")
    page_path = root / entry["page_relpath"]

    # Badges & tier labels
    ds = entry["discriminator_stability"]
    fl = entry["four_laws_adherence"]
    ds_text = f"Tier {ds['tier']} ({ds['importance']})" if ds else "Not in `discriminator_stability`"
    fl_text = f"{fl['law']} ({fl['role']})" if fl else "—"
    fs_count = catalog.get("full_suite_count", 59)
    suite_text = (
        f"Yes ({fs_count}-test active runnable denominator)"
        if entry["in_full_suite"]
        else "Unimplemented Tier-4 adapter backlog (planned; excluded from active runnable denominator)"
        if entry.get("in_unimplemented_backlog")
        else "No (diagnostic / auxiliary cohort)"
    )
    law_badges = " · ".join(
        f"[{law_by_id[lid]['short_label']}](../LAWS_HUB.md#{lid})"
        for lid in doc["laws"]
        if lid in law_by_id
    ) or "General / Infrastructure"
    edge_badges = " ".join(f"`{tag}`" for tag in doc["edge_cases"]) or "`general`"

    prior = entry["prior"]
    prior_summary = f"`{prior.get('kind', 'mog')}` (sigma=`{prior.get('sigma', 'default')}`, standardize=`{prior.get('standardize', False)}`)"

    lines: list[str] = [
        "<!-- Generated Forge experiment page — do not edit manually; regenerate via `python -m experiments.forge.experiment_docs` -->",
        "",
        f"[← Leaderboard](../technique-inventory.md) · [Experiment Catalog](README.md) · "
        f"[Explore by Laws](../LAWS_HUB.md) · [Research Portal]({RESEARCH_PORTAL_URL})",
        "",
        f"# {doc['title']} (`{tid}`)",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| **Laws Tested** | {law_badges} |",
        f"| **Edge Cases** | {edge_badges} |",
        f"| **Primary View (`discriminator_stability`)** | {ds_text} |",
        f"| **Four Laws Adherence Role** | {fl_text} |",
        f"| **Fixed Full-Suite Denominator** | {suite_text} |",
        f"| **Adapter / Domain** | `{entry['adapter']}` / `{entry['evaluation_kind']}` |",
        f"| **Prior** | {prior_summary} |",
        f"| **Update Budget** | `{entry['steps']}` steps |",
        "",
        "## What This Experiment Tests",
        "",
        doc["intuition"],
        "",
    ]

    if doc.get("protocol_notes"):
        lines.extend([
            "> [!NOTE]",
            f"> **Protocol notes:** {doc['protocol_notes']}",
            "",
        ])

    lines.extend([
        "## Laws & Edge Cases Targeted",
        "",
    ])
    if doc["laws"]:
        for lid in doc["laws"]:
            meta = law_by_id.get(lid)
            if not meta:
                continue
            theory_link = _rel_link(root, page_path, doc_path_laws, f"{meta['short_label']} theory", meta["doc_anchor"])
            lines.append(
                f"- **[{meta['short_label']}: {meta['title']}](../LAWS_HUB.md#{lid})** ({theory_link}) — {meta['question']}"
            )
    else:
        lines.append("- General Forge infrastructure / prerequisite check across laws.")
    if doc["edge_cases"]:
        lines.append(f"- **Edge-case tags:** {', '.join(f'`{t}`' for t in doc['edge_cases'])}")
    lines.append("")

    lines.extend([
        "## Common Failure Mode",
        "",
        doc["failure_mode"],
        "",
    ])
    pdef = entry.get("problem_definition")
    if isinstance(pdef, dict):
        lines.extend([
            f"**Historical audit diagnosis** (`reports/forge/problem-definitions.json`): {pdef.get('problem', '')}",
            "",
        ])
        if pdef.get("next_step"):
            lines.extend([f"- *Recommended research focus:* {pdef['next_step']}", ""])

    lines.extend([
        "## Visual Schematic",
        "",
        f"![Target and protocol schematic for {tid}](assets/{tid}.svg)",
        "",
        f"*{BADGE_TEXT} — generated deterministically from the declared task contract.*",
        "",
        "## Protocol & Architecture Diagram",
        "",
        "```mermaid",
        entry["mermaid"],
        "```",
        "",
        "## Task Contract & Numerical Gates",
        "",
        "### Execution & Sampling Contract",
        "",
        "| Contract Field | Declared Value |",
        "| --- | --- |",
        f"| Initializer | `{ex.get('initializer', 'deterministic_orthogonal')}` |",
        f"| Prior | {prior_summary} |",
        f"| Update Budget | `{entry['steps']}` steps |",
        f"| Batch Size | `{ex.get('batch_size') or res.get('batch_size') or hd.get('batch') or hd.get('batch_size') or 'host-declared'}` |",
        f"| Particles / Latent Dim | `P={res.get('num_particles') or hd.get('particles') or 'host'}` / `z_dim={res.get('z_dim') or hd.get('z_dim') or 'host'}` |",
        f"| Sampling Law | `{entry['sampling_law'] or 'host-declared'}` |",
        f"| Evaluation Output Noise | `{entry['eval_output_noise'] or 'clean'}` |",
        f"| Scoring Weights | `{entry['scoring_weights']}` |",
    ])
    if isinstance(model, dict) and model:
        compact_model = ", ".join(f"{k}={v}" for k, v in list(model.items())[:8])
        lines.append(f"| Architecture | `{_md_cell(compact_model)}` |")
    if ex.get("continuation_of"):
        lines.append(f"| Continuation Parent | [`{ex['continuation_of']}`]({ex['continuation_of']}.md) |")
    if ex.get("shift_step") or hd.get("shift_step"):
        lines.append(f"| Shift Step | `{ex.get('shift_step') or hd.get('shift_step')}` |")
    if entry["dependencies"]:
        dep_links = ", ".join(
            f"[`{d['task']}`]({d['task']}.md) (`{d['kind']}`)" for d in entry["dependencies"]
        )
        lines.append(f"| Dependencies | {dep_links} |")
    if entry["guards"]:
        guard_keys = ", ".join(f"`{k}`" for k, v in entry["guards"].items() if v)
        lines.append(f"| Integrity Guards | {guard_keys} |")
    if entry["views"]:
        view_str = ", ".join(f"`{v['view']}` (T{v['tier']} {v['importance']})" for v in entry["views"][:8])
        if len(entry["views"]) > 8:
            view_str += f" (+{len(entry['views']) - 8} more)"
        lines.append(f"| Assigned Views | {view_str} |")
    lines.append("")

    lines.extend([
        "### Numerical Pass/Fail Thresholds",
        "",
    ])
    if entry["thresholds"]:
        lines.extend([
            "| Metric | Operator | Threshold |",
            "| --- | :---: | ---: |",
        ])
        for metric, op, bound in entry["thresholds"]:
            lines.append(f"| `{_md_cell(metric)}` | `{_md_cell(op)}` | `{_md_cell(bound)}` |")
    else:
        lines.append("No fixed scalar threshold list declared (see evaluator contract above).")
    lines.append("")

    # Recorded Family Results
    lines.extend([
        "## Recorded Family Results",
        "",
        "> [!IMPORTANT]",
        "> Screening outcomes below come from recorded single-seed (`seed = 0`) Forge evidence. "
        "`NOT MEASURED`, `BLOCKED`, and `INCOMPATIBLE` are non-numerical states — not scientific failures — "
        "and confer no pass or qualification credit.",
        "",
    ])
    frows = entry["family_results"]
    if entry["scaling_results"]:
        lines.extend([
            "| Family | Candidate | Device | Status | Median ms/step | Images/s | Peak Allocated MiB | Family Report |",
            "| --- | --- | --- | :---: | ---: | ---: | ---: | --- |",
        ])
        for sr in sorted(entry["scaling_results"], key=lambda r: (r["step_ms_median"] is None, r["step_ms_median"] or 1e9)):
            ms_s = f"{sr['step_ms_median']:.2f}" if isinstance(sr["step_ms_median"], (int, float)) else "—"
            ips_s = f"{sr['images_per_second']:.1f}" if isinstance(sr["images_per_second"], (int, float)) else "—"
            mem_s = f"{sr['peak_allocated_mib']:.1f}" if isinstance(sr["peak_allocated_mib"], (int, float)) else "—"
            fam_link = _rel_link(root, page_path, sr["page"], sr["family_label"]) if sr.get("page") else "—"
            lines.append(
                f"| **{_md_cell(sr['family_label'])}** | `{_md_cell(sr['candidate_id'])}` | "
                f"{_md_cell(sr['device'])} | {_status_badge(sr['status'])} | {ms_s} | {ips_s} | {mem_s} | {fam_link} |"
            )
        lines.append("")
    elif frows:
        lines.extend([
            "| Family | Selected Candidate | Outcome | Key Recorded Metrics | Family Report | Training Media |",
            "| --- | --- | :---: | --- | --- | --- |",
        ])
        for fr in sorted(frows, key=lambda r: (0 if r["status"] == "PASS" else 1, r["family_label"].lower())):
            fam_link = _rel_link(root, page_path, fr["page"], fr["family_label"], fr["anchor"] or None)
            media = f"[HF GIF]({fr['gif_url']})" if fr.get("gif_url") else "—"
            lines.append(
                f"| **{_md_cell(fr['family_label'])}** | `{_md_cell(fr['candidate_id'])}` | "
                f"{_status_badge(fr['status'])} | {_md_cell(fr['metrics']) or '—'} | {fam_link} | {media} |"
            )
        lines.append("")
    elif not entry["four_laws_readouts"]:
        lines.extend([
            "No selected-family cohort cells are recorded for this task in the current `reports/forge/technique-inventory.json`.",
            "",
        ])

    if entry["four_laws_readouts"]:
        lines.extend([
            "### Four Laws Adherence Diagnostic Receipts (Separate Cohort)",
            "",
            "*Diagnostic readouts and Colab matched-GPU receipts for `four-laws-adherence-v1` (protocol seed `0`, unchanged recipes; research telemetry only — separate from ordinary selected-cohort qualification above and confers no qualification or default-adoption credit):*",
            "",
            "| Candidate | Family | GPU | Outcome | `min_margin` | NVML Energy | Key Recorded Metrics | Receipt |",
            "| --- | --- | --- | :---: | ---: | ---: | --- | --- |",
        ])
        for dr in entry["four_laws_readouts"]:
            rec_link = _rel_link(root, page_path, dr["readout"], Path(dr["readout"]).name)
            mm_s = f"`{dr['min_margin']:+.3f}`" if isinstance(dr.get("min_margin"), (int, float)) else "—"
            en_s = _format_energy_cell(dr.get("joules_total"), dr.get("joules_to_pass"), dr["status"])
            lines.append(
                f"| `{_md_cell(dr['candidate_id'])}` | `{_md_cell(dr['family_id'] or '—')}` | "
                f"{_md_cell(dr.get('gpu_name') or '—')} | {_status_badge(dr['status'])} | "
                f"{mm_s} | {_md_cell(en_s)} | {_md_cell(dr['metrics']) or '—'} | {rec_link} |"
            )
        lines.append("")

    if entry.get("api_contract_gif"):
        gif_link = _rel_link(root, page_path, entry["api_contract_gif"], "Actual-training reference GIF")
        lines.extend([f"- **Reference API-contract media:** {gif_link}", ""])

    # Reproduce section
    preferred_view = (
        "discriminator_stability"
        if ds
        else (
            "dcgan_performance"
            if entry["adapter"] == "dcgan_performance_v1"
            else (
                "four-laws-adherence-v1"
                if fl
                else (entry["views"][0]["view"] if entry["views"] else "discriminator_stability")
            )
        )
    )
    lines.extend([
        "## Reproduce This Experiment",
        "",
        "```bash",
        "# Validate this task declaration and inspect planned dependencies",
        "python -m experiments.forge validate",
        f"python -m experiments.forge plan --view {preferred_view} --candidate unified-v12 --tasks {tid}",
        "```",
        "",
        "**Declaration & Source Files:**",
        "",
        f"- Task contract: {_rel_link(root, page_path, entry['task_path'], entry['task_path'])}",
    ])
    for ast_src in entry["ast_sources"]:
        src_path = ast_src["path"]
        if (root / src_path).is_file():
            lines.append(f"- Source code: {_rel_link(root, page_path, src_path, src_path)}")
    lines.append("")

    # Annotation Provenance
    lines.extend([
        "## Annotation Provenance",
        "",
        "This page was assembled read-only from the following versioned inputs without altering any task execution/evaluation fingerprint:",
        "",
    ])
    for pfile in entry["provenance_files"]:
        if (root / pfile).is_file():
            role = "Task contract JSON" if pfile.startswith("configs/forge/tasks/") else (
                "Editorial task documentation sidecar" if pfile.startswith("configs/forge/task-documentation/")
                else "Read-only Python AST source"
            )
            syms = next((a["symbols"] for a in entry["ast_sources"] if a["path"] == pfile), [])
            sym_note = f" (AST symbols: {', '.join(f'`{s}`' for s in syms)})" if syms else ""
            lines.append(f"- **{role}:** {_rel_link(root, page_path, pfile, pfile)}{sym_note}")
    lines.append("")

    return "\n".join(lines)


def render_experiment_catalog_readme(root: Path, catalog: dict[str, Any]) -> str:
    """Render ``reports/forge/experiments/README.md`` grouped by Law, Tier, and Edge Case."""
    exps = catalog["experiments"]
    laws_reg = catalog["laws_registry"]
    law_by_id, _ = _law_lookup(laws_reg)
    readme_path = root / EXPERIMENTS_DIR_REL / "README.md"

    fl_count = sum(1 for e in exps.values() if e["four_laws_adherence"])
    dcgan_count = sum(1 for e in exps.values() if e["adapter"] == "dcgan_performance_v1")

    lines: list[str] = [
        "<!-- Generated Forge experiment catalog — regenerate via `python -m experiments.forge.experiment_docs` -->",
        "",
        f"[← Leaderboard](../technique-inventory.md) · [Explore by Laws](../LAWS_HUB.md) · "
        f"[Unresolved Problems](../problems.md) · [Research Portal]({RESEARCH_PORTAL_URL})",
        "",
        "# ParticleGAN Forge Experiment Catalog",
        "",
        f"Complete index of all **{len(exps)}** base Forge benchmark tasks (`{catalog['full_suite_count']}` in the fixed "
        f"full-suite denominator, `{fl_count}` in `four-laws-adherence-v1`, `{dcgan_count}` DCGAN scaling workloads). "
        "Every experiment has a dedicated page with plain-English intuition, common failure modes, a deterministic "
        "target-geometry SVG schematic, a Mermaid pipeline diagram, numerical gates, and recorded family outcomes.",
        "",
        "> [!NOTE]",
        f"> **Provisional & Evolving Laws:** {laws_reg.get('epistemic_note', '')}",
        "",
        "## Browse by Law",
        "",
        "| Law | Fundamental Question | Experiments | Explore |",
        "| --- | --- | ---: | --- |",
    ]

    for law in laws_reg["laws"]:
        lid = law["id"]
        matching = sorted(tid for tid, e in exps.items() if lid in e["top_laws"])
        lines.append(
            f"| **[{law['short_label']}: {law['title']}](../LAWS_HUB.md#{lid})** | "
            f"{_md_cell(law.get('question', ''))} | {len(matching)} | "
            f"[Laws Hub](../LAWS_HUB.md#{lid}) |"
        )
    lines.append("")

    # Browse by Tier (discriminator_stability + Four Laws Adherence + DCGAN + Legacy/Diagnostic)
    lines.extend([
        "## Browse by Tier (`discriminator_stability` & Specialized Suites)",
        "",
    ])

    def _emit_table(heading: str, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        lines.extend([
            f"### {heading} ({len(items)})",
            "",
            "| Experiment | Laws | Role / Suite | Steps | Intuition |",
            "| --- | --- | --- | ---: | --- |",
        ])
        for e in sorted(items, key=lambda x: x["task_id"]):
            tid = e["task_id"]
            doc = e["documentation"]
            law_str = ", ".join(law_by_id[l]["short_label"] for l in doc["laws"] if l in law_by_id) or "—"
            ds = e["discriminator_stability"]
            fl = e["four_laws_adherence"]
            role_parts = []
            if ds:
                role_parts.append(f"T{ds['tier']} {ds['importance']}")
            if fl:
                role_parts.append(f"FL:{fl['role']}")
            if not role_parts:
                role_parts.append(e["adapter"])
            snippet = doc["intuition"]
            if len(snippet) > 110:
                snippet = snippet[:107].rstrip() + "..."
            lines.append(
                f"| [`{tid}`]({tid}.md) — {_md_cell(doc['title'])} | {_md_cell(law_str)} | "
                f"`{_md_cell(' · '.join(role_parts))}` | `{e['steps']}` | {_md_cell(snippet)} |"
            )
        lines.append("")

    for tier in (1, 2, 3, 4):
        tier_items = [e for e in exps.values() if e["discriminator_stability"] and e["discriminator_stability"]["tier"] == tier]
        _emit_table(f"Tier {tier} (`discriminator_stability`)", tier_items)

    fl_probes_v2 = [
        e for e in exps.values()
        if e.get("four_laws_adherence_v2") and e["four_laws_adherence_v2"].get("role") == "four_laws_probe"
    ]
    if fl_probes_v2:
        _emit_table("Four Laws Adherence v2 Clean Dynamical Probes (`four-laws-adherence-v2`)", fl_probes_v2)

    fl_probes = [e for e in exps.values() if e["four_laws_adherence"] and e["four_laws_adherence"].get("role") == "four_laws_probe"]
    _emit_table("Four Laws Adherence Dynamical Falsification Probes (`four-laws-adherence-v1`)", fl_probes)

    dcgan_items = [e for e in exps.values() if e["adapter"] == "dcgan_performance_v1"]
    _emit_table("DCGAN Throughput & Memory Scaling (`dcgan_performance`)", dcgan_items)

    covered = {
        e["task_id"]
        for e in exps.values()
        if e["discriminator_stability"] or e["adapter"] == "dcgan_performance_v1"
        or (e["four_laws_adherence"] and e["four_laws_adherence"].get("role") == "four_laws_probe")
        or (e.get("four_laws_adherence_v2") and e["four_laws_adherence_v2"].get("role") == "four_laws_probe")
    }
    remaining = [e for e in exps.values() if e["task_id"] not in covered]
    fs_rem = [e for e in remaining if e["in_full_suite"]]
    legacy_rem = [e for e in remaining if not e["in_full_suite"] and e["task_id"].startswith("recipe-prior-refactor-legacy-")]
    diag_rem = [e for e in remaining if not e["in_full_suite"] and not e["task_id"].startswith("recipe-prior-refactor-legacy-")]

    _emit_table("Additional Fixed Full-Suite Tasks (Continuations & Adaptation)", fs_rem)
    _emit_table("Diagnostic, Ablation & Architectural Variants", diag_rem)
    _emit_table("Legacy Recipe-Prior-Refactor Comparison Cohort", legacy_rem)

    # Browse by Edge Case Tag
    by_tag: dict[str, list[str]] = defaultdict(list)
    for tid, e in sorted(exps.items()):
        for tag in e["documentation"]["edge_cases"]:
            by_tag[tag].append(tid)

    lines.extend([
        "## Browse by Edge Case Tag",
        "",
        "| Edge Case Tag | Count | Experiments |",
        "| --- | ---: | --- |",
    ])
    for tag, tids in sorted(by_tag.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        shown = ", ".join(f"[`{t}`]({t}.md)" for t in tids[:12])
        if len(tids) > 12:
            shown += f" *(+{len(tids) - 12} more)*"
        lines.append(f"| `{_md_cell(tag)}` | {len(tids)} | {shown} |")
    lines.append("")

    # Reference link to theory doc
    lines.append(f"Canonical theoretical specification: {_rel_link(root, readme_path, laws_reg.get('doc_path', 'docs/four-laws.md'), 'The Four Laws of Adversarial Generative Learning')}.")
    lines.append("")
    return "\n".join(lines)


def _render_four_laws_l4_leaderboard(
    exps: dict[str, dict[str, Any]],
    *,
    field_key: str = "four_laws_adherence",
    view_id: str = "four-laws-adherence-v1",
    heading: str = "Four Laws Adherence Diagnostic Leaderboard (NVIDIA L4 Cohort)",
) -> list[str]:
    """Render a matched NVIDIA L4 Four Laws Adherence diagnostic leaderboard from recorded receipts."""
    fl_probes = [
        e for e in exps.values()
        if e.get(field_key) and e[field_key].get("role") == "four_laws_probe"
    ]
    if not fl_probes:
        return []

    fl_probes.sort(key=lambda e: (e[field_key].get("law", ""), e["task_id"]))
    probe_ids = [e["task_id"] for e in fl_probes]
    law_of = {e["task_id"]: str(e[field_key].get("law") or "") for e in fl_probes}

    # Index strictly matched NVIDIA L4 receipts for the fl_law* probes + 2 ByteGAN Law-1 anchors
    by_cand: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    cand_family: dict[str, str] = {}
    for tid in [*probe_ids, "bytegan_hypercube8", "bytegan_bigan_dictionary"]:
        entry = exps.get(tid)
        if not entry:
            continue
        for dr in entry.get("four_laws_readouts") or []:
            if dr.get("gpu_name") != "NVIDIA L4":
                continue
            cid = dr["candidate_id"]
            by_cand[cid][tid] = dr
            if dr.get("family_id"):
                cand_family[cid] = dr["family_id"]

    l4_probe_cands = [cid for cid, m in by_cand.items() if any(pid in m for pid in probe_ids)]
    if not l4_probe_cands:
        return []

    law_order = ["law1", "law2", "law3", "law4"]
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

    ordered_cands = sorted(l4_probe_cands, key=_sort_key)

    lines: list[str] = [
        f"## {heading}",
        "",
        "> [!IMPORTANT]",
        f"> Matched **NVIDIA L4** hardware cohort (`{view_id}`, protocol seed `0`, unchanged global recipes). "
        "Supplies research diagnostic telemetry only — separate from ordinary selected-cohort qualification and confers no default-adoption credit. "
        "Statically preflight-blocked pairs (`atlas-particle-normal-v4`: requires `MoGParticlePrior`; `e22`: requires policy-control/served-sampling contract) remain `BLOCKED` and distinct from numerical `FAIL`.",
        "",
        f"### Cross-Technique Family × Law Summary ({len(probe_ids)} `fl_law*` Probes + ByteGAN)",
        "",
        f"| Candidate | Family | Law 1 ({len(probes_by_law['law1'])}) | Law 2 ({len(probes_by_law['law2'])}) | Law 3 ({len(probes_by_law['law3'])}) | Law 4 ({len(probes_by_law['law4'])}) | Probes PASS | ByteGAN (L4) | {len(probe_ids)}-Probe NVML Energy |",
        "| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |",
    ]

    for cid in ordered_cands:
        m = by_cand[cid]
        law_cells = []
        total_pass = 0
        total_measured = 0
        total_j = 0.0
        has_j = False
        for lid in law_order:
            lpids = probes_by_law[lid]
            lp = sum(1 for pid in lpids if (m.get(pid) or {}).get("status") == "PASS")
            lm = sum(1 for pid in lpids if pid in m)
            total_pass += lp
            total_measured += lm
            law_cells.append(f"`{lp}/{len(lpids)}`" if lm == len(lpids) else f"`{lp}/{lm}`")
        for pid in probe_ids:
            jt = (m.get(pid) or {}).get("joules_total")
            if isinstance(jt, (int, float)):
                total_j += jt
                has_j = True
        bg_m = [tid for tid in ("bytegan_hypercube8", "bytegan_bigan_dictionary") if tid in m]
        if bg_m:
            bg_p = sum(1 for tid in bg_m if m[tid]["status"] == "PASS")
            bg_cell = f"`{bg_p}/{len(bg_m)}`"
        else:
            bg_cell = "—"
        energy_cell = f"{total_j / 1000.0:.2f} kJ" if has_j else "—"
        lines.append(
            f"| `{_md_cell(cid)}` | `{_md_cell(cand_family.get(cid) or '—')}` | "
            + " | ".join(law_cells)
            + f" | **{total_pass}/{total_measured}** | {bg_cell} | {energy_cell} |"
        )

    # Also list explicitly blocked families if present in catalog without L4 probe runs
    for blocked_cid, blocked_fam, reason in (
        ("atlas-particle-normal-v4", "atlas", "requires `MoGParticlePrior`"),
        ("e22", "e22", "clean/live policy contract"),
    ):
        if blocked_cid not in by_cand or not any(pid in by_cand[blocked_cid] for pid in probe_ids):
            bg_m = [tid for tid in ("bytegan_hypercube8", "bytegan_bigan_dictionary") if tid in by_cand.get(blocked_cid, {})]
            bg_cell = (
                f"`{sum(1 for tid in bg_m if by_cand[blocked_cid][tid]['status'] == 'PASS')}/{len(bg_m)}`"
                if bg_m
                else "`BLOCKED`"
            )
            lines.append(
                f"| `{blocked_cid}` | `{blocked_fam}` | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` | "
                f"`0/0` ({reason}) | {bg_cell} | — |"
            )
    lines.append("")

    short_headers = {
        "fl_law1_aniso_unequal_modes": "L1 Aniso",
        "fl_law1_noisy_atomic_codebook": "L1 Noisy",
        "fl_law1_noisy_atomic_codebook_v2": "L1 Noisy v2",
        "fl_law1_highd_rank_d32_v2": "L1 Rank32 v2",
        "fl_law2_pure_adversarial_codim1": "L2 Codim1",
        "fl_law2_groupnorm_and_vit_sprites": "L2 GN+ViT",
        "fl_law2_groupnorm_and_vit_sprites_v2": "L2 GN+ViT v2",
        "fl_law3_highd_orphan_d32": "L3 R32",
        "fl_law3_highd_orphan_d32_v2": "L3 Orphan v2",
        "fl_law41_trust_rare1pct_tanh": "L4.1 Rare1%",
        "fl_law42_swap_deep_critic_16l": "L4.2 Swap16L",
        "fl_law42_deep_resbn_and_conv1x1": "L4.2 ResBN",
        "fl_law42_deep_resbn_and_conv1x1_v2": "L4.2 ResBN v2",
        "fl_law44_highd_sparse_prior_and_shock": "L4.4 R512",
        "fl_law44_scale_shock": "L4.4 Shock",
    }
    header_links = [
        f"[{short_headers.get(pid, pid)}](experiments/{pid}.md)" for pid in probe_ids
    ]
    lines.extend([
        "### Per-Probe Outcomes & Signed `min_margin` (NVIDIA L4)",
        "",
        "| Candidate | " + " | ".join(header_links) + " |",
        "| --- | " + " | ".join([":---:"] * len(probe_ids)) + " |",
    ])
    for cid in ordered_cands:
        m = by_cand[cid]
        cells = []
        for pid in probe_ids:
            dr = m.get(pid)
            if not dr:
                cells.append("—")
                continue
            st = dr["status"]
            mm = dr.get("min_margin")
            mm_s = f"{mm:+.2f}" if isinstance(mm, (int, float)) else ""
            if st == "PASS":
                jtp = dr.get("joules_to_pass")
                j_s = f" ({jtp / 1000.0:.2f} kJ)" if isinstance(jtp, (int, float)) else ""
                cells.append(f"**PASS** `{mm_s}`{j_s}")
            elif st == "FAIL":
                cells.append(f"FAIL `{mm_s}`")
            else:
                cells.append(f"`{st}`")
        lines.append(f"| `{_md_cell(cid)}` | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_laws_hub(root: Path, catalog: dict[str, Any]) -> str:
    """Render ``reports/forge/LAWS_HUB.md`` dynamically from ``configs/forge/laws.json``."""
    laws_reg = catalog["laws_registry"]
    exps = catalog["experiments"]
    fam_mechs = catalog["family_law_mechanisms"]
    hub_path = root / LAWS_HUB_REL
    doc_path = laws_reg.get("doc_path", "docs/four-laws.md")

    lines: list[str] = [
        "<!-- Generated Forge Laws Hub — regenerate via `python -m experiments.forge.experiment_docs` -->",
        "",
        f"[← Leaderboard](technique-inventory.md) · [Experiment Catalog](experiments/README.md) · "
        f"{_rel_link(root, hub_path, doc_path, 'Four Laws Theory')} · [Research Portal]({RESEARCH_PORTAL_URL})",
        "",
        "# Explore ParticleGAN by Laws",
        "",
        f"> *\"{laws_reg.get('epigraph', '')}\"*",
        "",
        "> [!NOTE]",
        f"> **Epistemic Status (`{laws_reg.get('epistemic_status', 'provisional_evolving')}`):** "
        f"{laws_reg.get('epistemic_note', '')}",
        "",
        "## Current Laws Overview",
        "",
        "| Law | Status | Domain Level | Fundamental Question | Probing Experiments | Theory |",
        "| --- | :---: | --- | --- | ---: | --- |",
    ]

    for law in laws_reg["laws"]:
        lid = law["id"]
        probing = [e for e in exps.values() if lid in e["top_laws"]]
        tlink = _rel_link(root, hub_path, doc_path, f"{law['short_label']} Section", law.get("doc_anchor"))
        law_status = str(law.get("status") or "active")
        lines.append(
            f"| **[{law['short_label']}](#{lid})** | `{_md_cell(law_status)}` | {_md_cell(law.get('domain_level', law['title']))} | "
            f"{_md_cell(law.get('question', ''))} | {len(probing)} | {tlink} |"
        )
    lines.append("")

    lines.extend(
        _render_four_laws_l4_leaderboard(
            exps,
            field_key="four_laws_adherence_v2",
            view_id="four-laws-adherence-v2",
            heading="Four Laws Adherence v2 Diagnostic Leaderboard (NVIDIA L4 Cohort)",
        )
    )
    lines.extend(
        _render_four_laws_l4_leaderboard(
            exps,
            field_key="four_laws_adherence",
            view_id="four-laws-adherence-v1",
            heading="Four Laws Adherence Diagnostic Leaderboard (NVIDIA L4 Cohort)",
        )
    )

    for law in laws_reg["laws"]:
        lid = law["id"]
        tlink = _rel_link(root, hub_path, doc_path, f"Read {law['short_label']} in `{doc_path}`", law.get("doc_anchor"))
        lines.extend([
            f'<a id="{lid}"></a>',
            "",
            f"## {law['short_label']} — {law['title']}",
            "",
            f"**Fundamental question:** *{law.get('question', '')}*",
            "",
            f"- **Canonical theory:** {tlink}",
            f"- **Status:** `{law.get('status', 'active')}` (subject to revision as counterexamples emerge)",
            "",
        ])

        sub_laws = law.get("sub_laws") or []
        if sub_laws:
            lines.extend(["### Sub-Laws", ""])
            for sub in sub_laws:
                sid = sub["id"]
                sub_exps = sorted(tid for tid, e in exps.items() if sid in e["documentation"]["laws"])
                exp_links = ", ".join(f"[`{t}`](experiments/{t}.md)" for t in sub_exps[:10])
                if len(sub_exps) > 10:
                    exp_links += f" *(+{len(sub_exps) - 10} more)*"
                sub_tlink = _rel_link(root, hub_path, doc_path, "theory", sub.get("doc_anchor", law.get("doc_anchor")))
                lines.append(f'<a id="{sid}"></a>')
                lines.append(
                    f"- **{sub['short_label']}: {sub['title']}** ({sub_tlink}) — "
                    f"{len(sub_exps)} targeted experiment(s): {exp_links or '*(covered via parent law)*'}"
                )
            lines.append("")

        # Experiments probing/falsifying this law
        probing = sorted(
            (e for e in exps.values() if lid in e["top_laws"]),
            key=lambda e: (
                0 if (e["four_laws_adherence"] and e["four_laws_adherence"].get("role") == "four_laws_probe") else 1,
                0 if e["in_full_suite"] else 1,
                e["task_id"],
            ),
        )
        lines.extend([
            f"### Experiments Probing or Falsifying {law['short_label']} ({len(probing)})",
            "",
            "| Experiment | Role / Suite | Edge Cases | What It Tests |",
            "| --- | --- | --- | --- |",
        ])
        for e in probing:
            tid = e["task_id"]
            doc = e["documentation"]
            fl = e["four_laws_adherence"]
            ds = e["discriminator_stability"]
            if fl and fl.get("role") == "four_laws_probe":
                role = "Four Laws Probe"
            elif ds:
                role = f"T{ds['tier']} {ds['importance']}"
            elif e["in_full_suite"]:
                role = "Full Suite"
            else:
                role = e["adapter"]
            tags = ", ".join(f"`{t}`" for t in doc["edge_cases"][:3]) or "—"
            snip = doc["intuition"]
            if len(snip) > 120:
                snip = snip[:117].rstrip() + "..."
            lines.append(
                f"| [`{tid}`](experiments/{tid}.md) — {_md_cell(doc['title'])} | `{role}` | {tags} | {_md_cell(snip)} |"
            )
        lines.append("")

        # Families addressing this law (from family-documentation law_mechanisms when available)
        lines.extend([f"### Trainer Families & {law['short_label']}", ""])
        mech_rows = []
        for fid, fmeta in sorted(fam_mechs.items()):
            entry = fmeta["law_mechanisms"].get(lid)
            if isinstance(entry, dict):
                mech_rows.append((fmeta, entry))
        if mech_rows:
            lines.extend([
                "| Family | Status | Mechanism | Known Limits |",
                "| --- | :---: | --- | --- |",
            ])
            for fmeta, m in mech_rows:
                flink = _rel_link(root, hub_path, fmeta["page"], fmeta["label"])
                lines.append(
                    f"| {flink} | `{_md_cell(m.get('status', 'unknown'))}` | "
                    f"{_md_cell(m.get('mechanism', ''))} | {_md_cell(m.get('known_limits', ''))} |"
                )
            lines.append("")
        else:
            lines.extend([
                "Family-level `law_mechanisms` annotations are loaded dynamically from "
                "`configs/forge/family-documentation/*.json` as they are added. "
                "See the [Technique Leaderboard](technique-inventory.md) and individual [Family Reports](families/) "
                "for current family comparisons.",
                "",
            ])

    return "\n".join(lines)


def write_experiment_docs(root: Path | str = ROOT) -> dict[str, Any]:
    """Generate all experiment SVGs, dedicated Markdown pages, catalog index, and Laws Hub."""
    root = Path(root).resolve()
    catalog = build_experiment_catalog(root)
    exp_dir = root / EXPERIMENTS_DIR_REL
    assets_dir = exp_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    written_pages: list[str] = []
    written_svgs: list[str] = []
    for tid, entry in sorted(catalog["experiments"].items()):
        svg_path = assets_dir / f"{tid}.svg"
        write_task_svg(entry["raw_task"], svg_path, entry["documentation"])
        written_svgs.append(svg_path.relative_to(root).as_posix())

        page_path = exp_dir / f"{tid}.md"
        page_md = render_experiment_page(root, entry, catalog)
        page_path.write_text(page_md, encoding="utf-8")
        written_pages.append(page_path.relative_to(root).as_posix())

    readme_path = exp_dir / "README.md"
    readme_path.write_text(render_experiment_catalog_readme(root, catalog), encoding="utf-8")

    laws_hub_path = root / LAWS_HUB_REL
    laws_hub_path.write_text(render_laws_hub(root, catalog), encoding="utf-8")

    return {
        "experiment_count": len(written_pages),
        "svg_count": len(written_svgs),
        "catalog_readme": readme_path.relative_to(root).as_posix(),
        "laws_hub": laws_hub_path.relative_to(root).as_posix(),
        "pages": written_pages,
        "svgs": written_svgs,
    }


def _check_mermaid_balance(text: str) -> list[str]:
    """Verify balanced Mermaid code fences and quoted node labels."""
    errors: list[str] = []
    blocks = re.findall(r"```mermaid\n(.*?)```", text, flags=re.S)
    open_count = len(re.findall(r"^```mermaid\s*$", text, flags=re.M))
    if open_count != len(blocks):
        errors.append("unbalanced ```mermaid fence")
    for idx, block in enumerate(blocks):
        if not block.strip():
            errors.append(f"empty mermaid block #{idx}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Forge experiment pages, target SVGs, catalog, and Laws Hub.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Print generation summary as JSON.")
    args = parser.parse_args(argv)
    summary = write_experiment_docs(args.root)
    if args.json:
        print(json.dumps({k: v for k, v in summary.items() if k not in ("pages", "svgs")}, indent=2))
    else:
        print(
            f"Generated {summary['experiment_count']} experiment pages, {summary['svg_count']} SVG schematics, "
            f"{summary['catalog_readme']}, and {summary['laws_hub']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
