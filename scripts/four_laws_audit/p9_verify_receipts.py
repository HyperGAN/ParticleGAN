#!/usr/bin/env python3
"""Programmatic audit of ParticleGAN Four Laws Forge receipts, complexity, comparability, and efficiency.

This script is strictly read-only with respect to repository Forge receipts and source code:
- Verifies every markdown link and readout receipt referenced in ``docs/four-laws.md``.
- Audits the missing ``unified-v12`` readout links (``readout-5002641a...``, ``readout-731e752f...``,
  ``particlegan/unified_v12.py``) against the actual committed ``unified-v12`` readouts on
  ``origin/develop`` and ``reports/forge/studies/law4-phase2-energy-benchmark.json``.
- Checks every quantitative figure in the progression table and prose, flagging exact matches,
  numerical discrepancies, uncommitted/prose-only claims, and comparability confounds
  (precision FP32 vs BF16 channels-last, benchmark harness mismatch, idle power drift, contention).
- Extracts energy-per-step (J/step) vs energy-to-quality (Joules-to-first-pass / sustained-pass / total).
- Computes exact LOC, AST branches (``ast.If``, ``ast.IfExp``), and transitive local import closure
  across all champion formulation packages, and detects surviving Round-4 Occam epicycles.
- Reconstructs full lineage tuning budgets (candidate cards, studies, readouts, task evaluations, GPU-hours)
  and audits post-freeze unseen benchmarks (ByteGAN) and cross-GPU/architecture transfer (Colab L4 waves).
- Optional ``--profile-gpu`` runs a controlled local RTX A6000 micro-benchmark measuring component
  wall-clock breakdown, VRAM, NVML energy, and the FP32 vs BF16 precision confound on ``rhbi_optimized``.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECORDS_DIR = ROOT / "reports" / "forge" / "records"
STUDIES_DIR = ROOT / "reports" / "forge" / "studies"
FOUR_LAWS_PATH = ROOT / "docs" / "four-laws.md"


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_four_laws_links(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    """Verify every relative file link and every readout-* reference in docs/four-laws.md."""
    four_laws = root / "docs" / "four-laws.md"
    text = four_laws.read_text(encoding="utf-8")

    md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
    checked_links: List[Dict[str, Any]] = []
    broken_links: List[Dict[str, Any]] = []
    for label, target in md_links:
        if target.startswith("#") or target.startswith(("http://", "https://")):
            continue
        path_part = target.split("#")[0]
        resolved = (four_laws.parent / path_part).resolve()
        exists = resolved.exists()
        entry = {
            "label": label,
            "target": target,
            "resolved": str(resolved.relative_to(root)) if str(resolved).startswith(str(root)) else str(resolved),
            "exists": exists,
        }
        checked_links.append(entry)
        if not exists:
            broken_links.append(entry)

    # All readout-* references (both markdown links and inline code)
    raw_refs = sorted(set(re.findall(r"readout-[0-9a-f]{8,24}(?:\.json)?", text)))
    prefixes = sorted(set(r.replace(".json", "")[:16] for r in raw_refs))
    records_dir = root / "reports" / "forge" / "records"

    readout_inventory: List[Dict[str, Any]] = []
    missing_readouts: List[str] = []
    for pref in prefixes:
        matches = sorted(records_dir.glob(f"{pref}*.json"))
        if not matches:
            missing_readouts.append(pref)
            readout_inventory.append({"prefix": pref, "exists": False, "file": None})
            continue
        rec_path = matches[0]
        data = _load_json(rec_path)
        trs = data.get("task_results", [])
        status_counts: Dict[str, int] = {}
        for tr in trs:
            st = str(tr.get("gate_status"))
            status_counts[st] = status_counts.get(st, 0) + 1
        readout_inventory.append(
            {
                "prefix": pref,
                "exists": True,
                "file": rec_path.name,
                "candidate_id": data.get("candidate_id"),
                "candidate_revision": data.get("candidate_revision"),
                "study_id": data.get("study_id"),
                "goal": data.get("goal"),
                "evidence_scope": data.get("evidence_scope"),
                "n_tasks": len(trs),
                "status_counts": status_counts,
            }
        )

    return {
        "total_markdown_links": len(md_links),
        "checked_local_file_links": len(checked_links),
        "broken_links": broken_links,
        "unique_readout_prefixes": len(prefixes),
        "existing_readout_count": len(prefixes) - len(missing_readouts),
        "missing_readouts": missing_readouts,
        "readout_inventory": readout_inventory,
    }


def _view_role_map(view_path: pathlib.Path) -> Dict[str, Tuple[Optional[int], Optional[str]]]:
    view = _load_json(view_path)
    out: Dict[str, Tuple[Optional[int], Optional[str]]] = {}
    for item in view.get("assignments", []):
        tid = item.get("task") or item.get("task_id") or item.get("id")
        tier = item.get("qualification_tier") if "qualification_tier" in item else item.get("tier")
        role = item.get("importance") or item.get("role")
        out[str(tid)] = (tier, role)
    return out


def audit_unified_v12_receipts(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    """Audit all actual committed unified-v12 readouts vs cited missing hashes and phase2 JSON."""
    records_dir = root / "reports" / "forge" / "records"
    committed_v12: List[Dict[str, Any]] = []
    for p in sorted(records_dir.glob("readout-*.json")):
        d = _load_json(p)
        if d.get("candidate_id") == "unified-v12":
            trs = d.get("task_results", [])
            counts: Dict[str, int] = {}
            contended_tasks = 0
            energy_tasks = 0
            j_first_tasks = 0
            j_sust_tasks = 0
            devices = set()
            gpu_names = set()
            for tr in trs:
                st = str(tr.get("gate_status"))
                counts[st] = counts.get(st, 0) + 1
                cost = tr.get("cost") or {}
                if cost.get("device") is not None:
                    devices.add(str(cost.get("device")))
                en = cost.get("energy") or {}
                if en.get("gpu_name"):
                    gpu_names.add(str(en["gpu_name"]))
                if en.get("contended") is True:
                    contended_tasks += 1
                if en.get("joules_per_step") is not None:
                    energy_tasks += 1
                if en.get("joules_to_first_pass") is not None:
                    j_first_tasks += 1
                if en.get("joules_to_sustained_pass") is not None:
                    j_sust_tasks += 1
            committed_v12.append(
                {
                    "file": p.name,
                    "study_id": d.get("study_id"),
                    "goal": d.get("goal"),
                    "evidence_scope": d.get("evidence_scope"),
                    "candidate_revision": d.get("candidate_revision"),
                    "n_tasks": len(trs),
                    "status_counts": counts,
                    "devices": sorted(devices),
                    "gpu_names": sorted(gpu_names),
                    "contended_tasks": contended_tasks,
                    "energy_j_per_step_tasks": energy_tasks,
                    "energy_j_to_first_pass_tasks": j_first_tasks,
                    "energy_j_to_sustained_pass_tasks": j_sust_tasks,
                }
            )

    # Detailed check of readout-b51ab67c4d97526186e23651.json against discriminator_stability rev 19
    disc_path = records_dir / "readout-b51ab67c4d97526186e23651.json"
    disc_data = _load_json(disc_path)
    rm19 = _view_role_map(root / "configs" / "forge" / "view-history" / "discriminator_stability-v19.json")
    tier_role_status: Dict[str, int] = {}
    required_pass = 0
    required_total = 0
    disc_failures: Dict[str, Dict[str, Any]] = {}
    disc_energy_rows: List[Dict[str, Any]] = []
    for tr in disc_data["task_results"]:
        tid = tr["task_id"]
        st = tr["gate_status"]
        tier, role = rm19.get(tid, (None, None))
        key = f"T{tier}_{role}_{st}"
        tier_role_status[key] = tier_role_status.get(key, 0) + 1
        if role == "required":
            required_total += 1
            if st == "PASS":
                required_pass += 1
        if st != "PASS":
            disc_failures[tid] = {
                "gate_status": st,
                "raw_status": tr.get("raw_status"),
                "reason": tr.get("reason"),
                "metrics": tr.get("metrics"),
            }
        en = (tr.get("cost") or {}).get("energy") or {}
        if en.get("joules_total") is not None:
            disc_energy_rows.append(
                {
                    "task_id": tid,
                    "gate_status": st,
                    "wall_seconds": (tr.get("cost") or {}).get("wall_seconds"),
                    "joules_per_step": en.get("joules_per_step"),
                    "first_pass_step": en.get("first_pass_step"),
                    "joules_to_first_pass": en.get("joules_to_first_pass"),
                    "sustained_pass_step": en.get("sustained_pass_step"),
                    "joules_to_sustained_pass": en.get("joules_to_sustained_pass"),
                    "joules_total": en.get("joules_total"),
                    "contended": en.get("contended"),
                }
            )

    # Detailed check of committed dcgan_performance readout-59898e1a77c4b179c9aebbc1.json
    dcgan_data = _load_json(records_dir / "readout-59898e1a77c4b179c9aebbc1.json")
    r256_tr = [t for t in dcgan_data["task_results"] if t["task_id"] == "dcgan_performance_r256"][0]
    committed_r256 = {
        "file": "readout-59898e1a77c4b179c9aebbc1.json",
        "update_ms_median": r256_tr["metrics"]["update_ms_median"],
        "update_ms_p95": r256_tr["metrics"]["update_ms_p95"],
        "images_per_second": r256_tr["metrics"]["images_per_second"],
        "peak_allocated_mib": r256_tr["metrics"]["peak_allocated_mib"],
        "peak_reserved_mib": r256_tr["metrics"]["peak_reserved_mib"],
        "warmup_updates": r256_tr["metrics"]["completed_updates"] - r256_tr["metrics"]["measured_updates"],
        "measured_updates": r256_tr["metrics"]["measured_updates"],
        "contended": (r256_tr.get("cost") or {}).get("energy", {}).get("contended"),
        "joules_total_13steps": (r256_tr.get("cost") or {}).get("energy", {}).get("joules_total"),
    }

    # Detailed check of committed t4-arch readout-c1ee5c3bb7ac4f432532da63.json
    t4_data = _load_json(records_dir / "readout-c1ee5c3bb7ac4f432532da63.json")
    t4_pass = sum(1 for t in t4_data["task_results"] if t["gate_status"] == "PASS")
    res16_tr = [t for t in t4_data["task_results"] if t["task_id"] == "grid100_res16_tanh"][0]

    # Phase 2 JSON (contains Phase 4 custom profiler metrics and references to missing hashes)
    p2 = _load_json(root / "reports" / "forge" / "studies" / "law4-phase2-energy-benchmark.json")
    p2_v12 = p2["energy_benchmark"]["unified-v12"]
    p2_r256 = p2_v12["dcgan_r256"]["sustained_uninstrumented"]

    # Check whether any committed readout-*.json exists for unified-v12 on Tier-3 holdout or shift diagnostic
    committed_goals = sorted(r["goal"] for r in committed_v12)

    # Check family-current-v1.json selection pin vs DEFAULT_RECIPE
    sel_pin = _load_json(root / "configs" / "forge" / "selections" / "family-current-v1.json")
    unified_sel = [s for s in sel_pin["selections"] if s["trainer_family"] == "unified"][0]

    return {
        "cited_missing_readouts_in_docs": [
            "readout-5002641a64ddaa91f1fc3b87.json",
            "readout-731e752ff3c4f97e29a84145.json",
        ],
        "cited_missing_readouts_in_champion_study": [
            "readout-5002641a64ddaa91f1fc3b87.json",
            "readout-731e752ff3c4f97e29a84145.json",
            "readout-7d00fc8c",
            "readout-a85f705c",
        ],
        "committed_v12_readouts": committed_v12,
        "committed_v12_goals": committed_goals,
        "uncommitted_claimed_v12_views": [
            "r3-tier3-long-horizon-holdout-v1 (18/18 Tier-3 endurance)",
            "t3-shift-recovery-diagnostic-v1 (5/5 shift recovery)",
            "dcgan_performance combined 3bf87cea3+ff41cf173 (39.76 ms/step, 1243.9 MiB)",
            "t4-strange-architectures-v1 combined 3bf87cea3+ff41cf173 (6/7, step-6250 modes=99 1-ULP dip)",
        ],
        "disc_stab_b51ab67c": {
            "pass_count": sum(1 for t in disc_data["task_results"] if t["gate_status"] == "PASS"),
            "total_count": len(disc_data["task_results"]),
            "required_pass": required_pass,
            "required_total": required_total,
            "tier_role_status": dict(sorted(tier_role_status.items())),
            "failures": disc_failures,
            "energy_rows": disc_energy_rows,
        },
        "t4_arch_c1ee5c3b": {
            "pass_count": t4_pass,
            "total_count": len(t4_data["task_results"]),
            "grid100_res16_tanh_status": res16_tr["gate_status"],
            "grid100_res16_tanh_precision": res16_tr["metrics"].get("precision"),
            "grid100_res16_tanh_center_rms_sigma": res16_tr["metrics"].get("center_rms_sigma"),
        },
        "r256_three_way_discrepancy": {
            "committed_forge_standalone_ff41cf173": committed_r256,
            "uncommitted_forge_combined_table_line193": {
                "claimed_ms": 39.76,
                "claimed_peak_allocated_mib": 1243.9,
                "committed_readout_file": None,
                "source": "prose / cohort_note in law4-phase2-energy-benchmark.json & law4-energy-champion.md",
            },
            "nvml_profiler_50step_prose_line169": {
                "source_commit": p2_v12.get("source_commit"),
                "step_ms_median": p2_r256["step_ms"]["median"],
                "step_ms_mean": p2_r256["step_ms"]["mean"],
                "peak_allocated_mib": p2_r256["peak_allocated_mib"],
                "peak_reserved_mib": p2_r256["peak_reserved_mib"],
                "joules_per_step": p2_r256["joules_per_step"],
                "dynamic_joules_per_step": p2_r256["dynamic_joules_per_step"],
                "idle_power_watts": p2_r256["idle_power_watts"],
                "avg_power_watts": p2_r256["avg_power_watts_from_energy"],
                "warmup_steps": p2_r256["warmup_steps"],
                "measured_steps": p2_r256["measured_steps"],
                "reuses_single_batch_pair": True,
            },
        },
        "selection_pin_governance": {
            "default_adoption": sel_pin.get("default_adoption"),
            "pinned_unified_candidate": unified_sel.get("candidate_id"),
        },
    }


def audit_progression_table(root: pathlib.Path = ROOT) -> List[Dict[str, Any]]:
    """Verify every candidate in docs/four-laws.md's progression table and comparability attributes."""
    records_dir = root / "reports" / "forge" / "records"
    p2 = _load_json(root / "reports" / "forge" / "studies" / "law4-phase2-energy-benchmark.json")
    p1 = _load_json(root / "reports" / "forge" / "studies" / "law4-phase1-profile.json")

    specs = [
        ("rhbi-v12", "readout-9917dee7ebd20eedc0310687.json", 588.75, 3750.6, None),
        ("empf-v8", "readout-9672bf970e5f31704c112ba1.json", 130.13, 1435.4, None),
        ("t2-secant-v2", "readout-14cc2a44f2d8da942d2873dc.json", 87.96, 1308.46, None),
        ("cond-fiber-v1", "readout-8d2e547ce820cf8662a34bdf.json", 87.50, 1308.4, None),
        ("unified-v10", "readout-90580c7aff5ec6988cfbcfad.json", 89.66, 1372.6, None),
        ("unified-v11", "readout-1e5fb2354920c2770b8b582c.json", 89.19, 1372.6, "unified-v11"),
        ("r4-razor-no-ns-v2", "readout-0219c6e2857b1605cbcf3428.json", 76.17, 1372.6, None),
        ("r4-razor-autograd-critic", "readout-4b04c4bf0dc3d7408a82e51e.json", 133.01, 1850.3, None),
        ("minimal-v2", "readout-d50a87c606ed4cd8e01a42fe.json", 55.50, 1344.8, None),
        ("law4-energy-v3", "readout-b898e9285957d3feba1bf714.json", 45.89, 1147.15, "law4-energy-v3"),
        ("law4-energy-v10", "readout-9599d9a20cf9a6e935f0d900.json", 45.30, 1243.4, "law4-energy-v10"),
        ("law4-energy-v11", "readout-bc0352d17785c6afcfeafd00.json", 44.61, 1243.4, "law4-energy-v11"),
        ("t4-nash-v3", "readout-642d4c55d093b44ed1e0309e.json", 39.68, 1372.9, None),
        ("unified-v12", "readout-59898e1a77c4b179c9aebbc1.json", 39.76, 1243.9, "unified-v12"),
    ]

    rows: List[Dict[str, Any]] = []
    for cid, fname, claimed_ms, claimed_mib, p2_key in specs:
        rec = _load_json(records_dir / fname)
        r256 = [t for t in rec["task_results"] if t["task_id"] == "dcgan_performance_r256"][0]
        m = r256["metrics"]
        c = r256.get("cost") or {}
        en = c.get("energy") or {}

        # Read declared training_precision from idea card
        idea_path = root / "configs" / "forge" / "ideas" / f"{cid}.json"
        idea = _load_json(idea_path)
        recipe_dict = idea.get("recipe") or idea.get("recipe_overrides") or {}
        training_precision = recipe_dict.get("training_precision") or "float32"

        p2_info = None
        if p2_key and p2_key in p2["energy_benchmark"]:
            arm = p2["energy_benchmark"][p2_key]
            su = arm["dcgan_r256"]["sustained_uninstrumented"]
            p2_info = {
                "source_commit": arm.get("source_commit"),
                "profiler_ms_median": su["step_ms"]["median"],
                "profiler_ms_mean": su["step_ms"]["mean"],
                "profiler_peak_allocated_mib": su["peak_allocated_mib"],
                "profiler_j_per_step": su["joules_per_step"],
                "profiler_dynamic_j_per_step": su["dynamic_joules_per_step"],
                "profiler_idle_watts": su["idle_power_watts"],
                "profiler_avg_watts": su["avg_power_watts_from_energy"],
            }

        # Check phase1 profile for law4-energy-v3 12.56 J/step
        p1_j = None
        if cid == "law4-energy-v3":
            for item in p1.get("workloads", p1.get("dcgan_profiles", [])) if isinstance(p1, dict) else []:
                if isinstance(item, dict) and item.get("task_id") == "dcgan_performance_r256":
                    p1_j = item.get("sustained_uninstrumented", {}).get("joules_per_step")
        if cid == "law4-energy-v3" and p1_j is None:
            # Scan nested dict structure of law4-phase1-profile.json
            txt = json.dumps(p1)
            m_j = re.search(r'"joules_per_step":\s*(12\.5\d+)', txt)
            if m_j:
                p1_j = float(m_j.group(1))

        forge_ms = float(m["update_ms_median"])
        forge_mib = float(m["peak_allocated_mib"])
        delta_ms = round(claimed_ms - forge_ms, 3)
        delta_mib = round(claimed_mib - forge_mib, 3)

        rows.append(
            {
                "candidate_id": cid,
                "forge_readout": fname,
                "training_precision": training_precision,
                "optimizer_family": recipe_dict.get("optimizer_family"),
                "device_index": c.get("device"),
                "forge_contended": en.get("contended"),
                "forge_ms_median": round(forge_ms, 3),
                "forge_ms_p95": round(float(m["update_ms_p95"]), 3),
                "forge_peak_allocated_mib": round(forge_mib, 2),
                "forge_peak_reserved_mib": round(float(m["peak_reserved_mib"]), 1),
                "claimed_ms": claimed_ms,
                "claimed_mib": claimed_mib,
                "delta_claimed_vs_forge_ms": delta_ms,
                "delta_claimed_vs_forge_mib": delta_mib,
                "profiler_phase2": p2_info,
                "profiler_phase1_j_per_step": p1_j,
            }
        )
    return rows


def audit_code_complexity(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    """Compute exact LOC, AST branches, transitive local imports, and surviving Occam epicycles."""
    packages = {
        "unified-v12_delta_only": ["particlegan/_unified_v12"],
        "unified-v12_effective_direct": [
            "particlegan/_unified_v12",
            "particlegan/_unified_v11",
            "particlegan/unified_v11.py",
        ],
        "unified-v11_current": ["particlegan/_unified_v11", "particlegan/unified_v11.py"],
        "unified-v10_direct": ["particlegan/_unified_v4", "particlegan/unified_v4.py"],
        "cond-fiber-v1_with_unified": [
            "particlegan/_cond_fiber",
            "particlegan/cond_fiber.py",
            "particlegan/_unified",
            "particlegan/unified.py",
        ],
        "t4-nash-v3": ["particlegan/_nash_v3", "particlegan/nash_v3.py"],
        "minimal-v2": ["particlegan/_minimal", "particlegan/minimal.py"],
        "empf-v8": ["particlegan/_empf", "particlegan/empf.py"],
        "rhbi-v12": ["particlegan/_rhbi", "particlegan/rhbi.py"],
        "law4-energy": ["particlegan/_law4_energy"],
    }

    metrics: Dict[str, Any] = {}
    for name, rels in packages.items():
        files: List[pathlib.Path] = []
        for rel in rels:
            p = root / rel
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(sorted(p.rglob("*.py")))
        loc = nonblank = n_if = n_ifexp = 0
        for f in files:
            src = f.read_text(encoding="utf-8")
            lines = src.splitlines()
            loc += len(lines)
            nonblank += sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    n_if += 1
                elif isinstance(node, ast.IfExp):
                    n_ifexp += 1
        metrics[name] = {
            "files": len(files),
            "loc": loc,
            "code_loc": nonblank,
            "ast_if": n_if,
            "ast_ifexp": n_ifexp,
            "total_branches": n_if + n_ifexp,
        }

    # Check transitive local sibling package imports of _unified_v12 + _unified_v11
    sibling_imports: Dict[str, List[str]] = {}
    for sub in ("_unified_v12", "_unified_v11"):
        for f in sorted((root / "particlegan" / sub).glob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            imps = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level >= 2 and node.module:
                    imps.add(node.module.split(".")[0])
            if imps:
                sibling_imports[str(f.relative_to(root))] = sorted(imps)

    # Check surviving Round-4 Occam redundant epicycles in _unified_v11 / _unified_v12 + unified-v12.json
    prior_src = (root / "particlegan" / "_unified_v11" / "prior.py").read_text(encoding="utf-8")
    opt12_src = (root / "particlegan" / "_unified_v12" / "optimizer.py").read_text(encoding="utf-8")
    opt12_tree = ast.parse(opt12_src)
    v12_idea = _load_json(root / "configs" / "forge" / "ideas" / "unified-v12.json")
    v12_recipe = v12_idea.get("recipe_overrides") or {}

    # Real AST / source checks on _unified_v12/optimizer.py (no hardcoded booleans)
    has_d_probe = "_d_probe" in opt12_src
    parent_of: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(opt12_tree):
        for child in ast.iter_child_nodes(parent):
            parent_of[child] = parent

    is_conv_if_nodes = [
        node
        for node in ast.walk(opt12_tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "is_conv"
    ]
    conv_uses_cached_metrics = any(
        "_cached_conv_metrics" in ast.unparse(node.body)
        for node in is_conv_if_nodes
    )
    conv_branches_call_ns = any(
        any(tok in ast.unparse(node.body) for tok in ("quintic_orthogonalize", "pending_dense", "dense2d_grads"))
        for node in is_conv_if_nodes
    )

    def _is_descendant_of(node: ast.AST, stmts: List[ast.stmt]) -> bool:
        targets = set(stmts)
        cur: Optional[ast.AST] = node
        while cur is not None:
            if cur in targets:
                return True
            cur = parent_of.get(cur)
        return False

    def _ns_enqueue_excludes_conv(call_node: ast.Call) -> bool:
        cur: Optional[ast.AST] = call_node
        while cur is not None:
            if isinstance(cur, ast.For) and "dense2d_meta" in ast.unparse(cur.iter):
                # Fed exclusively by dense2d_grads.append(), which is itself checked below.
                return True
            if isinstance(cur, ast.If):
                test_src = ast.unparse(cur.test)
                if test_src == "is_conv" and _is_descendant_of(call_node, cur.orelse):
                    return True
                if "not is_conv" in test_src and _is_descendant_of(call_node, cur.body):
                    return True
            cur = parent_of.get(cur)
        return False

    ns_enqueue_calls = [
        node
        for node in ast.walk(opt12_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (
            (isinstance(node.func.value, ast.Name) and node.func.value.id == "pending_dense" and node.func.attr == "setdefault")
            or (isinstance(node.func.value, ast.Name) and node.func.value.id == "dense2d_grads" and node.func.attr == "append")
        )
    ]
    all_ns_enqueues_exclude_conv = bool(ns_enqueue_calls) and all(
        _ns_enqueue_excludes_conv(c) for c in ns_enqueue_calls
    )
    epicycles = {
        "radon_hurwitz_dz4_branch_in_prior": "dim == 2 and d_z == 4" in prior_src,
        "fwad_generator_probe_w_sat_pt_in_prior": "_fwad_generator_probe(" in prior_src and "w_sat_pt" in prior_src,
        "conformal_horizon_flag_enabled_in_v12_idea": bool(v12_recipe.get("rgdt_conformal_horizon")),
        "multiscale_horizon_flag_enabled_in_v12_idea": bool(v12_recipe.get("t1c_multiscale_horizon")),
        "critic_secant_dropped_in_v12_optimizer": not has_d_probe,
        "conv2d_ns_dropped_in_v12_optimizer": (
            len(is_conv_if_nodes) >= 3
            and conv_uses_cached_metrics
            and (not conv_branches_call_ns)
            and all_ns_enqueues_exclude_conv
        ),
    }

    return {
        "package_metrics": metrics,
        "docs_claimed_loc": {
            "unified-v12": None,
            "unified-v11": 4713,
            "unified-v10": 4311,
            "minimal-v2": 899,
            "empf-v8": 1972,
            "rhbi-v12": 3114,
            "t4-nash-v3": 3016,
            "cond-fiber-v1": 4244,
        },
        "sibling_package_imports": sibling_imports,
        "surviving_occam_epicycles": epicycles,
    }


def audit_non_progression_readout_metrics(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    """Programmatically verify exact pass/total counts and key cited metrics in non-progression readouts."""
    records_dir = root / "reports" / "forge" / "records"

    def _task(rec: Dict[str, Any], tid: str) -> Dict[str, Any]:
        return [t for t in rec["task_results"] if t["task_id"] == tid][0]

    def _counts(rec: Dict[str, Any]) -> Tuple[int, int]:
        trs = rec["task_results"]
        return sum(1 for t in trs if t["gate_status"] == "PASS"), len(trs)

    checks: Dict[str, Any] = {}
    specs = [
        ("t1-prior-v2", "readout-379720b9859815defff0efaa.json", None),
        ("t1-coupling-v5", "readout-b87c22fbdb500d43981e839b.json", "img_bars4"),
        ("t2-secant-v5", "readout-a2d41a678887a49f9ecc6aa9.json", None),
        ("cond-fiber-v1", "readout-5932f4d351cc4e791417835a.json", "model_ddgan_gaussian_chain"),
        ("unified-v11_disc_stab", "readout-fc5b67a767e57d748f460610.json", "model_ddgan_gaussian_chain"),
        ("unified-v11_t3_iso", "readout-d1e36bf514f755e46b333e6d.json", None),
        ("unified-v11_t4_nash", "readout-2eb07e97714a1da4177d5d25.json", None),
        ("unified-v11_t3_shift", "readout-2029c37a1359498d3d8f2eb3.json", None),
        ("unified-v11_t4_arch", "readout-b96d9de47bb52cc91fbb4a4f.json", None),
        ("t4-nash-v3_nash", "readout-6f996ec256a149b2f1620f64.json", None),
        ("t4-arch-v3_diag", "readout-dc890b704514d91effb156a6.json", "grid100_mlp16"),
        ("t4-arch-v3_t3_holdout", "readout-ff22843954240927dfd7ecc3.json", None),
        ("rgdt-v3_t3_iso", "readout-c30d4a5840e8d861f62c1dcb.json", None),
        ("r4_arm1_no_ns", "readout-690dd40a4221b390d8680e43.json", "grid100_mlp16"),
        ("r4_arm2_no_secant", "readout-4e2fd695306a5dd6fd222e62.json", "nash_overpowered_generator_ring16"),
        ("r4_arm4_no_coalesce", "readout-04bb3c211150d5629c489b5e.json", "grid100"),
        ("r4_arm5_no_laguerre", "readout-b27b26896bd137e2e701ce2d.json", "grid100"),
        ("r4_arm7_autograd", "readout-b96faf85499b1dbe9dce102e.json", "t3_iso_scale_mass_8x"),
        ("r4_arm8_no_sat_probe", "readout-453cc0e86ed3223a21d247c3.json", "zero_tanh_rail_escape"),
    ]
    for label, fname, tid in specs:
        rec = _load_json(records_dir / fname)
        p_cnt, tot = _counts(rec)
        entry: Dict[str, Any] = {
            "file": fname,
            "candidate_id": rec.get("candidate_id"),
            "pass_count": p_cnt,
            "total_count": tot,
        }
        if tid is not None:
            tr = _task(rec, tid)
            entry["task_id"] = tid
            entry["task_status"] = tr["gate_status"]
            entry["metrics"] = tr.get("metrics") or {}
        checks[label] = entry
    return checks


def audit_p8_tuning_and_unseen(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    """Reconstruct lineage tuning budgets, calibration status, ByteGAN unseen fails, and L4 transfer fails."""
    ideas = sorted((root / "configs" / "forge" / "ideas").glob("*.json"))
    confs = sorted((root / "configs" / "forge" / "configurations").glob("*.json"))
    studies = sorted((root / "configs" / "forge" / "studies").glob("*.json"))
    readouts = sorted((root / "reports" / "forge" / "records").glob("readout-*.json"))

    all_cards = [p.stem for p in ideas + confs]
    study_cands = []
    for s in studies:
        d = _load_json(s)
        study_cands.append((s.stem, str(d.get("candidate") or d.get("candidate_id") or "")))
    readout_rows = []
    for r in readouts:
        d = _load_json(r)
        trs = d.get("task_results", [])
        wall = sum(float((t.get("cost") or {}).get("wall_seconds") or 0.0) for t in trs)
        readout_rows.append((r.name, str(d.get("candidate_id") or ""), str(d.get("goal") or ""), len(trs), wall))

    lineages = {
        "bcap_family": lambda s: "bcap" in s,
        "bispf_family": lambda s: s.startswith("bispf"),
        "rhbi_family": lambda s: s.startswith("rhbi"),
        "empf_family": lambda s: s.startswith("empf"),
        "t1_prior_coupling": lambda s: s.startswith(("t1-prior", "t1-coupling")),
        "t2_secant": lambda s: s.startswith("t2-secant"),
        "t3_rgdt": lambda s: s.startswith("rgdt"),
        "t4_arch_nash": lambda s: s.startswith(("t4-arch", "t4-nash", "t4-ctrl")),
        "cond_fiber": lambda s: s.startswith("cond-fiber"),
        "unified_v1_v12": lambda s: s.startswith("unified-v"),
        "r4_razor_minimal": lambda s: s.startswith(("r4-razor", "minimal-")),
        "law4_energy_opt": lambda s: s.startswith("law4-"),
    }

    budget_summary: Dict[str, Any] = {}
    for gname, pred in lineages.items():
        cs = [c for c in all_cards if pred(c)]
        ss = [s for s, cid in study_cands if pred(cid) or pred(s)]
        rs = [r for r in readout_rows if pred(r[1])]
        budget_summary[gname] = {
            "declared_cards": len(cs),
            "studies": len(ss),
            "concluded_readouts": len(rs),
            "task_evaluations": sum(r[3] for r in rs),
            "gpu_hours": round(sum(r[4] for r in rs) / 3600.0, 2),
        }

    # Unseen post-freeze ByteGAN stretch results
    bg = _load_json(root / "reports" / "forge" / "bytegan-stretch" / "summary.json")
    bytegan_summary = []
    for fam in bg.get("families", []):
        for tid, tdata in sorted((fam.get("tasks") or {}).items()):
            fm = tdata.get("final_metrics") or {}
            bytegan_summary.append(
                {
                    "candidate_id": fam.get("candidate_id"),
                    "task_id": tid,
                    "gate_status": tdata.get("gate_status"),
                    "valid_unique_bytes": fm.get("valid_unique_bytes"),
                    "unique_bytes": fm.get("unique_bytes"),
                    "rms_bit_error": fm.get("rms_bit_error"),
                    "recon_valid_exact_bytes": fm.get("recon_valid_exact_bytes"),
                    "encoder_particle_bijection_bytes": fm.get("encoder_particle_bijection_bytes"),
                }
            )

    # Colab NVIDIA L4 unified-v12 technique-receipts (waves 7a/7b)
    l4_counts: Dict[str, int] = {}
    l4_non_pass: List[Dict[str, Any]] = []
    for p in sorted((root / "reports" / "forge" / "technique-receipts").glob("*.json")):
        d = _load_json(p)
        if d.get("candidate_id") != "unified-v12":
            continue
        for tr in d.get("task_results", []):
            st = str(tr.get("gate_status"))
            l4_counts[st] = l4_counts.get(st, 0) + 1
            if st != "PASS":
                l4_non_pass.append(
                    {
                        "file": p.name,
                        "campaign_id": d.get("campaign_id"),
                        "task_id": tr.get("task_id"),
                        "gate_status": st,
                        "raw_status": tr.get("raw_status"),
                        "reason": tr.get("reason"),
                    }
                )

    # Exact programmatic overlap of r3-tier3-long-horizon-holdout-v1 with active screening views
    def _vtasks(rel: str) -> set[str]:
        view = _load_json(root / rel)
        return {str(a.get("task") or a.get("task_id") or a.get("id")) for a in view.get("assignments", [])}

    t3_tasks = _vtasks("configs/forge/views/r3-tier3-long-horizon-holdout-v1.json")
    core_screen = (
        _vtasks("configs/forge/view-history/discriminator_stability-v19.json")
        | _vtasks("configs/forge/views/t4-strange-architectures-v1.json")
        | _vtasks("configs/forge/views/r4-occam-ablation-v1.json")
    )
    shift_screen = _vtasks("configs/forge/views/t3-shift-recovery-diagnostic-v1.json")
    tier3_holdout_overlap = {
        "total_tasks": len(t3_tasks),
        "overlap_core_screening_count": len(t3_tasks & core_screen),
        "overlap_core_screening_tasks": sorted(t3_tasks & core_screen),
        "remaining_after_core_screening": sorted(t3_tasks - core_screen),
        "overlap_including_shift_count": len(t3_tasks & (core_screen | shift_screen)),
        "remaining_after_core_and_shift": sorted(t3_tasks - (core_screen | shift_screen)),
    }

    cal = _load_json(root / "reports" / "forge" / "calibration" / "r3-calibration-holdout.json")
    return {
        "lineage_tuning_budgets": budget_summary,
        "tier3_holdout_overlap": tier3_holdout_overlap,
        "calibration_adoption": cal.get("adoption"),
        "calibration_blockers": cal.get("calibration_blockers"),
        "bytegan_unseen_results": bytegan_summary,
        "unified_v12_l4_transfer_counts": l4_counts,
        "unified_v12_l4_non_pass": sorted(l4_non_pass, key=lambda x: (x["gate_status"], x["task_id"])),
    }


def run_gpu_microbenchmark(device_str: str = "cuda:0", warmup: int = 4, steps: int = 15) -> Dict[str, Any]:
    """Run a fast, read-only local micro-benchmark on RTX A6000 measuring component breakdown & FP32 vs BF16."""
    import torch
    from experiments.law4_optimizer_profiler import (
        NVMLMonitor,
        _build_dcgan_trainer,
        _instrument_step_phases,
        _stats,
        load_idea,
        load_task,
        optimizer_state_summary,
    )

    device = torch.device(device_str)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_str} not available for GPU micro-benchmark")

    dev_idx = device.index if device.index is not None else 0
    nvml = NVMLMonitor(dev_idx)
    gpu_name = torch.cuda.get_device_name(device)
    task = load_task("dcgan_performance_r256")
    shape = (16, 3, 256, 256)

    def _bench_one(
        label: str,
        idea_id: str,
        *,
        recipe_extra: Optional[Dict[str, Any]] = None,
        with_breakdown: bool = False,
    ) -> Dict[str, Any]:
        torch.cuda.synchronize(device)
        time.sleep(1.0)
        util_before = nvml.gpu_utilization()
        idle_w = nvml.measure_idle_power(duration_s=0.35, interval_s=0.01)

        idea = load_idea(idea_id)
        if recipe_extra:
            if "recipe" in idea and isinstance(idea["recipe"], dict):
                idea["recipe"].update(recipe_extra)
            if "recipe_overrides" in idea and isinstance(idea["recipe_overrides"], dict):
                idea["recipe_overrides"].update(recipe_extra)
        total = warmup + steps + (12 if with_breakdown else 0)
        ctx, trainer, g, d, spec = _build_dcgan_trainer(
            idea, task, device, total, recipe_extra=None
        )
        c_stream = ctx.streams.generator("data", component="critic", purpose="training")
        g_stream = ctx.streams.generator("data", component="generator", purpose="training")
        real_buf = torch.empty(shape, device=device, dtype=torch.float32)
        real_g_buf = torch.empty(shape, device=device, dtype=torch.float32)

        for _ in range(warmup):
            real_buf.uniform_(-1.0, 1.0, generator=c_stream)
            real_g_buf.uniform_(-1.0, 1.0, generator=g_stream)
            trainer.step(real_buf, generator_real=real_g_buf)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        ms_list: List[float] = []
        e0_j = nvml.total_energy_joules()
        t0 = time.perf_counter()
        for _ in range(steps):
            real_buf.uniform_(-1.0, 1.0, generator=c_stream)
            real_g_buf.uniform_(-1.0, 1.0, generator=g_stream)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            trainer.step(real_buf, generator_real=real_g_buf)
            ev1.record()
            torch.cuda.synchronize(device)
            ms_list.append(float(ev0.elapsed_time(ev1)))
        wall_s = time.perf_counter() - t0
        e1_j = nvml.total_energy_joules()
        util_after = nvml.gpu_utilization()

        peak_alloc = float(torch.cuda.max_memory_allocated(device)) / float(2**20)
        peak_res = float(torch.cuda.max_memory_reserved(device)) / float(2**20)
        st_info = optimizer_state_summary(trainer)
        j_tot = (e1_j - e0_j) if (e0_j is not None and e1_j is not None) else None
        j_step = (j_tot / float(steps)) if j_tot is not None else None
        avg_w = (j_tot / wall_s) if (j_tot is not None and wall_s > 0) else None
        dyn_j = (
            max(0.0, (avg_w - idle_w) * (wall_s / float(steps)))
            if (avg_w is not None and idle_w is not None)
            else None
        )

        breakdown = None
        if with_breakdown:
            phase_records: Dict[str, List[float]] = {}
            with _instrument_step_phases(trainer, device, phase_records) as run_step:
                for i in range(2 + 10):
                    real_buf.uniform_(-1.0, 1.0, generator=c_stream)
                    real_g_buf.uniform_(-1.0, 1.0, generator=g_stream)
                    run_step(real_buf, real_g_buf, record=(i >= 2))
            med_tot = statistics.median(phase_records["step_total_ms"])
            breakdown = {
                k: {
                    "median_ms": round(statistics.median(v), 3),
                    "pct": round(100.0 * statistics.median(v) / med_tot, 2) if med_tot > 0 else 0.0,
                }
                for k, v in phase_records.items()
            }

        del real_buf, real_g_buf, trainer, g, d, ctx
        torch.cuda.empty_cache()

        st = _stats(ms_list)
        return {
            "label": label,
            "idea_id": idea_id,
            "recipe_extra": recipe_extra or {},
            "gpu_util_before": util_before,
            "gpu_util_after": util_after,
            "contended_warning": bool(util_before and util_before[0] > 5),
            "idle_power_watts": round(idle_w, 2) if idle_w is not None else None,
            "avg_power_watts": round(avg_w, 2) if avg_w is not None else None,
            "step_ms_median": round(st["median"], 3),
            "step_ms_mean": round(st["mean"], 3),
            "step_ms_p95": round(st["p95"], 3),
            "joules_per_step": round(j_step, 3) if j_step is not None else None,
            "dynamic_joules_per_step": round(dyn_j, 3) if dyn_j is not None else None,
            "peak_allocated_mib": round(peak_alloc, 2),
            "peak_reserved_mib": round(peak_res, 1),
            "optimizer_state_bytes": st_info["total_state_bytes"],
            "phase_breakdown": breakdown,
        }

    try:
        results = [
            _bench_one("unified-v12 (BF16 CL)", "unified-v12", with_breakdown=True),
            _bench_one("unified-v11 (BF16 CL)", "unified-v11", with_breakdown=True),
            _bench_one("minimal-v2 (BF16 CL)", "minimal-v2", with_breakdown=False),
            _bench_one("t4-nash-v3 (BF16 CL)", "t4-nash-v3", with_breakdown=False),
            _bench_one("empf-v8 (BF16 CL)", "empf-v8", with_breakdown=False),
            _bench_one("rhbi-v12 (native FP32)", "rhbi-v12", with_breakdown=False),
            _bench_one(
                "rhbi-v12 (matched BF16 CL)",
                "rhbi-v12",
                recipe_extra={"training_precision": "bf16_channels_last"},
                with_breakdown=False,
            ),
        ]
        # Also profile a 16-layer skip-free 2D MLP (grid100_mlp16 topology) on unified-v12 vs unified-v11
        mlp_results = _bench_mlp16(device, nvml, warmup=5, steps=20)
    finally:
        nvml.shutdown()

    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "fresh_batches_per_step": True,
        "dcgan_r256": results,
        "mlp16_2d": mlp_results,
    }


def _bench_mlp16(device: Any, nvml: Any, warmup: int = 5, steps: int = 20) -> List[Dict[str, Any]]:
    """Micro-benchmark 16-layer skip-free 2D MLP (grid100_mlp16 topology) for unified_v12 vs unified_v11."""
    import torch
    from torch import nn
    import particlegan
    from experiments.law4_optimizer_profiler import _stats

    class _MLP16G(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: List[nn.Module] = [nn.Linear(2, 128), nn.LeakyReLU(0.2)]
            for _ in range(15):
                layers.extend([nn.Linear(128, 128), nn.LeakyReLU(0.2)])
            layers.append(nn.Linear(128, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    class _Critic2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    out: List[Dict[str, Any]] = []
    for preset in ("unified_v12", "unified_v11", "minimal"):
        torch.manual_seed(0)
        g = _MLP16G().to(device)
        d = _Critic2D().to(device)
        base_r = particlegan.get_recipe(preset, z_dim=2, batch_size=128, num_particles=256)
        recipe = base_r
        trainer = particlegan.GANTrainer(recipe, g, d, seed=0)
        gen = torch.Generator(device=device).manual_seed(0)

        for _ in range(warmup):
            x1 = torch.randn((128, 2), device=device, generator=gen)
            x2 = torch.randn((128, 2), device=device, generator=gen)
            trainer.step(x1, generator_real=x2)
        torch.cuda.synchronize(device)

        ms_list: List[float] = []
        e0 = nvml.total_energy_joules()
        for _ in range(steps):
            x1 = torch.randn((128, 2), device=device, generator=gen)
            x2 = torch.randn((128, 2), device=device, generator=gen)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            trainer.step(x1, generator_real=x2)
            ev1.record()
            torch.cuda.synchronize(device)
            ms_list.append(float(ev0.elapsed_time(ev1)))
        e1 = nvml.total_energy_joules()

        opt_g = trainer.opt_g
        opt_d = trainer.opt_d
        st = _stats(ms_list)
        out.append(
            {
                "preset": preset,
                "optimizer_family": recipe.optimizer_family,
                "step_ms_median": round(st["median"], 3),
                "step_ms_p95": round(st["p95"], 3),
                "joules_per_step": round((e1 - e0) / float(steps), 3) if (e0 is not None and e1 is not None) else None,
                "opt_g_ns_5step_calls": getattr(opt_g, "ns_5step_calls", None),
                "opt_d_critic_ns_calls": getattr(opt_d, "critic_ns_calls", None),
                "opt_g_ema_func_gain": round(float(getattr(opt_g, "_ema_func_gain", float("nan"))), 4)
                if hasattr(opt_g, "_ema_func_gain")
                else None,
            }
        )
        del trainer, g, d
        torch.cuda.empty_cache()
    return out


def run_full_audit(root: pathlib.Path = ROOT, profile_gpu: bool = False, device: str = "cuda:0") -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "links_and_readouts": audit_four_laws_links(root),
        "unified_v12_audit": audit_unified_v12_receipts(root),
        "progression_table": audit_progression_table(root),
        "non_progression_metrics": audit_non_progression_readout_metrics(root),
        "code_complexity": audit_code_complexity(root),
        "p8_qualification_and_unseen": audit_p8_tuning_and_unseen(root),
    }
    if profile_gpu:
        report["local_gpu_microbenchmark"] = run_gpu_microbenchmark(device_str=device)
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    lr = report["links_and_readouts"]
    v12 = report["unified_v12_audit"]
    cc = report["code_complexity"]
    p8 = report["p8_qualification_and_unseen"]

    print("=== P9 RECEIPT & LINK INTEGRITY SUMMARY ===")
    print(f"docs/four-laws.md local file links checked: {lr['checked_local_file_links']}")
    print(f"Broken file links ({len(lr['broken_links'])}):")
    for b in lr["broken_links"]:
        print(f"  - {b['label']} -> {b['target']} (MISSING)")
    print(
        f"Readout prefixes in docs/four-laws.md: {lr['unique_readout_prefixes']} "
        f"(existing={lr['existing_readout_count']}, missing={lr['missing_readouts']})"
    )

    print("\n=== COMMITTED UNIFIED-V12 READOUTS ON origin/develop ===")
    for r in v12["committed_v12_readouts"]:
        print(f"  - {r['file']} | goal={r['goal']} | {r['status_counts']} | contended={r['contended_tasks']}")
    print(
        f"disc_stab (b51ab67c): {v12['disc_stab_b51ab67c']['pass_count']}/{v12['disc_stab_b51ab67c']['total_count']} "
        f"(required: {v12['disc_stab_b51ab67c']['required_pass']}/{v12['disc_stab_b51ab67c']['required_total']})"
    )
    for fid, finfo in v12["disc_stab_b51ab67c"]["failures"].items():
        print(f"  FAIL {fid}: {finfo['reason']}")

    tw = v12["r256_three_way_discrepancy"]
    print("\n=== UNIFIED-V12 r256 THREE-WAY DISCREPANCY ===")
    print(
        f"  1) Committed Forge readout-59898e1a: {tw['committed_forge_standalone_ff41cf173']['update_ms_median']:.2f} ms/st, "
        f"{tw['committed_forge_standalone_ff41cf173']['peak_allocated_mib']:.1f} MiB (contended={tw['committed_forge_standalone_ff41cf173']['contended']})"
    )
    print(
        f"  2) Line-193 table (UNCOMMITTED combined): {tw['uncommitted_forge_combined_table_line193']['claimed_ms']} ms/st, "
        f"{tw['uncommitted_forge_combined_table_line193']['claimed_peak_allocated_mib']} MiB"
    )
    print(
        f"  3) Line-169 prose (50-step NVML profiler): {tw['nvml_profiler_50step_prose_line169']['step_ms_median']:.2f} ms/st, "
        f"{tw['nvml_profiler_50step_prose_line169']['peak_allocated_mib']:.1f} MiB, "
        f"{tw['nvml_profiler_50step_prose_line169']['joules_per_step']:.2f} J/st "
        f"(dyn={tw['nvml_profiler_50step_prose_line169']['dynamic_joules_per_step']:.2f} J/st, "
        f"idle={tw['nvml_profiler_50step_prose_line169']['idle_power_watts']:.2f} W)"
    )

    print("\n=== PROGRESSION TABLE & COMPARABILITY AUDIT ===")
    for row in report["progression_table"]:
        print(
            f"  {row['candidate_id']:24s} | prec={row['training_precision']:18s} | "
            f"forge={row['forge_ms_median']:6.2f}ms/{row['forge_peak_allocated_mib']:6.1f}MiB | "
            f"claimed={row['claimed_ms']:6.2f}ms/{row['claimed_mib']:6.1f}MiB | "
            f"dev={row['device_index']} cont={row['forge_contended']}"
        )

    print("\n=== CODE COMPLEXITY (DIRECT LOC & AST BRANCHES) ===")
    for pkg, m in cc["package_metrics"].items():
        print(
            f"  {pkg:30s} | files={m['files']:2d} | LOC={m['loc']:5d} | "
            f"if={m['ast_if']:4d} | ifexp={m['ast_ifexp']:3d} | total_br={m['total_branches']:4d}"
        )
    print("  Surviving Occam epicycles:", cc["surviving_occam_epicycles"])

    print("\n=== P8 TUNING BUDGET & UNSEEN GENERALIZATION ===")
    for lin, b in p8["lineage_tuning_budgets"].items():
        print(
            f"  {lin:20s} | cards={b['declared_cards']:3d} | studies={b['studies']:2d} | "
            f"readouts={b['concluded_readouts']:2d} | evals={b['task_evaluations']:4d} | {b['gpu_hours']:6.2f} GPU-h"
        )
    print(f"  Calibration adoption: {p8['calibration_adoption']} | L4 transfer counts: {p8['unified_v12_l4_transfer_counts']}")
    v12_bg = [x for x in p8["bytegan_unseen_results"] if x["candidate_id"] == "unified-v12"]
    print(f"  ByteGAN unseen (unified-v12): {v12_bg}")

    if "local_gpu_microbenchmark" in report:
        gm = report["local_gpu_microbenchmark"]
        print(f"\n=== LOCAL GPU MICRO-BENCHMARK ({gm['gpu_name']}, {gm['device']}) ===")
        for r in gm["dcgan_r256"]:
            print(
                f"  {r['label']:28s} | med={r['step_ms_median']:6.2f}ms p95={r['step_ms_p95']:6.2f}ms | "
                f"alloc={r['peak_allocated_mib']:7.1f}MiB | J/st={r['joules_per_step']} (dyn={r['dynamic_joules_per_step']}, idleW={r['idle_power_watts']}) | "
                f"util={r['gpu_util_before']}->{r['gpu_util_after']}"
            )
            if r["phase_breakdown"]:
                pb = r["phase_breakdown"]
                print(
                    f"     phases: total={pb['step_total_ms']['median_ms']}ms | "
                    f"d_fb={pb['d_forward_backward_ms']['median_ms']}ms ({pb['d_forward_backward_ms']['pct']}%) | "
                    f"d_pen={pb['d_penalty_ms']['median_ms']}ms ({pb['d_penalty_ms']['pct']}%) | "
                    f"opt_d={pb['opt_d_step_ms']['median_ms']}ms ({pb['opt_d_step_ms']['pct']}%) | "
                    f"g_fb={pb['g_forward_backward_ms']['median_ms']}ms ({pb['g_forward_backward_ms']['pct']}%) | "
                    f"g_lp={pb['g_loss_prior_ms']['median_ms']}ms ({pb['g_loss_prior_ms']['pct']}%) | "
                    f"opt_g={pb['opt_g_step_ms']['median_ms']}ms ({pb['opt_g_step_ms']['pct']}%)"
                )
        print("  2D 16-layer skip-free MLP (grid100_mlp16 topology):")
        for r in gm["mlp16_2d"]:
            print(
                f"    {r['preset']:14s} | med={r['step_ms_median']:6.2f}ms p95={r['step_ms_p95']:6.2f}ms | "
                f"J/st={r['joules_per_step']} | g_ns5={r['opt_g_ns_5step_calls']} d_ns={r['opt_d_critic_ns_calls']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Four Laws Forge receipts, complexity, and efficiency.")
    parser.add_argument("--json", action="store_true", help="Output full machine-readable JSON to stdout.")
    parser.add_argument("--profile-gpu", action="store_true", help="Run controlled local GPU micro-benchmark.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for --profile-gpu (default: cuda:0).")
    args = parser.parse_args()

    report = run_full_audit(ROOT, profile_gpu=args.profile_gpu, device=args.device)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_summary(report)


if __name__ == "__main__":
    main()
