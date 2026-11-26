#!/usr/bin/env python3
"""Parallel, crash-proof Google Colab GPU orchestrator & decentralized cache catch-up for Forge.

Designed for multi-agent branch development without a central coordinator:
- Branch A adds a new experiment/task -> `catchup` fills (families x new_task).
- Branch B adds a new family/candidate -> `catchup` fills (new_family x tasks).
- Branch C merges A + B -> `catchup` detects ONLY (new_family x new_task) is missing,
  spins up just the right-sized Colab GPU(s), fills the gap, projects compact
  content-addressed receipts (`reports/forge/technique-receipts/<attempt_id>.json`),
  and optionally publishes `--inventory` locally to Hugging Face (zero HF credentials
  ever leave the local machine).

Safety guarantees baked in from live Colab testing:
1. Two-layer teardown: local `try/finally` + SIGINT/SIGTERM/atexit stops every VM
   immediately on finish/error/Ctrl-C; VM-side detached `watchdog.py` calls
   `POST http://$TBE_RUNTIME_ADDR/unassign` if the local machine crashes, a job
   stalls, or `--max-minutes` is exceeded.
2. OS-level `timeout` + SIGKILL around every `colab` CLI subprocess so a wedged
   `google-colab-cli` websocket thread can never leak an idle GPU.
3. Per-worker isolated `--config <file>` + staggered VM launches avoid
   `sessions.json.lock` contention and Colab assignment 503 bursts.
4. Warm worker pool: bootstraps ParticleGAN once per VM (~3s via `uv`) and drains
   multiple catch-up work units before tearing down idle VMs immediately.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import queue as stdlib_queue
import shutil
import signal
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Empirically measured Colab CU/hr rates & VRAM (2026-11-22) for planning summaries.
COLAB_GPU_SPECS: dict[str, dict[str, Any]] = {
    "cpu": {"cuda_model": None, "vram_gb": 0.0, "cu_per_hr": 0.07, "colab_flag": None},
    "T4": {"cuda_model": "Tesla T4", "vram_gb": 15.6, "cu_per_hr": 1.15, "colab_flag": "T4"},
    "L4": {"cuda_model": "NVIDIA L4", "vram_gb": 23.7, "cu_per_hr": 1.94, "colab_flag": "L4"},
    "G4": {"cuda_model": "NVIDIA G4", "vram_gb": 24.0, "cu_per_hr": 3.00, "colab_flag": "G4"},
    "A100": {"cuda_model": "NVIDIA A100-SXM4-40GB", "vram_gb": 40.0, "cu_per_hr": 6.80, "colab_flag": "A100"},
    "H100": {"cuda_model": "NVIDIA H100 80GB HBM3", "vram_gb": 80.0, "cu_per_hr": 12.00, "colab_flag": "H100"},
}

# Tasks that empirically approach or exceed wall-time caps on L4 (e.g. five_word_joint_smoke
# hit 900.2s INCOMPLETE on L4 for unified-v11) or are long Tier 3-4 endurance/deep-arch tasks.
HEAVY_TASK_PREFIXES = (
    "five_word_joint_",
    "ring_extension",
    "grid100_14k",
    "rotated100_14k",
    "staggered100_14k",
    "grid100_mlp16",
    "grid100_res16_",
)

DEFAULT_REPO_URL = "https://github.com/HyperGAN/ParticleGAN.git"
TRANSIENT_TRANSPORT_RETRY_RCS = frozenset({124})
DEFAULT_TRANSPORT_RETRIES = 1
DEFAULT_TRANSPORT_RETRY_BACKOFF_SECONDS = 3.0
DEFAULT_REAP_MAX_RUNS = 50
DEFAULT_REAP_STOP_TIMEOUT_SECONDS = 30.0


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def normalize_gpu(gpu: str) -> str:
    raw = gpu.strip()
    if raw.lower() == "auto":
        return "auto"
    if raw.lower() == "cpu":
        return "cpu"
    upper = raw.upper()
    if upper not in COLAB_GPU_SPECS:
        raise ValueError(f"Unsupported --gpu {gpu!r}; choose from auto, cpu, T4, L4, G4, A100, H100")
    return upper


def choose_gpu_for_task(
    task_id: str,
    task_card: dict[str, Any],
    *,
    tier: int = 1,
    gpu_policy: str = "auto",
    prefer_t4: bool = False,
) -> str:
    """Right-size the Colab accelerator for a Forge task card."""
    policy = normalize_gpu(gpu_policy)
    if policy != "auto":
        return policy

    res = task_card.get("resources") or {}
    exe = task_card.get("execution") or {}
    timeout_s = float(res.get("timeout_seconds") or 0)
    steps = int(exe.get("steps") or exe.get("additional_steps") or 0)

    if res.get("gpus", 1) == 0:
        return "cpu"
    if (
        tier >= 3
        or any(task_id.startswith(prefix) for prefix in HEAVY_TASK_PREFIXES)
        or timeout_s >= 3600
        or steps >= 14000
    ):
        return "A100"
    if prefer_t4 and tier == 1 and timeout_s > 0 and timeout_s <= 300 and steps <= 1000:
        return "T4"
    return "L4"


def run_with_hard_timeout(
    cmd: list[str],
    *,
    timeout_seconds: float,
    kill_grace: float = 10.0,
    stdin_text: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
    capture: bool = False,
) -> tuple[int, str]:
    """Run a subprocess in its own process group with guaranteed SIGTERM -> SIGKILL timeout.

    Crucial for `google-colab-cli`, whose websocket background thread can hang indefinitely
    after printing `RuntimeError: Connection was lost.` instead of exiting.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    chunks: list[str] = []
    log_handle = log_file.open("a", encoding="utf-8") if log_file else None
    writer_lock = threading.Lock()

    def _kill_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if capture:
                chunks.append(line)
            if log_handle is not None:
                with writer_lock:
                    log_handle.write(line)
                    log_handle.flush()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        if stdin_text is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            rc = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            msg = f"[{utc_stamp()}] [timeout] Hard-killing hung command after {timeout_seconds}s: {' '.join(cmd[:6])}\n"
            if capture:
                chunks.append(msg)
            if log_handle is not None:
                with writer_lock:
                    log_handle.write(msg)
                    log_handle.flush()
            _kill_group(signal.SIGTERM)
            try:
                proc.wait(timeout=kill_grace)
            except subprocess.TimeoutExpired:
                _kill_group(signal.SIGKILL)
                proc.wait(timeout=5)
            rc = 124
    finally:
        reader_thread.join(timeout=2.0)
        if log_handle is not None:
            log_handle.close()
    return rc, "".join(chunks)


# ---------------------------------------------------------------------------
# Forge cache indexing & decentralized branch catch-up planner
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    candidate_id: str
    task_id: str
    status: str
    attempt_id: str | None = None
    candidate_revision: str | None = None
    source_digest: str | None = None
    gpu_name: str | None = None
    has_energy: bool = False
    origin: str = ""


def _extract_energy_present(cost: Any) -> tuple[bool, str | None]:
    if not isinstance(cost, dict):
        return False, None
    en = cost.get("energy")
    if not isinstance(en, dict):
        return False, None
    has = (
        isinstance(en.get("joules_total"), (int, float))
        or isinstance(en.get("joules_per_step"), (int, float))
        or isinstance(en.get("avg_power_watts"), (int, float))
    )
    gpu = en.get("gpu_name")
    return has, (str(gpu).strip() if isinstance(gpu, str) and gpu.strip() else None)


def index_forge_cache(root: Path) -> dict[tuple[str, str], list[CacheEntry]]:
    """Index all existing Forge cache entries from technique-receipts, local attempts, and records.

    Read-only: never modifies repo state or launches training.
    """
    root = Path(root).resolve()
    index: dict[tuple[str, str], list[CacheEntry]] = {}

    def _add(entry: CacheEntry) -> None:
        if not entry.candidate_id or not entry.task_id:
            return
        index.setdefault((entry.candidate_id, entry.task_id), []).append(entry)

    # 1. Compact committed technique receipts (primary cross-branch cache!)
    receipts_dir = root / "reports/forge/technique-receipts"
    if receipts_dir.is_dir():
        for path in sorted(receipts_dir.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cid = str(doc.get("candidate_id") or "")
            crev = doc.get("candidate_revision")
            prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
            sdig = prov.get("source_digest")
            aid = str(doc.get("attempt_id") or path.stem)
            for tr in doc.get("task_results") or []:
                if not isinstance(tr, dict):
                    continue
                tid = str(tr.get("task_id") or "")
                st = str(tr.get("gate_status") or tr.get("status") or "")
                has_en, gpu_name = _extract_energy_present(tr.get("cost"))
                _add(
                    CacheEntry(
                        candidate_id=cid,
                        task_id=tid,
                        status=st,
                        attempt_id=aid,
                        candidate_revision=str(crev) if crev else None,
                        source_digest=str(sdig) if sdig else None,
                        gpu_name=gpu_name,
                        has_energy=has_en,
                        origin=f"technique-receipt:{path.name}",
                    )
                )

    # 2. Local durable attempts (reports/forge/attempts/<id>/result.json)
    attempts_dir = root / "reports/forge/attempts"
    if attempts_dir.is_dir():
        for res_path in sorted(attempts_dir.glob("*/result.json")):
            try:
                doc = json.loads(res_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            req = doc.get("request") if isinstance(doc.get("request"), dict) else {}
            cand = req.get("candidate") if isinstance(req.get("candidate"), dict) else {}
            cid = str(cand.get("id") or doc.get("candidate_id") or "")
            crev = req.get("candidate_revision") or doc.get("candidate_revision")
            src = req.get("source") if isinstance(req.get("source"), dict) else {}
            sdig = src.get("digest")
            cprof = (req.get("compute_profiles") or {}).get("cuda") if isinstance(req.get("compute_profiles"), dict) else None
            cprof_gpu = cprof.get("model") if isinstance(cprof, dict) else None
            aid = res_path.parent.name
            for tr in doc.get("task_results") or []:
                if not isinstance(tr, dict):
                    continue
                tid = str(tr.get("task_id") or "")
                st = str(tr.get("gate_status") or tr.get("status") or "")
                has_en, gpu_name = _extract_energy_present(tr.get("cost"))
                _add(
                    CacheEntry(
                        candidate_id=cid,
                        task_id=tid,
                        status=st,
                        attempt_id=aid,
                        candidate_revision=str(crev) if crev else None,
                        source_digest=str(sdig) if sdig else None,
                        gpu_name=gpu_name or cprof_gpu,
                        has_energy=has_en,
                        origin=f"attempt:{aid}",
                    )
                )

    # 3. Committed readout records (reports/forge/records/readout-*.json)
    records_dir = root / "reports/forge/records"
    if records_dir.is_dir():
        for path in sorted(records_dir.glob("readout-*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cid = str(doc.get("candidate_id") or "")
            crev = doc.get("candidate_revision")
            study = doc.get("study") if isinstance(doc.get("study"), dict) else {}
            scope = study.get("scope") if isinstance(study.get("scope"), dict) else {}
            study_gpu = scope.get("cuda_model")
            for tr in doc.get("task_results") or []:
                if not isinstance(tr, dict):
                    continue
                tid = str(tr.get("task_id") or "")
                st = str(tr.get("gate_status") or tr.get("status") or "")
                has_en, gpu_name = _extract_energy_present(tr.get("cost"))
                _add(
                    CacheEntry(
                        candidate_id=cid,
                        task_id=tid,
                        status=st,
                        candidate_revision=str(crev) if crev else None,
                        gpu_name=gpu_name or study_gpu,
                        has_energy=has_en,
                        origin=f"record:{path.name}",
                    )
                )

    return index


def _entry_satisfies_cache(
    entry: CacheEntry,
    *,
    expected_revision: str | None,
    expected_source_digest: str | None,
    strict_source: bool,
    require_energy: bool,
    require_gpu_model: str | None,
    treat_incomplete_as_miss: bool = True,
) -> bool:
    valid_statuses = {"PASS", "FAIL"}
    if not treat_incomplete_as_miss:
        valid_statuses |= {"INCOMPLETE", "TIMEOUT"}
    if entry.status not in valid_statuses:
        return False
    # In strict_source mode, also require matching candidate_revision (which hashes recipe + source).
    if strict_source and expected_revision and entry.candidate_revision and entry.candidate_revision != expected_revision:
        return False
    if strict_source and expected_source_digest and entry.source_digest != expected_source_digest:
        return False
    if require_energy and not entry.has_energy:
        return False
    return not (require_gpu_model and entry.gpu_name != require_gpu_model)


def discover_catchup_candidates(
    root: Path,
    *,
    explicit_candidates: list[str] | None = None,
    include_selected_families: bool = True,
) -> list[str]:
    """Resolve the candidate roster for a branch catch-up."""
    names: list[str] = []
    seen: set[str] = set()

    if include_selected_families:
        from experiments.forge.trainer_families import (
            CURRENT_SELECTION,
            current_family_candidates,
        )

        if (root / CURRENT_SELECTION).is_file():
            for choice in current_family_candidates(root):
                cid = str(choice["candidate_id"])
                if cid not in seen:
                    seen.add(cid)
                    names.append(cid)

    for cid in explicit_candidates or []:
        cid = cid.strip()
        if cid and cid not in seen:
            seen.add(cid)
            names.append(cid)
    return names


@dataclass
class CatchupUnit:
    unit_id: str
    candidate_id: str
    view_id: str
    through_tier: int
    gpu: str
    tasks: list[str]
    is_full_view_slice: bool
    estimated_seconds: float
    study_id: str | None = None
    study_path: str | None = None


def detect_branch_changes(root: Path, base_ref: str | None) -> tuple[set[str], set[str]]:
    """Detect modified candidate/task declarations relative to `base_ref` and dirty worktree.

    Returns `(changed_candidates, changed_tasks)` so branch catch-up automatically
    invalidates edited recipes or tasks without rerunning unchanged cache cells.
    """
    changed_candidates: set[str] = set()
    changed_tasks: set[str] = set()
    if not base_ref:
        return changed_candidates, changed_tasks

    paths: set[str] = set()
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if diff.returncode == 0:
        paths.update(line.strip() for line in diff.stdout.splitlines() if line.strip())

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode == 0:
        for line in status.stdout.splitlines():
            if len(line) >= 4:
                rel = line[3:].strip()
                if " -> " in rel:
                    rel = rel.split(" -> ", 1)[1]
                paths.add(rel)

    for rel in paths:
        if rel.startswith("configs/forge/ideas/") and rel.endswith(".json"):
            changed_candidates.add(Path(rel).stem)
        elif rel.startswith("configs/forge/tasks/") and rel.endswith(".json"):
            changed_tasks.add(Path(rel).stem)
    return changed_candidates, changed_tasks


def _static_task_blockers(
    root: Path,
    idea: dict[str, Any],
    view: dict[str, Any],
    all_tasks: dict[str, dict[str, Any]],
    *,
    scope_task_ids: set[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Run Forge's static candidate & task preflight locally without requiring a target GPU."""
    from dataclasses import asdict

    from experiments.forge.api import CapabilityError, FormulationContext
    from experiments.forge.contracts import read_json
    from experiments.forge.preflight import task_preflight
    from experiments.forge.sampling import candidate_blockers

    defaults_path = root / "configs/forge/defaults.json"
    if not defaults_path.is_file():
        return [], {}
    defaults = read_json(defaults_path)
    protocol_path = root / "configs/forge/protocols" / f"{defaults['protocol']}.json"
    if not protocol_path.is_file():
        return [], {}
    protocol = read_json(protocol_path)
    prior = {**defaults.get("prior", {}), **idea.get("prior", {})}
    if "prior_update" in idea.get("recipe_overrides", {}):
        prior["learnable"] = idea["recipe_overrides"]["prior_update"] == "learned"

    cand_blockers: list[str] = []
    try:
        context = FormulationContext(
            recipe_preset=idea.get("recipe_preset"),
            recipe_overrides=idea.get("recipe_overrides", {}),
            prior=prior,
            prior_mechanism=idea.get("prior_mechanism"),
            seed=protocol["seed"],
            requires_capabilities=idea.get("requires_capabilities", []),
            extensions=idea.get("extensions", {}),
            initializer=idea.get("initializer", "deterministic_orthogonal"),
            execution_path=idea.get("execution_path", "public_trainer"),
        )
        recipe = asdict(context.recipe)
        capabilities = [name for name, enabled in context.capabilities().items() if enabled]
    except CapabilityError as exc:
        recipe, capabilities = idea.get("recipe_overrides", {}), []
        cand_blockers.extend(exc.blockers)

    candidate = {
        **idea,
        "resolved_recipe": recipe,
        "prior": prior,
        "capabilities": capabilities,
        "claim_contract": idea.get("claim_contract", {"schedule": "scheduled", "scoring_weights": "live"}),
    }
    cand_blockers.extend(candidate_blockers(candidate))
    tasks_in_view = {a["task"]: all_tasks[a["task"]] for a in view.get("assignments", []) if a["task"] in all_tasks}
    target_tids = scope_task_ids if scope_task_ids is not None else set(tasks_in_view)
    per_task: dict[str, list[str]] = {}
    for tid in target_tids:
        tcard = tasks_in_view.get(tid) or all_tasks.get(tid)
        if not tcard:
            continue
        if cand_blockers:
            per_task[tid] = list(cand_blockers)
        else:
            per_task[tid] = list(task_preflight(dict(tcard), candidate, protocol, root=root, tasks=tasks_in_view))
    return cand_blockers, per_task


def compute_catchup_plan(
    root: Path,
    *,
    view_id: str = "discriminator_stability",
    through_tier: int = 1,
    candidates: list[str] | None = None,
    include_selected_families: bool = True,
    only_tasks: list[str] | None = None,
    gpu_policy: str = "auto",
    prefer_t4: bool = False,
    strict_source: bool = False,
    require_energy: bool = False,
    require_media: bool = False,
    require_gpu_model: str | None = None,
    changed_since: str | None = "origin/develop",
    skip_preflight_blocked: bool = True,
    max_tasks_per_unit: int = 20,
) -> dict[str, Any]:
    """Compute the exact decentralized cache diff for the current branch checkout.

    Returns a structured plan listing CACHED, BLOCKED, and MISSING cells grouped into
    right-sized Colab GPU work units. Launches zero training and writes nothing.
    """
    from experiments.forge.leaderboard_media import load_leaderboard_media_index
    from experiments.forge.planning import load_idea
    from experiments.forge.sources import inspect_source
    from experiments.forge.views import load_tasks, load_view

    root = Path(root).resolve()
    gpu_policy = normalize_gpu(gpu_policy)
    roster = discover_catchup_candidates(
        root,
        explicit_candidates=candidates,
        include_selected_families=include_selected_families,
    )
    if not roster:
        raise ValueError("No candidates selected for catchup; pass --candidate or verify family-current-v1.json")

    view = load_view(root, view_id)
    all_tasks = load_tasks(root)
    only_set = set(only_tasks) if only_tasks else None
    assignments = [
        a
        for a in view.get("assignments", [])
        if int(a.get("qualification_tier", 99)) <= through_tier
        and (only_set is None or a["task"] in only_set)
    ]
    if not assignments:
        raise ValueError(f"No tasks in view {view_id!r} up to tier {through_tier}")

    cache = index_forge_cache(root)
    media_keys: set[tuple[str, str]] = set()
    if require_media:
        media_doc = load_leaderboard_media_index(root)
        for item in media_doc.get("items", []):
            if isinstance(item, dict) and item.get("candidate_id") and item.get("task_id"):
                media_keys.add((str(item["candidate_id"]), str(item["task_id"])))
    source_manifest = inspect_source(root)
    source_digest = source_manifest["digest"]
    changed_candidates, changed_tasks = detect_branch_changes(root, changed_since)

    # When strict GPU model isn't explicitly passed, keep require_gpu_model as None (any CUDA cohort
    # satisfies branch-merge cache unless --require-gpu-model or --require-energy is requested).
    effective_gpu_filter = require_gpu_model
    if effective_gpu_filter is None and gpu_policy not in ("auto", "cpu") and require_energy:
        effective_gpu_filter = COLAB_GPU_SPECS[gpu_policy]["cuda_model"]

    cached_cells: list[dict[str, Any]] = []
    blocked_cells: list[dict[str, Any]] = []
    missing_cells: list[dict[str, Any]] = []
    units: list[CatchupUnit] = []
    all_view_task_ids = [a["task"] for a in assignments]

    for cid in roster:
        idea = load_idea(root, cid)
        if not idea:
            continue
        per_task_blockers: dict[str, list[str]] = {}
        if skip_preflight_blocked:
            _, per_task_blockers = _static_task_blockers(
                root, idea, view, all_tasks, scope_task_ids=set(all_view_task_ids)
            )

        missing_by_gpu: dict[str, list[tuple[str, float]]] = {}
        for assignment in assignments:
            tid = assignment["task"]
            tier = int(assignment.get("qualification_tier", 1))
            tcard = all_tasks[tid]
            blockers = per_task_blockers.get(tid, [])
            if blockers:
                blocked_cells.append(
                    {
                        "candidate_id": cid,
                        "task_id": tid,
                        "tier": tier,
                        "blockers": blockers,
                    }
                )
                continue

            # If this candidate or task declaration changed on the branch, require current source_digest!
            cell_strict_source = strict_source or (cid in changed_candidates) or (tid in changed_tasks)
            entries = cache.get((cid, tid), [])
            has_media_hit = (not require_media) or ((cid, tid) in media_keys)
            hit = (
                next(
                    (
                        e
                        for e in entries
                        if _entry_satisfies_cache(
                            e,
                            expected_revision=None,
                            expected_source_digest=source_digest,
                            strict_source=cell_strict_source,
                            require_energy=require_energy,
                            require_gpu_model=effective_gpu_filter,
                        )
                    ),
                    None,
                )
                if has_media_hit
                else None
            )
            if hit is not None:
                cached_cells.append(
                    {
                        "candidate_id": cid,
                        "task_id": tid,
                        "tier": tier,
                        "status": hit.status,
                        "gpu_name": hit.gpu_name,
                        "has_energy": hit.has_energy,
                        "origin": hit.origin,
                    }
                )
            else:
                target_gpu = choose_gpu_for_task(
                    tid, tcard, tier=tier, gpu_policy=gpu_policy, prefer_t4=prefer_t4
                )
                timeout_s = float((tcard.get("resources") or {}).get("timeout_seconds") or 300.0)
                missing_cells.append(
                    {
                        "candidate_id": cid,
                        "task_id": tid,
                        "tier": tier,
                        "target_gpu": target_gpu,
                        "timeout_seconds": timeout_s,
                    }
                )
                missing_by_gpu.setdefault(target_gpu, []).append((tid, timeout_s))

        for target_gpu, task_pairs in sorted(missing_by_gpu.items()):
            # Separate dependency-chained tasks from independent tasks so a prerequisite chain
            # never delays or shares a slice with unrelated independent tasks.
            indep_pairs = [
                (t, s)
                for t, s in task_pairs
                if not (all_tasks[t].get("dependencies") or (all_tasks[t].get("execution") or {}).get("continuation_of"))
            ]
            dep_pairs = [
                (t, s)
                for t, s in task_pairs
                if (all_tasks[t].get("dependencies") or (all_tasks[t].get("execution") or {}).get("continuation_of"))
            ]
            buckets = [
                indep_pairs[start : start + max(1, max_tasks_per_unit)]
                for start in range(0, len(indep_pairs), max(1, max_tasks_per_unit))
            ] + [[pair] for pair in dep_pairs]
            for chunk in buckets:
                if not chunk:
                    continue
                chunk_tids = [t for t, _ in chunk]
                est_s = sum(s for _, s in chunk)
                is_full = (
                    len(missing_by_gpu) == 1
                    and len(buckets) == 1
                    and len(chunk_tids) == len(all_view_task_ids)
                    and set(chunk_tids) == set(all_view_task_ids)
                    and only_set is None
                )
                uid_seed = f"{cid}:{view_id}:t{through_tier}:{target_gpu}:{','.join(sorted(chunk_tids))}"
                uid_hash = hashlib.sha256(uid_seed.encode()).hexdigest()[:8]
                units.append(
                    CatchupUnit(
                        unit_id=f"{cid}-{target_gpu.lower()}-{uid_hash}",
                        candidate_id=cid,
                        view_id=view_id,
                        through_tier=through_tier,
                        gpu=target_gpu,
                        tasks=chunk_tids,
                        is_full_view_slice=is_full,
                        estimated_seconds=est_s,
                    )
                )

    total_cells = len(cached_cells) + len(blocked_cells) + len(missing_cells)
    return {
        "schema_version": 1,
        "generated_at": utc_stamp(),
        "git_head": _git_head(root),
        "source_digest": source_digest,
        "view_id": view_id,
        "through_tier": through_tier,
        "gpu_policy": gpu_policy,
        "strict_source": strict_source,
        "require_energy": require_energy,
        "require_gpu_model": effective_gpu_filter,
        "changed_candidates": sorted(changed_candidates),
        "changed_tasks": sorted(changed_tasks),
        "candidate_count": len(roster),
        "task_count": len(assignments),
        "total_cells": total_cells,
        "cached_count": len(cached_cells),
        "blocked_count": len(blocked_cells),
        "missing_count": len(missing_cells),
        "work_unit_count": len(units),
        "work_units": [
            {
                "unit_id": u.unit_id,
                "candidate_id": u.candidate_id,
                "view_id": u.view_id,
                "through_tier": u.through_tier,
                "gpu": u.gpu,
                "tasks": u.tasks,
                "is_full_view_slice": u.is_full_view_slice,
                "estimated_worst_case_seconds": u.estimated_seconds,
            }
            for u in units
        ],
        "cached_sample": cached_cells[:10],
        "blocked_sample": blocked_cells[:10],
        "missing_cells": missing_cells,
        "_units": units,
    }


# ---------------------------------------------------------------------------
# Git branch sync (handles pushed commits AND unpushed local agent branches!)
# ---------------------------------------------------------------------------


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _git_is_pushed(root: Path, sha: str) -> bool:
    rc = subprocess.run(
        ["git", "branch", "-r", "--contains", sha],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return rc.returncode == 0 and bool(rc.stdout.strip())


def prepare_branch_bundle(root: Path, run_dir: Path) -> dict[str, Any]:
    """Prepare minimal branch sync payload so Colab matches local branch HEAD + edits.

    - If HEAD is already pushed to `origin` and worktree has no source/config edits,
      returns `{"mode": "origin", "sha": HEAD}` (Colab clones directly from GitHub).
    - If HEAD is an unpushed local branch commit, creates a compact `git bundle` of
      `merge-base(origin/develop, HEAD)..HEAD` (typically <100 KB!).
    - If there are uncommitted tracked/untracked edits under `particlegan/`, `experiments/`,
      `benchmarks/`, `lib/`, or `configs/forge/`, packages them into a small overlay tarball.
    """
    head = _git_head(root)
    bundle_path: Path | None = None
    overlay_path: Path | None = None

    pushed = _git_is_pushed(root, head)
    if not pushed:
        base_proc = subprocess.run(
            ["git", "merge-base", "origin/develop", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        base_ref = base_proc.stdout.strip() if base_proc.returncode == 0 and base_proc.stdout.strip() else None
        bundle_path = run_dir / "branch.bundle"
        rev_range = f"{base_ref}..HEAD" if base_ref else "HEAD"
        subprocess.run(
            ["git", "bundle", "create", str(bundle_path), rev_range, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        )

    # Check for uncommitted modified or untracked files in code/config dirs
    status_out = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    )
    allowed_prefixes = ("particlegan/", "experiments/", "benchmarks/", "lib/", "configs/forge/", "pyproject.toml")
    dirty_files: list[str] = []
    for line in status_out.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip()
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        if any(rel == p or rel.startswith(p) for p in allowed_prefixes):
            full = root / rel
            if full.is_file():
                dirty_files.append(rel)

    if dirty_files:
        overlay_path = run_dir / "worktree-overlay.tar.gz"
        with tarfile.open(overlay_path, "w:gz") as tf:
            for rel in sorted(set(dirty_files)):
                tf.add(root / rel, arcname=rel)

    return {
        "sha": head,
        "pushed": pushed,
        "bundle_path": str(bundle_path) if bundle_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "dirty_file_count": len(dirty_files),
    }


# ---------------------------------------------------------------------------
# Ephemeral catch-up view/study synthesis (never touches local Git tree!)
# ---------------------------------------------------------------------------


def build_ephemeral_catchup_bundle(
    root: Path,
    unit: CatchupUnit,
    out_dir: Path,
) -> tuple[Path | None, Path, str]:
    """Write an ephemeral study (and optional task-subset view) into `out_dir` for Colab upload.

    Never writes into the local repository's `configs/forge/` directory.
    """
    from experiments.forge.views import load_tasks, load_view

    view = load_view(root, unit.view_id)
    all_tasks = load_tasks(root)
    cuda_model = COLAB_GPU_SPECS[unit.gpu]["cuda_model"]
    exec_backend = "cpu" if unit.gpu == "cpu" else "cuda"

    view_file: Path | None = None
    effective_view_id = unit.view_id
    if not unit.is_full_view_slice:
        effective_view_id = f"colab-catchup-{unit.unit_id}"
        # Transitively include prerequisite tasks so Forge's view validator accepts
        # dependency-chained tasks (e.g. five_word_joint_hold -> five_word_joint_smoke).
        wanted = set(unit.tasks)
        frontier = list(wanted)
        has_deps = False
        while frontier:
            tid = frontier.pop()
            tcard = all_tasks.get(tid) or {}
            deps = list(tcard.get("dependencies") or [])
            cont = (tcard.get("execution") or {}).get("continuation_of")
            if cont:
                deps.append(cont)
            for dep in deps:
                dep_id = dep.get("task") if isinstance(dep, dict) else dep
                if isinstance(dep_id, str) and dep_id:
                    has_deps = True
                    if dep_id not in wanted:
                        wanted.add(dep_id)
                        frontier.append(dep_id)
        subset_assignments = [a for a in view.get("assignments", []) if a["task"] in wanted]
        is_diagnostic_view = view.get("evidence_scope") in {"research_diagnostic", "calibration_diagnostic"}
        if is_diagnostic_view:
            # Diagnostic views require every assignment to be importance="diagnostic" in Tier 1
            # (Forge validate_view rejects non-diagnostic importance or tier > 1).
            effective_through_tier = 1
            remapped = [
                {
                    **{k: v for k, v in a.items() if k not in ("hold", "removal", "stretch")},
                    "qualification_tier": 1,
                    "importance": "diagnostic",
                }
                for a in subset_assignments
            ]
        elif has_deps:
            # Compress distinct original tiers onto contiguous 1..K so Forge's
            # "empty required tier: qualification tiers must be contiguous from Tier 1"
            # check always passes while strictly preserving parent-before-child gates.
            orig_tiers = sorted({int(a.get("qualification_tier", 1)) for a in subset_assignments})
            tier_map = {orig: idx + 1 for idx, orig in enumerate(orig_tiers)}
            effective_through_tier = len(orig_tiers)
            remapped = [
                {
                    **{k: v for k, v in a.items() if k not in ("hold", "removal", "stretch")},
                    "qualification_tier": tier_map[int(a.get("qualification_tier", 1))],
                    # Ephemeral catch-up views are non-qualifying; marking slice tasks
                    # required guarantees every compressed tier is non-empty.
                    "importance": "required",
                }
                for a in subset_assignments
            ]
        else:
            effective_through_tier = 1
            remapped = [
                {
                    **{k: v for k, v in a.items() if k not in ("hold", "removal", "stretch")},
                    "qualification_tier": 1,
                    "importance": "required",
                }
                for a in subset_assignments
            ]
        ephemeral_view = {
            **{
                k: v
                for k, v in view.items()
                if k not in ("id", "assignments", "revision", "previous_revision", "revision_reason", "law_groups")
            },
            "id": effective_view_id,
            "revision": 1,
            "assignments": remapped,
        }
        view_file = out_dir / f"{effective_view_id}.view.json"
        view_file.write_text(json.dumps(ephemeral_view, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        effective_through_tier = unit.through_tier

    study_id = f"colab-catchup-{unit.unit_id}"
    # Pick first task for a valid study contract metric anchor
    anchor_task = unit.tasks[0]
    budget_s = max(3600, int(unit.estimated_seconds * 2.5))
    control_cid = "rhbi-v12" if unit.candidate_id == "unified-v11" else "unified-v11"
    study_doc = {
        "schema_version": 1,
        "id": study_id,
        "status": "ready",
        "candidate": unit.candidate_id,
        "control": {"candidate_id": control_cid, "task_map": {}},
        "campaign": {
            "id": study_id,
            "budget_seconds": budget_s,
            "candidate_budget_seconds": budget_s,
        },
        "hypothesis": (
            f"Decentralized Colab {unit.gpu} cache catch-up for {unit.candidate_id} on "
            f"{len(unit.tasks)} missing task(s) of {unit.view_id} (through tier {unit.through_tier}) "
            "under protocol seed 0 with hardware NVML energy telemetry."
        ),
        "competing_explanation": (
            "If the recipe regresses or fails numerical bounds on any missing task under the "
            f"matched {cuda_model or 'CPU'} cohort, the task will record a faithful numerical FAIL."
        ),
        "prediction": {
            "task_id": anchor_task,
            "phase": "final",
            "metric": "step",
            "op": ">=",
            "threshold": 0.0,
        },
        "falsifier": {
            "task_id": anchor_task,
            "phase": "final",
            "metric": "step",
            "op": "<",
            "threshold": 0.0,
        },
        "prior_evidence": [
            {
                "path": "configs/forge/selections/family-current-v1.json",
                "selector": [],
                "identity": {
                    "scope": "whole_candidate_family_current",
                    "default_adoption": False,
                },
                "use": "motivation_only",
            }
        ],
        "max_rounds": 1,
        "scope": {
            "view": effective_view_id,
            "through_tier": effective_through_tier,
            "execution_backend": exec_backend,
            **({"cuda_model": cuda_model} if cuda_model else {}),
        },
        "terminal_rules": {
            "falsified": "stop_revision",
            "incomplete": "request_missing_evidence",
            "inconclusive": "stop_and_readout",
            "prediction_observed": "review_saved_diagnostics",
        },
    }
    study_file = out_dir / f"{study_id}.study.json"
    study_file.write_text(json.dumps(study_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return view_file, study_file, study_id


# ---------------------------------------------------------------------------
# VM-side crash-proof watchdog & Colab warm worker
# ---------------------------------------------------------------------------


WATCHDOG_ARM_PY = textwrap.dedent(
    """
    import os, subprocess, sys, textwrap
    addr = os.environ.get("TBE_RUNTIME_ADDR")
    if not addr:
        raise RuntimeError("TBE_RUNTIME_ADDR missing in Colab kernel environment")
    max_s = int(os.environ.get("CF_MAX_SECONDS", "7200"))
    grace_s = int(os.environ.get("CF_IDLE_GRACE_SECONDS", "600"))
    stall_s = int(os.environ.get("CF_STALL_SECONDS", "1500"))
    watchdog_code = textwrap.dedent('''
        import os, sys, time, urllib.request
        addr, max_s, grace_s, stall_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        t0 = time.monotonic()
        logf = "/content/watchdog.log"
        hb_file = "/content/forge_heartbeat"
        done_file = "/content/forge_job.done"
        job_log = "/content/forge_job.log"
        def log(msg):
            with open(logf, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}\\n")
        def unassign(reason):
            log(f"TRIGGERING UNASSIGN: {reason}")
            try:
                req = urllib.request.Request(f"http://{addr}/unassign", method="POST")
                with urllib.request.urlopen(req, timeout=15) as r:
                    log(f"unassign HTTP {r.status}")
            except Exception as e:
                log(f"unassign error: {e!r}")
            sys.exit(0)
        with open(hb_file, "w") as f:
            f.write(str(time.time()))
        log(f"Watchdog armed: addr={addr} max={max_s}s grace={grace_s}s stall={stall_s}s")
        while True:
            time.sleep(15)
            now_mono = time.monotonic()
            now_wall = time.time()
            if now_mono - t0 > max_s:
                unassign(f"max wall-clock {max_s}s exceeded")
            if os.path.exists(done_file):
                age = now_wall - os.path.getmtime(done_file)
                hb_age = now_wall - os.path.getmtime(hb_file) if os.path.exists(hb_file) else age
                if age > grace_s and hb_age > grace_s:
                    unassign(f"post-job idle grace {grace_s}s exceeded (local controller disconnected)")
            if os.path.exists(job_log):
                idle = now_wall - os.path.getmtime(job_log)
                hb_age = now_wall - os.path.getmtime(hb_file) if os.path.exists(hb_file) else idle
                if idle > stall_s and hb_age > stall_s:
                    unassign(f"log stall timeout {stall_s}s exceeded")
    ''').strip()
    with open("/content/watchdog.py", "w") as f:
        f.write(watchdog_code + "\\n")
    subprocess.Popen(
        [sys.executable, "/content/watchdog.py", addr, str(max_s), str(grace_s), str(stall_s)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"WATCHDOG_OK addr={addr} max={max_s}s grace={grace_s}s stall={stall_s}s")
    """
).strip()


@dataclass
class ActiveSessionRegistry:
    """Thread-safe registry ensuring every provisioned Colab session is stopped on exit/signal."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    sessions: dict[str, Path] = field(default_factory=dict)  # session_name -> config_path

    def register(self, session: str, cfg: Path) -> None:
        with self.lock:
            self.sessions[session] = cfg

    def unregister(self, session: str) -> None:
        with self.lock:
            self.sessions.pop(session, None)

    def stop_all(self) -> None:
        with self.lock:
            items = list(self.sessions.items())
            self.sessions.clear()
        for session, cfg in items:
            try:
                run_with_hard_timeout(
                    ["colab", "--config", str(cfg), "stop", "-s", session],
                    timeout_seconds=30,
                )
            except OSError as err:
                print(f"[colab_forge] stop_all failed for {session}: {err}", file=sys.stderr)


_GLOBAL_REGISTRY = ActiveSessionRegistry()
atexit.register(_GLOBAL_REGISTRY.stop_all)


class ColabWorker:
    """One warm Colab VM worker with isolated --config, VM watchdog, and guaranteed teardown."""

    def __init__(
        self,
        *,
        worker_name: str,
        gpu: str,
        run_dir: Path,
        repo_root: Path,
        repo_url: str,
        branch_sync: dict[str, Any],
        boot_timeout: int = 150,
        max_minutes: int = 120,
        idle_grace_minutes: int = 10,
        stall_minutes: int = 25,
        transport_retries: int = DEFAULT_TRANSPORT_RETRIES,
        transport_retry_backoff: float = DEFAULT_TRANSPORT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.worker_name = worker_name
        self.gpu = normalize_gpu(gpu)
        self.run_dir = run_dir
        self.repo_root = repo_root
        self.repo_url = repo_url
        self.branch_sync = branch_sync
        self.boot_timeout = boot_timeout
        self.max_seconds = max_minutes * 60
        self.idle_grace_seconds = idle_grace_minutes * 60
        self.stall_seconds = stall_minutes * 60
        self.transport_retries = max(0, int(transport_retries))
        self.transport_retry_backoff = max(0.0, float(transport_retry_backoff))
        self.cfg = run_dir / f"{worker_name}.colab.json"
        self.log = run_dir / f"{worker_name}.log"
        self.alive = False

    def _log(self, msg: str) -> None:
        line = f"[{utc_stamp()}] [{self.worker_name}] {msg}\n"
        with self.log.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="", flush=True)

    def _colab(self, args: list[str], *, timeout: float, stdin_text: str | None = None, capture: bool = False) -> tuple[int, str]:
        return run_with_hard_timeout(
            ["colab", "--config", str(self.cfg), *args],
            timeout_seconds=timeout,
            stdin_text=stdin_text,
            log_file=self.log,
            capture=capture,
        )

    def _colab_with_transport_retry(
        self,
        args: list[str],
        *,
        timeout: float,
        label: str,
        retries: int | None = None,
        backoff_seconds: float | None = None,
        stdin_text: str | None = None,
        capture: bool = False,
    ) -> tuple[int, str]:
        """Run an idempotent Colab transport command with bounded retry ONLY on rc=124 timeouts."""
        max_retries = self.transport_retries if retries is None else max(0, int(retries))
        backoff = self.transport_retry_backoff if backoff_seconds is None else max(0.0, float(backoff_seconds))
        attempts = 1 + max_retries
        rc, out = 0, ""
        for attempt in range(1, attempts + 1):
            rc, out = self._colab(args, timeout=timeout, stdin_text=stdin_text, capture=capture)
            if rc == 0 or rc not in TRANSIENT_TRANSPORT_RETRY_RCS or attempt >= attempts:
                return rc, out
            self._log(
                f"{label} hit transient transport timeout (rc={rc}) on attempt {attempt}/{attempts}; "
                f"retrying in {backoff:.1f}s..."
            )
            if backoff > 0:
                time.sleep(backoff)
        return rc, out

    def start(self) -> None:
        spec = COLAB_GPU_SPECS[self.gpu]
        cmd = ["new", "-s", self.worker_name]
        if spec["colab_flag"]:
            cmd += ["--gpu", spec["colab_flag"]]
        self._log(f"Provisioning Colab {self.gpu} VM (boot timeout {self.boot_timeout}s)...")
        rc, _ = self._colab(cmd, timeout=self.boot_timeout)
        if rc != 0:
            raise RuntimeError(f"colab new failed for {self.worker_name} (gpu={self.gpu}, rc={rc})")
        self.alive = True
        _GLOBAL_REGISTRY.register(self.worker_name, self.cfg)
        time.sleep(5)  # Websocket warmup guard

        # Arm VM-side crash/idle/max-wall-clock watchdog with 3 retries
        armed = False
        for attempt in range(1, 4):
            rc, _ = self._colab(
                [
                    "exec",
                    "-s",
                    self.worker_name,
                    "--timeout",
                    "30",
                    "--env",
                    f"CF_MAX_SECONDS={self.max_seconds}",
                    "--env",
                    f"CF_IDLE_GRACE_SECONDS={self.idle_grace_seconds}",
                    "--env",
                    f"CF_STALL_SECONDS={self.stall_seconds}",
                ],
                timeout=45,
                stdin_text=WATCHDOG_ARM_PY,
            )
            if rc == 0:
                armed = True
                break
            self._log(f"Watchdog arm attempt {attempt}/3 failed (rc={rc}); retrying in 5s...")
            time.sleep(5)
        if not armed:
            self.stop()
            raise RuntimeError(f"Failed to arm VM watchdog on {self.worker_name}; VM terminated for safety.")

        # Upload optional unpushed branch bundle / dirty worktree overlay
        if self.branch_sync.get("bundle_path"):
            self._log("Uploading unpushed branch git bundle...")
            rc, _ = self._colab_with_transport_retry(
                ["upload", "-s", self.worker_name, self.branch_sync["bundle_path"], "/content/branch.bundle"],
                timeout=120,
                label="Branch bundle upload",
            )
            if rc != 0:
                raise RuntimeError(f"Failed to upload branch.bundle (rc={rc})")
        if self.branch_sync.get("overlay_path"):
            self._log(f"Uploading worktree overlay ({self.branch_sync['dirty_file_count']} file(s))...")
            rc, _ = self._colab_with_transport_retry(
                ["upload", "-s", self.worker_name, self.branch_sync["overlay_path"], "/content/worktree-overlay.tar.gz"],
                timeout=120,
                label="Worktree overlay upload",
            )
            if rc != 0:
                raise RuntimeError(f"Failed to upload worktree-overlay.tar.gz (rc={rc})")

        # Bootstrap repo once on the warm VM
        sha = self.branch_sync["sha"]
        self._log(f"Bootstrapping ParticleGAN at {sha[:12]} via uv...")
        bootstrap_sh = textwrap.dedent(
            f"""
            set -euo pipefail
            exec > >(tee -a /content/forge_job.log) 2>&1
            if command -v nvidia-smi >/dev/null 2>&1; then
              nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
            fi
            git clone --depth 1 --branch develop {self.repo_url} /content/ParticleGAN
            cd /content/ParticleGAN
            ACTUAL=$(git rev-parse HEAD)
            if [ "$ACTUAL" != "{sha}" ]; then
              if [ -f /content/branch.bundle ]; then
                git fetch /content/branch.bundle HEAD:refs/heads/colab-branch || (git fetch --unshallow origin develop && git fetch /content/branch.bundle HEAD:refs/heads/colab-branch)
                git checkout {sha}
              else
                git fetch --depth 1 origin {sha}
                git checkout {sha}
              fi
            fi
            if [ -f /content/worktree-overlay.tar.gz ]; then
              tar -xzf /content/worktree-overlay.tar.gz -C /content/ParticleGAN
            fi
            uv pip install --system --no-build-isolation --no-deps -e .
            uv pip install --system 'POT>=0.9.5,<0.10' 'Pillow>=13.0.0,<14' 'matplotlib>=3.8'
            touch /content/forge_heartbeat
            echo "[vm] Bootstrap complete at $(git rev-parse --short HEAD)"
            """
        ).strip()
        rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", bootstrap_sh], timeout=300)
        if rc != 0:
            raise RuntimeError(f"VM bootstrap failed on {self.worker_name} (rc={rc})")
        self._log("Warm VM ready.")

    def run_unit(
        self,
        unit: CatchupUnit,
        *,
        study_file: Path,
        study_id: str,
        ephemeral_view_file: Path | None = None,
        render_media: bool = True,
    ) -> dict[str, Any]:
        """Run one Forge study/catch-up unit on this warm VM and download its output tarball."""
        self._log(f"Starting unit {unit.unit_id} (cand={unit.candidate_id}, tasks={len(unit.tasks)}, study={study_id}, render_media={render_media})")
        # Upload study (and ephemeral view if partial slice)
        rc, _ = self._colab_with_transport_retry(
            ["upload", "-s", self.worker_name, str(study_file), f"/content/ParticleGAN/configs/forge/studies/{study_id}.json"],
            timeout=60,
            label=f"Study upload ({unit.unit_id})",
        )
        if rc != 0:
            raise RuntimeError(f"Study upload failed for {unit.unit_id} (rc={rc})")
        if ephemeral_view_file is not None:
            view_name = ephemeral_view_file.name.replace(".view.json", "")
            rc, _ = self._colab_with_transport_retry(
                ["upload", "-s", self.worker_name, str(ephemeral_view_file), f"/content/ParticleGAN/configs/forge/views/{view_name}.json"],
                timeout=60,
                label=f"Ephemeral view upload ({unit.unit_id})",
            )
            if rc != 0:
                raise RuntimeError(f"Ephemeral view upload failed for {unit.unit_id} (rc={rc})")

        gpus_arg = "cpu" if self.gpu == "cpu" else "0"
        device_arg = "cpu" if self.gpu == "cpu" else "cuda"
        tar_remote = f"/content/out-{unit.unit_id}.tar.gz"
        tar_local = self.run_dir / f"{unit.unit_id}-out.tar.gz"
        render_media_sh = (
            textwrap.dedent(
                """
                rm -rf runs/forge/rendered_media
                mkdir -p runs/forge/rendered_media
                while IFS= read -r d; do
                  [ -n "$d" ] || continue
                  python3 -m experiments.forge.leaderboard_media export-attempt "reports/forge/attempts/$d" --output "runs/forge/rendered_media/$d" || echo "[vm] media export warning for $d"
                  touch /content/forge_heartbeat
                done < /content/new_attempts.txt
                [ -d runs/forge/rendered_media ] && TAR_ITEMS+=("runs/forge/rendered_media")
                """
            ).strip()
            if render_media
            else ""
        )

        run_sh = textwrap.dedent(
            f"""
            set -euo pipefail
            rm -f /content/forge_job.done
            touch /content/forge_heartbeat
            exec > >(tee -a /content/forge_job.log) 2>&1
            cd /content/ParticleGAN
            ls -1 reports/forge/attempts/ 2>/dev/null | sort > /content/before_attempts.txt || true
            mkdir -p runs/forge
            touch runs/forge/events.jsonl
            tail -n 0 -F runs/forge/events.jsonl 2>/dev/null &
            TAIL_PID=$!
            set +e
            python3 -m experiments.forge plan '{unit.candidate_id}' --study '{study_id}' --device '{device_arg}'
            python3 -m experiments.forge run '{unit.candidate_id}' --study '{study_id}' --gpus '{gpus_arg}'
            RUN_RC=$?
            set -e
            kill $TAIL_PID 2>/dev/null || true
            wait $TAIL_PID 2>/dev/null || true
            ls -1 reports/forge/attempts/ 2>/dev/null | sort > /content/after_attempts.txt || true
            comm -13 /content/before_attempts.txt /content/after_attempts.txt > /content/new_attempts.txt
            TAR_ITEMS=()
            while IFS= read -r d; do
              [ -n "$d" ] && TAR_ITEMS+=("reports/forge/attempts/$d")
            done < /content/new_attempts.txt
            {render_media_sh}
            for extra in runs/forge/events.jsonl runs/forge/queue/state.json; do
              [ -f "$extra" ] && TAR_ITEMS+=("$extra")
            done
            [ -f /content/watchdog.log ] && TAR_ITEMS+=("/content/watchdog.log")
            if [ "${{#TAR_ITEMS[@]}}" -gt 0 ]; then
              tar -czf '{tar_remote}' "${{TAR_ITEMS[@]}}" 2>&1 || true
            fi
            touch /content/forge_heartbeat /content/forge_job.done
            echo "[vm] unit {unit.unit_id} finished rc=$RUN_RC new_attempts=$(wc -l < /content/new_attempts.txt)"
            exit $RUN_RC
            """
        ).strip()

        t0 = time.monotonic()
        ssh_timeout = max(600, min(self.max_seconds, int(unit.estimated_seconds * 2.0) + 600))
        rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", run_sh], timeout=ssh_timeout)
        elapsed = time.monotonic() - t0

        dl_rc, _ = self._colab_with_transport_retry(
            ["download", "-s", self.worker_name, tar_remote, str(tar_local)],
            timeout=240,
            label=f"Result download ({unit.unit_id})",
        )
        self._log(f"Unit {unit.unit_id} done in {elapsed:.1f}s (run_rc={rc}, dl_rc={dl_rc}, tar_exists={tar_local.exists()})")
        return {
            "unit_id": unit.unit_id,
            "candidate_id": unit.candidate_id,
            "study_id": study_id,
            "gpu": self.gpu,
            "tasks": unit.tasks,
            "run_rc": rc,
            "download_rc": dl_rc,
            "tarball": str(tar_local) if tar_local.exists() else None,
            "elapsed_seconds": round(elapsed, 2),
        }

    def stop(self) -> None:
        if not self.alive:
            return
        self.alive = False
        _GLOBAL_REGISTRY.unregister(self.worker_name)
        self._log(f"Stopping Colab VM {self.worker_name}...")
        self._colab_with_transport_retry(
            ["stop", "-s", self.worker_name],
            timeout=30,
            label=f"Session stop ({self.worker_name})",
        )
        self._log("Session released.")


# ---------------------------------------------------------------------------
# Local ingestion: collision-safe attempt unpack + compact receipt projection
# ---------------------------------------------------------------------------


def ingest_tarball(
    root: Path,
    tar_path: Path,
    *,
    project_receipts: bool = True,
    register_submissions: bool = True,
    rendered_media_dir: Path | None = None,
) -> dict[str, Any]:
    """Unpack durable attempts (and optional rendered GIFs) from a Colab tarball."""
    from experiments.forge.contracts import atomic_json, file_lock, utc_now

    root = Path(root).resolve()
    attempts_root = root / "reports/forge/attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    media_root = (
        Path(rendered_media_dir).resolve()
        if rendered_media_dir is not None
        else (root / "runs/forge/rendered_media").resolve()
    )

    new_attempts: set[str] = set()
    imported_submissions: dict[str, dict[str, Any]] = {}
    rendered_media_gifs: list[str] = []

    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if member.name.startswith("runs/forge/rendered_media/"):
                rel_media = member.name[len("runs/forge/rendered_media/"):]
                if not rel_media:
                    continue
                dest_media = (media_root / rel_media).resolve()
                if not dest_media.is_relative_to(media_root):
                    raise ValueError(f"Unsafe archive member path: {member.name}")
                if member.isdir():
                    dest_media.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest_media.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, dest_media.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if dest_media.suffix.lower() == ".gif":
                        rendered_media_gifs.append(str(dest_media))
                continue
            if not member.name.startswith("reports/forge/attempts/"):
                continue
            dest = (root / member.name).resolve()
            if not dest.is_relative_to(attempts_root.resolve()):
                raise ValueError(f"Unsafe archive member path: {member.name}")
            parts = member.name.split("/")
            if len(parts) >= 4:
                new_attempts.add(parts[3])
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if member.name.endswith("/request.json"):
                    env = json.loads(dest.read_text(encoding="utf-8"))
                    rid = env.get("request_id")
                    req = env.get("request")
                    if rid and isinstance(req, dict) and rid not in imported_submissions:
                        imported_submissions[rid] = req

    if register_submissions and imported_submissions:
        state_path = root / "runs/forge/queue/state.json"
        lock_path = root / "runs/forge/queue/queue.lock"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(lock_path):
            state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.is_file()
                else {
                    "schema_version": 1,
                    "campaigns": {},
                    "charges": {},
                    "decision_rounds": {},
                    "events": [],
                    "jobs": {},
                    "studies": {},
                    "submissions": {},
                }
            )
            subs = state.setdefault("submissions", {})
            for rid, req in imported_submissions.items():
                if rid not in subs:
                    subs[rid] = {
                        "request": req,
                        "status": "blocked",
                        "lifecycle": "awaiting_readout",
                        "submitted_at": utc_now(),
                        "reason": "imported completed Colab Forge run",
                    }
            atomic_json(state_path, state)

    projected: list[str] = []
    if project_receipts and new_attempts:
        from reports.forge.regenerate_technique_inventory import (
            _json_text,
            _write_changed,
            project_receipt,
        )

        receipts_dir = root / "reports/forge/technique-receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        for aid in sorted(new_attempts):
            if not (attempts_root / aid / "result.json").is_file():
                continue
            summary = project_receipt(root, aid)
            out_path = receipts_dir / f"{aid}.json"
            _write_changed(out_path, _json_text(summary))
            projected.append(str(out_path.relative_to(root)))

    return {
        "tarball": str(tar_path),
        "attempts_extracted": sorted(new_attempts),
        "projected_receipts": projected,
        "submissions_registered": sorted(imported_submissions),
        "rendered_media_gifs": sorted(rendered_media_gifs),
    }


# ---------------------------------------------------------------------------
# Parallel warm-pool executor
# ---------------------------------------------------------------------------


def execute_work_units(
    root: Path,
    units: list[CatchupUnit],
    *,
    workers: int,
    stagger_seconds: float,
    boot_timeout: int,
    max_minutes: int,
    idle_grace_minutes: int,
    stall_minutes: int,
    repo_url: str,
    run_dir: Path,
    project_receipts: bool = True,
    render_media: bool = True,
    compile_memory: bool = False,
    publish_hf: bool = False,
    hf_dry_run: bool = False,
) -> dict[str, Any]:
    """Drain work units across a warm pool of right-sized Colab GPU workers."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if not units:
        return {"status": "already_caught_up", "units_executed": 0, "results": []}

    branch_sync = prepare_branch_bundle(root, run_dir)
    # Group units by GPU type so each warm VM only serves matching-GPU units
    by_gpu: dict[str, stdlib_queue.Queue[CatchupUnit]] = {}
    for unit in units:
        by_gpu.setdefault(unit.gpu, stdlib_queue.Queue()).put(unit)

    # Allocate worker slots proportionally without exceeding total `--workers` or unit count
    worker_plan: list[tuple[str, str]] = []  # (worker_name, gpu)
    total_needed = min(max(1, workers), len(units))
    gpu_keys = sorted(by_gpu.keys())
    idx = 0
    while len(worker_plan) < total_needed:
        added_any = False
        for gpu in gpu_keys:
            if len(worker_plan) >= total_needed:
                break
            already = sum(1 for _, g in worker_plan if g == gpu)
            if already < by_gpu[gpu].qsize():
                idx += 1
                worker_plan.append((f"forge-{gpu.lower()}-w{idx}-{run_dir.name[-6:]}", gpu))
                added_any = True
        if not added_any:
            break

    ingest_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    ingested: list[dict[str, Any]] = []
    started_by_gpu: dict[str, int] = {gpu: 0 for gpu in gpu_keys}
    worker_errors: list[dict[str, str]] = []

    def _worker_loop(slot_idx: int, worker_name: str, gpu: str) -> None:
        if slot_idx > 0 and stagger_seconds > 0:
            time.sleep(slot_idx * stagger_seconds)
        worker = ColabWorker(
            worker_name=worker_name,
            gpu=gpu,
            run_dir=run_dir,
            repo_root=root,
            repo_url=repo_url,
            branch_sync=branch_sync,
            boot_timeout=boot_timeout,
            max_minutes=max_minutes,
            idle_grace_minutes=idle_grace_minutes,
            stall_minutes=stall_minutes,
        )
        q = by_gpu[gpu]
        try:
            try:
                worker.start()
            except (RuntimeError, OSError, ValueError) as err:
                with ingest_lock:
                    worker_errors.append({"worker": worker_name, "gpu": gpu, "stage": "start", "error": str(err)})
                worker._log(f"Worker startup skipped/failed (other active workers will drain the queue): {err}")
                return
            with ingest_lock:
                started_by_gpu[gpu] = started_by_gpu.get(gpu, 0) + 1
            while True:
                try:
                    unit = q.get_nowait()
                except stdlib_queue.Empty:
                    break
                try:
                    if unit.study_path and unit.study_id:
                        ephemeral_view = None
                        study_file = Path(unit.study_path)
                        study_id = unit.study_id
                    else:
                        ephemeral_view, study_file, study_id = build_ephemeral_catchup_bundle(root, unit, run_dir)
                    res = worker.run_unit(
                        unit,
                        study_file=study_file,
                        study_id=study_id,
                        ephemeral_view_file=ephemeral_view,
                        render_media=render_media,
                    )
                    with ingest_lock:
                        results.append(res)
                        if res.get("tarball"):
                            ing = ingest_tarball(
                                root,
                                Path(res["tarball"]),
                                project_receipts=project_receipts,
                                rendered_media_dir=(run_dir / "rendered_media") if render_media else None,
                            )
                            ingested.append(ing)
                except (RuntimeError, OSError, ValueError, tarfile.TarError, json.JSONDecodeError) as err:
                    with ingest_lock:
                        worker_errors.append({"worker": worker_name, "gpu": gpu, "stage": f"unit:{unit.unit_id}", "error": str(err)})
                    worker._log(f"Unit {unit.unit_id} errored: {err}")
                finally:
                    q.task_done()
        finally:
            worker.stop()

    with ThreadPoolExecutor(max_workers=len(worker_plan)) as pool:
        futs = [pool.submit(_worker_loop, i, name, gpu) for i, (name, gpu) in enumerate(worker_plan)]
        for fut in as_completed(futs):
            fut.result()

    if not results and worker_errors:
        raise RuntimeError(f"All Colab workers failed to start/run: {worker_errors[:3]}")

    if compile_memory:
        subprocess.run([sys.executable, "-m", "experiments.forge", "compile", "--summaries-only"], cwd=root, check=True)

    rendered_media_dir = run_dir / "rendered_media"
    rendered_media_gif_count = sum(len(item.get("rendered_media_gifs") or []) for item in ingested)
    media_ingest = None
    media_upload = None
    hf_receipt = None
    if publish_hf:
        if render_media and rendered_media_gif_count > 0 and rendered_media_dir.is_dir():
            from experiments.forge.leaderboard_media import (
                ingest_rendered_media_dirs,
                upload_staged_media_to_hf,
            )

            staging_dir = root / "runs/forge/leaderboard_media_staging"
            media_ingest = ingest_rendered_media_dirs(
                root,
                [rendered_media_dir],
                staging_dir,
                cohort_match="colab_replay_media",
                write_index=not hf_dry_run,
            )
            staged_rels = set(media_ingest.get("staged_relpaths") or [])
            if staged_rels:
                media_upload = upload_staged_media_to_hf(
                    root,
                    staging_dir,
                    dry_run=hf_dry_run,
                    only_relpaths=staged_rels,
                    commit_message=f"Upload {len(staged_rels)} Colab actual-training leaderboard GIF(s)",
                )

        from experiments.forge.hf_publish import publish_to_hf

        hf_receipt = publish_to_hf(root, inventory=True, dry_run=hf_dry_run)

    summary = {
        "schema_version": 1,
        "completed_at": utc_stamp(),
        "run_dir": str(run_dir),
        "branch_sync": branch_sync,
        "render_media": render_media,
        "rendered_media_dir": str(rendered_media_dir) if render_media else None,
        "rendered_media_gif_count": rendered_media_gif_count,
        "workers_used": worker_plan,
        "started_by_gpu": started_by_gpu,
        "worker_errors": worker_errors,
        "units_executed": len(results),
        "results": results,
        "ingested": ingested,
        "media_ingest": media_ingest,
        "media_upload": media_upload,
        "hf_receipt": hf_receipt,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def cmd_catchup(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    require_media = bool(args.require_media or args.cache_mode == "media")
    render_media = bool(args.render_media or require_media)
    plan = compute_catchup_plan(
        root,
        view_id=args.view,
        through_tier=args.through_tier,
        candidates=args.candidate,
        include_selected_families=not args.only_candidates,
        only_tasks=args.task,
        gpu_policy=args.gpu,
        prefer_t4=args.prefer_t4,
        strict_source=args.strict_source or args.cache_mode == "exact",
        require_energy=args.require_energy or args.cache_mode == "energy",
        require_media=require_media,
        require_gpu_model=args.require_gpu_model,
        changed_since=None if args.no_git_diff else args.changed_since,
        skip_preflight_blocked=not args.include_blocked,
        max_tasks_per_unit=args.max_tasks_per_unit,
    )
    units: list[CatchupUnit] = plan.pop("_units")
    print(json.dumps({k: v for k, v in plan.items() if k != "missing_cells"}, indent=2))
    if args.plan_only or not units:
        if not units:
            print("\n[colab_forge] Cache is already 100% caught up — 0 Colab GPUs needed.")
        return 0

    run_dir = root / "runs/forge/colab" / f"catchup-{run_id_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"\n[colab_forge] Launching {min(args.workers, len(units))} worker(s) for {len(units)} work unit(s).")
    print(f"[colab_forge] Tail logs: tail -f {run_dir}/forge-*.log")

    summary = execute_work_units(
        root,
        units,
        workers=args.workers,
        stagger_seconds=args.stagger_seconds,
        boot_timeout=args.boot_timeout,
        max_minutes=args.max_minutes,
        idle_grace_minutes=args.idle_grace_minutes,
        stall_minutes=args.stall_minutes,
        repo_url=args.repo_url,
        run_dir=run_dir,
        project_receipts=not args.no_project_receipts,
        render_media=render_media,
        compile_memory=args.compile,
        publish_hf=args.publish_hf,
        hf_dry_run=args.hf_dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(r.get("run_rc") == 0 for r in summary.get("results", [])) else 1


def cmd_run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    units: list[CatchupUnit] = []
    gpu = normalize_gpu(args.gpu)
    if gpu == "auto":
        gpu = "L4"
    for spec in args.job:
        cand, sep, study_arg = spec.partition(":")
        if not sep or not cand or not study_arg:
            raise ValueError(f"--job requires CANDIDATE:STUDY, got {spec!r}")
        study_path = Path(study_arg)
        if not study_path.is_file():
            study_path = root / "configs/forge/studies" / (study_arg if study_arg.endswith(".json") else f"{study_arg}.json")
        if not study_path.is_file():
            raise FileNotFoundError(f"Study file not found: {study_path}")
        study_doc = json.loads(study_path.read_text(encoding="utf-8"))
        study_id = str(study_doc["id"])
        scope = study_doc.get("scope") or {}
        units.append(
            CatchupUnit(
                unit_id=f"{cand}-{study_id}"[:40],
                candidate_id=cand,
                view_id=str(scope.get("view", "discriminator_stability")),
                through_tier=int(scope.get("through_tier", 1)),
                gpu=gpu,
                tasks=[],
                is_full_view_slice=True,
                estimated_seconds=float((study_doc.get("campaign") or {}).get("candidate_budget_seconds", 3600)),
                study_id=study_id,
                study_path=str(study_path.resolve()),
            )
        )

    run_dir = root / "runs/forge/colab" / f"run-{run_id_stamp()}"
    print(f"[colab_forge] Running {len(units)} explicit study job(s) on {min(args.workers, len(units))} {gpu} worker(s).")
    print(f"[colab_forge] Tail logs: tail -f {run_dir}/forge-*.log")
    summary = execute_work_units(
        root,
        units,
        workers=args.workers,
        stagger_seconds=args.stagger_seconds,
        boot_timeout=args.boot_timeout,
        max_minutes=args.max_minutes,
        idle_grace_minutes=args.idle_grace_minutes,
        stall_minutes=args.stall_minutes,
        repo_url=args.repo_url,
        run_dir=run_dir,
        project_receipts=not args.no_project_receipts,
        render_media=bool(args.render_media),
        compile_memory=args.compile,
        publish_hf=args.publish_hf,
        hf_dry_run=args.hf_dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(r.get("run_rc") == 0 for r in summary.get("results", [])) else 1


def cmd_status(_args: argparse.Namespace) -> int:
    for sub in (["colab", "sessions"], ["colab", "usage"]):
        rc, out = run_with_hard_timeout(sub, timeout_seconds=30, capture=True)
        print(out, end="")
        if rc != 0:
            return rc
    return 0


def _run_colab_with_timeout_retry(
    cmd: list[str],
    *,
    timeout_seconds: float,
    label: str,
    retries: int = DEFAULT_TRANSPORT_RETRIES,
    backoff_seconds: float = DEFAULT_TRANSPORT_RETRY_BACKOFF_SECONDS,
    capture: bool = True,
) -> tuple[int, str]:
    """Run an idempotent `colab` CLI command with bounded retry ONLY on rc=124 hard timeouts."""
    attempts = 1 + max(0, int(retries))
    backoff = max(0.0, float(backoff_seconds))
    rc, out = 0, ""
    for attempt in range(1, attempts + 1):
        rc, out = run_with_hard_timeout(cmd, timeout_seconds=timeout_seconds, capture=capture)
        if rc == 0 or rc not in TRANSIENT_TRANSPORT_RETRY_RCS or attempt >= attempts:
            return rc, out
        print(
            f"[colab_forge] {label} hit transient timeout (rc={rc}) on attempt {attempt}/{attempts}; "
            f"retrying in {backoff:.1f}s...",
            flush=True,
        )
        if backoff > 0:
            time.sleep(backoff)
    return rc, out


def _parse_colab_sessions_output(output: str) -> list[dict[str, str | None]]:
    """Parse active session rows from `colab sessions` stdout.

    Supports both google-colab-cli >=0.8 lines (`[name] endpoint | Hardware: ...` and
    `[?] endpoint | Hardware: ...`) and legacy unbracketed `name | Hardware: ...` lines.
    """
    rows: list[dict[str, str | None]] = []
    for raw in output.splitlines():
        token = raw.strip()
        if not token or token.startswith("[colab]") or "|" not in token:
            continue
        head = token.split("|", 1)[0].strip()
        if not head:
            continue
        if head.startswith("[") and "]" in head:
            close = head.index("]")
            raw_name = head[1:close].strip()
            rest = head[close + 1 :].strip().split()
            endpoint = rest[0] if rest else None
            name = None if (not raw_name or raw_name == "?") else raw_name
            if name is None and endpoint is None:
                continue
            rows.append({"name": name, "endpoint": endpoint})
        else:
            parts = head.split()
            if not parts:
                continue
            rows.append({"name": parts[-1], "endpoint": None})
    return rows


def _discover_isolated_config_sessions(
    root: Path,
    *,
    max_runs: int = DEFAULT_REAP_MAX_RUNS,
) -> tuple[list[dict[str, Any]], int]:
    """Scan recent `runs/forge/colab/*/*.colab.json` worker configs for recorded sessions.

    Returns `(entries, configs_scanned)` ordered newest-first so deduplication prefers the
    most recent worker config.
    """
    colab_dir = root.resolve() / "runs/forge/colab"
    if not colab_dir.is_dir():
        return [], 0

    run_dirs: list[tuple[float, str, Path]] = []
    for entry in colab_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        run_dirs.append((mtime, entry.name, entry))
    run_dirs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if max_runs > 0:
        run_dirs = run_dirs[:max_runs]

    discovered: list[dict[str, Any]] = []
    configs_scanned = 0
    for _, _, run_dir in run_dirs:
        cfg_files: list[tuple[float, str, Path]] = []
        for cfg in run_dir.glob("*.colab.json"):
            if not cfg.is_file():
                continue
            try:
                mtime = cfg.stat().st_mtime
            except OSError:
                continue
            cfg_files.append((mtime, cfg.name, cfg))
        cfg_files.sort(key=lambda item: (item[0], item[1]), reverse=True)

        for _, _, cfg in cfg_files:
            configs_scanned += 1
            try:
                doc = json.loads(cfg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(doc, dict) or not doc:
                continue
            for key, val in doc.items():
                if isinstance(val, dict):
                    name = str(val.get("name") or key).strip()
                    raw_ep = val.get("endpoint")
                    endpoint = str(raw_ep).strip() if isinstance(raw_ep, str) and raw_ep.strip() else None
                else:
                    name = str(key).strip()
                    endpoint = None
                if not name:
                    continue
                discovered.append({"config": cfg, "name": name, "endpoint": endpoint})
    return discovered, configs_scanned


def reap_sessions(
    root: Path,
    *,
    max_runs: int = DEFAULT_REAP_MAX_RUNS,
    session_timeout: float = 30.0,
    stop_timeout: float = DEFAULT_REAP_STOP_TIMEOUT_SECONDS,
    retries: int = DEFAULT_TRANSPORT_RETRIES,
    backoff_seconds: float = DEFAULT_TRANSPORT_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Stop active Colab sessions in the default config and recent isolated worker configs."""
    sessions_rc, sessions_out = _run_colab_with_timeout_retry(
        ["colab", "sessions"],
        timeout_seconds=session_timeout,
        label="colab sessions",
        retries=retries,
        backoff_seconds=backoff_seconds,
        capture=True,
    )
    if sessions_out:
        print(sessions_out, end="" if sessions_out.endswith("\n") else "\n", flush=True)

    server_known = sessions_rc == 0
    parsed_rows = _parse_colab_sessions_output(sessions_out) if server_known else []
    active_server_endpoints = {r["endpoint"] for r in parsed_rows if r["endpoint"]}
    unknown_server_endpoints = {r["endpoint"] for r in parsed_rows if r["name"] is None and r["endpoint"]}
    server_has_any_sessions = bool(parsed_rows)
    server_has_endpoint_ids = bool(active_server_endpoints)

    stopped_default: list[str] = []
    stopped_isolated: list[dict[str, str | None]] = []
    not_found: list[dict[str, str | None]] = []
    failed: list[dict[str, Any]] = []
    skipped_stale = 0
    skipped_duplicate = 0

    seen_default_names: set[str] = set()
    handled_endpoints: set[str] = set()

    def _classify_stop(
        cmd: list[str],
        *,
        label: str,
        session: str,
        endpoint: str | None,
        config_path: Path | None,
    ) -> None:
        stop_rc, stop_out = _run_colab_with_timeout_retry(
            cmd,
            timeout_seconds=stop_timeout,
            label=label,
            retries=retries,
            backoff_seconds=backoff_seconds,
            capture=True,
        )
        if stop_out:
            print(stop_out, end="" if stop_out.endswith("\n") else "\n", flush=True)
        if stop_rc != 0:
            failed.append(
                {
                    "session": session,
                    "endpoint": endpoint,
                    "config": str(config_path) if config_path else None,
                    "rc": stop_rc,
                }
            )
            return
        if "not found" in stop_out.lower():
            not_found.append(
                {
                    "session": session,
                    "endpoint": endpoint,
                    "config": str(config_path) if config_path else None,
                }
            )
            return
        if config_path is None:
            stopped_default.append(session)
        else:
            stopped_isolated.append(
                {
                    "session": session,
                    "endpoint": endpoint,
                    "config": str(config_path),
                }
            )

    # 1. Stop named sessions tracked in the default Colab CLI config.
    for row in parsed_rows:
        sess = row["name"]
        ep = row["endpoint"]
        if sess is None:
            continue
        if sess in seen_default_names or (ep and ep in handled_endpoints):
            skipped_duplicate += 1
            continue
        seen_default_names.add(sess)
        print(f"[reap] Stopping default-config session {sess}...", flush=True)
        _classify_stop(
            ["colab", "stop", "-s", sess],
            label=f"colab stop {sess}",
            session=sess,
            endpoint=ep,
            config_path=None,
        )
        if ep:
            handled_endpoints.add(ep)

    if not server_known:
        print(
            f"[reap] Warning: `colab sessions` failed (rc={sessions_rc}); "
            "falling back to stopping sessions recorded in recent isolated worker configs.",
            flush=True,
        )

    # 2. Scan recent isolated worker configs (`runs/forge/colab/*/*.colab.json`).
    isolated_entries, configs_scanned = _discover_isolated_config_sessions(root, max_runs=max_runs)
    seen_isolated_keys: set[tuple[Path, str]] = set()
    for entry in isolated_entries:
        cfg: Path = entry["config"]
        sess = entry["name"]
        ep = entry["endpoint"]
        resolved_key = (cfg.resolve(), sess)
        if resolved_key in seen_isolated_keys or (ep and ep in handled_endpoints):
            skipped_duplicate += 1
            continue
        seen_isolated_keys.add(resolved_key)

        if server_known:
            if not server_has_any_sessions:
                skipped_stale += 1
                continue
            if server_has_endpoint_ids and ep is not None and ep not in active_server_endpoints:
                skipped_stale += 1
                continue
            if server_has_endpoint_ids and ep is None and not unknown_server_endpoints:
                skipped_stale += 1
                continue

        ep_label = f" (endpoint={ep})" if ep else ""
        print(f"[reap] Stopping isolated session {sess}{ep_label} via {cfg}...", flush=True)
        _classify_stop(
            ["colab", "--config", str(cfg), "stop", "-s", sess],
            label=f"colab --config {cfg.name} stop {sess}",
            session=sess,
            endpoint=ep,
            config_path=cfg,
        )
        if ep:
            handled_endpoints.add(ep)

    unmatched_orphans = sorted(ep for ep in unknown_server_endpoints if ep not in handled_endpoints)
    total_stopped = len(stopped_default) + len(stopped_isolated)
    print(
        f"[reap] Summary: stopped={total_stopped} "
        f"(default={len(stopped_default)}, isolated={len(stopped_isolated)}), "
        f"not_found={len(not_found)}, failed={len(failed)}, "
        f"skipped_stale={skipped_stale}, skipped_duplicate={skipped_duplicate}, "
        f"configs_scanned={configs_scanned}, unmatched_orphans={len(unmatched_orphans)}",
        flush=True,
    )
    if unmatched_orphans:
        print(
            "[reap] Warning: active server endpoint(s) not matched in default or recent local "
            f"worker configs: {', '.join(unmatched_orphans)}",
            flush=True,
        )

    return {
        "sessions_rc": sessions_rc,
        "stopped_default": stopped_default,
        "stopped_isolated": stopped_isolated,
        "not_found": not_found,
        "failed": failed,
        "skipped_stale": skipped_stale,
        "skipped_duplicate": skipped_duplicate,
        "configs_scanned": configs_scanned,
        "unmatched_orphans": unmatched_orphans,
    }


def cmd_reap(args: argparse.Namespace) -> int:
    """Stop all active Colab sessions in default and recent isolated worker configs."""
    root = getattr(args, "root", REPO_ROOT).resolve()
    max_runs = int(getattr(args, "max_runs", DEFAULT_REAP_MAX_RUNS))
    stop_timeout = float(getattr(args, "stop_timeout", DEFAULT_REAP_STOP_TIMEOUT_SECONDS))
    summary = reap_sessions(root, max_runs=max_runs, stop_timeout=stop_timeout)
    status_rc = cmd_status(args)
    if status_rc != 0:
        return status_rc
    if summary["sessions_rc"] != 0 or summary["failed"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT, help="ParticleGAN repository root")
    sub = p.add_subparsers(dest="command", required=True)

    def _add_pool_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--gpu", default="auto", help="auto (right-size), cpu, T4, L4, G4, A100, or H100")
        sp.add_argument("--workers", type=int, default=2, help="max parallel Colab GPU VMs (default 2)")
        sp.add_argument("--stagger-seconds", type=float, default=10.0, help="delay between VM allocations")
        sp.add_argument("--boot-timeout", type=int, default=150, help="seconds to wait for colab new")
        sp.add_argument("--max-minutes", type=int, default=120, help="hard VM watchdog wall-clock cap")
        sp.add_argument("--idle-grace-minutes", type=int, default=10, help="VM self-unassign grace after job done")
        sp.add_argument("--stall-minutes", type=int, default=25, help="VM self-unassign timeout if logs stall")
        sp.add_argument("--repo-url", default=DEFAULT_REPO_URL)
        sp.add_argument("--no-project-receipts", action="store_true", help="skip projecting compact technique-receipts")
        sp.add_argument(
            "--render-media",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="render actual-training GIFs on Colab VM from saved observations and download them (default: enabled; pass --no-render-media to skip)",
        )
        sp.add_argument("--compile", action="store_true", help="run `forge compile --summaries-only` locally after ingest")
        sp.add_argument("--publish-hf", action="store_true", help="publish `--inventory` locally to Hugging Face after ingest")
        sp.add_argument("--hf-dry-run", action="store_true", help="pass --dry-run to HF publish")

    cu = sub.add_parser("catchup", help="One-command decentralized branch cache catch-up across parallel Colab GPUs")
    cu.add_argument("--view", "--goal", dest="view", default="discriminator_stability")
    cu.add_argument("--through-tier", type=int, default=1)
    cu.add_argument("--candidate", action="append", default=[], help="extra candidate(s) to include; repeatable")
    cu.add_argument("--only-candidates", action="store_true", help="run only explicit --candidate values (skip family-current-v1)")
    cu.add_argument("--task", action="append", default=None, help="restrict catch-up to specific task id(s); repeatable")
    cu.add_argument("--cache-mode", choices=("branch", "exact", "energy", "media"), default="branch",
                    help="branch=fill missing cells across merged branches; exact=strict source digest; energy=fill missing NVML J-to-PASS; media=fill missing actual-training GIFs")
    cu.add_argument("--strict-source", action="store_true", help="require exact current source_digest for cache hit")
    cu.add_argument("--require-energy", action="store_true", help="require hardware NVML energy telemetry for cache hit")
    cu.add_argument("--require-media", action="store_true", help="require verified leaderboard training GIF for cache hit (implies --render-media)")
    cu.add_argument("--require-gpu-model", default=None, help="require specific GPU model (e.g. 'NVIDIA L4') for cache hit")
    cu.add_argument("--changed-since", default="origin/develop", help="git ref for smart recipe/task change invalidation")
    cu.add_argument("--no-git-diff", action="store_true", help="disable git diff recipe/task change invalidation")
    cu.add_argument("--include-blocked", action="store_true", help="do not skip statically preflight-blocked pairs")
    cu.add_argument("--prefer-t4", action="store_true", help="under --gpu auto, route lightweight Tier-1 smoke to T4")
    cu.add_argument("--max-tasks-per-unit", type=int, default=20, help="shard large missing task lists across workers")
    cu.add_argument("--plan-only", "--dry-run", dest="plan_only", action="store_true",
                    help="compute and print cache diff without allocating any Colab GPU")
    _add_pool_args(cu)

    rn = sub.add_parser("run", help="Run explicit CANDIDATE:STUDY job(s) across parallel Colab GPUs")
    rn.add_argument("--job", action="append", required=True, metavar="CANDIDATE:STUDY", help="repeatable")
    _add_pool_args(rn)

    sub.add_parser("status", help="Show active Colab sessions and compute-unit balance")
    rp = sub.add_parser("reap", help="Stop all active Colab sessions (default + isolated worker configs)")
    rp.add_argument("--max-runs", type=int, default=DEFAULT_REAP_MAX_RUNS,
                    help="max recent run directories under runs/forge/colab/ to scan for *.colab.json configs")
    rp.add_argument("--stop-timeout", type=float, default=DEFAULT_REAP_STOP_TIMEOUT_SECONDS,
                    help="hard timeout in seconds for each `colab stop` invocation")
    return p


def main(argv: list[str] | None = None) -> int:
    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"\n[colab_forge] Received signal {signum}; releasing all active Colab VMs...", file=sys.stderr, flush=True)
        _GLOBAL_REGISTRY.stop_all()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    args = build_parser().parse_args(argv)
    if args.command == "catchup":
        return cmd_catchup(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "reap":
        return cmd_reap(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
