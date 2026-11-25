"""Deterministic, CPU-only historical inventory and conservative evidence import.

Importing a recorded verdict does not certify it under a Forge protocol. Git is
used read-only; frozen sources are read at their original commits, never checked
out or executed. Generated Forge files are deliberately outside the source scope.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import gzip
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

SCHEMA_VERSION = 1
IMPORTER_VERSION = "forge-history-v1"
BASELINE_REVISION = "92dc0319"
LRFREE_REVISION = "0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b"
LRFREE_ROOT = "reports/toy100/lrfree-search"
UPSTREAM_VECTOR_REVISION = "a8b9d3977701ca700d9918ac66d40ac814b9f9ba"
UPSTREAM_VECTOR_ROOTS = ("reports/transfer_suite/anisotropic_core_metric",
                         "reports/transfer_suite/silu_rare_collapse")
SOURCE_ROOTS = ("benchmarks", "experiments", "configs", "reports")
EXCLUDED_PREFIXES = ("configs/forge/", "reports/forge/")
GATE_STATUSES = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "NOT_RUN", "BLOCKED"}
KNOWN_TASKS = {
    "grid100", "rotated100", "staggered100", "mode_hold", "two_pole",
    "trajectory", "residual_student", "unipolar", "ae_gan_hold", "cover_leftover",
    "unused_token_hold", "mid_scale_identity", "ring_shift", "stationary",
    "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2",
    "vector_anisotropic", "vector_overlap", "vector_spiral", "vector_two_broad",
    "vector_unequal_mass", "vector_unequal_width",
}
REPORT_FAMILIES = {
    "toy100", "behavioral_baseline", "locked_shared", "transfer_suite",
    "learned_lr", "smart_descent", "paired_error_2d", "cifar-ddgan",
    "cifar-particle-ae", "cifar-particle-ddgan", "denoising-toy", "gym",
    "lunar_fast", "mog-autoencoder", "mog-vae", "prior-comparison",
    "readme-100gaussians", "sparse-ucd", "trajectory", "transition", "releases",
    "atlas-explanation", "e22-animation", "r1-integrations", "develop-gates-20261001",
    "paired-residual-toy", "toy_audit", "gram-profile", "prior-ablation-2026-10-25",
    "axiom", "four-laws-audit",
}
BENCHMARK_FAMILIES = {"cifar_toy", "init_research", "learned_lr", "legacy", "locked_shared",
                      "paired_error_2d", "smart_descent", "toy100", "transfer_suite",
                      "routed_conditioning", "toy_audit"}
CONFIG_FAMILIES = {"100gaussians", "cifar_ddgan", "cifar_particle_ae",
                   "cifar_particle_ddgan", "denoising", "gym", "mog", "mog_vae",
                   "sparse", "toy100", "trajectory", "transition", "prior_ablation_cifar"}
SCIENCE_MARKERS = ("status", "raw_status", "verdict", "harness_result_status", "record_kind")


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def _scoped(path: str) -> bool:
    return (path.split("/", 1)[0] in SOURCE_ROOTS
            and not path.startswith(EXCLUDED_PREFIXES))


def _tree(root: Path, revision: str, prefixes: tuple[str, ...] = SOURCE_ROOTS) -> dict[str, str]:
    result = {}
    for item in _git(root, "ls-tree", "-r", "-z", revision, "--", *prefixes).split(b"\0"):
        if item:
            metadata, path = item.split(b"\t", 1)
            _, kind, oid = metadata.decode().split()
            name = path.decode()
            if kind == "blob" and _scoped(name):
                result[name] = oid
    return result


def _tracked(root: Path) -> dict[str, str]:
    result = {}
    for item in _git(root, "ls-files", "--stage", "-z", "--", *SOURCE_ROOTS).split(b"\0"):
        if item:
            metadata, path = item.split(b"\t", 1)
            _, oid, stage = metadata.decode().split()
            name = path.decode()
            if stage == "0" and _scoped(name):
                result[name] = oid
    return result


def classify(path: str) -> dict[str, Any]:
    """Classify role, never scientific outcome; unknown families fail coverage."""
    if path.startswith("experiments/forge/"):
        return {"family": "forge_engine", "role": "engine_support", "suggested_tiers": []}
    p = PurePosixPath(path)
    parts = p.parts
    root = parts[0]
    family = "unclassified"
    if path.startswith(LRFREE_ROOT + "/"):
        family = "lrfree/" + (parts[3] if len(parts) > 4 else "overview")
    elif root == "benchmarks":
        if len(parts) > 2 and parts[1] in BENCHMARK_FAMILIES:
            family = "benchmark/" + parts[1]
        elif len(parts) == 2:
            family = "benchmark/shared"
    elif root == "reports":
        if len(parts) == 2:
            family = "report/project"
        elif parts[1] in REPORT_FAMILIES:
            family = "report/" + parts[1]
            if parts[1] == "toy100" and len(parts) > 3:
                family += "/" + parts[2]
            elif any(path.startswith(prefix + "/") for prefix in UPSTREAM_VECTOR_ROOTS):
                family += "/" + parts[2]
    elif root == "configs":
        if len(parts) > 1 and (parts[1] in CONFIG_FAMILIES or parts[1].startswith("speed_")):
            family = "config/" + parts[1]
        elif len(parts) == 2 and parts[1] == "repo_bloat_allowlist.json":
            family = "config/repo_guard"
    elif root == "experiments":
        name = p.stem
        domain = next((x for x in ("cifar", "denoising", "mog", "gym", "sparse",
                                  "trajectory", "transition", "toy", "100gaussians")
                       if x in name), "shared")
        family = "experiment/" + domain

    name = p.name.lower()
    if p.suffix.lower() in {".md", ".txt"} or name in {"license", "source"}:
        role = "narrative"
    elif p.suffix in {".py", ".sh"}:
        if name in {"__init__.py", "config.py", "models.py", "metrics.py", "recipe.py"}:
            role = "support"
        elif name in {"mechanism.py", "latent.py", "response.py", "particle_update.py"}:
            role = "candidate_implementation"
        elif any(x in name for x in ("render", "plot", "preview")):
            role = "visualization"
        elif name.startswith(("test_", "check_", "verify", "audit")):
            role = "evaluator_or_audit"
        elif any(x in name for x in ("analyze", "summar", "score", "collect", "leaderboard")):
            role = "analysis"
        elif any(x in name for x in ("run", "train", "probe", "study", "screen", "search",
                                     "hold", "shift", "native", "replay", "__main__")):
            role = "driver"
        else:
            role = "support"
        if "/sources/" in path or "/reproduction/" in path:
            role = "frozen_" + role
    elif p.suffix in {".toml", ".yaml", ".yml"} or root == "configs":
        role = "configuration"
    elif p.suffix in {".json", ".jsonl", ".gz"}:
        role = "structured_evidence"
    elif p.suffix in {".pt", ".npz", ".npy", ".zip"}:
        role = "evidence_artifact"
    elif p.suffix in {".png", ".gif", ".jpg", ".svg", ".mp4"}:
        role = "visualization_artifact"
    elif p.suffix == ".patch":
        role = "candidate_patch"
    else:
        role = "support_artifact"
    suggested = [2]
    if any(x in path for x in ("init_research", "check_", "smoke", "verify")):
        suggested = [1]
    elif any(x in path for x in ("h_stability", "continuous", "shift", "extension", "14k")):
        suggested = [3]
    if role not in {"driver", "frozen_driver", "evaluator_or_audit", "frozen_evaluator_or_audit"}:
        suggested = []
    return {"family": family, "role": role, "suggested_tiers": suggested}


def validate_inventory(root: Path, catalog: dict | None = None) -> dict[str, Any]:
    """Detect added/removed tracked sources independently of classifier defaults."""
    if catalog is None:
        path = root / "configs/forge/catalog.json"
        catalog = json.loads(path.read_text()) if path.exists() else {"files": []}
    tracked = _tracked(root)
    declared = {entry["path"] for entry in catalog.get("files", [])}
    missing = sorted(set(tracked) - declared)
    stale = sorted(declared - set(tracked))
    unclassified = sorted(p for p in tracked if classify(p)["family"] == "unclassified")
    return {"valid": not (missing or stale or unclassified), "missing": missing,
            "stale": stale, "unclassified": unclassified,
            "tracked_count": len(tracked), "declared_count": len(declared)}


def inventory(root: Path) -> dict[str, Any]:
    """Return complete tracked source classification and coverage (no writes)."""
    root = Path(root)
    entries = [{"path": p, "git_blob": oid, **classify(p)}
               for p, oid in sorted(_tracked(root).items())]
    return {"schema_version": SCHEMA_VERSION, "source_roots": list(SOURCE_ROOTS),
            "excluded_prefixes": list(EXCLUDED_PREFIXES), "files": entries,
            "counts": {"files": len(entries),
                       "by_family": dict(sorted(Counter(e["family"] for e in entries).items())),
                       "by_role": dict(sorted(Counter(e["role"] for e in entries).items()))},
            "coverage": validate_inventory(root),
            "tier_note": "Suitability only; qualification_tier belongs to a versioned view.",
            "timing_note": "No measured runtime or FLOP claim is inferred from file roles."}


class Source:
    def __init__(self, root: Path, revision: str, prefixes: tuple[str, ...]):
        self.root = root
        self.revision = _git(root, "rev-parse", revision + "^{commit}").decode().strip()
        self.files = _tree(root, self.revision, prefixes)
        self.cache: dict[str, bytes] = {}

    def read(self, path: str) -> bytes:
        if path not in self.cache:
            self.cache[path] = _git(self.root, "show", self.revision + ":" + path)
        return self.cache[path]

    def json(self, path: str) -> Any:
        raw = self.read(path)
        return json.loads(gzip.decompress(raw) if path.endswith(".gz") else raw)

    def receipt(self, path: str) -> dict:
        return {"path": path, "revision": self.revision, "git_blob": self.files.get(path),
                "sha256": sha256(self.read(path)).hexdigest(),
                "url": f"https://github.com/255BITS/ParticleGAN/blob/{self.revision}/{path}"}


def normalize_status(raw: Any, *, capability_reason: str | None = None) -> tuple[str, str | None]:
    status = str(raw or "UNKNOWN").upper()
    if status == "ERROR" and capability_reason:
        return "BLOCKED", capability_reason
    if status in GATE_STATUSES:
        return status, None
    if status in {"NOT_CONVERGED", "FAILED"}:
        return "FAIL", "Recorded failure to converge within the declared protocol."
    if status in {"ERROR", "TIMEOUT", "CANCELLED", "INTERRUPTED"}:
        return "INCOMPLETE", "Execution did not produce a complete scientific gate verdict."
    return "NOT_RUN", "No recognized scientific verdict is recorded."


def _task(task_id: str, raw: dict, receipt: dict, *, capability_reason: str | None = None,
          negative_control: bool = False) -> dict:
    original = raw.get("raw_status", raw.get("status", raw.get("verdict", raw.get("harness_result_status"))))
    if original is None and raw.get("record_kind") == "not_run":
        original = "NOT_RUN"
    gate, default_reason = normalize_status(original, capability_reason=capability_reason)
    metrics = {k: raw[k] for k in (
        "final", "metrics", "passing_checks", "observations", "final_streak", "first_arrival",
        "native", "native_gate", "coverage", "accuracy", "terminal_checks", "terminal_accuracy",
        "holdout", "holdout_pass", "thresholds", "pass_rule", "stream_deviations",
        "post_convergence", "extension", "standard_hold", "subbudget_7000", "prefix_parity",
        "native_budgets", "seven_k_prefix_verification", "last_observation", "shift_recovery",
        "gate", "hold_window", "continued_hold", "checks", "deadline_passing_checks", "deadline_checks",
    ) if k in raw}
    # EMA and clean sampling are observations, not replacement gate statuses.
    diagnostics = {k: raw[k] for k in raw if k.startswith(("ema_", "clean_"))}
    return {"task_id": task_id, "gate_status": gate, "raw_status": original,
            "evidence_scope": "historical", "metrics": metrics, "diagnostics": diagnostics,
            "cost": {"seconds": raw.get("seconds"), "train_seconds": raw.get("train_seconds"),
                     "steps": raw.get("completed_steps", raw.get("steps", raw.get("step"))),
                     "peak_gpu_mib": raw.get("max_gpu_mib"), "flops": None,
                     "flops_status": "unavailable"},
            "evidence": [receipt], "reason": raw.get("reason") or raw.get("error") or default_reason,
            "role": "negative_control" if negative_control else "candidate",
            "raw_result": raw}


def _record(source: Source, path: str, candidate: str | None, tasks: list[dict], *,
            context: dict | None = None, hypothesis: str | None = None,
            record_type: str = "scientific") -> dict:
    context = context or {}
    candidate = candidate or "unresolved:" + path
    receipt = source.receipt(path)
    # Identity includes exact source/context; similarly named packages never merge.
    revision = _digest({"source": receipt, "candidate": candidate, "context": context})
    record_digest = _digest({"revision": revision, "tasks": tasks})
    slug = re.sub(r"[^a-z0-9]+", "-", candidate.lower()).strip("-")[:64]
    header = context.get("header") or {}
    recipe = (context.get("recipe") or context.get("overrides") or header.get("overrides")
              or context.get("declared_config", {}).get("content") or {})
    package = context.get("package_sha256") or header.get("package_sha256")
    mechanism = "sampling_only_patch" if "/row-em-" in path else "unknown"
    if "/dtrack2/" in path:
        mechanism = "parameter_change"
    elif any(part in path for part in ("/critic-floor/", "/structural100/", "/direct-particle-adam/")):
        mechanism = "training_formulation"
    prior = {"kind": recipe.get("prior_kind", "unknown"), "sigma": recipe.get("sigma_rel"),
             "standardize": recipe.get("standardize"), "trainable": "unknown",
             "scope": "historical; new learned-MoG defaults do not alter this receipt"}
    statuses = Counter(t["gate_status"] for t in tasks)
    return {"schema_version": SCHEMA_VERSION, "record_id": f"history-{slug}-{record_digest[:12]}",
            "record_type": record_type, "candidate_id": candidate, "candidate_revision": revision,
            "source": receipt, "evidence_scope": "historical", "task_results": tasks,
            "hypothesis": hypothesis or context.get("note") or "Unknown; consult the linked source narrative.",
            "mechanism_class": mechanism, "prior": prior,
            "claim_contract": {"learning_regime": "unverified", "clock_free_eligible": False,
                               "scoring_weights": "live" if tasks else "unknown",
                               "sampling_law": {"eval_output_noise": context.get("eval_output_noise"),
                                                "options": header.get("options", context.get("candidate_options"))},
                               "ema_decay": recipe.get("ema_decay"),
                               "training_and_public_sampling_identical": False if mechanism == "sampling_only_patch" else "unknown"},
            "provenance": {"importer_version": IMPORTER_VERSION, "source_revision": source.revision,
                           "package_sha256": package, "config": recipe or None,
                           "context": context, "verified_by_forge": False,
                           "reuse_eligible": False,
                           "verification": "Imported frozen recorded verdicts; no fresh regrading or qualification.",
                           "local_bulk_artifacts": "Availability unverified; original local references are retained."},
            "conclusion": ("Historical recorded task outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
                           if tasks else "Narrative context only; no scientific pass is inferred."),
            "next_action": "Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse."}


def _extract_task_groups(data: Any, candidate: str | None = None) -> list[tuple[str | None, list[tuple[str, dict]], dict]]:
    """Recognize explicit result structures, leaving ambiguous schemas unresolved."""
    groups: list[tuple[str | None, list[tuple[str, dict]], dict]] = []
    if not isinstance(data, dict):
        return groups
    candidate = data.get("cand", data.get("candidate", candidate))
    if "task" in data and any(k in data for k in SCIENCE_MARKERS):
        return [(candidate, [(str(data["task"]), data)], data)]
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        recognized = [(str(x["task"]), x) for x in tasks if isinstance(x, dict) and "task" in x
                      and any(k in x for k in SCIENCE_MARKERS)]
        if recognized:
            return [(candidate, recognized,
                     {k: v for k, v in data.items() if k != "tasks"})]
    if isinstance(tasks, dict):
        recognized = [(str(k), v) for k, v in tasks.items() if isinstance(v, dict)
                      and any(x in v for x in SCIENCE_MARKERS)]
        if recognized:
            return [(candidate, recognized,
                     {k: v for k, v in data.items() if k != "tasks"})]
    direct = [(k, v) for k, v in data.items() if k in KNOWN_TASKS and isinstance(v, dict)
              and any(x in v for x in SCIENCE_MARKERS)]
    if direct:
        return [(candidate, direct,
                 {k: v for k, v in data.items() if k not in KNOWN_TASKS})]
    for key, value in data.items():
        if isinstance(value, dict):
            container = key in {"arms", "extensions", "gates", "native", "tasks", "task_results", "records", "results", "candidates"}
            found = _extract_task_groups(value, candidate if container else key)
            for child_candidate, rows, context in found:
                inherited = {k: v for k, v in data.items() if k in (
                    "package_sha256", "overrides", "candidate_options", "config_hash", "initialization_scope")}
                groups.append((child_candidate, rows, {**inherited, **context}))
    return groups


def _related_receipts(source: Source, path: str) -> list[dict]:
    """Compact provenance stays in cards even when the frozen tree is not checked out."""
    directory = str(PurePosixPath(path).parent) + "/"
    stem = PurePosixPath(path).name.removesuffix("-result.json")
    exact = {directory + stem + suffix for suffix in ("-fixture.json", "-job-header.json")}
    exact.update(directory + name for name in ("manifest.json", "source-manifest.json",
                                              "source-sha256.json", "overrides.json", "parity.json",
                                              "candidate-options.json"))
    return [{**source.receipt(p), "content": source.json(p)} for p in sorted(exact)
            if p in source.files and p.endswith(".json")]


def _import_structured(source: Source) -> tuple[list[dict], set[str], list[dict]]:
    records, consumed, gaps = [], set(), []
    for path in sorted(source.files):
        name = PurePosixPath(path).name
        selected = (path.startswith(LRFREE_ROOT + "/") and path.endswith(".json")
                    and (name.endswith("-result.json") or "/results/" in path
                         or name in {"results.json", "native100-results.json", "all22-summary.json"})) or (
            path.startswith("reports/") and name in {"results.json", "summary.json", "leaderboard.json"})
        if not selected:
            continue
        try:
            data = source.json(path)
        except (ValueError, UnicodeError) as exc:
            gaps.append({"kind": "invalid_json", "source": source.receipt(path), "reason": str(exc)})
            continue
        groups = _extract_task_groups(data)
        if not groups:
            continue
        consumed.add(path)
        for candidate, rows, context in groups:
            # Preserve enclosing protocol statements when summaries group candidates.
            if isinstance(data, dict):
                context = {**{k: data[k] for k in ("schema", "protocol", "scoring", "primary_initialization", "native_gate")
                             if k in data}, **context}
            support = _related_receipts(source, path)
            context = {**context, "supporting_receipts": support}
            if len(rows) == 1:
                for key in ("package_sha256", "config_hash", "candidate_options", "eval_output_noise"):
                    if key in rows[0][1]:
                        context.setdefault(key, rows[0][1][key])
            config_paths = {str(PurePosixPath(path).parent / "configs" / (str(candidate).removesuffix("-14k") + ".json"))}
            for _, raw in rows:
                if raw.get("overrides_config"):
                    config_paths.add(str(PurePosixPath(path).parent / raw["overrides_config"]))
            for config_path in sorted(config_paths):
                if config_path in source.files:
                    context["declared_config"] = {**source.receipt(config_path), "content": source.json(config_path)}
            receipt = source.receipt(path)
            task_results = []
            for task_id, raw in rows:
                refusal = None
                if path == LRFREE_ROOT + "/row-em-renew/all22-summary.json" and raw.get("status") == "ERROR":
                    refusal = "Custom-host parity refuses the _sigma_intrinsic_scale capability before training; see the bound row-em-renew README."
                task_results.append(_task(task_id, raw, receipt, capability_reason=refusal))
            record = _record(source, path, candidate, task_results, context=context)
            if "/row-em-renew/" in path:
                record["mechanism_class"] = "sampling_only_patch"
                record["provenance"]["interpretation_source"] = source.receipt(LRFREE_ROOT + "/row-em-renew/README.md")
                record["conclusion"] += " Public sampling calibration changes the sampling law; this is not a training-mechanism improvement."
                for task in record["task_results"]:
                    reference = LRFREE_ROOT + "/critic-floor/" + task["task_id"] + "-result.json"
                    if reference in source.files and task["cost"]["seconds"] is not None:
                        reference_seconds = source.json(reference).get("seconds")
                        if reference_seconds:
                            task["cost"]["historical_wall_time_comparison"] = {
                                "reference_seconds": reference_seconds,
                                "ratio": task["cost"]["seconds"] / reference_seconds,
                                "reference": source.receipt(reference),
                                "scope": "Recorded runtime only; not a hardware-independent FLOP or speed claim."}
            records.append(record)
    return records, consumed, gaps


def _import_gap_fill(source: Source) -> tuple[list[dict], set[str]]:
    base = "reports/toy100/gap-fill-20260925/"
    qpath, rpath = base + "qualification-summary.json", base + "results-summary.json"
    if qpath not in source.files:
        return [], set()
    qualification, runs = source.json(qpath), source.json(rpath)
    records = []
    source_manifest = source.json(base + "manifest.json")
    for candidate, row in qualification["candidates"].items():
        tasks = []
        for gate in row["gates"]:
            receipt = {**source.receipt(qpath), "original_evidence": gate}
            snapshot = posixpath.normpath(base + gate["snapshot"])
            raw = dict(gate)
            if snapshot in source.files:
                payload = gzip.decompress(source.read(snapshot))
                if sha256(payload).hexdigest() != gate["artifact_sha256"]:
                    raise ValueError(f"Historical snapshot digest mismatch: {snapshot}")
                decoded = json.loads(payload)
                raw["snapshot_receipt"] = source.receipt(snapshot)
                # Curves, samples, and per-step histories stay in the hashed archive.
                raw["recorded_verdict"] = decoded.get("verdict")
                result = decoded.get("result", {})
                raw["metrics"] = result.get("live", decoded.get("final", {})) if isinstance(result, dict) else {}
                raw["snapshot_provenance"] = {k: decoded[k] for k in (
                    "config", "spec", "initialization_fixture_sha256", "worker_sha256", "torch", "cpu",
                    "environment", "latent_receipt", "response_receipt", "regularizer_receipt", "proof") if k in decoded}
            tasks.append(_task(gate["task"], raw, receipt))
        context = {k: v for k, v in row.items() if k != "gates"}
        context["scope"] = qualification.get("scope")
        context["candidate_source_bundle"] = [s for s in source_manifest["sources"]
                                               if s["saved"].startswith("sources/" + candidate + "/")]
        config = base + "sources/" + candidate + "/config.json"
        if config in source.files:
            context["recipe"] = source.json(config)
            context["config_receipt"] = source.receipt(config)
        record = _record(source, qpath, candidate, tasks, context=context)
        record["mechanism_class"] = "training_formulation"
        records.append(record)
    # Separate attempts preserve floors, negative controls, extension failures, and costs.
    for row in runs["records"]:
        candidate = row.get("candidate")
        task_id = row.get("task", row.get("kind", "unknown"))
        if row.get("kind") in {"hold", "shift", "shift_frozen"}:
            task_id += ":" + row["kind"]
        frozen = row.get("kind") == "shift_frozen" or "frozen" in str(row.get("id", ""))
        receipt = {**source.receipt(rpath), "original_evidence": {k: row[k] for k in (
            "id", "snapshot", "snapshot_sha256", "artifact_sha256", "artifact") if k in row}}
        tasks = [_task(task_id, row, receipt, negative_control=frozen)]
        extension = row.get("extension")
        if extension and row.get("kind") == "hold":
            reached = row.get("gate", {}).get("hold_budget_complete", False)
            status = ("PASS" if extension.get("passed") else "FAIL") if reached else "NOT_RUN"
            tasks.append(_task("mode_hold:extension", {"status": status, "metrics": extension,
                               "reason": None if reached else "Convergence/hold prerequisite was not reached."}, receipt))
        records.append(_record(source, rpath, candidate, tasks,
                               context={"attempt_id": row.get("id"), "floors": row.get("floors"),
                                        "environment": row.get("environment"), "role": "negative_control" if frozen else "candidate"}))
    selection_path = "reports/toy100/current-research-base.json"
    selection = source.json(selection_path)
    selection_tasks = [_task("mode_hold:" + key, selection[key], source.receipt(selection_path))
                       for key in ("ring_hold", "ring_extension", "shift_recovery")]
    records.append(_record(source, selection_path, selection["candidate"], selection_tasks,
                           context=selection, hypothesis="Selected K3P stationary endurance and target-shift recovery."))
    pair_path = base + "rg5-recovery-pair.json"
    pair = source.json(pair_path)
    paired = _task("mode_hold:paired_recovery", {"status": pair["status"], "metrics": pair}, source.receipt(pair_path))
    records.append(_record(source, pair_path, "rg5-a2-floor-0.1-0.1", [paired], context={"scope": pair["scope"]},
                           hypothesis="Matched frozen control distinguishes adaptation from stationary output."))
    return records, {qpath, rpath, selection_path, pair_path}


def _narrative_records(source: Source, consumed: set[str], *, include_unmapped_support=False) -> list[dict]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in source.files:
        if path.startswith("reports/") and path.endswith(".md"):
            grouped[classify(path)["family"]].append(path)
    records = []
    for family, paths in sorted(grouped.items()):
        paths.sort()
        first = next((p for p in paths if p.endswith("/README.md")), paths[0])
        narrative = source.read(first).decode("utf-8", errors="replace")
        headline = next((line.lstrip("# ").strip() for line in narrative.splitlines() if line.strip()), family)
        context = {"family": family, "narrative_sources": [source.receipt(p) for p in paths],
                   "summary_excerpt": narrative[:1400],
                   "structured_sources_imported": sorted(p for p in consumed if classify(p)["family"] == family)}
        if include_unmapped_support:
            context.update(
                narrative_headings=[line.lstrip("# ").strip() for line in narrative.splitlines()
                                    if line.startswith("#")],
                unmapped_structured_sources=[source.receipt(p) for p in sorted(source.files)
                    if p not in consumed and classify(p)["family"] == family
                    and classify(p)["role"] == "structured_evidence"],
                normalization_limit="JSONL summary/oracle schemas are not normalized by this importer; "
                    "linked prose and rows do not establish Forge task verdicts or compatible sampling laws.")
        record = _record(source, first, "context:" + family, [], record_type="family_context",
                         hypothesis=headline, context=context)
        record["next_action"] = "Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison."
        records.append(record)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() == content:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def import_history(root: Path) -> dict[str, Any]:
    """Write reproducible source catalog/cards; never launch training or claim reuse."""
    root = Path(root)
    catalog = inventory(root)
    catalog["coverage"] = validate_inventory(root, catalog)
    sources, cards, gaps, mappings = [], [], [], []
    for name, revision, prefixes, include_unmapped_support in (
        ("baseline", BASELINE_REVISION, SOURCE_ROOTS, False),
        ("pr155", LRFREE_REVISION, (LRFREE_ROOT,), False),
        ("develop-vector-protocols", UPSTREAM_VECTOR_REVISION, UPSTREAM_VECTOR_ROOTS, True),
    ):
        try:
            source = Source(root, revision, prefixes)
        except subprocess.CalledProcessError:
            gaps.append({"kind": "missing_revision", "revision": revision,
                         "reason": "Required frozen source commit is unavailable locally."})
            continue
        sources.append({"source_id": name, "revision": source.revision,
                        "files": source.files, "file_count": len(source.files),
                        "counts_by_suffix": dict(sorted(Counter(PurePosixPath(p).suffix for p in source.files).items()))})
        structured, consumed, problems = _import_structured(source)
        legacy, legacy_sources = _import_gap_fill(source)
        consumed |= legacy_sources
        cards.extend(structured + legacy + _narrative_records(
            source, consumed, include_unmapped_support=include_unmapped_support))
        gaps.extend(problems)
        for path in sorted(source.files):
            classification = classify(path)
            mappings.append({"revision": source.revision, "path": path, **classification,
                             "mapping": "structured" if path in consumed else "narrative" if path.endswith(".md") else "classified_only"})
        unmatched = [p for p in source.files if p.startswith("reports/")
                     and (p.endswith(".json") or include_unmapped_support and p.endswith(".jsonl"))
                     and p not in consumed]
        gaps.append({"kind": "structured_mapping_scope", "revision": source.revision,
                     "reason": "These files are inventoried support/config/evidence; no scientific result is inferred from unrecognized schemas.",
                     "paths": sorted(unmatched)})
    plan = "docs/better-experiment-automation-plan-2026-09-28.md"
    followup_audits = [
        {"path": path, "sha256": sha256((root / path).read_bytes()).hexdigest()}
        for path in ("reports/forge/calibration/followup-source-audit.json",
                     "reports/forge/calibration/local-followup-source-audit.json")
        if (root / path).is_file()
    ]
    for identifier, description in (
        ("dt075-14k", "Review reports grid/staggered 14k failures after stationarity-ladder release; exact package/fixture/results not located in pinned #155."),
        ("ema995-dtracking", "Review reports EMA .995 D-tracking 3/3; exact later live/EMA paired receipts are unbound."),
        ("center-sensitivity", "Review reports sub-0.03 sigma centre-error sensitivity; original scope and receipts remain unbound."),
        ("local-artifacts", "Historical /ml2 checkpoints, samples and runtimes have not been imported or assumed available."),
    ):
        gaps.append({"gap_id": identifier, "kind": "unbound_review_claim" if identifier != "local-artifacts" else "local_artifacts",
                     "source": plan, "reason": description, "gate_status": "NOT_RUN",
                     "followup_source_audits": followup_audits})
    cards.sort(key=lambda card: card["record_id"])
    ids = [card["record_id"] for card in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("Historical record identity collision; refusing to overwrite evidence")
    for card in cards:
        _write_json(root / "reports/forge/records" / (card["record_id"] + ".json"), card)
    # Remove only this importer's superseded materializations, never source evidence.
    for path in (root / "reports/forge/records").glob("history-*.json"):
        if path.stem not in ids:
            old = json.loads(path.read_text())
            if old.get("provenance", {}).get("importer_version") == IMPORTER_VERSION:
                path.unlink()
    catalog["pinned_sources"] = [{k: v for k, v in source.items() if k != "files"} for source in sources]
    _write_json(root / "configs/forge/catalog.json", catalog)
    manifest = {"schema_version": SCHEMA_VERSION, "importer_version": IMPORTER_VERSION,
                "sources": sources, "mappings": mappings, "record_ids": ids,
                "records_digest": _digest(cards), "evidence_scope": "historical",
                "followup_source_audits": followup_audits}
    _write_json(root / "reports/forge/history-sources.json", manifest)
    _write_json(root / "reports/forge/import-gaps.json", {"schema_version": SCHEMA_VERSION,
                "gaps": gaps, "inventory_coverage": catalog["coverage"],
                "historical_import_complete": not any(g["kind"] == "missing_revision" for g in gaps),
                "scientific_normalization_complete": False,
                "note": "Complete file classification is separate from complete experiment normalization and raw-evidence availability."})
    return {"schema_version": SCHEMA_VERSION, "files": len(catalog["files"]),
            "pinned_source_files": sum(source["file_count"] for source in sources),
            "records": len(cards), "scientific_records": sum(c["record_type"] == "scientific" for c in cards),
            "gaps": len(gaps), "coverage": catalog["coverage"],
            "catalog": "configs/forge/catalog.json", "sources": "reports/forge/history-sources.json",
            "import_gaps": "reports/forge/import-gaps.json", "records_digest": manifest["records_digest"]}
