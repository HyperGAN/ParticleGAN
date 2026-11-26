"""Deterministic experiment memory and goal views over immutable evidence.

These reducers never train, enqueue, or alter historical verdicts. Compatible
current receipts are independently regraded; historical imports remain visible
in separate cohorts until an explicit compatibility audit binds them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import re

from .contracts import atomic_json, atomic_text, file_hash, file_lock, identifier, read_json, stable_hash
from .views import diagnostic_evidence_scope

REDUCER_VERSION = "forge-knowledge-v2"
DECLARATION_ONLY_BOARD = "<!-- forge: declaration-only; no measured outcomes -->"


def _json_output(path: Path, value) -> None:
    """Avoid rewriting unaffected reducer materializations."""
    if path.exists() and read_json(path) == value:
        return
    atomic_json(path, value)


def _text_output(path: Path, value: str) -> None:
    if path.exists() and path.read_text() == value:
        return
    atomic_text(path, value)


def leaderboard_path(root: Path, view_id: str) -> Path:
    """Use a published technique inventory instead of a second goal table."""
    current = Path("reports/forge/technique-inventory.md")
    metadata = Path(root) / current.with_suffix(".json")
    if metadata.is_file() and (Path(root) / current).is_file():
        publication = read_json(metadata)
        if (publication.get("publication_scope") == "current_technique_inventory"
                and isinstance(publication.get("view"), str)):
            published_view = publication.get("view")
            if published_view == view_id:
                return current
            declaration = Path(root) / "configs/forge/views" / (view_id + ".json")
            if declaration.is_file():
                view = read_json(declaration)
                published_declaration = Path(root) / "configs/forge/views" / (published_view + ".json")
                goal = (read_json(published_declaration).get("goal", published_view)
                        if published_declaration.is_file() else published_view)
                if view.get("reporting", {}).get("family_totals") is False and view.get("goal") == goal:
                    return current
    return Path("reports/forge/leaderboards") / (view_id + ".md")


_RECORD_FILE_CACHE: dict[tuple[str, int, int, int], tuple[dict, str]] = {}
_ATTEMPT_DIR_CACHE: dict[tuple, tuple[dict, dict | None]] = {}
_NORMALIZE_CACHE: dict[tuple, list[dict]] = {}
_SEARCHABLE_CACHE: dict[int, tuple[dict, frozenset[str], str, frozenset[str], str]] = {}
_WORD_RE = re.compile(r"[a-z0-9_]+")


def _searchable_index(record: dict) -> tuple[frozenset[str], str, frozenset[str], str]:
    """Lazily build exact token-deduplicated substring index for recall()."""
    cached = _SEARCHABLE_CACHE.get(id(record))
    if cached is not None and cached[0] is record:
        return cached[1], cached[2], cached[3], cached[4]
    raw = json.dumps(record, separators=(",", ":")).lower()
    tokens = set(_WORD_RE.findall(raw))
    alpha_toks = frozenset(t for t in tokens if not t.isdigit())
    digit_toks = frozenset(t for t in tokens if t.isdigit())
    alpha_str = " ".join(alpha_toks)
    digit_str = " ".join(digit_toks)
    if len(_SEARCHABLE_CACHE) >= 8192:
        _SEARCHABLE_CACHE.clear()
    _SEARCHABLE_CACHE[id(record)] = (record, alpha_toks, alpha_str, digit_toks, digit_str)
    return alpha_toks, alpha_str, digit_toks, digit_str


def _records(root: Path) -> tuple[list[dict], list[dict]]:
    indexed, indexed_hashes, conflicts = {}, {}, []
    records_dir = os.path.join(os.fspath(root), "reports", "forge", "records")
    if not os.path.isdir(records_dir):
        return [], []
    resolved_dir = os.path.realpath(records_dir)
    entries = []
    with os.scandir(records_dir) as it:
        for entry in it:
            if entry.name.endswith(".json"):
                entries.append((entry.name, entry.path, entry.stat()))
    entries.sort(key=lambda item: item[0])
    for name, path_str, st in entries:
        cache_key = (os.path.join(resolved_dir, name), st.st_ino, st.st_mtime_ns, st.st_size)
        cached = _RECORD_FILE_CACHE.get(cache_key)
        if cached is not None:
            record, record_hash = cached
        else:
            record = read_json(Path(path_str))
            if record.get("schema_version") != 1 or not record.get("record_id"):
                raise ValueError(f"invalid experiment record: {path_str}")
            if not isinstance(record.get("task_results", []), list):
                raise ValueError(f"task_results must be a list: {path_str}")
            record_hash = stable_hash(record)
            _RECORD_FILE_CACHE[cache_key] = (record, record_hash)
        identity = record["record_id"]
        if identity in indexed and indexed_hashes[identity] != record_hash:
            conflicts.append({"record_id": identity, "reason": "Different records have the same record_id.",
                              "path": f"reports/forge/records/{name}"})
        else:
            indexed[identity] = record
            indexed_hashes[identity] = record_hash
    return [indexed[key] for key in sorted(indexed)], conflicts


def _gaps(root: Path) -> dict:
    path = root / "reports/forge/import-gaps.json"
    return read_json(path) if path.exists() else {
        "gaps": [{"kind": "history_not_imported", "reason": "Run forge history before treating memory as complete."}],
        "scientific_normalization_complete": False,
    }


_MEM_INPUT_CACHE: dict[tuple, tuple[tuple, list[dict]]] = {}
_MEM_FIXED_FILES = (
    "reports/forge/technique-inventory.json",
    "reports/forge/technique-inventory.md",
    "reports/forge/policy-family-inventory.json",
    "reports/forge/policy-family-inventory.md",
    "reports/forge/passive-publications.json",
    "reports/forge/api-model-toys/publication.json",
)
_MEM_SCAN_DIRS = (
    "reports/forge/configuration-search",
    "reports/forge/technique-evidence",
    "reports/forge/technique-receipts",
    "reports/forge/family-winner-round1",
)


def _pub_fast_signature(root_os: str, extra_paths: tuple[str, ...]) -> tuple:
    dir_entries: list[tuple] = []
    for rel_dir in _MEM_SCAN_DIRS:
        dpath = os.path.join(root_os, rel_dir)
        if not os.path.isdir(dpath):
            dir_entries.append((rel_dir, None))
            continue
        items: list[tuple[str, int, int, int]] = []
        for walk_root, _, files in os.walk(dpath):
            for fname in files:
                if fname.endswith(".json"):
                    fpath = os.path.join(walk_root, fname)
                    try:
                        st = os.stat(fpath)
                    except OSError:
                        continue
                    items.append((fpath, st.st_ino, st.st_mtime_ns, st.st_size))
        items.sort()
        dir_entries.append((rel_dir, tuple(items)))
    fixed: list[tuple] = []
    for rel_file in _MEM_FIXED_FILES:
        fpath = os.path.join(root_os, rel_file)
        try:
            st = os.stat(fpath)
            fixed.append((rel_file, st.st_ino, st.st_mtime_ns, st.st_size))
        except OSError:
            fixed.append((rel_file, None))
    extras: list[tuple] = []
    for fpath in extra_paths:
        try:
            st = os.stat(fpath)
            extras.append((fpath, st.st_ino, st.st_mtime_ns, st.st_size))
        except OSError:
            extras.append((fpath, None))
    return (tuple(dir_entries), tuple(fixed), tuple(extras))


import time as _time

_INPUT_FILE_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}
_INPUT_FILE_HASH_CACHE_LIMIT = 32768
_FRESH_MTIME_GUARD_NS = 50_000_000


def _cached_file_hash(path_str: str, now_ns: int) -> str:
    try:
        st = os.stat(path_str)
    except OSError:
        return file_hash(Path(path_str))
    if st.st_mtime_ns >= now_ns - _FRESH_MTIME_GUARD_NS:
        return file_hash(Path(path_str))
    key = (path_str, st.st_ino, st.st_mtime_ns, st.st_size)
    digest = _INPUT_FILE_HASH_CACHE.get(key)
    if digest is None:
        digest = file_hash(Path(path_str))
        if len(_INPUT_FILE_HASH_CACHE) >= _INPUT_FILE_HASH_CACHE_LIMIT:
            _INPUT_FILE_HASH_CACHE.clear()
        _INPUT_FILE_HASH_CACHE[key] = digest
    return digest


def _cached_normalize(root: Path) -> list[dict]:
    from .publication_memory import input_paths, load_passive_publications, normalize
    root_os = os.path.realpath(os.fspath(root))
    cache_slot = (id(normalize), id(input_paths), id(load_passive_publications), root_os)
    cached = _MEM_INPUT_CACHE.get(cache_slot)
    if cached is not None:
        extra_paths, prev_sig, pub_records = cached
        if _pub_fast_signature(root_os, extra_paths) == prev_sig:
            return pub_records
    scan_prefixes = tuple(os.path.join(root_os, d) + os.sep for d in _MEM_SCAN_DIRS)
    fixed_set = {os.path.join(root_os, f) for f in _MEM_FIXED_FILES}
    extra_paths = tuple(sorted({
        os.path.realpath(os.fspath(p))
        for p in input_paths(root)
        if os.path.realpath(os.fspath(p)) not in fixed_set
        and not os.path.realpath(os.fspath(p)).startswith(scan_prefixes)
    }))
    sig = _pub_fast_signature(root_os, extra_paths)
    pub_records = normalize(root)
    if len(_MEM_INPUT_CACHE) >= 32:
        _MEM_INPUT_CACHE.clear()
    _MEM_INPUT_CACHE[cache_slot] = (extra_paths, sig, pub_records)
    return pub_records


def _memory_records(root: Path) -> tuple[list[dict], list[dict]]:
    """Publication summaries enter recall only, never the qualification reducer."""
    records, conflicts = _records(root)
    pub_records = _cached_normalize(root)
    return sorted([*records, *pub_records], key=lambda record: record["record_id"]), conflicts


def _input_hashes(root: Path, *, summaries_only: bool = False) -> dict:
    """Hash the inputs consumed by this compilation mode.

    Compact publication recall reads records and publication bindings, while
    preserving existing qualifications. Hydrated original attempts are inputs
    only to the ordinary receipt reducer, so their local availability cannot
    invalidate a summaries-only publication in a clean checkout.
    """
    from .publication_memory import input_paths
    root_os = os.fspath(root)
    prefix = root_os if root_os.endswith(os.sep) else root_os + os.sep
    prefix_len = len(prefix)
    path_strs: set[str] = {os.fspath(p) for p in input_paths(root)}
    directories = ["reports/forge/records", "configs/forge",
                   "reports/forge/calibration-lanes", "reports/forge/promotions"]
    if not summaries_only:
        directories.append("reports/forge/attempts")
    for directory in directories:
        base = os.path.join(root_os, directory)
        if not os.path.isdir(base):
            continue
        for walk_root, _, files in os.walk(base):
            for fname in files:
                if fname.endswith(".json"):
                    path_strs.add(os.path.join(walk_root, fname))
    # The history catalog tracks Git blobs of unrelated engineering source.
    # Its drift is reported separately and cannot make scientific recall stale.
    path_strs.discard(os.path.join(root_os, "configs/forge/catalog.json"))
    for name in ("import-gaps.json", "history-sources.json"):
        fpath = os.path.join(root_os, "reports/forge", name)
        if os.path.isfile(fpath):
            path_strs.add(fpath)
    now_ns = _time.time_ns()
    result = {}
    for p_str in sorted(path_strs):
        rel = p_str[prefix_len:] if p_str.startswith(prefix) else str(Path(p_str).relative_to(root))
        result[rel] = _cached_file_hash(p_str, now_ns)
    return result


def _reducer_hashes() -> dict:
    from . import publication_memory
    return {__name__: file_hash(Path(__file__)),
            publication_memory.__name__: file_hash(Path(publication_memory.__file__))}


def freshness(root: Path) -> dict:
    """Read-only check suitable for CI and agents; no training or hydration."""
    from .publication_memory import OUTPUT
    root = Path(root)
    path = root / "reports/forge/compilation.json"
    manifest = read_json(path) if path.is_file() else {}
    summaries_only = manifest.get("qualification_refresh", {}).get("mode") == "preserved_published_evidence"
    current = _input_hashes(root, summaries_only=summaries_only)
    previous = manifest.get("input_hashes", {})
    added, removed = sorted(current.keys() - previous.keys()), sorted(previous.keys() - current.keys())
    changed = sorted(name for name in current.keys() & previous.keys() if current[name] != previous[name])
    projected = _cached_normalize(root)
    output = root / OUTPUT
    materialization_current = (output.is_file() and read_json(output).get("records") == projected)
    record_count = len(_records(root)[0]) + len(projected)
    view_count = len(list((root / "configs/forge/views").glob("*.json")))
    reducer_current = (manifest.get("reducer_version") == REDUCER_VERSION
                       and manifest.get("reducer_hashes") == _reducer_hashes())
    counts_current = manifest.get("record_count") == record_count and manifest.get("view_count") == view_count
    memory = root / "reports/forge/EXPERIMENT_MEMORY.md"
    memory_current = (memory.is_file() and manifest.get("output_hashes", {}).get("reports/forge/EXPERIMENT_MEMORY.md") == file_hash(memory))
    fresh = bool(manifest) and not (added or removed or changed) and reducer_current and counts_current and materialization_current and memory_current
    try:
        from .history import validate_inventory
        coverage = validate_inventory(root)
    except Exception as exc:
        coverage = {"valid": False, "reason": str(exc)}
    return {"fresh": fresh, "status": "CURRENT" if fresh else "STALE", "added_inputs": added,
            "removed_inputs": removed, "changed_inputs": changed, "reducer_current": reducer_current,
            "publication_materialization_current": materialization_current,
            "memory_current": memory_current,
            "record_count": record_count, "view_count": view_count,
            "publication_record_count": len(projected),
            "inventory_coverage": coverage,
            "coverage_scope": "Registered compact publication schemas and normalized records; original artifact availability is separate.",
            "qualification_refresh": manifest.get("qualification_refresh", {"mode": "unrecorded"}),
            "next_action": None if fresh else "Run python -m experiments.forge compile --summaries-only; inspect missing publication bindings before proposing work."}


def _cost(rows: list[dict]) -> dict:
    values = []
    unavailable = 0
    counted_attempts = set()
    for row in rows:
        attempt_id = row.get("_attempt_id")
        if attempt_id and attempt_id in counted_attempts:
            continue
        if attempt_id:
            counted_attempts.add(attempt_id)
        cost = row.get("cost", {})
        seconds = cost.get("wall_seconds", cost.get("seconds")) if isinstance(cost, dict) else None
        if type(seconds) in (int, float) and seconds >= 0:
            values.append(seconds)
        else:
            unavailable += 1
    return {"wall_seconds": sum(values) if values else None,
            "measured_tasks": len(values), "unmeasured_tasks": unavailable,
            "flops": {"kind": "unavailable", "value": None}}


def _attempts(root: Path) -> tuple[list[dict], list[dict]]:
    attempts, issues = [], []
    root_key = str(Path(root).resolve())
    for directory in sorted((root / "reports/forge/attempts").glob("*")):
        if not directory.is_dir():
            continue
        request_path, result_path = directory / "request.json", directory / "result.json"
        if not request_path.exists() or not result_path.exists():
            issues.append({"attempt_id": directory.name, "reason": "Attempt lacks a durable request/result pair."})
            continue
        evidence_path = directory / "evidence.json"
        req_st, res_st = request_path.stat(), result_path.stat()
        ev_st = evidence_path.stat() if evidence_path.exists() else None
        cache_key = (
            str(directory.resolve()),
            root_key,
            req_st.st_ino,
            req_st.st_mtime_ns,
            req_st.st_size,
            res_st.st_ino,
            res_st.st_mtime_ns,
            res_st.st_size,
            ev_st.st_ino if ev_st else None,
            ev_st.st_mtime_ns if ev_st else None,
            ev_st.st_size if ev_st else None,
        )
        cached = _ATTEMPT_DIR_CACHE.get(cache_key)
        if cached is not None:
            cached_attempt, cached_issue = cached
            attempts.append({**cached_attempt, "task_results": list(cached_attempt["task_results"])})
            if cached_issue is not None:
                issues.append(dict(cached_issue))
            continue
        resolved, result = read_json(request_path), read_json(result_path)
        request = resolved.get("request", resolved)
        certificate = read_json(evidence_path) if ev_st is not None else {}
        reason = None
        result_hash = stable_hash(result)
        if certificate.get("result_hash") != result_hash:
            reason = "Missing or mismatched durable result hash."
        elif certificate.get("source") != request.get("source"):
            reason = "Source certificate does not match the immutable request."
        elif certificate.get("runtime") != request.get("runtime"):
            reason = "Runtime certificate does not match the immutable request."
        elif result.get("candidate_revision") != request.get("candidate_revision"):
            reason = "Result and request candidate revisions differ."
        elif result.get("attempt_id") != directory.name:
            reason = "Result attempt id does not match its durable directory."
        elif result.get("retry_of") != resolved.get("retry_of"):
            reason = "Retry declaration differs between the resolved request and result."
        jobs = request.get("jobs", [])
        keys = {task_id: job["compatibility_key"] for job in jobs
                for task_id in job.get("task_ids", [job["task_id"]])}
        rows = []
        for row in result.get("task_results", []):
            if row.get("compatibility_key") != keys.get(row.get("task_id")) or row.get("task_id") not in keys:
                reason = "Task result compatibility key does not match its immutable request."
            rows.append({**row, "_attempt_id": directory.name})
        issue_entry = None
        if reason:
            issue_entry = {"attempt_id": directory.name, "reason": reason}
            issues.append(issue_entry)
            rows = [{**row, "gate_status": "INVALID", "evidence": {}, "error": reason,
                     "reason": reason} for row in rows]
        attempt_entry = {"attempt_id": directory.name, "request": request, "task_results": rows,
                         "request_id": request.get("request_id", resolved.get("request_id")),
                         "valid_receipt": reason is None, "receipt_error": reason,
                         "result_hash": result_hash,
                         "retry_of": result.get("retry_of"),
                         "attempt_status": result.get("raw", {}).get("attempt_status"),
                         "source": str(result_path.relative_to(root))}
        _ATTEMPT_DIR_CACHE[cache_key] = ({**attempt_entry, "task_results": list(rows)}, dict(issue_entry) if issue_entry else None)
        attempts.append(attempt_entry)
    issues.extend(_resolve_retries(attempts))
    return attempts, issues


def _resolve_retries(attempts):
    """Only certified repairs supersede infrastructure outcomes; keep every cost."""
    indexed = {attempt["attempt_id"]: attempt for attempt in attempts}
    successors = defaultdict(list)
    errors = {}
    for attempt in attempts:
        link = attempt.get("retry_of")
        if not link:
            continue
        previous = indexed.get(link.get("attempt_id")) if isinstance(link, dict) else None
        reason = None
        if (not previous or not attempt["valid_receipt"] or not previous["valid_receipt"]
                or link.get("result_hash") != previous["result_hash"]
                or not isinstance(link.get("reason"), str) or not link["reason"].strip()
                or not link.get("authorized_at")):
            reason = "Retry lacks a valid prior receipt and explicit repair authorization."
        elif (previous.get("attempt_status") not in {"error", "timeout", "cancelled"}
                or not previous["task_results"]
                or any(row.get("gate_status") != "INCOMPLETE" for row in previous["task_results"])):
            reason = "Retry cannot supersede scientific, applicability, or invalid evidence."
        else:
            old, new = previous["request"], attempt["request"]
            keys = lambda a: {(row["task_id"], row["compatibility_key"]) for row in a["task_results"]}
            if (old["candidate_revision"] != new["candidate_revision"]
                    or old.get("source", {}).get("digest") != new.get("source", {}).get("digest")
                    or _runtime_cohort(old) != _runtime_cohort(new)
                    or old.get("protocol") != new.get("protocol") or keys(previous) != keys(attempt)):
                reason = "Retry changes the scientific identity of the prior attempt."
        if reason:
            errors[attempt["attempt_id"]] = reason
        elif previous:
            successors[previous["attempt_id"]].append(attempt["attempt_id"])
    for children in successors.values():
        if len(children) > 1:
            errors.update({child: "Conflicting retry branches require investigation." for child in children})
    # A valid queue cannot produce cycles; imported receipts must prove this too.
    for identity in indexed:
        chain, cursor = set(), identity
        while cursor in indexed and isinstance(indexed[cursor].get("retry_of"), dict):
            if cursor in chain:
                errors.update({name: "Cyclic retry lineage is invalid." for name in chain})
                break
            chain.add(cursor)
            cursor = indexed[cursor]["retry_of"].get("attempt_id")
    for identity, reason in errors.items():
        attempt = indexed[identity]
        attempt.update(valid_receipt=False, receipt_error=reason)
        attempt["task_results"] = [{**row, "gate_status": "INVALID", "evidence": {}, "reason": reason}
                                   for row in attempt["task_results"]]
    for parent, children in successors.items():
        child = children[0]
        if len(children) == 1 and parent not in errors and child not in errors:
            indexed[parent]["superseded_by"] = child
    return [{"attempt_id": identity, "reason": reason} for identity, reason in sorted(errors.items())]


def _retry_history(attempts):
    return [{"attempt_id": a["attempt_id"], "superseded_by": a["superseded_by"],
             "task_results": a["task_results"]} for a in attempts if a.get("superseded_by")]


def _historical_row(record: dict) -> dict:
    tasks = record.get("task_results", [])
    counts = dict(sorted(Counter(row.get("gate_status", "NOT_RUN") for row in tasks).items()))
    controls = [row for row in tasks if row.get("role") == "negative_control"]
    return {"candidate_id": record.get("candidate_id"), "candidate_revision": record.get("candidate_revision"),
            "record_id": record["record_id"], "evidence_scope": "historical",
            "cohort": stable_hash({"revision": record.get("candidate_revision"),
                                   "source": record.get("source"), "claim": record.get("claim_contract")}),
            "qualified_tier": None, "status": "HISTORICAL", "rank": None,
            "counts": counts, "recorded_tasks": len(tasks), "negative_controls": len(controls),
            "cost": _cost(tasks), "mechanism_class": record.get("mechanism_class", "unknown"),
            "prior": record.get("prior"), "claim_contract": record.get("claim_contract"),
            "task_results": [{key: row[key] for key in ("task_id", "gate_status", "raw_status", "metrics", "cost", "reason", "role")
                              if key in row} for row in tasks], "source": record.get("source"),
            "conclusion": record.get("conclusion"), "next_action": record.get("next_action"),
            "note": "Historical recorded outcomes; overlapping summaries/attempts are not independent trials or a combined cost total."}


def _current_request(root: Path, idea_id: str, view_id: str, execution_backend="cuda", cuda_model=None) -> dict:
    from .planning import load_idea, resolve_idea
    from .resolution_scope import active
    cache = active(root)
    idea = cache.get_idea_ref(idea_id) if cache is not None else load_idea(root, idea_id)
    # Reconstruct the declared question. The resolver retains the full view's
    # tasks regardless of this authorization cap, so qualification is unchanged.
    through_tier = idea.get("decision_contract", {}).get("scope", {}).get("through_tier", 3)
    return resolve_idea(root, idea_id, view_id=view_id, through_tier=through_tier, freeze_source=False,
                        execution_backend=execution_backend, cuda_model=cuda_model)


def _runtime_cohort(request):
    profiles = request.get("compute_profiles") or {
        job["task_id"]: job.get("science", {}).get("compute") for job in request.get("jobs", [])
        if job.get("science", {}).get("compute")}
    return {"execution_backend": request.get("execution_backend", "unrecorded"),
            "runtime": request.get("runtime"), "compute_profiles": profiles}


def _queue_states(root, attempts):
    """Atomic queue snapshots are read without creating state or lock files."""
    from .__main__ import queue_location
    locations = {queue_location(root), root / "runs/forge"}
    locations.update(Path(a["request"]["queue_root"]) for a in attempts if a["request"].get("queue_root"))
    return {str(path.resolve()): read_json(path / "queue/state.json")
            for path in sorted(locations) if (path / "queue/state.json").exists()}


def _lifecycle(request, selected, records, queue_states):
    from .lifecycle import concluded_readout, overlay
    # Compatible aliases can share science; another idea's later repair does not
    # reopen this idea's immutable administrative disposition or readout.
    selected = [a for a in selected if a["request"]["candidate"]["id"] == request["candidate"]["id"]]
    ids = {a["attempt_id"] for a in selected}
    candidate_id, revision = request["candidate"]["id"], request["candidate_revision"]
    relevant = []
    for location, state in queue_states.items():
        for request_id, entry in state.get("submissions", {}).items():
            original = entry["request"]
            if (original.get("candidate", {}).get("id") == candidate_id
                    and original.get("candidate_revision") == revision
                    and _runtime_cohort(original) == _runtime_cohort(request)):
                relevant.append({"queue_root": location, "request_id": request_id,
                                 "status": entry["status"], "lifecycle": entry.get("lifecycle")})
    active = [e for e in relevant if e["status"] in {"queued", "running", "paused"}]
    # Compare the exact attempted cohort covered by a readout. A subsequent retry
    # adds an id and invalidates coverage even though candidate revision is fixed.
    has_readout = concluded_readout(request, selected, records) is not None
    lifecycle = ("running" if any(e["status"] == "running" for e in active) else "ready") if active else (
        "concluded" if has_readout else "awaiting_readout" if selected else "proposed")
    return overlay(request, selected, records, queue_states,
                   {"lifecycle": lifecycle, "pending_readout": bool(selected) and not has_readout and not active,
                    "queue_submissions": relevant, "readout_covers_attempt_ids": sorted(ids) if has_readout else []})


def _pinned_row(group, records, queue_states):
    request = group[0]["request"]
    tasks = [row for attempt in group for row in attempt["task_results"]]
    active = [row for attempt in group if not attempt.get("superseded_by") for row in attempt["task_results"]]
    counts = dict(sorted(Counter(row.get("gate_status", "NOT_RUN") for row in active).items()))
    status = next((s for s in ("INVALID", "FAIL", "BLOCKED", "INCOMPLETE", "NOT_RUN", "PASS") if counts.get(s)), "NOT_RUN")
    diagnostic = diagnostic_evidence_scope(request)
    return {"candidate_id": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
            "evidence_scope": diagnostic or "pinned",
            "calibration_lane": request.get("calibration_lane"),
            "cohort": stable_hash({"revision": request["candidate_revision"],
                "runtime": _runtime_cohort(request), "protocol": request.get("protocol"),
                "tasks": {k: v for k, v in request.get("tasks", {}).items()}}),
            "runtime_cohort": _runtime_cohort(request), "qualified_tier": 0 if diagnostic == "research_diagnostic" else None,
            "status": "DIAGNOSTIC" if diagnostic else status, "counts": counts, "cost": _cost(tasks),
            "attempt_ids": sorted(a["attempt_id"] for a in group),
            "retry_history": _retry_history(group),
            "task_results": [{k: r[k] for k in ("task_id", "gate_status", "raw_status", "cost", "reason", "metrics") if k in r}
                             for r in tasks],
            # The immutable request owns the full file manifest; repeating it in
            # every goal board grows reports without adding comparison evidence.
            "source": {**{k: request.get("source", {}).get(k) for k in ("digest", "origin_commit")},
                       "path": str(Path(group[0]["source"]).with_name("request.json"))},
            "recorded_view": request.get("view"),
            "qualification_reuse": False, "grading": "Recorded verdicts under pinned source; never regraded with a changed live evaluator.",
            **({"qualification_input": False, "eligible": False} if diagnostic == "research_diagnostic" else {}),
            **_lifecycle(request, group, records, queue_states),
            "next_action": "Publish the stopped cohort readout, or declare a new compatible revision before further qualification."}


def board(root: Path, view_id: str, *, include_bindings=False) -> dict:
    """Read-only view; current hardware cohorts and pinned observations stay separate."""
    from . import views
    from .planning import declaration_paths, resolution_cache
    root = Path(root)
    with resolution_cache(root) as cache:
        view = views.load_view(root, view_id)
        records, conflicts = _records(root)
        attempts, receipt_issues = _attempts(root)
        states = _queue_states(root, attempts)
        conflicts.extend(receipt_issues)
        current, matched_attempts, seen_cohorts = [], set(), set()
        attempt_configs: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
        attempts_by_cohort: dict[tuple, list[tuple[dict, dict[str, str]]]] = defaultdict(list)
        for attempt in attempts:
            original = attempt["request"]
            cid = original.get("candidate", {}).get("id")
            backend = original.get("execution_backend")
            if not diagnostic_evidence_scope(original):
                if cid and backend in {"cpu", "cuda"}:
                    model = original.get("compute_profiles", {}).get("cuda", {}).get("model") if backend == "cuda" else None
                    attempt_configs[cid].add((backend, model))
                c_key = (
                    original.get("candidate_revision"),
                    original.get("source", {}).get("digest"),
                    stable_hash(_runtime_cohort(original)),
                )
                old_fps = {tid: views.task_fingerprint(t) for tid, t in original.get("tasks", {}).items() if t}
                attempts_by_cohort[c_key].append((attempt, old_fps))
        view_task_fps: dict[str, str] = {}
        qualify_cache: dict[tuple, dict] = {}
        for path in declaration_paths(root):
            try:
                idea = cache.get_idea_ref(path.stem)
            except Exception:
                idea = read_json(path)
            idea_id = idea.get("id", path.stem)
            configurations = {("cuda", None), ("cpu", None)} | attempt_configs.get(idea_id, set())
            for backend, model in sorted(configurations, key=lambda item: (item[0], item[1] or "")):
                try:
                    request = _current_request(root, idea_id, view_id, backend, model)
                except (ValueError, FileNotFoundError, ImportError) as exc:
                    current.append({"candidate_id": idea_id, "candidate_revision": None, "qualified_tier": 0,
                                    "runtime_cohort": {"execution_backend": backend, "cuda_model": model},
                                    "status": "BLOCKED", "lifecycle": "proposed", "evidence_scope": "current",
                                    "blockers": [{"reason": str(exc)}], "counts": {}, "cost": _cost([]),
                                    "next_action": "Resolve the candidate declaration/capability blocker."})
                    continue
                expected = {task_id: job["compatibility_key"] for job in request["jobs"]
                            for task_id in job.get("task_ids", [job["task_id"]])}
                cohort = stable_hash({"candidate_revision": request["candidate_revision"], "keys": expected})
                if cohort in seen_cohorts:
                    continue
                seen_cohorts.add(cohort)
                selected, evidence_attempts, invalid_tasks = [], [], set()
                cohort_key = (
                    request["candidate_revision"],
                    request.get("source", {}).get("digest"),
                    stable_hash(_runtime_cohort(request)),
                )
                for attempt, old_fps in attempts_by_cohort.get(cohort_key, ()):
                    used = False
                    for row in attempt["task_results"]:
                        tid = row["task_id"]
                        if expected.get(tid) != row.get("compatibility_key"):
                            continue
                        if tid not in view_task_fps:
                            view_task_fps[tid] = views.task_fingerprint(request["tasks"][tid])
                        if old_fps.get(tid) != view_task_fps[tid]:
                            continue
                        selected.append(row)
                        if not attempt["valid_receipt"]:
                            invalid_tasks.add(tid)
                        used = True
                    if used:
                        evidence_attempts.append(attempt)
                        matched_attempts.add(attempt["attempt_id"])
                superseded = {a["attempt_id"] for a in evidence_attempts if a.get("superseded_by")}
                safe = [{**row, "applicability": {"status": "unknown", "reason": "Durable receipt validation failed."}}
                        if row["task_id"] in invalid_tasks else row for row in selected
                        if row["_attempt_id"] not in superseded]
                covered = {row["task_id"] for row in safe}
                for task_id, task in request["tasks"].items():
                    if task_id not in covered and task.get("preflight_blockers"):
                        safe.append({"task_id": task_id, "gate_status": "BLOCKED",
                                     "reason": "; ".join(task["preflight_blockers"])})
                if not selected:
                    q_key = (
                        tuple((r["task_id"], r["reason"]) for r in safe),
                        tuple(views._candidate_eligibility(view, request["candidate"])),
                    )
                    if q_key not in qualify_cache:
                        qualify_cache[q_key] = views.qualify(view, request["tasks"], safe, candidate=request["candidate"])
                    qualified = deepcopy(qualify_cache[q_key]) if request.get("preflight_blockers") else dict(qualify_cache[q_key])
                else:
                    qualified = views.qualify(view, request["tasks"], safe, candidate=request["candidate"])
                if request.get("preflight_blockers"):
                    qualified.update(status="BLOCKED", eligible=False, qualified_tier=0)
                    qualified["blockers"] = [{"task_id": None, "status": "BLOCKED", "reasons": request["preflight_blockers"]},
                                             *qualified["blockers"]]
                lifecycle = _lifecycle(request, evidence_attempts, records, states)
                row = {"candidate_id": idea_id, "candidate_revision": request["candidate_revision"],
                                "evidence_scope": "research_diagnostic" if view.get("evidence_scope") == "research_diagnostic" else "current", "cohort": cohort,
                                "runtime_cohort": _runtime_cohort(request),
                                "source_digest": request.get("source", {}).get("digest"),
                                "status": qualified["status"], "qualified_tier": qualified["qualified_tier"],
                                "qualification": qualified,
                                "counts": dict(sorted(Counter(qualified["task_statuses"].values()).items())),
                                "cost": _cost(selected), "attempt_ids": sorted(a["attempt_id"] for a in evidence_attempts),
                                "retry_history": _retry_history(evidence_attempts),
                                **lifecycle,
                                "mechanism_class": idea.get("mechanism_class"), "prior": request["candidate"].get("prior"),
                                "claim_contract": request["candidate"].get("claim_contract"),
                                "preflight_blockers": request.get("preflight_blockers", []),
                                "next_action": _next_action(qualified["status"], lifecycle["pending_readout"])}
                if qualified.get("held_total"):
                    # Held tasks are graded and shown, never counted; say so beside the status.
                    row["held"] = {"total": qualified["held_total"], "passed": qualified["held_passed"]}
                if include_bindings:
                    from .technique_board import request_bindings
                    row["scientific_bindings"] = request_bindings(request)
                current.append(row)
        current.sort(key=lambda row: (-row["qualified_tier"], row["candidate_id"], row.get("cohort", "")))
        historical = [_historical_row(record) for record in records
                      if record.get("evidence_scope") == "historical" and record.get("record_type") != "family_context"]
        historical.sort(key=lambda row: (str(row["candidate_id"]), row["record_id"]))
        archived_groups = defaultdict(list)
        for attempt in attempts:
            if attempt["attempt_id"] not in matched_attempts:
                request = attempt["request"]
                identity = {k: request.get(k) for k in ("candidate_revision", "source", "protocol", "tasks")}
                identity["runtime"] = _runtime_cohort(request)
                identity["evidence_use"] = diagnostic_evidence_scope(request) or "qualification"
                archived_groups[stable_hash(identity)].append(attempt)
        archived_rows = [_pinned_row(group, records, states) for _, group in sorted(archived_groups.items())]
        pinned = [row for row in archived_rows if row["evidence_scope"] == "pinned"]
        diagnostic = [row for row in archived_rows if row["evidence_scope"] == "calibration_diagnostic"]
        research = [row for row in archived_rows if row["evidence_scope"] == "research_diagnostic"]
        archived = [{"attempt_id": a["attempt_id"], "candidate_id": a["request"].get("candidate", {}).get("id"),
                     "candidate_revision": a["request"].get("candidate_revision"), "source": a["source"],
                     "cost": _cost(a["task_results"]), "reason": "Bounded diagnostics cannot qualify a candidate."
                     if diagnostic_evidence_scope(a["request"]) else "Pinned source/task/protocol/runtime differs from the current cohort."}
                    for a in attempts if a["attempt_id"] not in matched_attempts]
        result = {"schema_version": 1, "view": view_id, "view_revision": view["revision"],
                "policy_fingerprint": views.view_fingerprint(view), "calibration": view.get("calibration"),
                "rows": current + pinned + diagnostic + research + historical, "current_rows": current, "historical_rows": historical,
                "calibration_rows": diagnostic,
                "research_rows": research,
                "pinned_rows": pinned, "archived_attempts": archived, "conflicts": conflicts, "import_gaps": _gaps(root),
                "ranking_note": "Ordered by attained tier, then candidate name/cohort; raw metrics and cost remain separate, with no aggregate metric ranking. CPU/CUDA and GPU models remain separate; pinned and historical verdicts are unranked and never automatically reused."}
        from .board_filters import annotate_rows
        return annotate_rows(root, result, attempts=attempts, records=records)


def in_flight_work(root: Path, query: str = "", goal: str | None = None) -> list[dict]:
    """Surface active queue requests and open studies without hydrating attempts."""
    root = Path(root)
    records, _ = _records(root)
    concluded_studies = {
        record.get("study_id")
        for record in records
        if record.get("lifecycle") in {"concluded", "abandoned", "superseded"}
        and record.get("study_id")
    }
    words = re.findall(r"[a-z0-9_]+", query.lower())
    items = []
    states = _queue_states(root, [])
    for location, state in states.items():
        for request_id, entry in sorted(state.get("submissions", {}).items()):
            if entry.get("status") not in {"queued", "running", "paused"}:
                continue
            req = entry.get("request", {})
            view_id = req.get("view", {}).get("id") or req.get("view", {}).get("goal")
            if goal and view_id and view_id != goal:
                continue
            item = {
                "kind": "queue_submission",
                "request_id": request_id,
                "candidate_id": req.get("candidate", {}).get("id"),
                "candidate_revision": req.get("candidate_revision"),
                "study_id": req.get("study", {}).get("id") if isinstance(req.get("study"), dict) else None,
                "view": view_id,
                "status": entry.get("status"),
                "lifecycle": entry.get("lifecycle"),
                "queue_root": location,
            }
            searchable = json.dumps(item, sort_keys=True).lower()
            if words and not any(word in searchable for word in words):
                continue
            items.append(item)
    for path in sorted((root / "configs/forge/studies").glob("*.json")):
        study = read_json(path)
        study_id = study.get("id", path.stem)
        if study_id in concluded_studies or study.get("status") not in {"ready", "draft", "running"}:
            continue
        view_id = study.get("scope", {}).get("view", study.get("view"))
        if goal and view_id and view_id != goal:
            continue
        item = {
            "kind": "open_study",
            "study_id": study_id,
            "candidate_id": study.get("candidate"),
            "control_id": study.get("control"),
            "view": view_id,
            "status": study.get("status"),
            "hypothesis": study.get("hypothesis"),
            "source": str(path.relative_to(root)),
        }
        searchable = json.dumps(item, sort_keys=True).lower()
        if words and not any(word in searchable for word in words):
            continue
        items.append(item)
    return items


def _recall_source(source, *, full_provenance: bool):
    """Keep source identity readable without copying a complete file manifest."""
    if full_provenance or not isinstance(source, dict) or "files" not in source:
        return deepcopy(source)
    compact = {key: deepcopy(value) for key, value in source.items() if key != "files"}
    compact["file_count"] = len(source["files"])
    return compact


def recall(root: Path, query: str = "", goal: str | None = None, *, include_in_flight: bool = False,
           full_provenance: bool = False) -> list[dict]:
    """Search normalized records; missing goal tags remain visible as unknown."""
    records, _ = _memory_records(Path(root))
    query_norm = query.strip().lower()
    words = _WORD_RE.findall(query.lower())
    word_flags = [(word, word.isdigit()) for word in words]
    matches = []
    for record in records:
        declared_goal = record.get("goal")
        if goal and declared_goal and declared_goal != goal:
            continue
        if word_flags:
            alpha_toks, alpha_str, digit_toks, digit_str = _searchable_index(record)
            score = 0
            for word, is_digit in word_flags:
                if is_digit:
                    if word in digit_toks or word in alpha_str or word in digit_str:
                        score += 1
                else:
                    if word in alpha_toks or word in alpha_str:
                        score += 1
        else:
            score = 0
        identities = [str(record.get(key) or "").lower() for key in
                      ("record_id", "candidate_id", "candidate_revision", "configuration_id", "study_id")]
        # A literal complete id outranks broad token overlap (hyphens otherwise
        # split family ids into common, often misleading words).
        exact = bool(query_norm) and query_norm in identities
        if exact:
            score += len(words) + 1
        if words and score == 0:
            continue
        matches.append({"record_id": record["record_id"], "candidate_id": record.get("candidate_id"),
                        "candidate_revision": record.get("candidate_revision"), "evidence_scope": record.get("evidence_scope"),
                        "hypothesis": record.get("hypothesis"), "conclusion": record.get("conclusion"),
                        "next_action": record.get("next_action"),
                        "source": _recall_source(record.get("source"), full_provenance=full_provenance),
                        "cost": record.get("recorded_cost") or _cost(record.get("task_results", [])), "match_score": score,
                        "exact_identity_match": exact, "study_id": record.get("study_id"),
                        "qualification_input": record.get("qualification_input", record.get("evidence_scope") == "current"),
                        "task_results": [{key: row.get(key) for key in
                                          ("task_id", "gate_status", "original_gate", "metrics", "failed_bounds", "reason", "receipt")}
                                         for row in record.get("task_results", [])
                                         if record.get("evidence_scope") == "published_summary"
                                         and row.get("gate_status") in {"FAIL", "INVALID", "BLOCKED", "INCOMPLETE"}],
                        "goal_applicability": "unknown" if not declared_goal else declared_goal})
    ordered = sorted(matches, key=lambda row: (-row["match_score"], row["record_id"]))
    if include_in_flight:
        for item in in_flight_work(root, query=query, goal=goal):
            ordered.append({
                "record_id": f"in_flight:{item.get('study_id') or item.get('request_id')}",
                "candidate_id": item.get("candidate_id"),
                "candidate_revision": item.get("candidate_revision"),
                "evidence_scope": "in_flight",
                "hypothesis": item.get("hypothesis"),
                "conclusion": None,
                "next_action": f"In-flight ({item['kind']}: {item.get('status')})",
                "source": _recall_source(item.get("source") or item.get("queue_root"),
                                         full_provenance=full_provenance),
                "cost": _cost([]),
                "match_score": 0,
                "exact_identity_match": False,
                "study_id": item.get("study_id"),
                "qualification_input": False,
                "task_results": [],
                "goal_applicability": item.get("view") or "unknown",
            })
    return ordered


def _link(source) -> str:
    if isinstance(source, dict):
        return source.get("url") or "../../" + source.get("path", "")
    return "../../" + str(source or "")


def _evidence_navigation(record: dict) -> str:
    """Use durable provenance for raw receipts, independent of local hydration."""
    source = record.get("source")
    path = source.get("path", "") if isinstance(source, dict) else str(source or "")
    record_link = f"records/{record['record_id']}.json"
    if path.startswith("reports/forge/attempts/") and not (isinstance(source, dict) and source.get("url")):
        return (f"[Record and receipt provenance]({record_link}) · "
                "[Original artifact retrieval](../../docs/forge-artifact-resolver.md). "
                "Original receipt availability is separate from this record.")
    return f"[Evidence]({_link(source)}) · [Record]({record_link})"


def _cell(value) -> str:
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def _next_action(status: str, pending_readout: bool) -> str:
    if pending_readout:
        return "Publish an explanation, comparison, and next action."
    if status == "HELD":
        return ("No counted task remains: the rest of this view is held, so it shows HELD, never PASS, "
                "until a held test meets its re-entry rule.")
    return "Run the next eligible task within an explicit budget."


def held_note(row: dict) -> str:
    """'; N held (M pass, not counted)' for a view with held tasks, else empty."""
    held = row.get("held")
    return f"; {held['total']} held ({held['passed']} pass, not counted)" if held else ""


def _board_markdown(result: dict) -> str:
    lines = [f"# {result['view']} — revision {result['view_revision']}", "", result["ranking_note"], "",
             "| Candidate / exact revision | Scope | Tier | Outcomes | Wall seconds | Next action |",
             "| --- | --- | ---: | --- | ---: | --- |"]
    for row in result["rows"]:
        identity = str(row["candidate_id"]) + " / " + str(row.get("candidate_revision") or "unknown")[:12]
        runtime = row.get("runtime_cohort", {})
        backend = runtime.get("execution_backend")
        if backend:
            identity += " / " + backend
            models = sorted({p.get("model") for p in runtime.get("compute_profiles", {}).values() if p and p.get("model")})
            if models:
                identity += " (" + ", ".join(models) + ")"
        counts = (", ".join(f"{key} {value}" for key, value in row.get("counts", {}).items()) or "unmeasured") + held_note(row)
        seconds = row["cost"].get("wall_seconds")
        lines.append("| " + " | ".join(_cell(x) for x in (identity, row["evidence_scope"], row["qualified_tier"],
                     counts, round(seconds, 3) if seconds is not None else None, row.get("next_action"))) + " |")
    lines += ["", "Historical summaries overlap individual attempt rows. Do not sum them as independent trials or total cost.", ""]
    return "\n".join(lines)


def _metric_excerpt(rows: list[dict]) -> str:
    fragments = []
    keys = ("modes", "hq", "precision", "acc_center_rms_sigma", "center_rms_sigma", "mass_tv",
            "acc_radial_ks", "radial_ks", "recon_mse", "error", "passing_checks", "observations")
    for row in rows[:3]:
        metrics = row.get("metrics", {})
        source = metrics.get("final", metrics.get("metrics", metrics)) if isinstance(metrics, dict) else {}
        if not isinstance(source, dict):
            continue
        selected = {key: source[key] for key in keys if type(source.get(key)) in (int, float)}
        if selected:
            fragments.append(str(row.get("task_id")) + ": " + ", ".join(f"{key}={value:.5g}" for key, value in selected.items()))
    return "; ".join(fragments)


def compile_memory(root: Path, *, summaries_only: bool = False) -> dict:
    """Map outputs reduce deterministically, with one writer and explicit gaps."""
    root = Path(root)
    output = root / "reports/forge"
    with file_lock(root / "runs/forge/compile.lock"):
        previous = read_json(output / "compilation.json") if (output / "compilation.json").is_file() else {}
        records, conflicts = _memory_records(root)
        boards = []
        from . import planning, sources
        orig_inspect = sources.inspect_source
        orig_planning_inspect = planning.inspect_source
        orig_attempts = _attempts
        orig_records = _records
        source_cache: dict = {}
        attempts_cache: dict = {}
        records_cache: dict = {}

        def cached_inspect(r: Path, extra_paths=()):
            key = (str(Path(r).resolve()), tuple(sorted(str(p) for p in extra_paths)))
            if key not in source_cache:
                source_cache[key] = orig_inspect(r, extra_paths)
            return dict(source_cache[key])

        def cached_attempts(r: Path):
            key = str(Path(r).resolve())
            if key not in attempts_cache:
                attempts_cache[key] = orig_attempts(r)
            items, issues = attempts_cache[key]
            return list(items), list(issues)

        def cached_records(r: Path):
            key = str(Path(r).resolve())
            if key not in records_cache:
                records_cache[key] = orig_records(r)
            items, issues = records_cache[key]
            return list(items), list(issues)

        if not summaries_only:
            sources.inspect_source = cached_inspect
            planning.inspect_source = cached_inspect
            globals()["_attempts"] = cached_attempts
            globals()["_records"] = cached_records
        try:
            with planning.resolution_cache(root):
                for path in sorted((root / "configs/forge/views").glob("*.json")):
                    if read_json(path).get("evidence_scope") == "research_diagnostic":
                        continue
                    if summaries_only:
                        result = {"view": path.stem, "current_rows": [], "pinned_rows": [], "calibration_rows": [], "conflicts": []}
                    else:
                        result = board(root, path.stem)
                    boards.append(result)
                    if summaries_only:
                        markdown_path = leaderboard_path(root, path.stem)
                        published_path = root / markdown_path
                        legacy_pointer = f"# {path.stem}\n\nThis view has no previously published goal table. "
                        if not published_path.exists() or published_path.read_text().startswith(legacy_pointer):
                            _text_output(root / markdown_path,
                                         f"# {path.stem}\n\n{DECLARATION_ONLY_BOARD}\n\nThis view has no previously published goal table. "
                                         "The compact memory refresh records its declaration without replaying original receipts. "
                                         f"Inspect `forge board --goal {path.stem} --json` with compatible original evidence "
                                         "before making a qualification claim.\n")
                        continue
                    _json_output(output / "leaderboards" / (path.stem + ".json"), result)
                    markdown_path = leaderboard_path(root, path.stem)
                    if markdown_path.parent == Path("reports/forge/leaderboards"):
                        _text_output(root / markdown_path, _board_markdown(result))
                    else:
                        # This is generated output; numerical cohort evidence stays in
                        # the goal JSON and the inventory's evidence registry.
                        (output / "leaderboards" / (path.stem + ".md")).unlink(missing_ok=True)
        finally:
            if not summaries_only:
                sources.inspect_source = orig_inspect
                planning.inspect_source = orig_planning_inspect
                globals()["_attempts"] = orig_attempts
                globals()["_records"] = orig_records
        gaps = _gaps(root)
        inputs = _input_hashes(root, summaries_only=summaries_only)
        coverage = gaps.get("inventory_coverage", {"valid": False, "reason": "inventory unavailable"})
        try:
            from .history import validate_inventory
            coverage = validate_inventory(root)
        except Exception as exc:
            # Repo-less fixtures and exported artifacts can still be read, never called complete.
            coverage = {"valid": False, "reason": str(exc)}
        pending = sorted({row["candidate_id"] for result in boards for row in result["current_rows"] + result.get("pinned_rows", []) + result.get("calibration_rows", []) + result.get("research_rows", [])
                          if row.get("pending_readout")})
        scientific_sources = sorted({row["source_digest"] for result in boards for row in result["current_rows"]
                                     if row.get("source_digest")})
        if summaries_only:
            concluded_ids = {
                record.get("candidate_id")
                for record in records
                if record.get("lifecycle") in {"concluded", "abandoned", "superseded"}
                and record.get("candidate_id")
            }
            pending = [cid for cid in previous.get("pending_readout", []) if cid not in concluded_ids]
            scientific_sources = previous.get("scientific_source_digests", [])
        reducer_hashes = _reducer_hashes()
        reducer_hash = file_hash(Path(__file__))
        manifest = {"schema_version": 1, "reducer_version": REDUCER_VERSION, "reducer_sha256": reducer_hash,
                    "reducer_hashes": reducer_hashes,
                    "input_hashes": inputs, "scientific_source_digests": scientific_sources,
                    "input_digest": stable_hash({"files": inputs, "sources": scientific_sources, "reducers": reducer_hashes}),
                    "record_count": len(records),
                    "view_count": len(list((root / "configs/forge/views").glob("*.json"))),
                    "inventory_coverage": coverage,
                    "conflicts": conflicts + [issue for result in boards for issue in result["conflicts"]],
                    "pending_readout": pending, "import_gap_count": len(gaps.get("gaps", [])),
                    "scientific_normalization_complete": gaps.get("scientific_normalization_complete", False)}
        manifest["qualification_refresh"] = {
            "mode": "preserved_published_evidence" if summaries_only else "compatible_receipt_reducer",
            "source_manifest_digest": (previous.get("qualification_refresh", {}).get("source_manifest_digest")
                                       or previous.get("input_digest")) if summaries_only else None,
            "note": "Publication recall freshness grants no qualification; original evidence availability is separate."}
        manifest["operational_lifecycle_digest"] = stable_hash([
            {"cohort": row.get("cohort"), "lifecycle": row.get("lifecycle"), "requests": row.get("queue_submissions", [])}
            for result in boards for row in result["current_rows"] + result.get("pinned_rows", []) + result.get("calibration_rows", []) + result.get("research_rows", [])])
        if summaries_only:
            manifest["operational_lifecycle_digest"] = previous.get("operational_lifecycle_digest")
            automation = read_json(output / "automation.json") if (output / "automation.json").is_file() else {
                "schema_version": 1, "scope": "unavailable; compact memory refresh does not replay original receipts"}
        else:
            from .telemetry import summarize_automation
            automation = summarize_automation(root)
        _json_output(output / "automation.json", automation)
        manifest["automation_digest"] = stable_hash(automation)
        from .publication_memory import OUTPUT, VERSION
        publication_records = [record for record in records if record.get("evidence_scope") == "published_summary"]
        _json_output(root / OUTPUT, {"schema_version": 1, "reducer_version": VERSION,
                                    "qualification_input": False, "records": publication_records})
        manifest["publication_record_count"] = len(publication_records)
        lines = ["# ParticleGAN Forge experiment memory", "",
                 "Read the relevant prior evidence before declaring an idea. Historical outcomes retain their original scope. "
                 + ("Published qualification and automation snapshots are retained without replaying unavailable originals. " if summaries_only
                    else "Current qualification is recomputed from compatible receipts. ")
                 + "Compact publications enter recall as display-only summaries and grant no qualification.", "",
                 f"Records: {len(records)}. Inventory coverage: {'complete' if coverage.get('valid') else 'incomplete'}. "
                 f"Unresolved import items: {len(gaps.get('gaps', []))}.", "",
                 "## Goal views", ""]
        lines.extend(f"- [{result['view']}]({leaderboard_path(root, result['view']).relative_to('reports/forge').as_posix()})"
                     for result in boards)
        lines += ["", "[Measured automation costs, reuse, and avoided work](automation.json). "
                  "Run `python -m experiments.forge stats` for current accounting; unavailable measurements remain explicit.",
                  "", "## Pending readouts", "", ", ".join(pending) or "None recorded.", "",
                  "## Experiment and family records", ""]
        for record in records:
            if record.get("evidence_scope") == "published_summary":
                continue
            rows = record.get("task_results", [])
            counts = dict(sorted(Counter(row.get("gate_status", "NOT_RUN") for row in rows).items()))
            seconds = _cost(rows)["wall_seconds"]
            lines += [f"### {record.get('candidate_id', 'unknown')} · {record['record_id']}", "",
                      f"**Scope:** {record.get('evidence_scope', 'current')}; {record.get('record_type', 'scientific')}; "
                      f"revision `{record.get('candidate_revision', 'unknown')}`.", "",
                      str(record.get("hypothesis", "Hypothesis unrecorded.")), "",
                      f"**Observed:** {_cell(counts) if counts else 'No normalized scientific verdict'}; "
                      f"wall seconds {_cell(round(seconds, 3) if seconds is not None else None)}; mechanism `{record.get('mechanism_class', 'unknown')}`.", "",
                      _metric_excerpt(rows), "",
                      str(record.get("conclusion", "Conclusion pending.")), "",
                      f"**Next:** {record.get('next_action', 'Readout pending.')}", "",
                      _evidence_navigation(record), ""]
        closed_partial = any(record.get("lifecycle") == "closed_partial_cut" for record in publication_records)
        lines += ["## Compact publications and closed partial cuts" if closed_partial else "## Concluded compact publications", "",
                  "These study and trial projections preserve recorded outcomes, unknowns and source cohorts. "
                  "They do not regrade archived science or replace original receipts. Overlapping study/trial costs must not be summed.", "",
                  "[Normalized publication records](publication-records.json). "
                  "Search exact IDs, failed requirement metric names, mechanisms and goals with `forge recall`. "
                  "Check current input coverage with `forge compile --check`.", ""]
        for record in publication_records:
            if record["record_type"] != "published_study":
                continue
            lines += [f"- **{record['study_id']}**: {record['conclusion']} "
                      f"[Source]({_link(record['source'])}) · [Board](../../{record['source']['board']})"]
        lines.append("")
        lines += ["## Unresolved imports and limitations", ""]
        for gap in gaps.get("gaps", []):
            lines.append(f"- **{gap.get('gap_id', gap.get('kind', 'unknown'))}:** {gap.get('reason', 'Unresolved')}"
                         + (f" ({len(gap['paths'])} classified sources.)" if "paths" in gap else ""))
        lines += ["", "## Compilation provenance", "",
                  f"Reducer `{REDUCER_VERSION}`; input digest `{manifest['input_digest']}`. "
                  "[Full input hashes and coverage](compilation.json). No training or image inspection occurs during compilation.", ""]
        _text_output(output / "EXPERIMENT_MEMORY.md", "\n".join(lines))
        manifest["output_hashes"] = {name: file_hash(root / name)
                                     for name in ("reports/forge/EXPERIMENT_MEMORY.md", OUTPUT)}
        _json_output(output / "compilation.json", manifest)
        return {"records": len(records), "views": len(boards), "pending_readout": pending,
                "inventory_coverage": coverage, "input_digest": manifest["input_digest"],
                "qualification_refresh": manifest["qualification_refresh"],
                "memory": "reports/forge/EXPERIMENT_MEMORY.md", "conflicts": manifest["conflicts"]}


def readout_selection_claim(root: Path, candidate: dict, view: dict, task_results: list) -> dict | None:
    """Selection-versus-holdout claim for a candidate descending from a registered search.

    Returns None when no ancestor came from a configuration search. Otherwise
    the claim states whether the required tasks above the search's tuning tier
    (the untuned holdout) passed; tuning-tier passes are never independent
    confirmation. Missing results count as not measured.
    """
    from .leaderboard_contract import holdout_claim, selection_lineage
    lineage = selection_lineage(root, candidate)
    if not lineage.get("search_id"):
        return None
    statuses = {}
    for row in task_results:
        statuses[row.get("task_id")] = row.get("gate_status", "UNKNOWN")
    by_tier = {}
    for assignment in view.get("assignments", []):
        if assignment.get("importance") == "required":
            by_tier.setdefault(assignment["qualification_tier"], []).append(
                statuses.get(assignment["task"], "UNKNOWN"))
    return {**lineage, **holdout_claim(by_tier, lineage.get("tuning_through_tier"))}


def readout(root: Path, candidate_id: str, conclusion: str, comparison: str, next_action: str, *, study_id=None) -> dict:
    """Publish a readout for one exact attempted revision, then update memory."""
    root = Path(root)
    identifier(candidate_id, "candidate id")
    if not all(isinstance(value, str) and value.strip() for value in (conclusion, comparison, next_action)):
        raise ValueError("readout requires conclusion, comparison, and next_action")
    requested_revision = None
    if "@" in candidate_id and not (root / "configs/forge/ideas" / (candidate_id + ".json")).exists():
        candidate_id, requested_revision = candidate_id.rsplit("@", 1)
    attempts, _ = _attempts(root)
    states = _queue_states(root, attempts)
    study_request = None
    from .studies import frozen_declaration, selected_study_id
    if study_id is not None and (study_id.endswith(".json") or "/" in study_id):
        study_path = (root / study_id).resolve() if not Path(study_id).is_absolute() else Path(study_id)
        if study_path.exists():
            import json
            try:
                study_id = json.loads(study_path.read_text()).get("id", study_id)
            except Exception:
                pass
    selected = [a for a in attempts if a["request"].get("candidate", {}).get("id") == candidate_id
                and (requested_revision is None or a["request"]["candidate_revision"].startswith(requested_revision))
                and (study_id is None or a["request"].get("study", {}).get("id") == study_id)]
    if study_id is not None:
        from .studies import select_observations
        study_request, selected = select_observations(states, attempts, candidate_id, study_id, requested_revision)
    if study_id is None and any(selected_study_id(a["request"]) for a in selected):
        raise ValueError("Study-bound attempts require readout --study STUDY to keep research questions separate")
    revisions = {a["request"]["candidate_revision"] for a in selected}
    if not revisions:
        raise ValueError("No durable execution attempt exists for this candidate/revision.")
    if len(revisions) != 1:
        raise ValueError("Multiple revisions have attempts; use candidate-id@revision to bind the readout.")
    revision = revisions.pop()
    candidate = (study_request or selected[0]["request"])["candidate"]
    study = frozen_declaration(study_request) if study_request else None
    scopes = {diagnostic_evidence_scope(a["request"]) for a in selected}
    if "research_diagnostic" in scopes and len(scopes) != 1:
        raise ValueError("research diagnostic readouts cannot combine ordinary or calibration evidence")
    diagnostic_scope = next(iter(scopes)) if len(scopes) == 1 else None
    diagnostic_only = diagnostic_scope is not None
    identity = {"candidate": candidate_id, "revision": revision}
    if study:
        identity.update(study_id=study["id"], study_sha256=stable_hash(study))
    record = {"schema_version": 1, "record_id": "readout-" + stable_hash(identity)[:24],
              "record_type": "scientific", "candidate_id": candidate_id, "candidate_revision": revision,
              "evidence_scope": diagnostic_scope or "current",
              "qualification_reuse": not diagnostic_only, "lifecycle": "concluded",
              "goal": study.get("scope", {}).get("view", study.get("view")) if study else candidate.get("goal"),
              "hypothesis": study["hypothesis"] if study else candidate.get("hypothesis", "See immutable candidate declaration."),
              "mechanism_class": candidate.get("mechanism_class", "unknown"), "prior": candidate.get("prior"),
              **({"prior_mechanism": candidate["prior_mechanism"]} if candidate.get("prior_mechanism") is not None else {}),
              "claim_contract": candidate.get("claim_contract"),
              "source": {"path": selected[0]["source"]},
              "task_results": [row for attempt in selected for row in attempt["task_results"]],
              "attempt_ids": sorted(a["attempt_id"] for a in selected),
              "provenance": {"attempts": [{"attempt_id": a["attempt_id"], "result_hash": a["result_hash"],
                                             "valid_receipt": a["valid_receipt"]} for a in selected]},
              "conclusion": conclusion.strip(), "comparison": comparison.strip(), "next_action": next_action.strip()}
    if study:
        record.update(study_id=study["id"], study_sha256=stable_hash(study), study=study)
    if diagnostic_scope == "research_diagnostic":
        from reports.forge.regenerate_technique_inventory import _evaluator_summary
        record.update(qualification_input=False, qualified_tier=0, eligible=False)
        record["task_results"] = [{**{key: row[key] for key in (
            "task_id", "compatibility_key", "gate_status", "raw_status", "reason", "metrics", "cost") if key in row},
            "evaluator_summary": _evaluator_summary(row.get("evaluator_result", {}))}
            for attempt in selected for row in attempt["task_results"]]
    if study_request and "study" in study_request:
        from .studies import conclude_observations
        record["decision_outcomes"] = [conclude_observations(study_request, selected)]
    elif study_request and "search_plan" in study_request:
        record["search_plan"] = study_request["search_plan"]
    elif candidate.get("decision_contract") is not None:
        from .decision_contracts import concluded_outcomes
        record["decision_outcomes"] = concluded_outcomes(selected)
    claim = readout_selection_claim(root, candidate, (study_request or selected[0]["request"]).get("view", {}),
                                    record["task_results"])
    if claim is not None:
        record["selection_claim"] = claim
    with file_lock(root / "runs/forge/readout.lock"), ExitStack() as locks:
        for location in sorted(states):
            locks.enter_context(file_lock(Path(location) / "queue.lock"))
        stopped = []
        for location in sorted(states):
            state = read_json(Path(location) / "queue/state.json")
            matching = [e for e in state.get("submissions", {}).values()
                        if e["request"].get("candidate", {}).get("id") == candidate_id
                        and e["request"].get("candidate_revision") == revision
                        and (study_id is None or selected_study_id(e["request"]) == study_id)]
            if any(e["status"] in {"queued", "running", "paused"} for e in matching):
                raise ValueError("Candidate has active queued/running/paused work; stop or complete it before readout.")
            stopped.append((location, state, matching))
        atomic_json(root / "reports/forge/records" / (record["record_id"] + ".json"), record)
        for location, state, matching in stopped:
            if not matching:
                continue
            for entry in matching:
                entry.update(status="concluded", lifecycle="concluded", readout_record_id=record["record_id"],
                             readout_attempt_ids=record["attempt_ids"])
            atomic_json(Path(location) / "queue/state.json", state)
    if study is not None or diagnostic_scope == "research_diagnostic":
        compile_memory(root, summaries_only=True)
    else:
        compile_memory(root)
    return record
