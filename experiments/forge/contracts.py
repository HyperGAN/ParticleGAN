"""Small, strict, dependency-free contracts shared by Forge's file protocols.

Scientific graders live with the tasks. These helpers never infer a scientific
pass from an administrative status or a successful process exit.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

SCHEMA_VERSION = 1
GATES = frozenset({"PASS", "FAIL", "INCOMPLETE", "INVALID", "NOT_RUN", "BLOCKED"})
ATTEMPT_STATES = frozenset({"queued", "running", "completed", "error", "timeout", "cancelled"})
LIFECYCLE = {
    "proposed": {"ready", "abandoned", "superseded"},
    "ready": {"running", "awaiting_readout", "abandoned", "superseded"},
    "running": {"awaiting_readout"},
    "awaiting_readout": {"concluded", "ready", "abandoned", "superseded"},
    "concluded": {"superseded", "abandoned"},
    "abandoned": set(), "superseded": set(),
}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def clone_json(value):
    """Fast exact deepcopy for JSON-shaped dicts/lists/primitives; falls back to deepcopy."""
    t = type(value)
    if t is dict:
        return {k: clone_json(v) for k, v in value.items()}
    if t is list:
        return [clone_json(v) for v in value]
    if t is str or t is int or t is float or t is bool or value is None:
        return value
    if t is tuple:
        return tuple(clone_json(v) for v in value)
    return deepcopy(value)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    """Kernel-backed ownership; an old lock file does not imply a live owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"another coordinator owns {path}") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def identifier(value: str, label: str = "id") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@+-]{0,159}", value):
        raise ValueError(f"{label} must be a simple nonempty name (no directories): {value!r}")
    return value


def positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def require_fields(value: dict, fields, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = set(fields) - value.keys()
    if missing:
        raise ValueError(f"{label} missing {sorted(missing)}")
    canonical(value)


def validate_idea(idea: dict) -> None:
    require_fields(idea, ("schema_version", "id", "changed_factors", "mechanism_class"), "candidate")
    if type(idea["schema_version"]) is not int or idea["schema_version"] not in (1, 2, 3):
        raise ValueError("unsupported idea schema_version")
    if idea["schema_version"] == 3:
        # New recipes cannot carry an experiment's conditions or authorization.
        allowed = {"schema_version", "id", "parent", "changed_factors", "mechanism_class",
                   "mechanism_rationale", "recipe_preset", "recipe_overrides", "extensions",
                   "requires_capabilities", "api_changes", "api_version", "execution_path",
                   "initializer", "claim_contract", "host_adaptation", "prior_art", "guide",
                   "trainer_family", "configuration_id", "resolved_configuration_recipe",
                   "prior_mechanism"}
        if set(idea) - allowed:
            raise ValueError("v3 candidate has study/task-owned or unsupported fields: "
                             + ", ".join(sorted(set(idea) - allowed)))
        from .boundaries import TASK_RECIPE_FIELDS
        if set(idea.get("recipe_overrides", {})) & TASK_RECIPE_FIELDS:
            raise ValueError("v3 candidate recipe_overrides contain task-owned fields")
    else:
        require_fields(idea, ("hypothesis", "goal"), "legacy idea")
        identifier(idea["goal"], "goal")
        if not isinstance(idea["hypothesis"], str) or not idea["hypothesis"].strip():
            raise ValueError("state a hypothesis before submitting an idea")
    if "prior_mechanism" in idea:
        from .priors import validate_prior_mechanism
        validate_prior_mechanism(idea["prior_mechanism"])
    if idea["schema_version"] == 2 and "decision_contract" not in idea:
        raise ValueError("v2 ideas require a decision_contract; use forge new to create the draft scaffold")
    if "decision_contract" in idea:
        from .decision_contracts import validate_shape
        validate_shape(idea["decision_contract"])
    identifier(idea["id"], "idea id")
    factors = idea["changed_factors"]
    if not isinstance(factors, list) or not factors or not all(isinstance(x, str) and x.strip() for x in factors):
        raise ValueError("changed_factors must describe at least one mechanism/configuration change")
    if all(x.lower().strip() in {"seed", "random_seed", "rng_seed"} for x in factors):
        raise ValueError("seed-only ideas are forbidden; use a registered finished-candidate promotion stage")
    if idea["mechanism_class"] not in {"structural", "floor_constant", "sampling_only_patch"}:
        raise ValueError("unknown mechanism_class")
    if "seed" in idea:
        raise ValueError("screening seed belongs to the protocol, not the idea")
    from .taskrecipes import validate_host_adaptation
    validate_host_adaptation(idea)


def transition(record: dict, destination: str, *, reason: str | None = None, successor: str | None = None) -> dict:
    current = record.get("lifecycle", "proposed")
    if destination not in LIFECYCLE.get(current, set()):
        raise ValueError(f"invalid experiment transition {current} -> {destination}")
    if destination in {"abandoned", "superseded"} and not reason:
        raise ValueError(f"{destination} requires a reason")
    if destination == "superseded" and not successor:
        raise ValueError("superseded requires a successor")
    if destination == "concluded" and not all(record.get(k) for k in ("conclusion", "next_action", "comparison")):
        raise ValueError("concluded requires conclusion, comparison and next_action")
    result = {**record, "lifecycle": destination}
    result["lifecycle_events"] = [*record.get("lifecycle_events", []), {
        "from": current, "to": destination, "reason": reason, "successor": successor, "timestamp": utc_now(),
    }]
    return result
