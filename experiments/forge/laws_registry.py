"""Data-driven registry for ParticleGAN's provisional, evolving learning laws.

Never hardcode law identifiers in renderers or validators: read them dynamically
from ``configs/forge/laws.json`` so laws may be refined, split, merged, or
replaced as mathematical and empirical evidence advances.
"""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
import re

from .contracts import read_json

LAWS_REGISTRY = Path("configs/forge/laws.json")
VALID_LAW_STATUSES = frozenset({"active", "provisional", "superseded", "merged", "deprecated"})
VALID_MECHANISM_STATUSES = ("addressed", "partial", "unaddressed", "not_applicable")
_LAW_ID = re.compile(r"[a-z][a-z0-9_-]*")
_ANCHOR = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _nonempty_text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"laws registry: {label} must be nonempty text")
    return value


def _safe_relative_path(value, label: str) -> str:
    text = _nonempty_text(value, label)
    path = Path(text)
    if path.is_absolute() or "\\" in text or ".." in path.parts:
        raise ValueError(f"laws registry: {label} must be a repository-relative path")
    return text


def _validate_sub_law(entry, seen_ids: set[str], parent_id: str) -> dict:
    required = {"id", "short_label", "title", "doc_anchor"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise ValueError(
            f"laws registry: sub-law under {parent_id} must have exact fields {sorted(required)}"
        )
    sub_id = _nonempty_text(entry["id"], f"sub-law id under {parent_id}")
    if not _LAW_ID.fullmatch(sub_id):
        raise ValueError(f"laws registry: invalid sub-law id {sub_id!r}")
    if sub_id in seen_ids:
        raise ValueError(f"laws registry: duplicate law id {sub_id!r}")
    seen_ids.add(sub_id)
    _nonempty_text(entry["short_label"], f"sub-law {sub_id} short_label")
    _nonempty_text(entry["title"], f"sub-law {sub_id} title")
    anchor = _nonempty_text(entry["doc_anchor"], f"sub-law {sub_id} doc_anchor")
    if not _ANCHOR.fullmatch(anchor):
        raise ValueError(f"laws registry: invalid doc_anchor {anchor!r} for {sub_id}")
    return deepcopy(entry)


def _validate_registry_document(document: dict) -> dict:
    if not isinstance(document, dict):
        raise ValueError("laws registry must be a JSON object")
    required_top = {
        "schema_version",
        "epistemic_status",
        "epigraph",
        "epistemic_note",
        "doc_path",
        "laws",
    }
    if set(document) != required_top:
        raise ValueError(
            f"laws registry must contain exact top-level fields {sorted(required_top)}"
        )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("laws registry: schema_version must be integer 1")
    _nonempty_text(document["epistemic_status"], "epistemic_status")
    _nonempty_text(document["epigraph"], "epigraph")
    _nonempty_text(document["epistemic_note"], "epistemic_note")
    _safe_relative_path(document["doc_path"], "doc_path")
    laws = document["laws"]
    if not isinstance(laws, list) or not laws:
        raise ValueError("laws registry: laws must be a nonempty list")

    required_law = {
        "id",
        "short_label",
        "title",
        "domain_level",
        "question",
        "doc_anchor",
        "status",
        "sub_laws",
    }
    seen_ids: set[str] = set()
    for law in laws:
        if not isinstance(law, dict) or set(law) != required_law:
            raise ValueError(
                f"laws registry: each law entry must have exact fields {sorted(required_law)}"
            )
        law_id = _nonempty_text(law["id"], "law id")
        if not _LAW_ID.fullmatch(law_id):
            raise ValueError(f"laws registry: invalid law id {law_id!r}")
        if law_id in seen_ids:
            raise ValueError(f"laws registry: duplicate law id {law_id!r}")
        seen_ids.add(law_id)
        for field in ("short_label", "title", "domain_level", "question"):
            _nonempty_text(law[field], f"law {law_id} {field}")
        anchor = _nonempty_text(law["doc_anchor"], f"law {law_id} doc_anchor")
        if not _ANCHOR.fullmatch(anchor):
            raise ValueError(f"laws registry: invalid doc_anchor {anchor!r} for {law_id}")
        status = _nonempty_text(law["status"], f"law {law_id} status")
        if status not in VALID_LAW_STATUSES:
            raise ValueError(
                f"laws registry: law {law_id} status {status!r} must be one of {sorted(VALID_LAW_STATUSES)}"
            )
        sub_laws = law["sub_laws"]
        if not isinstance(sub_laws, list):
            raise ValueError(f"laws registry: law {law_id} sub_laws must be a list")
        for sub in sub_laws:
            _validate_sub_law(sub, seen_ids, law_id)
    return deepcopy(document)


def load_laws(root: Path | str, load=None) -> dict | None:
    """Load and validate ``configs/forge/laws.json`` if present, else return None."""
    root = Path(root)
    path = root / LAWS_REGISTRY
    if not path.is_file():
        return None
    raw = load(LAWS_REGISTRY) if load is not None else read_json(path)
    return _validate_registry_document(raw)


def top_level_laws(root: Path | str, load=None) -> list[dict]:
    """Return ordered top-level law definitions from ``configs/forge/laws.json``."""
    registry = load_laws(root, load=load)
    return list(registry["laws"]) if registry is not None else []


def laws_by_status(
    root: Path | str,
    statuses: str | Iterable[str],
    load=None,
) -> list[dict]:
    """Return ordered top-level laws whose ``status`` matches ``statuses``."""
    allowed = {statuses} if isinstance(statuses, str) else set(statuses)
    if not allowed or not allowed.issubset(VALID_LAW_STATUSES):
        invalid = sorted(allowed - VALID_LAW_STATUSES)
        raise ValueError(
            f"laws registry: status filter {invalid or sorted(allowed)!r} "
            f"must be one of {sorted(VALID_LAW_STATUSES)}"
        )
    return [law for law in top_level_laws(root, load=load) if law["status"] in allowed]


def active_laws(root: Path | str, load=None) -> list[dict]:
    """Return ordered top-level laws with ``status == 'active'``."""
    return laws_by_status(root, "active", load=load)


def provisional_laws(root: Path | str, load=None) -> list[dict]:
    """Return ordered top-level laws with ``status == 'provisional'``."""
    return laws_by_status(root, "provisional", load=load)


def top_level_law_ids(root: Path | str, load=None) -> list[str]:
    """Return ordered top-level law identifiers from ``configs/forge/laws.json``."""
    return [law["id"] for law in top_level_laws(root, load=load)]


def all_law_ids(root: Path | str, load=None) -> list[str]:
    """Return all law and sub-law identifiers in declaration order."""
    ids: list[str] = []
    for law in top_level_laws(root, load=load):
        ids.append(law["id"])
        for sub in law["sub_laws"]:
            ids.append(sub["id"])
    return ids


def validate_laws_registry(root: Path | str) -> dict:
    """Validate the evolving laws registry and verify its backing document path."""
    root = Path(root)
    if not (root / LAWS_REGISTRY).is_file():
        return {"status": "unavailable", "laws": 0, "sub_laws": 0}
    registry = load_laws(root)
    doc_path = Path(registry["doc_path"])
    resolved = (root / doc_path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"laws registry: doc_path must be an existing repository file: {registry['doc_path']}")
    sub_count = sum(len(law["sub_laws"]) for law in registry["laws"])
    return {
        "status": "valid",
        "schema_version": registry["schema_version"],
        "epistemic_status": registry["epistemic_status"],
        "laws": len(registry["laws"]),
        "sub_laws": sub_count,
    }
