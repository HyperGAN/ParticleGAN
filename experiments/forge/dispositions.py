"""Archive-first view revisions for held and removed-from-count test dispositions.

A dispositions registry (schema ``forge_test_dispositions_v1``) records one
decision per test. HOLD makes the view role ``held``: the test stays declared,
measured and shown, but it is not counted. REMOVE keeps the test assigned as
``diagnostic`` with a removal note naming why it left the count. KEEP (stays
counted), NO_CHANGE (variants) and FIX (pytest items) are registry-only records.
The registry is provenance and input to this command. Grading never reads it, so
view fingerprints never depend on a mutable file; the notes are copied into the
views, and changing one is a policy change that needs a new archived revision.

``check`` writes nothing and reports drift between the registry and the views.
``apply`` first copies each changed view's exact bytes to
configs/forge/view-history/<view>-v<revision>.json (refusing a different file
already there), validates every view, then writes the new revisions atomically
in each file's own key order. A second apply changes nothing. Each registry view
entry pins its base revision's policy fingerprint (``from_fingerprint``), so a
live file edited without a revision bump is refused instead of being archived
as that revision. Nothing here trains, grades or enqueues work.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile

from .views import (HOLD_REASONS, HOLD_SCHEMA, REMOVAL_REASONS, REMOVAL_SCHEMA, counted_by_matches,
                    explanation_reference, load_tasks, plain_summary, validate_view, view_fingerprint)

SCHEMA = "forge_test_dispositions_v1"
VIEW_ROLES = {"HOLD": "held", "REMOVE": "diagnostic"}
SCOPES = {"HOLD": {"required", "policy"}, "REMOVE": {"required", "policy"}, "KEEP": {"required", "policy"},
          "NO_CHANGE": {"variant"}, "FIX": {"pytest"}}
TOP_LEVEL = {"schema", "decided_on", "evidence_snapshot", "explanations", "qualification_input",
             "historical_verdicts_regraded", "training_launched", "views", "dispositions"}
ANCHOR = re.compile(r"""<a\s+(?:name|id)\s*=\s*["']([^"']+)["']""")


def archive_path(view_id: str, revision: int) -> str:
    return f"configs/forge/view-history/{view_id}-v{revision}.json"


def key_style(raw: str) -> str:
    """Files whose top-level keys are sorted take new keys in sorted position."""
    keys = list(json.loads(raw))
    return "sorted" if keys == sorted(keys) else "insertion"


def _put(mapping: dict, key, value, style: str) -> None:
    """Set a key without moving existing keys; a new key follows the file's style."""
    if key in mapping or style != "sorted":
        mapping[key] = value
        return
    items = list(mapping.items())
    position = next((i for i, (name, _) in enumerate(items) if name > key), len(items))
    items.insert(position, (key, value))
    mapping.clear()
    mapping.update(items)


def _ordered(note: dict, style: str) -> dict:
    return dict(sorted(note.items())) if style == "sorted" else note


def dump_view(view: dict, original: str) -> str:
    """Indent 2 and a trailing newline, escaping non-ASCII as the original file does."""
    return json.dumps(view, indent=2, ensure_ascii=original.isascii(), allow_nan=False) + "\n"


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
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


def _archive_state(root: Path, view_id: str, revision: int, live: bytes) -> str:
    """'create' or 'reuse' for the exact live bytes; a different archive is refused."""
    path = root / archive_path(view_id, revision)
    if not path.exists():
        return "create"
    if path.read_bytes() != live:
        raise ValueError(f"{archive_path(view_id, revision)} already exists with different bytes; "
                         "an archived revision is never overwritten")
    return "reuse"


def write_view_revision(root: Path | str, view: dict, reason: str, *, tasks: dict | None = None) -> dict:
    """Archive the live revision byte for byte, then write the next revision.

    ``view`` is the live view with its policy edits applied and its revision
    unchanged. The new revision records its archive and the old fingerprint.
    """
    root = Path(root)
    path = root / "configs/forge/views" / f"{view['id']}.json"
    live = path.read_bytes()
    raw = live.decode("utf-8")
    current = json.loads(raw)
    if current.get("revision") != view.get("revision"):
        raise ValueError("view revision changed while it was being edited; reload the live view")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a view revision needs a plain-language policy_change_reason")
    style = key_style(raw)
    revised = deepcopy(view)
    revision = current["revision"]
    _put(revised, "revision", revision + 1, style)
    _put(revised, "previous_revision", _ordered({"revision": revision, "archive": archive_path(view["id"], revision),
                                                 "policy_fingerprint": view_fingerprint(current)}, style), style)
    _put(revised, "policy_change_reason", reason, style)
    validate_view(revised, load_tasks(root) if tasks is None else tasks)
    if _archive_state(root, view["id"], revision, live) == "create":
        _atomic_bytes(root / archive_path(view["id"], revision), live)
    _atomic_bytes(path, dump_view(revised, raw).encode("utf-8"))
    return revised


def _text(value, label, minimum=1):
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise ValueError(f"{label} must be text of at least {minimum} characters")


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} needs exactly {sorted(keys)}")


def _applied(record) -> bool:
    return record.get("applied", True) is not False


def _validate_view_entry(view_id, entry):
    label = f"registry view {view_id}"
    optional = {"skipped_revision_note", "applied", "apply_in"}
    required = {"from_revision", "from_fingerprint", "to_revision", "archive", "policy_change_reason"}
    if (not isinstance(view_id, str) or Path(view_id).name != view_id or not isinstance(entry, dict)
            or not required <= set(entry) or set(entry) - required - optional):
        raise ValueError(f"{label} needs {sorted(required)} and optionally {sorted(optional)}")
    old, new = entry["from_revision"], entry["to_revision"]
    if type(old) is not int or type(new) is not int or not 1 <= old < new:
        raise ValueError(f"{label}: to_revision must follow from_revision")
    # The decided base revision's policy fingerprint: a live file edited without a
    # revision bump cannot be archived, or later accepted, as that revision.
    if not isinstance(entry["from_fingerprint"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["from_fingerprint"]):
        raise ValueError(f"{label}: from_fingerprint must be the base revision's SHA-256 policy fingerprint")
    if (new != old + 1) != ("skipped_revision_note" in entry):
        raise ValueError(f"{label}: a skipped revision number, and only one, needs skipped_revision_note")
    if "skipped_revision_note" in entry:
        _text(entry["skipped_revision_note"], f"{label} skipped_revision_note", 20)
    if entry["archive"] != archive_path(view_id, old):
        raise ValueError(f"{label}: archive must be {archive_path(view_id, old)}")
    _text(entry["policy_change_reason"], f"{label} policy_change_reason", 20)
    _validate_applied(entry, label)


def _validate_applied(value, label):
    if "applied" in value and type(value["applied"]) is not bool:
        raise ValueError(f"{label}: applied must be true or false")
    if (value.get("applied") is False) != ("apply_in" in value):
        raise ValueError(f"{label}: a record that is not applied names apply_in, and only such a record does")
    if "apply_in" in value:
        _text(value["apply_in"], f"{label} apply_in", 10)


def _validate_record(record, registry, tasks):
    if not isinstance(record, dict):
        raise ValueError("each disposition must be an object")
    kind = record.get("disposition")
    label = f"disposition {record.get('task')!r}"
    if kind not in SCOPES:
        raise ValueError(f"{label}: disposition must be one of {sorted(SCOPES)}")
    common = {"task", "scope", "disposition", "views", "group", "explanation"}
    extra = {"HOLD": {"hold"}, "REMOVE": {"removal"}, "KEEP": {"keep"}, "NO_CHANGE": {"tasks"},
             "FIX": {"in_handoff"}}[kind]
    optional = {"applied", "apply_in"} if kind in VIEW_ROLES else set()
    if not common | extra <= set(record) or set(record) - common - extra - optional:
        raise ValueError(f"{label} needs {sorted(common | extra)}"
                         + (f" and optionally {sorted(optional)}" if optional else ""))
    _text(record["task"], f"{label} task")
    if record["scope"] not in SCOPES[kind]:
        raise ValueError(f"{label}: {kind} applies to scope {sorted(SCOPES[kind])}")
    if record["group"] is not None:
        _text(record["group"], f"{label} group")
    explanation_reference(record["explanation"], label)
    if record["explanation"].split("#")[0] != registry["explanations"]:
        raise ValueError(f"{label}: explanation must link into {registry['explanations']}")
    views = record["views"]
    if not isinstance(views, dict) or any(type(tier) is not int or isinstance(tier, bool) or tier < 1 for tier in views.values()):
        raise ValueError(f"{label}: views must map view ids to positive integer tiers")
    if kind in {"NO_CHANGE", "FIX"}:
        if views:
            raise ValueError(f"{label}: {kind} records change no view")
    elif not views:
        raise ValueError(f"{label}: list every view that counts the test")
    if kind in {"HOLD", "REMOVE", "KEEP"} and record["task"] not in tasks:
        raise ValueError(f"{label}: unknown task")
    if kind == "HOLD":
        hold = record["hold"]
        _exact(hold, ("reason", "summary", "reentry"), f"{label} hold")
        if hold["reason"] not in HOLD_REASONS:
            raise ValueError(f"{label}: hold reason must be one of {sorted(HOLD_REASONS)}")
        plain_summary(hold["summary"], f"{label} hold")
        _text(hold["reentry"], f"{label} hold reentry", 20)
    elif kind == "REMOVE":
        removal = record["removal"]
        _exact(removal, ("reason", "counted_by", "summary"), f"{label} removal")
        if removal["reason"] not in REMOVAL_REASONS:
            raise ValueError(f"{label}: removal reason must be one of {sorted(REMOVAL_REASONS)}")
        if not counted_by_matches(removal["reason"], removal["counted_by"]):
            raise ValueError(f"{label}: phase and duplicate removals name counted_by; superseded and ill_posed use null")
        plain_summary(removal["summary"], f"{label} removal")
    elif kind == "KEEP":
        keep = record["keep"]
        _exact(keep, ("summary", "zero_cause"), f"{label} keep")
        plain_summary(keep["summary"], f"{label} keep")
        _text(keep["zero_cause"], f"{label} keep zero_cause", 20)
    elif kind == "NO_CHANGE":
        names = record["tasks"]
        if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(
                not isinstance(name, str) or name not in tasks for name in names):
            raise ValueError(f"{label}: tasks must list distinct declared tasks")
    elif type(record["in_handoff"]) is not bool:
        raise ValueError(f"{label}: in_handoff must be true or false")
    _validate_applied(record, label)


def validate_registry(registry: dict, tasks: dict) -> None:
    """Shape and catalog checks; checks against the live views happen in plan()."""
    if not isinstance(registry, dict) or registry.get("schema") != SCHEMA:
        raise ValueError(f"registry schema must be {SCHEMA}")
    if set(registry) != TOP_LEVEL:
        raise ValueError(f"registry needs exactly {sorted(TOP_LEVEL)}")
    if not isinstance(registry["decided_on"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", registry["decided_on"]):
        raise ValueError("decided_on must be a YYYY-MM-DD date")
    if not isinstance(registry["evidence_snapshot"], str) or not re.fullmatch(r"[0-9a-f]{40}", registry["evidence_snapshot"]):
        raise ValueError("evidence_snapshot must be a full commit id")
    explanations = registry["explanations"]
    if (not isinstance(explanations, str) or Path(explanations).suffix != ".md" or Path(explanations).is_absolute()
            or ".." in Path(explanations).parts or "#" in explanations or ":" in explanations
            or explanations.startswith("~")):
        raise ValueError("explanations must be a repo-relative Markdown path")
    for key in ("qualification_input", "historical_verdicts_regraded", "training_launched"):
        if registry[key] is not False:
            raise ValueError(f"{key} must be false: a disposition never qualifies, regrades or trains")
    if not isinstance(registry["views"], dict) or not registry["views"]:
        raise ValueError("registry views must name each revised view")
    for view_id, entry in registry["views"].items():
        _validate_view_entry(view_id, entry)
    records = registry["dispositions"]
    if not isinstance(records, list) or not records:
        raise ValueError("registry dispositions must be a nonempty list")
    seen = set()
    for record in records:
        _validate_record(record, registry, tasks)
        if record["disposition"] in {"HOLD", "REMOVE", "KEEP"}:
            if record["task"] in seen:
                raise ValueError(f"{record['task']}: one disposition per test")
            seen.add(record["task"])
            for view_id in record["views"]:
                entry = registry["views"].get(view_id)
                if record["disposition"] in VIEW_ROLES and _applied(record) and (entry is None or not _applied(entry)):
                    raise ValueError(f"{record['task']}: applied disposition names view {view_id} without an applied "
                                     "registry view entry")
    for view_id, entry in registry["views"].items():
        targeted = any(r["disposition"] in VIEW_ROLES and _applied(r) and view_id in r["views"] for r in records)
        if _applied(entry) and not targeted:
            raise ValueError(f"registry view {view_id} has no applied disposition; mark it applied false or drop it")


def load_registry(path: Path | str, tasks: dict) -> dict:
    registry = json.loads(Path(path).read_text())
    validate_registry(registry, tasks)
    return registry


def _anchors(path: Path) -> set:
    return set(ANCHOR.findall(path.read_text())) if path.is_file() else set()


def unresolved_explanations(root: Path | str, views) -> list[str]:
    """Hold, stretch and removal links whose Markdown file or explicit <a name> anchor is missing under root."""
    problems = []
    for view in views:
        for assignment in view["assignments"]:
            note = assignment.get("hold") or assignment.get("stretch") or assignment.get("removal")
            if not note or "explanation" not in note:
                continue
            path, _, anchor = note["explanation"].partition("#")
            if anchor not in _anchors(Path(root) / path):
                problems.append(f"{view['id']}: {assignment['task']} explanation {note['explanation']} needs an "
                                f'explicit <a name="{anchor}"></a> anchor in {path}')
    return problems


def _revise(base: dict, view_id: str, entry: dict, records: list, tasks: dict, style: str) -> dict:
    """The registry's revision of one archived base view; validated before return."""
    view = deepcopy(base)
    assigned = {assignment["task"]: assignment for assignment in view["assignments"]}
    for record in records:
        tier = record["views"][view_id]
        assignment = assigned.get(record["task"])
        if assignment is None:
            raise ValueError(f"{view_id}: {record['task']} is not assigned to this view")
        if assignment["importance"] != "required" or assignment["qualification_tier"] != tier:
            raise ValueError(f"{view_id}: {record['task']} must be a required Tier {tier} assignment "
                             f"in revision {entry['from_revision']}")
        assignment["importance"] = VIEW_ROLES[record["disposition"]]
    held = {assignment["task"] for assignment in view["assignments"] if assignment["importance"] == "held"}
    for record in records:
        assignment = assigned[record["task"]]
        if record["disposition"] == "HOLD":
            blocked = sorted(dep["task"] for dep in tasks[record["task"]]["dependencies"] if dep["task"] in held)
            note = {"schema": HOLD_SCHEMA, "previous_importance": "required", "reason": record["hold"]["reason"],
                    "blocked_by": blocked, "summary": record["hold"]["summary"],
                    "explanation": record["explanation"], "reentry": record["hold"]["reentry"]}
            _put(assignment, "hold", _ordered(note, style), style)
        else:
            note = {"schema": REMOVAL_SCHEMA, "previous_importance": "required",
                    "reason": record["removal"]["reason"], "counted_by": record["removal"]["counted_by"],
                    "summary": record["removal"]["summary"], "explanation": record["explanation"]}
            _put(assignment, "removal", _ordered(note, style), style)
    _put(view, "revision", entry["to_revision"], style)
    _put(view, "previous_revision", _ordered({"revision": entry["from_revision"], "archive": entry["archive"],
                                              "policy_fingerprint": view_fingerprint(base)}, style), style)
    _put(view, "policy_change_reason", entry["policy_change_reason"], style)
    try:
        validate_view(view, tasks)
    except ValueError as error:
        raise ValueError(f"{view_id} revision {entry['to_revision']} would be invalid: {error}") from error
    return view


def _differences(expected: dict, live: dict) -> list[str]:
    result = [f"field {key}" for key in sorted(set(expected) | set(live))
              if key != "assignments" and expected.get(key) != live.get(key)]
    old = {a.get("task"): a for a in expected.get("assignments", [])}
    new = {a.get("task"): a for a in live.get("assignments", [])}
    result += [f"assignment {name}" for name in sorted(set(old) | set(new), key=str) if old.get(name) != new.get(name)]
    return result


def _tier_counts(view: dict) -> dict:
    highest = max((a["qualification_tier"] for a in view.get("assignments", [])), default=1)
    def count(match):
        return [sum(match(a) and a["qualification_tier"] == tier for a in view["assignments"]) for tier in range(1, highest + 1)]
    return {"required": count(lambda a: a["importance"] == "required"),
            "held": count(lambda a: a["importance"] == "held"),
            "removed": count(lambda a: "removal" in a)}


def plan(root: Path | str, registry_path: Path | str) -> dict:
    """Read-only comparison of the registry with the live views and archives."""
    root = Path(root)
    tasks = load_tasks(root)
    registry = load_registry(registry_path, tasks)
    records = registry["dispositions"]
    applied = [r for r in records if r["disposition"] in VIEW_ROLES and _applied(r)]
    view_paths = {path.stem: path for path in sorted((root / "configs/forge/views").glob("*.json"))}
    live_views = {view_id: json.loads(path.read_text()) for view_id, path in view_paths.items()}
    explanation = root / registry["explanations"]
    anchors = _anchors(explanation)
    for record in applied:
        anchor = record["explanation"].split("#")[1]
        if anchor not in anchors:
            raise ValueError(f"{record['task']}: explanation {record['explanation']} needs an explicit "
                             f'<a name="{anchor}"></a> anchor in {registry["explanations"]}')
    for record in records:
        if record["disposition"] not in {"HOLD", "REMOVE", "KEEP"} or not _applied(record):
            continue
        for view_id, view in live_views.items():
            for assignment in view["assignments"]:
                if (assignment["task"] == record["task"] and assignment["importance"] == "required"
                        and view_id not in record["views"]):
                    raise ValueError(f"{record['task']} is counted in view {view_id}, which its disposition does not "
                                     "list; one disposition applies in every view that counts a test")
        if record["disposition"] == "KEEP":
            for view_id, tier in record["views"].items():
                assignment = next((a for a in live_views.get(view_id, {}).get("assignments", [])
                                   if a["task"] == record["task"]), None)
                if not assignment or assignment["importance"] != "required" or assignment["qualification_tier"] != tier:
                    raise ValueError(f"{record['task']}: KEEP names view {view_id} where it is not a required "
                                     f"Tier {tier} assignment")
    result = {}
    for view_id, entry in registry["views"].items():
        if not _applied(entry):
            continue
        if view_id not in view_paths:
            raise ValueError(f"registry view {view_id} is not declared in configs/forge/views")
        here = [r for r in applied if view_id in r["views"]]
        live_bytes = view_paths[view_id].read_bytes()
        raw = live_bytes.decode("utf-8")
        live = live_views[view_id]
        archive = root / entry["archive"]
        item = {"from_revision": entry["from_revision"], "to_revision": entry["to_revision"],
                "revision": live.get("revision"), "archive": entry["archive"],
                "held": sorted(r["task"] for r in here if r["disposition"] == "HOLD"),
                "removed": sorted(r["task"] for r in here if r["disposition"] == "REMOVE"), "differences": []}
        if live.get("revision") == entry["from_revision"]:
            try:
                if view_fingerprint(live) != entry["from_fingerprint"]:
                    raise ValueError(f"live {view_id} revision {entry['from_revision']} is not the registry's base "
                                     "revision (its policy fingerprint differs from from_fingerprint)")
                item["archive_action"] = _archive_state(root, view_id, entry["from_revision"], live_bytes)
            except ValueError as error:
                item.update(state="conflict", differences=[str(error)])
            else:
                revised = _revise(live, view_id, entry, here, tasks, key_style(raw))
                item.update(state="pending", counts_before=_tier_counts(live), counts_after=_tier_counts(revised),
                            differences=[f"assignment {name}" for name in item["held"] + item["removed"]],
                            _write=(view_paths[view_id], dump_view(revised, raw).encode("utf-8"), live_bytes))
                live_views[view_id] = revised
        elif (live.get("revision") == entry["to_revision"]
              or (type(live.get("revision")) is int and live["revision"] > entry["to_revision"])):
            if live["revision"] > entry["to_revision"]:
                # A later revision supersedes the disposition revision. Compare the
                # disposition revision itself, reached through the archive chain.
                superseded_by = live["revision"]
                try:
                    live = _archived_revision(root, view_id, live, entry["to_revision"])
                except ValueError as error:
                    item.update(state="drift", differences=[str(error)])
                    result[view_id] = item
                    continue
                item["superseded_by"] = superseded_by
            if not archive.is_file():
                item.update(state="drift", differences=[f"archive {entry['archive']} is missing"])
            else:
                base = json.loads(archive.read_text())
                if base.get("id") != view_id or base.get("revision") != entry["from_revision"]:
                    item.update(state="drift", differences=[f"archive {entry['archive']} is not revision "
                                                            f"{entry['from_revision']} of {view_id}"])
                elif view_fingerprint(base) != entry["from_fingerprint"]:
                    item.update(state="drift", differences=[f"archive {entry['archive']} is not the registry's base "
                                                            "revision (its policy fingerprint differs from "
                                                            "from_fingerprint)"])
                else:
                    expected = _revise(base, view_id, entry, here, tasks, key_style(raw))
                    differences = _differences(expected, live)
                    item.update(state="drift" if differences else "applied", differences=differences,
                                counts_before=_tier_counts(base), counts_after=_tier_counts(live))
        else:
            item.update(state="conflict", differences=[f"live revision {live.get('revision')} is neither "
                                                       f"{entry['from_revision']} nor {entry['to_revision']}"])
        result[view_id] = item
    # Every declared view must validate with the planned revisions in place.
    for view_id, view in live_views.items():
        if view.get("id") != view_id:
            raise ValueError(f"{view_id}: view filename and id differ")
        validate_view(view, tasks)
    skipped = {"not_applied": sorted(r["task"] for r in records if r["disposition"] in VIEW_ROLES and not _applied(r)),
               "views_not_applied": sorted(v for v, entry in registry["views"].items() if not _applied(entry)),
               "keep": sorted(r["task"] for r in records if r["disposition"] == "KEEP"),
               "no_change": sorted(r["task"] for r in records if r["disposition"] == "NO_CHANGE"),
               "fix": sorted(r["task"] for r in records if r["disposition"] == "FIX")}
    return {"schema": SCHEMA, "views": result, "skipped": skipped}


def _report(root, registry_path, planned, **extra):
    views = {name: {key: value for key, value in item.items() if not key.startswith("_")}
             for name, item in planned["views"].items()}
    try:
        registry = Path(registry_path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        registry = str(registry_path)
    return {"registry": registry, "schema": planned["schema"],
            "drift": any(item["state"] != "applied" for item in views.values()),
            "views": views, "skipped": planned["skipped"], **extra,
            "training_launched": False, "qualification_changed": False}


def _archived_revision(root: Path, view_id: str, view: dict, revision: int) -> dict:
    """Follow previous_revision archives (each pinned by fingerprint) back to ``revision``."""
    current = view
    while current.get("revision") != revision:
        previous = current.get("previous_revision")
        if not isinstance(previous, dict) or previous.get("archive") != archive_path(view_id, previous.get("revision")):
            raise ValueError(f"{view_id} revision {current.get('revision')} has no archive chain to revision {revision}")
        path = root / previous["archive"]
        if not path.is_file():
            raise ValueError(f"archive {previous['archive']} is missing")
        archived = json.loads(path.read_text())
        if (archived.get("id") != view_id or archived.get("revision") != previous["revision"]
                or view_fingerprint(archived) != previous["policy_fingerprint"]):
            raise ValueError(f"archive {previous['archive']} does not match its previous_revision fingerprint")
        if archived["revision"] < revision:
            raise ValueError(f"{view_id} archive chain skips revision {revision}")
        current = archived
    return current


def check(root: Path | str, registry_path: Path | str) -> dict:
    """Dry run: report each view as applied, pending, drift or conflict; write nothing."""
    return _report(root, registry_path, plan(root, registry_path))


def apply(root: Path | str, registry_path: Path | str) -> dict:
    """Archive first, validate every view, then write the pending revisions atomically."""
    root = Path(root)
    planned = plan(root, registry_path)
    blocked = {name: item["differences"] for name, item in planned["views"].items()
               if item["state"] in {"drift", "conflict"}}
    if blocked:
        raise ValueError("views differ from the registry; nothing was written: " + json.dumps(blocked, sort_keys=True))
    pending = {name: item for name, item in planned["views"].items() if item["state"] == "pending"}
    archived, written = [], []
    for name, item in sorted(pending.items()):
        if item["archive_action"] == "create":
            _atomic_bytes(root / item["archive"], item["_write"][2])
            archived.append(item["archive"])
    for name, item in sorted(pending.items()):
        path, data, _ = item["_write"]
        _atomic_bytes(path, data)
        written.append(path.relative_to(root).as_posix())
    after = plan(root, registry_path)
    if any(item["state"] != "applied" for item in after["views"].values()):
        raise ValueError("applied views do not match the registry on re-read")
    return _report(root, registry_path, after, archived=archived, written=written,
                   applied=sorted(pending))
