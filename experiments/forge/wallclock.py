"""Separate wall-clock limit hits from recipe verdicts; contention-aware limits by default.

A wall-clock limit (`resources.timeout_seconds`) is a safety cap on shared
hardware, not part of a gate. A run killed by it was never graded, so it is
reported as NOT MEASURED (wall-clock): kept out of fail totals and out of paired
recipe-versus-baseline deltas, together with any task it blocks downstream.

* ``outcome`` / ``tally`` / ``paired_delta`` classify saved attempt records.
* ``DEFAULT_POLICY`` is the default operational limit policy (user decision
  2026-10-16, "increase training": tests end on their declared update budget,
  not on a timer). For each listed task it sets

  - ``minimum_limit_seconds``: the operational limit used instead of a smaller
    declared ``timeout_seconds`` (from measured per-update times with a safety
    factor; the declared value stays the task's recorded budget contract, so
    task identities and archived receipts are unchanged), and
  - ``solo_seconds_per_update``: a solo reference for the contention-aware
    extension. At the (raised) limit the supervisor measures seconds per update
    so far, divides by the solo reference, and extends the limit by that
    slowdown factor, capped at ``ceiling_factor``. No measured progress means no
    extension (a run without a single observation is a defect, not contention),
    and a run whose measured rate cannot finish its declared updates within the
    extended limit is not extended either (it stays NOT MEASURED).

  Pass ``"declared"`` (or set ``FORGE_WALLCLOCK_POLICY=declared``) to enforce the
  declared limit exactly as before.

This module has no package imports so that ``attempt_worker.py`` can load it by
file path while a frozen source snapshot supplies every scientific module.

    python -m experiments.forge.wallclock report [--output-dir DIR]
    python -m experiments.forge.wallclock limits     # print the default operational limits
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

NOT_MEASURED = "NOT_MEASURED_WALLCLOCK"
LABEL = "NOT MEASURED (wall-clock)"
VERDICTS = {"PASS", "FAIL"}
DEFAULT_CEILING = 3.0
RERUNS = Path("reports/forge/wallclock-reruns.json")
REGISTRY = Path("reports/forge/attempt-coverage.json")
POLICY_ENV = "FORGE_WALLCLOCK_POLICY"
PROJECTION_MARGIN = 1.10

_WORD_EVIDENCE = ("reports/forge/dig2-20261016/wallclock-reruns.md (solo reruns under the lease: 0.0390-0.0406 s/update; "
                  "slowest measured word recipe 0.0686 s/update on a shared GPU) and reports/forge/dig-20261015/hygiene.md "
                  "(longest completed word smoke 847 s; recommended about 1,800 s)")
_GRID_EVIDENCE = ("reports/forge/attempt-coverage.json: completed 7,000-update runs take 326-2,133 s (median 0.076-0.091 "
                  "s/update); TIMEOUT hits at 3,600 s reached 6,500-6,750 updates (0.53 s/update) or 1,750-3,500 updates")
# Default operational limits. Update budgets are never changed here.
DEFAULT_POLICY = {
    "schema": "forge_wallclock_policy_v2",
    "ceiling_factor": DEFAULT_CEILING,
    "projection_margin": PROJECTION_MARGIN,
    "tasks": {
        "five_word_joint_smoke": {"updates": 20001, "declared_seconds": 900, "minimum_limit_seconds": 1800,
                                  "solo_seconds_per_update": 0.0398, "evidence": _WORD_EVIDENCE},
        # The hold continues the same word host for 4,000 updates; the A1 rerun hit 300 s at 3,667 of 4,000
        # updates (0.078 s/update on a shared GPU). 4,000 x 0.078 x 1.9 = about 600 s.
        "five_word_joint_hold": {"updates": 4000, "declared_seconds": 300, "minimum_limit_seconds": 600,
                                 "solo_seconds_per_update": 0.0398, "evidence": _WORD_EVIDENCE},
        # Counted ring16 (6,400 updates, declared 1,200 s): slowest measured 233 s (BISPF), median 0.0125 s/update.
        "ring16_acquisition_6400": {"updates": 6400, "declared_seconds": 1200, "minimum_limit_seconds": 1200,
                                    "solo_seconds_per_update": 0.0125,
                                    "evidence": "reports/forge/dig2-20261016/ring16-cap-runs-main.json and -runs-d1.json"},
        # Archived 1,600-update ring16: slowest completed 228 of 300 s; the TIMEOUT hit ran at 0.19 s/update shared.
        "ring16_acquisition": {"updates": 1600, "declared_seconds": 300, "minimum_limit_seconds": 600,
                               "solo_seconds_per_update": 0.0182,
                               "evidence": "reports/forge/dig2-20261016/wallclock-reruns.md cell 7 (33 s solo)"},
        "grid100": {"updates": 7000, "declared_seconds": 3600, "minimum_limit_seconds": 7200,
                    "solo_seconds_per_update": 0.0914, "evidence": _GRID_EVIDENCE},
        "rotated100": {"updates": 7000, "declared_seconds": 3600, "minimum_limit_seconds": 7200,
                       "solo_seconds_per_update": 0.0762, "evidence": _GRID_EVIDENCE},
        "staggered100": {"updates": 7000, "declared_seconds": 3600, "minimum_limit_seconds": 7200,
                         "solo_seconds_per_update": 0.0879, "evidence": _GRID_EVIDENCE},
    },
}


def is_wallclock_hit(record):
    """TIMEOUT, or INCOMPLETE whose raw terminal status is a timeout at or past the limit."""
    status = record.get("status")
    if status == "TIMEOUT":
        return True
    if status != "INCOMPLETE":
        return False
    raw = record.get("raw_terminal_status")
    seconds, limit = record.get("seconds"), record.get("timeout_seconds")
    return raw == "timeout" and (seconds is None or limit is None or seconds >= limit)


def outcome(record):
    return NOT_MEASURED if is_wallclock_hit(record) else record.get("status")


def blocked_by_wallclock(record):
    """A BLOCKED verdict whose only cause is an ungraded (TIMEOUT/INCOMPLETE) checkpoint parent."""
    reason = str(record.get("reason", ""))
    return record.get("status") == "BLOCKED" and "checkpoint parent" in reason and (
        "TIMEOUT" in reason or "INCOMPLETE" in reason)


def tally(records):
    """Outcome counts; wall-clock hits and their downstream blocks are never FAIL."""
    counts = Counter()
    for record in records:
        value = outcome(record)
        if value != NOT_MEASURED and blocked_by_wallclock(record):
            value = NOT_MEASURED + "_DOWNSTREAM"
        counts[value] += 1
    return {"passed": counts.get("PASS", 0), "failed": counts.get("FAIL", 0),
            "not_measured_wallclock": counts.get(NOT_MEASURED, 0),
            "not_measured_downstream": counts.get(NOT_MEASURED + "_DOWNSTREAM", 0),
            "counts": dict(sorted(counts.items()))}


def paired_delta(recipe, baseline):
    """PASS-count delta over tasks where both arms have a real verdict.

    ``recipe`` and ``baseline`` map task -> attempt record (or a bare status
    string). Tasks where either arm is NOT MEASURED, blocked by an ungraded
    parent, or otherwise lacks a PASS/FAIL verdict are excluded and listed.
    """
    def status(value):
        record = value if isinstance(value, dict) else {"status": value}
        if outcome(record) == NOT_MEASURED or blocked_by_wallclock(record):
            return NOT_MEASURED
        return record.get("status")
    compared, excluded = [], {}
    for task in sorted(set(recipe) | set(baseline)):
        left, right = status(recipe.get(task)), status(baseline.get(task))
        if left in VERDICTS and right in VERDICTS:
            compared.append((task, left, right))
        else:
            excluded[task] = {"recipe": left, "baseline": right}
    delta = sum((l == "PASS") - (r == "PASS") for _, l, r in compared)
    return {"delta": delta, "compared": len(compared), "excluded": excluded,
            "tasks": {t: {"recipe": l, "baseline": r} for t, l, r in compared}}


def effective_limit(declared, elapsed, progressed_updates, solo_seconds_per_update, ceiling_factor=DEFAULT_CEILING):
    """Contention-aware limit decided once, at the declared limit.

    Returns ``(limit_seconds, slowdown_factor)``. The factor is measured seconds
    per update divided by the solo reference; the limit is
    ``declared * clamp(factor, 1, ceiling_factor)``.
    """
    if not (declared > 0 and ceiling_factor >= 1):
        raise ValueError("declared limit must be positive and the ceiling at least 1")
    if not progressed_updates or progressed_updates <= 0 or not solo_seconds_per_update or solo_seconds_per_update <= 0:
        return float(declared), None
    factor = (elapsed / progressed_updates) / solo_seconds_per_update
    if not math.isfinite(factor):
        return float(declared), None
    return float(declared) * min(max(factor, 1.0), float(ceiling_factor)), factor


def logged_progress(log_path):
    """Largest training step in saved observation lines of a run log (optionally timestamp-prefixed).

    A continuation (for example the word hold) logs its own updates as
    ``continuation_offset``; that, not the inherited parent step, is its progress.
    """
    best = 0
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        text = line.rsplit("\t", 1)[-1].strip()
        if not text.startswith("{"):
            continue
        try:
            event = json.loads(text)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("event") != "observation":
            continue
        step = event.get("continuation_offset", event.get("step"))
        if isinstance(step, (int, float)) and not isinstance(step, bool):
            best = max(best, int(step))
    return best


def resolve_policy(value="default"):
    """``"default"`` (or missing) -> DEFAULT_POLICY; ``None``/``"declared"`` -> declared limits only.

    The environment variable ``FORGE_WALLCLOCK_POLICY=declared`` turns the default off.
    """
    if value == "default" and os.environ.get(POLICY_ENV) == "declared":
        return None
    if value in (None, "declared", False, {}):
        return None
    if value == "default":
        return DEFAULT_POLICY
    if not isinstance(value, dict):
        raise ValueError("wall-clock policy must be 'default', 'declared' or a policy document")
    return value


def task_policy(policy, task):
    """This task's limit entry, or None. Accepts v2 documents and the v1 opt-in form

    ``{"solo_seconds_per_update": {task: s}, "ceiling_factor": c}``.
    """
    policy = resolve_policy(policy) if not isinstance(policy, dict) else policy
    if not policy:
        return None
    ceiling = float(policy.get("ceiling_factor", DEFAULT_CEILING))
    if "tasks" in policy:
        entry = policy["tasks"].get(task)
        if entry is None:
            return None
        value = {"ceiling_factor": ceiling, "projection_margin": float(policy.get("projection_margin", PROJECTION_MARGIN)),
                 **entry}
    else:
        reference = policy.get("solo_seconds_per_update", {}).get(task)
        if reference is None:
            return None
        value = {"solo_seconds_per_update": reference, "ceiling_factor": ceiling}
    value["ceiling_factor"] = float(value["ceiling_factor"])
    if not value["ceiling_factor"] >= 1:
        raise ValueError("wall-clock ceiling factor must be at least 1")
    return value


def policy_for(policy, task):
    """Resolve a policy to this task's ``(solo_seconds_per_update, ceiling)`` or None."""
    entry = task_policy(policy, task) if policy else None
    if not entry or entry.get("solo_seconds_per_update") is None:
        return None
    return float(entry["solo_seconds_per_update"]), entry["ceiling_factor"]


def base_limit(declared, entry):
    """Operational limit before any contention extension: never below the declared limit."""
    if not entry:
        return float(declared)
    return float(max(declared, entry.get("minimum_limit_seconds") or declared))


def adapter_allowance(declared, entry):
    """Hard ceiling for an adapter's own in-process deadline."""
    return base_limit(declared, entry) * (entry["ceiling_factor"] if entry else 1.0)


def decide_extension(declared, elapsed, progressed_updates, entry):
    """Decide once, at the base limit, whether and how far to extend.

    Returns ``{base_limit, effective_limit, slowdown_factor, decision, projected_seconds}``.
    """
    base = base_limit(declared, entry)
    value = {"base_limit": base, "effective_limit": base, "slowdown_factor": None,
             "projected_seconds": None, "decision": "declared_only"}
    if not entry or not entry.get("solo_seconds_per_update"):
        return value
    limit, factor = effective_limit(base, elapsed, progressed_updates, float(entry["solo_seconds_per_update"]),
                                    entry["ceiling_factor"])
    value["slowdown_factor"] = factor
    if factor is None:
        value["decision"] = "no_progress"
        return value
    updates = entry.get("updates")
    if updates:
        projected = elapsed / progressed_updates * updates * float(entry.get("projection_margin", PROJECTION_MARGIN))
        value["projected_seconds"] = projected
        if projected > limit:
            value["decision"] = "cannot_finish_within_ceiling"
            return value
    value.update(effective_limit=limit, decision="extended" if limit > base else "not_needed")
    return value


def load_reruns(root):
    """Validated rerun overlay: real verdicts that supersede matching wall-clock hits (by exact attempt key)."""
    path = Path(root) / RERUNS
    if not path.is_file():
        return {}
    document = json.loads(path.read_text())
    if (document.get("schema") != "forge_wallclock_reruns_v1" or document.get("qualification_input") is not False
            or document.get("formulations_modified") is not False):
        raise ValueError("wall-clock reruns must remain unchanged-formulation diagnostics")
    reruns = {}
    for record in document["records"]:
        if record["key"] in reruns:
            raise ValueError("duplicate wall-clock rerun key")
        reruns[record["key"]] = record
    return reruns


def apply_reruns(records, reruns):
    """Replace a wall-clock hit with its same-key rerun only when the rerun has a real verdict.

    A task BLOCKED only because its checkpoint parent was a wall-clock hit is
    treated the same way: its rerun continues from the rerun parent's checkpoint.
    """
    out = []
    for record in records:
        rerun = reruns.get(record.get("key"))
        # A retained receipt may keep its raw terminal status only in its evidence
        # file; the overlay then quotes it (``original_hit``) for the same check.
        hit = rerun and (is_wallclock_hit({**record, **rerun.get("original_hit", {}), "status": record.get("status")})
                         or blocked_by_wallclock(record))
        if hit and rerun["status"] in VERDICTS:
            merged = {**record, "status": rerun["status"], "reason": rerun.get("reason", ""),
                      "seconds": rerun.get("seconds"), "updates": rerun.get("updates"),
                      "wallclock_rerun": {"attempt": rerun["attempt"], "evidence": str(RERUNS),
                                          "superseded_status": record["status"]}}
            out.append(merged)
        else:
            out.append(record)
    return out


def report(root, output_dir):
    root = Path(root)
    records = json.loads((root / REGISTRY).read_text())["records"]
    reruns = load_reruns(root)
    records = apply_reruns(records, reruns)
    by_recipe = {}
    for record in records:
        by_recipe.setdefault(record.get("label", record["candidate_id"]), []).append(record)
    rows = {label: tally(items) for label, items in sorted(by_recipe.items())}
    for record in records:
        quoted = (reruns.get(record.get("key")) or {}).get("original_hit")
        if quoted and not record.get("wallclock_rerun"):
            record.update({k: v for k, v in quoted.items() if k != "evidence"})
    hits = [r for r in records if is_wallclock_hit(r)]
    value = {"schema": "forge_wallclock_outcomes_v1", "qualification_input": False,
             "label": LABEL, "registry": str(REGISTRY), "reruns": str(RERUNS) if reruns else None,
             "summary": tally(records), "recipes": rows,
             "not_measured": [{k: r.get(k) for k in ("label", "task", "status", "seconds", "timeout_seconds", "key")}
                              for r in sorted(hits, key=lambda r: (r.get("label", ""), r["task"]))],
             "rerun_verdicts": [{k: r.get(k) for k in ("label", "task", "status", "seconds", "wallclock_rerun")}
                                for r in records if r.get("wallclock_rerun")]}
    output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wallclock-outcomes.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    s = value["summary"]
    lines = ["# Wall-clock outcomes", "",
             f"A run killed by its task's wall-clock limit was never graded. It is reported as **{LABEL}**, "
             "not FAIL, and is excluded (with any task it blocks downstream) from paired recipe-versus-baseline deltas. "
             "A same-key solo rerun under the one-job-per-GPU lease replaces the hit only when it produced a real "
             "PASS/FAIL verdict under the unchanged declared limit.", "",
             f"Totals: {s['passed']} PASS, {s['failed']} FAIL, {s['not_measured_wallclock']} not measured (wall-clock), "
             f"{s['not_measured_downstream']} blocked by an unmeasured parent.", "",
             "| Recipe | PASS | FAIL | Not measured (wall-clock) | Blocked by unmeasured parent |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, row in rows.items():
        if row["not_measured_wallclock"] or row["not_measured_downstream"] or any(
                r.get("wallclock_rerun") for r in by_recipe[label]):
            lines.append(f"| {label} | {row['passed']} | {row['failed']} | {row['not_measured_wallclock']} | "
                         f"{row['not_measured_downstream']} |")
    if value["rerun_verdicts"]:
        lines += ["", "## Real verdicts from solo reruns", "", "| Recipe | Task | Verdict | Seconds | Superseded |",
                  "| --- | --- | --- | ---: | --- |"]
        for r in value["rerun_verdicts"]:
            lines.append(f"| {r['label']} | {r['task']} | {r['status']} | {r['seconds']:.1f} | "
                         f"{r['wallclock_rerun']['superseded_status']} |")
    (output_dir / "WALLCLOCK_OUTCOMES.md").write_text("\n".join(lines) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("report", "limits"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/forge"))
    args = parser.parse_args()
    if args.action == "limits":
        for task, entry in sorted(DEFAULT_POLICY["tasks"].items()):
            print(f"{task}: declared {entry['declared_seconds']} s -> operational {entry['minimum_limit_seconds']} s, "
                  f"extension up to {entry['minimum_limit_seconds'] * DEFAULT_POLICY['ceiling_factor']:.0f} s "
                  f"({entry['updates']} updates, solo {entry['solo_seconds_per_update']} s/update)")
        return
    value = report(args.root, args.output_dir)
    print(json.dumps(value["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
