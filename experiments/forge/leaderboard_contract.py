"""Fixed leaderboard denominators, declared tuning budgets and holdout claims.

Presentation contracts shared by the technique inventory, configuration-search
reports and readouts. Nothing here grades, trains or changes a recorded verdict.

* A **fixed, dated test list** is the leaderboard denominator. Every listed
  test counts in every row as PASS, FAIL, not measured or another outcome,
  including tests that a view currently holds or removed from its count. The
  denominator never shrinks silently: a test leaves only through a new dated
  list, which keeps the old one as history.
* A row's **tuning budget** is what was actually declared: the registered
  configuration searches (and their grid sizes) for its family, and whether the
  selected recipe descends from one of them. A family with no declared search
  is labelled untuned.
* A configuration selected on tuning tiers carries a **holdout claim**: whether
  it passed the higher (untuned) tiers. Tuning-tier passes are selection
  evidence and are never presented as independent confirmation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .contracts import file_hash, read_json

FIXED_TEST_LIST = Path("configs/forge/leaderboards/technique-inventory-full-suite-2026-11-12.json")
FIXED_TEST_LIST_SCHEMA = "forge_fixed_test_list_v1"
NOT_MEASURED = frozenset({"UNKNOWN", "NOT_RUN"})
# Only a recorded FAIL is a scientific failure, matching ``fixed_outcomes``
# (INVALID and other non-PASS outcomes are "other": no pass credit, not a failure).
FAILED = frozenset({"FAIL"})
INDEPENDENT_CONFIRMATION = "not_performed"


def load_fixed_test_list(root, relative=FIXED_TEST_LIST) -> dict | None:
    """The registered dated denominator, or None when the checkout has none."""
    path = Path(root) / relative
    if not path.is_file():
        return None
    data = read_json(path)
    tests = data.get("tests")
    if (data.get("schema") != FIXED_TEST_LIST_SCHEMA or not isinstance(data.get("dated"), str)
            or not isinstance(tests, list) or not tests or len(set(tests)) != len(tests)
            or not all(isinstance(name, str) and name for name in tests)):
        raise ValueError(f"malformed fixed leaderboard test list: {relative}")
    return {**data, "path": Path(relative).as_posix(), "sha256": file_hash(path)}


def fixed_outcomes(statuses: dict, tests, not_counted) -> dict:
    """Count a row's recorded statuses against a fixed denominator.

    ``statuses`` maps test -> recorded status; a missing test is not measured.
    ``not_counted`` names listed tests that current views hold, removed from
    their count or no longer assign; they stay in the denominator and are also
    broken out in ``held`` so readers see what qualification totals exclude.
    """
    tests = list(tests)
    status = {name: statuses.get(name, "UNKNOWN") for name in tests}
    counts = Counter(status.values())
    held = [name for name in tests if name in set(not_counted)]
    held_counts = Counter(status[name] for name in held)
    other = {key: value for key, value in sorted(counts.items())
             if key not in {"PASS", "FAIL"} and key not in NOT_MEASURED}
    return {"total": len(tests), "passed": counts.get("PASS", 0), "failed": counts.get("FAIL", 0),
            "not_measured": sum(counts.get(key, 0) for key in NOT_MEASURED), "other": other,
            "held": {"total": len(held), "passed": held_counts.get("PASS", 0),
                     "failed": held_counts.get("FAIL", 0),
                     "not_measured": sum(held_counts.get(key, 0) for key in NOT_MEASURED)}}


def _grid_size(grid) -> int:
    """Declared configuration count of a search grid (union or Cartesian product)."""
    if isinstance(grid, list):
        return sum(_grid_size(part) for part in grid)
    if not isinstance(grid, dict) or not grid:
        return 0
    size = 1
    for values in grid.values():
        size *= len(values) if isinstance(values, list) else 1
    return size


def _family_ids(root, family_id: str) -> set[str]:
    path = Path(root) / "configs/forge/trainer-families.json"
    ids = {family_id}
    if not path.is_file():
        return ids
    registry = read_json(path)
    for family in registry.get("families", []) + registry.get("historical_families", []):
        if family.get("id") == family_id:
            ids.update(family.get("historical_family_ids", []))
        if family.get("reporting_family") == family_id:
            ids.add(family["id"])
    return ids


def family_search_budget(root, family_id: str) -> dict:
    """Registered configuration searches declared for a reporting family."""
    ids = _family_ids(root, family_id)
    searches = []
    for path in sorted((Path(root) / "configs/forge/searches").glob("*.json")):
        spec = read_json(path)
        if spec.get("trainer_family") in ids:
            searches.append({"id": spec.get("id", path.stem), "configurations": _grid_size(spec.get("grid")),
                             "tuning_through_tier": spec.get("tuning_through_tier")})
    cards = set()
    registry_path = Path(root) / "configs/forge/trainer-families.json"
    if registry_path.is_file():
        registry = read_json(registry_path)
        for family in registry.get("families", []) + registry.get("historical_families", []):
            if family.get("id") in ids:
                cards.update(family.get("candidates", []))
    for directory in ("ideas", "configurations"):
        for path in (Path(root) / "configs/forge" / directory).glob("*.json"):
            if read_json(path).get("trainer_family") in ids:
                cards.add(path.stem)
    return {"family": family_id, "searches": len(searches),
            "configurations": sum(item["configurations"] for item in searches),
            "search_ids": [s["id"] for s in searches], "declared_candidates": len(cards)}


def _declaration(root, candidate_id):
    for directory in ("configurations", "ideas"):
        path = Path(root) / "configs/forge" / directory / f"{candidate_id}.json"
        if path.is_file():
            return read_json(path)
    return None


def selection_lineage(root, candidate) -> dict:
    """Nearest registered configuration search in a declaration's parent lineage.

    ``candidate`` is a candidate id or a declaration. Returns the search id,
    the configuration in the lineage that it produced and its tuning tier cap,
    or ``{"search_id": None}`` when no ancestor came from a registered search.
    """
    current = _declaration(root, candidate) if isinstance(candidate, str) else candidate
    seen = set()
    while isinstance(current, dict) and current.get("id") not in seen:
        seen.add(current.get("id"))
        search_id = current.get("search_study_id")
        if search_id:
            spec_path = Path(root) / "configs/forge/searches" / f"{search_id}.json"
            spec = read_json(spec_path) if spec_path.is_file() else {}
            return {"search_id": search_id, "configuration": current.get("id"),
                    "tuning_through_tier": spec.get("tuning_through_tier"),
                    "configurations": _grid_size(spec.get("grid")) if spec else None}
        parent = current.get("parent")
        current = _declaration(root, parent) if isinstance(parent, str) else None
    return {"search_id": None}


def holdout_claim(tier_statuses: dict, tuning_through_tier) -> dict:
    """State whether a tuned selection passed its untuned holdout tiers.

    ``tier_statuses`` maps tier -> list of required-task statuses. Tiers at or
    below ``tuning_through_tier`` selected the configuration; higher tiers are
    the holdout. A tuning-tier PASS is never independent confirmation.
    """
    if not isinstance(tuning_through_tier, int):
        return {"status": "not_applicable", "independent_confirmation": INDEPENDENT_CONFIRMATION,
                "claim": "No registered selecting search; tier passes are recorded outcomes, not holdout evidence."}
    by_tier = {int(tier): list(statuses) for tier, statuses in tier_statuses.items()}
    tiers = sorted(tier for tier, statuses in by_tier.items() if tier > tuning_through_tier and statuses)
    statuses = [status for tier in tiers for status in by_tier[tier]]
    passed = sum(status == "PASS" for status in statuses)
    if not statuses:
        status = "none_declared"
    elif passed == len(statuses):
        status = "PASS"
    elif any(value in FAILED for value in statuses):
        status = "FAIL"
    else:
        status = "not_measured"
    span = (f"Tier {tiers[0]}" if len(tiers) == 1 else f"Tiers {tiers[0]}–{tiers[-1]}") if tiers else "none"
    verdict = {"PASS": "passed the untuned holdout",
               "FAIL": "did not pass the untuned holdout",
               "not_measured": "untuned holdout incomplete (not measured or not a PASS/FAIL outcome)",
               "none_declared": "no untuned holdout tier is declared"}[status]
    claim = (f"Selected on Tier ≤{tuning_through_tier}; holdout {span}: {passed}/{len(statuses)} PASS — {verdict}. "
             "Tuning-tier passes are selection evidence, not independent confirmation.")
    return {"status": status, "tuning_through_tier": tuning_through_tier, "holdout_tiers": tiers,
            "required_total": len(statuses), "required_passed": passed,
            "independent_confirmation": INDEPENDENT_CONFIRMATION, "claim": claim}


def budget_text(budget: dict, lineage: dict) -> str:
    """One-cell label: declared search budget plus the selected recipe's provenance."""
    cards = budget.get("declared_candidates", 0)
    card_text = f"{cards} declared candidate card" + ("" if cards == 1 else "s")
    if not budget["searches"]:
        text = (("untuned: " if cards <= 1 else "") + "no registered configuration search; " + card_text)
    else:
        text = (f"{budget['searches']} registered search" + ("" if budget["searches"] == 1 else "es") +
                f" ({budget['configurations']} grid configurations); " + card_text)
    if lineage.get("search_id"):
        text += (f"; selected recipe descends from `{lineage['search_id']}` "
                 f"(tuned on Tier ≤{lineage['tuning_through_tier']})")
    else:
        text += "; selected recipe declared directly (no selecting search in its lineage)"
    return text
