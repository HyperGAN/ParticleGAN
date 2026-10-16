"""Display retained experiments in their recorded tiers, without qualification.

Each row keeps one recipe/source/seed. Observation checks never enter task
counts, and untiered historical studies remain on the family detail pages.
"""
from pathlib import Path
from posixpath import relpath
from functools import lru_cache

from .contracts import read_json, file_hash

COVERAGE = Path("reports/forge/tier-coverage-20261014/results.json")


def coverage_report(root):
    path = Path(root) / COVERAGE
    if not path.is_file():
        return None
    report = read_json(path)
    if (report.get("schema_version") != 1 or report.get("scope") != "selected_family_diagnostic_coverage"
            or report.get("qualification_input") is not False or report.get("qualification_reuse") is not False):
        raise ValueError("selected family coverage must remain diagnostic-only")
    for relative, digest in report["inputs"].items():
        target = Path(root) / relative
        if not target.resolve().is_relative_to(Path(root).resolve()) or file_hash(target) != digest:
            raise ValueError("diagnostic coverage input changed: " + relative)
    return report


def _link(root, page, label, target):
    return f"[{label}]({relpath(str(Path(root) / target), str(Path(page).parent))})"


@lru_cache(maxsize=4)
def _attempt_registry(path, modified):
    # Same-key solo reruns replace wall-clock hits only when they produced a real
    # PASS/FAIL verdict; without reports/forge/wallclock-reruns.json this is a no-op.
    from .wallclock import apply_reruns, load_reruns
    return apply_reruns(read_json(path)["records"], load_reruns(Path(path).parents[2]))


def attempt_count(tasks, attempts, required, *, fill):
    """Count one research row's required tests, filling unmeasured cells.

    These are explicitly nonqualifying research rows. New same-source
    attempts fill previously unmeasured research cells (``fill``), never
    overwrite an archived verdict or a selected-family qualification result.
    """
    outcomes = {task: tasks.get(task) for task in required}
    if fill:
        for task in required:
            if outcomes[task] in {None, "UNKNOWN", "NOT_RUN"} and attempts.get(task, {}).get("status") in {"PASS", "FAIL"}:
                outcomes[task] = attempts[task]["status"]
            # An archived wall-clock hit was never graded; a same-key solo rerun's
            # real verdict (wallclock_rerun) is its first measurement.
            elif (outcomes[task] in {"INCOMPLETE", "TIMEOUT", "BLOCKED"} and attempts.get(task, {}).get("wallclock_rerun")
                  and attempts[task]["status"] in {"PASS", "FAIL"}):
                outcomes[task] = attempts[task]["status"]
    value = {"passed": sum(outcomes[task] == "PASS" for task in required), "total": len(required),
             "not_run": sum(tasks.get(task) in {None, "UNKNOWN", "NOT_RUN"} for task in required),
             "incompatible": sum(tasks.get(task) == "BLOCKED" for task in required)}
    if attempts:
        coverage = {}
        for task in required:
            if outcomes[task] in {"PASS", "FAIL"}:
                continue
            attempt = attempts.get(task, {})
            status = (outcomes[task] if outcomes[task] in {"ERROR", "INVALID", "INCOMPLETE"}
                      else attempt.get("status", outcomes[task]))
            key = ("queued" if status == "PENDING" else "running" if status == "RUNNING" else
                   "errors" if status in {"ERROR", "INVALID"} else
                   "timeouts" if status == "TIMEOUT" else
                   "incomplete_attempts" if status == "INCOMPLETE" else
                   "interrupted_attempts" if status == "INTERRUPTED" else
                   "incompatible" if status == "BLOCKED" and (not attempt or attempt.get("block_kind") == "incompatible") else
                   "blocked" if status == "BLOCKED" else
                   "diagnostic_attempts" if attempt.get("attempted") else "not_run")
            coverage[key] = coverage.get(key, 0) + 1
        value["attempt_coverage"] = coverage
    return value


def _row(root, page, label, target, tasks, requirements, *, scope=None, selected_family=None, policy_scope=None,
         repair_scope=None, records=None):
    from .family_reports import score
    registry = Path(root) / "reports/forge/attempt-coverage.json"
    attempts = {}
    if records is not None:
        attempts = dict(records)  # A cohort publication supplies its own records.
    elif registry.is_file() and (scope or selected_family or policy_scope or repair_scope):
        attempts = {r.get("parent_task", r["task"]): r for r in _attempt_registry(registry, registry.stat().st_mtime_ns)
                    if (scope and r.get("research_scope") == scope)
                    or (selected_family and r.get("family") == selected_family)
                    or (policy_scope and r.get("policy_scope") == policy_scope)
                    or (repair_scope and r.get("repair_scope") == repair_scope)}
    fill = bool(scope or policy_scope or repair_scope or records is not None)

    def count(required):
        return attempt_count(tasks, attempts, required, fill=fill)
    if set(tasks) - set().union(*map(set, requirements.values())):
        raise ValueError("research result has no recorded tier assignment")
    counts = []
    for tier in ("1", "2", "3"):
        required = requirements[tier]
        if not required:
            counts.append("—")
            continue
        counts.append(_link(root, page, score(count(required)), target))
    required = set().union(*map(set, requirements.values()))
    counts.append(_link(root, page, score(count(required)), target))
    return "| " + " | ".join([_link(root, page, label, target), *counts]) + " |"


def _confidence_rows(root, page, requirements):
    from .test_performance import Inputs, SUPPLEMENTAL
    from .test_performance_render import _render_confidence_packets
    inputs = Inputs(root)
    if not (Path(root) / SUPPLEMENTAL).is_file():
        return []
    names = {
        "pr399-confidence-lr-tier1-6-20261013": "screen",
        "pr399-confidence-lr-own-retention-2-20261013": "retention",
        "pr399-gaussian-finite-cap-1-20261013": "finite_cap",
        "pr399-critic-orientation-2-20261013": "orientation",
        "pr399-signed-confidence-2-20261013": "signed",
        "pr399-positive-evidence-2-20261013": "positive",
    }
    packets = {}
    for pin in inputs.read(SUPPLEMENTAL)["publications"]:
        document = inputs.read(pin["path"], sha256=pin["sha256"])
        key = names.get(document.get("discovery_summary", {}).get("id"))
        if key:
            if key in packets or any(document.get(flag) is not False for flag in
                                     ("qualification_input", "qualification_reuse")):
                raise ValueError("confidence tier rows require unique nonqualifying packets")
            packets[key] = pin, document
    # Reuse the existing receipt/continuity/count validators before projection.
    _render_confidence_packets(packets, lambda name: name)
    if not packets:
        return []
    rows = []
    for label, keys, anchor in (
        ("Confidence LR; seed 0", ("screen", "retention"), "confidence-lr-correction-six-task-screen-and-own-retention"),
        ("Finite critic cap; seed 0", ("finite_cap",), "later-gaussian-corrections-one-failed-acquisition-one-failed-retention"),
        ("Critic orientation; seed 0", ("orientation",), "later-gaussian-corrections-one-failed-acquisition-one-failed-retention"),
        ("Signed confidence; seed 0", ("signed",), "signed-and-positive-evidence-responses-retention-remains-unsolved"),
        ("Positive evidence; seed 0", ("positive",), "signed-and-positive-evidence-responses-retention-remains-unsolved"),
    ):
        tasks, identities = {}, set()
        for key in keys:
            document = packets[key][1]
            for cell in document["cells"]:
                task = cell["task_id"]
                if task in tasks:
                    raise ValueError("research tier row repeats a task")
                tasks[task] = cell["status"]
                condition = document["condition_contracts"][cell["conditions_sha256"]]
                identities.add((cell["candidate_revision"], condition["source_digest"], condition["seed"]))
        if len(identities) != 1:
            raise ValueError("research tier row cannot pool recipes, sources or seeds")
        from .research_attempts import scope_id
        revision, source, seed = next(iter(identities))
        rows.append(_row(root, page, label, "reports/forge/families/bcap-pure.md#" + anchor,
                         tasks, requirements, scope=scope_id(source, revision) if seed == 0 else None))
    return rows


def checkpoint_repair_rows(root, page):
    rows = {}
    registry = Path(root) / "reports/forge/attempt-coverage.json"
    if registry.is_file():
        from collections import defaultdict
        repairs = defaultdict(list)
        for record in _attempt_registry(registry, registry.stat().st_mtime_ns):
            if record.get("repair_scope") and not record.get("repair_superseded"):
                repairs[record["repair_scope"]].append(record)
        for scope, records in repairs.items():
            identities = {(r["source_digest"], r["candidate_revision"], r["owning_family"]) for r in records}
            if len(identities) != 1 or len({r["task"] for r in records}) != len(records):
                raise ValueError("checkpoint repair rows cannot pool sources, recipes or repeated tasks")
            requirements = {str(tier): {r["task"] for r in records if r["reporting_tier"] == tier} for tier in (1, 2, 3)}
            family = records[0]["owning_family"]
            rows.setdefault(family, []).append(_row(root, page, records[0]["label"],
                "reports/forge/ATTEMPT_COVERAGE.md#checkpoint-recovery", {}, requirements, repair_scope=scope))
    return rows


def policy_cohort_rows(root, page):
    """One row per family and scoring law of the policy-cloud cohorts.

    Records come only from the cohort's own committed publication
    (``policy_cloud_publication.RECORDS``); the live attempt service rewrites
    attempt-coverage.json, so cohort records never live there. The clean and
    as-served laws are separate rows (clean first). A law-independent record
    (one run for both laws) enters both rows with its single verdict, and the
    as-served row says how many verdicts it shares.
    """
    from .policy_cloud_publication import anchor, cohort_rows, load_records, row_label
    registry = Path(root) / "reports/forge/attempt-coverage.json"
    if registry.is_file() and any(r.get("cohort_scope") for r in _attempt_registry(registry, registry.stat().st_mtime_ns)):
        raise ValueError("policy-cloud cohort records belong to their own publication, not attempt-coverage.json")
    value = load_records(root)
    rows = {}
    for row in cohort_rows(value) if value is not None else []:
        rows.setdefault(row["family"], []).append(_row(root, page, row_label(row), anchor(row["family"]), {},
                                                       row["requirements"], records=row["records"]))
    return rows


def tier_rows(root, page):
    """Rows keyed by owning family; no modification of selected publication data."""
    rows = {}
    from .source_bound_board import BOARD_SOURCE, validate_board
    from .confidence_coherence_scores import validate_original_rows
    if (Path(root) / BOARD_SOURCE).is_file():
        board = validate_board(read_json(Path(root) / BOARD_SOURCE))
        validate_original_rows(root, board)
        requirements = {str(tier): {a["task"] for a in board["recorded_view"]["assignments"]
                                   if a["importance"] == "required" and a["qualification_tier"] == tier}
                        for tier in (1, 2, 3)}
        if ({tier: len(tasks) for tier, tasks in requirements.items()} != {"1": 6, "2": 21, "3": 2}
                or len(set().union(*requirements.values())) != 29):
            raise ValueError("research tiers must retain the recorded revision 9 requirements")
        bcap = rows.setdefault("bcap-pure", [])
        bcap.extend(_confidence_rows(root, page, requirements))
        for record in board["rows"]:
            tasks = {task: cell["status"] for task, cell in record["benchmark_tasks"].items()
                     if cell["status"] is not None}
            from .research_attempts import scope_id
            bcap.append(_row(root, page, record["label"], record["readout"], tasks, requirements,
                             scope=scope_id(record["source_digest"], record["candidate_revision"])))
        from .pr399_scores import tier_rows as pr399_tier_rows
        for family, additions in pr399_tier_rows(root, page, requirements).items():
            rows.setdefault(family, []).extend(additions)
    from .atlas_catalog_scores import ATLAS, _read, atlas_score_lines
    if (Path(root) / ATLAS / "results.json").is_file():
        atlas_score_lines(root)  # Existing byte pins, original verdicts and scope.
        result = _read(root, f"{ATLAS}/results.json")
        tasks = {row["task"]: row["status"] for row in result["rows"]}
        tasks.update({task: "BLOCKED" for task in result["blocked_unstarted"]})
        if len(tasks) != 7 or any("tier1_policy_selected_cloud_v1" not in task for task in tasks):
            raise ValueError("Atlas policy questions must retain their original Tier 1 scope")
        rows["atlas"] = [_row(root, page, "Selected-cloud policy; seed 0 (separate scope)",
                              f"{ATLAS}/README.md", tasks, {"1": set(tasks), "2": set(), "3": set()})]
    coverage = coverage_report(root)
    for record in (coverage or {}).get("diagnostic_rows", []):
        requirements = {tier: set(names) for tier, names in record["requirements"].items()}
        rows.setdefault(record["family"], []).append(_row(root, page, record["label"],
            "reports/forge/tier-coverage-20261014/README.md", record["tasks"], requirements))
    followup = Path("reports/forge/actual-missing-diagnostics-20261014/results.json")
    if (Path(root) / followup).is_file():
        measured = read_json(Path(root) / followup)
        if (measured.get("schema") != "actual_missing_diagnostics_v1"
                or measured.get("qualification_input") is not False
                or measured.get("qualification_reuse") is not False
                or measured.get("formulations_modified") is not False):
            raise ValueError("actual execution readout must remain an unchanged-formulation diagnostic")
        for record in measured["diagnostic_rows"]:
            requirements = {tier: set(names) for tier, names in record["requirements"].items()}
            policy = None
            if record["label"] == "Selected-cloud execution after adapter investigation":
                identities = {(r["source_digest"], r["candidate_revision"]) for r in measured["records"]
                              if r["family"] == record["family"] and r["lane"] == "policy"}
                # Older E22 summaries already retain multiple source cohorts.
                # Do not attach new results to such an aggregate.
                if len(identities) == 1:
                    from .remaining_attempts import policy_scope
                    policy = policy_scope(record["family"], *next(iter(identities)))
            target = str(followup.with_name("README.md"))
            registry = Path(root) / "reports/forge/attempt-coverage.json"
            if policy and registry.is_file() and any(r.get("policy_scope") == policy for r in
                    _attempt_registry(registry, registry.stat().st_mtime_ns)):
                target = "reports/forge/ATTEMPT_COVERAGE.md#remaining-attempts"
            label = record["label"]
            if (label == "Selected-cloud execution after adapter investigation" and registry.is_file()
                    and any(r.get("repair_scope") and r.get("owning_family") == record["family"]
                            for r in _attempt_registry(registry, registry.stat().st_mtime_ns))):
                label += " (pre-repair source)"
            rows.setdefault(record["family"], []).append(_row(root, page, label,
                target, record["tasks"], requirements,
                selected_family=record["family"] if record["label"].startswith("Unchanged selected recipe:") else None,
                policy_scope=policy))
    for family, additions in checkpoint_repair_rows(root, page).items():
        rows.setdefault(family, []).extend(additions)
    for family, additions in policy_cohort_rows(root, page).items():
        rows.setdefault(family, []).extend(additions)
    registry = Path(root) / "reports/forge/attempt-coverage.json"
    if registry.is_file():
        for record in _attempt_registry(registry, registry.stat().st_mtime_ns):
            if record.get("origin") != "reused_execution":
                continue
            if record.get("qualification_input", False) or not record.get("reused_execution"):
                raise ValueError("saved group reuse must remain an explicit diagnostic")
            rows.setdefault(record["family"], []).append(_row(root, page,
                "Verified extension from the existing uninterrupted run; source " + record["source_digest"][:12],
                record["evidence"], {record["task"]: record["status"]},
                {"1": set(), "2": set(), "3": {record["task"]}}))
    return rows


def historical_seed_rows(root, page):
    """Keep excluded seed-only studies visible without presenting them as queued work."""
    from .source_bound_board import BOARD_SOURCE, validate_board
    from .confidence_coherence_scores import PAIRED, _pinned, validate_paired_results
    if not (Path(root) / BOARD_SOURCE).is_file() or not (Path(root) / PAIRED).is_file():
        return {}
    board = validate_board(read_json(Path(root) / BOARD_SOURCE))
    requirements = {str(tier): {a["task"] for a in board["recorded_view"]["assignments"]
                               if a["importance"] == "required" and a["qualification_tier"] == tier}
                    for tier in (1, 2, 3)}
    paired = _pinned(root, PAIRED)
    validate_paired_results(paired)
    rows = []
    for seed in range(1, 7):
        for arm in ("candidate", "control"):
            study = f"paired-g-coherence-seed{seed}-{arm}"
            tasks = {r["task"]: r["gate"]["status"] for r in paired["records"] if r["study"] == study}
            rows.append(_row(root, page, f"G-coherence {arm}; research seed {seed}",
                             "reports/forge/families/bcap-pure.md#g-only-coherence-registered-paired-research",
                             tasks, requirements))
    return {"bcap-pure": rows}


def family_research_lines(root, page, family_id):
    """Keep detailed scores and untiered historical diagnostics off the scoreboard."""
    if family_id == "atlas":
        from .atlas_catalog_scores import atlas_score_lines
        from .pr399_scores import family_lines
        return atlas_score_lines(root) + family_lines(root, page, family_id)
    if family_id != "bcap-pure":
        return []
    from .atlas_catalog_scores import catalog_score_lines
    from .source_bound_board import main_board_rows
    from .test_performance_render import render_discovery_navigation
    lines = catalog_score_lines(root, page) + main_board_rows(root, page)
    lines.append(render_discovery_navigation(root, page))
    from .pr399_scores import family_lines
    lines += family_lines(root, page, family_id)
    text = "\n".join(lines)
    for heading, anchor in (
        ("G-only coherence: registered paired research", "g-only-coherence-registered-paired-research"),
        ("Confidence-LR correction: six-task screen and own retention", "confidence-lr-correction-six-task-screen-and-own-retention"),
        ("Later Gaussian corrections: one failed acquisition, one failed retention", "later-gaussian-corrections-one-failed-acquisition-one-failed-retention"),
        ("Signed and positive-evidence responses: retention remains unsolved", "signed-and-positive-evidence-responses-retention-remains-unsolved"),
    ):
        text = text.replace("## " + heading, f'<a name="{anchor}"></a>\n\n## ' + heading, 1)
    return [text]
