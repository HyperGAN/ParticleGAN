"""Read-only task grading and versioned qualification views.

Receipts supplied here must already be filtered by the coordinator's candidate,
seed, source, and runtime compatibility key. A trainer's PASS string is never
evidence: graders reconstruct verdicts from curves or the existing file audits.
No function in this module launches training, rewrites receipts, or enqueues work.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .sampling import grade_sampling, validate_declaration
from .priors import task_prior
from .initialization import task_initializer


ROOT = Path(__file__).resolve().parents[2]
# A stretch (or legacy held) task stays assigned, graded and shown in the
# Stretch table, but it is not counted in required totals, blockers or next
# tasks. A task removed from the count stays assigned as diagnostic and carries
# a removal note naming why.
STRETCH_ROLES = frozenset({"stretch", "held"})
IMPORTANCES = {"required", "stretch", "ranking", "diagnostic", "held"}
COUNTED_OR_HELD = frozenset({"required", *STRETCH_ROLES})
HOLD_SCHEMA, STRETCH_SCHEMA, REMOVAL_SCHEMA = "forge_hold_v1", "forge_stretch_v1", "forge_count_removal_v1"
# unsolved: no recorded recipe has ever passed the current declaration.
# held_prerequisite: the task depends on a held task (listed in blocked_by).
HOLD_REASONS = frozenset({"no_global_recipe", "unsolved", "held_prerequisite"})
STRETCH_REASONS = frozenset({"frontier", "extra_credit", "optional", *HOLD_REASONS, "stretch_prerequisite"})
# phase and duplicate name the counted task that carries the question (counted_by).
REMOVAL_REASONS = frozenset({"phase", "duplicate", "superseded", "ill_posed"})
GATE_STATUSES = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "NOT_RUN", "BLOCKED"}
DIAGNOSTIC_SCOPES = frozenset({"calibration_diagnostic", "research_diagnostic"})


def is_stretch(assignment_or_role: dict | str | None) -> bool:
    """True when an assignment dict or importance string represents a stretch/held task."""
    role = assignment_or_role.get("importance") if isinstance(assignment_or_role, dict) else assignment_or_role
    return role in STRETCH_ROLES


def stretch_note(assignment: dict | None) -> dict | None:
    """Return the stretch or legacy hold note attached to a stretch assignment, if any."""
    if not isinstance(assignment, dict) or not is_stretch(assignment):
        return None
    note = assignment.get("stretch") or assignment.get("hold")
    return note if isinstance(note, dict) else None


def diagnostic_evidence_scope(request: dict) -> str | None:
    """Keep diagnostic evidence nonqualifying across saved-request reducers."""
    markers = {request.get("view", {}).get("evidence_scope")}
    markers.update(job.get("science", {}).get("evidence_use") for job in request.get("jobs", []))
    if request.get("calibration_lane") or "calibration_diagnostic" in markers:
        return "calibration_diagnostic"
    return "research_diagnostic" if "research_diagnostic" in markers else None


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def task_execution_fingerprint(task: dict) -> str:
    """Scientific execution identity deliberately excludes view policy."""
    return _hash({key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")})


def task_evaluation_fingerprint(task: dict) -> str:
    return _hash(task["evaluation"])


def task_fingerprint(task: dict) -> str:
    return _hash({"execution": task_execution_fingerprint(task),
                  "evaluation": task_evaluation_fingerprint(task)})


def view_fingerprint(view: dict) -> str:
    return _hash(view)


def _positive_int(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _cost(receipt):
    cost = receipt.get("cost", {})
    value = cost.get("wall_seconds", cost.get("seconds", 0)) if isinstance(cost, dict) else 0
    return value if _finite(value) and value >= 0 else 0


def _dependencies(task):
    result = []
    for entry in task.get("dependencies", []):
        if not isinstance(entry, dict) or entry.get("kind") not in {"gate", "checkpoint"}:
            raise ValueError(f"{task['id']}: dependency needs task and gate/checkpoint kind")
        if not isinstance(entry.get("task"), str) or not entry["task"]:
            raise ValueError(f"{task['id']}: dependency task must be named")
        result.append(entry)
    if len({entry["task"] for entry in result}) != len(result):
        raise ValueError(f"{task['id']}: duplicate dependency")
    return result


def _validate_measurement_contract(task):
    """Repeated evaluator constants are declarations, never ignored overrides."""
    validate_declaration(task)
    evaluation = task["evaluation"]
    kind = evaluation["kind"]
    fixed = {"scoring_weights": "live"}
    measured = task
    from .tier1_policy import is_policy_task
    if is_policy_task(task):
        from .tier1_policy import validate
        # A scoped variant changes only its declared binding; its unchanged
        # parent must still satisfy the referenced evaluator's constants.
        measured = validate(task)
        fixed["scoring_weights"] = "state_selected"
    if kind in {"gaussian_smoke", "gaussian_stability"}:
        from .gaussian_tasks import validate_task
        validate_task(measured)
    elif kind in {"word_smoke", "word_hold"}:
        from .word_adapter import validate_word_task
        validate_word_task(measured)
    elif kind == "transfer_sustained":
        from benchmarks.locked_shared.observation import OBSERVATIONS, MIN_STABLE_CHECKS
        declared = evaluation.get("evaluator") == "experiments.forge.transfer_cadence:test_verdict"
        if declared:
            count = evaluation.get("observations")
            if type(count) is not int or not MIN_STABLE_CHECKS <= count <= task["execution"]["steps"]:
                raise ValueError(f"{task['id']}: declared observations must fit the full update budget")
        fixed.update(evaluator="experiments.forge.transfer_cadence:test_verdict" if declared else "benchmarks.transfer_suite.protocol:test_verdict",
                     observations=evaluation["observations"] if declared else OBSERVATIONS,
                     minimum_stable_checks=MIN_STABLE_CHECKS)
    elif kind == "native_accuracy":
        from benchmarks.toy100.accuracy import LIMITS
        from benchmarks.toy100.accuracy_gate import HOLDOUT_N
        from benchmarks.toy100.gate import MIN_STABLE_CHECKS
        from benchmarks.toy100.metrics import REQUIREMENTS
        metrics_pin = measured.get("evaluation", {}).get("sources", {}).get("benchmarks/toy100/metrics.py")
        expected_req = (
            {**REQUIREMENTS, "min_precision": 0.97}
            if metrics_pin == "a7fbef19082f8470eb392948da52a336a90a3e99c6b3b893343a9e75bcda9c7b"
            or (metrics_pin is None and measured.get("evaluation", {}).get("coverage_thresholds", {}).get("min_precision") == 0.97)
            else REQUIREMENTS
        )
        fixed.update(evaluator="benchmarks.toy100.accuracy_gate:evaluate_suite",
                     coverage_evaluator="benchmarks.toy100.gate:evaluate_suite",
                     minimum_stable_checks=MIN_STABLE_CHECKS, holdout_samples=HOLDOUT_N,
                     coverage_thresholds=expected_req, accuracy_limits=LIMITS)
    elif kind in {"ring_hold", "ring_extension"}:
        declaration = _convergence_class()().declaration()
        fixed.update(evaluator="reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py:ConvergenceGate",
                     thresholds=[["modes", "==", declaration["modes"]],
                                 ["hq", ">=", declaration["min_hq"]], ["hq", "<=", 1.]])
    elif kind == "paired_adaptation":
        from benchmarks.toy100.continuous_probe import RECOVERY_DEADLINE
        fixed.update(evaluator="benchmarks.toy100.continuous_probe:match_frozen_control",
                     recovery_deadline=RECOVERY_DEADLINE, diagnostic_every=10,
                     stationary_checks=5, minimum_frozen_passing=0)
        # The complete late window is recomputed from these consumed fields.
        fixed["deadline_checks"] = (evaluation["steps"] - evaluation["shift_step"] - RECOVERY_DEADLINE) // 10 + 1
    elif kind == "clockfree_parity":
        fixed.update(evaluator="experiments.forge.views:_clockfree", training_feedback=False)
        conditions = evaluation.get("conditions")
        if not isinstance(conditions, list) or sorted(conditions) != sorted(("step_label", "horizon", "evaluation_cadence", "restart")):
            raise ValueError(f"{task['id']}: all four fixed clock-free comparisons are required")
    elif kind == "image_training_performance":
        from .scaling_adapter import validate_task
        validate_task(task)
        fixed.update(evaluator="experiments.forge.scaling_adapter:grade", training_feedback=False,
                     qualification_input=False)
    elif kind == "zero_tanh_smoke":
        # The implemented point-mass adapter draws 4,096 clean live samples.
        # This task has final bounds, without a sustained-suffix requirement.
        fixed.update(eval_samples=4096, minimum_stable_checks=1)
    else:
        return
    for key, expected in fixed.items():
        if key in evaluation and _hash(evaluation[key]) != _hash(expected):
            raise ValueError(f"{task['id']}: evaluation.{key} is fixed by the referenced evaluator; unsupported override")


def _validate_task(task):
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    required = {"schema_version", "id", "adapter", "execution", "evaluation",
                "resources", "requires_capabilities", "dependencies"}
    if required - task.keys():
        raise ValueError(f"task lacks {sorted(required - task.keys())}")
    if "qualification_tier" in task:
        raise ValueError("qualification_tier belongs only in view assignments")
    _positive_int(task["schema_version"], "task schema_version")
    for key in ("id", "adapter"):
        if not isinstance(task[key], str) or not task[key]:
            raise ValueError(f"task {key} must be nonempty")
    for key in ("execution", "evaluation", "resources"):
        if not isinstance(task[key], dict):
            raise ValueError(f"task {key} must be an object")
    if not isinstance(task["evaluation"].get("kind"), str):
        raise ValueError("task evaluation.kind is required")
    if not isinstance(task["dependencies"], list):
        raise ValueError("task dependencies must be a list")
    caps = task["requires_capabilities"]
    if not isinstance(caps, list) or any(not isinstance(x, str) or not x for x in caps):
        raise ValueError("requires_capabilities must be a list of names")
    _dependencies(task)
    task_prior(task)
    task_initializer(task)
    _validate_measurement_contract(task)


def load_tasks(root: Path | str) -> dict:
    """Validated task catalogue; memoized while a resolution cache is bound.

    Inside ``planning.resolution_cache(root)`` the catalogue is read once and
    each caller receives a shallow copy of the memoized mapping (the task
    declarations themselves are shared and must be treated as read-only).
    """
    from .resolution_scope import active
    cache = active(root)
    if cache is not None:
        return dict(cache.get_all_tasks())
    return read_task_catalog(root)


def read_task_catalog(root: Path | str) -> dict:
    """Read and validate every task declaration, bypassing any resolution cache."""
    tasks = {}
    for path in sorted((Path(root) / "configs/forge/tasks").glob("*.json")):
        task = _read(path)
        _validate_task(task)
        if task["id"] in tasks:
            raise ValueError(f"duplicate task id {task['id']}")
        tasks[task["id"]] = task
    if not tasks:
        raise ValueError("no Forge tasks found")
    from .tier1_policy import load_variants
    for name, task in load_variants(root, tasks).items():
        _validate_task(task)
        tasks[name] = task
    return tasks


def load_view(root: Path | str, view_id: str) -> dict:
    if not isinstance(view_id, str) or Path(view_id).name != view_id or view_id in {"", ".", ".."}:
        raise ValueError("view id must be a filename-safe name")
    view = _read(Path(root) / "configs/forge/views" / f"{view_id}.json")
    if view.get("id") != view_id:
        raise ValueError("view filename and id differ")
    validate_view(view, load_tasks(root))
    return view


def explanation_reference(value, label):
    """Repo-relative Markdown path#anchor; report tests check that the anchor exists.

    A URL scheme, drive letter or home directory never resolves inside the
    repository, so ':' and a leading '~' are refused like absolute paths.
    """
    if not isinstance(value, str) or value.count("#") != 1:
        raise ValueError(f"{label} needs a repo-relative path#anchor explanation")
    path, anchor = value.split("#")
    if (not anchor or Path(path).suffix != ".md" or Path(path).is_absolute() or ".." in Path(path).parts
            or ":" in path or path.startswith("~") or "\\" in path or any(c.isspace() for c in value)):
        raise ValueError(f"{label} needs a repo-relative path#anchor explanation")


def plain_summary(value, label):
    if not isinstance(value, str) or value != value.strip() or not 20 <= len(value) <= 280:
        raise ValueError(f"{label} needs a plain-language summary of 20-280 characters")


def counted_by_matches(reason, counted_by) -> bool:
    """Phase and duplicate removals name the counted task; the other reasons use null."""
    if reason in {"phase", "duplicate"}:
        return isinstance(counted_by, str) and bool(counted_by)
    return counted_by is None


def _validate_hold(name, hold):
    keys = {"schema", "previous_importance", "reason", "blocked_by", "summary", "explanation", "reentry"}
    if not isinstance(hold, dict) or set(hold) != keys:
        raise ValueError(f"{name}: held assignments need hold metadata with exactly {sorted(keys)}")
    if hold["schema"] != HOLD_SCHEMA or hold["previous_importance"] != "required":
        raise ValueError(f"{name}: only a previously required task can be held ({HOLD_SCHEMA})")
    if hold["reason"] not in HOLD_REASONS:
        raise ValueError(f"{name}: hold reason must be one of {sorted(HOLD_REASONS)}")
    blocked = hold["blocked_by"]
    if (not isinstance(blocked, list) or any(not isinstance(x, str) or not x for x in blocked)
            or blocked != sorted(set(blocked))):
        raise ValueError(f"{name}: hold blocked_by must list distinct task ids in sorted order")
    plain_summary(hold["summary"], f"{name}: hold")
    explanation_reference(hold["explanation"], f"{name}: hold")
    if not isinstance(hold["reentry"], str) or len(hold["reentry"].strip()) < 20:
        raise ValueError(f"{name}: hold needs a concrete re-entry criterion")


def _validate_stretch(name, stretch):
    if stretch is None:
        return
    allowed = {"schema", "previous_importance", "reason", "blocked_by", "summary", "explanation", "reentry", "graduation"}
    required = {"reason", "summary"}
    if not isinstance(stretch, dict) or not required <= set(stretch) <= allowed:
        raise ValueError(f"{name}: stretch metadata needs at least {sorted(required)} and only {sorted(allowed)}")
    if "schema" in stretch and stretch["schema"] not in {STRETCH_SCHEMA, HOLD_SCHEMA}:
        raise ValueError(f"{name}: stretch schema must be {STRETCH_SCHEMA}")
    if stretch["reason"] not in STRETCH_REASONS:
        raise ValueError(f"{name}: stretch reason must be one of {sorted(STRETCH_REASONS)}")
    if "blocked_by" in stretch:
        blocked = stretch["blocked_by"]
        if (not isinstance(blocked, list) or any(not isinstance(x, str) or not x for x in blocked)
                or blocked != sorted(set(blocked))):
            raise ValueError(f"{name}: stretch blocked_by must list distinct task ids in sorted order")
    plain_summary(stretch["summary"], f"{name}: stretch")
    if "explanation" in stretch:
        explanation_reference(stretch["explanation"], f"{name}: stretch")
    for field in ("reentry", "graduation"):
        if field in stretch and (not isinstance(stretch[field], str) or len(stretch[field].strip()) < 20):
            raise ValueError(f"{name}: stretch {field} needs a concrete criterion of at least 20 characters")


def _validate_removal(name, removal):
    keys = {"schema", "previous_importance", "reason", "counted_by", "summary", "explanation"}
    if not isinstance(removal, dict) or set(removal) != keys:
        raise ValueError(f"{name}: removal metadata needs exactly {sorted(keys)}")
    if removal["schema"] != REMOVAL_SCHEMA or removal["previous_importance"] != "required":
        raise ValueError(f"{name}: only a previously required task can be removed from the count ({REMOVAL_SCHEMA})")
    if removal["reason"] not in REMOVAL_REASONS:
        raise ValueError(f"{name}: removal reason must be one of {sorted(REMOVAL_REASONS)}")
    if not counted_by_matches(removal["reason"], removal["counted_by"]):
        raise ValueError(f"{name}: phase and duplicate removals name counted_by; superseded and ill_posed use null")
    plain_summary(removal["summary"], f"{name}: removal")
    explanation_reference(removal["explanation"], f"{name}: removal")


def _validate_previous_revision(view):
    previous = view.get("previous_revision")
    if previous is None:
        return
    if not isinstance(previous, dict) or set(previous) != {"revision", "archive", "policy_fingerprint"}:
        raise ValueError("previous_revision needs exactly revision, archive and policy_fingerprint")
    revision = previous["revision"]
    if type(revision) is not int or not 1 <= revision < view["revision"]:
        raise ValueError("previous_revision must name an earlier revision")
    if previous["archive"] != f"configs/forge/view-history/{view['id']}-v{revision}.json":
        raise ValueError("previous_revision archive must be configs/forge/view-history/<view>-v<revision>.json")
    if not _digest(previous["policy_fingerprint"]):
        raise ValueError("previous_revision policy_fingerprint must be a SHA-256 digest")


def _phase_parent(index, tasks, parent, child):
    """A removed phase may precede the counted verdict that replays it on the same run.

    Both must share one uninterrupted execution group and tier, and the parent's
    removal note must name the child. Callers also require a checkpoint edge, and
    validate_view requires the phase's own prerequisites to be counted.
    """
    removal = index[parent].get("removal") or {}
    group = tasks[child]["execution"].get("execution_group")
    return bool(index[parent]["importance"] == "diagnostic" and removal.get("reason") == "phase"
                and removal.get("counted_by") == child and group
                and tasks[child]["execution"].get("uninterrupted") and tasks[parent]["execution"].get("uninterrupted")
                and tasks[parent]["execution"].get("execution_group") == group
                and index[parent]["qualification_tier"] == index[child]["qualification_tier"])


def _same_question(removed, counted):
    """Duplicates grade one question: equal evaluation declarations, apart from a
    source pin of the other task's own declaration (a derived copy's provenance)."""
    def declared(task, other):
        evaluation = deepcopy(task["evaluation"])
        sources = evaluation.pop("sources", {})
        if isinstance(sources, dict):
            sources = {path: digest for path, digest in sources.items()
                       if path != f"configs/forge/tasks/{other['id']}.json"}
        return {**evaluation, **({"sources": sources} if sources else {})}
    return _hash(declared(removed, counted)) == _hash(declared(counted, removed))


def validate_view(view: dict, tasks: dict) -> None:
    if not isinstance(view, dict):
        raise ValueError("view must be an object")
    for key in ("id", "goal"):
        if not isinstance(view.get(key), str) or not view[key]:
            raise ValueError(f"view needs {key}")
    _positive_int(view.get("schema_version"), "view schema_version")
    _positive_int(view.get("revision"), "view revision")
    assignments = view.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("view needs nonempty assignments")
    if not isinstance(view.get("eligibility", {}), dict):
        raise ValueError("view eligibility must be an object")
    _validate_previous_revision(view)
    index = {}
    required_tiers = set()  # Tiers with a counted or stretch/held task.
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("assignment must be an object")
        name = assignment.get("task")
        if not isinstance(name, str) or name not in tasks:
            raise ValueError(f"missing task reference {name!r}")
        if name in index:
            raise ValueError(f"duplicate assignment {name}")
        _validate_task(tasks[name])
        if tasks[name]["id"] != name:
            raise ValueError("task index and id disagree")
        tier = assignment.get("qualification_tier")
        if type(tier) is not int or isinstance(tier, bool) or tier < 1:
            raise ValueError("qualification_tier must be an integer >= 1")
        if assignment.get("importance") not in IMPORTANCES:
            raise ValueError("unknown task importance")
        if assignment["importance"] == "held":
            _validate_hold(name, assignment.get("hold"))
        elif "hold" in assignment:
            raise ValueError(f"{name}: hold metadata requires importance held")
        if assignment["importance"] == "stretch":
            _validate_stretch(name, assignment.get("stretch"))
        elif "stretch" in assignment:
            raise ValueError(f"{name}: stretch metadata requires importance stretch")
        if "removal" in assignment:
            if assignment["importance"] != "diagnostic":
                raise ValueError(f"{name}: a task removed from the count stays assigned as diagnostic")
            _validate_removal(name, assignment["removal"])
        _positive_int(assignment.get("order"), "assignment order", minimum=0)
        index[name] = assignment
        if assignment["importance"] in COUNTED_OR_HELD:
            required_tiers.add(tier)
    highest = max(a["qualification_tier"] for a in assignments)
    diagnostic = view.get("evidence_scope") in DIAGNOSTIC_SCOPES
    if any(tasks[name]["adapter"] == "dcgan_performance_v1" for name in index) and view.get("evidence_scope") != "research_diagnostic":
        raise ValueError("image performance tasks require a separate research_diagnostic view and cannot grant qualification")
    if diagnostic and any(a["importance"] != "diagnostic" or a["qualification_tier"] != 1 for a in assignments):
        raise ValueError("diagnostic views require only diagnostic tasks in Tier 1")
    if not diagnostic and required_tiers != set(range(1, highest + 1)):
        raise ValueError("empty required tier: qualification tiers must be contiguous from Tier 1")
    execution_groups = {}
    for name, assignment in index.items():
        execution = tasks[name]["execution"]
        if execution.get("uninterrupted") and execution.get("execution_group"):
            group = execution["execution_group"]
            tier = assignment["qualification_tier"]
            if group in execution_groups and execution_groups[group] != tier:
                raise ValueError(f"retier uninterrupted execution group {group} together; it cannot cross tier caps")
            execution_groups[group] = tier
    visiting, visited = set(), set()

    def require_counted(name, parent, via=None):
        """A counted verdict rests only on counted tasks, directly or through its removed phase."""
        if index[parent]["importance"] == "required":
            return
        if index[parent]["importance"] == "held" and via:
            raise ValueError(f"{name}: its removed phase {via} depends on held task {parent}; count {parent}, "
                             f"or hold {name} and {via} with reason held_prerequisite")
        if index[parent]["importance"] == "held":
            raise ValueError(f"{name}: required task depends on held task {parent}; hold {name} "
                             f"with reason held_prerequisite and list {parent} in blocked_by")
        raise ValueError("a required task cannot depend on a nonrequired task"
                         + (f" (its removed phase {via} depends on {parent})" if via else ""))

    def visit(name):
        if name in visiting:
            raise ValueError(f"dependency cycle involving {name}")
        if name in visited:
            return
        visiting.add(name)
        for dep in _dependencies(tasks[name]):
            parent = dep["task"]
            if parent not in index:
                raise ValueError(f"{name}: dependency {parent} missing from view")
            if index[parent]["qualification_tier"] > index[name]["qualification_tier"]:
                raise ValueError(f"{name}: dependency {parent} is in a later tier")
            if index[name]["importance"] == "required" and index[parent]["importance"] != "required":
                if dep["kind"] == "checkpoint" and _phase_parent(index, tasks, parent, name):
                    # The removed phase runs inside this counted job, so its own
                    # prerequisites gate the counted verdict and must be counted too.
                    for inner in _dependencies(tasks[parent]):
                        if inner["task"] not in index:
                            raise ValueError(f"{parent}: dependency {inner['task']} missing from view")
                        require_counted(name, inner["task"], via=parent)
                else:
                    require_counted(name, parent)
            if dep["kind"] == "checkpoint":
                if not tasks[parent]["execution"].get("produces_state"):
                    raise ValueError(f"{name}: checkpoint dependency does not produce state")
                if tasks[name]["execution"].get("continuation_of") != parent:
                    raise ValueError(f"{name}: continuation_of must name its checkpoint dependency")
            visit(parent)
        visiting.remove(name)
        visited.add(name)

    for name in index:
        visit(name)
    for name, assignment in index.items():
        if assignment["importance"] == "held":
            hold = assignment["hold"]
            held_parents = sorted(dep["task"] for dep in _dependencies(tasks[name])
                                  if is_stretch(index[dep["task"]]))
            if hold["blocked_by"] != held_parents:
                raise ValueError(f"{name}: hold blocked_by must list exactly its held prerequisites {held_parents}")
            if (hold["reason"] == "held_prerequisite") != bool(held_parents):
                raise ValueError(f"{name}: a task held behind a held prerequisite uses reason held_prerequisite, "
                                 "and only such a task does")
        if assignment["importance"] == "stretch" and isinstance(assignment.get("stretch"), dict):
            stretch = assignment["stretch"]
            stretch_parents = sorted(dep["task"] for dep in _dependencies(tasks[name])
                                     if is_stretch(index[dep["task"]]))
            if "blocked_by" in stretch and stretch["blocked_by"] != stretch_parents:
                raise ValueError(f"{name}: stretch blocked_by must list its stretch prerequisites {stretch_parents}")
            if stretch["reason"] in {"held_prerequisite", "stretch_prerequisite"}:
                if not stretch_parents or stretch.get("blocked_by") != stretch_parents:
                    raise ValueError(f"{name}: stretch prerequisite reason requires blocked_by listing stretch prerequisites {stretch_parents}")
        removal = assignment.get("removal")
        if removal and removal["reason"] == "phase":
            target = removal["counted_by"]
            if (target not in index or index[target]["importance"] != "required"
                    or not any(dep["task"] == name and dep["kind"] == "checkpoint"
                               for dep in _dependencies(tasks[target]))
                    or not _phase_parent(index, tasks, name, target)):
                raise ValueError(f"{name}: a phase removal must be the checkpoint parent of its counted task, "
                                 "in one uninterrupted execution group and tier")
        if removal and removal["reason"] == "duplicate":
            target = removal["counted_by"]
            if (target == name or target not in index or index[target]["importance"] != "required"
                    or index[target]["qualification_tier"] > assignment["qualification_tier"]
                    or tasks[target]["adapter"] != tasks[name]["adapter"]
                    or not _same_question(tasks[name], tasks[target])):
                raise ValueError(f"{name}: a duplicate removal must name a counted task of this view at the same "
                                 "or an earlier tier, with the same adapter and the same evaluation declaration")


def _verdict(status, reason, **details):
    return {"status": status, "gate_status": status, "reasons": [reason], **details}


def _curve(points, requirements, expected=None):
    if not isinstance(points, list) or not points:
        raise ValueError("missing observation curve")
    steps = []
    for point in points:
        if not isinstance(point, dict) or type(point.get("step")) is not int or point["step"] < 0:
            raise ValueError("observation steps must be nonnegative integers")
        steps.append(point["step"])
        for name, _, _ in requirements:
            if name not in point or not _finite(point[name]):
                raise ValueError(f"missing or nonfinite observed metric {name}")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("observation steps must be unique and strictly increasing")
    if expected is not None and steps != list(expected):
        raise ValueError("observation schedule is incomplete or differs from the task")
    return points


def _guards(task, evidence):
    expected = task["evaluation"].get("guards", {})
    if not expected:
        return None
    guards = evidence.get("guards")
    if not isinstance(guards, dict):
        return _verdict("INCOMPLETE", "missing finite-state/update/mechanism guards")
    if expected.get("finite_state") and guards.get("all_finite") is not True:
        return _verdict("FAIL" if guards.get("all_finite") is False else "INCOMPLETE",
                        "finite-state guard did not pass")
    if not isinstance(guards.get("optimizer_updates", {}), dict):
        return _verdict("INCOMPLETE", "optimizer update counts must be an object")
    for role in expected.get("optimizer_roles", []):
        count = guards.get("optimizer_updates", {}).get(role)
        if type(count) is not int or count < 0:
            return _verdict("INCOMPLETE", f"missing valid {role} optimizer update count")
        from .priors import expected_prior_updates
        expected_count = (expected_prior_updates(task, evidence, task["execution"]["steps"])
                          if role == "prior" else task["execution"]["steps"])
        if count == 0 and expected_count != 0:
            return _verdict("FAIL", f"{role} did not perform an intended optimizer update")
        if (expected.get("exact_optimizer_updates") or expected_count == 0) and count != expected_count:
            return _verdict("INCOMPLETE", f"{role} optimizer updates do not complete the declared task budget")
    if expected.get("mechanism_exercised"):
        from .mechanisms import mechanism_blockers
        reasons = mechanism_blockers(guards.get("mechanism_audit"))
        if guards.get("hooks_exercised") is not True or reasons:
            return _verdict("BLOCKED", "; ".join(reasons) or "required mechanism activation has not been demonstrated")
    if expected.get("rng_isolation"):
        deviations = guards.get("unintended_rng_deviations")
        if type(deviations) is not int or deviations < 0:
            return _verdict("INCOMPLETE", "missing RNG deviation audit")
        if deviations:
            return _verdict("INVALID", "unintended RNG stream deviations invalidate comparison")
    return None


def _transfer(task, evidence):
    from benchmarks.transfer_suite.protocol import test_verdict
    from .transfer_cadence import test_verdict as test_verdict_declared

    evaluation = task["evaluation"]
    declared = evaluation.get("evaluator") == "experiments.forge.transfer_cadence:test_verdict"
    count = evaluation["observations"] if declared else 24
    spec = {"steps": task["execution"]["steps"], "thresholds": evaluation["thresholds"], "observations": count}
    expected = [math.ceil(i * spec["steps"] / count) for i in range(1, count + 1)]
    points = evidence.get("observations", evidence.get("curve"))
    try:
        _curve(points, spec["thresholds"])
    except ValueError as exc:
        return _verdict("INVALID" if points else "INCOMPLETE", str(exc))
    if [p["step"] for p in points] != expected:
        return _verdict("INCOMPLETE", f"all {count} declared observation checkpoints are required")
    live = evidence.get("live")
    if not isinstance(live, dict) or any(not _finite(live.get(key)) for key, _, _ in spec["thresholds"]):
        return _verdict("INCOMPLETE", "missing finite final live metrics")
    grade = (test_verdict_declared if declared else test_verdict)(spec, {"observations": points, "live": live})
    return _verdict(grade["status"], "recomputed complete live curve and terminal suffix",
                    metrics=live, evaluator_result=grade)


def _zero_tanh_smoke(task, evidence):
    """Grade the declared origin-Dirac endpoint, without adding hold gates."""
    from benchmarks.locked_shared.baseline import score_metrics
    from .tanh_point_mass import accepts_origin_smoke

    if not accepts_origin_smoke(task, task["execution"].get("host_definition")):
        return _verdict("BLOCKED", "only the declared origin zero_tanh_smoke host is supported")
    evaluation = task["evaluation"]
    count = evaluation.get("observations", 24)
    _positive_int(count, "evaluation observations")
    steps = task["execution"]["steps"]
    # Match the adapter's deduplicated cadence, including tiny software runs.
    expected = sorted({math.ceil(i * steps / count) for i in range(1, count + 1)})
    points = evidence.get("observations", evidence.get("curve"))
    try:
        _curve(points, evaluation["thresholds"])
    except ValueError as exc:
        return _verdict("INVALID" if points else "INCOMPLETE", str(exc))
    if [point["step"] for point in points] != expected:
        return _verdict("INCOMPLETE", "all declared zero-tanh observation checkpoints are required")
    live = evidence.get("live")
    if (not isinstance(live, dict) or live.get("step") != steps
            or any(not _finite(live.get(name)) for name, _, _ in evaluation["thresholds"])):
        return _verdict("INCOMPLETE", "finite final live metrics at the complete update budget are required")
    samples = evaluation.get("eval_samples", 4096)
    if any(type(row.get("sample_count")) is not int or row["sample_count"] != samples
           for row in [*points, live]):
        return _verdict("INCOMPLETE", "every zero-tanh observation requires the declared sample count")
    cells = score_metrics(live, evaluation["thresholds"])
    passed = all(cell["status"] == "PASS" for cell in cells)
    return _verdict("PASS" if passed else "FAIL", "recomputed the declared final zero-tanh bounds",
                    metrics=live, evaluator_result={"passed": passed, "metrics": cells})


def _native(task, evidence, *, diagnostic=False):
    from benchmarks.toy100.accuracy_gate import evaluate_suite
    from .artifacts import verify_artifacts

    artifact = evidence.get("artifact_root")
    if not isinstance(artifact, str) or not artifact:
        return _verdict("INCOMPLETE", "native coverage/accuracy needs saved samples and artifact_root")
    manifest = evidence.get("artifact_manifest")
    if manifest is None:
        return _verdict("INCOMPLETE", "native evaluator inputs require a certified artifact manifest")
    path = Path(artifact)
    verify_artifacts(path, manifest)
    problem = task["execution"]["problem"]
    config = _read(path / problem / "config.json")
    if config.get("steps") != task["execution"]["steps"]:
        return _verdict("INVALID", "native artifact budget differs from declared task")
    if config.get("eval_interval", 250) != task["evaluation"]["eval_interval"]:
        return _verdict("INVALID", "native artifact observation cadence differs from declared task")
    if config.get("eval_samples") != task["evaluation"]["eval_samples"]:
        return _verdict("INVALID", "native evaluation sample count differs from declared task")
    if config.get("early_eval_steps") != task["evaluation"]["early_eval_steps"]:
        return _verdict("INVALID", "native early observation schedule differs from declared task")
    if task["execution"].get("preserve_prefix_steps"):
        prefix = evidence.get("prefix_parity", {})
        if prefix.get("steps") != task["execution"]["preserve_prefix_steps"] or not _digest(prefix.get("reference_sha256")):
            return _verdict("INCOMPLETE", "continuation requires bound original-prefix parity evidence")
        if prefix.get("continued_sha256") != prefix["reference_sha256"]:
            return _verdict("FAIL", "continuation changed the original training prefix")
        from .continuation import verify_native_prefix
        verify_native_prefix(task, evidence, diagnostic=diagnostic)
    grade = evaluate_suite(path, problem=problem, write=False)["problems"][problem]
    verify_artifacts(path, manifest)
    return _verdict(grade["status"], grade["reason"],
                    metrics=grade.get("holdout_metrics", {}), evaluator_result=grade)


# Evidence identity of the ring grader. Task declarations (ring_hold,
# ring_extension and their recipe-prior-refactor variants) keep binding the
# archived path and hash below, so their evaluation fingerprints and every
# recorded verdict are unchanged. Scoring code is imported from
# ``benchmarks/``; the archived copy under ``reports/`` is evidence only and is
# never placed on the import path.
CONVERGENCE_GATE_RECEIPT = {
    "module": "benchmarks.toy100.k3p_convergence_gate",
    "source": "benchmarks/toy100/k3p_convergence_gate.py",
    "archived_source": "reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py",
    "sha256": "a29c8c21083195c2944a2b2509ec7d4b96e6fc1eb7c3fa6df70d51dea6fe1ce0",
    "archived_git_blob": "736563558b82bf7385881699cb12f59739f02e99",
    "archived_commit": "74e1bae1e9c890d84debfff88eac5b16604b791f",
    "relation": "byte_identical_copy",
}


def _convergence_class():
    """The archived K3P ring ``ConvergenceGate``, loaded from ``benchmarks/``.

    The imported module must stay byte-identical to the archived evidence
    source (same SHA-256 and Git blob), otherwise grading refuses.
    """
    from benchmarks.toy100 import k3p_convergence_gate as module
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest != CONVERGENCE_GATE_RECEIPT["sha256"]:
        raise ValueError("ring convergence grader differs from its archived evidence identity "
                         f"({CONVERGENCE_GATE_RECEIPT['archived_source']})")
    return module.ConvergenceGate


def _ring(task, evidence):
    evaluation = task["evaluation"]
    points = evidence.get("dense", evidence.get("observations"))
    if not points:
        return _verdict("INCOMPLETE", "dense first-convergence and hold observations are required")
    _curve(points, [["modes", ">=", 8], ["hq", ">=", .9]])
    if any(type(p["modes"]) is not int or not 0 <= p["modes"] <= 8 or not 0 <= p["hq"] <= 1 for p in points):
        return _verdict("INVALID", "ring metrics have invalid mode counts or quality bounds")
    _curve(points, [], range(evaluation["start_step"] + 1, points[-1]["step"] + 1))
    if points[-1]["step"] > evaluation["max_total_steps"]:
        return _verdict("INVALID", "ring observations exceed the frozen total budget")
    gate = _convergence_class()(confirmation=evaluation["confirmation_checks"],
                               settling_budget=evaluation["settling_budget"],
                               hold_budget=evaluation["hold_budget"],
                               start_step=evaluation["start_step"])
    for point in points:
        if gate.done:
            break
        gate.observe(point)
    if gate.status in {"NOT_CONVERGED", "POST_CONVERGENCE_FAIL"}:
        return _verdict("FAIL", "first qualifying window failed acquisition or uninterrupted hold",
                        metrics=gate.summary())
    if gate.status != "PASS":
        return _verdict("INCOMPLETE", "first-convergence hold has not completed", metrics=gate.summary())
    if evaluation["kind"] == "ring_extension":
        extension = [p for p in points if p["step"] > gate.last_step]
        if len(extension) != evaluation["extension_steps"]:
            return _verdict("INCOMPLETE", "the complete immediately following extension is required")
        continuity = evidence.get("continuity", {})
        if continuity.get("mode") != "uninterrupted" or not continuity.get("run_id"):
            return _verdict("BLOCKED", "extension requires bound uninterrupted own-state provenance")
        passing = sum(p["modes"] == 8 and .9 <= p["hq"] <= 1 for p in extension)
        return _verdict("PASS" if passing == len(extension) else "FAIL",
                        "recomputed every extension observation without selecting another window",
                        metrics={**gate.summary(), "extension_checks": len(extension),
                                 "extension_passing": passing})
    return _verdict("PASS", "recomputed first-convergence hold", metrics=gate.summary())


def _adaptation(task, evidence):
    from benchmarks.toy100.continuous_probe import _window, match_frozen_control

    active, frozen = deepcopy(evidence.get("active")), deepcopy(evidence.get("frozen"))
    if not isinstance(active, dict) or not isinstance(frozen, dict):
        return _verdict("INCOMPLETE", "matched active and frozen raw control evidence is required")
    if not evidence.get("artifact_root") or not evidence.get("artifact_manifest"):
        return _verdict("INCOMPLETE", "paired adaptation requires certified own-state and control artifacts")
    from .adaptation import verify_pair_artifacts
    verify_pair_artifacts(evidence)
    evaluation = task["evaluation"]
    required = ("config_sha256", "source_sha256", "runtime", "mode", "noise_horizon",
                "diagnostic_every", "dense_after", "dense_until", "shift", "shift_pair", "optimizer_final")
    from .tier1_policy import schedule_free_policy
    from .contracts import read_json
    # A schedule-free policy recipe has no noise/schedule horizon; its actual
    # None is verified against the saved recipe by verify_pair_artifacts.
    optional = {"dense_after", "dense_until"} | ({"noise_horizon"} if schedule_free_policy(task) else set())
    if "noise_horizon" not in optional and active.get("noise_horizon") is None:
        try:
            cfg = read_json(Path(evidence["artifact_root"]) / "config.json")
            if cfg.get("recipe", {}).get("total_steps") is None:
                optional.add("noise_horizon")
        except Exception:
            pass
    for run in (active, frozen):
        if any(key not in run or (run[key] is None and key not in optional) for key in required):
            return _verdict("INCOMPLETE", "paired control lacks complete provenance/optimizer state")
        if run.get("steps") != evaluation["steps"] or run.get("shift_step") != evaluation["shift_step"]:
            return _verdict("INVALID", "shift timing differs from task")
        points = run.get("diagnostic")
        _curve(points, [["modes", ">=", 8], ["hq", ">=", .9]])
        if any(type(p["modes"]) is not int or not 0 <= p["modes"] <= 8 or not 0 <= p["hq"] <= 1 for p in points):
            return _verdict("INVALID", "invalid ring mode count or quality bounds")
        if run["diagnostic_every"] != evaluation["diagnostic_every"]:
            return _verdict("INVALID", "diagnostic observation cadence differs from protocol")
        if [p["step"] for p in points] != list(range(10, evaluation["steps"] + 1, 10)):
            return _verdict("INCOMPLETE", "paired adaptation requires the complete declared diagnostic curve")
        at_shift = run["shift_pair"].get("optimizer_at_shift", [])
        if not at_shift or any(r.get("updates") != evaluation["shift_step"] for r in at_shift):
            return _verdict("INVALID", "optimizer state at the target shift is not preserved")
        by_step = {p["step"]: p for p in points}
        stationary = list(range(1000, 1201, 50))
        continued = list(range(1210, evaluation["shift_step"] + 1, 10))
        recovery = list(range(evaluation["shift_step"] + 10, evaluation["steps"] + 1, 10))
        if any(step not in by_step for step in stationary + continued + recovery):
            return _verdict("INCOMPLETE", "complete stationary, hold and recovery windows are required")
        run["stationary"] = _window([by_step[s] for s in stationary])
        run["continued_hold"] = _window([by_step[s] for s in continued])
        deadline = evaluation["shift_step"] + evaluation["recovery_deadline"]
        late = _window([by_step[s] for s in recovery if s >= deadline])
        run["shift_recovery"] = {"deadline_window": late, "deadline_pass": late["pass_all"]}
    for run in (active, frozen):
        final = run["optimizer_final"]
        expected = evaluation["shift_step"] if run.get("freeze_after_shift") else evaluation["steps"]
        if not isinstance(final, list) or not final or any(r.get("updates") != expected for r in final):
            return _verdict("INVALID", "optimizer updates do not match declared active/frozen continuation")
    grade = match_frozen_control(active, frozen)
    return _verdict(grade["status"], "recomputed live recovery and matched frozen negative control",
                    metrics={"live_deadline": active["shift_recovery"]["deadline_window"],
                             "frozen_deadline": frozen["shift_recovery"]["deadline_window"]})


def _clockfree(task, evidence):
    if not evidence.get("artifact_root") or not evidence.get("artifact_manifest"):
        return _verdict("INCOMPLETE", "clock comparisons require certified saved state artifacts")
    from .clockfree import verify_probe
    comparisons, audit = verify_probe(task, evidence)
    if not isinstance(comparisons, list) or not isinstance(audit, dict):
        return _verdict("INCOMPLETE", "state/horizon parity comparisons and source audit are required")
    conditions = task["evaluation"]["conditions"]
    names = [r.get("condition") for r in comparisons if isinstance(r, dict)]
    if sorted(names) != sorted(conditions):
        return _verdict("INCOMPLETE", "each declared clock/evaluation/restart comparison is required exactly once")
    for row in comparisons:
        for key in ("permitted_state_sha256", "rng_state_sha256", "reference_sha256", "changed_sha256"):
            value = row.get(key)
            if not _digest(value):
                return _verdict("INCOMPLETE", "parity comparisons require state and output digests")
        if row["reference_sha256"] != row["changed_sha256"]:
            return _verdict("FAIL", f"{row['condition']} changed the update or common-prefix state")
    hashes = audit.get("source_sha256")
    dependencies = audit.get("unexplained_clock_dependencies")
    if (not isinstance(hashes, dict) or not hashes or not all(_digest(v) for v in hashes.values())
            or not isinstance(audit.get("allowed_state"), list) or not audit["allowed_state"]
            or not isinstance(dependencies, list)):
        return _verdict("INCOMPLETE", "source audit must bind code and explain permitted state/counters")
    if audit["unexplained_clock_dependencies"]:
        if task["execution"].get("clock_audit_scope") == "measure_known_dependencies":
            return _verdict("FAIL", "; ".join(audit["unexplained_clock_dependencies"]),
                            metrics={"clock_dependency_count": len(audit["unexplained_clock_dependencies"]), "parity_comparisons": len(comparisons)})
        return _verdict("BLOCKED", "unexplained clock dependencies prevent clock-free eligibility")
    return _verdict("PASS", "declared state/horizon/cadence/restart comparisons agree; source audit bound",
                    metrics={"parity_comparisons": len(comparisons)})


def _zero_tanh(task, evidence):
    evaluation = task["evaluation"]
    thresholds = evaluation["thresholds"]
    points = evidence.get("observations", evidence.get("curve"))
    if not isinstance(points, list) or not points:
        return _verdict("INCOMPLETE", "missing zero_tanh observation curve")
    try:
        _curve(points, thresholds)
    except ValueError as exc:
        return _verdict("INVALID" if points else "INCOMPLETE", str(exc))
    live = evidence.get("live")
    if not isinstance(live, dict) or any(not _finite(live.get(key)) for key, _, _ in thresholds):
        return _verdict("INCOMPLETE", "missing finite final live metrics")
    failed = []
    for name, op, bound in thresholds:
        val = live.get(name)
        ok = (val <= bound if op == "<=" else
              val >= bound if op == ">=" else
              val == bound if op == "==" else False)
        if not ok:
            failed.append(f"{name} {op} {bound}")
    passed = len(failed) == 0
    status = "PASS" if passed else "FAIL"
    reason = ("recomputed origin point-mass contraction gates"
              if passed else f"failed thresholds: {', '.join(failed)}")
    return _verdict(status, reason, metrics=live,
                    evaluator_result={"passed": passed, "failed_bounds": failed})


STRETCH_EVALUATOR_KINDS = frozenset({
    "zero_tanh_rail_escape",
    "misgan_block_recovery",
    "misgan_block_neural_imputer",
    "misgan_mcar_p50_recovery",
    "misgan_mcar_p80_anchored",
    "pacgan_unequal_mass",
    "pacgan8_grid100",
    "ring16_bigan_continuous",
    "alice_spiral_bigan",
    "infogan_anisotropic_modes",
    "cgan_overlapping_classes",
    "cramer_energy_overlap",
    "cyclegan_unpaired_constellation",
    "interp_sparse_arc",
    "interp_img_patch_shift",
    "interp_saddle_barycentric",
})


def _stretch_thresholds(task, evidence):
    evaluation = task["evaluation"]
    thresholds = evaluation["thresholds"]
    raw_points = evidence.get("observations", evidence.get("curve"))
    if not isinstance(raw_points, list) or not raw_points:
        return _verdict("INCOMPLETE", f"missing {evaluation['kind']} observation curve")
    points = []
    for p in raw_points:
        if not isinstance(p, dict):
            return _verdict("INVALID", "observation entry must be an object")
        nested = p.get("metrics") if isinstance(p.get("metrics"), dict) else {}
        merged = {**nested, **{k: v for k, v in p.items() if k != "metrics"}}
        points.append(merged)
    try:
        _curve(points, thresholds)
    except ValueError as exc:
        return _verdict("INVALID" if points else "INCOMPLETE", str(exc))
    live = evidence.get("live")
    if not isinstance(live, dict):
        live = {k: v for k, v in points[-1].items() if k not in {"step", "passed", "failed_bounds"}}
    if any(not _finite(live.get(key)) for key, _, _ in thresholds):
        return _verdict("INCOMPLETE", "missing finite final live metrics")

    def _point_failures(pt):
        bad = []
        for name, op, bound in thresholds:
            val = pt.get(name)
            ok = (val <= bound if op == "<=" else
                  val >= bound if op == ">=" else
                  val == bound if op == "==" else False)
            if not ok:
                bad.append(f"{name} {op} {bound}")
        return bad

    min_stable = int(evaluation.get("minimum_stable_checks", 1))
    if len(points) < min_stable:
        return _verdict("INCOMPLETE", f"requires at least {min_stable} observations, got {len(points)}")
    tail = points[-min_stable:]
    tail_failures = [_point_failures(pt) for pt in tail]
    terminal_passed = sum(1 for f in tail_failures if not f)
    failed = list(dict.fromkeys(b for f in tail_failures for b in f))
    passed = terminal_passed == len(tail)
    if evaluation["kind"] == "zero_tanh_rail_escape":
        shift_step = int(task.get("execution", {}).get("shift_step", 1001))
        recovery_deadline = int(evaluation.get("recovery_deadline", 400))
        pre_shift = [pt for pt in points if pt.get("step") == shift_step - 1]
        post_deadline = [pt for pt in points if int(pt.get("step", 0)) >= shift_step + recovery_deadline]
        if pre_shift and any(_point_failures(pt) for pt in pre_shift):
            passed = False
            for pt in pre_shift:
                for b in _point_failures(pt):
                    if f"pre_shift:{b}" not in failed:
                        failed.append(f"pre_shift:{b}")
        if post_deadline and any(_point_failures(pt) for pt in post_deadline):
            passed = False
            for pt in post_deadline:
                for b in _point_failures(pt):
                    if b not in failed:
                        failed.append(b)
    status = "PASS" if passed else "FAIL"
    reason = (f"recomputed {evaluation['kind']} gates ({terminal_passed}/{len(tail)} terminal checks passed)"
              if passed else f"failed {evaluation['kind']} thresholds: {', '.join(failed)}")
    return _verdict(status, reason, metrics=live,
                    evaluator_result={"passed": passed, "terminal_passed": terminal_passed,
                                      "minimum_stable_checks": min_stable, "failed_bounds": failed})


def grade_result(task: dict, result: dict | None, *, diagnostic=False) -> dict:
    """Independently grade compatible raw evidence, without changing it."""
    try:
        _validate_measurement_contract(task)
    except (KeyError, TypeError, ValueError) as exc:
        return _verdict("INVALID", str(exc))
    if result is None:
        return _verdict("NOT_RUN", "no compatible result")
    if not isinstance(result, dict):
        return _verdict("INVALID", "receipt must be an object")
    applicability = result.get("applicability", {})
    if isinstance(applicability, dict) and applicability.get("status") in {"unsupported", "unknown"}:
        return _verdict("BLOCKED", applicability.get("reason") or "host applicability has not been demonstrated",
                        raw_status=result.get("gate_status", result.get("status")))
    raw = result.get("gate_status", result.get("status"))
    if raw == "BLOCKED":
        return _verdict("BLOCKED", result.get("reason", "execution blocked"))
    if raw in {"ERROR", "error", "timeout", "cancelled"} or result.get("error"):
        return _verdict("INCOMPLETE", result.get("error") or f"execution ended with {raw}", raw_status=raw)
    if result.get("gate_status") in {"INCOMPLETE", "INVALID"}:
        return _verdict(result["gate_status"], result.get("reason") or "execution did not produce valid complete evidence")
    evidence = result.get("evidence", result.get("metrics"))
    if not isinstance(evidence, dict):
        return _verdict("INCOMPLETE", "missing raw evaluator evidence; status stamps cannot qualify")
    from .priors import validate_prior_policy
    applied = result.get("applied", result)
    prior_grade = validate_prior_policy(task, evidence, applied.get("recipe", {}) if isinstance(applied, dict) else None)
    if prior_grade is not None:
        return _verdict(prior_grade["status"], prior_grade["reason"])
    if evidence.get("scoring_weights", "live") != task["evaluation"].get("scoring_weights", "live"):
        return _verdict("INVALID", "live/EMA scoring policies differ")
    sampling = grade_sampling(task, evidence)
    if sampling is not None:
        return _verdict(sampling["status"], sampling["reason"])
    if "linear_precision" in task.get("execution", {}):
        from .precision import grade as precision_grade
        precision = precision_grade(task, evidence)
        if precision is not None:
            return _verdict(*precision)
    if task.get("adapter") == "particle_interpolation":
        from .interpolation import grade as interpolation_grade
        interp_check = interpolation_grade(task, evidence)
        if interp_check is not None:
            return _verdict(*interp_check)
    from .tier1_policy import is_policy_task
    if is_policy_task(task) and task["adapter"] != "clockfree_audit":
        from .tier1_policy import validate_evidence
        policy_grade = validate_evidence(task, evidence)
        if policy_grade is not None:
            return _verdict(policy_grade["status"], policy_grade["reason"])
    guard = _guards(task, evidence)
    if guard is not None:
        return guard
    if is_policy_task(task):
        from .policy_gaussian import grade as gaussian_grade
    else:
        from .gaussian_tasks import grade as gaussian_grade
    from .word_tasks import grade as word_grade
    from .scaling_adapter import grade as performance_grade
    graders = {"gaussian_smoke": gaussian_grade, "gaussian_stability": gaussian_grade,
               "word_smoke": word_grade, "word_hold": word_grade,
               "transfer_sustained": _transfer, "native_accuracy": _native,
               "ring_hold": _ring, "ring_extension": _ring,
               "paired_adaptation": _adaptation, "clockfree_parity": _clockfree,
               "image_training_performance": performance_grade,
               "zero_tanh_smoke": _zero_tanh_smoke,
               **{kind: _stretch_thresholds for kind in STRETCH_EVALUATOR_KINDS}}
    grader = graders.get(task["evaluation"]["kind"])
    if grader is None:
        return _verdict("BLOCKED", f"unsupported evaluator {task['evaluation']['kind']}")
    try:
        from .tier1_policy import diagnostic_continuation_allowed
        # Only a registered cohort's declared Gaussian stability variant may
        # continue its own scoped smoke state after a failed gate; every other
        # task keeps native-only diagnostic checkpoint grading.
        gaussian_continuation = (task["evaluation"]["kind"] == "gaussian_stability"
                                 and diagnostic_continuation_allowed(task))
        scope = (evidence.get("continuity") or {}) if gaussian_continuation else {}
        if diagnostic and task.get("adapter") != "native100_continuation" and not gaussian_continuation:
            return _verdict("INVALID", "diagnostic checkpoint grading is limited to native continuations")
        if gaussian_continuation and (scope.get("diagnostic_continuation") is True) != bool(diagnostic):
            return _verdict("INVALID", "diagnostic Gaussian continuation requires its explicit nonqualifying scope"
                            if diagnostic else "diagnostic continuation cannot supply ordinary qualification")
        if diagnostic and gaussian_continuation and scope.get("parent_smoke_status") not in {"PASS", "FAIL"}:
            return _verdict("INVALID", "diagnostic Gaussian continuation must record its parent's measured gate")
        grade = (_native(task, evidence, diagnostic=True) if diagnostic and not gaussian_continuation
                 else grader(task, evidence))
        if diagnostic:
            grade.update(qualification_input=False, qualification_reuse=False)
        # Normalized metrics must not replace the raw curve when the input used
        # the metrics fallback. The queue can safely merge this grade into it.
        if "evidence" not in result:
            grade["evidence"] = deepcopy(evidence)
        return grade
    except FileNotFoundError as exc:
        return _verdict("INCOMPLETE", f"required saved evidence unavailable: {exc}")
    except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
        return _verdict("INVALID", str(exc))


def _candidate_eligibility(view, candidate):
    rules = view.get("eligibility", {})
    required = rules.get("requires_capabilities", [])
    claims = rules.get("claim_contract", {})
    if not required and not claims:
        return []
    if not isinstance(candidate, dict):
        return ["candidate capability/claim metadata is required for this view"]
    available = candidate.get("capabilities", candidate.get("provides_capabilities", []))
    reasons = [f"missing capability {name}" for name in required
               if (available.get(name) is not True if isinstance(available, dict) else name not in available)]
    actual = candidate.get("claim_contract", {})
    reasons.extend(f"claim_contract.{name} must equal {value!r}"
                   for name, value in claims.items() if actual.get(name) != value)
    return reasons


def check_summary(grade: dict) -> dict | None:
    """Extract continuous check pass counts (k/M) from a graded task result."""
    if not isinstance(grade, dict):
        return None
    passed = total = None
    evaluator = grade.get("evaluator_result")
    if isinstance(evaluator, dict):
        checks = evaluator.get("checks")
        if isinstance(checks, dict) and checks:
            if all(isinstance(v, bool) for v in checks.values()):
                passed = sum(1 for v in checks.values() if v)
                total = len(checks)
            elif all(isinstance(v, dict) for v in checks.values()):
                passed = sum(1 for v in checks.values() if v.get("passed") is True or v.get("status") == "PASS")
                total = len(checks)
        elif isinstance(checks, list) and checks and all(isinstance(c, dict) for c in checks):
            passed = sum(1 for c in checks if c.get("passed") is True or c.get("status") == "PASS")
            total = len(checks)
        elif type(evaluator.get("passing_checks")) is int and type(evaluator.get("observations")) is int:
            passed, total = evaluator["passing_checks"], evaluator["observations"]
    if total is None:
        metrics = grade.get("metrics")
        if isinstance(metrics, dict):
            for pass_key, total_key in (
                ("passing_checks", "observations"),
                ("extension_passing", "extension_checks"),
                ("passed_checks", "total_checks"),
            ):
                if type(metrics.get(pass_key)) is int and type(metrics.get(total_key)) is int:
                    passed, total = metrics[pass_key], metrics[total_key]
                    break
    if type(passed) is int and type(total) is int and total > 0 and 0 <= passed <= total:
        return {"passed_checks": passed, "total_checks": total, "check_pass_rate": round(passed / total, 6)}
    return None


def qualify(view: dict, tasks: dict, results: list[dict], *, candidate: dict | None = None,
            allow_held_highest_tier: bool = False) -> dict:
    """Reduce one candidate's compatible receipts; tier attainment is computed.

    Stretch and held tasks are graded and shown but never counted in required
    totals. A tier with no required task has status STRETCH (when its nonrequired
    tasks are stretch) or HELD (legacy held): it neither passes nor stops
    progression. Pure stretch tiers never block a view from reporting PASS once
    every counted tier passes; legacy held highest tiers still report HELD unless
    allow_held_highest_tier is explicitly authorized.
    """
    validate_view(view, tasks)
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    indexed = {}
    for receipt in results:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("task_id"), str):
            raise ValueError("each receipt needs task_id")
        indexed.setdefault(receipt["task_id"], []).append(receipt)
    assignments = sorted(view["assignments"], key=lambda a: (a["qualification_tier"], a["order"], a["task"]))
    rows, selected = {}, {}
    for assignment in assignments:
        name = assignment["task"]
        receipts = indexed.get(name, [])
        grades = [grade_result(tasks[name], receipt) for receipt in receipts]
        if not grades:
            grade = grade_result(tasks[name], None)
        elif len({_hash(grade) for grade in grades}) > 1:
            grade = _verdict("INVALID", "conflicting compatible attempts require explicit resolution")
        else:
            grade = grades[0]
            selected[name] = receipts[0]
        summary = check_summary(grade)
        rows[name] = {"task_id": name, "qualification_tier": assignment["qualification_tier"],
                      "importance": assignment["importance"], **grade,
                      **({"check_summary": summary} if summary is not None else {}),
                      **{key: deepcopy(assignment[key]) for key in ("hold", "stretch", "removal") if key in assignment},
                      "cost": sum(_cost(r) for r in receipts)}
        rows[name].pop("evidence", None)
    # Retiering may place children before parents in presentation order. Resolve
    # prerequisite statuses in dependency order before exposing any PASS rows.
    ordered, visited = [], set()
    def dependency_order(name):
        if name in visited:
            return
        for dep in _dependencies(tasks[name]):
            dependency_order(dep["task"])
        visited.add(name)
        ordered.append(name)
    for name in rows:
        dependency_order(name)
    for name in ordered:
        row = rows[name]
        if row["status"] != "PASS":
            continue
        for dep in _dependencies(tasks[name]):
            if rows[dep["task"]]["status"] != "PASS":
                row.update(_verdict("BLOCKED", f"dependency {dep['task']} has not passed"))
                continue
            if dep["kind"] == "checkpoint":
                child = selected.get(name, {}).get("evidence", {})
                parent = selected.get(dep["task"], {}).get("evidence", {})
                a, b = child.get("continuity") or {}, parent.get("continuity") or {}
                uninterrupted = a.get("mode") == b.get("mode") == "uninterrupted" and a.get("run_id") and a.get("run_id") == b.get("run_id")
                native_resume = False
                if tasks[name]["adapter"] == "native100_continuation":
                    # grade_result has already verified actual prefix/restore
                    # artifacts. Bind that proof to this selected parent too.
                    proof = child.get("prefix_parity", {})
                    prerequisite = proof.get("prerequisite", {})
                    producer = selected.get(dep["task"], {})
                    native_resume = bool(parent.get("checkpoint") and
                        proof.get("checkpoint") == parent["checkpoint"] and
                        proof.get("artifact_manifest") == parent.get("artifact_manifest") and
                        prerequisite.get("compatibility_key") == producer.get("compatibility_key") and
                        (not producer.get("_attempt_id") or prerequisite.get("attempt_id") == producer["_attempt_id"]))
                if not uninterrupted and not native_resume:
                    reason = ("fresh-process checkpoint restoration has no supported proof contract; uninterrupted continuation required"
                              if a.get("mode") == "verified_checkpoint" else
                              "checkpoint continuation is not bound to the producer's own state")
                    row.update(_verdict("BLOCKED", reason))
    held_rows = [row for row in rows.values() if row["importance"] == "held"]
    stretch_rows = [row for row in rows.values() if is_stretch(row)]
    held_summary = {
        "held_total": len(held_rows),
        "held_passed": sum(row["status"] == "PASS" for row in held_rows),
        "held_tasks": [row["task_id"] for row in held_rows],
        "stretch_total": len(stretch_rows),
        "stretch_passed": sum(row["status"] == "PASS" for row in stretch_rows),
        "stretch_tasks": [row["task_id"] for row in stretch_rows],
    }
    if view.get("evidence_scope") in DIAGNOSTIC_SCOPES:
        return {"view": view["id"], "view_revision": view["revision"], "policy_fingerprint": view_fingerprint(view),
                "status": "DIAGNOSTIC", "qualified_tier": 0, "eligible": False, "counted_eligible": False,
                "evidence_scope": view["evidence_scope"], "current_qualification_reuse": False,
                "qualification_input": False, "qualification_reuse": False,
                "diagnostic_complete": all(row["status"] in {"PASS", "FAIL"} for row in rows.values()),
                "tiers": [], "tasks": list(rows.values()),
                "task_statuses": {name: row["status"] for name, row in rows.items()},
                "next_task": None, "next_tasks": [], "blockers": [], "required_passed": 0, "required_total": 0,
                **held_summary, "held_tiers": [], "stretch_tiers": [], "counted_tiers": [],
                "cost_seconds": sum(row["cost"] for row in rows.values())}
    index = {a["task"]: a for a in assignments}

    def ready(name):
        # A removed phase runs inside its counted child's uninterrupted job, so
        # its own prerequisites decide whether that job can be proposed.
        for dep in _dependencies(tasks[name]):
            parent = dep["task"]
            if rows[parent]["status"] == "PASS":
                continue
            if (rows[parent]["status"] == "NOT_RUN" and dep["kind"] == "checkpoint"
                    and _phase_parent(index, tasks, parent, name) and ready(parent)):
                continue
            return False
        return True

    eligibility = _candidate_eligibility(view, candidate)
    tiers, qualified_tier, next_tasks, blockers = [], 0, [], []
    stopped = False
    for tier in sorted({a["qualification_tier"] for a in assignments}):
        subset = [rows[a["task"]] for a in assignments if a["qualification_tier"] == tier]
        required = [row for row in subset if row["importance"] == "required"]
        held = [row for row in subset if row["importance"] == "held"]
        stretch = [row for row in subset if is_stretch(row)]
        has_explicit_stretch = any(row["importance"] == "stretch" for row in subset)
        shared = {
            "held_total": len(held),
            "held_passed": sum(row["status"] == "PASS" for row in held),
            "diagnostic_total": sum(r["importance"] == "diagnostic" for r in subset),
            **({"stretch_total": len(stretch), "stretch_passed": sum(row["status"] == "PASS" for row in stretch)}
               if has_explicit_stretch else {}),
        }
        if not required:
            tier_status = "STRETCH" if (stretch and not held) else "HELD"
            tiers.append({"qualification_tier": tier, "status": tier_status, "required_passed": 0,
                          "required_total": 0, "counts": {}, **shared})
            continue
        counts = dict(Counter(row["status"] for row in required))
        all_pass = all(row["status"] == "PASS" for row in required)
        status = "PASS" if all_pass else next((s for s in ("FAIL", "INVALID", "BLOCKED", "INCOMPLETE", "NOT_RUN") if counts.get(s)), "INCOMPLETE")
        tiers.append({"qualification_tier": tier, "status": status,
                      "required_passed": counts.get("PASS", 0), "required_total": len(required),
                      "counts": counts, **shared})
        if stopped:
            continue
        if all_pass and not eligibility:
            qualified_tier = tier
            continue
        stopped = True
        blockers = [{"task_id": row["task_id"], "status": row["status"], "reasons": row["reasons"]}
                    for row in required if row["status"] != "PASS"]
        hard_stop = eligibility or any(row["status"] in {"FAIL", "INVALID", "BLOCKED", "INCOMPLETE"} for row in required)
        if not hard_stop:
            next_tasks = [row["task_id"] for row in required if row["status"] == "NOT_RUN" and ready(row["task_id"])]
    if eligibility:
        blockers = [{"task_id": None, "status": "BLOCKED", "reasons": eligibility}, *blockers]
    counted = [t["qualification_tier"] for t in tiers if t["status"] not in {"HELD", "STRETCH"}]
    non_stretch = [t for t in tiers if t["status"] != "STRETCH"]
    complete = bool(counted) and qualified_tier == counted[-1]
    counted_eligible = bool(complete and not eligibility)
    allow_held = bool(allow_held_highest_tier or view.get("eligibility", {}).get("allow_held_highest_tier"))
    status = ("BLOCKED" if eligibility else
              "PASS" if complete and (non_stretch[-1]["status"] != "HELD" or allow_held) else
              "STRETCH" if not non_stretch and bool(tiers) else
              "HELD" if complete or not counted else
              next((t["status"] for t in tiers if t["status"] not in {"HELD", "STRETCH"} and t["qualification_tier"] > qualified_tier),
                   "INCOMPLETE"))
    if status == "NOT_RUN":
        status = "INCOMPLETE"
    return {"view": view["id"], "view_revision": view["revision"], "policy_fingerprint": view_fingerprint(view),
            "status": status, "qualified_tier": qualified_tier, "eligible": status == "PASS",
            "counted_eligible": counted_eligible,
            "tiers": tiers, "tasks": list(rows.values()), "task_statuses": {name: row["status"] for name, row in rows.items()},
            "next_task": next_tasks[0] if next_tasks else None, "next_tasks": next_tasks,
            "blockers": blockers, "required_passed": sum(row["status"] == "PASS" for row in rows.values() if row["importance"] == "required"),
            "required_total": sum(row["importance"] == "required" for row in rows.values()),
            **held_summary,
            "held_tiers": [t["qualification_tier"] for t in tiers if t["status"] == "HELD"],
            "stretch_tiers": [t["qualification_tier"] for t in tiers if t["status"] in {"HELD", "STRETCH"}],
            "counted_tiers": counted,
            "cost_seconds": sum(row["cost"] for row in rows.values())}
