"""Bounded deterministic Recipe grids through ordinary Forge gates.

Trials are declarations, not trainer copies. A study freezes one protocol,
source cohort and objective before submission. Selection chooses one complete
configuration; tuning evidence never constitutes independent confirmation or
public-default adoption.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, fields
from itertools import product
import math
from pathlib import Path

from particlegan import Recipe

from .api import FormulationContext, task_formulation_context
from .boundaries import TUNABLE_FIELDS, task_owned_recipe_fields
from .contracts import atomic_json, identifier, positive_number, read_json, stable_hash, validate_idea
from .planning import FORMULATION_FIELDS, load_idea, plan_summary, resolve_idea
from .queue import Queue, drain
from .technique_inventory import _first_task_blockers, _signature
from .techniques import recipe_field_active, technique_signature, validate_same_technique


SPEC_FIELDS = frozenset({"schema_version", "id", "trainer_family", "base_candidate", "grid",
                         "tuning_through_tier", "view", "execution_backend", "cuda_model", "protocol",
                         "protocol_hash", "campaign", "hypothesis", "rationale", "guide"})


def _report_path(root, study_id):
    return Path(root) / "reports/forge/configuration-search" / f"{study_id}.json"


def _comparison_binding(task):
    """Freeze the task law and allowance, excluding candidate field ownership."""
    from .views import task_fingerprint
    return stable_hash({"task": task_fingerprint(task), "resources": task.get("resources")})


def _load_spec(root, spec, *, validate_current_protocol=True):
    if isinstance(spec, dict):
        value = deepcopy(spec)
    else:
        path = Path(spec)
        if path.suffix != ".json" and len(path.parts) == 1:
            path = Path("configs/forge/searches") / f"{path}.json"
        value = read_json(Path(root) / path)
    if not isinstance(value, dict) or set(value) - SPEC_FIELDS:
        raise ValueError("search spec contains unsupported fields")
    modern = value.get("schema_version") == 2
    required = SPEC_FIELDS - {"cuda_model", "hypothesis", "rationale", "guide"}
    if modern:
        required = (required - {"protocol_hash"}) | {"hypothesis"}
        if "protocol_hash" in value:
            raise ValueError("v2 search protocol_hash is generated, not authored in a study")
        if not isinstance(value.get("hypothesis"), str) or not value["hypothesis"].strip() or "TODO" in value["hypothesis"]:
            raise ValueError("v2 search requires a finished study hypothesis")
    if required - value.keys() or type(value.get("schema_version")) is not int or value["schema_version"] not in (1, 2):
        raise ValueError("search spec missing required fields or unsupported schema")
    for key in ("id", "trainer_family", "base_candidate", "view", "protocol"):
        identifier(value[key], f"search {key}")
    if not isinstance(value["tuning_through_tier"], int) or isinstance(value["tuning_through_tier"], bool) or value["tuning_through_tier"] < 1:
        raise ValueError("tuning_through_tier must be an integer >= 1")
    if value["execution_backend"] not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be cpu or cuda")
    if validate_current_protocol:
        defaults = read_json(Path(root) / "configs/forge/defaults.json")
        protocol = read_json(Path(root) / "configs/forge/protocols" / f"{value['protocol']}.json")
        if defaults["protocol"] != value["protocol"] or (not modern and stable_hash(protocol) != value["protocol_hash"]):
            raise ValueError("search fixed protocol differs from the declared Forge protocol")
    campaign = value["campaign"]
    identifier(campaign["id"], "search campaign")
    positive_number(campaign["budget_seconds"], "search campaign budget_seconds")
    positive_number(campaign["candidate_budget_seconds"], "search candidate_budget_seconds")
    if campaign["candidate_budget_seconds"] > campaign["budget_seconds"]:
        raise ValueError("candidate budget exceeds campaign budget")
    # Validate that every grid key ultimately binds a public Recipe field.
    _grid(value["grid"])
    return value


def _grid(grid):
    # Re-evaluate an exact union of previously declared grids without adding
    # redundant overrides or taking their unwanted Cartesian product. Explicit
    # overrides participate in configuration identity, including default values.
    if isinstance(grid, list):
        if not grid or any(not isinstance(part, dict) for part in grid):
            raise ValueError("search grid union requires nonempty ordinary grid objects")
        choices = [choice for part in grid for choice in _grid(part)]
        if len(choices) > 256:
            raise ValueError("search grid exceeds the 256-configuration declaration limit")
        if len({stable_hash(choice) for choice in choices}) != len(choices):
            raise ValueError("search grid union contains duplicate choices")
        return choices
    if not isinstance(grid, dict) or not grid:
        raise ValueError("search grid must declare at least one Recipe axis")
    recipe_fields = {field.name for field in fields(Recipe)}
    axes, assigned = [], set()
    for axis, values in sorted(grid.items()):
        identifier(axis, "grid axis")
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid axis {axis} must have nonempty choices")
        # A named group allows correlated endpoints without an unwanted product.
        choices = ([{axis: value} for value in values] if axis in recipe_fields else values)
        if not all(isinstance(choice, dict) and choice for choice in choices):
            raise ValueError(f"unknown or forbidden Recipe grid field: {axis}")
        names = set(choices[0])
        if any(set(choice) != names for choice in choices):
            raise ValueError(f"grouped grid axis {axis} must bind the same Recipe fields in every choice")
        for name in names:
            if name not in recipe_fields:
                raise ValueError(f"unknown Recipe grid field: {name}")
            if name not in TUNABLE_FIELDS:
                raise ValueError(f"forbidden Recipe grid field (fixed host/protocol/sampling law): {name}")
            for choice in choices:
                _validate_grid_value(name, choice[name])
        if assigned & names:
            raise ValueError("grid axes overlap Recipe fields")
        if len({stable_hash(choice) for choice in choices}) != len(choices):
            raise ValueError("grid axis contains duplicate choices")
        assigned.update(names)
        axes.append(choices)
    # Bound declaration growth as well as paid work; this is an explicit grid.
    count = 1
    for axis in axes:
        count *= len(axis)
    if count > 256:
        raise ValueError("search grid exceeds the 256-configuration declaration limit")
    return [dict(item for choice in combination for item in choice.items())
            for combination in product(*axes)]


def _validate_grid_value(name, value):
    """JSON booleans are not numeric hyperparameters; optional values are typed."""
    if name == "amsgrad":
        valid = type(value) is bool
    elif name in {"reg_every", "lr_decay_steps"}:
        valid = type(value) is int and value > 0
    elif name in {"betas", "prior_betas", "d_betas", "direct_particle_betas"}:
        valid = (name in {"prior_betas", "d_betas"} and value is None) or (
            isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(item) in (int, float) and math.isfinite(item) and 0 <= item < 1
                    for item in value))
    elif value is None:
        valid = name in {"network_lr_floor", "beta2_end", "reg_coeff_end", "optimizer_adam_lr", "d_eps", "prior_eps"}
    else:
        valid = type(value) in (int, float) and math.isfinite(value)
    if not valid:
        raise ValueError(f"invalid hyperparameter value for Recipe.{name}: {value!r}")


def recipe_identity_fields(recipe):
    """Preserve the formerly implicit objective without rewriting saved cards."""
    recipe = deepcopy(recipe)
    from particlegan.recipe_compat import without_default_additions
    recipe = without_default_additions(recipe)
    if recipe.get("loss") == "relativistic":
        recipe.pop("loss")
    # Default selectors added for optimizer experiments are absent from all
    # earlier resolved recipes. Their implicit values must not rename saved
    # configuration cards; authored overrides remain in formulation identity.
    if recipe.get("optimizer_momentum") == 0:
        recipe.pop("optimizer_momentum")
    if recipe.get("optimizer_adam_lr") is None:
        recipe.pop("optimizer_adam_lr", None)
    return recipe


def configuration_id(candidate, *, resolved_recipe=None):
    """Hash actual configuration and fixed laws; exclude labels and source code."""
    formulation = {key: candidate.get(key) for key in FORMULATION_FIELDS}
    if candidate.get("schema_version") == 3:
        formulation["prior"] = None  # Runtime task binding never enters an authored recipe identity.
    formulation["recipe_overrides"] = deepcopy(formulation.get("recipe_overrides") or {})
    formulation["recipe_overrides"].pop("name", None)
    if resolved_recipe is None:
        resolved_recipe = candidate.get("resolved_configuration_recipe") or _resolved_recipe(candidate)
    recipe = recipe_identity_fields(resolved_recipe)
    recipe.pop("name", None)
    # Historical recipes had one implicit paired-logistic objective. Adding
    # its public selector must not rename those immutable configuration cards.
    # Explicit loss overrides remain in formulation; alternatives stay bound.
    formulation.update(resolved_recipe=recipe, host_adaptation=candidate.get("host_adaptation"),
                       initializer=candidate.get("initializer", "deterministic_orthogonal"),
                       execution_path=candidate.get("execution_path", "public_trainer"))
    return stable_hash(formulation)


def _resolved_recipe(candidate):
    """Use the same fixed prior and API bindings as an actual Forge request."""
    context = FormulationContext(recipe_preset=candidate.get("recipe_preset"),
        recipe_overrides=candidate.get("recipe_overrides", {}), prior=candidate.get("prior"),
        prior_mechanism=candidate.get("prior_mechanism"),
        requires_capabilities=candidate.get("requires_capabilities", ()),
        extensions=candidate.get("extensions", {}),
        initializer=candidate.get("initializer", "deterministic_orthogonal"),
        execution_path=candidate.get("execution_path", "public_trainer"))
    return asdict(context.recipe)


def _base_declaration(root, spec):
    base = load_idea(root, spec["base_candidate"])
    defaults = read_json(Path(root) / "configs/forge/defaults.json")
    if base["schema_version"] != 3:
        base["prior"] = {**defaults["prior"], **base.get("prior", {})}
    return base


def validate_configuration_declaration(candidate, *, root=None, _lineage=()):
    """Validate recorded identity, and current ordinary-entry technique lineage.

    Evidence readers pass no checkout and retain pure frozen-hash validation.
    Ordinary declaration loaders additionally bind the named parent technique.
    """
    family = identifier(candidate.get("trainer_family"), "configuration trainer_family")
    frozen = candidate.get("resolved_configuration_recipe")
    if not isinstance(frozen, dict) or not frozen:
        raise ValueError("configuration declaration requires its complete frozen resolved Recipe")
    for key, value in candidate.get("recipe_overrides", {}).items():
        # Public factories normalize JSON integer moments/labels to floats.
        # Compare that binding while retaining the authored override in its
        # original content identity (including archived integer choices).
        bound = (tuple(float(v) for v in value) if key in {
            "betas", "prior_betas", "direct_particle_betas", "d_betas", "loss_labels"}
            and isinstance(value, (list, tuple)) else value)
        if key != "name" and (key not in frozen or stable_hash(bound) != stable_hash(frozen[key])):
            raise ValueError("configuration declaration overrides contradict its frozen Recipe")
    digest = configuration_id(candidate, resolved_recipe=frozen)
    if (candidate.get("configuration_id") != digest or candidate.get("id") != f"{family}--{digest}"):
        raise ValueError("configuration declaration hash does not match its actual Recipe and fixed laws")
    if root is None:
        return
    if stable_hash(recipe_identity_fields(_resolved_recipe(candidate))) != stable_hash(recipe_identity_fields(frozen)):
        raise ValueError("configuration frozen Recipe differs from current public defaults or API bindings; "
                         "declare a new configuration")
    parent_id = identifier(candidate.get("parent"), "configuration parent")
    lineage = (*_lineage, candidate["id"])
    if parent_id in lineage:
        raise ValueError("configuration technique lineage contains a parent cycle")
    # Read declarations without re-entering the ordinary loader; validate each
    # configuration ancestor with an explicit cycle guard instead.
    from .planning import declaration_paths
    paths = [path for path in declaration_paths(root) if path.stem == parent_id]
    if len(paths) != 1:
        raise ValueError("configuration parent must name one declared technique/configuration")
    parent = read_json(paths[0])
    validate_idea(parent)
    if parent.get("id") != parent_id:
        raise ValueError("configuration parent filename differs from its declaration")
    if paths[0].parent.name == "configurations":
        validate_configuration_declaration(parent, root=root, _lineage=lineage)
        parent_recipe = parent["resolved_configuration_recipe"]
        reference = parent
    else:
        defaults = read_json(Path(root) / "configs/forge/defaults.json")
        reference = {**parent, "prior": {**defaults["prior"], **parent.get("prior", {})}}
        parent_recipe = _resolved_recipe(reference)
    from .trainer_families import family_for_candidate, load_families
    if (parent.get("trainer_family") not in (None, family)
            or (load_families(root) and family not in {
                family_for_candidate(root, parent_id, parent)["id"],
                family_for_candidate(root, parent_id, parent, current_presentation=True)["id"]})):
        raise ValueError("configuration parent belongs to a different trainer_family")
    validate_same_technique(parent_recipe, frozen)
    laws = (set(FORMULATION_FIELDS) - {"recipe_overrides"}) | {"host_adaptation", "execution_path"}
    if candidate.get("schema_version") == 3:
        laws -= {"prior"}  # Modern priors come only from the task, including search trials.
    if any(stable_hash(candidate.get(name)) != stable_hash(reference.get(name)) for name in laws):
        raise ValueError("configuration changes fixed parent technique/protocol declarations")
    changed = {name for name in frozen if name != "name"
               and stable_hash(frozen[name]) != stable_hash(parent_recipe.get(name))}
    if changed - TUNABLE_FIELDS:
        raise ValueError("configuration changes fields outside the hyperparameter search whitelist: "
                         + ", ".join(sorted(changed - TUNABLE_FIELDS)))
    for name in changed:
        _validate_grid_value(name, frozen[name])


def _declarations(root, spec):
    base = _base_declaration(root, spec)
    base_recipe = _resolved_recipe(base)
    from .trainer_families import family_for_candidate, load_families
    if load_families(root) and spec["trainer_family"] not in {
            family_for_candidate(root, base["id"], base)["id"],
            family_for_candidate(root, base["id"], base, current_presentation=True)["id"]}:
        raise ValueError("base candidate belongs to a different registered trainer_family")
    if base.get("trainer_family") not in (None, spec["trainer_family"]):
        raise ValueError("base candidate belongs to a different trainer_family")
    declarations = []
    for settings in _grid(spec["grid"]):
        idea = deepcopy(base)
        modern = spec["schema_version"] == 2
        if modern:
            for key in ("decision_contract", "prior", "goal", "hypothesis", "lifecycle", "search_study_id", "search_report"):
                idea.pop(key, None)
            idea["schema_version"] = 3
        idea["recipe_overrides"] = {**base.get("recipe_overrides", {}), **settings}
        frozen_recipe = _resolved_recipe(idea)  # Actual public context, never ignored kwargs.
        validate_same_technique(base_recipe, frozen_recipe)
        digest = configuration_id(idea, resolved_recipe=frozen_recipe)
        name = f"{spec['trainer_family']}--{digest}"
        identifier(name, "configuration candidate")
        idea.update(id=name, parent=spec["base_candidate"], trainer_family=spec["trainer_family"],
                    configuration_id=digest,
                    resolved_configuration_recipe=frozen_recipe,
                    changed_factors=[f"Recipe.{key}={settings[key]!r}" for key in sorted(settings)],
                    mechanism_class="floor_constant")
        if not modern:
            idea.update(search_study_id=spec["id"], search_report=f"reports/forge/configuration-search/{spec['id']}.json",
                        hypothesis=spec.get("hypothesis") or base.get("hypothesis", "See the search study"), lifecycle="proposed")
        validate_idea(idea)
        path = Path(root) / "configs/forge/configurations" / f"{name}.json"
        if path.exists():
            existing = read_json(path)
            validate_configuration_declaration(existing)
            # Cards are globally reusable across studies. Their first study's
            # metadata remains immutable; the new study binds its own selection.
            if (existing.get("id") != name or existing.get("configuration_id") != digest
                    or existing.get("trainer_family") != spec["trainer_family"]
                    or configuration_id(existing, resolved_recipe=existing["resolved_configuration_recipe"]) != digest):
                raise ValueError(f"configuration declaration is immutable: {name}")
            idea = existing
        declarations.append((idea, settings))
    _validate_tuning_axes(root, spec, base_recipe, declarations)
    return sorted(declarations, key=lambda item: item[0]["configuration_id"])


def _validate_tuning_axes(root, spec, base_recipe, declarations):
    """Do not spend a search axis on values every tuning host ignores."""
    from .views import load_tasks, load_view

    tasks = load_tasks(root)
    view = load_view(root, spec["view"])
    tuning = [tasks[row["task"]] for row in view["assignments"]
              if row["qualification_tier"] <= spec["tuning_through_tier"]]
    requested = {name for _, settings in declarations for name in settings}
    names = {name for name in requested
             if any(stable_hash(settings.get(name, base_recipe[name])) != stable_hash(base_recipe[name])
                    for _, settings in declarations)}
    for name in sorted(names):
        active = False
        uncertain = False
        for task in tuning:
            if name in task_owned_recipe_fields(task):
                continue
            for idea, _ in declarations:
                if not recipe_field_active(name, idea["resolved_configuration_recipe"], task=task):
                    continue
                try:
                    context = task_formulation_context(idea, task, root=root)
                except ValueError:
                    # An incompatible formulation remains a visible preflight
                    # blocker; an unresolved binding is not proof of inactivity.
                    uncertain = True
                    continue
                if recipe_field_active(name, context.recipe, task=task):
                    active = True
                    break
            if active:
                break
        if not active and not uncertain:
            raise ValueError(f"search Recipe.{name} is inactive or task-owned on every tuning task; "
                             "remove this axis or declare a task where it is effective")


def materialize_search(root: Path, spec) -> list[Path]:
    """Write immutable derived cards, with no queue submission or training."""
    root = Path(root).resolve()
    spec = _load_spec(root, spec)
    _check_spec_registration(root, spec)
    declarations = _declarations(root, spec)  # Validate all before writing any.
    paths = []
    for idea, _ in declarations:
        path = root / "configs/forge/configurations" / f"{idea['id']}.json"
        if not path.exists():
            atomic_json(path, idea)
        paths.append(path)
    return paths


def _check_spec_registration(root, spec):
    path = _report_path(root, spec["id"])
    if path.exists():
        saved = read_json(path)
        if saved.get("input_digest") != stable_hash({key: value for key, value in saved.items() if key != "input_digest"}):
            raise ValueError("search report input digest does not match its recorded contents")
        if saved.get("spec_hash") != stable_hash(spec):
            raise ValueError("search spec is immutable after enqueue; use a new study id")
        return saved
    return None


def _persist(root, summary):
    summary["input_digest"] = stable_hash({key: value for key, value in summary.items() if key != "input_digest"})
    atomic_json(_report_path(root, summary["study_id"]), summary)


def _qualification(trial, through_tier):
    """Retain the full view denominator independently of the tuning cap.

    Held tasks are never counted. A held-only tier neither passes nor breaks
    progression, and a view whose highest tier is held is never fully qualified.
    """
    required = [task for task in trial["tasks"] if task["importance"] == "required"]
    held = [task for task in trial["tasks"] if task["importance"] == "held"]
    stretch = [task for task in trial["tasks"] if task["importance"] in {"stretch", "held"}]
    tuning = [task for task in required if task["qualification_tier"] <= through_tier]
    held_only = {task["qualification_tier"] for task in stretch} - {task["qualification_tier"] for task in required}
    tiers, qualified_tier = [], 0
    advancing = not trial.get("submission_blockers")
    for tier in sorted({task["qualification_tier"] for task in required}):
        tasks = [task for task in required if task["qualification_tier"] == tier]
        counts = dict(sorted(Counter(task["gate_status"] for task in tasks).items()))
        passed = bool(tasks) and counts.get("PASS", 0) == len(tasks)
        advancing = advancing and passed and all(t in held_only for t in range(qualified_tier + 1, tier))
        if advancing:
            qualified_tier = tier
        tiers.append({"tier": tier, "required_total": len(tasks),
                      "required_passed": counts.get("PASS", 0), "statuses": counts})
    tuning_passed = bool(tuning) and not trial.get("submission_blockers") and all(
        task["gate_status"] == "PASS" for task in tuning)
    full_passed = bool(required) and not trial.get("submission_blockers") and all(
        task["gate_status"] == "PASS" for task in required)
    return {"qualified_tier": qualified_tier, "tuning_qualified": tuning_passed,
            "full_view_qualified": full_passed and qualified_tier == max(
                (task["qualification_tier"] for task in required + held), default=0) and all(
                task["qualification_tier"] <= through_tier for task in required),
            "required_total": len(required),
            "required_passed": sum(task["gate_status"] == "PASS" for task in required),
            "required_statuses": dict(sorted(Counter(task["gate_status"] for task in required).items())),
            "tuning_required_total": len(tuning),
            "tuning_required_passed": sum(task["gate_status"] == "PASS" for task in tuning),
            **({"held_total": len(held), "held_passed": sum(task["gate_status"] == "PASS" for task in held)}
               if held else {}),
            "tiers": tiers}


def _selected_holdout(selected, through_tier):
    """Whether the selected configuration passed the untuned tiers above the tuning cap.

    Selection uses only tiers <= ``through_tier``; required tasks above it are
    the holdout. The claim never presents a tuning-tier PASS as independent
    confirmation (``independent_confirmation`` stays ``not_performed``).
    """
    from .leaderboard_contract import INDEPENDENT_CONFIRMATION, holdout_claim
    if selected is None:
        return {"status": "pending", "independent_confirmation": INDEPENDENT_CONFIRMATION,
                "claim": "No configuration is selected yet; no holdout claim."}
    by_tier = {}
    for task in selected["tasks"]:
        if task["importance"] == "required":
            by_tier.setdefault(task["qualification_tier"], []).append(task["gate_status"])
    return holdout_claim(by_tier, through_tier)


def _annotate_progression(summary):
    """Add reporting without changing the archived PASS-count/hash selector."""
    trials = summary["trials"]
    scopes = [sorted((task["task"], task["qualification_tier"], task["importance"])
                     for task in trial["tasks"]) for trial in trials]
    if any(scope != scopes[0] for scope in scopes[1:]) or any(
            len({task[0] for task in scope}) != len(scope) for scope in scopes):
        raise ValueError("search progression requires the same complete view task denominator for every configuration")
    for trial in trials:
        trial["qualification"] = _qualification(trial, summary["tuning_through_tier"])
    selected = next((trial for trial in trials if trial["candidate_id"] ==
                     summary["selection"]["selected_candidate_id"]), None)
    outcome = ("pending" if not summary["selection"]["selection_complete"] else
               "full_winner" if selected and selected["qualification"]["full_view_qualified"] else
               "tuning_only_winner" if summary["selection"]["qualified"] else "best_observed")
    summary["progression"] = {
        "policy": "every declared configuration advances independently through ordinary prerequisites to the tier cap",
        "configured_through_tier": summary["tuning_through_tier"],
        "smoke_survivor_candidate_ids": sorted(trial["candidate_id"] for trial in trials
            if trial["qualification"]["qualified_tier"] >= 1),
        "full_view_qualified_candidate_ids": sorted(trial["candidate_id"] for trial in trials
            if trial["qualification"]["full_view_qualified"]),
        "outcome": outcome,
        "full_view_winner_candidate_id": selected["candidate_id"] if outcome == "full_winner" else None,
        "comparison_complete": summary["selection"]["selection_complete"],
        "default_adoption": False,
        "selected_holdout": _selected_holdout(selected, summary["tuning_through_tier"]),
    }
    summary["speed_selection"] = {
        "status": "UNAVAILABLE", "selected_candidate_id": None,
        "reason": "No frozen comparable first-acquisition timing contract is implemented for this search. "
                  "Not all required adapters record per-observation seconds; original terminal-suffix "
                  "or coverage-only times cannot supply full-quality first-acquisition speed.",
        "wall_time_ranking": False,
    }


def _evaluator_timing(task, grade):
    """Preserve recorded evaluator semantics; never infer acquisition seconds."""
    kind = task["evaluation"]["kind"]
    if kind == "transfer_sustained":
        original = grade.get("evaluator_result", {}).get("convergence", {})
        semantics = "complete live curve and terminal passing suffix"
    elif kind in {"ring_hold", "ring_extension"}:
        original = grade.get("metrics", {})
        semantics = "first qualifying dense window followed by uninterrupted hold/extension"
    else:
        original = {}
        semantics = "no first full-quality acquisition timestamp in this evaluator result"
    original = _compact(original)
    invalid = [key for key, value in original.items() if key.endswith("seconds") and value is not None
               and (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0)]
    for key in invalid:
        original[key] = None  # Raw values remain in the certified original receipt.
    seconds = original.get("confirmed_seconds")
    recorded = (not isinstance(seconds, bool) and isinstance(seconds, (int, float))
                and math.isfinite(seconds) and seconds >= 0 and not invalid)
    return {"status": "RECORDED_EVALUATOR_ONLY" if recorded else "UNAVAILABLE",
            "semantics": semantics, "original_evaluator_convergence": original,
            "invalid_timing_fields": invalid,
            "speed_qualified": False}


def _prepare(root, queue_root, spec, queue):
    spec = _load_spec(root, spec)
    protocol_hash = stable_hash(read_json(Path(root) / f"configs/forge/protocols/{spec['protocol']}.json"))
    previous = _check_spec_registration(root, spec)
    state = queue.inspect()
    existing = state.get("campaigns", {}).get(spec["campaign"]["id"])
    if existing and existing["definition"] != spec["campaign"]:
        raise ValueError("search campaign definition is immutable")
    requests, trials = [], []
    for idea, settings in _declarations(root, spec):
        request = resolve_idea(root, idea["id"], declaration=idea, view_id=spec["view"],
                               through_tier=spec["tuning_through_tier"],
                               execution_backend=spec["execution_backend"], cuda_model=spec.get("cuda_model"))
        if spec["schema_version"] == 2:
            request["search_plan"] = {"study_id": spec["id"], "spec_sha256": stable_hash(spec), "declaration": deepcopy(spec)}
        requests.append(request)
        planned = plan_summary(request, state)
        allowed = {row["task"] for row in planned["tasks"] if row["permitted_by_tier_cap"]}
        from .wallclock import reservation_seconds
        ceiling = sum(reservation_seconds(job, request.get("wallclock_policy", "default"))
                      for job in request["jobs"] if set(job["task_ids"]) <= allowed)
        if ceiling > spec["campaign"]["candidate_budget_seconds"]:
            raise ValueError("search candidate budget cannot cover its declared tuning task ceiling")
        blockers = list(request["preflight_blockers"]) + _first_task_blockers(request)
        trial = {"candidate_id": idea["id"], "configuration_id": idea["configuration_id"],
                 "trainer_family": spec["trainer_family"], "settings": settings,
                 "declaration": deepcopy(idea),
                 "recipe_preset": idea.get("recipe_preset"), "recipe_overrides": idea["recipe_overrides"],
                 "resolved_recipe": request["candidate"]["resolved_recipe"],
                 "technique_signature": technique_signature(idea["resolved_configuration_recipe"]),
                 "candidate_revision": request["candidate_revision"], "source_digest": request["source"]["digest"],
                 "scientific_signature": _signature(request), "policy_fingerprint": request["policy_fingerprint"],
                 "runtime_cohort": {"runtime": request["runtime"], "execution_backend": request["execution_backend"],
                                    "compute_profiles": request["compute_profiles"]},
                 "protocol_hash": stable_hash(request["protocol"]),
                 "evidence_scope": request["view"].get("evidence_scope", "ordinary"),
                 "submission_status": "BLOCKED" if blockers else "READY", "submission_blockers": blockers,
                 "declared_worst_case_seconds": ceiling, "unreused_worst_case_seconds": planned["worst_case_seconds"],
                 "tasks": [{**a, "comparison_binding": _comparison_binding(request["tasks"][a["task"]]),
                            "compatibility_key": next(job["compatibility_key"] for job in request["jobs"]
                            if a["task"] in job["task_ids"]), "gate_status": "UNKNOWN"}
                           for a in planned["tasks"]], "request_id": None, "attempt_ids": [],
                 "cost": {"wall_seconds": 0.0}, "status": "UNKNOWN"}
        trials.append(trial)
    ceiling = sum(trial["declared_worst_case_seconds"] for trial in trials)
    if ceiling > spec["campaign"]["budget_seconds"]:
        raise ValueError("search campaign budget cannot cover all declared configuration ceilings")
    if len({trial["configuration_id"] for trial in trials}) != len(trials):
        raise ValueError("search grid resolves duplicate configurations")
    sources = sorted({trial["source_digest"] for trial in trials})
    if (len(sources) != 1 or len({stable_hash(t["runtime_cohort"]) for t in trials}) != 1
            or {t["protocol_hash"] for t in trials} != {protocol_hash}):
        raise ValueError("search requires one frozen source/runtime/protocol cohort; plan again after inputs stabilize")
    if previous:
        old = {trial["configuration_id"]: trial for trial in previous["trials"]}
        if (set(old) != {trial["configuration_id"] for trial in trials}
                or any(old[t["configuration_id"]]["scientific_signature"] != t["scientific_signature"] for t in trials)):
            raise ValueError("search source or scientific declarations changed; use a new study id")
    summary = {"schema_version": 1, "stage": "planned", "study_id": spec["id"],
               "trainer_family": spec["trainer_family"], "base_candidate": spec["base_candidate"],
               "base_declaration": _base_declaration(root, spec),
               "technique_signature": technique_signature(_resolved_recipe(_base_declaration(root, spec))),
               "spec": spec, "spec_hash": stable_hash(spec), "view": spec["view"],
               "evidence_scope": requests[0]["view"].get("evidence_scope", "ordinary"),
               "tuning_through_tier": spec["tuning_through_tier"],
               "execution_backend": spec["execution_backend"], "source_digests": sources,
               "source_digest": sources[0] if len(sources) == 1 else None,
               "policy_fingerprint": requests[0]["policy_fingerprint"], "protocol_hash": protocol_hash,
               "runtime_cohort": trials[0]["runtime_cohort"], "campaign": spec["campaign"],
               "declared_worst_case_seconds": ceiling, "trials": trials,
               "queue_root": str(queue_root), "logs": str(Path(queue_root) / "events.jsonl"),
               "independent_confirmation": "not_performed", "default_adoption": False,
               "qualification_scope": "declared tuning tasks only; provisional Forge gates",
               "holdout_tasks": [a["task"] for a in requests[0]["view"]["assignments"]
                                 if a["qualification_tier"] > spec["tuning_through_tier"]]}
    summary["selection"] = select_configuration(trials, spec["tuning_through_tier"])
    _annotate_progression(summary)
    return requests, summary, previous


def select_configuration(trials: list[dict], through_tier: int) -> dict:
    """Frozen lexicographic objective, with a content-hash-only tie break."""
    if not trials or not isinstance(through_tier, int) or isinstance(through_tier, bool) or through_tier < 1:
        raise ValueError("selection requires trials and a declared tuning tier >= 1")
    def counts(trial):
        return tuple(sum(t["importance"] == "required" and t["qualification_tier"] == tier
                         and t["gate_status"] == "PASS" for t in trial["tasks"])
                     for tier in range(1, through_tier + 1))
    def terminal(trial):
        if trial.get("submission_status", "").lower() not in {"completed", "blocked", "terminal", "concluded"}:
            return False
        required = [t for t in trial["tasks"] if t["importance"] == "required" and t["qualification_tier"] <= through_tier]
        return bool(trial.get("submission_blockers") or (required and (
            all(t["gate_status"] == "PASS" for t in required)
            or any(t["gate_status"] in {"FAIL", "INVALID", "INCOMPLETE", "BLOCKED"} for t in required))))
    complete = all(terminal(trial) for trial in trials)
    objective = "required PASS count by ascending tuning tier (descending); configuration hash (ascending)"
    common = {"selection_complete": complete, "all_trials_terminal": complete,
              "calibration_status": "provisional", "independent_confirmation": "not_performed", "default_adoption": False,
              "tuning_through_tier": through_tier, "objective": objective}
    if not complete:
        return {**common, "selected_candidate_id": None, "selected_configuration_id": None, "qualified": False,
                "selection_kind": "pending", "required_pass_count": None, "required_total": None,
                "required_passes_by_tier": None, "source_digest": None, "required_task_bindings": []}
    selected = min(trials, key=lambda t: (*(-n for n in counts(t)), t["configuration_id"]))
    required = [t for t in selected["tasks"] if t["importance"] == "required"
                and t["qualification_tier"] <= through_tier]
    passed = sum(t["gate_status"] == "PASS" for t in required)
    qualified = bool(required) and passed == len(required)
    return {**common, "selected_candidate_id": selected["candidate_id"],
            "selected_configuration_id": selected["configuration_id"], "qualified": qualified,
            "selection_kind": "qualified_winner" if qualified else "best_observed",
            "required_pass_count": passed, "required_total": len(required),
            "required_passes_by_tier": list(counts(selected)), "tuning_through_tier": through_tier,
            "source_digest": selected.get("source_digest"),
            "required_task_bindings": [{key: t[key] for key in ("task", "qualification_tier", "compatibility_key")}
                                       for t in required]}


def plan_search(root: Path, queue_root: Path, spec, *, queue=None) -> dict:
    """Read-only plan; undeclared configurations are resolved in memory."""
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    _, summary, _ = _prepare(root, queue_root, spec, queue or Queue(queue_root))
    return summary


def _freeze_search(root: Path, queue_root: Path, spec, *, queue) -> tuple:
    """Prepare immutable requests without admission, including crash recovery."""
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    if queue.root != queue_root:
        raise ValueError("search queue_root differs from supplied Queue")
    materialize_search(root, spec)
    requests, summary, previous = _prepare(root, queue_root, spec, queue)
    frozen = []
    for request, trial in zip(requests, summary["trials"]):
        if trial["submission_blockers"]:
            continue
        resolved = resolve_idea(root, trial["candidate_id"], view_id=summary["view"],
                                through_tier=summary["tuning_through_tier"], execution_backend=summary["execution_backend"],
                                cuda_model=summary["spec"].get("cuda_model"), queue_root=queue_root, freeze_source=True)
        if summary["spec"]["schema_version"] == 2:
            resolved["search_plan"] = {"study_id": summary["study_id"], "spec_sha256": summary["spec_hash"],
                                       "declaration": deepcopy(summary["spec"])}
        if _signature(resolved) != trial["scientific_signature"]:
            raise ValueError("search source or declarations changed before submission; plan and enqueue again")
        # A Git commit with identical scientific bytes cannot create another
        # study request solely by changing origin_commit metadata.
        if previous:
            saved = next(t for t in previous["trials"] if t["configuration_id"] == trial["configuration_id"])
            state = queue.inspect()
            request_id = saved.get("request_id")
            if not request_id:
                # Crash between queue admission and the compact report update:
                # recover its exact immutable request rather than create one.
                request_id = next((key for key, entry in state["submissions"].items()
                                   if entry["request"].get("campaign_id") == summary["campaign"]["id"]
                                   and entry["request"]["candidate"]["id"] == trial["candidate_id"]
                                   and _signature(entry["request"]) == trial["scientific_signature"]), None)
            if request_id:
                if request_id not in state["submissions"]:
                    raise ValueError("registered search request is missing from its queue; restore the queue")
                resolved = state["submissions"][request_id]["request"]
                resolved = {key: value for key, value in resolved.items()
                            if key not in {"request_id", "campaign_id", "queue_root"}}
        frozen.append((resolved, trial))
    summary.update(stage="registered", submitted_count=0, blocked_count=len(requests) - len(frozen))
    return frozen, summary


def _admit_search(root, queue, frozen, summary):
    queue_root = queue.root
    _persist(root, summary)  # Freeze study identity before the first admission.
    for request, trial in frozen:
        entry = queue.submit(request, summary["campaign"])
        trial.update(request_id=entry["request"]["request_id"], submission_status=entry["status"],
                     request_path=str(queue_root / "queue/requests" / f"{entry['request']['request_id']}.json"))
        summary["submitted_count"] += 1
        _persist(root, summary)
    summary.update(stage="enqueued", submitted_count=len(frozen))
    _persist(root, summary)
    return report_search(root, queue_root, summary["spec"], queue=queue)


def enqueue_search(root: Path, queue_root: Path, spec, *, queue=None) -> dict:
    """Materialize all cards, freeze all source, then admit ordinary requests."""
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    frozen, summary = _freeze_search(root, queue_root, spec, queue=queue)
    return _admit_search(root, queue, frozen, summary)


def report_search(root: Path, queue_root: Path, spec, *, queue=None) -> dict:
    """Refresh compact trial evidence using the study's frozen request keys."""
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    # Reporting a recorded study does not resolve the latest source or regrade it
    # as current. It preserves the exact frozen request/source/runtime cohort.
    if isinstance(spec, dict):
        value = deepcopy(spec)
    else:
        path = Path(spec)
        if path.suffix != ".json" and len(path.parts) == 1:
            path = Path("configs/forge/searches") / f"{path}.json"
        value = read_json(root / path)
    identifier(value["id"], "search study")
    summary = _check_spec_registration(root, value)
    if summary is None:
        raise ValueError("search has not been enqueued; use search plan for a read-only preview")
    if Path(summary["queue_root"]).resolve() != queue_root:
        raise ValueError("report queue differs from the frozen study queue")
    queue = queue or Queue(queue_root)
    state = queue.inspect()
    # The existing certificate validator marks corrupted receipts INVALID and
    # resolves authorized retries. Summary projections never qualify a trial.
    from .knowledge import _attempts
    attempts, _ = _attempts(root)
    by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    from .views import grade_result
    all_observed = {}
    for trial in summary["trials"]:
        trial.pop("selection", None)
        entry = state.get("submissions", {}).get(trial.get("request_id"))
        if trial.get("request_id") and entry is None:
            raise ValueError("registered search request missing; restore the recorded queue")
        if entry:
            request = entry["request"]
            if _signature(request) != trial["scientific_signature"]:
                raise ValueError("search frozen request scientific signature mismatch")
            trial.update(submission_status=entry["status"], submission_reason=entry.get("reason"))
        trial["attempt_ids"] = []
        evidence, observed = [], {}
        for task in trial["tasks"]:
            if entry:
                task.setdefault("comparison_binding", _comparison_binding(entry["request"]["tasks"][task["task"]]))
            job = state.get("jobs", {}).get(task["compatibility_key"], {})
            for item in job.get("attempts", []):
                original = by_id.get(item["attempt_id"])
                if original:
                    observed[original["attempt_id"]] = original
            result = job.get("result")
            task.update(gate_status="UNKNOWN", raw_status=None, metrics={}, cost={}, cadence=None,
                        evaluator_timing={"status": "UNAVAILABLE", "speed_qualified": False,
                                          "reason": "no currently certified evaluator result"})
            if not result:
                if task["blockers"] or trial["submission_blockers"]:
                    task["gate_status"] = "BLOCKED"
                continue
            attempt = by_id.get(result["attempt_id"])
            # Missing originals and mismatched certificates cannot grant PASS.
            valid = bool(attempt and attempt["valid_receipt"] and attempt["result_hash"] == stable_hash(result)
                         and not attempt.get("superseded_by")
                         and attempt["request"]["candidate_revision"] == trial["candidate_revision"]
                         and attempt["request"]["source"]["digest"] == trial["source_digest"]
                         and attempt["request"]["runtime"] == trial["runtime_cohort"]["runtime"]
                         and attempt["request"]["compute_profiles"] == trial["runtime_cohort"]["compute_profiles"])
            row = next((r for r in result["task_results"] if r["task_id"] == task["task"]), None)
            if row:
                grade = grade_result(entry["request"]["tasks"][task["task"]], row) if valid else {}
                task.update(gate_status=grade.get("gate_status", grade.get("status", "INVALID")) if valid else "INVALID",
                            raw_status=row.get("raw_status"), metrics=_compact(grade.get("metrics", row.get("metrics", {}))),
                            cost=deepcopy(row.get("cost", {})),
                            reason=grade.get("reason", row.get("reason")) if valid else "missing or mismatched original receipt certificate",
                            attempt_id=result["attempt_id"])
                if valid:
                    task["evaluator_timing"] = _evaluator_timing(entry["request"]["tasks"][task["task"]], grade)
                    evidence_row = row.get("evidence")
                    evidence_row = evidence_row if isinstance(evidence_row, dict) else {}
                    guards = evidence_row.get("guards")
                    guards = guards if isinstance(guards, dict) else {}
                    audit = guards.get("mechanism_audit", evidence_row.get("mechanism_audit"))
                    task["cadence"] = deepcopy(audit.get("penalty_cadence")) if isinstance(audit, dict) else None
                trial["attempt_ids"].append(result["attempt_id"])
                evidence.append({"attempt_id": result["attempt_id"], "result_hash": stable_hash(result),
                                 "valid_receipt": valid})
        trial["selected_attempt_ids"] = sorted(set(trial["attempt_ids"]))
        trial["attempt_ids"] = sorted(set(trial["attempt_ids"]) | set(observed))
        trial["receipt_bindings"] = sorted({e["attempt_id"]: e for e in evidence}.values(), key=lambda e: e["attempt_id"])
        trial["attempt_history"] = [{"attempt_id": a["attempt_id"], "result_hash": a["result_hash"],
                                     "valid_receipt": a["valid_receipt"], "superseded_by": a.get("superseded_by"),
                                     "gate_statuses": {r["task_id"]: r["gate_status"] for r in a["task_results"]},
                                     "wall_seconds": _attempt_seconds(a)}
                                    for _, a in sorted(observed.items())]
        all_observed.update(observed)
        observation_cost = sum(a["wall_seconds"] for a in trial["attempt_history"])
        paid = sum(charge["seconds"] for charge in state.get("charges", [])
                   if charge["owner"]["campaign"] == summary["campaign"]["id"]
                   and charge["owner"]["revision"] == trial["candidate_revision"])
        trial["cost"] = {"wall_seconds": observation_cost, "evidence_wall_seconds": observation_cost,
                         "new_paid_wall_seconds": paid}
        trial["counts"] = dict(sorted(Counter(t["gate_status"] for t in trial["tasks"]).items()))
        required = [t for t in trial["tasks"] if t["importance"] == "required"
                    and t["qualification_tier"] <= summary["tuning_through_tier"]]
        statuses = {t["gate_status"] for t in required}
        trial["status"] = ("PASS" if required and statuses == {"PASS"} else
                           next((s for s in ("INVALID", "FAIL", "INCOMPLETE", "BLOCKED") if s in statuses), "UNKNOWN"))
    summary["selection"] = select_configuration(summary["trials"], summary["tuning_through_tier"])
    _annotate_progression(summary)
    selected = next((t for t in summary["trials"] if t["candidate_id"] == summary["selection"]["selected_candidate_id"]), None)
    if selected:
        selected["selection"] = deepcopy(summary["selection"])
    summary["campaign_accounting"] = state.get("campaigns", {}).get(summary["campaign"]["id"])
    summary["cost"] = {"wall_seconds": sum(_attempt_seconds(a) for a in all_observed.values()),
                       "new_paid_wall_seconds": (summary["campaign_accounting"] or {}).get("spent_seconds", 0.0),
                       "observed_attempts": len(all_observed)}
    summary["cost"]["evidence_wall_seconds"] = summary["cost"]["wall_seconds"]
    _persist(root, summary)
    return summary


def _compact(value):
    """Final scalar metrics only; event/observation arrays stay in originals."""
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if not isinstance(item, (list, tuple))}
    return value


def _attempt_seconds(attempt):
    # Group members can repeat the physical execution cost or assign it only to
    # the producer. One certified physical attempt receives one charge.
    return max((row.get("cost", {}).get("wall_seconds", 0) for row in attempt["task_results"]), default=0)


def run_search(root: Path, queue_root: Path, spec, *, devices=None, queue=None) -> dict:
    """Explicit execution of only this bounded campaign; ordinary stop rules."""
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    definition = _load_spec(root, spec)
    devices = list(devices or (["cpu"] if definition["execution_backend"] == "cpu" else ["0", "1"]))
    if (definition["execution_backend"] == "cpu") != (devices == ["cpu"]):
        raise ValueError("search devices must agree with the fixed execution_backend")
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    summary = enqueue_search(root, queue_root, definition, queue=queue)
    if summary["submitted_count"]:
        drain(queue, devices, campaign=summary["campaign"]["id"])
    summary = report_search(root, queue_root, definition, queue=queue)
    summary["stage"] = "drained"
    _persist(root, summary)
    return summary
