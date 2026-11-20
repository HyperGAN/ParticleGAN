"""Explicit trainer families and whole-configuration publication selection.

Family names never establish scientific compatibility. Selecting a row copies
its complete evidence; alternatives cannot contribute individual passing cells.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import identifier, read_json, stable_hash

REGISTRY = Path("configs/forge/trainer-families.json")
CURRENT_SELECTION = Path("configs/forge/selections/family-current-v1.json")
DECLARED_REVISIONS = Path("configs/forge/selections/family-revisions-20261016.json")

# Editorial labels and publication receipts do not change a measured row. Bind
# every scientific field, including the complete denominator and UNKNOWN cells.
_DISPLAY_FIELDS = {"technique", "publication_key", "qualification_input", "qualification_reuse",
                   "trainer_family", "configuration_id", "comparison_cohort", "selection",
                   "selected_configuration", "alternative_scope"}


def scientific_row_hash(row: dict) -> str:
    return stable_hash({key: value for key, value in row.items() if key not in _DISPLAY_FIELDS})


def family_row_pin(row: dict, *, selection_kind: str, reason: str, measurement_views=None, measurement_tasks=None) -> dict:
    """Describe one exact ordinary measured row, never an aggregate of cells."""
    bindings = row.get("bindings", {})
    pin = {"trainer_family": row["trainer_family"], "candidate_id": row["candidate_id"],
            "candidate_revision": row.get("candidate_revision"), "cohort": row.get("cohort"),
            "execution_backend": row.get("runtime_cohort", {}).get("execution_backend"),
            "runtime_cohort_sha256": stable_hash(row.get("runtime_cohort")),
            **{key: bindings.get(key) for key in ("source_digest", "recipe_sha256", "task_keys_sha256",
                                                "protocol_sha256", "rng_sha256")},
            "scientific_row_sha256": scientific_row_hash(row),
            "selection_kind": selection_kind, "reason": reason}
    if measurement_views is not None:
        pin["measurement_views"] = list(measurement_views)
    if measurement_tasks is not None:
        pin["measurement_tasks"] = list(measurement_tasks)
    return pin


def load_current_selection(root: Path | str, *, view_id: str, policy_fingerprint: str,
                           selection_card: dict | None = None) -> dict:
    path = Path(root) / CURRENT_SELECTION
    if selection_card is None and not path.is_file():
        return {}
    card = read_json(path) if selection_card is None else selection_card
    if (card.get("schema_version") != 1 or card.get("scope") != "whole_candidate_family_current"
            or card.get("default_adoption") is not False or not isinstance(card.get("selections"), list)):
        raise ValueError("unsupported whole-row family selection")
    if card.get("view") != view_id or card.get("policy_fingerprint") != policy_fingerprint:
        raise ValueError("current family selection differs from the view policy")
    pins = {}
    for pin in card["selections"]:
        family = identifier(pin.get("trainer_family"), "selected trainer family")
        identifier(pin.get("candidate_id"), "selected candidate")
        if family in pins:
            raise ValueError("current family selection must contain one whole row per family")
        if (pin.get("selection_kind") not in {"historical_incumbent", "configured_standard", "current_measurement"}
                or pin.get("execution_backend") not in {"cpu", "cuda"}
                or not isinstance(pin.get("reason"), str) or not pin["reason"].strip()):
            raise ValueError("current family selection needs an explicit cohort choice and reason")
        measurement_views = pin.get("measurement_views")
        if pin["selection_kind"] == "current_measurement":
            if (not isinstance(measurement_views, list) or not measurement_views
                    or len(set(measurement_views)) != len(measurement_views)):
                raise ValueError("current measurement requires explicit distinct measurement views")
            for view in measurement_views:
                identifier(view, "measurement view")
        elif measurement_views is not None:
            raise ValueError("measurement views require a current_measurement selection")
        if "measurement_tasks" in pin:
            tasks = pin["measurement_tasks"]
            if (pin["selection_kind"] != "current_measurement" or not isinstance(tasks, list)
                    or not tasks or len(set(tasks)) != len(tasks)):
                raise ValueError("additional measurement tasks require an explicit current_measurement selection")
            for task in tasks:
                identifier(task, "measurement task")
        pins[family] = pin
    return pins


def load_declared_revisions(root: Path | str, *, card: dict | None = None) -> dict:
    """Declared next selected recipe per family (user decision), awaiting measurement.

    The measured pins in ``family-current-v1.json`` are never rewritten here: a
    declared revision names the recipe the family is measured with next, and must
    supersede exactly that family's currently pinned candidate with a recipe that
    differs only in the declared recipe fields. Returns ``{family: revision}``.
    """
    from .planning import load_idea
    root = Path(root)
    path = root / DECLARED_REVISIONS
    if card is None:
        if not path.is_file():
            return {}
        card = read_json(path)
    if (card.get("schema_version") != 1 or card.get("scope") != "declared_selected_recipe_revisions"
            or card.get("default_adoption") is not False or not isinstance(card.get("revisions"), list)):
        raise ValueError("unsupported declared family revision card")
    current = read_json(root / CURRENT_SELECTION)
    if card.get("view") != current.get("view"):
        raise ValueError("declared family revisions must target the current selection view")
    pins = {pin["trainer_family"]: pin for pin in current["selections"]}
    result = {}
    for revision in card["revisions"]:
        family = identifier(revision.get("trainer_family"), "declared revision family")
        if family in result:
            raise ValueError("one declared revision per family")
        if family not in pins or pins[family]["candidate_id"] != revision.get("supersedes"):
            raise ValueError(f"{family}: a declared revision must supersede the family's pinned candidate")
        if revision.get("status") not in {"awaiting_measurement", "measured"}:
            raise ValueError(f"{family}: declared revision status must be awaiting_measurement or measured")
        old, new = load_idea(root, revision["supersedes"]), load_idea(root, revision["candidate_id"])
        if new.get("trainer_family") != family or new.get("parent") != old["id"]:
            raise ValueError(f"{family}: declared revision must be a same-family child of the superseded recipe")
        changes = revision.get("changed_recipe_fields")
        if not isinstance(changes, dict) or not changes:
            raise ValueError(f"{family}: declared revision must name its changed recipe fields")
        before, after = old["recipe_overrides"], new["recipe_overrides"]
        if set(before) != set(after):
            raise ValueError(f"{family}: declared revision changed the recipe field set")
        differing = {key for key in before if before[key] != after[key]}
        if differing != set(changes) or any([before[key], after[key]] != list(changes[key]) for key in changes):
            raise ValueError(f"{family}: recipe differs from its superseded selection beyond the declared fields")
        scientific = ("claim_contract", "execution_path", "host_adaptation", "recipe_preset", "requires_capabilities",
                      "schema_version", "api_version", "api_changes")
        if any(old.get(key) != new.get(key) for key in scientific):
            raise ValueError(f"{family}: declared revision changed a non-recipe scientific field")
        result[family] = deepcopy(revision)
    return result


def _current_pin(root, family_id, rows, pin, *, view_id, catalogs):
    from .views import load_view, load_tasks, task_evaluation_fingerprint, task_execution_fingerprint
    matches = [row for row in rows if family_row_pin(
        row, selection_kind=pin["selection_kind"], reason=pin["reason"],
        measurement_views=pin.get("measurement_views"), measurement_tasks=pin.get("measurement_tasks")) == pin]
    if len(matches) != 1:
        raise ValueError("current family selection must match one exact verified scientific row")
    selected = matches[0]
    if pin["selection_kind"] in {"configured_standard", "current_measurement"} and not selected.get("attempt_ids"):
        raise ValueError("current family selection requires ordinary measured evidence")
    required = {assignment["task"] for assignment in load_view(root, view_id).get("assignments", [])
                if assignment["importance"] == "required" and assignment["qualification_tier"] == 1}
    tasks = selected.get("tasks", [])
    statuses = {task["task_id"]: task["status"] for task in tasks}
    tier = selected.get("tiers", {}).get("1", {})
    qualified = (bool(required) and all(statuses.get(task) == "PASS" for task in required)
                 and tier.get("passed") == tier.get("total") == len(required)
                 and selected.get("qualified_tier", 0) >= 1
                 and len(statuses) == len(tasks))
    if pin["selection_kind"] == "configured_standard" and not qualified:
        raise ValueError("configured family standard requires every required Tier 1 task to PASS")
    measurement = {}
    if pin["selection_kind"] == "current_measurement":
        required_measurements = {assignment["task"] for name in pin["measurement_views"]
                                 for assignment in load_view(root, name)["assignments"]
                                 if assignment["importance"] == "required" and assignment["qualification_tier"] == 1}
        required_measurements.update(pin.get("measurement_tasks", []))
        measured = selected.get("tasks", []) + selected.get("nonrequired_tasks", [])
        observed = {task["task_id"]: task["status"] for task in measured}
        if (not required_measurements or len(observed) != len(measured)
                or any(observed.get(task) not in {"PASS", "FAIL"} for task in required_measurements)):
            raise ValueError("current measurement requires every required Tier 1 task in its measurement views to be PASS or FAIL")
        declarations = load_tasks(root)
        for name in required_measurements:
            task = declarations[name]
            digest = selected.get("bindings", {}).get("task_contracts", {}).get(name)
            contract = catalogs.get("task_contracts", {}).get(digest, {})
            if (contract.get("execution_sha256") != task_execution_fingerprint(task)
                    or contract.get("evaluation_sha256") != task_evaluation_fingerprint(task)
                    or contract.get("timeout_seconds") != task["resources"]["timeout_seconds"]):
                raise ValueError("current measurement requires current execution, evaluation and budget contracts")
        measurement = {"measurement_views": pin["measurement_views"], "measurement_complete": True,
                       "measured_required_tasks": sorted(required_measurements)}
        if pin.get("measurement_tasks"):
            measurement["measurement_tasks"] = pin["measurement_tasks"]
    return selected, {"selection_kind": pin["selection_kind"], "qualified": qualified, **measurement,
                      "default_adoption": False, "reason": pin["reason"],
                      "selection_card": CURRENT_SELECTION.as_posix(),
                      "comparison_claim": "No ranking across incompatible source or runtime cohorts.",
                      "calibration_status": "provisional", "independent_confirmation": "not_performed"}


def load_families(root: Path | str) -> dict:
    path = Path(root) / REGISTRY
    if not path.is_file():
        return {}
    registry = read_json(path)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("families"), list):
        raise ValueError("unsupported trainer family registry")
    result, candidates = {}, set()
    for family in registry["families"]:
        if not isinstance(family, dict):
            raise ValueError("trainer family must be an object")
        name = identifier(family.get("id"), "trainer family")
        if name in result or not isinstance(family.get("label"), str) or not family["label"].strip():
            raise ValueError("trainer families need unique ids and nonempty labels")
        members = family.get("candidates")
        if not isinstance(members, list) or not members or len(set(members)) != len(members):
            raise ValueError("trainer family needs distinct candidate ids")
        current_members = family.get("current_presentation_candidates", [])
        if not isinstance(current_members, list) or len(set(current_members)) != len(current_members):
            raise ValueError("current family presentation needs distinct candidate ids")
        for member in members + current_members:
            identifier(member, "candidate")
            if member in candidates:
                raise ValueError("candidate belongs to more than one trainer family")
            candidates.add(member)
        if family.get("canonical_candidate") not in members:
            raise ValueError("trainer family canonical candidate must be a family member")
        for backend, study in family.get("active_search_by_backend", {}).items():
            if backend not in {"cpu", "cuda"}:
                raise ValueError("family active search backend must be cpu or cuda")
            identifier(study, "family active search study")
        display_fields = {"unmeasured_display_backend", "unmeasured_display_reason"}
        if display_fields & family.keys():
            if (not display_fields <= family.keys()
                    or family["unmeasured_display_backend"] not in {"cpu", "cuda"}
                    or not isinstance(family["unmeasured_display_reason"], str)
                    or not family["unmeasured_display_reason"].strip()):
                raise ValueError(f"family {name}: unmeasured display requires an explicit backend and reason")
            if family.get("active_search_by_backend"):
                raise ValueError(f"family {name}: unmeasured display cannot select an active search")
        result[name] = deepcopy(family)
    historical = registry.get("historical_families", [])
    if not isinstance(historical, list):
        raise ValueError("historical trainer families must be a list")
    historical_ids = [identifier(family.get("id"), "historical trainer family") for family in historical]
    if len(set(historical_ids)) != len(historical_ids) or set(historical_ids) & result.keys():
        raise ValueError("historical trainer family identities must be distinct")
    referenced = []
    for family in result.values():
        reporting = family.get("reporting_family", family["id"])
        identifier(reporting, "reporting family")
        if (reporting not in result or result[reporting].get("reporting_family", reporting) != reporting
                or (reporting != family["id"] and result[reporting].get("inventory_visible") is False)):
            raise ValueError("reporting family must reference a visible registered solution without another alias")
        if not isinstance(family.get("inventory_visible", True), bool):
            raise ValueError("inventory visibility must be boolean")
        preferred = family.get("current_configuration_family")
        if preferred is not None:
            identifier(preferred, "current configuration family")
            if (reporting != family["id"] or preferred not in result
                    or result[preferred].get("reporting_family", preferred) != family["id"]):
                raise ValueError("current configuration must belong to its reporting family")
        aliases = family.get("historical_family_ids", [])
        if not isinstance(aliases, list) or len(set(aliases)) != len(aliases) or set(aliases) - set(historical_ids):
            raise ValueError("trainer family aliases require exact registered historical identities")
        referenced.extend(aliases)
        for previous in historical:
            if previous["id"] in aliases and not set(previous.get("candidates", [])) <= set(family["candidates"]):
                raise ValueError("historical family members must remain in their current solution family")
    if len(set(referenced)) != len(referenced):
        raise ValueError("historical trainer family cannot belong to multiple solution families")
    return result


def current_family_candidates(root: Path | str, *, family_ids=None, selection_card: dict | None = None) -> list[dict]:
    """One explicitly selected whole configuration per visible solution family.

    Reporting aliases leave historical formulation and receipt identities intact.
    A selected optimizer is not replaced by a better-scoring archived alternative.
    """
    root = Path(root)
    families = load_families(root)
    card = read_json(root / CURRENT_SELECTION) if selection_card is None else selection_card
    pins = load_current_selection(root, view_id=card["view"], policy_fingerprint=card["policy_fingerprint"],
                                  selection_card=card)
    choices = []
    for name, family in sorted(families.items()):
        if family.get("reporting_family", name) != name or not family.get("inventory_visible", True):
            continue
        if family_ids is not None and name not in family_ids:
            continue
        configuration_family = family.get("current_configuration_family", name)
        pin = pins.get(configuration_family)
        if pin is None:
            declaration = families[configuration_family]
            if declaration.get("unmeasured_display_backend") is None:
                raise ValueError(f"family {name}: current benchmark requires an explicit whole-row selection")
            pin = {"trainer_family": configuration_family, "candidate_id": declaration["canonical_candidate"],
                   "selection_kind": "unmeasured_declaration", "execution_backend": declaration["unmeasured_display_backend"]}
        choices.append({"family": name, "label": family["label"],
                        "configuration_family": configuration_family, **deepcopy(pin)})
    if len({choice["candidate_id"] for choice in choices}) != len(choices):
        raise ValueError("current family benchmarks cannot repeat a selected candidate")
    return choices


def family_for_candidate(root: Path | str, candidate_id: str, declaration: dict | None = None,
                         *, current_presentation: bool = False) -> dict:
    families = load_families(root)
    historical = (read_json(Path(root) / REGISTRY).get("historical_families", [])
                  if (Path(root) / REGISTRY).is_file() else [])
    historical_by_id = {family["id"]: family for family in historical}
    for family in families.values():
        members = family["candidates"] + (family.get("current_presentation_candidates", [])
                                           if current_presentation else [])
        if candidate_id in members:
            explicit = (declaration or {}).get("trainer_family")
            aliases = family.get("historical_family_ids", [])
            if explicit is not None and explicit not in [family["id"], *aliases]:
                raise ValueError("candidate trainer_family contradicts its explicit registry")
            if not current_presentation and explicit != family["id"]:
                for name in aliases:
                    previous = historical_by_id[name]
                    if candidate_id in previous["candidates"]:
                        return deepcopy(previous)
            return family
    explicit = (declaration or {}).get("trainer_family")
    if explicit is not None:
        if explicit in historical_by_id:
            if not current_presentation:
                return deepcopy(historical_by_id[explicit])
            return next(family for family in families.values() if explicit in family.get("historical_family_ids", []))
        if explicit not in families:
            raise ValueError(f"unregistered trainer family {explicit}")
        return families[explicit]
    # A new independent mechanism remains visible before a family is declared.
    return {"id": candidate_id, "label": candidate_id, "canonical_candidate": candidate_id,
            "candidates": [candidate_id], "registration": "unclassified_independent_technique"}


def comparison_cohort(row: dict, catalogs: dict, *, task_ids=None) -> str:
    """Identity held fixed when comparing configurations of a trainer family."""
    bindings = row.get("bindings", {})
    contracts = {}
    selected_tasks = bindings.get("task_contracts", {})
    names = sorted(task_ids if task_ids is not None else selected_tasks)
    for name in names:
        digest = selected_tasks.get(name)
        if digest not in catalogs.get("task_contracts", {}):
            # Unknown scientific identities are not comparable by coincidence.
            return stable_hash({"unbound_row": row.get("candidate_id"), "cohort": row.get("cohort"), "task": name})
        contract = deepcopy(catalogs["task_contracts"][digest])
        # The adaptation receipt repeats the varying resolved recipe. Host,
        # prior, initialization, steps and sampling remain bound independently.
        contract.pop("host_adaptation", None)
        contracts[name] = contract
    if not bindings.get("source_digest"):
        return stable_hash({"unbound_row": row.get("candidate_id"), "cohort": row.get("cohort")})
    return stable_hash({"source_digest": bindings["source_digest"], "runtime": row.get("runtime_cohort"),
                        "prior": bindings.get("prior"), "initializer": bindings.get("initializer"),
                        "protocol_sha256": bindings.get("protocol_sha256"), "rng_sha256": bindings.get("rng_sha256"),
                        "claim_contract": bindings.get("claim_contract"), "task_contracts": contracts})


def _declarations(root):
    from .planning import declaration_paths
    # Shared discovery includes full derived configuration cards.
    return {path.stem: read_json(path) for path in declaration_paths(root)}


def _search_pin(root, family_id, backend, rows, declarations, catalogs, *, view_id, policy_fingerprint,
                view_policy=None):
    from .configuration_search import _grid, _load_spec, configuration_id, select_configuration
    from .planning import FORMULATION_FIELDS
    from .views import load_view
    paths = set((Path(root) / "reports/forge/configuration-search").glob("*.json"))
    paths.update(Path(root) / card["search_report"] for card in declarations.values()
                 if card.get("trainer_family") == family_id and isinstance(card.get("search_report"), str))
    registry = load_families(root)
    registry_path = Path(root) / REGISTRY
    if registry_path.is_file():
        registry.update({family["id"]: family for family in read_json(registry_path).get("historical_families", [])})
    active = registry.get(family_id, {}).get("active_search_by_backend", {}).get(backend)
    if active is None:
        return None
    pins = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        report = read_json(path)
        if (report.get("study_id") != active or report.get("trainer_family") != family_id or report.get("view") != view_id
                or report.get("execution_backend") != backend):
            continue
        if report.get("runtime_cohort") != rows[0].get("runtime_cohort"):
            # The same family/backend may have recorded different hardware.
            # A study can select only inside its exact runtime cohort.
            continue
        if report.get("input_digest") != stable_hash({key: value for key, value in report.items() if key != "input_digest"}):
            raise ValueError("search report input digest mismatch")
        spec = _load_spec(root, report["spec"], validate_current_protocol=False)
        if report.get("spec_hash") != stable_hash(spec):
            raise ValueError("search frozen specification hash mismatch")
        frozen_path = Path(root) / "configs/forge/searches" / (spec["id"] + ".json")
        if not frozen_path.is_file() or stable_hash(read_json(frozen_path)) != report["spec_hash"]:
            raise ValueError("search report differs from its registered specification")
        modern = spec["schema_version"] == 2
        # V2 studies select a protocol by name; its hash is generated in the
        # frozen report rather than authored in the specification. Verify that
        # generated identity against every independently graded row below.
        protocol_hash = report.get("protocol_hash") if modern else spec["protocol_hash"]
        if modern and (not isinstance(protocol_hash, str) or len(protocol_hash) != 64
                       or any(character not in "0123456789abcdef" for character in protocol_hash)):
            raise ValueError("v2 search report requires its generated protocol hash")
        for key, value in (("study_id", spec["id"]), ("trainer_family", spec["trainer_family"]),
                           ("view", spec["view"]), ("execution_backend", spec["execution_backend"]),
                           ("tuning_through_tier", spec["tuning_through_tier"]), ("protocol_hash", protocol_hash),
                           ("policy_fingerprint", policy_fingerprint)):
            if report.get(key) != value:
                raise ValueError("search report contradicts its frozen study contract")
        selection = report.get("selection", {})
        if not selection.get("selected_candidate_id"):
            continue
        trials = report.get("trials", [])
        base = report.get("base_declaration", {})
        if base.get("id") != spec["base_candidate"]:
            raise ValueError("search frozen base declaration differs from its specification")
        expected_settings = {stable_hash(settings) for settings in _grid(spec["grid"])}
        observed_settings = [stable_hash(trial.get("settings")) for trial in trials]
        if len(set(observed_settings)) != len(observed_settings) or set(observed_settings) != expected_settings:
            raise ValueError("search report omitted or changed a declared configuration trial")
        # Use immutable public Recipes, never today's public defaults. Cards are
        # globally reusable across studies; their editorial study metadata can
        # retain the first study while the scientific configuration stays fixed.
        expected = {}
        fixed_fields = set(FORMULATION_FIELDS) | {"host_adaptation", "initializer", "execution_path"}
        fixed_fields.discard("recipe_overrides")
        for trial in trials:
            frozen = trial.get("declaration", {})
            recipe = trial.get("resolved_recipe")
            if not isinstance(recipe, dict) or not recipe:
                raise ValueError("search trial lacks its frozen resolved Recipe")
            overrides = {**base.get("recipe_overrides", {}), **trial["settings"]}
            if (frozen.get("recipe_overrides") != overrides
                    or any(frozen.get(key) != base.get(key) for key in fixed_fields)
                    or trial.get("recipe_overrides") != overrides):
                raise ValueError("search trial changed its frozen base formulation or grid settings")
            digest = configuration_id(frozen, resolved_recipe=recipe)
            name = f"{family_id}--{digest}"
            if (trial.get("candidate_id") != name or trial.get("configuration_id") != digest
                    or frozen.get("id") != name or frozen.get("configuration_id") != digest
                    or frozen.get("trainer_family") != family_id
                    or any(stable_hash(recipe.get(key)) != stable_hash(value)
                           for key, value in overrides.items() if key != "name")):
                raise ValueError("search frozen configuration identity differs from its Recipe")
            expected[name] = frozen
        ids = [trial.get("candidate_id") for trial in trials]
        if len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("search report omitted or changed a declared configuration trial")
        cohort_rows = [row for row in rows if row["candidate_id"] in expected]
        if len(cohort_rows) != len(expected) or {row["candidate_id"] for row in cohort_rows} != set(expected):
            raise ValueError("search comparison is missing verified configuration evidence")
        assignments = {a["task"]: a for a in
                       (view_policy if view_policy is not None else load_view(root, view_id))["assignments"]}
        verified = []
        for trial in trials:
            name = trial["candidate_id"]
            row = next(item for item in cohort_rows if item["candidate_id"] == name)
            card = declarations.get(name, {})
            if (card.get("configuration_id") != expected[name]["configuration_id"]
                    or configuration_id(card, resolved_recipe=trial["resolved_recipe"]) != card.get("configuration_id")
                    or trial.get("configuration_id") != card.get("configuration_id")
                    or trial.get("trainer_family") != family_id
                    or trial.get("candidate_revision") != row.get("candidate_revision")):
                raise ValueError("search configuration or revision differs from verified evidence")
            bindings = row.get("bindings", {})
            protocol = catalogs.get("protocol_contracts", {}).get(bindings.get("protocol_sha256"))
            if (protocol is None or stable_hash(protocol) != protocol_hash
                    or (modern and protocol.get("id") != spec["protocol"])):
                raise ValueError("search frozen protocol differs from verified protocol contracts")
            if (trial.get("source_digest") != bindings.get("source_digest")
                    or trial.get("runtime_cohort") != row.get("runtime_cohort")
                    or trial.get("protocol_hash") != bindings.get("protocol_sha256")
                    or stable_hash(trial["resolved_recipe"]) != bindings.get("recipe_sha256")):
                raise ValueError("search source/runtime/protocol differs from verified evidence")
            task_rows = trial.get("tasks", [])
            keys = {task["task"]: task.get("compatibility_key") for task in task_rows}
            if len(keys) != len(task_rows) or set(keys) != set(assignments):
                raise ValueError("search trial changed the frozen view task denominator")
            if stable_hash(keys) != bindings.get("task_keys_sha256"):
                raise ValueError("search trial task identities differ from verified evidence")
            grades = {task["task_id"]: task["status"] for task in row.get("tasks", []) + row.get("nonrequired_tasks", [])}
            verified_tasks = []
            for task in task_rows:
                assignment = assignments[task["task"]]
                if any(task.get(key) != assignment[key] for key in ("importance", "qualification_tier")):
                    raise ValueError("search trial changed frozen task importance or tier")
                observed = grades.get(task["task"], "UNKNOWN")
                declared = task.get("gate_status")
                if ("UNKNOWN" if declared == "NOT_RUN" else declared) != observed:
                    raise ValueError("search trial status differs from independently graded publication")
                verified_tasks.append({**assignment, "compatibility_key": task["compatibility_key"], "gate_status": observed})
            required = [task["gate_status"] for task in verified_tasks if task["importance"] == "required"
                        and task["qualification_tier"] <= spec["tuning_through_tier"]]
            terminal = (any(status in {"FAIL", "BLOCKED", "INVALID", "INCOMPLETE"} for status in required)
                        or bool(required) and all(status == "PASS" for status in required))
            verified.append({"candidate_id": name, "configuration_id": card["configuration_id"],
                             "source_digest": bindings["source_digest"], "tasks": verified_tasks,
                             "submission_status": "terminal" if terminal else "pending"})
        cohorts = {comparison_cohort(row, catalogs) for row in cohort_rows}
        if len(cohorts) != 1:
            raise ValueError("search cannot rank different source/runtime/task/prior/protocol cohorts")
        sources = report.get("source_digests", [])
        if set(sources) != {row["bindings"]["source_digest"] for row in cohort_rows}:
            raise ValueError("search report source identity differs from its scientific evidence")
        recomputed = select_configuration(verified, spec["tuning_through_tier"])
        if not recomputed.get("selection_complete"):
            raise ValueError("search configuration selection requires all trials to be terminal")
        if selection != recomputed:
            raise ValueError("search selection differs from verified whole-configuration objective")
        selected = next(row for row in cohort_rows if row["candidate_id"] == recomputed["selected_candidate_id"])
        pins.append((selected, {**recomputed, "study_report": path.relative_to(root).as_posix(),
                                "study_id": report["study_id"], "comparison_cohort": next(iter(cohorts))}))
    if len(pins) > 1:
        raise ValueError("multiple search selections require an explicit family study choice")
    return pins[0] if pins else None


def _declaration_only_row(row):
    """No executed outcome or cost may enter an unmeasured display choice."""
    tasks = row.get("tasks", []) + row.get("nonrequired_tasks", [])
    unmeasured = {"UNKNOWN", "BLOCKED", "NOT_RUN"}
    # qualify() labels a wholly NOT_RUN aggregate INCOMPLETE. The individual
    # task grades below must still prove that no execution outcome exists.
    return (bool(tasks) and not row.get("attempt_ids") and row.get("qualified_tier", 0) == 0
            and row.get("status") in unmeasured | {"INCOMPLETE"}
            and all(task.get("status") in unmeasured
                    and task.get("gate_status", task.get("status")) in unmeasured for task in tasks)
            and not any(tier.get("passed", 0) or any(count and status not in unmeasured
                        for status, count in tier.get("counts", {}).items())
                        for tier in row.get("tiers", {}).values())
            and row.get("cost", {}).get("measured_tasks", 0) == 0
            and row.get("cost", {}).get("wall_seconds") in {None, 0})


def select_family_rows(root: Path | str, rows: list[dict], catalogs: dict, *, view_id: str,
                       policy_fingerprint: str, declarations: dict | None = None,
                       view_policy: dict | None = None, execution_backend: str | None = None,
                       historical_rows: list[dict] | None = None,
                       selection_card: dict | None = None) -> dict:
    """Choose one current whole row per family; archived policies retain cohorts."""
    if view_policy is not None and (view_policy.get("id") != view_id
                                    or stable_hash(view_policy) != policy_fingerprint):
        raise ValueError("explicit family-selection view differs from its recorded policy")
    declarations = _declarations(root) if declarations is None else declarations
    current_pins = (load_current_selection(root, view_id=view_id, policy_fingerprint=policy_fingerprint,
                                         selection_card=selection_card)
                    if view_policy is None else {})
    grouped, families, variants = {}, {}, []
    for original in rows:
        if (execution_backend is not None
                and original.get("runtime_cohort", {}).get("execution_backend") != execution_backend):
            continue
        row = deepcopy(original)
        family = family_for_candidate(root, row["candidate_id"], declarations.get(row["candidate_id"]),
                                      current_presentation=view_policy is None)
        family_id = family["id"]
        families[family_id] = family
        row.update(trainer_family=family_id,
                   configuration_id=declarations.get(row["candidate_id"], {}).get("configuration_id", row["candidate_id"]),
                   comparison_cohort=comparison_cohort(row, catalogs))
        backend = row.get("runtime_cohort", {}).get("execution_backend", "unrecorded")
        runtime_key = stable_hash(row.get("runtime_cohort"))
        grouped.setdefault((family_id, backend, runtime_key), []).append(row)
        variants.append(row)
    expected_pins = {family for family, pin in current_pins.items()
                     if execution_backend is None or pin["execution_backend"] == execution_backend}
    if expected_pins - set(families):
        raise ValueError("current family selection is missing its verified family evidence")
    selected_rows, selected_families = [], set()
    for (family_id, backend, _), alternatives in sorted(grouped.items()):
        family = families[family_id]
        explicit = current_pins.get(family_id)
        all_alternatives = [row for row in variants if row["trainer_family"] == family_id]
        use_explicit = explicit and (execution_backend is None or explicit["execution_backend"] == execution_backend)
        display_backend = family.get("unmeasured_display_backend") if view_policy is None else None
        use_unmeasured = display_backend is not None and execution_backend in {None, display_backend}
        if use_explicit:
            if family_id in selected_families:
                continue
            selected, metadata = _current_pin(root, family_id, all_alternatives, explicit, view_id=view_id, catalogs=catalogs)
            alternatives = all_alternatives
        elif use_unmeasured:
            if family.get("active_search_by_backend") or not all(_declaration_only_row(row) for row in all_alternatives):
                raise ValueError(f"family {family_id}: unmeasured display requires exclusively declaration-only rows; "
                                 "measured evidence needs an explicit whole-row family selection")
            if family_id in selected_families:
                continue
            canonical = [row for row in all_alternatives if row["candidate_id"] == family["canonical_candidate"]
                         and row.get("runtime_cohort", {}).get("execution_backend") == display_backend]
            if len(canonical) != 1:
                raise ValueError(f"family {family_id}: unmeasured display requires exactly one canonical row "
                                 f"on declared backend {display_backend}")
            selected = canonical[0]
            alternatives = all_alternatives
            metadata = {"selection_kind": "unmeasured_declaration", "qualified": False, "default_adoption": False,
                        "reason": family["unmeasured_display_reason"], "execution_backend": display_backend,
                        "comparison_claim": "Explicit declaration-only display; no source or outcome ranking."}
        else:
            if view_policy is None and family_id in selected_families:
                raise ValueError(f"family {family_id}: multiple current runtime cohorts require "
                                 "an explicit whole-row family selection")
            pin = _search_pin(root, family_id, backend, alternatives, declarations, catalogs,
                              view_id=view_id, policy_fingerprint=policy_fingerprint, view_policy=view_policy)
            if pin:
                selected, metadata = pin
            else:
                canonical = [row for row in alternatives if row["candidate_id"] == family["canonical_candidate"]]
                if len(canonical) > 1:
                    raise ValueError(f"family {family_id}: canonical configuration has multiple scientific rows "
                                     "in one runtime; select its recorded revision first")
                if not canonical:
                    raise ValueError(f"family {family_id}: runtime cohort needs its canonical configuration row before selection")
                selected = canonical[0]
                metadata = {"selection_kind": "canonical_fallback", "qualified": selected.get("status") == "PASS",
                            "reason": "Canonical configuration preferred; no outcome ranking across incomparable sources."}
        selected_families.add(family_id)
        selected_identity = scientific_row_hash(selected)
        display = deepcopy(selected)
        display.update(technique=family["label"], selection=metadata)
        selected_rows.append(display)
        for row in alternatives:
            identity = scientific_row_hash(row)
            row["selected_configuration"] = identity == selected_identity
            row["alternative_scope"] = ("selected" if row["selected_configuration"] else
                                        "archived_alternative" if use_explicit else "comparable_trial"
                                        if row["comparison_cohort"] == selected["comparison_cohort"] else "archived_alternative")
    selected_rows.sort(key=lambda row: (-row.get("qualified_tier", 0), row["technique"],
                                        row.get("runtime_cohort", {}).get("execution_backend", "")))
    result = {"rows": selected_rows, "configuration_rows": variants, "trainer_families": families}
    selection_path = Path(root) / CURRENT_SELECTION
    if view_policy is None and (selection_card is not None or selection_path.is_file()):
        history = []
        card = read_json(selection_path) if selection_card is None else selection_card
        for pin in card.get("historical_selections", []):
            matches = []
            for original in rows + (historical_rows or []):
                previous = deepcopy(original)
                previous["trainer_family"] = pin["trainer_family"]
                if family_row_pin(previous, selection_kind=pin["selection_kind"], reason=pin["reason"]) == pin:
                    matches.append(previous)
            if len(matches) != 1:
                raise ValueError("historical family page must match one exact verified scientific row")
            previous = matches[0]
            family = family_for_candidate(root, previous["candidate_id"], {"trainer_family": pin["trainer_family"]})
            if family["id"] != pin["trainer_family"]:
                raise ValueError("historical family page needs its original registered family")
            previous.update(technique=family["label"], selection={"selection_kind": "historical_incumbent",
                            "reason": pin["reason"], "qualified": False, "default_adoption": False})
            history.append(previous)
        if history:
            result["historical_family_rows"] = history
    return result
