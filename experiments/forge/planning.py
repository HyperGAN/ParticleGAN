"""Resolve small idea declarations to frozen scientific requests, without running."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import threading

from .contracts import atomic_json, canonical, identifier, read_json, stable_hash, validate_idea
from .sampling import candidate_blockers
from .execution_policy import DEFAULT as DEFAULT_EXECUTION_POLICY, group_blockers, policy
from .preflight import task_preflight
from .resolution_scope import active as active_resolution_cache, bind as bind_resolution_cache
from .sources import compute_profile, inspect_source, runtime_manifest, snapshot_source
from .views import (load_tasks, load_view, read_task_catalog, task_evaluation_fingerprint,
                    task_execution_fingerprint, validate_view, view_fingerprint)

FORMULATION_FIELDS = ("recipe_preset", "recipe_overrides", "prior", "extensions", "requires_capabilities",
                      "api_changes", "implementation", "initializer", "claim_contract")


def _blocker_sig(candidate: dict) -> str:
    """Memo key for per-task *preflight* blockers within one resolution cache.

    Preflight results are shared across candidates (the key has no candidate
    id). Tunable override values enter only through ``tunable_flags``; this is
    the historical preflight sharing contract and is unchanged. Bindings, which
    report effective recipe values, use ``_binding_sig`` instead.
    """
    from .boundaries import TUNABLE_FIELDS
    overrides = candidate.get("recipe_overrides") or {}
    resolved = candidate.get("resolved_recipe") or {}
    return canonical({
        "recipe_preset": candidate.get("recipe_preset"),
        "override_keys": sorted(overrides.keys()),
        "non_tunable_overrides": {k: v for k, v in overrides.items() if k not in TUNABLE_FIELDS},
        "tunable_flags": {
            "lr_floor_ne1": resolved.get("lr_floor", 1) != 1,
            "net_floor_active": resolved.get("network_lr_floor") not in (None, 1),
            "reg_coeff_nz": bool(resolved.get("reg_coeff")),
            "reg_every_ne1": resolved.get("reg_every", 1) != 1,
        },
        "prior": candidate.get("prior"),
        "requires_capabilities": candidate.get("requires_capabilities"),
        "extensions": candidate.get("extensions"),
        "initializer": candidate.get("initializer", "deterministic_orthogonal"),
        "execution_path": candidate.get("execution_path", "public_trainer"),
        "host_adaptation": candidate.get("host_adaptation"),
        "claim_contract": candidate.get("claim_contract"),
        "capabilities": candidate.get("capabilities"),
        **({"prior_mechanism": candidate["prior_mechanism"]} if candidate.get("prior_mechanism") is not None else {}),
    })


def _binding_sig(candidate: dict) -> str:
    """Memo key for per-task decision-contract bindings (finding #23).

    Bindings report effective recipe *values*, so every declared override
    value is part of the key, not only which overrides are present: two
    declarations with the same id and override names but a different tunable
    value (e.g. ``lr``) must never share a cached binding.
    """
    return canonical({"blockers": _blocker_sig(candidate),
                      "recipe_overrides": candidate.get("recipe_overrides") or {}})


class ResolutionCache:
    """Explicit memo tables for one read-only resolution scope over one checkout.

    Bound by ``resolution_cache(root)`` through ``resolution_scope`` (a
    ``ContextVar``), never by rebinding another module's functions. Readers that
    memoize consult ``resolution_scope.active(root)``. Memo population is
    serialized by ``self.lock`` so a deliberately shared cache stays consistent.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self._paths: list[Path] | None = None
        self._path_by_id: dict[str, Path] = {}
        self._ideas: dict[str, dict] = {}
        self._defaults: dict | None = None
        self._protocol: dict | None = None
        self._runtime: dict | None = None
        self._all_tasks: dict | None = None
        self._all_tasks_loader = None
        self._base_extra_sources: set[str] | None = None
        self._sources: dict[tuple, dict] = {}
        self._compute_profiles: dict[tuple, dict] = {}
        self._task_static: dict[str, tuple[str, str, str]] = {}
        self._views: dict[tuple, tuple[dict, str, dict]] = {}
        self._candidates: dict[str, dict] = {}
        self.preflight_cache: dict[tuple, list[str]] = {}
        self.tier1_validate_cache: dict[tuple, dict] = {}
        self.binding_task_cache: dict[tuple, dict] = {}
        self.contract_static_cache: dict[tuple, dict] = {}

    def get_declaration_paths(self) -> list[Path]:
        if self._paths is None:
            paths = sorted(path for directory in ("ideas", "configurations")
                           for path in (self.root / "configs/forge" / directory).glob("*.json"))
            if len({path.stem for path in paths}) != len(paths):
                raise ValueError("candidate ids must be unique across ideas and configurations")
            self._paths = paths
            self._path_by_id = {path.stem: path for path in paths}
        return self._paths

    def get_idea_ref(self, idea_id: str) -> dict:
        identifier(idea_id, "idea")
        if idea_id not in self._ideas:
            self.get_declaration_paths()
            path = self._path_by_id.get(idea_id)
            if path is None:
                raise FileNotFoundError(f"no declared candidate: {idea_id}")
            idea = read_json(path)
            validate_idea(idea)
            if idea["id"] != idea_id:
                raise ValueError("idea filename must match id")
            if path.parent.name == "configurations":
                from .configuration_search import validate_configuration_declaration
                validate_configuration_declaration(idea, root=self.root)
            self._ideas[idea_id] = idea
        return self._ideas[idea_id]

    def get_idea(self, idea_id: str) -> dict:
        return deepcopy(self.get_idea_ref(idea_id))

    def get_defaults(self) -> dict:
        if self._defaults is None:
            self._defaults = read_json(self.root / "configs/forge/defaults.json")
        return self._defaults

    def get_protocol(self) -> dict:
        if self._protocol is None:
            protocol_id = self.get_defaults()["protocol"]
            self._protocol = read_json(self.root / "configs/forge/protocols" / f"{protocol_id}.json")
        return self._protocol

    def get_runtime(self) -> dict:
        if self._runtime is None:
            self._runtime = runtime_manifest()
        return self._runtime

    def get_all_tasks(self) -> dict:
        with self.lock:
            if self._all_tasks is None:
                self._all_tasks = read_task_catalog(self.root)
                self._task_static.clear()
                self._views.clear()
            return self._all_tasks

    def loaded_tasks(self) -> dict | None:
        """The memoized catalogue, or None before anything has loaded it."""
        return self._all_tasks

    def tier1_validation(self, task: dict, root, validate):
        """Memoize ``tier1_policy`` validation of a catalogued task by id."""
        tid = task.get("id") if isinstance(task, dict) else None
        catalog = self._all_tasks
        if not tid or catalog is None or tid not in catalog:
            return validate(task, root=root)
        key = (tid, root is not None)
        with self.lock:
            if key not in self.tier1_validate_cache:
                self.tier1_validate_cache[key] = validate(task, root=root)
            return self.tier1_validate_cache[key]

    def bindings(self, candidate: dict, tasks: dict, protocol: dict, root, resolve):
        """Per-task memo of ``decision_contracts`` bindings, keyed by ``_binding_sig`` (override values)."""
        cid = candidate.get("id")
        sig = _binding_sig(candidate)
        result = {name: {} for name in ("recipe", "prior", "initialization", "budget", "sampling", "host",
                                        "evaluation", "task_identity")}
        for name, task in sorted(tasks.items()):
            tid = task.get("id", name)
            b_key = (cid, sig, tid)
            with self.lock:
                if b_key not in self.binding_task_cache:
                    single = resolve(candidate, {name: task}, protocol, root)
                    self.binding_task_cache[b_key] = {k: single[k][name] for k in result}
                entry = self.binding_task_cache[b_key]
            for k in result:
                result[k][name] = entry[k]
        return result

    def get_task_static(self, task_id: str, task: dict) -> tuple[str, str, str]:
        if task_id not in self._task_static:
            from .initialization import task_initializer
            self._task_static[task_id] = (
                task_execution_fingerprint(task),
                task_evaluation_fingerprint(task),
                task_initializer(task),
            )
        return self._task_static[task_id]

    def get_base_extra_sources(self) -> set[str]:
        if self._base_extra_sources is None:
            from .hostprofiles import profile_source_paths
            from .model_toy_sources import model_toy_source_paths
            extra: set[str] = set()
            for task in self.get_all_tasks().values():
                extra.update(relative for relative in task["evaluation"].get("sources", {})
                             if (self.root / relative).is_file())
                try:
                    profile_sources = profile_source_paths(task)
                except (KeyError, TypeError, ValueError):
                    profile_sources = {}
                extra.update(relative for relative in profile_sources
                             if (self.root / relative).is_file())
                extra.update(relative for relative in model_toy_source_paths(task)
                             if (self.root / relative).is_file())
            self._base_extra_sources = extra
        return self._base_extra_sources

    def get_source(self, extra_tuple: tuple[str, ...]) -> dict:
        key = (extra_tuple, inspect_source)
        if key not in self._sources:
            self._sources[key] = inspect_source(self.root, list(extra_tuple))
        return dict(self._sources[key])

    def get_compute_profile(self, backend: str, cuda_model: str | None) -> dict:
        key = (backend, cuda_model)
        if key not in self._compute_profiles:
            self._compute_profiles[key] = compute_profile(backend, cuda_model)
        return dict(self._compute_profiles[key])

    def get_view_info(self, view_id: str) -> tuple[dict, str, dict]:
        all_tasks = self.get_all_tasks()
        key = (view_id, load_view)
        if key not in self._views:
            view = load_view(self.root, view_id)
            validate_view(view, all_tasks)
            policy_fp = view_fingerprint(view)
            task_ids = [a["task"] for a in view["assignments"]]
            task_set = set(task_ids)
            groups: dict[str, list[str]] = {}
            for tid in task_ids:
                task = all_tasks[tid]
                group = task["execution"].get("execution_group", tid)
                groups.setdefault(group, []).append(tid)
            group_list = []
            for group, members in groups.items():
                member_set = set(members)
                members = sorted(members, key=lambda x: (
                    sum(d["task"] in member_set for d in all_tasks[x].get("dependencies", [])), x))
                lead = all_tasks[members[0]]
                exec_fp, eval_fp, task_inits = {}, {}, {}
                for m in members:
                    ex_f, ev_f, ti_f = self.get_task_static(m, all_tasks[m])
                    exec_fp[m] = ex_f
                    eval_fp[m] = ev_f
                    task_inits[m] = ti_f
                prerequisites: set[str] = set()
                def collect_dependencies(name, _mset=member_set, _prereq=prerequisites):
                    for dependency in all_tasks[name].get("dependencies", []):
                        required = dependency["task"] if isinstance(dependency, dict) else dependency
                        if required not in _mset and required not in _prereq:
                            _prereq.add(required)
                            collect_dependencies(required, _mset, _prereq)
                for member in members:
                    collect_dependencies(member)
                prereq_map = {}
                for name in sorted(prerequisites):
                    ex_f, ev_f, _ = self.get_task_static(name, all_tasks[name])
                    prereq_map[name] = {"execution": ex_f, "evaluation": ev_f}
                resources = lead["resources"]
                base_res = {
                    "memory_mb": resources["gpu_memory_mb"],
                    "gpus": resources["gpus"],
                    **({"host_memory_mb": resources["host_memory_mb"]} if "host_memory_mb" in resources else {}),
                    "cpu_threads": resources["cpu_threads"],
                }
                group_list.append({
                    "group": group,
                    "task_id": members[0],
                    "members": members,
                    "gpus": resources.get("gpus"),
                    "cpu_threads": resources["cpu_threads"],
                    "execution_fp": exec_fp,
                    "evaluation_fp": eval_fp,
                    "task_inits": task_inits,
                    "prerequisites": prereq_map,
                    "budget_seconds": max(all_tasks[m]["resources"]["timeout_seconds"] for m in members),
                    "base_resources": base_res,
                })
            self._views[key] = (view, policy_fp, {
                "task_ids": task_ids,
                "task_set": task_set,
                "groups": group_list,
            })
        return self._views[key]

    def get_candidate_info(self, idea_id: str) -> dict:
        key = (idea_id, inspect_source)
        if key not in self._candidates:
            idea = self.get_idea(idea_id)
            defaults = self.get_defaults()
            protocol = self.get_protocol()
            self.get_all_tasks()
            prior = {**defaults["prior"], **idea.get("prior", {})}
            if "prior_update" in idea.get("recipe_overrides", {}):
                prior["learnable"] = idea["recipe_overrides"]["prior_update"] == "learned"
            base_blockers = []
            if "TODO" in idea.get("hypothesis", "") or any("TODO" in x for x in idea["changed_factors"]):
                base_blockers.append("finish the scaffold's hypothesis and changed_factors before enqueue")
            if idea.get("extensions") and not idea.get("api_changes"):
                base_blockers.append("extensions require api_changes describing variables, provider, affected hosts and migration")
            if idea.get("implementation"):
                base_blockers.append("custom implementation loaders are unsupported; implement reusable changes in the public package and declare their Recipe/API bindings")
            from .api import CapabilityError, FormulationContext
            try:
                context = FormulationContext(
                    recipe_preset=idea.get("recipe_preset"),
                    recipe_overrides=idea.get("recipe_overrides", {}),
                    prior=prior,
                    prior_mechanism=idea.get("prior_mechanism"),
                    seed=protocol["seed"],
                    requires_capabilities=idea.get("requires_capabilities", []),
                    extensions=idea.get("extensions", {}),
                    initializer=idea.get("initializer", "deterministic_orthogonal"),
                    execution_path=idea.get("execution_path", "public_trainer"),
                )
                from particlegan.recipe_compat import without_default_additions
                recipe = without_default_additions(asdict(context.recipe), fields=("ebi_beta", "kappa_radial", "generator_gradient_response", 'reg_real_weight', 'reg_fake_weight', 'joint_reconstruction_weight', 'conditional_energy_weight', 'conditional_energy_samples', 'masked_mixture_weight', 'masked_transport_weight', 'imputer_component_weight', "rhbi_weight", "rhbi_rho", "rhbi_tau", "rhbi_eps", "rhbi_iters", "rhbi_c_max", "rhbi_beta_ell", "rhbi_nn_quantile", "rhbi_critic_weight", "rhbi_self_coupling", "rhbi_relativistic_weight", "rhbi_lambda_floor", "rhbi_gamma_gauge", "rhbi_rapidity_screening", "rhbi_scale_log_odds", "rhbi_conditional_fibre", "rhbi_akt_horizon", "rhbi_akt_mode", "rhbi_slack_gain", "rhbi_chord_weight", "rhbi_thermal_tau", "rhbi_thermal_shock_mu", "rhbi_settle_beta", "rhbi_settle_mode", "rhbi_fibre_rel_blend", "rhbi_step_damp", "rhbi_cond_ot_weight", "rhbi_cond_ot_mode", "rhbi_simplex_weight", "rhbi_cvx_mu_gain", "rhbi_simplex_chart_weight", "rhbi_tail_spill_gain", "rhbi_simplex_macro_settle"))
                if "configuration_id" in idea:
                    from .configuration_search import recipe_identity_fields
                    if stable_hash(recipe_identity_fields(recipe)) != stable_hash(recipe_identity_fields(idea["resolved_configuration_recipe"])):
                        raise ValueError("configuration frozen Recipe differs from current public defaults or API bindings; declare a new configuration")
                capabilities = [name for name, enabled in context.capabilities().items() if enabled]
                rng = context.streams.manifest()
            except CapabilityError as exc:
                recipe, capabilities, rng = idea.get("recipe_overrides", {}), [], protocol["rng"]
                base_blockers.extend(exc.blockers)
            candidate = {
                **idea,
                "resolved_recipe": recipe,
                "prior": prior,
                "capabilities": capabilities,
                "claim_contract": idea.get("claim_contract", {"schedule": "scheduled", "scoring_weights": "live"}),
            }
            base_blockers.extend(candidate_blockers(candidate))
            extra_sources = set(idea.get("source_files", [])) | self.get_base_extra_sources()
            source = self.get_source(tuple(sorted(extra_sources)))
            candidate_revision = candidate_revision_for(source["digest"], candidate)
            legacy_blockers = []
            if "decision_contract" not in candidate:
                if candidate.get("schema_version") == 3 and "configuration_id" not in candidate:
                    legacy_blockers.append("v3 candidate requires an explicit study or registered configuration search before enqueue")
                elif "configuration_id" not in candidate and (self.root / "configs/forge/legacy-ideas-v1.json").is_file():
                    from .decision_contracts import validate_legacy_admission
                    try:
                        validate_legacy_admission(
                            {"candidate": candidate, "requires_independent_grading": True},
                            root=self.root,
                        )
                    except ValueError as error:
                        legacy_blockers.append(str(error))
            self._candidates[key] = {
                "idea": idea,
                "candidate": candidate,
                "capabilities": capabilities,
                "rng": rng,
                "base_blockers": base_blockers,
                "source": source,
                "candidate_revision": candidate_revision,
                "blocker_sig": _blocker_sig(candidate),
                "legacy_blockers": legacy_blockers,
            }
        return self._candidates[key]

    def resolve_cached(self, idea_id: str, *, view_id: str | None = None,
                       through_tier: int | None = None, execution_backend: str | None = None,
                       cuda_model: str | None = None) -> dict:
        cand_info = self.get_candidate_info(idea_id)
        through_tier = 1 if through_tier is None else through_tier
        execution_backend = execution_backend or "cuda"
        if not isinstance(through_tier, int) or isinstance(through_tier, bool) or through_tier < 1:
            raise ValueError("through-tier must be an integer >= 1")
        idea = cand_info["idea"]
        candidate = cand_info["candidate"]
        capabilities = cand_info["capabilities"]
        rng = cand_info["rng"]
        protocol = self.get_protocol()
        runtime = self.get_runtime()
        all_tasks = self.get_all_tasks()
        actual_view_id = view_id or idea.get("goal", "discriminator_stability")
        view, policy_fp, group_specs = self.get_view_info(actual_view_id)
        blockers = list(cand_info["base_blockers"])
        for name, value in view.get("eligibility", {}).get("claim_contract", {}).items():
            if candidate["claim_contract"].get(name) != value:
                blockers.append(f"view requires claim_contract.{name}={value!r}")
        for name in view.get("eligibility", {}).get("requires_capabilities", []):
            if name not in capabilities:
                blockers.append(f"view requires capability {name}")
        view_task_set = group_specs["task_set"]
        tasks = {tid: dict(all_tasks[tid]) for tid in group_specs["task_ids"]}
        blocker_sig = cand_info["blocker_sig"]
        for tid, task in tasks.items():
            cont = task["execution"].get("continuation_of") if task.get("adapter") == "native100_continuation" else None
            pf_key = (blocker_sig, tid, cont in view_task_set if cont else None)
            if pf_key not in self.preflight_cache:
                probe = dict(all_tasks[tid])
                self.preflight_cache[pf_key] = task_preflight(probe, candidate, protocol, root=self.root, tasks=tasks)
            task["preflight_blockers"] = list(self.preflight_cache[pf_key])
        candidate_revision = cand_info["candidate_revision"]
        jobs = []
        compute_profiles = {}
        initializer = idea.get("initializer", "deterministic_orthogonal")
        is_diag = view.get("evidence_scope") == "research_diagnostic"
        for gspec in group_specs["groups"]:
            backend = "cpu" if gspec["gpus"] == 0 else execution_backend
            if backend not in compute_profiles:
                compute_profiles[backend] = self.get_compute_profile(backend, cuda_model)
            science = {
                "candidate_revision": candidate_revision,
                "execution": dict(gspec["execution_fp"]),
                "evaluation": dict(gspec["evaluation_fp"]),
                "protocol": protocol,
                "seed": protocol["seed"],
                "rng": rng,
                "initializer": initializer,
                "runtime": runtime,
                "task_initializers": dict(gspec["task_inits"]),
                "compute": {**compute_profiles[backend], "threads": gspec["cpu_threads"]},
            }
            if is_diag:
                science["evidence_use"] = "research_diagnostic"
            if gspec["prerequisites"]:
                science["prerequisites"] = {name: dict(val) for name, val in gspec["prerequisites"].items()}
            jobs.append({
                "task_id": gspec["task_id"],
                "task_ids": list(gspec["members"]),
                "execution_group": gspec["group"],
                "compatibility_key": stable_hash(science),
                "science": science,
                "budget_seconds": gspec["budget_seconds"],
                "resources": {
                    **gspec["base_resources"],
                    "allow_cpu": backend == "cpu",
                    "backend": backend,
                    "gpu_model": compute_profiles[backend].get("model") if backend == "cuda" else None,
                },
            })
        rekey_jobs(jobs)
        request = {
            "schema_version": 1,
            "candidate": candidate,
            "candidate_revision": candidate_revision,
            "execution_policy": dict(DEFAULT_EXECUTION_POLICY),
            "requires_independent_grading": True,
            "source": dict(cand_info["source"]),
            "runtime": runtime,
            "protocol": protocol,
            "rng": rng,
            "execution_backend": execution_backend,
            "compute_profiles": compute_profiles,
            "view": view,
            "policy_fingerprint": policy_fp,
            "tasks": tasks,
            "jobs": jobs,
            "through_tier": through_tier,
            "preflight_blockers": blockers,
        }
        if "decision_contract" in candidate:
            from .decision_contracts import inspect_contract
            review = inspect_contract(self.root, request)
            request["decision_review"] = review
            request["decision_admission"] = review["receipt"]
            blockers.extend(review["blockers"])
        else:
            blockers.extend(cand_info["legacy_blockers"])
        return request


_ResolutionCache = ResolutionCache  # Backwards-compatible private alias.


@contextmanager
def resolution_cache(root: Path):
    """Memoize read-only resolution for ``root`` within this context.

    Re-entrant: an inner scope for the same checkout reuses the active cache.
    Nothing outside this context (other threads included) observes the cache.
    """
    resolved = Path(root).resolve()
    current = active_resolution_cache(resolved)
    if current is not None:
        yield current
        return
    with bind_resolution_cache(ResolutionCache(resolved)) as cache:
        yield cache


def candidate_revision_for(source_digest: str, candidate: dict) -> str:
    """Identity of an already resolved formulation, independent of its label.

    Keep the field selection in one place for planning and frozen-request
    validation. Callers at execution boundaries must first reconstruct and
    validate resolved_recipe from the actual public formulation declaration.
    """
    from particlegan.recipe_compat import without_default_additions
    formulation = {key: candidate.get(key) for key in FORMULATION_FIELDS}
    # Preserve identities of existing declarations with no adaptation contract.
    if candidate.get("host_adaptation") is not None:
        formulation["host_adaptation"] = candidate["host_adaptation"]
    if candidate.get("prior_mechanism") is not None:
        formulation["prior_mechanism"] = candidate["prior_mechanism"]
    resolved_recipe = without_default_additions(candidate["resolved_recipe"], fields=("ebi_beta", "kappa_radial", "generator_gradient_response", 'reg_real_weight', 'reg_fake_weight', 'joint_reconstruction_weight', 'conditional_energy_weight', 'conditional_energy_samples', 'masked_mixture_weight', 'masked_transport_weight', 'imputer_component_weight', "rhbi_weight", "rhbi_rho", "rhbi_tau", "rhbi_eps", "rhbi_iters", "rhbi_c_max", "rhbi_beta_ell", "rhbi_nn_quantile", "rhbi_critic_weight", "rhbi_self_coupling", "rhbi_relativistic_weight", "rhbi_lambda_floor", "rhbi_gamma_gauge", "rhbi_rapidity_screening", "rhbi_scale_log_odds", "rhbi_conditional_fibre", "rhbi_akt_horizon", "rhbi_akt_mode", "rhbi_slack_gain", "rhbi_chord_weight", "rhbi_thermal_tau", "rhbi_thermal_shock_mu", "rhbi_settle_beta", "rhbi_settle_mode", "rhbi_fibre_rel_blend", "rhbi_step_damp", "rhbi_cond_ot_weight", "rhbi_cond_ot_mode", "rhbi_simplex_weight", "rhbi_cvx_mu_gain", "rhbi_simplex_chart_weight", "rhbi_tail_spill_gain", "rhbi_simplex_macro_settle"))
    formulation.update(resolved_recipe=resolved_recipe, prior=candidate["prior"],
                       api_version=candidate.get("api_version", "forge-api-v1"))
    return stable_hash({"source": source_digest, "formulation": formulation})


def rekey_jobs(jobs):
    """Rebind complete prerequisite identities after a protocol or lane change."""
    by_task = {member: job for job in jobs for member in job.get("task_ids", [job["task_id"]])}
    bound, visiting = set(), set()
    def bind(job):
        group = job.get("execution_group", job["task_id"])
        if group in bound:
            return job["compatibility_key"]
        if group in visiting:
            raise ValueError("cyclic execution-group prerequisites")
        visiting.add(group)
        parents = job["science"].get("prerequisites", {})
        if parents:
            job["science"]["prerequisites"] = {name: bind(by_task[name]) for name in sorted(parents)}
        job["compatibility_key"] = stable_hash(job["science"])
        visiting.remove(group)
        bound.add(group)
        return job["compatibility_key"]
    for job in jobs:
        bind(job)
    return jobs


def declaration_paths(root: Path) -> list[Path]:
    """Ordinary ideas and immutable configurations share the same planner."""
    cache = active_resolution_cache(root)
    if cache is not None:
        return list(cache.get_declaration_paths())
    paths = sorted(path for directory in ("ideas", "configurations")
                   for path in (Path(root) / "configs/forge" / directory).glob("*.json"))
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("candidate ids must be unique across ideas and configurations")
    return paths


def discover_candidate_ids(root: Path) -> list[str]:
    return sorted(path.stem for path in declaration_paths(root))


def load_idea(root: Path, idea_id: str) -> dict:
    cache = active_resolution_cache(root)
    if cache is not None:
        return cache.get_idea(idea_id)
    identifier(idea_id, "idea")
    matches = [path for path in declaration_paths(root) if path.stem == idea_id]
    if not matches:
        raise FileNotFoundError(f"no declared candidate: {idea_id}")
    path = matches[0]
    idea = read_json(path)
    validate_idea(idea)
    if idea["id"] != idea_id:
        raise ValueError("idea filename must match id")
    if path.parent.name == "configurations":
        from .configuration_search import validate_configuration_declaration
        validate_configuration_declaration(idea, root=root)
    return idea


def new_idea(root: Path, idea_id: str, parent: str, *, goal="discriminator_stability", hypothesis=None,
             study_id=None) -> Path:
    identifier(idea_id, "idea")
    path = Path(root) / "configs/forge/ideas" / f"{idea_id}.json"
    if path.exists():
        raise ValueError(f"idea already exists: {path}")
    inherited = load_idea(root, parent)
    idea = {k: deepcopy(inherited[k]) for k in FORMULATION_FIELDS if k in inherited}
    if "host_adaptation" in inherited:
        idea["host_adaptation"] = deepcopy(inherited["host_adaptation"])
    if "prior_mechanism" in inherited:
        idea["prior_mechanism"] = deepcopy(inherited["prior_mechanism"])
    idea.pop("prior", None)  # Priors are task conditions, not reusable recipes.
    from .boundaries import TASK_RECIPE_FIELDS
    idea["recipe_overrides"] = {key: value for key, value in idea.get("recipe_overrides", {}).items()
                                if key not in TASK_RECIPE_FIELDS}
    if "host_adaptation" in idea:
        fields = [key for key in idea["host_adaptation"]["recipe_fields"] if key not in TASK_RECIPE_FIELDS]
        if fields:
            idea["host_adaptation"]["recipe_fields"] = fields
        else:
            idea.pop("host_adaptation")
    from .studies import scaffold, study_path
    study = scaffold(study_id or f"{idea_id}-study", idea_id, parent, goal, hypothesis=hypothesis)
    if study_path(root, study["id"]).exists():
        raise ValueError("study already exists")
    idea.update(schema_version=3, id=idea_id, parent=parent,
                changed_factors=["TODO: describe the substantive change before enqueue"],
                mechanism_class=inherited.get("mechanism_class", "structural"),
                mechanism_rationale="TODO: explain structural, constant/floor or sampling-only change",
                api_version="forge-api-v1", execution_path="public_trainer",
                prior_art=[parent], guide="EXPERIMENTATION.md")
    validate_idea(idea)
    atomic_json(path, idea)
    atomic_json(study_path(root, study["id"]), study)
    return path


def resolve_idea(root: Path, idea_id: str, *, view_id: str | None = None,
                 through_tier: int | None = None, queue_root: Path | None = None,
                 freeze_source: bool = False, execution_backend: str | None = None,
                 cuda_model: str | None = None, declaration: dict | None = None,
                 study=None) -> dict:
    """Tier placement never enters candidate/task compatibility keys."""
    root = Path(root).resolve()
    cache = active_resolution_cache(root)
    if cache is not None and declaration is None and study is None and not freeze_source:
        return cache.resolve_cached(
            idea_id, view_id=view_id, through_tier=through_tier,
            execution_backend=execution_backend, cuda_model=cuda_model,
        )
    idea = load_idea(root, idea_id) if declaration is None else deepcopy(declaration)
    validate_idea(idea)
    if idea["id"] != idea_id:
        raise ValueError("declaration id must match requested candidate")
    if declaration is not None and "configuration_id" in idea:
        from .configuration_search import validate_configuration_declaration
        validate_configuration_declaration(idea, root=root)
    if study is not None:
        from .studies import load_study
        study = load_study(root, study)
        if study["candidate"] != idea_id:
            raise ValueError("study selects a different candidate")
        if "decision_contract" in idea:
            raise ValueError("legacy embedded decision_contract cannot be replaced by a study; declare a v3 successor")
        for supplied, expected in ((view_id, study["scope"]["view"]),
                                   (through_tier, study["scope"]["through_tier"]),
                                   (execution_backend, study["scope"]["execution_backend"]),
                                   (cuda_model, study["scope"].get("cuda_model"))):
            if supplied is not None and supplied != expected:
                raise ValueError("requested view/tier/backend/model differs from study scope")
        view_id, through_tier, execution_backend = (study["scope"][key] for key in
                                                   ("view", "through_tier", "execution_backend"))
        cuda_model = study["scope"].get("cuda_model")
    through_tier = 1 if through_tier is None else through_tier
    execution_backend = execution_backend or "cuda"
    if not isinstance(through_tier, int) or isinstance(through_tier, bool) or through_tier < 1:
        raise ValueError("through-tier must be an integer >= 1")
    defaults = read_json(root / "configs/forge/defaults.json")
    view = load_view(root, view_id or idea.get("goal", "discriminator_stability"))
    all_tasks = load_tasks(root)
    validate_view(view, all_tasks)
    tasks = {a["task"]: deepcopy(all_tasks[a["task"]]) for a in view["assignments"]}
    protocol_id = defaults["protocol"]
    protocol = read_json(root / "configs/forge/protocols" / f"{protocol_id}.json")
    prior = {**defaults["prior"], **idea.get("prior", {})}
    # Candidate reference construction cannot let a historical convenience
    # prior choose a new explicit recipe learning policy.
    if "prior_update" in idea.get("recipe_overrides", {}):
        prior["learnable"] = idea["recipe_overrides"]["prior_update"] == "learned"
    blockers = []
    if "TODO" in idea.get("hypothesis", "") or any("TODO" in x for x in idea["changed_factors"]):
        blockers.append("finish the scaffold's hypothesis and changed_factors before enqueue")
    if idea.get("extensions") and not idea.get("api_changes"):
        blockers.append("extensions require api_changes describing variables, provider, affected hosts and migration")
    if idea.get("implementation"):
        blockers.append("custom implementation loaders are unsupported; implement reusable changes in the public package and declare their Recipe/API bindings")
    from .api import CapabilityError, FormulationContext
    from .initialization import task_initializer
    try:
        context = FormulationContext(recipe_preset=idea.get("recipe_preset"),
            recipe_overrides=idea.get("recipe_overrides", {}), prior=prior,
            prior_mechanism=idea.get("prior_mechanism"),
            seed=protocol["seed"], requires_capabilities=idea.get("requires_capabilities", []),
            extensions=idea.get("extensions", {}), initializer=idea.get("initializer", "deterministic_orthogonal"),
            execution_path=idea.get("execution_path", "public_trainer"))
        from particlegan.recipe_compat import without_default_additions
        recipe = without_default_additions(asdict(context.recipe), fields=("ebi_beta", "kappa_radial", "generator_gradient_response", 'reg_real_weight', 'reg_fake_weight', 'joint_reconstruction_weight', 'conditional_energy_weight', 'conditional_energy_samples', 'masked_mixture_weight', 'masked_transport_weight', 'imputer_component_weight', "rhbi_weight", "rhbi_rho", "rhbi_tau", "rhbi_eps", "rhbi_iters", "rhbi_c_max", "rhbi_beta_ell", "rhbi_nn_quantile", "rhbi_critic_weight", "rhbi_self_coupling", "rhbi_relativistic_weight", "rhbi_lambda_floor", "rhbi_gamma_gauge", "rhbi_rapidity_screening", "rhbi_scale_log_odds", "rhbi_conditional_fibre", "rhbi_akt_horizon", "rhbi_akt_mode", "rhbi_slack_gain", "rhbi_chord_weight", "rhbi_thermal_tau", "rhbi_thermal_shock_mu", "rhbi_settle_beta", "rhbi_settle_mode", "rhbi_fibre_rel_blend", "rhbi_step_damp", "rhbi_cond_ot_weight", "rhbi_cond_ot_mode", "rhbi_simplex_weight", "rhbi_cvx_mu_gain", "rhbi_simplex_chart_weight", "rhbi_tail_spill_gain", "rhbi_simplex_macro_settle"))
        if "configuration_id" in idea:
            from .configuration_search import recipe_identity_fields
            if stable_hash(recipe_identity_fields(recipe)) != stable_hash(recipe_identity_fields(idea["resolved_configuration_recipe"])):
                raise ValueError("configuration frozen Recipe differs from current public defaults or API bindings; declare a new configuration")
        capabilities = [name for name, enabled in context.capabilities().items() if enabled]
        rng = context.streams.manifest()
    except CapabilityError as exc:
        recipe, capabilities, rng = idea.get("recipe_overrides", {}), [], protocol["rng"]
        blockers.extend(exc.blockers)
    candidate = {**idea, "resolved_recipe": recipe, "prior": prior, "capabilities": capabilities,
                 "claim_contract": idea.get("claim_contract", {"schedule": "scheduled", "scoring_weights": "live"})}
    blockers.extend(candidate_blockers(candidate))
    for name, value in view.get("eligibility", {}).get("claim_contract", {}).items():
        if candidate["claim_contract"].get(name) != value:
            blockers.append(f"view requires claim_contract.{name}={value!r}")
    for name in view.get("eligibility", {}).get("requires_capabilities", []):
        if name not in capabilities:
            blockers.append(f"view requires capability {name}")
    # Referenced evaluator source outside normal code roots must travel with a job.
    extra_sources = set(idea.get("source_files", []))
    # A view selects evidence, not a different candidate implementation. Capture
    # the same catalog evaluator support set for every view so quality/stability
    # requests can share their identical task receipts.
    for task in all_tasks.values():
        extra_sources.update(relative for relative in task["evaluation"].get("sources", {})
                             if (root / relative).is_file())
        # Published architecture declarations can live outside the code roots.
        # Capture the catalog's support set regardless of selected view. The
        # selected task's preflight validates every hash and reports missing or
        # malformed declarations as BLOCKED, before any worker is allocated.
        from .hostprofiles import profile_source_paths
        try:
            profile_sources = profile_source_paths(task)
        except (KeyError, TypeError, ValueError):
            profile_sources = {}
        extra_sources.update(relative for relative in profile_sources
                             if (root / relative).is_file())
        from .model_toy_sources import model_toy_source_paths
        extra_sources.update(relative for relative in model_toy_source_paths(task)
                             if (root / relative).is_file())
    for task in tasks.values():
        for relative in task["evaluation"].get("sources", {}):
            if (root / relative).is_file():
                extra_sources.add(relative)
        # The task owns its sampling law. Candidate priors describe the reference
        # formulation and cannot replace even a task's MoG width or code path.
        task["preflight_blockers"] = task_preflight(task, candidate, protocol, root=root, tasks=tasks)
    source = inspect_source(root, sorted(extra_sources))
    candidate_revision = candidate_revision_for(source["digest"], candidate)
    runtime = runtime_manifest()
    groups = {}
    for task_id, task in tasks.items():
        group = task["execution"].get("execution_group", task_id)
        groups.setdefault(group, []).append(task_id)
    jobs = []
    compute_profiles = {}
    for group, members in groups.items():
        # Stable member order must not depend on view order/retiering. The
        # execution producer precedes its checkpoint-derived evaluation tasks.
        member_set = set(members)
        members.sort(key=lambda x: (sum(d["task"] in member_set for d in tasks[x].get("dependencies", [])), x))
        task = tasks[members[0]]
        backend = "cpu" if task["resources"].get("gpus") == 0 else execution_backend
        if backend not in compute_profiles:
            compute_profiles[backend] = compute_profile(backend, cuda_model)
        science = {"candidate_revision": candidate_revision,
                   "execution": {m: task_execution_fingerprint(tasks[m]) for m in members},
                   "evaluation": {m: task_evaluation_fingerprint(tasks[m]) for m in members},
                   "protocol": protocol, "seed": protocol["seed"], "rng": rng,
                   "initializer": idea.get("initializer", "deterministic_orthogonal"), "runtime": runtime,
                   "task_initializers": {m: task_initializer(tasks[m]) for m in members},
                   "compute": {**compute_profiles[backend], "threads": task["resources"]["cpu_threads"]}}
        if view.get("evidence_scope") == "research_diagnostic":
            # Diagnostic measurements cannot alias an ordinary qualification job.
            science["evidence_use"] = "research_diagnostic"
        prerequisites = set()
        def collect_dependencies(name):
            for dependency in tasks[name].get("dependencies", []):
                required = dependency["task"] if isinstance(dependency, dict) else dependency
                if required not in member_set and required not in prerequisites:
                    prerequisites.add(required)
                    collect_dependencies(required)
        for member in members:
            collect_dependencies(member)
        if prerequisites:
            science["prerequisites"] = {name: {
                "execution": task_execution_fingerprint(tasks[name]),
                "evaluation": task_evaluation_fingerprint(tasks[name]),
            } for name in sorted(prerequisites)}
        resources = task["resources"]
        jobs.append({"task_id": members[0], "task_ids": members,
                     "execution_group": group, "compatibility_key": stable_hash(science),
                     "science": science, "budget_seconds": max(tasks[m]["resources"]["timeout_seconds"] for m in members),
                     "resources": {"memory_mb": resources["gpu_memory_mb"], "gpus": resources["gpus"],
                                   **({"host_memory_mb": resources["host_memory_mb"]} if "host_memory_mb" in resources else {}),
                                   "cpu_threads": resources["cpu_threads"], "allow_cpu": backend == "cpu",
                                   "backend": backend, "gpu_model": compute_profiles[backend].get("model") if backend == "cuda" else None}})
    rekey_jobs(jobs)
    request = {"schema_version": 1, "candidate": candidate, "candidate_revision": candidate_revision,
               "execution_policy": deepcopy(DEFAULT_EXECUTION_POLICY),
               "requires_independent_grading": True,
               "source": source, "runtime": runtime, "protocol": protocol, "rng": rng,
               "execution_backend": execution_backend, "compute_profiles": compute_profiles,
               "view": view, "policy_fingerprint": view_fingerprint(view), "tasks": tasks,
               "jobs": jobs, "through_tier": through_tier, "preflight_blockers": blockers}
    if study is not None:
        from .studies import inspect_study
        request["study"] = study
        review = inspect_study(root, request)
        request["study_review"] = review
        request["study_admission"] = review["receipt"]
        blockers.extend(review["blockers"])
    elif "decision_contract" in candidate:
        from .decision_contracts import inspect_contract
        review = inspect_contract(root, request)
        request["decision_review"] = review
        request["decision_admission"] = review["receipt"]
        blockers.extend(review["blockers"])
    elif candidate.get("schema_version") == 3 and "configuration_id" not in candidate:
        blockers.append("v3 candidate requires an explicit study or registered configuration search before enqueue")
    elif "configuration_id" not in candidate and (root / "configs/forge/legacy-ideas-v1.json").is_file():
        from .decision_contracts import validate_legacy_admission
        try:
            validate_legacy_admission(request, root=root)
        except ValueError as error:
            blockers.append(str(error))
    if freeze_source:
        if blockers:
            raise ValueError("submission blocked: " + "; ".join(blockers))
        if queue_root is None:
            raise ValueError("queue_root is required to freeze source")
        source["snapshot_path"] = str(snapshot_source(root, queue_root, source))
    return request


def plan_summary(request: dict, queue_state: dict | None = None, *, include_ownership=False) -> dict:
    existing = (queue_state or {}).get("jobs", {})
    by_task = {m: job for job in request["jobs"] for m in job.get("task_ids", [job["task_id"]])}
    tasks = []
    seen_groups = set()
    total = 0
    for assignment in sorted(request["view"]["assignments"], key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"])):
        task_id = assignment["task"]
        job = by_task[task_id]
        saved = existing.get(job["compatibility_key"])
        reusable = bool(saved and saved.get("status") == "terminal")
        allowed = assignment["qualification_tier"] <= request["through_tier"]
        if allowed and not reusable and job["compatibility_key"] not in seen_groups:
            total += job["budget_seconds"]
            seen_groups.add(job["compatibility_key"])
        tasks.append({**assignment, "reusable": reusable, "shared_pending": bool(saved and not reusable),
                      "permitted_by_tier_cap": allowed, "budget_seconds": job["budget_seconds"],
                      "execution_group": job["execution_group"],
                      "blockers": request["tasks"][task_id].get("preflight_blockers", []),
                      "execution_group_blockers": group_blockers(request, job),
                      **({"field_ownership": request["tasks"][task_id].get("field_ownership")}
                         if include_ownership else {})})
    return {"candidate": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
            "view": request["view"]["id"], "policy_fingerprint": request["policy_fingerprint"],
            "through_tier": request["through_tier"], "tasks": tasks, "worst_case_seconds": total,
            "execution_policy": policy(request),
            **({"study": request["study"], "study_binding": request["study_review"]} if "study" in request else {}),
            **({"decision_contract": request["decision_review"]} if "decision_review" in request else {}),
            "preflight_blockers": request["preflight_blockers"], "guide": "EXPERIMENTATION.md"}
