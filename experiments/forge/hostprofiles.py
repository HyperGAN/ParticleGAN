"""Revalidate experiment ownership at frozen-request execution boundaries.

Only snapshots containing this module opt into this boundary. Older frozen
registrations retain their original execution rules. Cached preflight results
are never an authority for current host architecture, prior or resource locks.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import canonical, stable_hash
from .initialization import MODULE as INITIALIZATION_MODULE, task_initializer
from .priors import task_prior

MODULE = "experiments/forge/hostprofiles.py"
PROFILE_ADAPTERS = {"image_profile": {"transfer_image"},
                    "vector_profile": {"transfer_vector"},
                    "native_profile": {"native100", "native100_continuation"}}
SUPPORTED_ADAPTERS = frozenset().union(*PROFILE_ADAPTERS.values())
BOUNDARY_ADAPTERS = SUPPORTED_ADAPTERS | {
    "transfer_behavior", "ring_endurance", "clockfree_audit", "paired_adaptation", "word_joint", "particle_interpolation",
}



def profile_source_paths(task: dict) -> list[str]:
    """One catalog support-file inventory; resolvers own exact source pins."""
    execution = task.get("execution", {})
    paths = set()
    if "image_profile" in execution:
        from .imageprofiles import profile_declaration
        if canonical(execution["image_profile"]) != canonical(profile_declaration()):
            raise ValueError("unsupported image profile source declaration")
        paths.add(profile_declaration()["source"]["path"])
    if "vector_profile" in execution:
        from .vectorprofiles import profile_source_files as vector_sources
        paths.update(vector_sources(task))
    if "native_profile" in execution:
        from .nativeprofiles import native_profile_source_files
        paths.update(native_profile_source_files(task))
    if (task.get("adapter") == "public_api_model_toy"
            and execution.get("host_definition", {}).get("provider") == "benchmarks.toy_audit.api_model_ae"):
        paths.update(("configs/forge/tasks/model_ae_reconstruction.json",
                      "configs/forge/tasks/model_ae_gan_multimodal.json"))
    return sorted(paths)


def _members(request, task_ids):
    if task_ids is not None:
        return set(task_ids)
    authorized = {row["task"] for row in request["view"]["assignments"]
                  if row["qualification_tier"] <= request["through_tier"]}
    members = set(authorized)
    for job in request["jobs"]:
        group = set(job.get("task_ids", [job["task_id"]]))
        if group & authorized:
            members.update(group)
    return members


def _gaussian_continuation(task):
    """Recognize only the fixed own-smoke continuation, without construction."""
    if task.get("evaluation", {}).get("kind") != "gaussian_stability":
        return False
    from .gaussian_tasks import validate_task
    validate_task(task)
    execution = task["execution"]
    for field, expected in (("steps", 6000), ("preserve_prefix_steps", 1000),
                            ("original_schedule_horizon", 1000)):
        if type(execution.get(field)) is not int or execution[field] != expected:
            raise ValueError(f"Gaussian continuation requires integer {field}={expected}")
    parent = execution.get("continuation_of")
    if (execution["host_definition"]["steps"] != 1000
            or execution.get("produces_state") is not True
            or not isinstance(parent, str) or not parent or parent == task["id"]):
        raise ValueError("Gaussian continuation requires its distinct own-smoke parent and 1,000-step host horizon")
    return True


def _validate_gaussian_parent(parent, child):
    from .gaussian_tasks import SMOKE_KIND, validate_task
    validate_task(parent)
    if (parent["evaluation"]["kind"] != SMOKE_KIND
            or parent["id"] != child["execution"]["continuation_of"]
            or type(parent["execution"]["steps"]) is not int
            or parent["execution"]["steps"] != 1000):
        raise ValueError("Gaussian continuation requires its matching 1,000-update smoke parent")
    for field in ("host", "host_source", "host_definition", "prior", "initializer", "protocol",
                  "original_schedule_horizon"):
        if canonical(parent["execution"].get(field)) != canonical(child["execution"].get(field)):
            raise ValueError(f"Gaussian continuation changes task-owned {field}")
    for field in ("sampling_contract_version", "eval_output_noise", "sampling_law", "scoring_weights",
                  "sample_evaluator", "eval_samples", "thresholds", "observations", "sources"):
        if canonical(parent["evaluation"].get(field)) != canonical(child["evaluation"].get(field)):
            raise ValueError(f"Gaussian continuation changes evaluation {field}")


def _validate_task(task, candidate, root, *, explicit_initializer=True):
    from .taskrecipes import bind_task_candidate
    initializer = task_initializer(task, candidate, explicit=explicit_initializer)
    reference_candidate = candidate
    candidate = bind_task_candidate(candidate, task)
    execution, adapter = task["execution"], task["adapter"]
    gaussian_continuation = _gaussian_continuation(task)
    for marker, adapters in PROFILE_ADAPTERS.items():
        if marker in execution and adapter not in adapters:
            raise ValueError(f"{marker} is unsupported by adapter {adapter}")
    if "host_initialization" in execution and (adapter in SUPPORTED_ADAPTERS or explicit_initializer):
        raise ValueError("host initialization must be declared inside a supported native profile")
    if adapter not in SUPPORTED_ADAPTERS:
        if explicit_initializer and adapter in BOUNDARY_ADAPTERS:
            if adapter == "word_joint":
                from .word_adapter import word_context
                word_context({"candidate": reference_candidate, "protocol": {"seed": 0}}, task, "cpu", root=root)
            elif adapter == "particle_interpolation":
                from .interpolation import interpolation_context
                interpolation_context({"candidate": reference_candidate, "protocol": {"seed": 0}}, task, "cpu", root=root)
            else:
                from .api import task_formulation_context
                task_formulation_context(reference_candidate, task, device="cpu", root=root)
            return True
        return False
    if adapter.startswith("native100"):
        from .nativeprofiles import native_profile_blockers
        blockers = native_profile_blockers(task, candidate, root=root, explicit_initializer=explicit_initializer)
        if blockers:
            raise ValueError("; ".join(blockers))
        # Legacy native tasks retain their old resource/init path. Current MLP
        # tasks own their resources and participate in the same frozen checks.
        if "native_profile" not in execution:
            if "resources" not in execution:
                return False
            resources = execution["resources"]
        else:
            from .nativeprofiles import resolve_native_spec
            resources = resolve_native_spec(task, root=root)["resources"]
    else:
        if adapter == "transfer_image":
            from .imageprofiles import resolve_image_spec
            spec = resolve_image_spec(task, root=root)
        else:
            from .vectorprofiles import resolve_vector_spec
            spec = resolve_vector_spec(task, root=root)
        if execution["steps"] != spec["steps"] and not gaussian_continuation:
            raise ValueError("execution budget differs from the explicit host card")
        resources = {"num_particles": spec["particles"], "z_dim": spec["z_dim"],
                     "batch_size": spec["batch_size"] if adapter == "transfer_image" else spec["batch"]}
    prior = task_prior(task)
    from .api import host_recipe_overrides, task_policy_blockers
    blockers = task_policy_blockers(task, candidate)
    if blockers:
        raise ValueError("; ".join(blockers))
    overrides = host_recipe_overrides(candidate, execution, resources)
    # Shared public validation without model construction or GPU allocation.
    # Native policy syntax/applicability was already resolved by its helper.
    from .api import FormulationContext
    if explicit_initializer:
        from .api import task_formulation_context
        task_formulation_context(reference_candidate, task, device="cpu", root=root)
    else:
        FormulationContext(recipe_preset=candidate.get("recipe_preset"), recipe_overrides=overrides, prior=prior,
            prior_mechanism=candidate.get("prior_mechanism"),
            device="cpu", requires_capabilities=tuple(candidate.get("requires_capabilities", ()))
                + tuple(task["requires_capabilities"]), extensions=candidate.get("extensions", {}),
            initializer=initializer)
    return True



def _optimizer_recipe_identity(recipe):
    """Keep newly implicit defaults compatible with frozen resolved recipes."""
    if not isinstance(recipe, dict):
        return recipe
    from particlegan.recipe_compat import without_default_additions
    recipe = without_default_additions(recipe)
    momentum = recipe.get("optimizer_momentum")
    if type(momentum) in (int, float) and momentum == 0:
        recipe.pop("optimizer_momentum")
    if recipe.get("optimizer_adam_lr") is None:
        recipe.pop("optimizer_adam_lr", None)
    return recipe


def _validate_candidate_identity(request):
    from dataclasses import asdict
    from .api import FormulationContext
    from .planning import candidate_revision_for

    candidate = request["candidate"]
    from .taskrecipes import validate_host_adaptation
    validate_host_adaptation(candidate)
    # Rebuild from declarations, not candidate-provided resolved_recipe echoes.
    # Recipe resolution has no model construction or training draw and does not
    # depend on promotion seed; the registered RNG protocol is checked separately.
    context = FormulationContext(recipe_preset=candidate.get("recipe_preset"),
        recipe_overrides=candidate.get("recipe_overrides", {}),
        prior=candidate.get("prior"), prior_mechanism=candidate.get("prior_mechanism"),
        device="cpu",
        requires_capabilities=candidate.get("requires_capabilities", ()),
        extensions=candidate.get("extensions", {}),
        initializer=candidate.get("initializer", "deterministic_orthogonal"),
        execution_path=candidate.get("execution_path", "public_trainer"))
    resolved_recipe = asdict(context.recipe)
    if "configuration_id" in candidate:
        from .configuration_search import recipe_identity_fields, validate_configuration_declaration
        validate_configuration_declaration(candidate)
        # Older immutable cards recorded the one paired-logistic objective
        # implicitly. Normalize only that default for this legacy comparison;
        # the actual request, alternative objectives and source identity stay
        # explicit and continue through the strict checks below.
        if canonical(recipe_identity_fields(candidate["resolved_configuration_recipe"])) != canonical(recipe_identity_fields(resolved_recipe)):
            raise ValueError("configuration frozen Recipe differs from its actual public formulation")
    if canonical(_optimizer_recipe_identity(candidate.get("resolved_recipe"))) != canonical(_optimizer_recipe_identity(resolved_recipe)):
        raise ValueError("candidate resolved_recipe differs from its actual public formulation")
    # The semantic check above permits only the new implicit defaults. Hash
    # the exact recorded dictionary to retain each frozen request's identity.
    expected = candidate_revision_for(request["source"]["digest"], candidate)
    if request.get("candidate_revision") != expected:
        raise ValueError("candidate scientific identity differs from its actual formulation/source")


def _validate_job_identity(request, checked, *, explicit_initializer=True):
    from .views import task_execution_fingerprint, task_evaluation_fingerprint
    covered = set()
    for job in request["jobs"]:
        members = set(job.get("task_ids", [job["task_id"]]))
        relevant = members & checked
        if not relevant:
            continue
        if covered & relevant:
            raise ValueError("host task appears in multiple execution jobs")
        covered.update(relevant)
        science = job["science"]
        leader_resources = request["tasks"][job["task_id"]]["resources"]
        locked_resources = {"memory_mb": leader_resources["gpu_memory_mb"],
                            "gpus": leader_resources["gpus"], "cpu_threads": leader_resources["cpu_threads"]}
        if "host_memory_mb" in leader_resources:
            locked_resources["host_memory_mb"] = leader_resources["host_memory_mb"]
        if (any(job["resources"].get(key) != value for key, value in locked_resources.items())
                or job["budget_seconds"] != max(request["tasks"][name]["resources"]["timeout_seconds"] for name in members)
                or science.get("compute", {}).get("threads") != locked_resources["cpu_threads"]):
            raise ValueError("host job resources differ from its frozen task/compute budget")
        for member in relevant:
            task = request["tasks"][member]
            gaussian_continuation = _gaussian_continuation(task)
            if task.get("adapter") == "native100_continuation" or gaussian_continuation:
                parent_id = task["execution"]["continuation_of"]
                parents = [row for row in request["jobs"] if parent_id in row.get("task_ids", [row["task_id"]])]
                if (len(parents) != 1 or science.get("prerequisites", {}).get(parent_id)
                        != parents[0]["compatibility_key"] or (gaussian_continuation and parents[0] is job)):
                    name = "Gaussian" if gaussian_continuation else "native"
                    raise ValueError(f"{name} continuation scientific identity lacks its exact own-parent job")
        expected_execution = {name: task_execution_fingerprint(request["tasks"][name]) for name in members}
        expected_evaluation = {name: task_evaluation_fingerprint(request["tasks"][name]) for name in members}
        if explicit_initializer:
            expected_initializers = {name: task_initializer(request["tasks"][name], request["candidate"])
                                     for name in members}
            if science.get("task_initializers") != expected_initializers:
                raise ValueError("host job initialization differs from its experiment-owned task policies")
        if (science.get("execution") != expected_execution or science.get("evaluation") != expected_evaluation
                or science.get("candidate_revision") != request["candidate_revision"]
                or science.get("initializer") != request["candidate"].get("initializer", "deterministic_orthogonal")
                or job["compatibility_key"] != stable_hash(science)):
            raise ValueError("host job scientific identity differs from its resolved task/formulation")
    if covered != checked:
        raise ValueError("authorized host task lacks its execution job")


def validate_request_host_profiles(request: dict, *, task_ids=None) -> None:
    """Check all authorized grouped members against the actual frozen root.

    Call before queue mutation/reservation, and again immediately before dispatch.
    Valid registered diagnostics and promotions retain their own namespaced job
    keys; those identities still bind the same measured host and effective prior.
    """
    source = request.get("source", {})
    snapshot = source.get("snapshot_path")
    present = isinstance(snapshot, str) and bool(snapshot) and (Path(snapshot) / MODULE).is_file()
    if MODULE not in source.get("files", {}) and not present:
        return
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("host profile validation requires a frozen source snapshot")
    from .sources import verify_snapshot
    root = Path(snapshot)
    verify_snapshot(root, source)
    # The new source contract owns initialization in tasks. Existing frozen
    # requests without the helper retain their original candidate fallback and
    # scientific keys; their old source executes its original binding rules.
    explicit_initializer = INITIALIZATION_MODULE in source.get("files", {})
    blockers, checked = [], set()
    tasks = request.get("tasks", {})
    for member in sorted(_members(request, task_ids)):
        if member not in tasks:
            blockers.append(f"{member}: authorized task lacks its frozen definition")
            continue
        task = tasks[member]
        try:
            if _validate_task(task, request["candidate"], root, explicit_initializer=explicit_initializer):
                checked.add(member)
            if task.get("adapter") == "native100_continuation":
                from .nativeprofiles import validate_native_continuation
                parent_id = task["execution"]["continuation_of"]
                if parent_id not in tasks:
                    raise ValueError("native continuation lacks its declared parent task")
                parent = tasks[parent_id]
                if _validate_task(parent, request["candidate"], root, explicit_initializer=explicit_initializer):
                    checked.add(parent_id)
                validate_native_continuation(parent, task, root=root)
            elif _gaussian_continuation(task):
                parent_id = task["execution"]["continuation_of"]
                if parent_id not in tasks:
                    raise ValueError("Gaussian continuation lacks its declared smoke parent task")
                parent = tasks[parent_id]
                if _validate_task(parent, request["candidate"], root, explicit_initializer=explicit_initializer):
                    checked.add(parent_id)
                _validate_gaussian_parent(parent, task)
        except (KeyError, TypeError, ValueError, OSError) as error:
            blockers.append(f"{member}: {error}")
    if not blockers:
        try:
            _validate_candidate_identity(request)
            _validate_job_identity(request, checked, explicit_initializer=explicit_initializer)
        except (KeyError, TypeError, ValueError) as error:
            blockers.append(str(error))
    if blockers:
        raise ValueError("host profile blocked: " + "; ".join(blockers))
