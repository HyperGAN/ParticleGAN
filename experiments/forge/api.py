"""Forge's thin construction contract around the public ParticleGAN API.

There is no second optimizer or training loop here. New experiment variables
bind explicitly to a documented public Recipe or GANTrainer argument.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import inspect
import math

import torch

from particlegan import (
    GANTrainer, Recipe, get_recipe, init, unwrap_prior,
)
from .rng import NamedStreams, RNG_VERSION
from .priors import (
    prior_noise_sigma, prior_uses_noise_stream, resolve_prior, resolve_prior_mechanism,
)


API_VERSION = "forge-api-v1"
TRAINER_STREAM_BINDINGS = {
    "latent_generator": ("prior", "latent", "indices"),
    "penalty_generator": ("noise", "penalty", "training"),
    "noise_generator": ("noise", "generator", "output"),
    "input_noise_generator": ("noise", "critic", "input"),
    "eval_generator": ("eval", "sampler", "samples"),
    "model_generator": ("noise", "models", "stochastic_layers"),
    "prior_noise_generator": ("noise", "prior", "gaussian"),
}
BUILTIN_CAPABILITIES = (
    "public_trainer", "public_components", "scalar_gan", "mog_prior", "particle_cloud", "learned_locations",
    "uniform_masses", "fixed_prior_width", "a2", "checkpoint", "named_rng", "live_sampling",
    "policy_controls", "policy_serving", "served_sampling",
)
COMPONENT_OBJECTIVE_HOSTS = frozenset({
    "two_pole", "unused_token_hold", "ae_gan_hold", "trajectory",
    "residual_student", "unipolar", "cover_leftover", "mid_scale_identity",
    "five_word_joint",
})
COMPONENT_TRANSPORT_COHORTS = frozenset({
    "conditional_transport_round5_v1", "component_output_transport_v1",
})
_MISSING = object()


class CapabilityError(ValueError):
    """Pretraining incompatibility; callers record BLOCKED, not scientific FAIL."""
    status = "BLOCKED"

    def __init__(self, blockers):
        self.blockers = list(blockers)
        super().__init__("; ".join(self.blockers))


def require_legacy_autograd_source(trainer_class, expected_sha256, *, owner):
    """Refuse an original owner before construction when its trainer changed."""
    from pathlib import Path
    source = Path(inspect.getfile(trainer_class)).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise CapabilityError([
            f"{owner} requires its pinned original GANTrainer source; the current "
            "fixed serial autograd policy cannot substitute for that historical execution cohort"])


def resolve_public_recipe(candidate, **overrides):
    """Resolve the versioned declaration binding; Recipe.name is only a label.

    Forge API v1 freezes ``bcap`` to the original native-Adam preset, now
    exposed publicly as ``bcap_adam``. Public ``get_recipe("bcap")`` may select
    the winning public default without changing archived Forge cards or
    their identities. Explicit optimizer and rate overrides still take effect.
    Its unnamed historical cards explicitly bind the KA2 preset. New cards
    should declare ``recipe_preset``; this versioned protocol does not select
    restoration defaults as a substitute for the public recipe factory.
    """
    options = {**candidate.get("recipe_overrides", {}), **overrides}
    preset = candidate.get("recipe_preset")
    try:
        if API_VERSION == "forge-api-v1" and preset == "bcap":
            label = options.pop("name", "bcap")
            return get_recipe("bcap_adam", **options).replace(name=label)
        if preset is None:
            label = options.pop("name", "ka2")
            return get_recipe("ka2", **options).replace(name=label)
        return get_recipe(preset, **options)
    except (TypeError, ValueError) as error:
        raise CapabilityError([f"unsupported recipe: {error}"]) from error


def policy_controls(recipe):
    """Mechanisms requiring the ordered public UpdatePolicy lifecycle."""
    return bool(recipe.continuous_policy is not None or recipe.lr_control == "stationarity"
                or recipe.row_evidence_gate or recipe.particle_birth_death
                or recipe.serve_average or recipe.output_noise_mode != "fixed")


def host_recipe_overrides(candidate, execution, resources):
    """Preserve schedule-free None; task steps bound GANTrainer.max_steps."""
    recipe = resolve_public_recipe(candidate)
    fixed = dict(resources)
    if recipe.total_steps is not None:
        fixed["total_steps"] = execution.get("original_schedule_horizon", execution["steps"])
    conflicts = [key for key, value in fixed.items()
                 if key in candidate.get("recipe_overrides", {})
                 and candidate["recipe_overrides"][key] != value]
    if conflicts:
        raise CapabilityError([f"candidate overrides frozen host resource {key}; declare a different task"
                               for key in sorted(conflicts)])
    return {**candidate.get("recipe_overrides", {}), **fixed}


def task_recipe_resources(task):
    """Resolve only the resources consumed by the task's existing adapter."""
    execution, adapter = task["execution"], task["adapter"]
    if adapter in {"transfer_vector", "paired_adaptation", "transfer_image"}:
        spec = execution["host_definition"]
        return {"num_particles": spec.get("particles", 256), "z_dim": spec["z_dim"],
                "batch_size": spec["batch_size"] if adapter == "transfer_image" else spec["batch"]}
    if adapter == "ring_endurance" or (adapter == "transfer_behavior" and execution.get("host") == "mode_hold"):
        from benchmarks.legacy.locked_shared import LOCKED_SHARED
        return {"num_particles": LOCKED_SHARED.n_particles, "z_dim": 4, "batch_size": 128}
    if adapter in {"native100", "native100_continuation"} and "native_profile" in execution:
        return deepcopy(execution["host_definition"]["resources"])
    if adapter in {"word_joint", "public_api_model_toy", "particle_interpolation"}:
        return deepcopy(execution["host_definition"]["resources"])
    return deepcopy(execution.get("resources", {}))


# Common component host boundary overrides: hosts without UpdatePolicy lifecycle
# support must deactivate continuous policy, stationarity ladders, and birth-death
# mechanics uniformly across all candidate families.
COMPONENT_HOST_BOUNDARY_OVERRIDES = {
    "particle_birth_death": False,
    "row_evidence_gate": False,
    "serve_average": 0.0,
    "birth_death_feature_scale": "none",
    "birth_death_isolation": False,
    "birth_death_backend": "knn",
    "birth_death_cells": 64,
    "continuous_policy": None,
    "lr_control": "mobility",
    "reopen_guard": None,
}


def task_recipe_overrides(candidate, task):
    """Materialize task binding once, identically for planning and execution."""
    from .taskrecipes import bind_task_candidate
    from .tier1_policy import is_policy_task
    bound = bind_task_candidate(candidate, task)
    execution = task["execution"]
    if task["adapter"] == "public_api_model_toy":
        from .model_toy_adapter import validate_task
        validate_task(task)
        resources = {**task_recipe_resources(task), **execution["recipe_fields"]}
        overrides = host_recipe_overrides(bound, execution, resources)
        overrides.update(
            COMPONENT_HOST_BOUNDARY_OVERRIDES,
            prior_kind="mog",
            prior_lr_mult=0.02,
            total_steps=execution.get("original_schedule_horizon", execution.get("steps", 1200)),
        )
        return overrides
    if task["adapter"] == "word_joint" and not is_policy_task(task):
        overrides = host_recipe_overrides(bound, execution, task_recipe_resources(task))
        overrides.update(
            COMPONENT_HOST_BOUNDARY_OVERRIDES,
            input_noise_std=0.5,
            total_steps=execution.get("original_schedule_horizon", execution["steps"]),
        )
        return overrides
    if (task["adapter"] == "transfer_behavior" and execution.get("host") != "mode_hold"
            and not is_policy_task(task)):
        from .behavior_adapters import behavior_preflight
        blockers = behavior_preflight(task, bound)
        if blockers:
            raise CapabilityError(blockers)
        overrides = dict(bound.get("recipe_overrides", {}))
        resolve_public_recipe(bound)
        overrides.update(
            COMPONENT_HOST_BOUNDARY_OVERRIDES,
            total_steps=execution.get("original_schedule_horizon", execution.get("steps", 100)),
        )
        if execution.get("host") == "ae_gan_hold":
            from benchmarks.locked_shared.hosts.ae_gan_hold import HoldConfig
            cfg = HoldConfig(name="forge")
            overrides.update(encoder_mode="ae", z_dim=2, num_particles=cfg.n_particles, batch_size=cfg.batch)
        return overrides
    return host_recipe_overrides(bound, execution, task_recipe_resources(task))


# Hosts whose every training and evaluation latent is drawn through the public
# ``prior.sample`` path (GANTrainer or the locked mode_hold loop) on a MoG
# table. Other hosts enumerate, encode against or directly optimize table rows,
# so Recipe.prior_code_law='fresh_gaussian' has no defined meaning there.
FRESH_GAUSSIAN_ADAPTERS = frozenset({"transfer_vector", "native100", "native100_continuation",
                                     "clockfree_audit", "ring_endurance"})
FRESH_GAUSSIAN_BEHAVIOR_HOSTS = frozenset({"mode_hold"})


def fresh_gaussian_blockers(task, recipe):
    """Explicit blockers for the opt-in fresh-Gaussian latent control."""
    if getattr(recipe, "prior_code_law", "table") == "table":
        return []
    execution = task.get("execution", {})
    prior = execution.get("prior") or {}
    supported = (task.get("adapter") in FRESH_GAUSSIAN_ADAPTERS
                 or (task.get("adapter") == "transfer_behavior"
                     and execution.get("host") in FRESH_GAUSSIAN_BEHAVIOR_HOSTS))
    if prior.get("kind") != "mog" or execution.get("prior_applicability") == "not_sampled" or not supported:
        return [f"{task.get('id')}: Recipe.prior_code_law='fresh_gaussian' requires a sampled MoG latent "
                "drawn only through prior.sample; this host reads, encodes or optimizes table rows"]
    return []


def task_formulation_context(candidate, task, protocol=None, *, device="cpu", root=None):
    """Resolve a task's actual public configuration without constructing models."""
    from .initialization import task_initializer
    from .nativeprofiles import native_host_initialization
    from .priors import task_prior
    from .taskrecipes import adaptation_receipt, bind_task_candidate
    from .tier1_policy import is_policy_task
    bound = bind_task_candidate(candidate, task)
    blockers = task_policy_blockers(task, bound)
    if blockers:
        raise CapabilityError(blockers)
    overrides = task_recipe_overrides(candidate, task)
    is_encoder = overrides.get("encoder_mode", "none") != "none"
    is_model_toy = task.get("adapter") == "public_api_model_toy"
    is_clean_word = task.get("adapter") == "word_joint" and not is_policy_task(task)
    is_clean_behavior = (task.get("adapter") == "transfer_behavior"
                         and task.get("execution", {}).get("host") != "mode_hold"
                         and not is_policy_task(task))
    cand_caps = [c for c in bound.get("requires_capabilities", ())
                 if not (is_encoder and c == "particle_cloud")
                 and not (is_model_toy and c in ("policy_controls", "policy_serving", "particle_cloud"))
                 and not ((is_clean_word or is_clean_behavior) and c in ("policy_controls", "policy_serving"))]
    prior_mechanism = None if is_model_toy else bound.get("prior_mechanism")
    context = FormulationContext(recipe_preset=bound.get("recipe_preset"),
        recipe_overrides=overrides, prior=task_prior(task),
        prior_mechanism=prior_mechanism,
        seed=(protocol or {}).get("seed", 0), device=device,
        requires_capabilities=tuple(cand_caps) + tuple(task["requires_capabilities"]),
        extensions=bound.get("extensions", {}), initializer=task_initializer(task, candidate),
        host_initialization=native_host_initialization(task, root=root),
        execution_path=task["execution"].get("execution_path", bound.get("execution_path", "public_trainer")),
        component_transport=task["execution"].get("transport_consumer"),
        prior_contract=task["execution"].get("prior_contract"),
        policy_task=task if is_policy_task(task) else None)
    # Typed extensions receive the same ownership and policy checks as ordinary
    # overrides before a worker can reserve this task.
    blockers = task_policy_blockers(task, {"recipe_overrides": asdict(context.recipe),
                                           "prior_mechanism": bound.get("prior_mechanism")})
    blockers.extend(fresh_gaussian_blockers(task, context.recipe))
    if (task["adapter"] == "transfer_behavior" and task["execution"].get("host") != "mode_hold"
            and not is_policy_task(task)):
        from .behavior_adapters import behavior_preflight
        blockers.extend(behavior_preflight(task, {"recipe_overrides": context.bindings["recipe"]}))
    if blockers:
        raise CapabilityError(blockers)
    from .boundaries import ownership_receipt
    cand_for_receipt = deepcopy(candidate)
    if is_encoder or is_model_toy:
        cand_for_receipt.pop("prior_mechanism", None)
    ownership_receipt(cand_for_receipt, task, asdict(context.recipe), protocol,
                      initializer=context.initializer,
                      extension_recipe_bindings=context.bindings["recipe"])
    if context.recipe.constraint_geometry_mode != "none":
        if (context.execution_path == "public_components"
                and task["adapter"] != "public_api_model_toy"
                and task["execution"].get("host") not in COMPONENT_OBJECTIVE_HOSTS):
            raise CapabilityError(["constraint_geometry protected-loss hook is unavailable for this host"])
        if context.recipe.constraint_geometry_mode == "strict_progress" and (
                context.recipe.input_noise_std or context.recipe.output_noise_std or context.recipe.standardize):
            raise CapabilityError(["strict_progress finite replay requires zero additive noise and an unstandardized prior"])
    context.host_adaptation = adaptation_receipt(candidate, task)
    context.ownership_contract = {"candidate": deepcopy(cand_for_receipt), "task": deepcopy(task),
                                  "protocol": deepcopy(protocol or {}),
                                  "extension_recipe_bindings": deepcopy(context.bindings["recipe"])}
    return context


PUBLIC_TRAINER_POLICY_ADAPTERS = frozenset({
    "transfer_vector", "transfer_image", "native100", "native100_continuation",
    "ring_endurance", "paired_adaptation", "clockfree_audit",
    "word_joint", "public_api_model_toy", "transfer_behavior", "particle_interpolation",
})


def task_policy_blockers(task, candidate):
    """Current Forge tasks certify clean/live component laws, not E22 controls."""
    try:
        recipe = resolve_public_recipe(candidate)
    except CapabilityError as error:
        return error.blockers
    if recipe.conditional_energy_weight:
        host = task.get("execution", {}).get("host_definition", {})
        if (task.get("adapter") != "public_api_model_toy"
                or host.get("provider") != "benchmarks.toy_audit.api_model_ddgan"):
            return ["conditional energy score requires an explicitly supported paired conditional host"]
    if ((recipe.masked_mixture_weight or recipe.masked_transport_weight or recipe.imputer_component_weight)
            and task.get("execution", {}).get("host_definition", {}).get("provider") != "benchmarks.toy_audit.api_model_misgan"):
        return ["masked mixture/imputer objectives require the explicit MisGAN component consumer"]
    from .tier1_policy import blocked_host, is_policy_task
    policy_task = is_policy_task(task)
    if policy_task and blocked_host(task) is not None:
        # An explicit ownership boundary is the actual reason; report it before
        # inherited metadata (for example a component transport consumer).
        from .tier1_policy import blockers
        return blockers(task, recipe)
    consumer = task.get('execution', {}).get('transport_consumer')
    # The selected-policy two-pole declaration inherits its parent's optional
    # transport metadata. Its existing owner implements the original objective
    # when transport is disabled; it does not implement a transport loss.
    policy_two_pole = (policy_task
        and task.get('execution', {}).get('host') == 'two_pole'
        and not (recipe.kinetic_transport_weight or recipe.kinetic_transport_local_weight))
    execution = task.get('execution', {})
    model_vae_transport = (task.get('adapter') == 'public_api_model_toy'
        and task.get('task_cohort') == 'api_model_vae_seed0_v1'
        and execution.get('host') in ('api-model-vae-hard', 'api-model-vae-categorical')
        and execution.get('host_definition', {}).get('provider') == 'benchmarks.toy_audit.api_model_vae')
    if consumer is not None and not model_vae_transport and (consumer != 'output_marginal_v1'
            or task.get('adapter') not in ('transfer_behavior', 'word_joint')
            or task.get('execution', {}).get('host') not in COMPONENT_OBJECTIVE_HOSTS
            or (task.get('task_cohort') not in COMPONENT_TRANSPORT_COHORTS and not policy_two_pole)):
        return ['unsupported explicit component transport consumer']
    if model_vae_transport and consumer not in (None, 'output_marginal_v1'):
        return ['unsupported explicit model VAE transport consumer']
    if (recipe.kinetic_transport_weight or recipe.kinetic_transport_local_weight) and task.get("execution", {}).get("execution_path") == "public_components" and consumer is None:
        return [f"{task.get('id', '<task>')}: frozen public_components host does not consume "
                "Recipe kinetic transport sample-space losses; mechanism is unsupported"]
    if policy_task:
        from .tier1_policy import blockers
        return blockers(task, recipe)
    # Update throughput does not certify the generated sampling distribution.
    # This explicit diagnostic host runs GANTrainer's full policy lifecycle;
    # it never disables policy controls to fit an ordinary clean/live gate.
    from .scaling_adapter import is_performance_task, validate_task as validate_performance_task
    if is_performance_task(task):
        try:
            validate_performance_task(task)
        except (KeyError, TypeError, ValueError) as error:
            return [str(error)]
        return []
    if not policy_controls(recipe):
        return []
    if candidate.get("prior_mechanism") is not None:
        adapter = task.get("adapter")
        if adapter in PUBLIC_TRAINER_POLICY_ADAPTERS:
            return []
    host = "public components" if task.get("adapter") == "transfer_behavior" else "clean/live scoring"
    return [f"{task.get('id', '<task>')}: {host} has no declared policy-control evidence and "
            "served-sampling contract; freeze a policy-aware task before reservation"]


@dataclass(frozen=True)
class ExtensionSpec:
    """One typed binding to a public API field, declared centrally.

    ``target`` is ``recipe`` or ``trainer``; no ignored catch-all arguments.
    Stateful/trainable additions must declare their public implementation's
    ownership, RNG and checkpoint behavior in this card.
    """
    name: str
    value_type: str
    target: str
    argument: str
    description: str
    required: bool = False
    default: object = _MISSING
    ownership: str = "public API"
    rng_policy: str = "no additional draws"
    checkpoint: str = "resolved recipe or public trainer state"
    shape: str = "scalar or public argument schema"
    producer: str = "candidate declaration"
    gradient_ownership: str = "public API field implementation"
    optimizer_binding: str = "public Recipe optimizer factories"
    initialization: str = "public API field implementation"
    supported_paths: tuple = ("public_trainer", "public_components")

    def declaration(self):
        result = {name: getattr(self, name) for name in (
            "name", "value_type", "target", "argument", "description", "required",
            "ownership", "rng_policy", "checkpoint", "shape", "producer",
            "gradient_ownership", "optimizer_binding", "initialization", "supported_paths")}
        result["public_definition"] = f"particlegan.{'Recipe' if self.target == 'recipe' else 'GANTrainer'}.{self.argument}"
        result["has_default"] = self.default is not _MISSING
        if self.default is not _MISSING:
            result["default"] = deepcopy(self.default)
        return result


class CapabilityRegistry:
    """Explicit public argument bindings shared by all task families."""
    def __init__(self):
        self.extensions = {}

    def register_extension(self, spec):
        if not isinstance(spec, ExtensionSpec):
            raise TypeError("register_extension requires an ExtensionSpec")
        if (not spec.name or spec.name in BUILTIN_CAPABILITIES or spec.name in self.extensions
                or spec.target not in ("recipe", "trainer")
                or spec.value_type not in ("bool", "int", "float", "str", "list", "dict")
                or any(not isinstance(value, str) or not value.strip()
                       for value in (spec.argument, spec.description, spec.ownership,
                                     spec.rng_policy, spec.checkpoint, spec.shape, spec.producer,
                                     spec.gradient_ownership, spec.optimizer_binding, spec.initialization))
                or not spec.supported_paths
                or set(spec.supported_paths) - {"public_trainer", "public_components"}):
            raise ValueError("invalid or duplicate extension declaration")
        arguments = {field.name for field in fields(Recipe)} if spec.target == "recipe" else \
                    set(inspect.signature(GANTrainer).parameters)
        reserved = {"recipe", "generator", "discriminator", "prior", "seed", "optimizer_options",
                    "penalty_options", "require_latent_damping"}
        if spec.argument not in arguments or (spec.target == "trainer" and (
                spec.argument in reserved or spec.argument.endswith("_generator"))):
            raise ValueError("extension must bind a supported nonreserved public argument")
        if any(old.target == spec.target and old.argument == spec.argument for old in self.extensions.values()):
            raise ValueError("a public argument already has an extension binding")
        self.extensions[spec.name] = spec
        return self

    @staticmethod
    def _validate_value(spec, value):
        expected = {"bool": bool, "int": int, "str": str, "list": list, "dict": dict}.get(spec.value_type)
        valid = (type(value) in (int, float) and math.isfinite(value)) if spec.value_type == "float" else type(value) is expected
        if not valid:
            raise CapabilityError([f"extension {spec.name} requires {spec.value_type}"])

    def resolve(self, supplied):
        if not isinstance(supplied, dict):
            raise CapabilityError(["extensions must be a declared mapping"])
        unknown = set(supplied) - set(self.extensions)
        if unknown:
            raise CapabilityError([f"unsupported extension {name}" for name in sorted(unknown)])
        values, bindings = {}, {"recipe": {}, "trainer": {}}
        for name, spec in self.extensions.items():
            value = supplied.get(name, spec.default)
            if value is _MISSING:
                if spec.required:
                    raise CapabilityError([f"missing required extension {name}"])
                continue
            self._validate_value(spec, value)
            values[name] = deepcopy(value)
            bindings[spec.target][spec.argument] = deepcopy(value)
        return values, bindings


def default_registry():
    """The single catalog for public extension bindings shared by every host.

    Add a supported ExtensionSpec here after implementing its public API field;
    ordinary Recipe fields require no extension declaration.
    """
    return CapabilityRegistry()


def _resolved_prior(value):
    try:
        return resolve_prior(value)
    except ValueError as error:
        raise CapabilityError([str(error)]) from error


class FormulationContext:
    """Resolve one formulation once, then build public trainers across tasks.

    Recipe-only variables stay in ``recipe_overrides``. Prior width is explicit
    and does not run historical nearest-neighbor calibration. ``extensions``
    are validated bindings registered once through ``CapabilityRegistry``.
    This is the frozen Forge-v1 protocol: an omitted preset means explicit
    historical KA2. Declare ``recipe_preset`` for a current public recipe.
    """
    def __init__(self, *, recipe_preset=None, recipe_overrides=None, prior=None, seed=0, device="cpu",
                 requires_capabilities=(), registry=None, extensions=None,
                 initializer="deterministic_orthogonal", rng_version=RNG_VERSION,
                 execution_path="public_trainer", host_initialization=None,
                 initializer_requirements=None, policy_task=None, component_transport=None,
                 prior_contract=None, prior_mechanism=None):
        from .priors import RECIPE_PRIOR_CONTRACT
        if prior_contract not in (None, RECIPE_PRIOR_CONTRACT):
            raise CapabilityError(["unsupported prior_contract"])
        if prior_contract == RECIPE_PRIOR_CONTRACT and "learnable" in (prior or {}):
            raise CapabilityError(["recipe-owned initial prior must omit learnable"])
        self.prior_contract = prior_contract
        if execution_path not in ("public_trainer", "public_components"):
            raise CapabilityError(["unsupported public execution path"])
        self.execution_path = execution_path
        if component_transport not in (None, 'output_marginal_v1'):
            raise CapabilityError(['unsupported component transport consumer'])
        self.component_transport = component_transport
        self.policy_task = deepcopy(policy_task)
        if self.policy_task is not None:
            from .tier1_policy import validate
            validate(self.policy_task)
        self.registry = registry if registry is not None else default_registry()
        self.extension_values, self.bindings = self.registry.resolve(extensions or {})
        incompatible = [name for name in self.extension_values
                        if execution_path not in self.registry.extensions[name].supported_paths
                        or (execution_path == "public_components" and self.registry.extensions[name].target == "trainer")]
        if incompatible:
            raise CapabilityError([f"extension {name} is unsupported by {execution_path}" for name in incompatible])
        self.prior_config = _resolved_prior(prior)
        self.task_prior_config = deepcopy(self.prior_config)
        self.prior_mechanism = deepcopy(prior_mechanism)
        overrides = dict(recipe_overrides or {})
        duplicates = set(overrides) & set(self.bindings["recipe"])
        if duplicates:
            raise CapabilityError([f"recipe/extension conflict: {sorted(duplicates)}"])
        overrides.update(self.bindings["recipe"])

        is_encoder = overrides.get("encoder_mode", "none") != "none"
        if self.prior_mechanism is not None and not is_encoder:
            try:
                self.prior_config = resolve_prior_mechanism(self.prior_mechanism, self.task_prior_config)
            except ValueError as error:
                raise CapabilityError([str(error)]) from error
        elif is_encoder:
            # Preserve task-owned MoG prior for encoders, and deactivate birth-death/row-evidence
            overrides.update(
                particle_birth_death=False,
                row_evidence_gate=False,
                birth_death_feature_scale="none",
                birth_death_isolation=False,
                birth_death_backend="knn",
                birth_death_cells=64,
            )

        if prior_contract is None and prior is not None and "learnable" in prior:
            policy = "learned" if prior["learnable"] else "frozen"
            if "prior_update" in overrides and overrides["prior_update"] != policy:
                raise CapabilityError(["Recipe.prior_update conflicts with legacy task-owned learnable"])
            overrides["prior_update"] = policy
        kind = "mog" if self.prior_config["kind"] == "mog" else "particles"
        prior_fields = {"prior_kind": kind, "sigma_rel": 0., "standardize": self.prior_config["standardize"]}
        for key, value in prior_fields.items():
            if key in overrides and overrides[key] != value:
                raise CapabilityError([f"recipe {key} conflicts with explicit prior declaration"])
        self.recipe_preset = recipe_preset
        self.recipe = resolve_public_recipe({"recipe_preset": recipe_preset,
                                             "recipe_overrides": overrides}, **prior_fields)
        self.prior_config["learnable"] = self.recipe.prior_update == "learned"
        if execution_path == "public_components" and (self.recipe.kinetic_transport_weight or self.recipe.kinetic_transport_local_weight) and component_transport is None:
            raise CapabilityError(["public_components must explicitly consume Recipe kinetic transport losses; "
                                   "the current Forge host bindings do not support it"])
        if self.recipe.row_policy != "independent":
            raise CapabilityError(["Forge has no RoutedRows host binding; declare a separate routed task"])
        if self.prior_config["kind"] != "particle_cloud" and (
                self.recipe.row_evidence_gate or self.recipe.particle_birth_death):
            raise CapabilityError(["independent policy row controls require an explicit particle_cloud cohort; MoG is unsupported"])
        if (execution_path == "public_components" and policy_controls(self.recipe)
                and self.policy_task is None and self.prior_mechanism is None and not is_encoder):
            raise CapabilityError(["public_components hosts do not bind the ordered UpdatePolicy lifecycle"])
        if initializer not in ("deterministic_orthogonal", "supplied"):
            raise CapabilityError(["unsupported initializer"])
        self.initializer = initializer
        from .nativeprofiles import resolve_host_initialization
        try:
            self.host_initialization = resolve_host_initialization(host_initialization,
                initializer=initializer, prior=self.prior_config, requirements=initializer_requirements)
        except ValueError as error:
            raise CapabilityError([str(error)]) from error
        self._host_initialized_models = {}
        self._host_prior = None
        self.device = torch.device(device)
        self.streams = NamedStreams(seed, version=rng_version, device=self.device)
        # Declare the fixed purpose names before any task executes. Unused
        # streams are deliberately left untouched (including sigma=0 noise).
        for family, component, purpose in (
                ("init", "prior", "locations"), ("data", "target", "training"),
                ("prior", "latent", "indices"), ("noise", "penalty", "training"),
                ("noise", "generator", "output"), ("noise", "critic", "input"),
                ("noise", "models", "stochastic_layers"), ("noise", "prior", "gaussian"),
                ("eval", "sampler", "samples"), ("eval", "target", "reference")):
            self.streams.generator(family, component=component, purpose=purpose)
        for component in ("generator", "discriminator"):
            self.streams.generator("init", component=component, purpose="construction", device="cpu")
        self.requires_capabilities = tuple(requires_capabilities)
        self.initialization = {}
        self._trainer = None
        self._policy = None
        self._policy_max_steps = None
        self._check_capabilities()

    def capabilities(self):
        scalar = self.recipe.model == "gan" and self.recipe.conditioning == "scalar" and self.recipe.encoder_mode == "none"
        p = self.prior_config
        a2 = (p["learnable"] and not p["standardize"] and self.recipe.latent_damping_max_rate > 0
              and (self.recipe.prior_betas or self.recipe.betas)[0] == 0)
        has_gaussian_kernel = p["kind"] == "mog" or prior_noise_sigma(p) > 0.0
        live_ok = (not (self.recipe.serve_average or self.recipe.continuous_policy)
                   or (self.prior_mechanism is not None and self.execution_path in ("public_trainer", "public_components")))
        return {"public_trainer": scalar, "public_components": True, "scalar_gan": scalar,
                "mog_prior": p["kind"] == "mog" or (self.prior_mechanism is not None and has_gaussian_kernel),
                "particle_cloud": p["kind"] == "particle_cloud",
                "learned_locations": p["learnable"], "uniform_masses": True, "fixed_prior_width": True,
                "a2": a2, "checkpoint": True, "named_rng": True,
                "live_sampling": live_ok,
                "policy_controls": policy_controls(self.recipe),
                "policy_serving": self.recipe.serve_average > 0,
                "served_sampling": bool(self.policy_task) and policy_controls(self.recipe),
                **{name: True for name in self.extension_values}}

    def _check_capabilities(self):
        available = self.capabilities()
        required = set(self.requires_capabilities) | {self.execution_path}
        if self.prior_config["learnable"] and self.recipe.latent_damping_max_rate > 0:
            required.add("a2")
        blockers = [f"required capability unavailable: {name}" for name in sorted(required) if not available.get(name, False)]
        if blockers:
            raise CapabilityError(blockers)

    def construct(self, factory, *, component):
        """Construct a network without perturbing data/prior/global streams."""
        with self.streams.fork("init", component=component, purpose="construction", device="cpu"):
            return factory()

    def initialize(self, model, *, component):
        from .state import state_digest
        policies = (self.host_initialization or {}).get("components", {})
        if component in policies:
            if component in self._host_initialized_models:
                if self._host_initialized_models[component] is not model:
                    raise CapabilityError([f"pinned {component} initialization already belongs to another model"])
                return model
            return self._initialize_host_component(model, component)
        if self.initializer == "supplied":
            self.initialization[component] = {
                "initializer": "supplied", "initial_state_sha256": state_digest(model.state_dict())}
            return model
        target = unwrap_prior(model) if component == "prior" else model
        seeds = {name: self.streams.seed_for("init", component=component, purpose=name)
                 for name, p in target.named_parameters() if p.requires_grad and p.numel()}
        init.deterministic_orthogonal_(target, parameter_seeds=seeds)
        self.initialization[component] = {"initializer": "deterministic_orthogonal_named_parameters_v1",
                                         "parameter_seeds": seeds,
                                         "initial_state_sha256": state_digest(model.state_dict())}
        return model

    def _host_init_options(self, model, component, *, probe=False):
        """Bind policy descriptors and named streams to the additive public API."""
        target = unwrap_prior(model) if component == "prior" else model
        policy = self.host_initialization["components"][component]
        options = {"method": policy["method"]}
        parameters = dict(target.named_parameters())
        if policy["method"] == "xavier_uniform_zero_bias_v1":
            options["gain"] = policy["gain"]
            names = [name for name, value in parameters.items() if value.requires_grad and name.split(".")[-1] == "weight"]
        elif policy["method"] == "sample_distributions_v1":
            descriptors = {}
            for name, value in policy["parameters"].items():
                kind = value["kind"]
                descriptors[name] = (init.Uniform(value["low"], value["high"]) if kind == "uniform" else
                    init.Normal(value["mean"], value["std"]) if kind == "normal" else init.KEEP)
            options["distributions"] = descriptors
            resolved = {**init.declarations(target), **descriptors}
            names = [name for name, value in parameters.items() if value.requires_grad and
                     isinstance(resolved.get(name), (init.Uniform, init.Normal))]
        else:
            names = []
        if probe:
            # Validate every actual model before mutating any component. These
            # scratch generators never register/advance a live named stream.
            generators = {name: torch.Generator(device="cpu").manual_seed(
                self.streams.seed_for("init", component=component, purpose=name)) for name in names}
        else:
            generators = {name: self.streams.generator("init", component=component, purpose=name, device="cpu") for name in names}
        options["parameter_generators"] = generators
        return options

    def _initialize_host_component(self, model, component):
        from .state import state_digest
        target = unwrap_prior(model) if component == "prior" else model
        # Public validation/staging protects the component against partial
        # writes; cloned streams protect the context against registration on a
        # rejected component. G/D are also validated together in build_trainer.
        init.initialize_(deepcopy(target), **self._host_init_options(target, component, probe=True))
        options = self._host_init_options(target, component)
        generators = options["parameter_generators"]
        digest = lambda stream: hashlib.sha256(stream.get_state().numpy().tobytes()).hexdigest()
        before = {name: digest(stream) for name, stream in generators.items()}
        init.initialize_(target, **options)
        policy = self.host_initialization["components"][component]
        manifest = {}
        for name, parameter in target.named_parameters():
            row = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                   "requires_grad": parameter.requires_grad, "tensor_sha256": state_digest(parameter),
                   "policy": deepcopy(policy if parameter.requires_grad else {"method": "retain_frozen"})}
            if name in generators:
                row["random_stream"] = {"family": "init", "component": component, "purpose": name,
                    "seed": self.streams.seed_for("init", component=component, purpose=name), "device": "cpu",
                    "initial_state_sha256": before[name], "final_state_sha256": digest(generators[name])}
            manifest[name] = row
        self.initialization[component] = {"initializer": policy["method"], "owner": "task",
            "fallback_initializer": self.initializer, "effective_policy": deepcopy(policy),
            "profile_sha256": self.host_initialization["profile_sha256"],
            "model_class": type(model).__module__ + "." + type(model).__qualname__, "parameters": manifest,
            "buffers": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "tensor_sha256": state_digest(value)}
                        for name, value in target.named_buffers()}}
        if component == "prior":
            self.initialization[component]["constructor_locations"] = "implicit factory init_std creates temporary locations; public policy replaces them once"
        self._host_initialized_models[component] = model
        return model

    def _construct_prior(self, *, dtype, probe=False):
        """Build through the same public factory, optionally using a scratch RNG."""
        p = self.prior_config
        stream = self.streams.generator("init", component="prior", purpose="locations")
        if probe:
            stream = torch.Generator(device=stream.device).set_state(stream.get_state())
        options = {"device": self.device, "dtype": dtype,
                   "generator": stream, "init_std": p.get("init_std", 1.)}
        noise_sigma = prior_noise_sigma(p)
        if p["kind"] == "mog":
            options["sigma"] = p["sigma"]
        elif noise_sigma > 0.0:
            options["sigma"] = noise_sigma
        # Initialize identical trainable locations, then bind recipe freezing.
        return self.recipe.replace(prior_update="learned").make_prior(**options)

    def build_prior(self, *, dtype=torch.float32):
        if self.host_initialization and self._host_prior is not None:
            if self._host_prior.z.dtype != dtype:
                raise CapabilityError(["pinned prior was already constructed with a different dtype"])
            return self._host_prior
        prior = self._construct_prior(dtype=dtype)
        self.initialize(prior, component="prior")
        self.recipe.apply_prior_policy(prior)
        if self.host_initialization:
            self._host_prior = prior
        return prior

    def bind_update_policy(self, policy, *, external_max_steps):
        """Bind a caller-owned public lifecycle for the scoped direct-table host."""
        from particlegan import UpdatePolicy
        if self.policy_task is None or not isinstance(policy, UpdatePolicy):
            raise CapabilityError(["a declared task and actual public UpdatePolicy are required"])
        if self._trainer is not None or self._policy is not None:
            raise ValueError("context already owns a public lifecycle")
        if (policy.recipe.to_dict() != self.recipe.to_dict() or
                external_max_steps != self.policy_task["execution"]["steps"] or
                policy.table.shape != (self.recipe.num_particles, self.recipe.z_dim) or
                policy.row_policy != "independent" or policy.prior.z is not policy.table):
            raise CapabilityError(["policy recipe/table/external budget differs from its task"])
        self._policy, self._policy_max_steps = policy, external_max_steps

    def build_trainer(self, generator, discriminator, *, initialize=True, max_steps=None):
        """Build the shared public update path; no task-private training loop."""
        self._check_capabilities()
        if self.execution_path != "public_trainer":
            raise CapabilityError(["public_components context does not own a scalar GANTrainer"])
        if self._trainer is not None:
            raise ValueError("a context owns one trainer; create a new context for another task/attempt")
        if self.host_initialization:
            if not initialize:
                raise CapabilityError(["initialize=False cannot bypass task-pinned component initialization"])
            if any(next(model.parameters()).device != self.device for model in (generator, discriminator)):
                raise CapabilityError(["context and network devices differ"])
            for component, model in (("generator", generator), ("discriminator", discriminator)):
                if component in self.host_initialization["components"]:
                    init.initialize_(deepcopy(model), **self._host_init_options(model, component, probe=True))
                else:
                    seeds = {name: self.streams.seed_for("init", component=component, purpose=name)
                             for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.numel()}
                    init.deterministic_orthogonal_(deepcopy(model), parameter_seeds=seeds)
            dtype = next(generator.parameters()).dtype
            if self._host_prior is not None:
                if self._host_prior.z.dtype != dtype:
                    raise CapabilityError(["pinned prior was already constructed with a different dtype"])
            else:
                # Prior shape/dtype checks can reject a syntactically valid
                # policy. Perform them before either caller network changes.
                # The temporary factory and initializer use scratch generators;
                # successful construction still consumes the original one draw.
                probe_prior = self._construct_prior(dtype=dtype, probe=True)
                probe_target = unwrap_prior(probe_prior)
                if "prior" in self.host_initialization["components"]:
                    init.initialize_(probe_target, **self._host_init_options(probe_target, "prior", probe=True))
                else:
                    seeds = {name: self.streams.seed_for("init", component="prior", purpose=name)
                             for name, parameter in probe_target.named_parameters() if parameter.requires_grad and parameter.numel()}
                    init.deterministic_orthogonal_(probe_target, parameter_seeds=seeds)
        if initialize:
            self.initialize(generator, component="generator")
            self.initialize(discriminator, component="discriminator")
        first = next(generator.parameters())
        if first.device != self.device:
            raise CapabilityError(["context and network devices differ"])
        prior = self.build_prior(dtype=first.dtype)
        use_prior_noise = prior_uses_noise_stream(self.prior_config)
        options = {
            **{name: self.streams.generator(family, component=component, purpose=purpose)
               for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()
               if name != "prior_noise_generator" or use_prior_noise},
            "require_latent_damping": self.recipe.latent_damping_max_rate > 0,
            **self.bindings["trainer"],
        }
        if max_steps is not None:
            if "max_steps" in options:
                raise CapabilityError(["execution budget conflicts with trainer extension"])
            options["max_steps"] = max_steps
        streams = [options[name] for name in TRAINER_STREAM_BINDINGS if name in options]
        if len({id(stream) for stream in streams}) != len(streams):
            raise CapabilityError(["Forge requires distinct named trainer streams"])
        self._trainer = GANTrainer(self.recipe, generator, discriminator, prior=prior,
                                   seed=self.streams.seed, **options)
        return self._trainer

    def receipt(self):
        ownership = {}
        if getattr(self, "ownership_contract", None) is not None:
            from .boundaries import ownership_receipt
            ownership["field_ownership"] = ownership_receipt(**self.ownership_contract,
                resolved_recipe=asdict(self.recipe), initializer=self.initializer)
        policy = None
        if (self._trainer is not None or self._policy is not None) and policy_controls(self.recipe):
            owner = self._trainer if self._trainer is not None else self._policy
            birth = owner.birth_death
            policy = {"owner": "particlegan.UpdatePolicy", "completed_steps": owner.completed_steps,
                      "external_max_steps": self._trainer.max_steps if self._trainer is not None else self._policy_max_steps,
                      "continuous_policy": self.recipe.continuous_policy,
                      "lr_control": self.recipe.lr_control, "row_policy": self.recipe.row_policy,
                      "serving": "state_selected" if self.recipe.serve_average else "fast",
                      "quality_qualification": False,
                      "private_rng": [] if birth is None else [{
                          "owner": "UpdatePolicy.birth_death", "seed": self.streams.seed + 6,
                          "derivation": "public policy seed + 6; independent of Forge named streams",
                          "state_sha256": hashlib.sha256(birth.stream.get_state().cpu().numpy().tobytes()).hexdigest(),
                          "checkpoint_path": "trainer.birth_death.stream"}]}
        return {"api_version": API_VERSION, "execution_path": self.execution_path,
                **({"prior_contract": self.prior_contract,
                    "initial_prior": {key: value for key, value in self.prior_config.items() if key != "learnable"}}
                   if self.prior_contract is not None else {}),
                **ownership,
                **({"host_adaptation": self.host_adaptation} if getattr(self, "host_adaptation", None) else {}),
                "recipe_preset": self.recipe_preset,
                "recipe": self.recipe.to_dict(), "prior": deepcopy(self.prior_config),
                "capabilities": self.capabilities(), "requires_capabilities": list(self.requires_capabilities),
                "extensions": deepcopy(self.extension_values),
                "api_changes": [self.registry.extensions[name].declaration() for name in sorted(self.extension_values)],
                "initializer": self.initializer, "initialization": deepcopy(self.initialization),
                "rng": self.streams.manifest(), "policy_lifecycle": policy,
                "prior_mechanisms": None if self._trainer is None else deepcopy(self._trainer.prior_mechanisms)}

    def state_dict(self):
        if self._trainer is None:
            raise ValueError("build a trainer before checkpointing")
        return {"schema": 1, "api_version": API_VERSION, "recipe": self.recipe.to_dict(),
                **({"prior_contract": self.prior_contract} if self.prior_contract is not None else {}),
                "prior": deepcopy(self.prior_config), "extensions": deepcopy(self.extension_values),
                "initializer": self.initializer, "initialization": deepcopy(self.initialization),
                "streams": self.streams.state_dict(), "trainer": self._trainer.state_dict()}

    def load_state_dict(self, state):
        if self._trainer is None:
            raise ValueError("build a trainer before restoring")
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid context checkpoint schema")
        for key in ("schema", "api_version", "recipe", "prior", "extensions", "initializer", "initialization"):
            if state[key] != expected[key]:
                raise ValueError(f"context checkpoint {key} mismatch")
        if state.get("prior_contract") != expected.get("prior_contract"):
            raise ValueError("context checkpoint prior_contract mismatch")
        self.streams.validate_state_dict(state["streams"])
        # A trainer and named registry refer to the same stream objects. Reject
        # conflicting duplicated states before either loader mutates anything.
        for name, value in state["trainer"].get("streams", {}).items():
            stream = getattr(self._trainer, name, None)
            key = next((key for key, registered in self.streams._streams.items() if registered is stream), None)
            if key is None or key not in state["streams"]["states"] or not torch.equal(value, state["streams"]["states"][key]):
                raise ValueError("trainer/named RNG checkpoint mismatch")
        self._trainer.load_state_dict(state["trainer"])
        self.streams.load_state_dict(state["streams"])
