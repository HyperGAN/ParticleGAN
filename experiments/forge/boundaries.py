"""Ownership and provenance for an already resolved Forge task formulation.

This module describes the binding performed by the public adapters. It never
resolves a second Recipe, imports an adapter, or treats historical host training
constants as effective candidate settings.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields

from particlegan import Recipe

from .taskrecipes import BEHAVIOR_HOST_FIELDS, RESOURCE_FIELDS, delegated_fields
from .priors import recipe_owned_prior


BOUNDARIES_VERSION = "forge-field-boundaries-v2"
OWNERS = frozenset({"task", "technique", "hyperparameter", "protocol"})

# Conservative finite search permission is narrower than numerical ownership.
# In particular, a numerical ablation switch still changes a technique when its
# value removes an active mechanism; techniques.py checks that invariant.
TUNABLE_FIELDS = frozenset({
    "lr", "d_lr_mult", "prior_lr_mult", "betas", "prior_betas", "d_betas", "d_eps", "prior_eps",
    "direct_particle_betas", "eps", "amsgrad", "lr_decay_rate", "lr_decay_steps",
    "optimizer_momentum", "optimizer_adam_lr", "optimizer_smoothing",
    "reg_coeff", "reg_coeff_end", "reg_coeff_anneal_end", "reg_kappa", "reg_every", "ebi_beta", "prior_reg",
    "prior_reg_target_std", "prior_reg_eps", "prior_l2", "prior_proximal_strength",
    "prior_coulomb_softening", "prior_coulomb_confinement", "prior_coulomb_neighbors",
    "kinetic_transport_weight", "kinetic_transport_local_weight",
    "rhbi_weight", "rhbi_rho", "rhbi_tau", "rhbi_eps", "rhbi_c_max", "rhbi_beta_ell", "rhbi_nn_quantile", "rhbi_critic_weight",
    "rhbi_relativistic_weight", "rhbi_lambda_floor", "rhbi_gamma_gauge", "rhbi_slack_gain", "rhbi_chord_weight", "rhbi_thermal_tau",
    "rhbi_thermal_shock_mu", "rhbi_settle_beta", "rhbi_fibre_rel_blend",
    "rhbi_step_damp", "rhbi_cond_ot_weight", "rhbi_simplex_weight",
    "rhbi_cvx_mu_gain", "rhbi_simplex_chart_weight", "rhbi_tail_spill_gain",
    "lr_anneal_start", "lr_floor", "network_lr_floor", "beta2_end", "beta2_anneal_end",
})
TASK_RECIPE_FIELDS = RESOURCE_FIELDS | {"prior_kind", "sigma_rel", "standardize"}
TECHNIQUE_RECIPE_FIELDS = frozenset({
    "training_precision",
    "masked_mixture_weight", "masked_transport_weight", "imputer_component_weight",
    "optimizer_svd_backend", "optimizer_convolution_rule", "optimizer_dense_profile",
    "prior_update", "prior_regularizer",
    "reg_real_weight", "reg_fake_weight", "joint_reconstruction_weight",
    "name", "critic_formulation", "model", "loss", "num_classes", "conditioning", "ucd_target", "encoder_mode",
    "distance_reduction", "continuous_policy", "reg_arm", "direct_particle_gain",
    "critic_r1_real", "critic_payoff_damping", "output_noise_mode", "lr_control",
    "particle_birth_death", "row_evidence_gate", "table_release_rule", "row_evidence_hot",
    "row_evidence_exclude", "row_evidence_hold", "birth_death_space", "reopen_signal",
    "reopen_anchor", "reopen_guard", "row_evidence_null", "birth_death_isolation",
    "birth_death_feature_scale", "birth_death_backend", "birth_death_parent_policy",
    "constraint_geometry_mode", "row_policy", "optimizer_family", "optimizer_convolution", "loss_labels", "adam_variant", "lr_schedule", "lr_decay_staircase",
    "critic_step_mode", "bispf_shape_gates", "bispf_gate_overrides", "prior_code_law", "generator_gradient_response",
    "rhbi_self_coupling", "rhbi_rapidity_screening", "rhbi_scale_log_odds", "rhbi_conditional_fibre", "rhbi_akt_horizon", "rhbi_akt_mode", "rhbi_settle_mode", "rhbi_cond_ot_mode", "rhbi_simplex_macro_settle",
})
HYPERPARAMETER_RECIPE_FIELDS = TUNABLE_FIELDS | {
    "conditional_energy_weight", "conditional_energy_samples",
    "prior_sigreg_projections", "prior_sigreg_knots",
    "kinetic_transport_projections", "rhbi_iters",
    "ucd_weight", "alpha_bar", "ema_decay", "network_lr_horizon_cap",
    "reg_anchor_min_decay", "reg_anchor_weight", "d_guard_ratio", "d_guard_min_steps",
    "latent_damping_max_rate", "direct_particle_betas", "input_noise_std",
    "input_noise_anneal_end", "output_noise_std", "output_noise_warmup",
    "routing_temperature", "observation_sigma", "reconstruction_weight", "serve_average",
    "birth_death_cells", "birth_death_metric_rank", "birth_death_chunk",
    "kappa_radial",
}
RECIPE_FIELD_OWNERS = {
    **dict.fromkeys(TASK_RECIPE_FIELDS, "task"),
    **dict.fromkeys(TECHNIQUE_RECIPE_FIELDS, "technique"),
    **dict.fromkeys(HYPERPARAMETER_RECIPE_FIELDS, "hyperparameter"),
}


@dataclass(frozen=True)
class RecipeFieldSpec:
    name: str
    owner: str
    tunable: bool
    default: object


_DEFAULT_RECIPE = Recipe.from_dict({})
RECIPE_FIELD_SPECS: dict[str, RecipeFieldSpec] = {
    field.name: RecipeFieldSpec(
        name=field.name,
        owner=RECIPE_FIELD_OWNERS[field.name],
        tunable=field.name in TUNABLE_FIELDS,
        default=getattr(_DEFAULT_RECIPE, field.name),
    )
    for field in fields(Recipe)
    if field.name in RECIPE_FIELD_OWNERS
}


def validate_registry():
    """Adding a public field requires an explicit Forge ownership decision."""
    groups = (TASK_RECIPE_FIELDS, TECHNIQUE_RECIPE_FIELDS, HYPERPARAMETER_RECIPE_FIELDS)
    if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
        raise ValueError("Recipe field ownership groups overlap")
    public = {field.name for field in fields(Recipe)}
    missing, unknown = public - RECIPE_FIELD_OWNERS.keys(), RECIPE_FIELD_OWNERS.keys() - public
    if missing or unknown:
        raise ValueError(f"Recipe ownership registry differs from public API: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if not TUNABLE_FIELDS <= HYPERPARAMETER_RECIPE_FIELDS:
        raise ValueError("search fields must be declared hyperparameters")
    if set(RECIPE_FIELD_SPECS) != public:
        raise ValueError("RecipeFieldSpec registry does not match public Recipe fields")


validate_registry()


def _behavior_host(task):
    execution = task.get("execution", {})
    return (execution.get("host", task.get("id"))
            if task.get("adapter") == "transfer_behavior" else None)


def prior_control_binding(task):
    """Separate direct generated coordinates from a sampled latent prior."""
    from .tier1_policy import is_policy_task
    if _behavior_host(task) == "two_pole" and is_policy_task(task):
        return {"representation": "policy_owned_direct_sample_coordinates",
                "latent_table_controls": True, "construction": "public ParticlePrior with the original zero initialization",
                "optimizer": "Recipe.make_generator_optimizer(latent_table=..., direct_particles=...)",
                "base_lr": "Recipe.lr", "base_betas": "Recipe.betas",
                "lr_schedule": "public_UpdatePolicy_stationarity",
                "note": "Explicit selected-policy variant binds a policy table and preserves direct-coordinate base LR/betas; original nonpolicy direct-parameter evidence grants no credit."}
    if _behavior_host(task) == "two_pole":
        return {"representation": "direct_sample_coordinates", "latent_table_controls": False,
                "construction": "direct nn.Parameter; no ParticlePrior or latent-table optimizer",
                "optimizer": "Recipe.make_generator_optimizer(direct_particles=...)",
                "base_lr": "Recipe.lr", "base_betas": "Recipe.betas",
                "lr_schedule": "prior", "note": "The existing direct-coordinate group uses the prior "
                "decay schedule, but prior_lr_mult and prior_betas do not bind it. Formulation "
                "optimizers retain their public DirectParticleResponse step controls."}
    if task.get("execution", {}).get("prior_applicability") == "not_sampled":
        return {"representation": "not_sampled", "latent_table_controls": False,
                "construction": "no sampled latent prior or prior optimizer group"}
    return {"representation": "latent_prior_locations", "latent_table_controls": True,
            "base_lr": "Recipe.lr * Recipe.prior_lr_mult",
            "base_betas": "Recipe.prior_betas or Recipe.betas", "lr_schedule": "prior"}


def task_owned_recipe_fields(task):
    """Fields whose effective binding belongs to this frozen task contract."""
    owned = TASK_RECIPE_FIELDS
    host = _behavior_host(task)
    if host is not None and host != "mode_hold":
        owned |= BEHAVIOR_HOST_FIELDS
        if host != "ae_gan_hold":
            owned |= {"routing_temperature", "distance_reduction"}
    if task.get("adapter") == "public_api_model_toy":
        owned |= frozenset(task["execution"].get("recipe_fields", {}))
    if recipe_owned_prior(task):
        owned = owned - {"prior_reg"}
    return frozenset(owned)


def recipe_field_owner(name, task=None):
    if name not in RECIPE_FIELD_OWNERS:
        raise ValueError(f"unknown public Recipe field: {name}")
    if task is not None and name in task_owned_recipe_fields(task):
        return "task"
    return RECIPE_FIELD_OWNERS[name]


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return deepcopy(value)


def _record(value, owner, source, **extra):
    return {"value": _json_value(value), "owner": owner, "source": source, **extra}


def _resources(task):
    """Read resource declarations; host factories remain responsible for binding."""
    execution = task.get("execution", {})
    host = execution.get("host_definition", {})
    result = {}
    for name in sorted(RESOURCE_FIELDS - {"total_steps"}):
        for mapping, prefix, key in (
                (execution.get("resources", {}), "task.execution.resources", name),
                (host.get("resources", {}), "task.execution.host_definition.resources", name),
                (host, "task.execution.host_definition", name),
                (host, "task.execution.host_definition", {"num_particles": "particles", "batch_size": "batch", "z_dim": "z_dim"}[name])):
            if key in mapping:
                result[name] = (mapping[key], f"{prefix}.{key}")
                break
    return result


def _reference_source(candidate, name, value, extension_recipe_bindings):
    if name in extension_recipe_bindings:
        origin = f"registered candidate.extensions binding for Recipe.{name}"
        if _json_value(extension_recipe_bindings[name]) != _json_value(value):
            return f"public Recipe normalization of {origin}"
        return origin
    if name in candidate.get("recipe_overrides", {}):
        if _json_value(candidate["recipe_overrides"][name]) != _json_value(value):
            return f"public Recipe normalization of candidate.recipe_overrides.{name}"
        return f"candidate.recipe_overrides.{name}"
    if name == "critic_formulation" and candidate.get("recipe_overrides", {}).get("reg_arm") is not None:
        return "public Recipe normalization of candidate.recipe_overrides.reg_arm"
    if name == "critic_formulation" and extension_recipe_bindings.get("reg_arm") is not None:
        return "public Recipe normalization of registered candidate.extensions binding for Recipe.reg_arm"
    if candidate.get("recipe_preset") is not None:
        return f"public Recipe preset {candidate['recipe_preset']}"
    return "public Recipe default"


def ownership_receipt(candidate, task, resolved_recipe, protocol=None, initializer=None,
                      *, extension_recipe_bindings=None):
    """Describe effective values supplied by the actual public task resolver.

    Behavioral hosts own their resource/objective code. Recipe defaults for
    those fields do not describe its trained components, so the receipt records
    null and keeps the inactive reference value explicitly labelled. The AE
    encoder's resource fields are exceptions: its public encoder recipe binds
    those fields directly. No models are built and no RNG draws are made here.
    """
    validate_registry()
    if not isinstance(resolved_recipe, dict) or set(resolved_recipe) != set(RECIPE_FIELD_OWNERS):
        raise ValueError("ownership receipt requires the complete effective public Recipe")
    extension_recipe_bindings = extension_recipe_bindings or {}
    if set(extension_recipe_bindings) - RECIPE_FIELD_OWNERS.keys():
        raise ValueError("extension ownership receipt contains unknown public Recipe fields")
    execution = task.get("execution", {})
    prior = execution.get("prior")
    if not isinstance(prior, dict) or prior.get("kind") not in {"mog", "particle_cloud"}:
        raise ValueError("ownership receipt requires the task's explicit prior")
    mechanism = (candidate or {}).get("prior_mechanism")
    if mechanism is not None and resolved_recipe.get("encoder_mode", "none") == "none":
        from .priors import prior_code_path, resolve_prior_mechanism
        effective_prior = resolve_prior_mechanism(mechanism, prior)
        expected_prior = {"prior_kind": "mog" if effective_prior["kind"] == "mog" else "particles",
                          "sigma_rel": 0., "standardize": effective_prior["standardize"]}
    else:
        effective_prior = None
        expected_prior = {"prior_kind": "mog" if prior["kind"] == "mog" else "particles",
                          "sigma_rel": 0., "standardize": prior["standardize"]}
    for name, value in expected_prior.items():
        if resolved_recipe[name] != value:
            label = "technique-owned prior_mechanism" if effective_prior is not None else "task-owned prior"
            raise ValueError(f"effective Recipe {name} contradicts {label}")
    resources = _resources(task)
    for name, (value, _) in resources.items():
        if resolved_recipe[name] != value:
            raise ValueError(f"effective Recipe {name} contradicts task-owned resource")
    horizon = execution.get("original_schedule_horizon", execution.get("steps"))
    if resolved_recipe["total_steps"] is not None and resolved_recipe["total_steps"] != horizon:
        raise ValueError("effective Recipe total_steps contradicts task-owned schedule horizon")
    host = _behavior_host(task)
    prior_binding = prior_control_binding(task)
    behavior_owned = task_owned_recipe_fields(task) - TASK_RECIPE_FIELDS
    if host is not None and host != "mode_hold":
        behavior_owned |= RESOURCE_FIELDS - {"total_steps"}
        if host == "ae_gan_hold":
            behavior_owned -= {"encoder_mode", "z_dim", "num_particles", "batch_size"}
    result = {}
    for name in sorted(resolved_recipe):
        owner = recipe_field_owner(name, task)
        value = resolved_recipe[name]
        source = _reference_source(candidate, name, value, extension_recipe_bindings)
        status = "metadata" if name == "name" else "effective"
        if name in expected_prior:
            if effective_prior is not None and name in {"prior_kind", "standardize"}:
                owner = "technique"
                source = ("candidate.prior_mechanism.kind" if name == "prior_kind"
                          else "candidate.prior_mechanism.standardize")
            else:
                source = ("task.execution.prior.kind" if name == "prior_kind" else
                          "task.execution.prior.standardize" if name == "standardize" else
                          "public prior binding: absolute task sigma replaces relative calibration")
        elif name == "total_steps":
            if value is None:
                owner, source = "technique", "schedule-free technique; task.execution.steps bounds execution separately"
            else:
                source = ("task.execution.original_schedule_horizon" if "original_schedule_horizon" in execution
                          else "task.execution.steps")
        elif name in resources:
            source = resources[name][1]
        elif host == "ae_gan_hold" and name in {"encoder_mode", "z_dim", "num_particles", "batch_size"}:
            source = "frozen behavioral host HoldConfig and public encoder recipe binding"
        elif (name in RESOURCE_FIELDS - {"total_steps"}
              and task.get("adapter") in {"native100", "native100_continuation"}
              and "native_profile" not in execution):
            owner, status = "technique", "legacy_reference"
        elif owner == "task":
            source = "frozen task host/adapter resource binding"
        prior_fields = {"prior_lr_mult", "prior_betas", "prior_eps", "prior_proximal_strength"}
        if recipe_owned_prior(task):
            prior_fields |= {"prior_update", "prior_regularizer", "prior_reg", "prior_l2",
                             "prior_reg_target_std", "prior_reg_eps", "prior_sigreg_projections", "prior_sigreg_knots",
                             "prior_coulomb_softening", "prior_coulomb_confinement", "prior_coulomb_neighbors"}
        optimizer_inactive = (name in prior_fields - {"prior_update"}
                              and resolved_recipe["prior_update"] == "frozen" and recipe_owned_prior(task))
        if (name in prior_fields and not prior_binding["latent_table_controls"]) or optimizer_inactive:
            result[name] = _record(None, owner, "task prior-control applicability",
                                   status="not_applicable", reference_recipe_value=_json_value(value),
                                   representation=prior_binding["representation"])
        elif name == "joint_reconstruction_weight" and task.get("adapter") != "word_joint":
            result[name] = _record(None, owner, "task joint-inverse reconstruction applicability",
                                   status="not_applicable", reference_recipe_value=_json_value(value))
        elif name in {"reg_real_weight", "reg_fake_weight"} and (
                resolved_recipe["reg_arm"] not in {"a_r1r2", "b_cap"}
                or (task.get("adapter") == "public_api_model_toy"
                    and execution.get("host_definition", {}).get("adversarial") is False)):
            result[name] = _record(None, owner, "task fixed critic-penalty applicability",
                                   status="not_applicable", reference_recipe_value=_json_value(value))
        elif name in behavior_owned:
            result[name] = _record(None, owner, "frozen behavioral host objective/component source",
                                   status="host_owned", reference_recipe_value=_json_value(value))
        else:
            result[name] = _record(value, owner, source, status=status)
    host_definition = execution.get("host_definition", {})
    # Legacy optimizer and penalty settings are retained provenance. Scalar
    # adapters copy architecture, data and resources, never these settings.
    inactive_names = ((set(TECHNIQUE_RECIPE_FIELDS) | set(HYPERPARAMETER_RECIPE_FIELDS))
                      - task_owned_recipe_fields(task)) | {
                          "d_every", "g_every", "loss_type", "gan_mode", "reg_norm", "reg_lazy", "target_anneal"}
    inactive = {name: _record(value, "task", f"task.execution.host_definition.{name}", status="inactive_provenance")
                for name, value in sorted(host_definition.items()) if name in inactive_names}
    active_host = {name: value for name, value in host_definition.items() if name not in inactive}
    initialized = initializer if initializer is not None else execution.get("initializer")
    if "initializer" in execution and initialized != execution["initializer"]:
        raise ValueError("effective initializer contradicts task-owned initialization")
    if "initializer" in execution:
        init_source, init_owner = "task.execution.initializer", "task"
    elif "fixed_initialization" in execution or "initialization" in host_definition:
        init_source, init_owner = "frozen task host initialization", "task"
    else:
        init_source, init_owner = "legacy candidate/API initializer fallback", "technique"
    initialization = _record(initialized, init_owner, init_source)
    for key in ("fixed_initialization", "host_initialization"):
        if key in execution:
            initialization[key] = _json_value(execution[key])
    if "initialization" in host_definition:
        initialization["component_policies"] = _json_value(host_definition["initialization"])
    prior_record = _record(prior, "task", "task.execution.prior")
    if recipe_owned_prior(task):
        prior_record["update_policy"] = _record(resolved_recipe["prior_update"], "technique", "Recipe.prior_update")
        prior_record["contract"] = execution["prior_contract"]
    if effective_prior is not None:
        prior_record["technique_mechanism"] = _record(effective_prior, "technique", "candidate.prior_mechanism")
        prior_record["code_path"] = prior_code_path(effective_prior)
    else:
        prior_record["code_path"] = "MoGParticlePrior" if prior["kind"] == "mog" else "ParticlePrior"
    if not prior_binding["latent_table_controls"]:
        prior_record["declared_code_path"] = prior_record["code_path"]
        prior_record["code_path"] = None
    prior_record["control_binding"] = prior_binding
    prior_record["applicability"] = execution.get("prior_applicability", "sampled")
    budget = {name: execution[name] for name in ("steps", "incremental_steps", "preserve_prefix_steps", "original_schedule_horizon", "horizon_diagnostic") if name in execution}
    budget["execution_resources"] = task.get("resources", {})
    receipt = {
        "schema_version": 1, "version": BOUNDARIES_VERSION, "task_id": task.get("id"),
        "recipe_fields": result,
        "task_contract": {
            "prior": prior_record,
            "initialization": initialization,
            "budget": _record(budget, "task", "task.execution and task.resources"),
            "host": _record({"definition": active_host, "source": execution.get("host_source"),
                              "model": execution.get("model"), "recipe_resources": execution.get("resources"),
                              "problem": execution.get("problem"), "native_profile": execution.get("native_profile")},
                             "task", "frozen task architecture/data declaration and host source"),
            "evaluation": _record(task.get("evaluation", {}), "task", "task.evaluation"),
        },
        "delegated_reference_values": _json_value(delegated_fields(candidate, task)),
        "inactive_legacy_host_fields": inactive,
    }
    if recipe_owned_prior(task):
        receipt["version"] = "forge-field-boundaries-v3-recipe-priors"
    if protocol is not None:
        receipt["protocol"] = {name: _record(protocol[name], "protocol", f"protocol.{name}")
                               for name in ("id", "revision", "seed", "rng", "scoring", "robustness") if name in protocol}
    # These declarations are references, never a fallback for effective task
    # priors or initialization. Protocol.prior is likewise retained as policy.
    reference = {name: candidate[name] for name in ("prior", "initializer") if name in candidate}
    if protocol is not None and "prior" in protocol:
        reference["protocol_prior"] = protocol["prior"]
    receipt["reference_declarations"] = _json_value(reference)
    return receipt
