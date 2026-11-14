"""Mechanism identity for configuration search, separate from family lineage.

Numeric tuning may change a mechanism's strength or timing, but cannot add or
remove it. These projections never replace recorded scientific identities.
"""
from dataclasses import asdict

from particlegan import Recipe

from .contracts import stable_hash


def _recipe(value):
    return value if isinstance(value, Recipe) else Recipe.from_dict(value)


def technique_signature(value):
    """Describe fixed public mechanisms and activation boundaries of knobs."""
    from .boundaries import recipe_field_owner

    recipe = _recipe(value)
    resolved = asdict(recipe)
    from particlegan.recipe_compat import without_default_additions
    resolved = without_default_additions(resolved)
    fixed = {name: item for name, item in resolved.items()
             if name != "name" and recipe_field_owner(name) == "technique"}
    prior_betas = recipe.prior_betas or recipe.betas
    network_floor = recipe.resolved_network_lr_floor
    penalty_end = recipe.reg_coeff if recipe.reg_coeff_end is None else recipe.reg_coeff_end
    beta2_end = recipe.betas[1] if recipe.beta2_end is None else recipe.beta2_end
    # Magnitude grafting consumes Adam moments; raw and normalized SGD do not.
    # Hybrid D-only retains Adam for G/E/prior, whereas row-only retains it only
    # for networks. Record the consumed rule, not inert Recipe defaults.
    adam_networks = recipe.optimizer_family in {
        "formulation", "adam", "ada_nsgda", "dualnorm_D_only", "particle_rownorm_only"}
    adam_prior = recipe.optimizer_family in {
        "formulation", "adam", "ada_nsgda", "dualnorm_D_only"}
    mechanisms = {
        "critic_penalty": {"initial": recipe.reg_coeff > 0, "terminal": penalty_end > 0},
        "penalty_schedule": {"declared": recipe.reg_coeff_end is not None,
                             "changing": penalty_end != recipe.reg_coeff},
        "beta2_schedule": {"declared": recipe.beta2_end is not None,
                           "network_changing": beta2_end != recipe.betas[1],
                           "prior_changing": recipe.beta2_end is not None and beta2_end != prior_betas[1]},
        "network_moments": [value > 0 for value in recipe.betas] if adam_networks else None,
        "prior_moments": [value > 0 for value in prior_betas] if adam_prior else None,
        "direct_particle_moments": ([value > 0 for value in recipe.direct_particle_betas]
                                    if recipe.optimizer_family == "formulation" else None),
        "terminal_second_moment": beta2_end > 0 if adam_networks or adam_prior else None,
        "amsgrad": recipe.amsgrad if adam_networks or adam_prior else None,
        "prior_regularization": recipe.prior_update == "learned" and recipe.prior_reg > 0,
        "critic_anchor": recipe.reg_anchor_weight > 0,
        "critic_guard": recipe.d_guard_ratio > 0,
        "latent_damping": recipe.latent_damping_max_rate > 0,
        "training_input_noise": recipe.input_noise_std > 0,
        "training_output_noise": recipe.output_noise_std > 0,
        "output_noise_warmup": recipe.output_noise_std > 0 and recipe.output_noise_warmup > 0,
        "ucd_objective": recipe.ucd_weight > 0 if recipe.conditioning == "ucd" else None,
        "reconstruction_objective": recipe.reconstruction_weight > 0 if recipe.encoder_mode != "none" else None,
        "averaged_serving": recipe.serve_average > 0,
        "gradient_cap": recipe.reg_kappa > 0 if recipe.reg_arm != "a_r1r2" else None,
        "scheduled_learning_rates": recipe.total_steps is not None,
        "network_horizon_cap": recipe.network_lr_horizon_cap is not None,
        "network_terminal_updates": network_floor > 0,
        "prior_terminal_updates": recipe.lr_floor > 0,
        # K3P passes floor=0 when the resolved network floor reaches .5.
        "k3p_penalty_lr_floor": (0 < network_floor < .5
                                 if recipe.effective_critic_formulation == "k3p" and recipe.reg_arm is None
                                 else None),
    }
    if recipe.conditional_energy_weight:
        mechanisms["conditional_energy_objective"] = True
    if recipe.optimizer_family in {"dualnorm", "dualnorm_D_only"}:
        mechanisms["dualnorm_momentum"] = recipe.optimizer_momentum > 0
    if recipe.prior_update == "learned" and recipe.prior_l2 > 0:
        mechanisms["prior_l2"] = True
    if recipe.prior_update == "learned" and recipe.prior_proximal_strength > 0:
        mechanisms["prior_proximal"] = "initial_position_after_base_update_v1"
    if recipe.kinetic_transport_weight > 0:
        mechanisms["kinetic_transport"] = True
    if recipe.kinetic_transport_local_weight > 0:
        mechanisms["kinetic_transport_local"] = True
    if recipe.rhbi_weight > 0:
        mechanisms["rhbi_transport"] = True
    if recipe.rhbi_critic_weight > 0:
        mechanisms["rhbi_critic_slack"] = True
    if recipe.rhbi_weight > 0 and recipe.rhbi_relativistic_weight > 0:
        mechanisms["rhbi_relativistic"] = True
    if recipe.effective_critic_formulation == "rhbi" and recipe.rhbi_lambda_floor > 0:
        mechanisms["rhbi_lambda_floor"] = True
    if recipe.effective_critic_formulation == "rhbi" and recipe.rhbi_gamma_gauge > 0:
        mechanisms["rhbi_gamma_gauge"] = True
    if recipe.effective_critic_formulation == "rhbi" and recipe.reg_coeff > 0 and recipe.rhbi_settle_beta > 0:
        mechanisms["rhbi_settle_beta"] = True
    if (
        (recipe.effective_critic_formulation == "rhbi" or recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0)
        and recipe.rhbi_conditional_fibre == "slack"
        and recipe.rhbi_fibre_rel_blend > 0
    ):
        mechanisms["rhbi_fibre_rel_blend"] = True
    if (
        (recipe.effective_critic_formulation == "rhbi" or recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0)
        and recipe.optimizer_family != "formulation"
        and recipe.optimizer_family != "adam"
        and recipe.rhbi_step_damp > 0
    ):
        mechanisms["rhbi_step_damp"] = True
    if (
        recipe.rhbi_weight > 0
        and recipe.rhbi_conditional_fibre == "slack"
        and recipe.rhbi_cond_ot_weight > 0
    ):
        mechanisms["rhbi_cond_ot"] = True
    if recipe.rhbi_weight > 0 and recipe.rhbi_simplex_weight > 0:
        mechanisms["rhbi_simplex"] = True
        if recipe.rhbi_simplex_chart_weight > 0:
            mechanisms["rhbi_simplex_chart"] = True
            if recipe.rhbi_simplex_macro_settle:
                mechanisms["rhbi_simplex_macro_settle"] = True
    if recipe.rhbi_weight > 0 and recipe.rhbi_thermal_tau > 0 and recipe.rhbi_cvx_mu_gain != 0.40:
        mechanisms["rhbi_cvx_mu"] = True
    if recipe.rhbi_weight > 0 and recipe.rhbi_thermal_tau > 0 and recipe.rhbi_tail_spill_gain > 0:
        mechanisms["rhbi_tail_spill"] = True
    if recipe.optimizer_smoothing:
        mechanisms["smoothed_dualnorm"] = True
    if recipe.optimizer_family in {"dualnorm_D_only", "particle_rownorm_only"}:
        mechanisms["hybrid_adam_rate_override"] = recipe.optimizer_adam_lr is not None
    critic_betas = recipe.d_betas or recipe.betas
    adam_critic = recipe.optimizer_family in {"formulation", "adam", "ada_nsgda", "particle_rownorm_only"}
    if adam_critic and [b > 0 for b in critic_betas] != mechanisms["network_moments"]:
        mechanisms["critic_moments"] = [b > 0 for b in critic_betas]
    if recipe.beta2_end is not None:
        critic_changing = recipe.beta2_end != critic_betas[1]
        if critic_changing != mechanisms["beta2_schedule"]["network_changing"]:
            mechanisms["critic_beta2_changing"] = critic_changing
    if recipe.lr_schedule == "exponential":
        mechanisms["exponential_decay"] = recipe.lr_decay_rate < 1
    contract = {"schema_version": 1, "fixed_recipe_fields": fixed, "mechanisms": mechanisms}
    return {**contract, "digest": stable_hash(contract)}


def validate_same_technique(base, trial):
    """Reject structural changes hidden in otherwise searchable Recipe knobs."""
    before, after = technique_signature(base), technique_signature(trial)
    changed = [name for name in before["mechanisms"].keys() | after["mechanisms"].keys()
               if before["mechanisms"].get(name) != after["mechanisms"].get(name)]
    changed += [f"Recipe.{name}" for name in before["fixed_recipe_fields"].keys() | after["fixed_recipe_fields"].keys()
                if before["fixed_recipe_fields"].get(name) != after["fixed_recipe_fields"].get(name)]
    if changed:
        raise ValueError("search changes technique mechanisms: " + ", ".join(changed)
                         + "; declare a structural idea instead of a hyperparameter trial")
    return after


def recipe_field_active(name, value, *, task=None):
    """Conservative declared activity, without constructing or sampling models."""
    from .boundaries import prior_control_binding

    recipe = _recipe(value)
    execution = {} if task is None else task.get("execution", {})
    if name == "joint_reconstruction_weight":
        return task is None or task.get("adapter") == "word_joint"
    if name in {"reg_real_weight", "reg_fake_weight"}:
        return recipe.reg_arm in {"a_r1r2", "b_cap"} and not (
            task is not None and task.get("adapter") == "public_api_model_toy"
            and execution.get("host_definition", {}).get("adversarial") is False)
    from .priors import recipe_owned_prior
    learned = (recipe.prior_update == "learned" if task is None or recipe_owned_prior(task)
               else execution.get("prior", {}).get("learnable", True))
    prior_active = ((task is None or prior_control_binding(task)["latent_table_controls"])
                    and learned)
    if name in {"prior_update", "prior_regularizer"} and task is not None and not prior_control_binding(task)["latent_table_controls"]:
        return False
    if name == "prior_regularizer" and not learned:
        return False
    if name == "optimizer_momentum":
        return recipe.optimizer_family in {"dualnorm", "dualnorm_D_only"}
    if name == "optimizer_smoothing":
        return recipe.optimizer_family == "dualnorm"
    if name == "optimizer_convolution":
        return recipe.optimizer_family == "dualnorm"
    if name == "optimizer_convolution_rule":
        return recipe.optimizer_family == "dualnorm" and recipe.optimizer_convolution == "per_offset"
    if name == "optimizer_adam_lr":
        return recipe.optimizer_family in {"dualnorm_D_only", "particle_rownorm_only"}
    if name in {"betas", "amsgrad", "beta2_end"}:
        return recipe.optimizer_family in {
            "formulation", "adam", "ada_nsgda", "dualnorm_D_only", "particle_rownorm_only"}
    if name == "prior_betas" and recipe.optimizer_family not in {
            "formulation", "adam", "ada_nsgda", "dualnorm_D_only"}:
        return False
    if name == "d_betas":
        return recipe.optimizer_family in {"formulation", "adam", "ada_nsgda", "particle_rownorm_only"}
    if name in {"lr_decay_rate", "lr_decay_steps"}:
        return recipe.lr_schedule == "exponential" and recipe.lr_decay_rate < 1
    if name in {"lr_anneal_start", "lr_floor", "network_lr_floor"} and recipe.lr_schedule != "cosine":
        return False
    if name == "direct_particle_betas":
        return recipe.optimizer_family == "formulation" and (
            task is None or prior_control_binding(task)["representation"] == "direct_sample_coordinates")
    penalty_end = recipe.reg_coeff if recipe.reg_coeff_end is None else recipe.reg_coeff_end
    if name in {"reg_kappa", "reg_every", "reg_coeff_anneal_end"}:
        if recipe.reg_coeff == 0 and penalty_end == 0:
            return False
    if name == "reg_kappa" and recipe.reg_arm == "a_r1r2":
        return False
    if name == "reg_coeff_anneal_end":
        return recipe.reg_coeff_end is not None and recipe.reg_coeff_end != recipe.reg_coeff
    if name == "beta2_anneal_end":
        return recipe.beta2_end is not None and (recipe.beta2_end != recipe.betas[1]
                or recipe.beta2_end != (recipe.d_betas or recipe.betas)[1]
                or (prior_active and recipe.beta2_end != (recipe.prior_betas or recipe.betas)[1]))
    if name in {"lr_anneal_start", "lr_floor", "network_lr_floor"} and recipe.total_steps is None:
        return False
    if name == "lr_anneal_start" and recipe.lr_floor == recipe.resolved_network_lr_floor == 1:
        return False
    if name == "lr_floor" and task is not None and not prior_active and recipe.network_lr_floor is not None:
        return False
    if name in {"prior_lr_mult", "prior_betas", "prior_eps", "prior_reg", "prior_l2",
                "prior_reg_target_std", "prior_reg_eps", "prior_proximal_strength"}:
        if not prior_active:
            return False
    if name.startswith("prior_coulomb_"):
        return prior_active and recipe.prior_regularizer == "coulomb" and recipe.prior_reg > 0
    if name in {"prior_reg_target_std", "prior_reg_eps"} and recipe.prior_reg == 0:
        return False
    if name == "prior_reg_target_std" and recipe.prior_regularizer not in {"vicreg", "sbnp", "empf"}:
        return False
    if name == "prior_reg_eps" and recipe.prior_regularizer not in {"vicreg", "moment_tether", "sbnp", "empf"}:
        return False
    if name in {"prior_sigreg_projections", "prior_sigreg_knots"}:
        return prior_active and recipe.prior_regularizer == "sigreg" and recipe.prior_reg > 0
    if name == "kinetic_transport_weight":
        return recipe.kinetic_transport_weight > 0
    if name == "kinetic_transport_local_weight":
        return recipe.kinetic_transport_local_weight > 0
    if name == "kinetic_transport_projections":
        return recipe.kinetic_transport_weight > 0 or recipe.kinetic_transport_local_weight > 0
    if name == "rhbi_weight":
        return recipe.rhbi_weight > 0
    if name == "rhbi_critic_weight":
        return recipe.rhbi_critic_weight > 0
    if name in {"rhbi_self_coupling", "rhbi_relativistic_weight", "rhbi_slack_gain", "rhbi_thermal_tau"}:
        return recipe.rhbi_weight > 0
    if name == "rhbi_thermal_shock_mu":
        return recipe.rhbi_weight > 0 and recipe.rhbi_thermal_tau > 0
    if name in {"rhbi_lambda_floor", "rhbi_gamma_gauge", "rhbi_chord_weight", "rhbi_settle_beta"}:
        return recipe.effective_critic_formulation == "rhbi" and recipe.reg_coeff > 0
    if name == "rhbi_settle_mode":
        return recipe.effective_critic_formulation == "rhbi" and recipe.reg_coeff > 0 and recipe.rhbi_settle_beta > 0
    if name == "rhbi_akt_horizon":
        return recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0 or (
            recipe.effective_critic_formulation == "rhbi" and recipe.reg_coeff > 0 and recipe.rhbi_chord_weight > 0
        )
    if name == "rhbi_akt_mode":
        return recipe.rhbi_akt_horizon and (
            recipe.rhbi_weight > 0
            or recipe.rhbi_critic_weight > 0
            or (recipe.effective_critic_formulation == "rhbi" and recipe.reg_coeff > 0 and recipe.rhbi_chord_weight > 0)
        )
    if name == "rhbi_fibre_rel_blend":
        return (
            recipe.effective_critic_formulation == "rhbi"
            or recipe.rhbi_weight > 0
            or recipe.rhbi_critic_weight > 0
        ) and recipe.rhbi_conditional_fibre == "slack"
    if name == "rhbi_step_damp":
        return (
            (recipe.effective_critic_formulation == "rhbi" or recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0)
            and recipe.optimizer_family not in {"formulation", "adam"}
            and recipe.rhbi_thermal_tau > 0
        )
    if name == "rhbi_cond_ot_weight":
        return recipe.rhbi_weight > 0 and recipe.rhbi_conditional_fibre == "slack"
    if name == "rhbi_cond_ot_mode":
        return recipe.rhbi_weight > 0 and recipe.rhbi_conditional_fibre == "slack" and recipe.rhbi_cond_ot_weight > 0
    if name == "rhbi_simplex_weight":
        return recipe.rhbi_weight > 0
    if name == "rhbi_simplex_chart_weight":
        return recipe.rhbi_weight > 0 and recipe.rhbi_simplex_weight > 0
    if name == "rhbi_simplex_macro_settle":
        return recipe.rhbi_weight > 0 and recipe.rhbi_simplex_weight > 0 and recipe.rhbi_simplex_chart_weight > 0
    if name in {"rhbi_cvx_mu_gain", "rhbi_tail_spill_gain"}:
        return recipe.rhbi_weight > 0 and recipe.rhbi_thermal_tau > 0
    if name in {"rhbi_rho", "rhbi_tau", "rhbi_eps", "rhbi_iters", "rhbi_c_max", "rhbi_beta_ell", "rhbi_nn_quantile", "rhbi_rapidity_screening", "rhbi_scale_log_odds", "rhbi_conditional_fibre"}:
        return recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0
    return True
