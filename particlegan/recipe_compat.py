"""Default projections for Recipe additions; archived packets keep their identity."""

SEARCH_RECIPE_DEFAULTS = {
    "reg_real_weight": 1.0, "reg_fake_weight": 1.0,
    "joint_reconstruction_weight": 0.0,
    "conditional_energy_weight": 0.0, "conditional_energy_samples": 8,
    "training_precision": "float32",
    "prior_update": "learned", "prior_regularizer": "vicreg",
    "prior_coulomb_softening": .1, "prior_coulomb_confinement": .35, "prior_coulomb_neighbors": 8,
    "prior_reg_target_std": 1.0, "prior_reg_eps": 1e-4, "prior_l2": 0.0,
    "prior_proximal_strength": 0.0,
    "prior_sigreg_projections": 32, "prior_sigreg_knots": 17,
    "critic_step_mode": "none",
    "constraint_geometry_mode": "none",
    "kinetic_transport_weight": 0.0,
    "masked_mixture_weight": 0.0,
    "masked_transport_weight": 0.0,
    "imputer_component_weight": 0.0,
    "kinetic_transport_local_weight": 0.0,
    "kinetic_transport_projections": 32,
    "optimizer_smoothing": 0.0,
    "optimizer_convolution": "none",
    "optimizer_convolution_rule": "polar",
    "optimizer_svd_backend": "native",
    "optimizer_dense_profile": "default",
    "d_betas": None, "d_eps": None, "prior_eps": None,
    "loss_labels": (0.0, 1.0, 1.0), "adam_variant": "pytorch",
    "lr_schedule": "cosine", "lr_decay_rate": 0.96, "lr_decay_steps": 50_000,
    "lr_decay_staircase": False,
    "bispf_shape_gates": True, "bispf_gate_overrides": (),
    "prior_code_law": "table",
    "kappa_radial": 1.0,
    "ebi_beta": 30.0,
    "generator_gradient_response": "none",
}


def without_default_additions(values, *, fields=None):
    """Omit registered defaults, optionally limited to a serializer's additions.

    Older full-recipe receipts retain their existing fields; they can project
    only fields added after that receipt contract instead of changing its shape.
    """
    result = dict(values)
    defaults = (SEARCH_RECIPE_DEFAULTS if fields is None else
                {name: SEARCH_RECIPE_DEFAULTS[name] for name in fields})
    for name, default in defaults.items():
        value = result.get(name, default)
        if isinstance(value, list) and isinstance(default, tuple):
            value = tuple(value)
        if value == default:
            result.pop(name, None)
    return result
