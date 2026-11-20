"""Software contracts for mechanism identity; these do not qualify science."""
from dataclasses import asdict

import pytest

from particlegan import Recipe

from experiments.forge.techniques import recipe_field_active, technique_signature, validate_same_technique


@pytest.fixture
def modern():
    return Recipe.from_dict(dict(optimizer_family="adam", reg_arm="a_r1r2", d_guard_ratio=0,
                  reg_anchor_weight=0, latent_damping_max_rate=0, direct_particle_gain=False,
                  betas=(0, .9), beta2_end=.99, reg_coeff_end=.1,
                  lr_floor=1, network_lr_floor=1, network_lr_horizon_cap=None))


def test_positive_strengths_and_schedule_timing_keep_mechanisms(modern):
    trial = modern.replace(lr=.0085, reg_coeff=.5, reg_coeff_end=.05, reg_coeff_anneal_end=.3,
                           beta2_anneal_end=.4, betas=(0, .95), eps=1e-7, reg_every=2,
                           lr_floor=.25, network_lr_floor=.25)
    assert validate_same_technique(modern, trial) == technique_signature(asdict(modern))


@pytest.mark.parametrize("settings, mechanism", [
    ({"reg_coeff": 0}, "critic_penalty"),
    ({"reg_coeff_end": 0}, "critic_penalty"),
    ({"reg_coeff_end": None}, "penalty_schedule"),
    ({"reg_coeff_end": 1}, "penalty_schedule"),
    ({"beta2_end": None}, "beta2_schedule"),
    ({"beta2_end": .9}, "beta2_schedule"),
    ({"beta2_end": 0}, "terminal_second_moment"),
    ({"amsgrad": True}, "amsgrad"),
    ({"betas": (.1, .9)}, "network_moments"),
    ({"prior_betas": (.1, .9)}, "prior_moments"),
    ({"prior_reg": .01}, "prior_regularization"),
    ({"lr_floor": 0}, "prior_terminal_updates"),
    ({"network_lr_floor": 0}, "network_terminal_updates"),
])
def test_search_cannot_add_or_remove_mechanisms(modern, settings, mechanism):
    with pytest.raises(ValueError, match=mechanism):
        validate_same_technique(modern, modern.replace(**settings))


def test_an_existing_zero_tail_is_a_separate_preserved_technique(modern):
    base = modern.replace(reg_coeff_end=0)
    validate_same_technique(base, base.replace(reg_coeff=.5, reg_coeff_anneal_end=.3))
    with pytest.raises(ValueError, match="critic_penalty"):
        validate_same_technique(base, modern)


def test_task_owned_prior_and_resources_do_not_change_technique(modern):
    same = modern.replace(prior_kind="mog", sigma_rel=.2, standardize=False,
                          num_particles=256, z_dim=4, batch_size=64, total_steps=80, name="other")
    assert technique_signature(same) == technique_signature(modern)


def test_family_names_cannot_mask_fixed_mechanism_changes(modern):
    other = modern.replace(reg_arm="b_cap")
    with pytest.raises(ValueError, match="Recipe.reg_arm"):
        validate_same_technique(modern, other)


def test_r1r2_repair_controls_are_structural_and_default_signatures_stay_sparse(modern):
    names = {"reg_real_weight", "reg_fake_weight", "joint_reconstruction_weight"}
    assert not names & technique_signature(modern)["fixed_recipe_fields"].keys()
    for name, value in (("reg_fake_weight", 0.), ("reg_real_weight", .5),
                        ("joint_reconstruction_weight", 1.)):
        with pytest.raises(ValueError, match=f"Recipe.{name}"):
            validate_same_technique(modern, modern.replace(**{name: value}))
    word = {"adapter": "word_joint", "execution": {}}
    scalar = {"adapter": "transfer_vector", "execution": {}}
    ae = {"adapter": "public_api_model_toy", "execution": {"host_definition": {"adversarial": False}}}
    assert recipe_field_active("joint_reconstruction_weight", modern, task=word)
    assert not recipe_field_active("joint_reconstruction_weight", modern, task=scalar)
    assert recipe_field_active("reg_fake_weight", modern, task=scalar)
    assert not recipe_field_active("reg_fake_weight", modern, task=ae)
    assert not recipe_field_active("reg_fake_weight", Recipe.from_dict({}), task=scalar)


def test_k3p_floor_cannot_disable_the_blended_penalty_phase():
    base = Recipe.from_dict(dict(critic_formulation="k3p", network_lr_floor=.01))
    validate_same_technique(base, base.replace(network_lr_floor=.25))
    with pytest.raises(ValueError, match="k3p_penalty_lr_floor"):
        validate_same_technique(base, base.replace(network_lr_floor=.5))


def test_inactive_fields_follow_recipe_and_actual_prior_contract(modern):
    assert not recipe_field_active("reg_kappa", modern)
    assert not recipe_field_active("reg_coeff_anneal_end", modern.replace(reg_coeff_end=None))
    assert not recipe_field_active("beta2_anneal_end", modern.replace(beta2_end=None))
    assert not recipe_field_active("lr_anneal_start", modern)
    task = {"execution": {"prior": {"learnable": False}}}
    assert not recipe_field_active("prior_lr_mult", modern, task=task)
    assert not recipe_field_active("prior_betas", modern, task=task)
    assert not recipe_field_active("prior_reg", modern, task=task)
    task["execution"] = {"prior_applicability": "not_sampled"}
    assert not recipe_field_active("prior_lr_mult", modern, task=task)


def test_numeric_ablation_activation_is_visible_even_outside_search_whitelist():
    recipe = Recipe.from_dict({})
    for name in ("reg_anchor_weight", "d_guard_ratio", "latent_damping_max_rate",
                 "input_noise_std", "output_noise_std"):
        with pytest.raises(ValueError, match="changes technique"):
            validate_same_technique(recipe, recipe.replace(**{name: 0}))


def test_kinetic_transport_weights_tune_when_active_but_reject_zero_ablation():
    active = Recipe.from_dict(dict(kinetic_transport_weight=0.05, kinetic_transport_local_weight=0.02))
    tuned = active.replace(kinetic_transport_weight=0.1, kinetic_transport_local_weight=0.04)
    assert validate_same_technique(active, tuned)["mechanisms"]["kinetic_transport"] is True
    assert recipe_field_active("kinetic_transport_weight", active) is True
    assert recipe_field_active("kinetic_transport_local_weight", active) is True
    assert recipe_field_active("kinetic_transport_projections", active) is True
    inactive = Recipe.from_dict({})
    assert recipe_field_active("kinetic_transport_weight", inactive) is False
    assert recipe_field_active("kinetic_transport_projections", inactive) is False
    for name in ("kinetic_transport_weight", "kinetic_transport_local_weight"):
        with pytest.raises(ValueError, match="search changes technique mechanisms"):
            validate_same_technique(active, active.replace(**{name: 0.0}))
