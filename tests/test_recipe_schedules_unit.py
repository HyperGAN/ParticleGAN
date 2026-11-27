"""Direct tests for each public schedule helper in ``particlegan.recipe_schedules``.

test_recipe_schedules.py covers the composed trainer path; these pin the
individual functions so a refactor of one cannot hide behind another.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import Recipe
from particlegan.recipe_schedules import (
    apply_optimizer_schedule, apply_penalty_schedule, cosine_value, make_plain_adam,
    validate_plain_adam_state,
)


def plain_recipe(**changes):
    return Recipe.from_dict({
        "name": "schedule-unit", "optimizer_family": "adam", "reg_arm": "a_r1r2",
        "num_particles": 8, "z_dim": 2, "batch_size": 4, "total_steps": 20,
        "standardize": False, "lr": .0002, "d_lr_mult": 1., "prior_lr_mult": 1.,
        "betas": (0., .9), "beta2_end": .99, "beta2_anneal_end": .2,
        "reg_coeff": 1., "reg_coeff_end": .1, "reg_coeff_anneal_end": .2,
        "reg_every": 1, "reg_anchor_weight": 0., "d_guard_ratio": 0.,
        "latent_damping_max_rate": 0., "direct_particle_gain": False,
        "input_noise_std": 0., "output_noise_std": 0., "amsgrad": False,
        "network_lr_floor": 1., "network_lr_horizon_cap": None, "lr_floor": 1., **changes})


@pytest.mark.parametrize("step,expected", [(0, 1.), (2, .55), (4, .1), (5, .1), (400, .1), (1, .1 + .9 * .5 * (1 + .5 ** .5))])
def test_cosine_value_anneals_over_the_declared_fraction_then_holds(step, expected):
    assert cosine_value(step, 1., .1, .2, 20) == pytest.approx(expected)


@pytest.mark.parametrize("step", [-1, 1.0, True, None])
def test_cosine_value_rejects_non_integer_or_negative_steps(step):
    with pytest.raises(ValueError, match="nonnegative integer"):
        cosine_value(step, 1., .1, .2, 20)


def test_optimizer_schedule_sets_beta2_for_every_group_from_its_recipe_initial_betas():
    recipe = plain_recipe()
    a, b = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(3))
    optimizer = make_plain_adam(recipe, [dict(params=[a]), dict(params=[b], betas=(.5, .8))], betas=(0., .9))
    apply_optimizer_schedule(2, recipe, optimizer)
    assert optimizer.param_groups[0]["betas"] == pytest.approx((0., .945))
    assert optimizer.param_groups[1]["betas"] == pytest.approx((.5, .8 + (.99 - .8) * .5))
    apply_optimizer_schedule(20, recipe, optimizer)
    assert [group["betas"][1] for group in optimizer.param_groups] == pytest.approx([.99, .99])


def test_optimizer_schedule_is_a_no_op_without_beta2_end_and_refuses_foreign_optimizers():
    parameter = nn.Parameter(torch.zeros(2))
    foreign = torch.optim.Adam([parameter], betas=(0., .9))
    apply_optimizer_schedule(3, plain_recipe(beta2_end=None), foreign)
    assert foreign.param_groups[0]["betas"] == (0., .9)
    with pytest.raises(ValueError, match="optimizer from the recipe factories"):
        apply_optimizer_schedule(3, plain_recipe(), foreign)


def test_penalty_schedule_sets_the_coefficient_and_skips_absent_schedules():
    penalty = SimpleNamespace(initial_coeff=1., regularizer=SimpleNamespace(coeff=1.))
    apply_penalty_schedule(2, plain_recipe(), penalty)
    assert penalty.regularizer.coeff == pytest.approx(.55)
    penalty.regularizer.coeff = 7.
    apply_penalty_schedule(2, plain_recipe(reg_coeff_end=None), penalty)
    assert penalty.regularizer.coeff == 7.
    apply_penalty_schedule(2, plain_recipe(), None)


def test_plain_adam_checkpoint_validation_rejects_bad_observer_and_schedule_state():
    recipe = plain_recipe()
    critic = nn.Linear(2, 1)
    optimizer = make_plain_adam(recipe, critic.parameters(), critic=critic, lr=.1, betas=(0., .9))
    state = optimizer.state_dict()
    validate_plain_adam_state(optimizer, state)
    for corrupt, match in (
            (lambda s: s.pop("regularizer"), "critic observer checkpoint"),
            (lambda s: s["regularizer"].update(ema=1), "critic observer checkpoint"),
            (lambda s: s["regularizer"]["record"].update(calls=-1), "critic step count"),
            (lambda s: s["regularizer"]["record"].update(anchor_started=True), "critic anchor"),
            (lambda s: s["regularizer"]["record"].update(lr_max=float("nan")), "critic LR record"),
            (lambda s: s["param_groups"][0].update(_recipe_initial_betas=(0., .5)), "beta2 schedule initialization")):
        saved = optimizer.state_dict()
        corrupt(saved)
        with pytest.raises(ValueError, match=match):
            validate_plain_adam_state(optimizer, saved)
    with pytest.raises(ValueError, match="invalid plain Adam checkpoint"):
        validate_plain_adam_state(optimizer, [])
    # Optimizers outside the plain-Adam family are not this validator's concern.
    validate_plain_adam_state(torch.optim.SGD(critic.parameters(), lr=.1), [])


@pytest.mark.parametrize("formulation,expected_cls_name", [
    ("unified_v15", "_UnifiedV15ObserverStepRecord"),
    ("axiom", "AxiomStepRecord"),
    ("minimal", "MinimalStepRecord"),
])
def test_explicit_formulations_precede_rhbi_weight_in_plain_adam_dualnorm_and_binding(formulation, expected_cls_name):
    from particlegan import bind_formulation_records, get_recipe
    from particlegan.optim.dualnorm import make_normalized_optimizer

    critic = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    # Explicit formulation with rhbi_weight > 0 must NOT be shadowed by RHBI observer
    r_adam = get_recipe(
        formulation,
        optimizer_family="adam",
        optimizer_smoothing=0.0,
        constraint_geometry_mode="none",
        rhbi_weight=0.5,
    )
    opt_adam = make_plain_adam(r_adam, list(critic.parameters()), critic=critic, lr=1e-3)
    assert type(opt_adam.record).__name__ == expected_cls_name
    validate_plain_adam_state(opt_adam, opt_adam.state_dict())
    opt_adam.load_state_dict(opt_adam.state_dict())

    r_norm = get_recipe(
        formulation,
        optimizer_family="dualnorm",
        optimizer_smoothing=0.0,
        constraint_geometry_mode="none",
        rhbi_weight=0.5,
    )
    opt_norm = make_normalized_optimizer(r_norm, [{"params": list(critic.parameters()), "role": "critic"}], critic=critic, lr=1e-3)
    assert type(opt_norm.record).__name__ == expected_cls_name

    # bind_formulation_records binds the formulation's own record (not shadowed by rhbi_weight)
    loss = r_norm.make_loss()
    reg = r_norm.make_prior_regularizer()
    rec = bind_formulation_records(r_norm, opt_norm, loss, reg, generator=nn.Linear(2, 2))
    assert rec is opt_norm.record
