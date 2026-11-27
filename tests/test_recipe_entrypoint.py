"""New runs must resolve the public preset, never restoration defaults."""
from dataclasses import asdict, replace
import json

import pytest

from particlegan import Recipe, get_recipe, get_recipe_fields


@pytest.mark.parametrize("overrides", [{}, {"lr": 2e-5},
    {"z_dim": 4, "num_particles": 128, "prior_reg": 1.},
    {"name": "bispf_optimized"}])
def test_partial_constructor_fails_before_implicit_ka2_selection(overrides):
    with pytest.raises(TypeError, match=r"Use get_recipe\(\).*Recipe.from_dict"):
        Recipe(**overrides)


def test_selected_default_has_no_effective_decay_or_ema_anchor():
    from particlegan.recipes import learning_rate_scales
    recipe = get_recipe()
    assert recipe.name == "axiom"
    assert recipe.reg_anchor_weight == 0
    for step in (0, 200, 800, 960, 1600, 7000, 14000):
        assert learning_rate_scales(step, recipe) == (1., 1.)
    assert get_recipe("unified_v12").ema_decay == 0


def test_restoration_keeps_recorded_recipe_when_public_default_changes(monkeypatch):
    import particlegan.recipes as module
    historical = get_recipe("ka2", lr=2e-5)
    saved = json.loads(json.dumps(historical.to_dict()))
    monkeypatch.setattr(module, "DEFAULT_RECIPE", "bispf")
    assert get_recipe().name == "bispf"
    assert Recipe.from_dict(saved) == historical
    assert Recipe.from_dict(get_recipe().to_dict()) == get_recipe()


def test_compact_historical_fields_restore_without_new_default_migration():
    restored = Recipe.from_dict({"lr": 2e-5})
    assert restored.name == "ka2" and restored.lr == 2e-5
    assert restored.ema_decay == .995
    assert Recipe.from_dict(restored.to_dict()) == restored


def test_fully_resolved_construction_and_dataclass_replace_remain_safe():
    recipe = get_recipe()
    assert Recipe(**get_recipe_fields()) == recipe
    changed = replace(recipe, lr=.02)
    assert changed == recipe.replace(lr=.02)
    assert asdict(changed) == {**asdict(recipe), "lr": .02}


@pytest.mark.parametrize("fields", [None, [], [("lr", .02)]])
def test_restore_requires_a_mapping(fields):
    with pytest.raises(TypeError, match="must be a mapping"):
        Recipe.from_dict(fields)


def test_restored_values_and_unknown_fields_still_validate():
    with pytest.raises(ValueError, match="lr"):
        Recipe.from_dict({"lr": -1})
    with pytest.raises(TypeError, match="Unknown recipe field"):
        Recipe.from_dict({"lrr": .02})
    with pytest.raises(TypeError, match="Unknown recipe field"):
        get_recipe().replace(lrr=.02)


def test_recipe_validation_and_standalone_axiom_critic_optimizer():
    import torch
    from torch import nn
    from particlegan import AxiomCriticPenalty, AxiomOptimizer, AxiomStepRecord, scale_learning_rates
    from particlegan.priors import ManipulatedPrior, SimplexRoutingManipulator

    # 1. kappa_radial rejects nan/inf
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="kappa_radial"):
            get_recipe("axiom", kappa_radial=bad)

    # 2. axiom and minimal reject unknown penalty_overrides with TypeError and non-default real/fake weights
    for preset in ("axiom", "minimal"):
        r = get_recipe(preset)
        with pytest.raises(TypeError, match="unknown"):
            r._penalty_options(unknown_key=1.0)
        with pytest.raises(ValueError, match="real/fake penalty weights"):
            r.replace(reg_real_weight=2.0)
        with pytest.raises(ValueError, match="real/fake penalty weights"):
            r.replace(reg_fake_weight=0.5)

    # 3. make_critic_optimizer on axiom with optimizer_family="formulation" works standalone with AxiomCriticPenalty
    r_ax = get_recipe(
        "axiom",
        optimizer_family="formulation",
        optimizer_smoothing=0.0,
        constraint_geometry_mode="none",
    )
    critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1))
    opt_d = r_ax.make_critic_optimizer(critic)
    assert isinstance(opt_d, AxiomOptimizer)
    assert isinstance(opt_d.record, AxiomStepRecord)
    penalty = r_ax.make_critic_penalty(opt_d)
    assert isinstance(penalty, AxiomCriticPenalty)
    real, fake = torch.randn(8, 2), torch.randn(8, 2)
    pen = penalty(critic, real, fake)
    assert torch.isfinite(pen)
    pen.backward()
    opt_d.step()
    assert opt_d.record.observed_steps == 1

    # 4. scale_learning_rates classifies router params as network (not prior) and applies penalty annealing
    r_sched = get_recipe(
        "bcap_adam",
        num_particles=16,
        z_dim=2,
        total_steps=20,
        lr_anneal_start=0.0,
        lr_floor=0.1,
        network_lr_floor=0.5,
        reg_coeff=1.0,
        reg_coeff_end=0.1,
        reg_coeff_anneal_end=0.2,
    )
    base_prior = r_sched.make_prior()
    router = SimplexRoutingManipulator(num_modes=4, z_dim=2, learnable_projection=True)
    wrapped_prior = ManipulatedPrior(base_prior, [router])
    gen = nn.Linear(2, 2)
    crit = nn.Linear(2, 1)
    opt_g, opt_d = r_sched.make_optimizers(gen, crit, wrapped_prior)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in (opt_g, opt_d)]
    pen_obj = r_sched.make_critic_penalty(opt_d)
    net_s, prior_s = scale_learning_rates(4, r_sched, (opt_g, opt_d), base_lrs, wrapped_prior, penalty=pen_obj)
    assert net_s != prior_s
    # Param groups of opt_g: [generator, prior table, router]
    assert opt_g.param_groups[0]["lr"] == pytest.approx(base_lrs[0][0] * net_s)
    assert opt_g.param_groups[1]["lr"] == pytest.approx(base_lrs[0][1] * prior_s)
    assert opt_g.param_groups[2]["lr"] == pytest.approx(base_lrs[0][2] * net_s)
    assert pen_obj.regularizer.coeff == pytest.approx(0.1)
