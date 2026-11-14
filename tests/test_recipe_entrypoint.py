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


def test_selected_default_has_no_effective_decay_or_ema():
    from particlegan.recipes import learning_rate_scales
    recipe = get_recipe()
    assert recipe.name == "empf"
    assert recipe.ema_decay == recipe.reg_anchor_weight == 0
    for step in (0, 200, 800, 960, 1600, 7000, 14000):
        assert learning_rate_scales(step, recipe) == (1., 1.)


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
