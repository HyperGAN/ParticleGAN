"""Merged optional objectives must execute on their declared caller or refuse."""
from importlib import import_module

import pytest
import torch

from benchmarks.toy_audit import api_contract


MASKED = ("masked_mixture_weight", "masked_transport_weight", "imputer_component_weight")
UNSUPPORTED = (
    [("api_model_misgan", field) for field in ("conditional_energy_weight", "joint_reconstruction_weight")]
    + [("api_model_vae", field) for field in ("conditional_energy_weight", "joint_reconstruction_weight", *MASKED)]
    + [("api_model_ddgan", field) for field in ("joint_reconstruction_weight", *MASKED)]
    + [("api_model_ae", field) for field in ("conditional_energy_weight", "joint_reconstruction_weight", *MASKED)]
)


@pytest.mark.parametrize("provider,field", UNSUPPORTED)
def test_direct_model_provider_refuses_unconsumed_objective_before_construction(provider, field):
    module = import_module("benchmarks.toy_audit." + provider)
    case = module.list_cases()[0]
    recipe = module.task_recipe(case).replace(**{field: 1.})
    before = torch.get_rng_state().clone()
    with pytest.raises(api_contract.UnsupportedRecipeOverrides, match="component consumer|word joint consumer"):
        module.task_recipe(case, recipe=recipe)
    with pytest.raises(api_contract.UnsupportedRecipeOverrides, match="component consumer|word joint consumer"):
        module.build_case(case["id"], recipe=recipe, max_steps=1)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("provider,field", [
    *UNSUPPORTED,
])
def test_public_build_rejects_unconsumed_injection_before_calling_provider(monkeypatch, provider, field):
    module = import_module("benchmarks.toy_audit." + provider)
    case = dict(module.list_cases()[0], provider=provider)
    recipe = module.task_recipe(case).replace(**{field: 1.})

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported recipe reached provider construction")

    monkeypatch.setattr(module, "build_case", forbidden)
    with pytest.raises(api_contract.UnsupportedRecipeOverrides, match="component consumer|word joint consumer"):
        api_contract.build(case, recipe=recipe, max_steps=1)


@pytest.mark.parametrize("provider,overrides", [
    ("api_model_ddgan", {"conditional_energy_weight": 1.}),
    ("api_model_misgan", {field: 1. for field in MASKED}),
    ("api_model_vae", {"kinetic_transport_weight": .2, "kinetic_transport_local_weight": .3}),
])
def test_supported_model_objectives_keep_the_exact_recipe(provider, overrides):
    module = import_module("benchmarks.toy_audit." + provider)
    case = dict(module.list_cases()[0], provider=provider)
    recipe = module.task_recipe(case).replace(**overrides)
    fixture = api_contract.build(case, recipe=recipe, max_steps=1)
    assert fixture.recipe.to_dict() == recipe.to_dict()
    assert fixture.completed_steps == 0


def test_word_joint_consumer_keeps_its_declared_objective(monkeypatch):
    from benchmarks.toy_audit import api_images
    case = next(dict(case, provider="api_images") for case in api_images.list_cases() if case["kind"] == "word")
    # The provider returns a fixture carrying the actual resolved word recipe;
    # keep this guard test independent of the word model's large constructor.
    from particlegan import Recipe
    from types import SimpleNamespace
    recipe = Recipe.from_dict(dict(joint_reconstruction_weight=1.))
    fixture = SimpleNamespace(recipe=recipe, api_components=("Recipe.joint_reconstruction_loss",),
                              step=lambda: None, observe=lambda: None, state_dict=lambda: {})
    monkeypatch.setattr(api_images, "build_case", lambda *args, **kwargs: fixture)
    assert api_contract.build(case, max_steps=1) is fixture
