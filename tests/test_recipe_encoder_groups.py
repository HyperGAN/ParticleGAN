"""Native Adam encoder group parity and public model-provider continuation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy_audit import api_model_ae, api_model_ae_ddgan, api_model_vae
from experiments.forge.state import state_digest
from particlegan import Recipe, init
from particlegan.recipe_schedules import apply_training_schedules


ROOT = Path(__file__).resolve().parents[1]


def r1_recipe(card="r1r2-repair-control-20261022"):
    return Recipe.from_dict(json.loads((ROOT / "configs/forge/ideas" / (card + ".json")).read_text())["recipe_overrides"])


@pytest.fixture(autouse=True)
def serial_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def components(recipe, dtype=torch.float32):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        generator, encoder, critic = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 1)
        prior = recipe.make_prior(sigma=.1)
        for offset, module in enumerate((generator, encoder, critic, prior)):
            init.deterministic_orthogonal_(module, seed=offset)
            module.to(dtype=dtype)
    return generator, encoder, critic, prior


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("card", ["r1r2-repair-control-20261022", "recipe-prior-refactor-r1r2-v1"])
def test_native_adam_split_is_bit_exact_per_parameter_and_state(dtype, card):
    recipe = r1_recipe(card).replace(num_particles=8, z_dim=2, batch_size=4, total_steps=20,
        prior_kind="mog", sigma_rel=0., standardize=False)
    left = components(recipe, dtype)
    right = deepcopy(left)
    g, e, d, prior = left
    split_g, split_e, split_d, split_prior = right
    combined, combined_d = recipe.make_optimizers(g, d, prior, encoder=e, foreach=False, fused=False)
    split, split_critic = recipe.make_optimizers(split_g, split_d, split_prior,
        encoder=split_e, separate_encoder_groups=True, foreach=False, fused=False)
    assert type(combined) is type(split) is torch.optim.Adam
    assert len(combined.param_groups) == 3
    assert [group["role"] for group in split.param_groups] == ["generator", "encoder", "prior"]
    assert split.param_groups[0]["lr"] == split.param_groups[1]["lr"]
    assert split.param_groups[0]["betas"] == split.param_groups[1]["betas"]
    left_params = [p for group in combined.param_groups for p in group["params"]]
    right_params = [p for group in split.param_groups for p in group["params"]]
    for step in range(6):
        apply_training_schedules(step, recipe, (combined, combined_d))
        apply_training_schedules(step, recipe, (split, split_critic))
        for index, (a, b) in enumerate(zip(left_params, right_params)):
            gradient = torch.arange(a.numel(), dtype=dtype).reshape_as(a) / 11 + (index + 1) * .13 - step * .07
            # A missing gradient keeps its own Adam clock, independently of group boundaries.
            a.grad = None if (step + index) % 4 == 0 else gradient.clone()
            b.grad = None if a.grad is None else gradient.clone()
        combined.step()
        split.step()
        for a, b in zip(left_params, right_params):
            assert torch.equal(a, b)
            assert state_digest(combined.state.get(a, {})) == state_digest(split.state.get(b, {}))


def test_default_keeps_legacy_groups_and_optimizer_checkpoint_layout():
    recipe = r1_recipe().replace(num_particles=8, z_dim=2, prior_kind="mog", sigma_rel=0., standardize=False)
    g, e, d, prior = components(recipe)
    default, _ = recipe.make_optimizers(g, d, prior, encoder=e, foreach=False)
    explicit, _ = recipe.make_optimizers(g, d, prior, encoder=e,
        separate_encoder_groups=False, foreach=False)
    assert len(default.param_groups) == 3
    assert [id(p) for p in default.param_groups[0]["params"]] == [id(p) for p in g.parameters()]
    assert default.param_groups[1]["params"] == list(e.parameters())
    assert default.param_groups[2]["params"] == [prior.z]
    assert all("role" not in group for group in default.param_groups)
    assert state_digest(default.state_dict()) == state_digest(explicit.state_dict())
    restored, _ = recipe.make_optimizers(g, d, prior, encoder=e, foreach=False)
    restored.load_state_dict(default.state_dict())
    assert state_digest(restored.state_dict()) == state_digest(default.state_dict())
    with pytest.raises(TypeError, match="must be a boolean"):
        recipe.make_optimizers(g, d, prior, encoder=e, separate_encoder_groups=1)


CASES = [(provider, case) for provider in (api_model_ae, api_model_vae, api_model_ae_ddgan)
         for case in provider.list_cases()]


@pytest.mark.parametrize("provider,case", CASES, ids=[case["id"] for _, case in CASES])
def test_plain_ns_amsgrad_public_model_policy_and_exact_resume(provider, case):
    task_recipe = provider.task_recipe(case)
    task_fields = ("model", "conditioning", "encoder_mode", "num_classes", "alpha_bar",
        "prior_kind", "sigma_rel", "standardize", "num_particles", "z_dim", "batch_size",
        "total_steps", "observation_sigma", "reconstruction_weight")
    recipe = r1_recipe().replace(**{name: getattr(task_recipe, name) for name in task_fields})
    assert recipe.loss == "non_saturating" and recipe.amsgrad
    fixture = provider.build_case(case["id"], recipe=recipe, max_steps=3)
    assert type(fixture.opt_g) is torch.optim.Adam
    assert [group["role"] for group in fixture.opt_g.param_groups] == ["generator", "encoder", "prior"]
    assert fixture.policy.roles[0] == ["generator", "encoder", "table"]
    fixture.step()
    checkpoint = fixture.state_dict()
    expected_step = fixture.step()
    expected = state_digest(fixture.state_dict())
    restored = provider.build_case(case["id"], recipe=recipe, max_steps=3)
    restored.load_state_dict(checkpoint)
    assert restored.step() == expected_step
    assert state_digest(restored.state_dict()) == expected
