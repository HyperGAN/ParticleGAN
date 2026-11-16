"""Public recipe binding and module-aware construction, not training gates."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import DEFAULT_RECIPE, Recipe, get_recipe, get_recipe_fields


ROOT = Path(__file__).resolve().parents[1]


def test_public_default_matches_every_selected_candidate_field_except_label():
    empf_candidate = json.loads((ROOT / "configs/forge/ideas/empf-v8.json").read_text())
    actual = json.loads(json.dumps(asdict(get_recipe())))
    assert DEFAULT_RECIPE == actual["name"] == "empf"
    for field, value in empf_candidate["recipe_overrides"].items():
        if field != "name":
            assert actual[field] == value, field
    assert get_recipe() == get_recipe("empf")
    assert Recipe.from_dict(get_recipe().to_dict()) == get_recipe()
    assert actual["optimizer_svd_backend"] == "batched"
    assert actual["training_precision"] == "bf16_channels_last"
    # Explicit bispf_optimized, historical snapshot restoration, and presets keep their meaning.
    bispf_candidate = json.loads((ROOT / "configs/forge/ideas/bispf-throughput-v1.json").read_text())
    bispf_actual = json.loads(json.dumps(asdict(get_recipe("bispf_optimized"))))
    for field, value in bispf_candidate["recipe_overrides"].items():
        if field != "name":
            assert bispf_actual[field] == value, field
    assert Recipe.from_dict({}) == get_recipe("ka2")
    assert get_recipe("bispf").optimizer_family == "formulation"
    assert get_recipe("bispf").loss == "relativistic"
    assert get_recipe("bispf").prior_reg == .05


@pytest.mark.parametrize("name", ["gan", "empf", "bispf_optimized", "bispf", "mog", "ae_ddgan", "ka2"])
def test_plain_fields_match_validated_recipe(name):
    assert get_recipe_fields(name) == asdict(get_recipe(name))


def test_plain_fields_load_without_torch_or_package_import():
    source = ROOT / "particlegan/recipes.py"
    script = """
import importlib.util, sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('configuration resolution imported Torch')
sys.meta_path.insert(0, BlockTorch())
spec = importlib.util.spec_from_file_location('standalone_recipes', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
values = module.get_recipe_fields('ae_gan', z_dim=3)
assert values['name'] == 'ae_gan'
assert values['z_dim'] == 3 and values['optimizer_family'] == 'dualnorm'
assert values['optimizer_convolution_rule'] == 'quintic'
assert values['optimizer_dense_profile'] == 'newton_schulz_hybrid_v1'
assert values['optimizer_svd_backend'] == 'batched'
assert values['training_precision'] == 'bf16_channels_last'
assert 'torch' not in sys.modules and 'particlegan' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(source)], check=True)


def test_plain_fields_reject_unknown_fields_without_value_validation():
    with pytest.raises(TypeError, match="Unknown recipe field"):
        get_recipe_fields(missing_option=True)
    assert get_recipe_fields(lr=-1)["lr"] == -1
    with pytest.raises(ValueError):
        get_recipe(lr=-1)


def configuration(**overrides):
    return get_recipe("bispf_optimized", constraint_geometry_mode="none", **overrides)


def test_custom_groups_bind_only_supplied_parameters_and_preserve_fields():
    model = nn.Sequential(nn.Conv2d(2, 3, 1), nn.Conv2d(3, 1, 1))
    model[0].bias.requires_grad_(False)
    optimizer = configuration().make_generator_optimizer([
        {"params": model[0].parameters(), "role": "encoder", "lr": .123, "host_label": "head"},
    ], modules=[model])
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["params"] == [model[0].weight]
    assert group["lr"] == .123 and group["role"] == "encoder" and group["host_label"] == "head"
    assert group["dualnorm_convolution"]["layout"] == "conv2d_out_in"
    assert group["dualnorm_convolution"]["update_version"] == "per_offset_quintic_hybrid_v1"
    assert all(parameter is not model[1].weight for parameter in group["params"])


def test_bound_groups_have_identical_updates_and_resume_to_module_factory():
    left = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ConvTranspose2d(3, 2, 1))
    right = deepcopy(left)
    recipe = configuration(optimizer_momentum=.5)
    explicit = recipe.make_generator_optimizer([{"params": left.parameters(), "role": "generator"}], modules=left)
    automatic = recipe.make_generator_optimizer(right)
    for step in range(2):
        for a, b in zip(left.parameters(), right.parameters()):
            gradient = torch.arange(a.numel(), dtype=a.dtype).reshape(a.shape).sin() + step
            a.grad, b.grad = gradient, gradient.clone()
        explicit.step()
        automatic.step()
        for a, b in zip(left.parameters(), right.parameters()):
            assert torch.equal(a, b)
        if step == 0:
            explicit.load_state_dict(deepcopy(automatic.state_dict()))


@pytest.mark.parametrize("trainable", [False, True])
def test_pretrained_parameters_keep_values_and_input_gradients(trainable):
    trunk = nn.Sequential(nn.Linear(3, 4), nn.Tanh()).requires_grad_(trainable)
    head = nn.Linear(4, 2)
    model = nn.Sequential(trunk, head)
    original = deepcopy(model.state_dict())
    # A stale gradient must not make a permanently frozen weight update-owned.
    trunk[0].weight.grad = torch.ones_like(trunk[0].weight)
    optimizer = configuration().make_generator_optimizer([
        {"params": model.parameters(), "role": "generator"}
    ], modules=model)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert owned == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())
    inputs = torch.ones(2, 3, requires_grad=True)
    model(inputs).square().mean().backward()
    assert inputs.grad is not None and inputs.grad.norm() > 0
    optimizer.step()
    if not trainable:
        for key, value in trunk.state_dict().items():
            assert torch.equal(value, original["0." + key])
    else:
        assert not torch.equal(trunk[0].weight, original["0.0.weight"])
    assert not torch.equal(head.weight, original["1.weight"])


def test_module_binding_rejects_non_module_owners():
    model = nn.Linear(2, 1)
    with pytest.raises(TypeError, match="torch.nn.Module"):
        configuration().make_generator_optimizer(model.parameters(), modules=[object()])


def test_shared_weights_cannot_claim_conflicting_module_layouts():
    convolution = nn.Conv2d(2, 3, 1, bias=False)
    transpose = nn.ConvTranspose2d(3, 2, 1, bias=False)
    transpose.weight = convolution.weight
    with pytest.raises(ValueError, match="incompatible module layouts"):
        configuration().make_generator_optimizer([convolution.weight], modules=[convolution, transpose])
