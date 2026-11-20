"""Software contracts for opt-in joint inverse training, not quality gates."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from particlegan import Recipe
from particlegan.joint_reconstruction import joint_reconstruction_loss
from experiments.forge.state import state_digest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_confident_wrong_categorical_logits_keep_corrective_gradients():
    logits = torch.tensor([[[400., -400.], [-400., 400.]]], requires_grad=True)
    target = torch.tensor([[[0., 1.], [1., 0.]]], requires_grad=True)
    loss = joint_reconstruction_loss(logits, target, categorical_dim=1)
    assert loss.item() == 800.
    loss.backward()
    assert torch.equal(logits.grad, torch.tensor([[[.5, -.5], [-.5, .5]]]))
    assert target.grad is None
    # The original probability-space metric clamp has zero derivative here.
    probabilities = logits.detach().requires_grad_().softmax(1)
    clamped = -(probabilities * target.detach()).sum(1).clamp_min(1e-8).log().mean()
    assert torch.count_nonzero(torch.autograd.grad(clamped, probabilities)[0]) == 0


def test_continuous_inverse_loss_detaches_targets_and_weights_gradient():
    values = torch.tensor([[1., 3.]], requires_grad=True)
    target = torch.zeros_like(values, requires_grad=True)
    loss = Recipe.from_dict(dict(joint_reconstruction_weight=2.)).joint_reconstruction_loss(values, target)
    assert loss.item() == 10.
    loss.backward()
    assert torch.equal(values.grad, torch.tensor([[2., 6.]]))
    assert target.grad is None


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True, "1"])
def test_invalid_weight_rejected(value):
    with pytest.raises(ValueError):
        Recipe.from_dict(dict(joint_reconstruction_weight=value))


def build(weight):
    from experiments.forge.api import task_formulation_context
    from benchmarks.toy_audit.api_images import WordFixture
    candidate = json.loads((ROOT / "configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json").read_text())
    candidate["recipe_overrides"]["joint_reconstruction_weight"] = weight
    task = json.loads((ROOT / "configs/forge/tasks/five_word_joint_smoke.json").read_text())
    context = task_formulation_context(candidate, task, {"seed": 0})
    return context, WordFixture(device="cpu", seed=0, recipe_name=None, max_steps=4, components=context)


def test_public_word_inverse_auxiliary_preserves_inputs_and_consumed_streams():
    left_context, left = build(0.)
    right_context, right = build(1.)
    assert state_digest(left.G.state_dict()) == state_digest(right.G.state_dict())
    assert state_digest(left.E.state_dict()) == state_digest(right.E.state_dict())
    calls = [0, 0]
    left.G.net.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    right.G.net.register_forward_hook(lambda *_: calls.__setitem__(1, calls[1] + 1))
    left.step()
    observed = right.step()
    assert calls == [2, 3]
    assert observed["joint_reconstruction"].item() > 0
    assert state_digest(left.D.state_dict()) == state_digest(right.D.state_dict())
    assert state_digest(left.G.state_dict()) != state_digest(right.G.state_dict())
    assert state_digest(left.E.state_dict()) != state_digest(right.E.state_dict())
    assert torch.equal(left.data_generator.get_state(), right.data_generator.get_state())
    assert state_digest(left_context.streams.state_dict()) == state_digest(right_context.streams.state_dict())
    assert "joint_reconstruction_weight" not in left.recipe.to_dict()
    assert right.recipe.to_dict()["joint_reconstruction_weight"] == 1.


def test_public_word_active_checkpoint_resumes_exactly_and_rejects_recipe_change():
    context, first = build(1.)
    first.step()
    saved, streams = deepcopy(first.state_dict()), deepcopy(context.streams.state_dict())
    first.step()
    expected = first.state_dict()
    other_context, restored = build(1.)
    other_context.streams.load_state_dict(streams)
    restored.policy.load_state_dict(saved["api_state"])
    restored.data_generator.set_state(saved["data_generator"])
    restored.step()
    assert state_digest(expected) == state_digest(restored.state_dict())
    _, inactive = build(0.)
    before = state_digest(inactive.state_dict())
    with pytest.raises(ValueError):
        inactive.policy.load_state_dict(saved["api_state"])
    assert state_digest(inactive.state_dict()) == before
