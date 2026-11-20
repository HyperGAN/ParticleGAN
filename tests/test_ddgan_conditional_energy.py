"""Proper-score controls and isolation of the DDGAN auxiliary objective."""
from copy import deepcopy

import pytest
import torch

from particlegan import Recipe, conditional_energy_score
from benchmarks.toy_audit.api_model_ddgan import build_case
from experiments.forge.state import state_digest


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_energy_score_uses_unbiased_offdiagonal_repulsion_and_both_gradients():
    samples = torch.tensor([[[-1.], [1.]]], requires_grad=True)
    score = conditional_energy_score(samples, torch.zeros(1, 1))
    assert score.item() == 0.  # biased V-statistic would give .5
    score.backward()
    torch.testing.assert_close(samples.grad, torch.zeros_like(samples))
    shifted = (samples.detach() + 2).requires_grad_()
    conditional_energy_score(shifted, torch.zeros(1, 1)).backward()
    assert shifted.grad.sum().item() == 1.


def test_energy_score_oracle_beats_collapse_wrong_mean_and_wrong_spread():
    rng = torch.Generator().manual_seed(0)
    n, k = 32768, 8
    target = torch.randn(n, 2, generator=rng)
    oracle = torch.randn(n, k, 2, generator=rng)
    good = conditional_energy_score(oracle, target)
    for wrong in (torch.zeros_like(oracle), oracle + 1., oracle * .2, oracle * 2.):
        assert conditional_energy_score(wrong, target) > good + .03
    # Two-mode law: preserving stochastic modes also beats their conditional mean.
    target = torch.where(target > 0, 1., -1.)
    oracle = torch.where(oracle > 0, 1., -1.)
    assert conditional_energy_score(oracle, target) < conditional_energy_score(torch.zeros_like(oracle), target) - .1


@pytest.mark.parametrize("geometry", ["direction_blend", "none"])
def test_energy_objective_checkpoint_replay_and_matched_data_streams(geometry):
    baseline = build_case("api-model-ddgan-gaussian-chain", max_steps=3)
    recipe = baseline.recipe.replace(conditional_energy_weight=1., conditional_energy_samples=8,
                                    constraint_geometry_mode=geometry)
    candidate = build_case("api-model-ddgan-gaussian-chain", recipe=recipe, max_steps=3)
    assert state_digest(baseline.G.state_dict()) == state_digest(candidate.G.state_dict())
    assert state_digest(baseline.prior.state_dict()) == state_digest(candidate.prior.state_dict())
    baseline.step(); candidate.step()
    assert baseline.batch_digest == candidate.batch_digest
    assert candidate.last_losses["conditional_energy"] > 0
    saved = deepcopy(candidate.state_dict())
    expected = candidate.step()
    restored = build_case("api-model-ddgan-gaussian-chain", recipe=recipe, max_steps=3)
    restored.load_state_dict(saved)
    assert restored.step() == expected
    assert state_digest(restored.state_dict()) == state_digest(candidate.state_dict())
    baseline.step()
    assert baseline.batch_digest == candidate.batch_digest
    before = state_digest(candidate.state_dict())
    candidate.observe(n=64)
    assert state_digest(candidate.state_dict()) == before


@pytest.mark.parametrize("kwargs", [{"conditional_energy_weight": -1.},
    {"conditional_energy_weight": float("nan")}, {"conditional_energy_samples": 1},
    {"conditional_energy_samples": 2.5}])
def test_recipe_rejects_invalid_energy_settings(kwargs):
    with pytest.raises(ValueError):
        Recipe.from_dict(kwargs)


def test_disabled_energy_fields_preserve_archived_recipe_packets():
    assert "conditional_energy_weight" not in Recipe.from_dict({}).to_dict()
    assert "conditional_energy_samples" not in Recipe.from_dict({}).to_dict()
    assert Recipe.from_dict(dict(conditional_energy_weight=1.)).to_dict()["conditional_energy_weight"] == 1.


def test_target_is_fixed_and_nonfinite_predictions_fail_finite_guard():
    target = torch.ones(2, 1, requires_grad=True)
    samples = torch.zeros(2, 3, 1, requires_grad=True)
    conditional_energy_score(samples, target).backward()
    assert target.grad is None
    assert samples.grad is not None
    assert not torch.isfinite(conditional_energy_score(torch.full((2, 3, 1), float("nan")), target))


def test_unsupported_ae_composition_refuses_energy_before_construction():
    from benchmarks.toy_audit.api_model_ae_ddgan import CASES as AE_CASES
    from benchmarks.toy_audit.api_model_ddgan import task_recipe
    base = task_recipe(AE_CASES["api-model-ae-ddgan-gaussian-chain"])
    with pytest.raises(ValueError, match="not bound"):
        task_recipe(AE_CASES["api-model-ae-ddgan-gaussian-chain"],
                    recipe=base.replace(conditional_energy_weight=1.))


def test_auxiliary_uses_independent_same_context_draws_and_no_analytic_posterior(monkeypatch):
    base = build_case("api-model-ddgan-gaussian-chain", max_steps=1)
    fixture = build_case("api-model-ddgan-gaussian-chain", max_steps=1,
                        recipe=base.recipe.replace(conditional_energy_weight=1.))
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation-only posterior leaked into training")
    for name in ("posterior_parameters", "oracle_clean", "transition_parameters"):
        monkeypatch.setattr(fixture.target, name, forbidden)
    calls = []
    original = fixture.G.forward
    def recorded(xt, z, t):
        calls.append((xt.detach().clone(), z.detach().clone(), t.detach().clone()))
        return original(xt, z, t)
    monkeypatch.setattr(fixture.G, "forward", recorded)
    fixture.step()
    assert [len(xt) for xt, _, _ in calls] == [64, 64, 64, 512]
    xt, z, t = calls[-1]
    torch.testing.assert_close(xt.reshape(64, 8, 2), calls[1][0][:, None].expand(64, 8, 2))
    torch.testing.assert_close(t.reshape(64, 8), calls[1][2][:, None].expand(64, 8))
    assert ((z.reshape(64, 8, 4)[:, 1:] - z.reshape(64, 8, 4)[:, :1]).abs().sum(-1) > 0).all()
