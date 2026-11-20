"""Fixed-arm weighting contracts; tiny software fixtures, not quality runs."""
import pytest
import torch
from torch import nn

from particlegan import GANTrainer, MoGParticlePrior, Recipe, get_recipe
from particlegan.grad_regularizers import GradientPenalty
from particlegan.init import deterministic_orthogonal_
from experiments.forge.state import state_digest


class PolynomialCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.3, -.7], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([.2, .4], dtype=torch.float64))

    def forward(self, x):
        return (self.weight * x.square() + self.bias * x).sum(1)


def recipe(**changes):
    return Recipe.from_dict(dict(optimizer_family="adam", reg_arm="a_r1r2", num_particles=8, z_dim=2,
                  batch_size=4, total_steps=8, prior_kind="mog", standardize=False,
                  reg_anchor_weight=0., d_guard_ratio=0., latent_damping_max_rate=0.,
                  direct_particle_gain=False, input_noise_std=0., output_noise_std=0.,
                  prior_reg=0., prior_regularizer="none", ema_decay=0., **changes))


def trainer(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        generator = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 2)).double()
        critic = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 1)).double()
        prior = MoGParticlePrior(8, 2, sigma=.1, standardize=False, dtype=torch.float64)
    for component, seed in ((generator, 0), (critic, 1), (prior, 0)):
        deterministic_orthogonal_(component, seed=seed)
    return GANTrainer(config, generator, critic, prior=prior, seed=0)


@pytest.mark.parametrize("weights", [(1., 0.), (0., 1.), (.25, 2.)])
@pytest.mark.parametrize("arm", ["a_r1r2", "b_cap"])
def test_weighted_nonlinear_penalty_and_parameter_gradients(weights, arm):
    critic = PolynomialCritic()
    real = torch.tensor([[-.8, .4], [.2, -.1], [.7, 1.2]], dtype=torch.float64, requires_grad=True)
    fake = (real.detach() + .6).requires_grad_()
    coefficient, kappa = 3., .15
    penalty = GradientPenalty(arm=arm, coeff=coefficient, kappa=kappa,
                              real_weight=weights[0], fake_weight=weights[1])
    actual = penalty(critic, real, fake)
    # Differentiate the explicit polynomial input derivative, rather than the
    # implementation's autograd input-gradient helper.
    expected_terms = []
    for weight, batch in zip(weights, (real, fake)):
        derivative = 2 * critic.weight * batch.detach() + critic.bias
        squared_norm = derivative.square().sum(1)
        term = squared_norm if arm == "a_r1r2" else (squared_norm.add(1e-12).sqrt() - kappa).relu().square()
        expected_terms.append(weight * term.mean())
    expected = coefficient / 2 * sum(expected_terms)
    torch.testing.assert_close(actual, expected, rtol=0., atol=1e-14)
    actual_grad = torch.autograd.grad(actual, tuple(critic.parameters()))
    expected_grad = torch.autograd.grad(expected, tuple(critic.parameters()))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=0., atol=1e-14)
    assert real.grad is None and fake.grad is None


@pytest.mark.parametrize("weights,calls", [((1., 0.), 1), ((0., 1.), 1), ((0., 0.), 0)])
def test_inactive_terms_do_not_forward_differentiate_or_draw(weights, calls):
    critic = PolynomialCritic()
    rng, reference_rng = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    visited = []
    def observe(module, inputs):
        visited.append(inputs[0].detach().clone())
        torch.rand((), generator=rng)
    critic.register_forward_pre_hook(observe)
    real = torch.ones(4, 2, dtype=torch.float64)
    fake = torch.full_like(real, 2.)
    penalty = GradientPenalty(arm="a_r1r2", coeff=2., lazy_k=3,
                              real_weight=weights[0], fake_weight=weights[1])
    skipped, stats = penalty.penalty(critic, real, fake, step=1)
    assert not stats["applied"] and skipped.item() == 0 and not visited
    actual = penalty(critic, real, fake, step=3)
    assert len(visited) == calls
    for _ in range(calls):
        torch.rand((), generator=reference_rng)
    assert torch.equal(rng.get_state(), reference_rng.get_state())
    if calls:
        expected_inputs = real if weights[0] else fake
        assert torch.equal(visited[0], expected_inputs)
        expected = 3 * (2 * critic.weight * expected_inputs + critic.bias).square().sum(1).mean()
        torch.testing.assert_close(actual, expected)
    else:
        assert actual.item() == 0


@pytest.mark.parametrize("name", ["reg_real_weight", "reg_fake_weight"])
@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True, "0"])
def test_invalid_weights_rejected_at_recipe_and_kernel_boundary(name, value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        recipe(**{name: value})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GradientPenalty(arm="a_r1r2", **{name.removeprefix("reg_"): value})


def test_nonfixed_formulations_reject_unsupported_weighting():
    for name in ("ka2", "k3p", "bispf"):
        with pytest.raises(ValueError, match="fixed"):
            get_recipe(name, reg_fake_weight=0.)
    with pytest.raises(ValueError, match="fixed"):
        GradientPenalty(arm="k3p", fake_weight=0.)


def test_default_identity_and_paired_training_are_exactly_unchanged():
    for name in ("ka2", "k3p", "e22", "mog", "bcap_adam"):
        config = get_recipe(name)
        assert "reg_real_weight" not in config.to_dict()
        assert "reg_fake_weight" not in config.to_dict()
    left = trainer(recipe())
    right = trainer(recipe(reg_real_weight=1., reg_fake_weight=1.))
    batch = torch.arange(8, dtype=torch.float64).reshape(4, 2) / 7
    for _ in range(3):
        left.step(batch)
        right.step(batch)
    assert state_digest(left.state_dict()) == state_digest(right.state_dict())


def test_r1_only_factory_roundtrip_resume_and_recipe_mismatch_are_strict():
    config = recipe(reg_fake_weight=0., reg_coeff=2., reg_coeff_end=.2)
    assert Recipe.from_dict(config.to_dict()) == config
    assert config.to_dict()["reg_fake_weight"] == 0.
    current = trainer(config)
    assert current.penalty.regularizer.real_weight == 1.
    assert current.penalty.regularizer.fake_weight == 0.
    batch = torch.arange(8, dtype=torch.float64).reshape(4, 2) / 7
    current.step(batch)
    checkpoint = current.state_dict()
    for _ in range(3):
        current.step(batch)
    restored = trainer(config)
    restored.load_state_dict(checkpoint)
    for _ in range(3):
        restored.step(batch)
    assert state_digest(current.state_dict()) == state_digest(restored.state_dict())
    incompatible = trainer(config.replace(reg_fake_weight=1.))
    unchanged = state_digest(incompatible.state_dict())
    with pytest.raises(ValueError, match="recipe"):
        incompatible.load_state_dict(checkpoint)
    assert state_digest(incompatible.state_dict()) == unchanged
