"""Software oracle for R1/R2 updates, not a scientific quality experiment.

The reference spells out the pinned R3GAN Trainer.py scalar equations and
uses native Adam directly, without repository loss, penalty or schedule helpers.
It adapts only the input shape and adds the task's learned MoG parameter group.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from particlegan import GANTrainer, MoGParticlePrior, Recipe
from particlegan.init import deterministic_orthogonal_


def _assert_tensors_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0., atol=2e-12)


def test_public_r1r2_updates_match_independent_native_adam_oracle():
    # Constructor draws are isolated and replaced by the public initializer.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        generator = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 2)).double()
        critic = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 1)).double()
        prior = MoGParticlePrior(8, 2, sigma=.1, standardize=False, dtype=torch.float64)
    deterministic_orthogonal_(generator, seed=0)
    deterministic_orthogonal_(critic, seed=1)
    deterministic_orthogonal_(prior, seed=0)
    recipe = Recipe.from_dict(dict(
        optimizer_family="adam", reg_arm="a_r1r2", num_particles=8, z_dim=2,
        batch_size=4, total_steps=20, prior_kind="mog", standardize=False,
        lr=.0085, d_lr_mult=1.5, prior_lr_mult=2., betas=(0., .9),
        beta2_end=.99, beta2_anneal_end=.2, reg_coeff=1., reg_coeff_end=.1,
        reg_coeff_anneal_end=.2, reg_every=1, reg_anchor_weight=0., d_guard_ratio=0.,
        latent_damping_max_rate=0., direct_particle_gain=False, input_noise_std=0.,
        output_noise_std=0., prior_reg=0., prior_regularizer="none", ema_decay=0.,
        lr_anneal_start=.2, lr_floor=.1, network_lr_floor=.1, network_lr_horizon_cap=None))
    trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=0)
    expected_g, expected_d, expected_prior = map(deepcopy, (generator, critic, prior))
    expected_opt_g = torch.optim.Adam([
        {"params": expected_g.parameters(), "lr": .0085},
        {"params": expected_prior.parameters(), "lr": .017}], betas=(0., .9), eps=1e-8)
    expected_opt_d = torch.optim.Adam(expected_d.parameters(), lr=.01275, betas=(0., .9), eps=1e-8)
    index_rng, noise_rng = torch.Generator(), torch.Generator()
    index_rng.set_state(trainer.latent_generator.get_state())
    noise_rng.set_state(trainer.prior_noise_generator.get_state())
    data_rng = torch.Generator().manual_seed(0)
    critic_after_d = []
    trainer.opt_d.register_step_post_hook(
        lambda *_: critic_after_d.append([p.detach().clone() for p in critic.parameters()]))

    # Seven independent one-update comparisons span the midpoint, endpoint and
    # hold of beta2/gamma, and the beginning of the separate LR anneal.
    for step in range(7):
        real = torch.randn(4, 2, generator=data_rng, dtype=torch.float64)
        generator_real = torch.randn(4, 2, generator=data_rng, dtype=torch.float64) + .7
        weight = .5 * (1. + math.cos(math.pi * min(1., step / 4.)))
        beta2, gamma = .99 + (.9 - .99) * weight, .1 + .9 * weight
        lr_fraction = min(1., max(0., (step - 4.) / 16.))
        lr_scale = .1 + .9 * .5 * (1. + math.cos(math.pi * lr_fraction))
        for optimizer in (expected_opt_g, expected_opt_d):
            for group in optimizer.param_groups:
                group["betas"] = (0., beta2)
        expected_opt_g.param_groups[0]["lr"] = .0085 * lr_scale
        expected_opt_g.param_groups[1]["lr"] = .017 * lr_scale
        expected_opt_d.param_groups[0]["lr"] = .0085 * 1.5 * lr_scale

        with torch.autograd.set_multithreading_enabled(False):
            with torch.no_grad():
                latent, _ = expected_prior.sample(4, generator=index_rng, noise_generator=noise_rng)
                fake = expected_g(latent)
            real_input = real.detach().clone().requires_grad_(True)
            fake_input = fake.detach().requires_grad_(True)
            real_scores, fake_scores = expected_d(real_input), expected_d(fake_input)
            real_gradient, = torch.autograd.grad(real_scores.sum(), real_input, create_graph=True)
            fake_gradient, = torch.autograd.grad(fake_scores.sum(), fake_input, create_graph=True)
            penalty = gamma / 2. * (real_gradient.square().sum(1).mean()
                                    + fake_gradient.square().sum(1).mean())
            loss_d = F.softplus(fake_scores - real_scores).mean() + penalty
            expected_opt_d.zero_grad()
            loss_d.backward()
            expected_opt_d.step()

            expected_d.requires_grad_(False)
            # Draw fresh differentiable fakes and evaluate the updated critic.
            latent, _ = expected_prior.sample(4, generator=index_rng, noise_generator=noise_rng)
            fake_scores = expected_d(expected_g(latent))
            real_scores = expected_d(generator_real)
            loss_g = F.softplus(real_scores - fake_scores).mean()
            expected_opt_g.zero_grad()
            loss_g.backward()
            expected_opt_g.step()
            expected_d.requires_grad_(True)

        result = trainer.step(real, generator_real=generator_real)
        for actual, expected in ((result["loss_d"], loss_d), (result["loss_g"], loss_g),
                                 (result["penalty"], penalty)):
            _assert_tensors_close(actual, expected)
        for actual_module, expected_module in ((generator, expected_g), (critic, expected_d),
                                                (prior, expected_prior)):
            for actual, expected in zip(actual_module.parameters(), expected_module.parameters()):
                _assert_tensors_close(actual, expected)
                _assert_tensors_close(actual.grad, expected.grad)
        # D cannot move during the generator/prior phase.
        assert all(torch.equal(saved, parameter) for saved, parameter in
                   zip(critic_after_d[-1], critic.parameters()))
        assert all(parameter.requires_grad for parameter in critic.parameters())
        for actual_optimizer, expected_optimizer in ((trainer.opt_g, expected_opt_g),
                                                       (trainer.opt_d, expected_opt_d)):
            assert type(actual_optimizer) is torch.optim.Adam
            for actual_group, expected_group in zip(actual_optimizer.param_groups, expected_optimizer.param_groups):
                assert actual_group["betas"] == expected_group["betas"]
                assert actual_group["lr"] == expected_group["lr"]
                for actual_parameter, expected_parameter in zip(actual_group["params"], expected_group["params"]):
                    actual_state, expected_state = (actual_optimizer.state[actual_parameter],
                                                     expected_optimizer.state[expected_parameter])
                    assert actual_state.keys() == expected_state.keys()
                    for name in expected_state:
                        _assert_tensors_close(actual_state[name], expected_state[name])
        assert trainer.completed_steps == trainer.opt_d.record.observed_steps == step + 1
        assert trainer.penalty.regularizer.coeff == gamma
        assert torch.equal(index_rng.get_state(), trainer.latent_generator.get_state())
        assert torch.equal(noise_rng.get_state(), trainer.prior_noise_generator.get_state())
    assert not torch.equal(prior.z, deterministic_orthogonal_(deepcopy(expected_prior), seed=0).z)
