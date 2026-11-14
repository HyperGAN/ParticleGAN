"""A caller-owned PyTorch loop using only Torch and the installed particlegan.

CPU smoke: python -u examples/pytorch_loop.py --steps 5 --batch-size 16
TOML:     python -u examples/pytorch_loop.py --config examples/api.toml

This small MLP demonstrates integration using the recommended defaults --
the same update GANTrainer performs, with the control flow in your hands:

* role-wise LR schedule: ``learning_rate_scales`` (network horizon + prior),
* critic input noise (``InputNoise``) / generator output noise (annealed, one stream),
* ``recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))``:
  recipe-owned optimizers, currently EMPF DualNorm with direction blending,
* sampled prior rows and protected generator loss bound before the G step,
* ``recipe.make_critic_penalty(opt_d)``: the critic penalty, added to the
  critic loss like any other term.

To checkpoint, save the modules and both optimizers' ``state_dict()`` (they
carry the EMA critic and every counter). With several critics, build one
``recipe.make_critic_optimizer(D_k, ema_critic=...)`` and one penalty per
critic, and call ``particlegan.init.deterministic_orthogonal_(D_k, seed=k)``
on a fresh extra critic first (seeds 0/1/2 are the examples' G/D/E). Replace
the networks and synthetic batches with your own.
"""

import argparse
import copy
import json
import time

import torch
from torch import nn

from particlegan import InputNoise, get_recipe, init, learning_rate_scales, scale_learning_rates
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.training import input_noise_std, output_noise_std


@torch.no_grad()
def update_ema(average, current, decay):
    for target, source in zip(average.parameters(), current.parameters()):
        target.lerp_(source, 1.0 - decay)
    for target, source in zip(average.buffers(), current.buffers()):
        target.copy_(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="TOML file with a [particlegan] section")
    parser.add_argument("--steps", type=int, help="Override the recipe's training horizon")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")

    options = {}
    if args.config:
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10: pip install tomli
            import tomli as tomllib
        with open(args.config, "rb") as stream:
            options = tomllib.load(stream)["particlegan"]
    if args.steps is not None:
        options["total_steps"] = args.steps
    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    recipe = get_recipe(**options)
    if (recipe.model != "gan" or recipe.conditioning != "scalar"
            or recipe.encoder_mode != "none"):
        parser.error("this one-shot example requires model='gan', conditioning='scalar', encoder_mode='none'")

    device = torch.device(args.device)
    # These choices belong to this application, not the library.
    torch.manual_seed(42)
    if device.type == "cpu":
        torch.set_num_threads(1)
    generator = nn.Sequential(
        nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 2),
    ).to(device)
    critic = nn.Sequential(
        nn.Linear(2, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1),
    ).to(device)
    prior = recipe.make_prior().to(device)
    # Deterministic initial weights and R2 particle table; optimizers keep weights as given.
    init.deterministic_orthogonal_(generator, seed=0)
    init.deterministic_orthogonal_(critic, seed=1)
    init.deterministic_orthogonal_(prior)
    gan = recipe.make_loss()
    # Joint generator/prior optimizer and critic optimizer, selected by the recipe.
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, ema_critic=copy.deepcopy(critic))
    # Optimizer construction binds the initialized prior's reference buffers.
    spread = recipe.make_prior_regularizer(prior=prior)
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    penalty = recipe.make_critic_penalty(opt_d)
    noise = torch.Generator(device=device).manual_seed(43)
    noisy_critic = InputNoise(critic, generator=noise)  # fresh input noise per evaluation

    def with_noise(x, sigma):
        if sigma == 0:
            return x
        return x + sigma * torch.randn(x.shape, generator=noise, device=x.device, dtype=x.dtype)
    ema_g = copy.deepcopy(generator).eval().requires_grad_(False)
    ema_prior = copy.deepcopy(prior).eval().requires_grad_(False)

    axis = torch.linspace(-1.0, 1.0, 10, device=device)
    centers = torch.cartesian_prod(axis, axis)
    print(json.dumps({"event": "config", **recipe.to_dict(), "device": str(device)}), flush=True)
    started = time.monotonic()
    for step in range(1, recipe.total_steps + 1):
        network, prior_scale = learning_rate_scales(step - 1, recipe)
        scale_learning_rates(step - 1, recipe, (opt_g, opt_d), base_lrs, prior)
        noisy_critic.std = input_noise_std(recipe, step - 1)
        sigma_out = output_noise_std(recipe, step - 1)

        # Replace this synthetic batch with a batch from your DataLoader.
        ids = torch.randint(len(centers), (recipe.batch_size,), device=device)
        real = centers[ids] + 0.015 * torch.randn(recipe.batch_size, 2, device=device)
        z, particle_ids = prior.sample(recipe.batch_size)
        fake = with_noise(generator(z), sigma_out)

        d_loss = gan.d_loss(noisy_critic(real), noisy_critic(fake.detach()))
        d_loss = d_loss + penalty(noisy_critic, real, fake.detach())
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        # Freeze critic weights while preserving gradients through critic(fake).
        flags = [parameter.requires_grad for parameter in critic.parameters()]
        critic.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            g_loss = gan.g_loss(noisy_critic(fake), noisy_critic(real).detach())
            prior_loss = spread(prior.z[particle_ids.unique()])
            total_g = g_loss + prior_loss  # spread already carries recipe.prior_reg
            if hasattr(opt_g, "set_sampled_rows"):
                opt_g.set_sampled_rows(prior.z, particle_ids)
            constraint_geometry_backward(total_g, opt_g, (g_loss,))
            opt_g.step()
        finally:
            for parameter, flag in zip(critic.parameters(), flags):
                parameter.requires_grad_(flag)
        update_ema(ema_g, generator, recipe.ema_decay)
        update_ema(ema_prior, prior, recipe.ema_decay)

        if step == 1 or step % args.log_every == 0 or step == recipe.total_steps:
            print(json.dumps({
                "event": "train", "step": step,
                "d_loss": d_loss.detach().item(), "g_loss": g_loss.detach().item(),
                "prior_loss": prior_loss.detach().item(), "lr_scale": network,
                **penalty.diagnostics(),
                "seconds": round(time.monotonic() - started, 3),
            }), flush=True)

    with torch.no_grad():
        z, _ = ema_prior.sample(256)
        samples = ema_g(z)
    if not torch.isfinite(samples).all():
        raise RuntimeError("non-finite generated samples")
    print(json.dumps({"event": "complete", "sample_shape": list(samples.shape)}), flush=True)


if __name__ == "__main__":
    main()
