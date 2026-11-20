"""Constrained particle VAE research toys using public recipe components.

This is a new seed-0, two-mode cohort. The historical 100-mode VAE studies
remain evidence for their own initialization, optimizer and sampling contracts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from particlegan import Recipe, get_recipe, init
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.policy import UpdatePolicy
from particlegan.training import InputNoise
from .reproducibility import construction_rng


ROOT = Path(__file__).resolve().parents[2]
VERSION = "api-model-vae-v1"
BASELINE = "bispf-optimized-quintic-v1"
STEPS, BATCH, PARTICLES, Z_DIM = 1200, 64, 16, 2
PRIOR_SIGMA, DATA_SIGMA, OBSERVATION_SIGMA = .025, .05, .03
ANCHORS = ((-1.5, 0.), (1.5, 0.))
THRESHOLDS = [
    ["reconstruction_mse", "<=", .05],
    ["shuffle_reconstruction_ratio", ">=", 5.],
    ["posterior_mutual_information", ">=", .3],
    ["posterior_noise_variance_ratio", ">=", .8],
    ["posterior_noise_variance_ratio", "<=", 1.2],
    ["kl_identity_error", "<=", 1e-5],
    ["elbo_identity_error", "<=", 1e-4],
    ["decoder_mean_hq", ">=", .85],
    ["decoder_mean_mass_tv", "<=", .15],
    ["decoder_mean_modes", ">=", 2],
    ["likelihood_hq", ">=", .85],
    ["likelihood_mass_tv", "<=", .15],
    ["likelihood_modes", ">=", 2],
]


def list_cases():
    result = []
    for mode in ("hard", "categorical"):
        result.append(dict(
            id=f"api-model-vae-{mode}", title=f"{mode.capitalize()} particle VAE: two noisy anchors",
            goal="Can a genuine particle VAE retain the input and posterior noise while independently generating both target modes?",
            kind="vector", legacy_ids=["mog-vae-scout", "mog-vae-stability"],
            scope="New constrained two-mode seed-0 public-component cohort; does not reproduce or qualify historical 100-mode results. Hard routing has constant log(K) KL and biased straight-through gradients; categorical routing uses the public leave-one-out score estimator and a variable KL. A stochastic no-KL AE is not a VAE.",
            default_steps=STEPS, batch_size=BATCH, eval_samples=4096,
            evaluation_observations=24, terminal_observations=5, default_recipe=BASELINE,
            encoder_mode=mode, protocol_seed=0, thresholds=deepcopy(THRESHOLDS), forge_thresholds=deepcopy(THRESHOLDS),
            sampling="Independent uniform learned MoG prior with fixed latent sigma=.025; live decoder means and separate Gaussian likelihood draws G(z)+.03*epsilon. Reconstructions use eight posterior draws with prior-matching local Gaussian noise.",
            initialization="particlegan.init.deterministic_orthogonal_: E seed0, G seed1, D seed2, prior R2 seed3; isolated constructor scope seed0",
            target=dict(anchors=ANCHORS, weights=[.5, .5], sigma=DATA_SIGMA),
            architecture=dict(encoder=[2, 32, 2], decoder=[2, 32, 2], critic=[2, 32, 1], activation="LeakyReLU(.2)"),
            prior=dict(kind="mog", sigma=PRIOR_SIGMA, standardize=False, num_particles=PARTICLES, z_dim=Z_DIM),
        ))
    return result


def _rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def sample_target(n, rng, device):
    anchors = torch.tensor(ANCHORS, device=device)
    ids = torch.randint(2, (n,), device=device, generator=rng)
    return anchors[ids] + DATA_SIGMA * torch.randn(n, 2, device=device, generator=rng)


def _hash_model(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes() if isinstance(tensor, torch.Tensor)
                      else json.dumps(tensor, sort_keys=True).encode())
    return digest.hexdigest()


def _distribution_metrics(samples, prefix):
    anchors = samples.new_tensor(ANCHORS)
    distances, ids = torch.cdist(samples, anchors).min(1)
    good = distances <= .2
    mass = torch.bincount(ids[good], minlength=2).to(samples.dtype) / len(samples)
    return {f"{prefix}_hq": float(good.float().mean()),
            f"{prefix}_mass_tv": float(((mass - .5).abs().sum() + 1 - mass.sum()) / 2),
            f"{prefix}_modes": int((mass >= .1).sum())}


def score_outputs(*, data, reconstruction, shuffled, posterior_codes, posterior_centers,
                  log_probs, kl, negative_elbo, decoder_means, likelihood):
    """Score actual posterior and prior outputs; destructive controls share this path."""
    # Metrics promote the already-observed values, never model execution.
    # Float32 ELBO terms can be thousands of nats before learning; cancellation
    # at that scale must not spuriously fail the fixed 1e-4 identity gate.
    recon = (reconstruction.double() - data[:, None].double()).square().mean()
    shuffled_mse = (shuffled.double() - data[:, None].double()).square().mean()
    q = log_probs.double().exp()
    entropy = -(q * log_probs.double().nan_to_num(neginf=0.)).sum(1)
    aggregate = q.mean(0)
    aggregate_entropy = -(aggregate * aggregate.clamp_min(1e-30).log()).sum()
    expected_kl = math.log(log_probs.shape[1]) - entropy
    # Independent Gaussian log-density reference for the public ELBO value.
    expected_elbo = (-torch.distributions.Normal(reconstruction.double(), OBSERVATION_SIGMA)
                     .log_prob(data[:, None].double()).sum(-1).mean() + kl.double().mean())
    metrics = dict(
        reconstruction_mse=float(recon), shuffled_reconstruction_mse=float(shuffled_mse),
        shuffle_reconstruction_ratio=float(shuffled_mse / recon.clamp_min(1e-12)),
        posterior_mutual_information=float(aggregate_entropy - entropy.mean()),
        posterior_noise_variance_ratio=float((posterior_codes - posterior_centers).square().mean() / PRIOR_SIGMA ** 2),
        posterior_pair_rms=float((reconstruction[:, 0] - reconstruction[:, 1]).square().mean().sqrt()),
        posterior_effective_components=float(entropy.mean().exp()),
        aggregate_effective_components=float(aggregate_entropy.exp()),
        kl_nats=float(kl.mean()), negative_elbo_nats=float(negative_elbo),
        kl_identity_error=float((kl - expected_kl).abs().max()),
        elbo_identity_error=float((negative_elbo.double() - expected_elbo).abs()),
        **_distribution_metrics(decoder_means, "decoder_mean"),
        **_distribution_metrics(likelihood, "likelihood"),
    )
    failed = []
    for name, op, bound in THRESHOLDS:
        value = metrics[name]
        if not math.isfinite(value) or not ({"<=": value <= bound, ">=": value >= bound}[op]):
            failed.append(f"{name} {op} {bound:g} (observed {value:g})")
    for name, value in metrics.items():
        if not math.isfinite(value):
            failed.append(f"{name}: nonfinite")
    return metrics, failed


def observed_negative_elbo(encoded, reconstruction, data):
    """Public ELBO on float64 copies of actual float32 observations.

    No forward pass or draw is repeated. The training objective keeps its
    original dtype; this promotion is solely a numerical identity diagnostic.
    """
    diagnostic = replace(encoded, codes=encoded.codes.double(), kl=encoded.kl.double(),
                         log_probs=None if encoded.log_probs is None else encoded.log_probs.double())
    return diagnostic.negative_elbo(reconstruction.double(), data.double(), observation_sigma=OBSERVATION_SIGMA)


def task_recipe(case, recipe_name=None, recipe=None):
    """Adapt a named recipe once; validate an already-resolved Forge injection."""
    fields = dict(model="gan", conditioning="scalar", encoder_mode=case["encoder_mode"],
                  observation_sigma=OBSERVATION_SIGMA, prior_kind="mog", sigma_rel=0., standardize=False,
                  num_particles=PARTICLES, z_dim=Z_DIM, batch_size=BATCH)
    if recipe is not None:
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe injection must be a resolved public Recipe")
        for name, expected in fields.items():
            if getattr(recipe, name) != expected:
                raise ValueError(f"resolved Recipe contradicts task-owned {name}")
        if recipe.total_steps not in (None, case["default_steps"]):
            raise ValueError("resolved Recipe contradicts task-owned schedule horizon")
        result = recipe
    else:
        if recipe_name in (None, BASELINE):
            card = json.loads((ROOT / f"configs/forge/ideas/{BASELINE}.json").read_text())
            base = Recipe.from_dict(card["recipe_overrides"])
        else:
            base = get_recipe(recipe_name)
        result = base.replace(**fields, total_steps=None if base.total_steps is None else case["default_steps"])
    from .api_contract import validate_caller_objectives
    validate_caller_objectives(result, provider="api_model_vae")
    if result.constraint_geometry_mode == "strict_progress":
        raise ValueError("VAE stochastic posterior has no strict_progress finite-replay contract")
    return result


class VAEFixture:
    api_components = ("particlegan.Recipe", "Recipe.encode", "ParticleEncoding.negative_elbo",
                      "Recipe.make_optimizers", "Recipe.make_critic_penalty", "Recipe.prior_regularization",
                      "particlegan.policy.UpdatePolicy", "particlegan.MoGParticlePrior")

    def __init__(self, case, *, device, seed, recipe_name, max_steps, recipe):
        if seed != 0 or type(seed) is not int:
            raise ValueError("VAE comparison uses protocol seed 0 only")
        self.case, self.device = deepcopy(case), torch.device(device)
        self.limit = case["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0 < self.limit <= case["default_steps"]:
            raise ValueError("execution cap must be within the frozen task allowance")
        self.recipe = task_recipe(case, recipe_name=recipe_name, recipe=recipe)
        with construction_rng(seed, self.device):
            def mlp(out):
                return nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(.2), nn.Linear(32, out)).to(self.device)
            self.E, self.G, self.D = mlp(2), mlp(2), mlp(1)
            # Initialize the same trainable R2 table before applying the
            # recipe's freeze policy in make_optimizers. Frozen buffers would
            # otherwise retain constructor draws instead of this shared law.
            self.prior = self.recipe.replace(prior_update="learned").make_prior(sigma=PRIOR_SIGMA).to(self.device)
            self.constructor_rng = torch.get_rng_state().clone()
            self.constructor_cuda_rng = (torch.cuda.get_rng_state(self.device).clone()
                                         if self.device.type == "cuda" else None)
        for offset, module in enumerate((self.E, self.G, self.D, self.prior)):
            init.deterministic_orthogonal_(module, seed=seed + offset)
        self.initial_model_hashes = {name: _hash_model(getattr(self, name)) for name in ("E", "G", "D", "prior")}
        self.opt_g, self.opt_d = self.recipe.make_optimizers(self.G, self.D, self.prior, encoder=self.E,
                                                           ema_critic=deepcopy(self.D),
                                                           separate_encoder_groups=True, foreach=False)
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.loss = self.recipe.make_loss()
        self.prior_regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        self.streams = {name: _rng(self.device, seed + offset) for name, offset in
                        (("data_d", 10), ("data_g", 11), ("latent", 12), ("prior_noise", 13),
                         ("posterior", 14), ("noise", 15), ("input_noise", 16), ("eval", 17), ("penalty", 18))}
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, encoder=self.E, prior=self.prior,
                                   generator_optimizer=self.opt_g, critic_optimizer=self.opt_d, penalty=self.penalty,
                                   streams={"latent_generator": self.streams["latent"],
                                            "penalty_generator": self.streams["penalty"],
                                            "eval_generator": self.streams["eval"],
                                            "noise_generator": self.streams["noise"]})
        self.critic = InputNoise(self.D, generator=self.streams["input_noise"])
        self.batch_sequence_digest = hashlib.sha256(b"").hexdigest()
        self.last = {}

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def _prior_sample(self, n):
        return self.prior.sample(n, generator=self.streams["latent"], noise_generator=self.streams["prior_noise"])

    def step(self):
        if self.completed_steps >= self.limit:
            raise RuntimeError("frozen VAE execution allowance exhausted")
        with torch.autograd.set_multithreading_enabled(False):
            try:
                return self._step()
            except Exception:
                self.policy.abort_step()
                raise

    def _step(self):
        real = sample_target(BATCH, self.streams["data_d"], self.device)
        real_g = sample_target(BATCH, self.streams["data_g"], self.device)
        digest = hashlib.sha256(bytes.fromhex(self.batch_sequence_digest))
        for batch in (real, real_g):
            digest.update(batch.cpu().numpy().tobytes())
        self.batch_sequence_digest = digest.hexdigest()
        noise = self.policy.begin_step(real, game_record=self.penalty.regularizer.record, execution_limit=self.limit)
        self.critic.std = noise.input_sigma
        with torch.no_grad():
            latent, rows_d = self._prior_sample(BATCH)
            self.policy.observe_support(latent)
            fake_d = self.policy.generate(latent, sigma=noise.output_sigma, rows=rows_d)
        self.policy.observe_critic_pair(real, fake_d)
        loss_d = self.loss.d_loss(self.critic(real), self.critic(fake_d))
        penalty = self.penalty(self.critic, real, fake_d)
        self.opt_d.zero_grad(set_to_none=True)
        (loss_d + penalty).backward()
        self.opt_d.step()
        self.policy.after_critic_step()
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            latent, rows = self._prior_sample(BATCH)
            generated = self.policy.generate(latent, sigma=noise.output_sigma, rows=rows)
            with torch.no_grad():
                real_logits = self.critic(real_g)
            adversarial = self.loss.g_loss(self.critic(generated), real_logits)
            encoded = self.recipe.encode(self.E(real_g), self.prior, draws=2, generator=self.streams["posterior"])
            # Public row ownership includes actual posterior samples, not every
            # row with a soft-routing or whole-table regularizer gradient.
            self.policy.observe_sampled_rows(encoded.indices.reshape(-1))
            reconstruction = self.policy.generate(encoded.codes.flatten(0, 1), sigma=noise.output_sigma).reshape(BATCH, 2, 2)
            elbo = encoded.negative_elbo(reconstruction, real_g, observation_sigma=OBSERVATION_SIGMA)
            # Positive rescaling preserves the Gaussian ELBO objective; in 2D
            # its reconstruction term is exactly MSE, matching AE task units.
            variational = OBSERVATION_SIGMA ** 2 * elbo
            prior_loss = self.recipe.prior_regularization(self.prior.z, regularizer=self.prior_regularizer, prior=self.prior)
            objective = adversarial + self.recipe.reconstruction_weight * variational + prior_loss
            if self.recipe.kinetic_transport_weight:
                objective = objective + self.recipe.kinetic_transport_loss(generated, real_g)
            if self.recipe.kinetic_transport_local_weight:
                objective = objective + self.recipe.kinetic_transport_local_loss(generated, real_g)
            self.opt_g.zero_grad(set_to_none=True)
            constraint_geometry_backward(objective, self.opt_g, (adversarial, variational))
            self.policy.after_generator_backward(loss_gan=adversarial.detach(), loss_critic=loss_d.detach())
            self.opt_g.step()
            self.policy.after_generator_step()
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
        self.policy.finish_step()
        self.last = dict(step=self.completed_steps, loss_d=float(loss_d.detach()), loss_g=float(objective.detach()),
                         negative_elbo_nats=float(elbo.detach()), prior_regularization=float(prior_loss.detach()))
        return deepcopy(self.last)

    @torch.no_grad()
    def observe(self, n=4096, seed=713):
        if n < 2:
            raise ValueError("VAE observation needs at least two held-out inputs")
        rng = _rng(self.device, seed)
        data = sample_target(n, rng, self.device)
        encoded = self.recipe.encode(self.E(data), self.prior, draws=8, generator=rng)
        reconstruction = self.G(encoded.codes.flatten(0, 1)).reshape(n, 8, 2)
        shuffled = reconstruction[torch.randperm(n, device=self.device, generator=rng)]
        latent, _ = self.prior.sample(n, generator=rng, noise_generator=rng)
        decoder_means = self.G(latent)
        likelihood = decoder_means + OBSERVATION_SIGMA * torch.randn(n, 2, device=self.device, generator=rng)
        elbo = observed_negative_elbo(encoded, reconstruction, data)
        metrics, failed = score_outputs(data=data, reconstruction=reconstruction, shuffled=shuffled,
                                       posterior_codes=encoded.codes, posterior_centers=self.prior.means()[encoded.indices],
                                       log_probs=encoded.log_probs, kl=encoded.kl, negative_elbo=elbo,
                                       decoder_means=decoder_means, likelihood=likelihood)
        def view(title, targets, samples):
            return dict(kind="scatter", title=title, target=targets.cpu().numpy(), samples=samples.cpu().numpy(),
                        xlim=[-2.2, 2.2], ylim=[-.7, .7], xlabel="x", ylabel="y")
        return dict(metrics=metrics, passed=not failed, failed_bounds=failed, views=[
            view("Held-out inputs and sampled posterior reconstructions", data, reconstruction[:, 0]),
            view("Independent prior: live decoder means", data, decoder_means),
            view("Gaussian likelihood samples: G(z) + .03 epsilon", data, likelihood),
        ])

    def state_dict(self):
        return deepcopy(dict(version=VERSION, case_id=self.case["id"], limit=self.limit, policy=self.policy.state_dict(),
                             streams={name: stream.get_state() for name, stream in self.streams.items()},
                             constructor_rng=self.constructor_rng, constructor_cuda_rng=self.constructor_cuda_rng,
                             initial_model_hashes=self.initial_model_hashes,
                             batch_sequence_digest=self.batch_sequence_digest, last=self.last,
                             prior_regularizer=self.recipe.prior_regularizer_checkpoint(self.prior_regularizer)))

    @property
    def batch_sequence_sha256(self):
        return self.batch_sequence_digest

    def load_state_dict(self, state):
        if (state.get("version"), state.get("case_id"), state.get("limit")) != (VERSION, self.case["id"], self.limit):
            raise ValueError("incompatible VAE task checkpoint")
        if set(state.get("streams", {})) != set(self.streams):
            raise ValueError("incomplete VAE RNG checkpoint")
        self.policy.load_state_dict(state["policy"])
        for name, stream in self.streams.items():
            stream.set_state(state["streams"][name].cpu())
        self.recipe.restore_prior_regularizer(self.prior_regularizer, state.get("prior_regularizer", {}).get("prior_regularizer"))
        for name in ("constructor_rng", "constructor_cuda_rng", "initial_model_hashes", "batch_sequence_digest", "last"):
            setattr(self, name, deepcopy(state[name]))


def build_case(case_id, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe=None):
    case = next((case for case in list_cases() if case["id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown VAE case: {case_id}")
    return VAEFixture(case, device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps, recipe=recipe)
