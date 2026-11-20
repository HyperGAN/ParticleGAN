"""Frozen multimodal particle-AE callers using shared public training components.

The reconstruction control has no adversarial objective. AE-GAN additionally
learns independent prior generation; good reconstructions cannot satisfy its
prior gates. Historical ae_gan_hold and api-ae-anchor-hold are separate cohorts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from particlegan import Recipe, UpdatePolicy, get_recipe, init
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.training import InputNoise
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from .reproducibility import construction_rng

ROOT = Path(__file__).resolve().parents[2]
VERSION = "api-model-ae-v1"
DEFAULT_RECIPE = "bispf-optimized-quintic-v1"
CENTERS = ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))
TARGET_SIGMA = .08
CASES = {"api-model-ae-reconstruction": "model_ae_reconstruction",
         "api-model-ae-gan-multimodal": "model_ae_gan_multimodal"}
RECON_BOUNDS = (("reconstruction_mse", "<=", .025), ("reconstruction_mode_agreement", ">=", .98))
FROZEN_RESOURCES = dict(num_particles=64, z_dim=2, batch_size=128)
FROZEN_RECIPE_FIELDS = dict(model="gan", conditioning="scalar", encoder_mode="ae",
                            **FROZEN_RESOURCES, prior_kind="mog", sigma_rel=0.,
                            standardize=False, reconstruction_weight=1.)
GENERATION_BOUNDS = (("modes", ">=", 4), ("precision", ">=", .90),
                     ("mass_tv", "<=", .15), ("component_min_eigen_ratio", ">=", .10),
                     ("component_max_eigen_ratio", "<=", 4.), ("component_center_rms_sigma", "<=", 1.))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def list_cases():
    result = []
    for case_id, task_id in CASES.items():
        task = json.loads((ROOT / f"configs/forge/tasks/{task_id}.json").read_text())
        adversarial = case_id == "api-model-ae-gan-multimodal"
        execution, evaluation = task["execution"], task["evaluation"]
        host = execution["host_definition"]
        bounds = [list(bound) for bound in RECON_BOUNDS + (GENERATION_BOUNDS if adversarial else ())]
        expected_target = dict(kind="gaussian_mixture", means=[list(center) for center in CENTERS],
                               sigma=TARGET_SIGMA, masses=[.25] * 4)
        expected_host = dict(provider="benchmarks.toy_audit.api_model_ae", case_id=case_id,
                             adversarial=adversarial, **FROZEN_RESOURCES, hidden=64, layers=2,
                             fourier=2, init_std=.5, resources=FROZEN_RESOURCES,
                             target=expected_target)
        if evaluation["thresholds"] != bounds:
            raise ValueError(f"{task_id}: task bounds differ from the static AE scorer")
        if (execution["steps"] != 1200 or evaluation["eval_samples"] != 4096
                or evaluation["observations"] != 24 or evaluation["minimum_stable_checks"] != 5
                or execution["initializer"] != "deterministic_orthogonal"
                or execution["prior"] != dict(kind="mog", sigma=.025, standardize=False)
                or any(host.get(key) != value for key, value in expected_host.items())):
            raise ValueError(f"{task_id}: task conditions differ from the frozen AE protocol")
        result.append(dict(
            id=case_id, legacy_ids=["develop-ae_gan_hold", "api-ae-anchor-hold"],
            title="Particle AE: " + ("multimodal reconstruction and prior generation" if adversarial else "reconstruction control"),
            goal=task["description"], kind="vector", default_steps=1200,
            batch_size=128, eval_samples=4096, default_recipe=DEFAULT_RECIPE,
            evaluation_observations=24, terminal_observations=5, protocol_seed=0,
            thresholds=deepcopy(bounds), forge_thresholds=deepcopy(bounds),
            sampling="Clean live G(MoG): 64 uniform learned locations in R2, fixed sigma=.025, standardize=False. Reconstruction uses deterministic public AE routing and bounded local offsets; no output noise, EMA or best-checkpoint selection.",
            scope="Four equally weighted Gaussians at (+/-1,+/-1), sigma=.08; independently drawn target and prior samples. " +
                  ("Reconstruction and all prior gates must pass together." if adversarial else "Explicit reconstruction-only objective control; independent generation is measured but not required."),
            adaptation="New four-mode target, deterministic public initializer, 1200 updates and strict shape/mass gates; preserves the historical two-anchor 250-update source and api-ae-anchor-hold identities.",
            initialization="particlegan.init.deterministic_orthogonal_: G seed0, D seed1, E seed2, prior seed3 R2Normal(init_std=.5); seeds denote isolated components within protocol seed0.",
            host_definition=deepcopy(task["execution"]["host_definition"]),
            task_sha256=hashlib.sha256((ROOT / f"configs/forge/tasks/{task_id}.json").read_bytes()).hexdigest(),
            trainer_configuration="configs/forge/ideas/bispf-optimized-quintic-v1.json"))
    return result


def sample_target(n, rng, *, device="cpu"):
    centers = torch.tensor(CENTERS, device=device)
    return centers[torch.randint(4, (n,), generator=rng, device=device)] + TARGET_SIGMA * torch.randn(n, 2, generator=rng, device=device)


def score_samples(target, reconstructed, samples, *, adversarial=True):
    """Score the paired AE and independent-prior contracts; reject NaNs explicitly."""
    values = [torch.as_tensor(value).detach().float().cpu() for value in (target, reconstructed, samples)]
    if any(value.ndim != 2 or value.shape[1] != 2 or not len(value) for value in values):
        raise ValueError("target, reconstruction and prior samples must be nonempty [N,2]")
    target, reconstructed, samples = values
    if reconstructed.shape != target.shape:
        raise ValueError("reconstruction must preserve the target's paired rows")
    bounds = RECON_BOUNDS + (GENERATION_BOUNDS if adversarial else ())
    if not all(torch.isfinite(value).all() for value in values):
        return dict(metrics={key: float("nan") for key, _, _ in bounds}, passed=False, failed_bounds=["nonfinite_output"])
    centers = torch.tensor(CENTERS)
    ids = torch.cdist(samples, centers).argmin(1)
    distances = (samples - centers[ids]).norm(dim=1)
    mass = torch.bincount(ids, minlength=4).float() / len(samples)
    quality_counts = torch.bincount(ids[distances <= 3 * TARGET_SIGMA], minlength=4)
    eigens, center_errors = [], []
    for index in range(4):
        points = samples[ids == index]
        if len(points) < 10:
            eigens.append(torch.tensor([0., 1e6])); center_errors.append(1e6)
            continue
        centered = points - points.mean(0)
        eigens.append(torch.linalg.eigvalsh(centered.T @ centered / len(points)) / TARGET_SIGMA**2)
        center_errors.append(float((points.mean(0) - centers[index]).square().sum() / TARGET_SIGMA**2))
    eigen = torch.stack(eigens)
    metrics = dict(reconstruction_mse=float((reconstructed - target).square().mean()),
                   reconstruction_mode_agreement=float((torch.cdist(reconstructed, centers).argmin(1) == torch.cdist(target, centers).argmin(1)).float().mean()),
                   modes=int((quality_counts >= max(10, math.ceil(.01 * len(samples)))).sum()),
                   precision=float((distances <= 3 * TARGET_SIGMA).float().mean()),
                   mass_tv=float((mass - .25).abs().sum() / 2),
                   component_min_eigen_ratio=float(eigen.min()), component_max_eigen_ratio=float(eigen.max()),
                   component_center_rms_sigma=math.sqrt(sum(center_errors) / 4), sample_count=len(samples))
    failures = [f"{key}={metrics[key]:.6g} {op} {bound} failed" for key, op, bound in bounds
                if not (metrics[key] <= bound if op == "<=" else metrics[key] >= bound)]
    return dict(metrics=metrics, passed=not failures, failed_bounds=failures)


def task_recipe(case, recipe_name=None, recipe=None):
    """Adapt a named trainer; a resolved Forge recipe must already match the task."""
    injected = recipe is not None
    if recipe is None:
        name = recipe_name if recipe_name not in (None, "auto") else DEFAULT_RECIPE
        if name == DEFAULT_RECIPE:
            candidate = json.loads((ROOT / f"configs/forge/ideas/{DEFAULT_RECIPE}.json").read_text())
            recipe = Recipe.from_dict(candidate["recipe_overrides"])
        else:
            recipe = get_recipe(name)
    if not isinstance(recipe, Recipe):
        raise TypeError("recipe injection must be a public Recipe")
    conflicts = [key for key, value in FROZEN_RECIPE_FIELDS.items() if getattr(recipe, key) != value]
    if recipe.total_steps not in (None, 1200):
        conflicts.append("total_steps")
    if injected and conflicts:
        raise ValueError(f"injected recipe conflicts with task-owned AE fields: {', '.join(conflicts)}")
    # Named recipes adapt task fields only. All prior policy, regularization,
    # rates, losses and optimizer remain recipe-owned. Scheduled trainers bind
    # the task horizon; intentionally schedule-free trainers retain None.
    result = recipe if injected else recipe.replace(**FROZEN_RECIPE_FIELDS, total_steps=None if recipe.total_steps is None else 1200)
    from .api_contract import validate_caller_objectives
    validate_caller_objectives(result, provider="api_model_ae")
    if (result.row_policy != "independent" or result.row_evidence_gate
            or result.particle_birth_death or result.serve_average
            or result.constraint_geometry_mode == "strict_progress"):
        raise ValueError("AE caller has no adaptive-row/averaged-serving/strict-progress task contract")
    if not case["host_definition"]["adversarial"] and (result.kinetic_transport_weight or result.kinetic_transport_local_weight):
        raise ValueError("reconstruction-only AE does not bind adversarial transport objectives")
    return result


class AEFixture:
    api_components = ("particlegan.Recipe", "particlegan.Recipe.encode", "particlegan.ParticleEncoding.reconstruction_loss",
                      "particlegan.Recipe.make_optimizers", "particlegan.Recipe.make_critic_penalty",
                      "particlegan.Recipe.prior_regularization", "particlegan.UpdatePolicy", "particlegan.init")

    def __init__(self, case, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe=None):
        if type(seed) is not int or seed != 0:
            raise ValueError("this prospective protocol is bound to seed 0, not a seed study")
        self.metadata, self.case_id, self.seed = deepcopy(case), case["id"], seed
        self.device = torch.device(device)
        self.execution_steps = case["default_steps"] if max_steps is None else max_steps
        if type(self.execution_steps) is not int or not 1 <= self.execution_steps <= case["default_steps"]:
            raise ValueError("max_steps must be a positive prefix of the frozen budget")
        self.adversarial = case["host_definition"]["adversarial"]
        self.recipe = task_recipe(case, recipe_name=recipe_name, recipe=recipe)
        with construction_rng(seed, self.device):
            self.G = SimpleMLPGenerator(2, 64, 2, 2).to(self.device)
            self.D = SimpleMLPDiscriminator(2, 64, 2, 2).to(self.device)
            self.E = SimpleMLPGenerator(2, 64, 2, 4).to(self.device)
            self.prior_constructor_rng = torch.Generator(device=self.device).manual_seed(seed)
            self.prior = self.recipe.replace(prior_update="learned").make_prior(
                sigma=.025, init_std=.5, device=self.device, generator=self.prior_constructor_rng)
            self.constructor_rng = torch.random.get_rng_state().clone()
            self.constructor_cuda_rng = torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None
            for offset, model in enumerate((self.G, self.D, self.E, self.prior)):
                init.deterministic_orthogonal_(model, seed=seed + offset)
        ema = None if self.recipe.effective_critic_formulation == "bispf" else deepcopy(self.D)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(self.G, self.D, self.prior, encoder=self.E,
                                                           ema_critic=ema, separate_encoder_groups=True,
                                                           foreach=False, fused=False)
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, prior=self.prior, encoder=self.E,
                                   generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
                                   penalty=self.penalty, seed=seed)
        self.data_rng = torch.Generator(device=self.device).manual_seed(seed + 101)
        self.generator_data_rng = torch.Generator(device=self.device).manual_seed(seed + 102)
        self.prior_noise_rng = torch.Generator(device=self.device).manual_seed(seed + 103)
        self.input_noise_rng = torch.Generator(device=self.device).manual_seed(seed + 104)
        self._critic = InputNoise(self.D, generator=self.input_noise_rng)
        self.batch_sequence_sha256 = hashlib.sha256(b"api-model-ae-batches-v1").hexdigest()

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def _prior_draw(self):
        return self.prior.sample(self.recipe.batch_size, generator=self.policy.latent_generator, noise_generator=self.prior_noise_rng)

    def _reconstruct(self, real, *, training_rows=False):
        query, offset = self.E(real).chunk(2, dim=1)
        encoding = self.recipe.encode(query, self.prior, offset=offset)
        if training_rows:
            self.policy.observe_sampled_rows(encoding.indices.reshape(-1))
        prediction = self.G(encoding.codes[:, 0])
        return prediction, encoding.reconstruction_loss(prediction[:, None], real)

    def step(self):
        if self.completed_steps >= self.execution_steps:
            raise RuntimeError("declared execution cap reached")
        with torch.autograd.set_multithreading_enabled(False):
            return self._step()

    def _step(self):
        real_d = sample_target(self.recipe.batch_size, self.data_rng, device=self.device)
        real_g = sample_target(self.recipe.batch_size, self.generator_data_rng, device=self.device)
        from experiments.forge.state import state_digest
        digest = state_digest({"critic": real_d, "generator": real_g})
        self.batch_sequence_sha256 = hashlib.sha256(bytes.fromhex(self.batch_sequence_sha256) + bytes.fromhex(digest)).hexdigest()
        noise = self.policy.begin_step(real_d, game_record=self.penalty.regularizer.record, execution_limit=self.execution_steps)
        self._critic.std = noise.input_sigma
        self.G.eval(); self.E.eval(); self.D.train()
        # Both controls draw the same training batches and prior streams. The
        # explicit reconstruction control omits all adversarial optimizer work.
        with torch.no_grad():
            latent, rows = self._prior_draw()
            fake = self.policy.generate(latent, sigma=noise.output_sigma, stream=self.policy.noise_generator, rows=rows if self.adversarial else None)
        self.policy.observe_support(latent)
        self.policy.observe_critic_pair(real_d, fake)
        zero = real_d.new_zeros(())
        critic_loss, penalty = zero, zero
        if self.adversarial:
            critic_loss = self.loss.d_loss(self._critic(real_d), self._critic(fake))
            penalty = self.penalty(self._critic, real_d, fake)
            self.opt_d.zero_grad(set_to_none=True)
            (critic_loss + penalty).backward(); self.opt_d.step()
        self.policy.after_critic_step()
        self.D.eval(); self.G.train(); self.E.train()
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            latent, rows = self._prior_draw()
            fake = self.policy.generate(latent, sigma=noise.output_sigma, stream=self.policy.noise_generator, rows=rows if self.adversarial else None)
            adversarial = self.loss.g_loss(self._critic(fake), self._critic(real_g)) if self.adversarial else zero
            _, reconstruction = self._reconstruct(real_g, training_rows=True)
            regularization = self.recipe.prior_regularization(self.prior.z, regularizer=self.regularizer)
            transport = self.recipe.kinetic_transport_loss(fake, real_g) if self.adversarial and self.recipe.kinetic_transport_weight else zero
            local = self.recipe.kinetic_transport_local_loss(fake, real_g) if self.adversarial and self.recipe.kinetic_transport_local_weight else zero
            total = self.recipe.reconstruction_weight * reconstruction + adversarial + regularization + transport + local
            self.opt_g.zero_grad(set_to_none=True)
            constraint_geometry_backward(total, self.opt_g, (reconstruction, adversarial) if self.adversarial else (reconstruction,))
            self.policy.after_generator_backward(loss_gan=adversarial.detach(), loss_critic=critic_loss.detach())
            self.opt_g.step(); self.policy.after_generator_step()
        finally:
            for p, flag in zip(self.D.parameters(), flags):
                p.requires_grad_(flag)
        self.policy.finish_step()
        result = dict(step=self.completed_steps, reconstruction=float(reconstruction.detach()),
                      generator_gan=float(adversarial.detach()), critic_gan=float(critic_loss.detach()),
                      prior_regularization=float(regularization.detach()), critic_penalty=float(penalty.detach()))
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("nonfinite public AE update")
        return result

    @torch.no_grad()
    def observe(self, n=4096, seed=10000):
        if type(n) is not int or n < 2 or type(seed) is not int or seed < 0:
            raise ValueError("evaluation requires n>=2 and a nonnegative seed")
        modes = [(m, m.training) for model in (self.G, self.D, self.E, self.prior) for m in model.modules()]
        try:
            for model in (self.G, self.D, self.E, self.prior):
                model.eval()
            target = sample_target(n, torch.Generator(device=self.device).manual_seed(seed), device=self.device)
            reconstruction, _ = self._reconstruct(target)
            latent, _ = self.prior.sample(n, generator=torch.Generator(device=self.device).manual_seed(seed + 1),
                                           noise_generator=torch.Generator(device=self.device).manual_seed(seed + 2))
            samples = self.G(latent)
            observed = score_samples(target, reconstruction, samples, adversarial=self.adversarial)
            observed["views"] = [dict(kind="scatter", title="Paired noisy inputs and AE reconstructions", target=target.cpu(), samples=reconstruction.cpu(), xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], caption="Matched held-out rows; reconstruction and mode agreement are scored separately from prior draws."),
                                 dict(kind="scatter", title="Independent clean prior generation", target=target.cpu(), samples=samples.cpu(), xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], caption="Uniform learned MoG plus fixed latent noise. All four modes, masses and within-mode widths enter the AE-GAN gate.")]
            return observed
        finally:
            for module, mode in modes:
                module.training = mode

    def state_dict(self):
        return deepcopy(dict(version=VERSION, case_id=self.case_id, metadata_sha256=_digest(self.metadata),
                             seed=self.seed, execution_steps=self.execution_steps, recipe=self.recipe.to_dict(),
                             policy=self.policy.state_dict(), prior=self.prior.state_dict(),
                             data_rng=self.data_rng.get_state(), generator_data_rng=self.generator_data_rng.get_state(),
                             prior_noise_rng=self.prior_noise_rng.get_state(), input_noise_rng=self.input_noise_rng.get_state(),
                             constructor_rng=self.constructor_rng, constructor_cuda_rng=self.constructor_cuda_rng,
                             prior_constructor_rng=self.prior_constructor_rng.get_state(),
                             batch_sequence_sha256=self.batch_sequence_sha256,
                             modes={name: {path: module.training for path, module in model.named_modules()} for name, model in (("G", self.G), ("D", self.D), ("E", self.E), ("prior", self.prior))},
                             **self.recipe.prior_regularizer_checkpoint(self.regularizer)))

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("version", "case_id", "metadata_sha256", "seed", "execution_steps", "recipe"):
            if state.get(key) != expected[key]:
                raise ValueError(f"incompatible AE checkpoint {key}")
        self.recipe.restore_prior_regularizer(self.regularizer, state.get("prior_regularizer"), validate_only=True)
        self.policy.load_state_dict(state["policy"])
        self.prior.load_state_dict(state["prior"])
        self.recipe.restore_prior_regularizer(self.regularizer, state.get("prior_regularizer"))
        for key in ("data_rng", "generator_data_rng", "prior_noise_rng", "input_noise_rng", "prior_constructor_rng"):
            getattr(self, key).set_state(state[key])
        self.constructor_rng = state["constructor_rng"].clone()
        self.constructor_cuda_rng = None if state["constructor_cuda_rng"] is None else state["constructor_cuda_rng"].clone()
        self.batch_sequence_sha256 = state["batch_sequence_sha256"]
        for name in ("G", "D", "E", "prior"):
            for path, module in getattr(self, name).named_modules():
                module.training = state["modes"][name][path]


def build_case(case_id, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe=None):
    cases = {case["id"]: case for case in list_cases()}
    if case_id not in cases:
        raise ValueError(f"unknown AE case: {case_id}")
    return AEFixture(cases[case_id], device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps, recipe=recipe)
