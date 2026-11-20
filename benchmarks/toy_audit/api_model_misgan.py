"""BISPF MisGAN with genuinely incomplete training and a neural amortized imputer.

Explicit four-mode architecture/target variant; never qualifies the historical
100-mode MLP recovery or imputer tasks. Complete examples are evaluation-only.
"""
from copy import copy, deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from particlegan import Recipe, UpdatePolicy, init
from particlegan.masked_mixture import NeuralComponentImputer, masked_mixture_nll, masked_quantile_transport_loss
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from .reproducibility import construction_rng
from .api_sources.misgan import BLOCK_PATTERNS, sample_masks, pattern_index, pattern_probs

ROOT = Path(__file__).resolve().parents[2]
VERSION = "misgan-four-mode-independent-games-v2"
DEFAULT_RECIPE = "bispf-optimized-quintic-v1"
CASE_ID = "api-model-misgan-four-mode"
TASK_ID = "model_misgan_four_mode"
RESOURCES = dict(num_particles=64, z_dim=2, batch_size=256)
CENTERS = ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))
SIGMA = .08
BOUNDS = (("modes", ">=", 4), ("hq", ">=", .90), ("mass_tv", "<=", .10),
          ("off_plane", "<=", .04), ("mask_tv", "<=", .05),
          ("imputer_observed_error", "<=", 1e-6), ("imputer_accuracy", ">=", .94),
          ("ambiguous_hq", ">=", .90), ("ambiguous_mass_tv", "<=", .12),
          ("ambiguous_query_coverage", ">=", .90), ("ambiguous_std_ratio", ">=", .70),
          ("ambiguous_std_ratio", "<=", 1.30))


def _digest(value):
    from experiments.forge.state import state_digest
    return state_digest(value)


FROZEN_HOST = {'provider': 'benchmarks.toy_audit.api_model_misgan', 'case_id': 'api-model-misgan-four-mode', 'resources': {'num_particles': 64, 'z_dim': 2, 'batch_size': 256}, 'num_particles': 64, 'z_dim': 2, 'batch_size': 256, 'init_std': 0.5, 'target': {'kind': 'four_corner_hadamard8', 'means': [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], 'sigma': 0.08, 'masses': [0.25, 0.25, 0.25, 0.25]}, 'architecture': {'generator': 'affine_2_to_8', 'mask': 'categorical4_linear1', 'imputer': '16_64_64_64_neural_logits', 'critics': '64_64'}, 'recipe_fields': {'model': 'gan', 'conditioning': 'scalar', 'encoder_mode': 'none', 'observation_sigma': 0.08, 'reconstruction_weight': 1.0}}
FROZEN_HOST["execution_contract"] = VERSION
FROZEN_QUESTION = 'Learn an 8D four-mode complete distribution and the block-mask law using only incomplete rows, then learn a stochastic neural imputer whose held-out single-sensor queries cover both ambiguous branches with correct mass.'

def list_cases():
    return [dict(id=CASE_ID, legacy_ids=["pr196-block-recovery", "pr196-block-neural-imputer"],
                 title="MisGAN: incomplete four-mode recovery and neural multimodal imputation",
                 goal=FROZEN_QUESTION, kind="vector", default_steps=1200,
                 batch_size=256, eval_samples=4096, default_recipe=DEFAULT_RECIPE,
                 evaluation_observations=24, terminal_observations=5, protocol_seed=0, execution_contract=VERSION,
                 thresholds=[list(x) for x in BOUNDS], forge_thresholds=[list(x) for x in BOUNDS],
                 sampling="Clean live learned MoG64 R2 sigma=.025 -> learned affine 8D G_x; learned four-pattern categorical G_m. G_i neural logits on x*m,m sample a component and independent MoG noise through G_x; no component distances, ppost or oracle at inference.",
                 scope="Explicit four-corner sigma=.08 target with a fixed Hadamard 8x2 lift. Same four block masks and masses as original MisGAN; sensor7 sees x+y, leaving two equally probable branches at zero. Original 100-mode MLP questions remain unchanged.",
                 adaptation="New identifiable target, affine generator, categorical mask support learned from observed masks, and neural categorical amortizer; no complete real data reaches training.",
                 host_definition=deepcopy(FROZEN_HOST))]


class Problem:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.A = torch.tensor([[1., 1.], [1., -1.]] * 3 + [[1., -1.], [1., 1.]], dtype=torch.float32, device=device) / math.sqrt(8)
        rng = torch.Generator().manual_seed(0)
        centers = torch.tensor(CENTERS)
        def draw(n):
            y = torch.randint(4, (n,), generator=rng)
            z = centers[y] + SIGMA * torch.randn(n, 2, generator=rng)
            return (z @ self.A.cpu().T).to(device), y.to(device)
        full, _ = draw(20000)
        self.m_train = sample_masks("block", 20000, rng, device=device)
        count = self.m_train.sum(0)
        self.mean = (full * self.m_train).sum(0) / count
        self.std = (((full-self.mean).square() * self.m_train).sum(0) / count).sqrt()
        # The complete training pool is discarded here, before training exists.
        self.observed = self.standardize(full) * self.m_train
        raw, self.y_test = draw(4096)
        self.x_test = self.standardize(raw)
        self.m_test = sample_masks("block", 4096, rng, device=device)
        self.data_law_rng = rng.get_state().clone()
        self.centers = centers.to(device)
        self.patterns = torch.zeros(4, 8, device=device)
        for row, (sensors, _) in enumerate(BLOCK_PATTERNS):
            self.patterns[row, list(sensors)] = 1

    def standardize(self, x):
        return (x-self.mean)/self.std

    def to2d(self, x):
        return (x*self.std+self.mean) @ self.A

    def batch(self, count, generator):
        ids = torch.randint(len(self.observed), (count,), generator=generator, device=self.device)
        return self.observed[ids], self.m_train[ids]

    def ambiguous_queries(self):
        # Sixteen held-out noisy sums, all overwhelmingly identify the two
        # mixed-sign modes; each conditional retains equiprobable branches.
        raw = torch.zeros(16, 8, device=self.device)
        raw[:, 7] = torch.linspace(-.03, .03, 16, device=self.device)
        masks = torch.zeros_like(raw); masks[:, 7] = 1
        return self.standardize(raw) * masks, masks


class Generators(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = nn.Linear(2, 8)
        self.mask = nn.Linear(1, 4)
        self.imputer = NeuralComponentImputer(8, 64, 64)

    def forward(self, z):
        return self.data(z)


def critic(dim):
    return nn.Sequential(nn.Linear(dim, 64), nn.LeakyReLU(.2), nn.Linear(64, 64), nn.LeakyReLU(.2), nn.Linear(64, 1))


def task_recipe(case, recipe_name=None, recipe=None):
    if case.get("execution_contract") != VERSION or case.get("host_definition") != FROZEN_HOST:
        raise ValueError("incompatible historical four-mode MisGAN execution; use its original source")
    injected = recipe is not None
    if recipe is None:
        name = DEFAULT_RECIPE if recipe_name in (None, "auto") else recipe_name
        recipe = Recipe.from_dict(json.loads((ROOT / f"configs/forge/ideas/{name}.json").read_text())["recipe_overrides"])
    fixed = dict(**RESOURCES, prior_kind="mog", standardize=False, sigma_rel=0.,
                 model="gan", conditioning="scalar", encoder_mode="none", observation_sigma=.08,
                 reconstruction_weight=1.)
    if injected and any(getattr(recipe, k) != v for k, v in fixed.items()):
        raise ValueError("injected recipe conflicts with frozen MisGAN task resources/objectives")
    if not injected:
        recipe = recipe.replace(**fixed, total_steps=1200)
    from .api_contract import validate_caller_objectives
    validate_caller_objectives(recipe, provider="api_model_misgan")
    unsupported = [name for name in ("continuous_policy", "output_noise_std", "input_noise_std",
                    "serve_average", "kinetic_transport_weight", "kinetic_transport_local_weight",
                    "row_evidence_gate", "particle_birth_death", "latent_damping_max_rate")
                   if getattr(recipe, name)]
    if recipe.training_precision != "float32":
        unsupported.append("training_precision")
    if recipe.kappa_radial != 1.:
        unsupported.append("kappa_radial")
    if recipe.output_noise_mode != "fixed":
        unsupported.append("output_noise_mode")
    if recipe.constraint_geometry_mode == "strict_progress":
        unsupported.append("constraint_geometry_mode=strict_progress")
    if recipe.lr_control == "stationarity":
        unsupported.append("lr_control=stationarity")
    if recipe.row_policy != "independent":
        unsupported.append("row_policy")
    if recipe.prior_code_law != "table":
        unsupported.append("prior_code_law")
    if recipe.effective_critic_formulation != "bispf":
        unsupported.append("critic_formulation")
    if unsupported:
        from .api_contract import UnsupportedRecipeOverrides
        raise UnsupportedRecipeOverrides("four-mode MisGAN cannot execute the intact recipe mechanisms: "
                                         + ", ".join(unsupported))
    return recipe


def score(problem, generated, masks, predictions, ambiguous):
    x2 = problem.to2d(generated)
    distances = torch.cdist(x2, problem.centers)
    nearest, ids = distances.min(1)
    counts = torch.bincount(ids, minlength=4).float()/len(ids)
    quality_counts = torch.bincount(ids[nearest <= 3*SIGMA], minlength=4)
    raw = generated*problem.std+problem.mean
    residual = raw - x2 @ problem.A.T
    empiric = torch.bincount(pattern_index(masks), minlength=256).float()/len(masks)
    imp2 = problem.to2d(predictions)
    pred_ids = torch.cdist(imp2, problem.centers).argmin(-1)
    ax2 = problem.to2d(ambiguous)
    adist, aid = torch.cdist(ax2, problem.centers).min(-1)
    mass1 = (aid == 1).float().mean(0)
    mass2 = (aid == 2).float().mean(0)
    observed_error = ((predictions - problem.x_test) * problem.m_test).abs().max()
    amask = torch.zeros(16, 8, device=generated.device); amask[:, 7] = 1
    ax, _ = problem.ambiguous_queries()
    observed_error = torch.maximum(observed_error, ((ambiguous - ax) * amask).abs().max())
    coordinate = (ax2[..., 0] - ax2[..., 1])/math.sqrt(2)
    metrics = dict(modes=int((quality_counts >= 10).sum()), hq=float((nearest <= 3*SIGMA).float().mean()),
                   mass_tv=float((counts-.25).abs().sum()/2), off_plane=float(residual.square().mean().sqrt()),
                   mask_tv=float((empiric-pattern_probs("block").to(empiric)).abs().sum()/2),
                   imputer_observed_error=float(observed_error),
                   imputer_accuracy=float((pred_ids == problem.y_test).float().mean()),
                   ambiguous_hq=float((adist <= 3*SIGMA).float().mean()),
                   ambiguous_mass_tv=float(((mass1-.5).abs()+(mass2-.5).abs()+1-mass1-mass2).mean()/2),
                   ambiguous_query_coverage=float(((mass1 >= .15) & (mass2 >= .15)).float().mean()),
                   ambiguous_std_ratio=float(coordinate.std(0).mean()/math.sqrt(2+SIGMA**2)))
    failures = [f"{name}={metrics[name]:.6g} {op} {bound} failed" for name, op, bound in BOUNDS
                if not (math.isfinite(metrics[name]) and (metrics[name] >= bound if op == ">=" else metrics[name] <= bound))]
    return dict(metrics=metrics, passed=not failures, failed_bounds=failures,
                views=[dict(kind="scatter", title="Complete generator: target and actual", target=problem.centers.cpu(), samples=x2[:1024].cpu(), xlim=[-1.7,1.7], ylim=[-1.7,1.7]),
                       dict(kind="scatter", title="Held-out single-sensor posterior: two branches", target=problem.centers[[1,2]].cpu(), samples=ax2[:,0].cpu(), xlim=[-1.7,1.7], ylim=[-1.7,1.7], caption="Actual neural conditional logits are sampled; no component distances or ppost at inference.")])


class MisGANFixture:
    api_components = ("particlegan.Recipe", "Recipe.make_prior", "Recipe.make_optimizers", "Recipe.make_critic_penalty",
                      "Recipe.prior_regularization", "particlegan.UpdatePolicy", "particlegan.init",
                      "particlegan.masked_mixture.masked_mixture_nll", "particlegan.masked_mixture.NeuralComponentImputer")

    def __init__(self, case, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe=None):
        if seed != 0:
            raise ValueError("this comparison is bound to protocol seed 0")
        self.metadata = deepcopy(case); self.device = torch.device(device)
        self.execution_steps = case["default_steps"] if max_steps is None else max_steps
        self.recipe = task_recipe(case, recipe_name, recipe)
        self.problem = Problem(device)
        with construction_rng(seed, self.device):
            self.G = Generators().to(device)
            self.D = nn.ModuleDict(dict(data=critic(16), mask=critic(8), imputer=critic(16))).to(device)
            self.prior_constructor_rng = torch.Generator(device=self.device).manual_seed(0)
            self.prior = self.recipe.make_prior(sigma=.025, init_std=.5, device=device, generator=self.prior_constructor_rng)
            self.constructor_rng = torch.random.get_rng_state().clone()
            self.constructor_cuda_rng = torch.cuda.get_rng_state(device).clone() if self.device.type == "cuda" else None
            for index, model in enumerate((self.G, self.D, self.prior)):
                init.deterministic_orthogonal_(model, seed=index)
        # The shared generator/prior update retains its original ordering.
        # Each distinct adversarial game owns its complete critic history.
        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G, self.D["data"], self.prior, foreach=False, fused=False)
        self.critic_optimizers = {"data": self.opt_d}
        for role in ("mask", "imputer"):
            self.critic_optimizers[role] = self.recipe.make_critic_optimizer(
                self.D[role], foreach=False, fused=False)
        self.losses = {role: self.recipe.make_loss() for role in self.D}
        self.penalties = {role: self.recipe.make_critic_penalty(optimizer, collect_stats=True)
                          for role, optimizer in self.critic_optimizers.items()}
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        from particlegan.bispf import bind_thermal_source
        for role, optimizer in self.critic_optimizers.items():
            bind_thermal_source(optimizer.record, self.losses[role],
                                self.regularizer if role == "data" else None)
        self.penalty = self.penalties["data"]
        # V2 explicitly anchors the single generator policy to the data game.
        # Unsupported multi-game controllers are rejected in task_recipe.
        self.policy = UpdatePolicy(self.recipe, self.G, self.D["data"], prior=self.prior,
                                   generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
                                   penalty=self.penalty, seed=seed)
        self.critic_initial_lrs = {role: [group["lr"] for group in optimizer.param_groups]
                                   for role, optimizer in self.critic_optimizers.items()}
        self.stream_names = ("data_rng", "generator_data_rng", "prior_noise_rng", "mask_rng", "imputer_rng", "model_rng")
        for index, name in enumerate(self.stream_names):
            setattr(self, name, torch.Generator(device=self.device).manual_seed(101+index))
        self.batch_sequence_sha256 = hashlib.sha256(b"misgan-four-mode-batches-v1").hexdigest()
        self._checkpoint_ready = True

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def _draw(self):
        return self.prior.sample(self.recipe.batch_size, generator=self.policy.latent_generator, noise_generator=self.prior_noise_rng)

    def _mask(self, n, rng):
        logits = self.G.mask(torch.ones(n, 1, device=self.device))
        u = torch.rand(logits.shape, device=self.device, generator=rng).clamp(1e-6,1-1e-6)
        soft = ((logits - torch.log(-torch.log(u)))/.5).softmax(-1)
        hard = F.one_hot(soft.argmax(-1),4).to(soft)
        return ((hard-soft).detach()+soft) @ self.problem.patterns, logits

    def _impute(self, x, m, rng, draws=1):
        ids = self.G.imputer.draw_indices(x, m, draws=draws, generator=rng)
        eps = torch.randn((*ids.shape, 2), generator=rng, device=self.device)
        values = self.G.data(self.prior.z[ids] + self.prior.sigma*eps)
        return x*m + values*(1-m)

    def step(self):
        if self.completed_steps >= self.execution_steps:
            raise RuntimeError("declared update budget reached")
        if not self._checkpoint_ready:
            raise RuntimeError("interrupted MisGAN update requires complete checkpoint restoration")
        self._checkpoint_ready = False
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.autograd.set_multithreading_enabled(False), torch.random.fork_rng(devices=devices):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self.model_rng.get_state(), self.device)
            else:
                torch.set_rng_state(self.model_rng.get_state())
            try:
                result = self._step()
            finally:
                state = (torch.cuda.get_rng_state(self.device) if self.device.type == "cuda"
                         else torch.get_rng_state())
                self.model_rng.set_state(state)
        self._checkpoint_ready = True
        return result

    def _step(self):
        x_d,m_d = self.problem.batch(self.recipe.batch_size,self.data_rng)
        x_g,m_g = self.problem.batch(self.recipe.batch_size,self.generator_data_rng)
        digest = _digest(dict(x_d=x_d,m_d=m_d,x_g=x_g,m_g=m_g))
        self.batch_sequence_sha256 = hashlib.sha256(bytes.fromhex(self.batch_sequence_sha256)+bytes.fromhex(digest)).hexdigest()
        self.policy.begin_step(x_d,game_record=self.penalty.regularizer.record,execution_limit=self.execution_steps)
        from particlegan.recipes import scale_learning_rates
        from particlegan.recipe_schedules import apply_penalty_schedule
        for role in ("mask", "imputer"):
            scale_learning_rates(self.completed_steps, self.recipe,
                                 [self.critic_optimizers[role]], [self.critic_initial_lrs[role]])
            apply_penalty_schedule(self.completed_steps, self.recipe, self.penalties[role])
        self.G.eval(); self.D.train()
        with torch.no_grad():
            z,rows = self._draw(); fake = self.policy.generate(z,rows=rows)
            mf,_ = self._mask(len(x_d),self.mask_rng)
            complete = self._impute(x_d,m_d,self.imputer_rng)[0]
        self.policy.observe_support(z)
        real = (torch.cat((x_d,m_d),1), m_d, torch.cat((fake,m_d),1))
        fakes = (torch.cat((fake*mf,mf),1), mf, torch.cat((complete,m_d),1))
        dl = x_d.new_zeros(())
        critic_adversarial = x_d.new_zeros(())
        for (role, model), r, f in zip(self.D.items(), real, fakes):
            adversarial_d = self.losses[role].d_loss(model(r), model(f))
            critic_adversarial = critic_adversarial + adversarial_d
            dl = dl + adversarial_d + self.penalties[role](model, r, f)
        for optimizer in self.critic_optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        dl.backward()
        for optimizer in self.critic_optimizers.values():
            optimizer.step()
        self.policy.after_critic_step()
        self.D.eval(); self.G.train(); self.D.requires_grad_(False)
        try:
            z,rows = self._draw(); fake = self.policy.generate(z,rows=rows)
            mf,logits = self._mask(len(x_g),self.mask_rng)
            # The categorical imputer's GAN gradient uses a straight-through
            # Gumbel draw, while serving uses its actual categorical law.
            ilogits = self.G.imputer(x_g,m_g)
            u = torch.rand(ilogits.shape,generator=self.imputer_rng,device=self.device).clamp(1e-6,1-1e-6)
            soft = ((ilogits-torch.log(-torch.log(u)))/.5).softmax(-1)
            hard = F.one_hot(soft.argmax(-1),64).to(soft)
            table = self.G.data(self.prior.z)
            component_codes = ((hard-soft).detach()+soft) @ self.prior.z.detach()
            imputer_noise = torch.randn(component_codes.shape,generator=self.imputer_rng,device=self.device)
            # Gradients reach the conditional logits through a frozen affine
            # decoder; the sampled kernel is identical to serving.
            decoded = F.linear(component_codes+self.prior.sigma*imputer_noise,
                               self.G.data.weight.detach(),self.G.data.bias.detach())
            imputed = x_g + (1-m_g) * decoded
            gx = self.losses["data"].g_loss(self.D["data"](torch.cat((fake*mf.detach(),mf.detach()),1)),self.D["data"](torch.cat((x_g,m_g),1)))
            gm = self.losses["mask"].g_loss(self.D["mask"](mf),self.D["mask"](m_g))
            gi = self.losses["imputer"].g_loss(self.D["imputer"](torch.cat((imputed,m_g),1)),self.D["imputer"](torch.cat((fake.detach(),m_g),1)))
            adversarial = gx+gm+gi
            # Draw synthetic labels in every arm, so an auxiliary cannot shift
            # the later prior/noise/data sequence relative to the baseline.
            teacher_z,teacher_ids = self._draw()
            synthetic = self.G.data(teacher_z).detach()
            categorical = self.G.imputer.synthetic_loss(synthetic,m_g,teacher_ids)
            mixture = masked_mixture_nll(x_g,m_g,table,sigma=self.recipe.observation_sigma)
            mask_ids = torch.cdist(m_g,self.problem.patterns).argmin(1)
            mask_nll = F.cross_entropy(logits,mask_ids)
            transport = (masked_quantile_transport_loss(fake,x_g,m_g,projections=self.recipe.kinetic_transport_projections)
                         if self.recipe.masked_transport_weight else fake.new_zeros(()))
            auxiliary = (self.recipe.masked_mixture_weight*mixture + self.recipe.masked_transport_weight*transport
                         + (self.recipe.masked_mixture_weight+self.recipe.masked_transport_weight)*mask_nll
                         + self.recipe.imputer_component_weight*categorical)
            regularization = self.recipe.prior_regularization(self.prior.z,regularizer=self.regularizer)
            self.policy.observe_sampled_rows(torch.arange(64,device=self.device))
            self.opt_g.zero_grad(set_to_none=True)
            protected = (adversarial,auxiliary) if auxiliary.requires_grad and (self.recipe.masked_mixture_weight or self.recipe.masked_transport_weight or self.recipe.imputer_component_weight) else (adversarial,)
            constraint_geometry_backward(adversarial+auxiliary+regularization,self.opt_g,protected)
            self.policy.after_generator_backward(loss_gan=adversarial.detach(),loss_critic=critic_adversarial.detach())
            self.opt_g.step(); self.policy.after_generator_step()
        finally:
            self.D.requires_grad_(True)
        self.policy.finish_step()
        return dict(step=self.completed_steps,loss_d=float(dl.detach()),loss_g=float(adversarial.detach()),masked_nll=float(mixture.detach()),masked_transport=float(transport.detach()),synthetic_ce=float(categorical.detach()))

    @torch.no_grad()
    def observe(self,n=4096,seed=10000):
        rng = torch.Generator(device=self.device).manual_seed(seed)
        z,_ = self.prior.sample(n,generator=rng)
        fake = self.G.data(z)
        masks,_ = self._mask(n,rng)
        predictions = self._impute(self.problem.x_test*self.problem.m_test,self.problem.m_test,rng,draws=1)[0]
        x,m = self.problem.ambiguous_queries()
        ambiguous = self._impute(x,m,rng,draws=256)
        result = score(self.problem,fake,masks,predictions,ambiguous)
        result["observed_sampling"] = dict(contract=VERSION, scoring_weights="live",
            predictor="G.imputer neural categorical logits and G.data", posterior_resampling=False,
            sampling_law="uniform_MoG64_sigma_0.025", eval_output_noise="clean")
        return result

    def _checkpoint_state(self, *, allow_interrupted=False):
        if not allow_interrupted and not self._checkpoint_ready:
            raise RuntimeError("MisGAN checkpoints require a completed update boundary")
        policy = copy(self.policy)
        policy._phase, policy._fast = "ready", None
        owners = {"G": self.G, "D": self.D, "prior": self.prior, "regularizer": self.regularizer}
        penalties = {}
        for role, penalty in self.penalties.items():
            penalties[role] = dict(record=penalty.optimizer.record.state_dict(),
                                   cooler=penalty.regularizer.cooler.state_dict(),
                                   coeff=penalty.regularizer.coeff, last_stats=penalty.last_stats)
        return deepcopy(dict(
            version=VERSION, metadata_sha256=_digest(self.metadata), recipe=self.recipe.to_dict(),
            execution_steps=self.execution_steps, policy=policy.state_dict(),
            models={role: model.state_dict() for role, model in owners.items()},
            critic_optimizers={role: optimizer.state_dict() for role, optimizer in self.critic_optimizers.items()},
            critic_initial_lrs=self.critic_initial_lrs, penalties=penalties,
            streams={name: getattr(self, name).get_state() for name in (*self.stream_names, "prior_constructor_rng")},
            constructor_rng=self.constructor_rng, constructor_cuda_rng=self.constructor_cuda_rng,
            batch_sequence_sha256=self.batch_sequence_sha256, data_law_rng=self.problem.data_law_rng,
            modes={role: {path: module.training for path, module in model.named_modules()} for role, model in owners.items()},
            requires_grad={role: {name: param.requires_grad for name, param in model.named_parameters()}
                           for role, model in owners.items()},
            gradients={role: {name: None if param.grad is None else param.grad for name, param in model.named_parameters()}
                       for role, model in owners.items()}))

    def state_dict(self):
        return self._checkpoint_state()

    def load_state_dict(self, state):
        from particlegan.policy import _validate_optimizer_state
        from particlegan.recipe_schedules import cosine_value
        if not isinstance(state, dict) or state.get("version") != VERSION:
            raise ValueError("incompatible historical MisGAN checkpoint; resume with its original source")
        # Loaders may retain moment tensors: never alias the caller's snapshot.
        state = deepcopy(state)
        expected = self._checkpoint_state(allow_interrupted=True)
        if state.keys() != expected.keys():
            raise ValueError("invalid complete MisGAN checkpoint fields")
        for name in ("metadata_sha256", "recipe", "execution_steps", "critic_initial_lrs", "requires_grad"):
            if state[name] != expected[name]:
                raise ValueError(f"incompatible MisGAN checkpoint {name}")
        owners = {"G": self.G, "D": self.D, "prior": self.prior, "regularizer": self.regularizer}
        try:
            for name in ("models", "critic_optimizers", "penalties", "streams", "modes", "gradients"):
                if not isinstance(state[name], dict) or state[name].keys() != expected[name].keys():
                    raise ValueError(f"incomplete {name}")
            validator = copy(self.policy)
            validator._phase, validator._fast = "ready", None
            validator._check_state(state["policy"])
            steps = state["policy"]["completed_steps"]
            if type(steps) is not int or not 0 <= steps <= self.execution_steps:
                raise ValueError("invalid update count")
            # The policy and explicit owners must describe the same shared objects.
            for role, policy_role in (("G", "generator"), ("prior", "prior")):
                if _digest(state["models"][role]) != _digest(state["policy"]["models"][policy_role]):
                    raise ValueError("contradictory policy/model state")
            data_model = {name.removeprefix("data."): value for name, value in state["models"]["D"].items()
                          if name.startswith("data.")}
            if _digest(data_model) != _digest(state["policy"]["models"]["critic"]):
                raise ValueError("contradictory data critic model")
            if _digest(state["critic_optimizers"]["data"]) != _digest(state["policy"]["optimizers"][1]):
                raise ValueError("contradictory data critic optimizer")
            for role, model in owners.items():
                saved = state["models"][role]
                UpdatePolicy._check_tensors(saved, expected["models"][role], f"MisGAN/{role}")
                if any(not torch.isfinite(value).all() for value in saved.values() if isinstance(value, torch.Tensor)):
                    raise ValueError("nonfinite model state")
                deepcopy(model).load_state_dict(saved)
                modes = state["modes"][role]
                if (not isinstance(modes, dict) or modes.keys() != expected["modes"][role].keys()
                        or any(type(flag) is not bool for flag in modes.values())):
                    raise ValueError("invalid module modes")
                gradients = state["gradients"][role]
                parameters = dict(model.named_parameters())
                if not isinstance(gradients, dict) or gradients.keys() != parameters.keys():
                    raise ValueError("invalid gradient fields")
                for name, value in gradients.items():
                    parameter = parameters[name]
                    if value is not None and (not isinstance(value, torch.Tensor)
                            or value.shape != parameter.shape or value.dtype != parameter.dtype
                            or not torch.isfinite(value).all()):
                        raise ValueError("invalid saved gradient")
            for role, optimizer in self.critic_optimizers.items():
                saved = state["critic_optimizers"][role]
                _validate_optimizer_state(optimizer, saved)
                values = state["penalties"][role]
                if not isinstance(values, dict) or values.keys() != expected["penalties"][role].keys():
                    raise ValueError("invalid penalty fields")
                record = saved.get("regularizer", {}).get("record")
                if _digest(record) != _digest(values["record"]):
                    raise ValueError("contradictory optimizer/penalty history")
                if record["observed_steps"] != steps:
                    raise ValueError("critic history has the wrong update count")
                penalty = self.penalties[role]
                deepcopy(optimizer.record).load_state_dict(values["record"])
                cooler = values["cooler"]
                declaration = expected["penalties"][role]["cooler"]
                if not isinstance(cooler, dict) or cooler.keys() != declaration.keys():
                    raise ValueError("incomplete cooler state")
                if any(cooler[k] != declaration[k] for k in declaration.keys() - {"ema_residual", "initialized"}):
                    raise ValueError("changed cooler declaration")
                if (type(cooler["initialized"]) is not bool or type(cooler["ema_residual"]) not in (int, float)
                        or not math.isfinite(cooler["ema_residual"]) or cooler["ema_residual"] < 0):
                    raise ValueError("invalid cooler history")
                deepcopy(penalty.regularizer.cooler).load_state_dict(cooler)
                coefficient = penalty.initial_coeff
                if self.recipe.reg_coeff_end is not None:
                    coefficient = cosine_value(max(0, steps - 1), coefficient, self.recipe.reg_coeff_end,
                                               self.recipe.reg_coeff_anneal_end, self.recipe.total_steps)
                if type(values["coeff"]) not in (int, float) or values["coeff"] != coefficient:
                    raise ValueError("penalty coefficient differs from the recipe schedule")
                if not isinstance(values["last_stats"], dict):
                    raise ValueError("invalid penalty diagnostics")
                json.dumps(values["last_stats"], allow_nan=False)
            for name, value in state["streams"].items():
                torch.Generator(device=getattr(self, name).device).set_state(value.cpu())
            for name in ("constructor_rng", "constructor_cuda_rng", "data_law_rng"):
                if _digest(state[name]) != _digest(expected[name]):
                    raise ValueError("immutable constructor/data law differs")
            digest = state["batch_sequence_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid consumed-batch digest")
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid complete MisGAN checkpoint state") from error
        # Every owned object is validated before changing live weights or RNGs.
        self.policy._phase = "ready"
        self.policy.load_state_dict(state["policy"])
        for role, model in owners.items():
            model.load_state_dict(state["models"][role])
        for role, optimizer in self.critic_optimizers.items():
            optimizer.load_state_dict(state["critic_optimizers"][role])
            penalty = self.penalties[role]
            penalty.regularizer.cooler.load_state_dict(state["penalties"][role]["cooler"])
            penalty.regularizer.coeff = state["penalties"][role]["coeff"]
            penalty.last_stats = deepcopy(state["penalties"][role]["last_stats"])
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        self.constructor_rng = state["constructor_rng"].clone()
        self.constructor_cuda_rng = state["constructor_cuda_rng"]
        self.problem.data_law_rng = state["data_law_rng"].clone()
        self.batch_sequence_sha256 = state["batch_sequence_sha256"]
        for role, model in owners.items():
            for path, module in model.named_modules():
                module.training = state["modes"][role][path]
            for name, parameter in model.named_parameters():
                value = state["gradients"][role][name]
                parameter.grad = None if value is None else value.to(parameter).clone()
        self._checkpoint_ready = True


def build_case(case_id, **kwargs):
    if case_id != CASE_ID:
        raise ValueError("unknown MisGAN case")
    return MisGANFixture(list_cases()[0],**kwargs)
