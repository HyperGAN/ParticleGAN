"""Clean oracle-free v2 DDGAN transition games and ancestral-chain research toys.

Compared with the frozen v1 cohort (`api_model_ddgan.py`), v2 removes the
host-owned target-variance `div_loss` regularizer so that the public Recipe
owns 100% of training updates and ground-truth target parameters are read
exclusively for evaluation and analytic scorer controls.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import torch

from particlegan import DDGAN, MoGParticlePrior, UpdatePolicy, init
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from experiments.forge.rng import NamedStreams

from .api_model_ddgan import (
    ALPHA_BAR,
    BASELINE as BASELINE,
    BOUNDS as BOUNDS,
    ChainCritic,
    ChainGenerator,
    DDGANFixture,
    MixtureTarget,
    RESOURCES as RESOURCES,
    distribution_failures as distribution_failures,
    distribution_metrics as distribution_metrics,
    final_metrics as final_metrics,
    list_cases as ddgan_v1_cases,
    sample_chain as sample_chain,
    sample_transition as sample_transition,
    task_recipe as _v1_task_recipe,
)
from .reproducibility import DEFAULT_SEED, construction_rng

VERSION = "api-model-ddgan-v2"

_CASES = []
for _v1 in ddgan_v1_cases():
    _case = deepcopy(_v1)
    _case["id"] = _v1["id"] + "-v2"
    _case["legacy_ids"] = [_v1["id"], *_v1["legacy_ids"]]
    _case["title"] = _v1["title"] + " (oracle-free v2)"
    _case["terminal_observations"] = 1
    _case["oracle_free"] = True
    _case["scope"] = (
        "Clean oracle-free v2 DDGAN cohort: identical analytic target, schedule, G/D architecture, "
        "prior and calibrated terminal_observations=1 protocol as v1, with host-owned target-variance "
        "div_loss removed so the public Recipe owns every training update."
    )
    _CASES.append(_case)
CASES = {case["id"]: case for case in _CASES}


def list_cases():
    return deepcopy(_CASES)


def task_recipe(case, recipe_name=None, recipe=None):
    return _v1_task_recipe(case, recipe_name=recipe_name, recipe=recipe)


class DDGANV2Fixture(DDGANFixture):
    """Oracle-free DDGAN fixture: zero host-owned target-variance `div_loss`."""

    schema_version: str = VERSION
    lambda_div: float = 0.0

    def __init__(self, case, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe=None):
        self.case, self.seed, self.device = deepcopy(case), int(seed), torch.device(device)
        self.oracle_free = True
        self.lambda_div = 0.0
        self.cond_post_std = None
        self.limit = case["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0 < self.limit <= case["default_steps"]:
            raise ValueError("execution cap must be within the frozen task budget")
        self.recipe = task_recipe(case, recipe_name, recipe)
        self.target = MixtureTarget(case["target"])
        self.completed_steps, self.last_losses = 0, {}
        self.streams = NamedStreams(seed, device=self.device)
        with construction_rng(seed, "cpu"), self.streams.fork("init", component="models", device="cpu"):
            self.G, self.D = ChainGenerator(self.recipe.z_dim), ChainCritic()
            self.prior = MoGParticlePrior(
                sigma=0.25,
                standardize=False,
                **{name: getattr(self.recipe, name) for name in ("num_particles", "z_dim")},
            )
            self.E = self.make_encoder()
        for offset, module in enumerate((self.G, self.D, self.prior, self.E)):
            if module is not None:
                init.deterministic_orthogonal_(module, seed=seed + offset)
                module.to(self.device)
        self.schedule = DDGAN(ALPHA_BAR, device=self.device)
        self.recipe.apply_prior_policy(self.prior)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G,
            self.D,
            self.prior,
            encoder=self.E,
            separate_encoder_groups=self.E is not None,
            ema_critic=None if self.recipe.effective_critic_formulation == "bispf" else deepcopy(self.D),
        )
        self.base_rates = [
            [group["lr"] for group in optimizer.param_groups]
            for optimizer in (self.opt_g, self.opt_d)
        ]
        self.loss = self.recipe.make_loss()
        self.prior_regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.policy = UpdatePolicy(
            self.recipe,
            self.G,
            self.D,
            prior=self.prior,
            encoder=self.E,
            generator_optimizer=self.opt_g,
            critic_optimizer=self.opt_d,
            row_semantics="conditional",
            penalty=self.penalty,
            seed=seed,
            streams={
                "latent_generator": self._rng("prior", "policy_latent"),
                "penalty_generator": self._rng("noise", "policy_penalty"),
                "noise_generator": self._rng("noise", "policy_output"),
                "eval_generator": self._rng("eval", "policy_evaluation", "evaluation"),
            },
        )
        self.batch_digest = hashlib.sha256().hexdigest()
        self.initialization = (
            "public deterministic_orthogonal: G=seed, D=seed+1, prior=seed+2, E=seed+3; "
            "R2Normal prior; oracle-free v2 (zero host-owned target-variance div_loss)"
        )

    def _step(self):
        _, real, xt, t = self._training_pair("critic")
        self.policy.begin_step(real, game_record=self.penalty.regularizer.record, execution_limit=self.limit)
        self.G.eval()
        self.D.train()
        if self.E is not None:
            self.E.eval()
        with torch.no_grad():
            fake, _ = self._transition(xt, t, "critic")
        self.policy.observe_critic_pair(real, fake)
        self.opt_d.zero_grad()
        dl = self.loss.d_loss(self.D(real, xt=xt, t=t), self.D(fake, xt=xt, t=t))
        with self.streams.fork("noise", component="critic_penalty"):
            penalty = self.penalty(self.D, real, fake, xt=xt, t=t)
        (dl + penalty).backward()
        self.opt_d.step()
        self.policy.after_critic_step()

        clean, real, xt, t = self._training_pair("generator")
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.eval()
            self.G.train()
            self.D.requires_grad_(False)
            if self.E is not None:
                self.E.train()
            z, indices = self.prior.sample(
                len(xt),
                generator=self._rng("prior", "generator_latent"),
                noise_generator=self._rng("noise", "generator_kernel"),
            )
            self.bind_prior_rows(indices)
            clean_pred = self.G(xt, z, t)
            eta = torch.randn(
                xt.shape,
                generator=self._rng("noise", "generator_reverse"),
                device=xt.device,
                dtype=xt.dtype,
            )
            fake = self.schedule.reverse(clean_pred, xt, t, eta)
            gl = self.loss.g_loss(self.D(fake, xt=xt, t=t), self.D(real, xt=xt, t=t).detach())
            prior_loss = self.recipe.prior_regularization(
                self.prior.z, regularizer=self.prior_regularizer, prior=self.prior
            )
            extra, protected, diagnostics = self.extra_generator_objective(clean, xt, t)
            self.opt_g.zero_grad()
            constraint_geometry_backward(gl + prior_loss + extra, self.opt_g, (gl,) + tuple(protected))
            self.policy.after_generator_backward(loss_gan=gl.detach(), loss_critic=dl.detach())
            self.opt_g.step()
            self.policy.after_generator_step()
        finally:
            for p, flag in zip(self.D.parameters(), flags):
                p.requires_grad_(flag)
        self.policy.finish_step()
        self.completed_steps = self.policy.completed_steps
        self.last_losses = dict(
            g_gan=float(gl.detach()),
            d_gan=float(dl.detach()),
            div_loss=0.0,
            critic_penalty=float(penalty.detach()),
            prior_regularization=float(prior_loss.detach()),
            **diagnostics,
        )
        if not all(math.isfinite(float(value)) for value in self.last_losses.values()):
            raise FloatingPointError("nonfinite DDGAN v2 transition update")
        return dict(step=self.completed_steps, **self.last_losses)

    def state_dict(self):
        return deepcopy(
            dict(
                schema=self.schema_version,
                case_id=self.case["id"],
                seed=self.seed,
                device=str(self.device),
                limit=self.limit,
                recipe=self.recipe.to_dict(),
                completed_steps=self.completed_steps,
                models={name: model.state_dict() for name, model in self._models().items()},
                modes={
                    name: {key: module.training for key, module in model.named_modules()}
                    for name, model in self._models().items()
                },
                gradients={
                    name: {key: None if p.grad is None else p.grad.clone() for key, p in model.named_parameters()}
                    for name, model in self._models().items()
                },
                optimizers=[self.opt_g.state_dict(), self.opt_d.state_dict()],
                base_rates=self.base_rates,
                policy=self.policy.state_dict(),
                streams=self.streams.state_dict(),
                batch_digest=self.batch_digest,
                last_losses=self.last_losses,
                **self.recipe.prior_regularizer_checkpoint(self.prior_regularizer),
            )
        )

    def load_state_dict(self, state):
        for name, value in (
            ("schema", self.schema_version),
            ("case_id", self.case["id"]),
            ("seed", self.seed),
            ("device", str(self.device)),
            ("limit", self.limit),
            ("recipe", self.recipe.to_dict()),
        ):
            if state.get(name) != value:
                raise ValueError("DDGAN v2 checkpoint identity mismatch: " + name)
        if type(state.get("completed_steps")) is not int or not 0 <= state["completed_steps"] <= self.limit:
            raise ValueError("invalid checkpoint update count")
        self.streams.validate_state_dict(state["streams"])
        self.recipe.restore_prior_regularizer(
            self.prior_regularizer, state.get("prior_regularizer"), validate_only=True
        )
        for name, model in self._models().items():
            model.load_state_dict(state["models"][name])
            for key, module in model.named_modules():
                module.training = state["modes"][name][key]
            for key, parameter in model.named_parameters():
                gradient = state["gradients"][name][key]
                parameter.grad = None if gradient is None else gradient.to(parameter).clone()
        for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
            optimizer.load_state_dict(values)
        self.streams.load_state_dict(state["streams"])
        self.recipe.restore_prior_regularizer(self.prior_regularizer, state.get("prior_regularizer"))
        self.policy.load_state_dict(state["policy"])
        self.completed_steps, self.base_rates = state["completed_steps"], deepcopy(state["base_rates"])
        self.last_losses, self.batch_digest = deepcopy(state["last_losses"]), state["batch_digest"]


def build_case(case_id, *, device="cpu", seed=DEFAULT_SEED, recipe_name=None, max_steps=None, recipe=None):
    if case_id not in CASES:
        raise ValueError("unknown DDGAN v2 API toy: " + case_id)
    return DDGANV2Fixture(
        CASES[case_id],
        device=device,
        seed=seed,
        recipe_name=recipe_name,
        max_steps=max_steps,
        recipe=recipe,
    )
