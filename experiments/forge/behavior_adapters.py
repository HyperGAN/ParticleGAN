"""Bind the eight frozen behavioral objectives to public ParticleGAN components.

The existing hosts retain their data, objectives, and measurement loops. Their
explicit ``components`` argument replaces construction/update primitives once;
there is no copied training loop and no global optimizer patch. These hosts
honor their requested device, separately from the scalar GANTrainer path.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch
from torch import nn

from particlegan import init, prior_capabilities
from particlegan.recipes import learning_rate_scales
from particlegan.recipe_compat import without_default_additions
from particlegan.recipe_schedules import apply_optimizer_schedule
from particlegan.training import input_noise_std, output_noise_std
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, _InputAdapter, _OutputAdapter

from .api import CapabilityError, task_formulation_context, task_policy_blockers
from .contracts import file_hash
from .artifacts import save_provenance_checkpoint
from .contracts import atomic_json
from .initialization import task_initializer
from .mechanisms import MechanismAudit, mechanism_blockers
from .priors import task_prior, recipe_owned_prior
from .sampling import BEHAVIOR_POLICIES, POLICIES, executed_receipt
from .taskrecipes import BEHAVIOR_HOST_FIELDS, adaptation_receipt, bind_task_candidate


HOSTS = ("two_pole", "trajectory", "residual_student", "unipolar", "ae_gan_hold",
         "cover_leftover", "unused_token_hold", "mid_scale_identity")

# These resources/objectives belong to the frozen host, not the formulation.
# Even a matching explicit value must be removed from the candidate declaration:
# it cannot be a shared variable when another host owns a different value.
FROZEN_HOST_RECIPE_FIELDS = BEHAVIOR_HOST_FIELDS


def behavior_preflight(task: dict, candidate: dict) -> list[str]:
    """Return unsupported explicit overrides before constructing any host state."""
    try:
        from .two_pole_observer import diagnostic_contract
        diagnostic_contract(task)
        prior = task_prior(task)
        task_initializer(task, candidate)
        candidate = bind_task_candidate(candidate, task)
    except ValueError as error:
        return [str(error)]
    frozen_fields = FROZEN_HOST_RECIPE_FIELDS - ({"prior_reg"} if recipe_owned_prior(task) else set())
    fields = set(candidate.get("recipe_overrides", {})) & frozen_fields
    host = task["execution"].get("host", task["id"])
    if host != "ae_gan_hold":
        fields |= set(candidate.get("recipe_overrides", {})) & {"routing_temperature", "distance_reduction"}
    prior_blockers = []
    if host in HOSTS:
        expected = "mog" if host == "ae_gan_hold" else "particle_cloud"
        if prior["kind"] != expected:
            prior_blockers.append(f"{task['id']}: frozen behavioral host requires prior kind {expected}")
    return (prior_blockers + task_policy_blockers(task, candidate)
            + [f"{task['id']}: recipe override {name!r} is owned by the frozen host; revise its task specification"
               for name in sorted(fields)])


def _base(module):
    return module.model if isinstance(module, (_InputAdapter, _OutputAdapter)) else module


def _observe_range(receipt, value):
    """Retain compact actual schedule observations, without a per-update stream."""
    value = float(value)
    if not receipt:
        receipt.update(observations=0, first=value, minimum=value, maximum=value)
    receipt.update(observations=receipt["observations"] + 1, last=value,
                   minimum=min(receipt["minimum"], value), maximum=max(receipt["maximum"], value))


class _OptimizerBundle:
    """One host G step over public optimizers with independently owned priors."""
    def __init__(self, optimizers, *, enumerated_tables=()):
        self.optimizers = optimizers
        self.enumerated_tables = tuple(enumerated_tables)
        self.param_groups = [g for optimizer in optimizers for g in optimizer.param_groups]

    def zero_grad(self, *args, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        for optimizer in self.optimizers:
            setter = getattr(optimizer, "set_sampled_rows", None)
            if setter is not None:
                for table in self.enumerated_tables:
                    if any(table is p for group in optimizer.param_groups for p in group["params"]):
                        # These hosts feed the entire latent table to G instead
                        # of sampling. Ownership is exactly every table row.
                        setter(table, torch.arange(table.shape[0], device=table.device))
            optimizer.step(*args, **kwargs)

    def bind_protected_losses(self, losses, *, protected_evaluator=None):
        enabled = [optimizer for optimizer in self.optimizers
                   if hasattr(optimizer, "bind_protected_losses")]
        if not enabled:
            return
        if len(enabled) != 1 or len(self.optimizers) != 1:
            raise CapabilityError(["constraint_geometry requires a single joint generator/prior optimizer"])
        optimizer = enabled[0]
        for table in self.enumerated_tables:
            optimizer.set_sampled_rows(table, torch.arange(len(table), device=table.device))
        optimizer.bind_protected_losses(losses, protected_evaluator=protected_evaluator)

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]


class _ConditionedScore(nn.Module):
    def __init__(self, critic, scale):
        super().__init__()
        self.critic, self.scale = critic, scale

    def forward(self, values):
        return self.critic.score(values, self.scale)


class _PenaltyBinding:
    """Adapt old host call syntax to Recipe.make_critic_penalty's public API."""
    def __init__(self, recipe, optimizer, audit):
        self.bound = recipe.make_critic_penalty(optimizer, collect_stats=True)
        self.arm, self.coeff, self.kappa = recipe.reg_arm or recipe.critic_formulation, recipe.reg_coeff, recipe.reg_kappa
        self.norm, self.lazy_k, self.target_anneal = "rms", recipe.reg_every, "none"
        self.calls = 0
        self.audit = audit
        self.coefficient_observations = {}

    def penalty(self, critic, real, fake, step=None, **kwargs):
        self.calls += 1
        value = self.bound(critic, real, fake)
        if getattr(self, "diagnostic", None) is not None:
            self.diagnostic.penalty_gradient(value)
        _observe_range(self.coefficient_observations, self.bound.regularizer.coeff)
        self.audit.observe_penalty(self.bound.last_stats)
        return value, self.bound.last_stats

    def __call__(self, critic, real, fake, step=None, **kwargs):
        return self.penalty(critic, real, fake, step=step, **kwargs)[0]


class _NamedNoise(NoisePolicy):
    """Reuse the shared host wrappers, with public schedules and Forge streams."""
    def __init__(self, components):
        self.components = components
        recipe = components.recipe
        super().__init__(recipe.output_noise_std, recipe.input_noise_std,
                         recipe.input_noise_anneal_end, recipe.total_steps,
                         seed=components.context.streams.seed,
                         output_noise_warmup=recipe.output_noise_warmup,
                         output_noise_rng="isolated" if recipe.output_noise_std else None)
        streams = components.context.streams
        self.input_stream = streams.generator("noise", component="critic", purpose="input")
        self.output_stream = streams.generator("noise", component="generator", purpose="output")
        self._output_stream_initial_sha256 = hashlib.sha256(self.output_stream.get_state().numpy().tobytes()).hexdigest()
        for component, purpose in (("host", "global"), ("critic", "input"), ("generator", "output")):
            streams.generator("eval", component=component, purpose=purpose)

    def _output_sigma_for(self, completed_steps):
        return output_noise_std(self.components.recipe, completed_steps)

    def set_step(self, completed_steps):
        super().set_step(completed_steps)  # retain the host's mechanism-use counters
        self.input_sigma = input_noise_std(self.components.recipe, completed_steps)
        self.output_sigma = output_noise_std(self.components.recipe, completed_steps)
        return self.input_sigma

    def receipt(self):
        receipt = super().receipt()
        # The inherited policy's fixed seed offsets describe its own generators.
        # This adapter replaces them with named streams; report the generators
        # actually used, without drawing or resetting their current state.
        receipt.update(
            rng_derivation=self.components.context.streams.version,
            d_noise_seed=self.input_stream.initial_seed(),
            output_noise_seed=self.output_stream.initial_seed(),
            output_noise_seed_offset=None,
        )
        return receipt

    @contextmanager
    def evaluation(self, step):
        streams = self.components.context.streams
        before = streams.audit()
        old = self.input_stream, self.output_stream, self.input_sigma, self.output_sigma, self._evaluating
        try:
            with streams.preserve(), streams.fork("eval", component="host", purpose="global"):
                self.input_stream = streams.generator("eval", component="critic", purpose="input")
                self.output_stream = streams.generator("eval", component="generator", purpose="output")
                self.input_sigma = 0.0
                self.output_sigma = self._output_sigma_for(step)
                self._evaluating = True
                yield
        finally:
            self.input_stream, self.output_stream, self.input_sigma, self.output_sigma, self._evaluating = old
            self.components.rng_audits.append(streams.compare(before, streams.audit()))


class BehaviorComponents:
    """Explicit shared construction, optimizer, noise, and observation binder."""
    def __init__(self, request, task, *, device="cpu"):
        self.task = task
        reference_candidate = request.get("candidate", {})
        self.reference_candidate = deepcopy(reference_candidate)
        self.protocol = deepcopy(request.get("protocol", {}))
        candidate = bind_task_candidate(reference_candidate, task)
        self.host_adaptation = adaptation_receipt(reference_candidate, task)
        blockers = behavior_preflight(task, candidate)
        if blockers:
            raise CapabilityError(blockers)
        self.context = task_formulation_context(reference_candidate, task, self.protocol, device=device)
        extension_blockers = behavior_preflight(task, {"recipe_overrides": self.context.bindings["recipe"]})
        if extension_blockers:
            raise CapabilityError(extension_blockers)
        self.recipe = self.context.recipe
        self.uses_recipe_prior = recipe_owned_prior(task)
        self.prior_regularizer = None  # Active reference-based penalties bind after public initialization.
        if candidate.get("claim_contract", {}).get("learning") == "clockfree":
            raise CapabilityError(["these scheduled public components do not implement a clock-free optimizer"])
        self.models, self.optimizers, self.base_rates, self.role_parameters = {}, {}, {}, {}
        self.direct_particle_ids, self.initial_group_betas = set(), {}
        self.sampled_prior_rows = {}
        self.rng_audits, self.observations = [], []
        self.diagnostic, self.diagnostic_observations = None, []
        self.penalty = None
        self.schedule_observations = {}
        self.noise = _NamedNoise(self)
        self.bound = False

    def encoder_recipe(self, cfg):
        self.recipe = self.recipe.replace(encoder_mode="ae", z_dim=2, num_particles=cfg.n_particles,
                                          batch_size=cfg.batch)
        return self.recipe

    def make_prior(self, recipe):
        prior = recipe.replace(prior_update="learned").make_prior(sigma=self.context.prior_config["sigma"],
                                  init_std=self.context.prior_config.get("init_std", 1.),
                                  generator=self.context.streams.generator("init", component="prior", purpose="locations"))
        return prior

    def add_prior_regularization(self, loss, *priors):
        """One recipe objective over this host's actual latent tables."""
        if not self.uses_recipe_prior:
            raise ValueError("legacy hosts retain their original prior objective")
        priors = tuple(prior for prior in priors if prior.z.requires_grad)
        rows = [prior.z for prior in priors]
        if self.recipe.prior_regularizer == "coulomb" and self.recipe.prior_reg > 0:
            self.prior_regularizer = self.recipe.make_prior_regularizer(prior=priors)
        if rows and (self.recipe.prior_reg > 0 or self.recipe.prior_l2 > 0):
            loss = loss + self.recipe.prior_regularization(
                rows[0] if len(rows) == 1 else torch.cat(rows), regularizer=self.prior_regularizer)
        return loss

    @staticmethod
    def critic_copy(critic):
        return deepcopy(_base(critic))

    @staticmethod
    def conditioned_score(critic, scale):
        return _ConditionedScore(critic, scale)

    def _sample_binding(self, prior, component):
        original = prior.sample
        streams = self.context.streams
        streams.generator("eval", component=component, purpose="indices")
        streams.generator("eval", component=component, purpose="width")

        def sample(count, *args, generator=None, **kwargs):
            family = "eval" if self.noise._evaluating else "prior"
            kwargs["generator"] = streams.generator(family, component=component, purpose="indices")
            if self.context.prior_config["kind"] == "mog":
                kwargs["noise_generator"] = streams.generator(family, component=component, purpose="width")
            result = original(count, *args, **kwargs)
            if family != "eval":
                self.sampled_prior_rows[prior.z] = result[1]
                # Row-normalized tables must use the actual sampled rows.
                # Auxiliary all-table losses can create gradients elsewhere,
                # so a nonzero-gradient mask is not an equivalent contract.
                for bundle in self.optimizers.values():
                    owned = bundle.optimizers if isinstance(bundle, _OptimizerBundle) else [bundle]
                    for optimizer in owned:
                        setter = getattr(optimizer, "set_sampled_rows", None)
                        if setter is not None and any(
                                parameter is prior.z for group in optimizer.param_groups
                                for parameter in group["params"]):
                            # These retained hosts have exactly one prior draw
                            # per G step after the D draw. Their zero_grad can
                            # occur after sampling, so replace earlier D rows
                            # here instead of clearing them at zero_grad.
                            optimizer.clear_sampled_rows()
                            # An active joint optimizer may own multiple
                            # priors. Replacing this table's D draw with its
                            # G draw must retain the other table's actual rows.
                            for table, rows in self.sampled_prior_rows.items():
                                if any(parameter is table for group in optimizer.param_groups
                                       for parameter in group["params"]):
                                    setter(table, rows)
            return result
        prior.sample = sample

    def bind(self, *, generator, critic, opt_g, opt_d, priors=(), encoder=None, direct_particles=()):
        if self.bound:
            raise RuntimeError("one component binder cannot own two host runs")
        declared = self.context.prior_config
        for prior in priors:
            actual = prior_capabilities(prior)
            expected = {"kind": declared["kind"], "sigma": declared["sigma"],
                        "standardize": declared["standardize"], "learned_locations": declared["learnable"]}
            if self.uses_recipe_prior:
                expected.pop("learned_locations")  # Policy binds after identical initialization.
            if actual["kind"] == "mog":
                expected["sigma"] = float(prior.sigma.new_tensor(declared["sigma"]))
            if any(actual[key] != value for key, value in expected.items()):
                raise CapabilityError([f"{self.task['id']}: constructed prior differs from execution.prior"])
        self.bound = True
        critic = _base(critic)
        for role, model in (("generator", generator), ("encoder", encoder), ("discriminator", critic)):
            if model is None:
                continue
            model = _base(model)
            self.models[role] = model
            self.role_parameters[role] = list(model.parameters())
            if self.task["execution"].get("host", self.task["id"]) != "two_pole":
                self.context.initialize(model, component=role)
        for index, prior in enumerate(priors):
            name = f"prior{index}"
            self.models[name] = prior
            self.context.initialize(prior, component=name)
            self.recipe.apply_prior_policy(prior)
            self._sample_binding(prior, name)
        self.prior_regularizer = self.recipe.make_prior_regularizer(reference=priors) if priors else None
        prior_parameters = [p for prior in priors for p in prior.parameters() if p.requires_grad] + list(direct_particles)
        if prior_parameters or priors:
            self.role_parameters["prior"] = prior_parameters
        parts = []
        g_parameters = self.role_parameters.get("generator", []) + self.role_parameters.get("encoder", [])
        if g_parameters:
            parts.append(self.recipe.make_generator_optimizer(g_parameters))
        for prior in priors:
            parameters = [p for p in prior.parameters() if p.requires_grad]
            if not parameters:
                continue
            parts.append(self.recipe.make_generator_optimizer(
                [{"params": parameters, "lr": self.recipe.lr * self.recipe.prior_lr_mult,
                  "betas": self.recipe.prior_betas or self.recipe.betas, "forge_role": "prior",
                  **({"eps": self.recipe.prior_eps} if self.recipe.prior_eps is not None else {})}], latent_table=prior.z))
        if direct_particles:
            self.direct_particle_ids = {id(parameter) for parameter in direct_particles}
            parts.append(self.recipe.make_generator_optimizer(
                [{"params": list(direct_particles), "forge_role": "prior"}],
                direct_particles=list(direct_particles)))
        if not parts:
            raise CapabilityError(["host exposes no trainable generator-side component"])
        if self.recipe.constraint_geometry_mode != "none" and len(parts) > 1:
            # Zero-momentum full DualNorm has independent per-group histories.
            # Joining their groups changes no base direction, rate or ownership;
            # it permits one projection of the actual combined displacement.
            groups = [{key: value for key, value in group.items() if key != "algorithm"}
                      for optimizer in parts for group in optimizer.param_groups]
            parts = [self.recipe.make_generator_optimizer(groups)]
        full_table = self.task["execution"].get("host", self.task["id"]) in {"trajectory", "residual_student"}
        public_g = _OptimizerBundle(parts, enumerated_tables=[prior.z for prior in priors] if full_table else ())
        public_d = self.recipe.make_critic_optimizer(critic, ema_critic=self.critic_copy(critic))
        self.optimizers = {"generator": public_g, "discriminator": public_d}
        for optimizer in self.optimizers.values():
            self.base_rates[id(optimizer)] = [group["lr"] for group in optimizer.param_groups]
            self.initial_group_betas[id(optimizer)] = [tuple(group["betas"]) for group in optimizer.param_groups]
        self.mechanism_audit = MechanismAudit(self.recipe, public_d, parts)
        self.penalty = _PenaltyBinding(self.recipe, public_d, self.mechanism_audit)
        # Host constructors may seed global RNG for fixed fixtures. Training
        # starts on the independent data stream after construction is complete.
        data = self.context.streams.generator("data", component="host", purpose="batches")
        if data.device.type == "cuda":
            torch.cuda.set_rng_state(data.get_state(), data.device)
        else:
            torch.set_rng_state(data.get_state())
        if self.diagnostic is not None:
            if len(parts) != 1 or len(direct_particles) != 1:
                raise CapabilityError(["two-pole diagnostics require the existing single direct-coordinate optimizer"])
            self.diagnostic.bind(self, direct_particles[0], parts[0], public_d)
            self.penalty.diagnostic = self.diagnostic
        return public_g, public_d, self.recipe.make_loss(), self.penalty

    def schedule_optimizer(self, optimizer, completed_updates):
        apply_optimizer_schedule(completed_updates, self.recipe, optimizer)
        network, prior = learning_rate_scales(completed_updates, self.recipe)
        for group, base in zip(optimizer.param_groups, self.base_rates[id(optimizer)]):
            group["lr"] = base * (prior if group.get("forge_role") == "prior" else network)
            role = next(name for name, owned in self.optimizers.items() if owned is optimizer)
            role = "prior" if group.get("forge_role") == "prior" else role
            observed = self.schedule_observations.setdefault(role, {"lr": {}, "beta2": {}})
            _observe_range(observed["lr"], group["lr"])
            _observe_range(observed["beta2"], group["betas"][1])

    def checkpoint(self, step, measure):
        budget = self.task["execution"]["steps"]
        expected = {math.ceil(i * budget / 24) for i in range(1, 25)}
        diagnostic_steps = self.diagnostic.observation_steps if self.diagnostic is not None else set()
        if step not in expected | diagnostic_steps:
            return
        with self.noise.evaluation(step):
            values = measure()
        if step in expected:
            self.observations.append({**values, "step": step})
        if step in diagnostic_steps:
            self.diagnostic_observations.append({**values, "step": step})
        print(json.dumps(dict(event="observation" if step in expected else "diagnostic_observation", task=self.task["id"], step=step,
                              budget=budget, metrics=values), allow_nan=False), flush=True)

    def constraint_geometry_capture(self, step, target, prediction, metrics, *, kind="image"):
        """Save the already scored deterministic panel; no additional sampling."""
        records = getattr(self, "constraint_geometry_outputs", [])
        self.constraint_geometry_outputs = records
        thresholds = self.task["evaluation"]["thresholds"]
        passed = all((metrics[name] <= value if op == "<=" else
                      metrics[name] >= value if op == ">=" else metrics[name] == value)
                     for name, op, value in thresholds)
        failures = [] if passed else [name for name, op, value in thresholds
                                       if not (metrics[name] <= value if op == "<=" else
                                               metrics[name] >= value if op == ">=" else metrics[name] == value)]
        records.append(dict(step=step, metrics=dict(metrics), passed=passed, failed_bounds=failures,
                            views=[dict(kind=kind, title="Declared paired target and actual trained prediction",
                                        target=target.detach().cpu(), samples=prediction.detach().cpu(),
                                        caption="Same scored state and pairing; fixed target, no replay training.",
                                        vmin=-1.5, vmax=1.5)]))

    def add_transport_loss(self, total, fake, real, *, conditioning=None):
        """Only a declared variant consumes marginal output transport."""
        if self.context.component_transport is None or not (
                self.recipe.kinetic_transport_weight or self.recipe.kinetic_transport_local_weight):
            return total
        if not hasattr(self, 'transport_consumer'):
            from particlegan.conditional_transport import OutputMarginalTransport
            self.transport_consumer = OutputMarginalTransport(self.recipe)
        return self.transport_consumer.add(total, fake.flatten(1), real.flatten(1), conditioning=conditioning)

    def guards(self):
        steps = {}
        all_state = {}
        for optimizer in self.optimizers.values():
            owned = optimizer.optimizers if isinstance(optimizer, _OptimizerBundle) else [optimizer]
            for sub in owned:
                all_state.update(sub.state)
        for role, params in self.role_parameters.items():
            observed = [int(all_state.get(p, {}).get("step", 0)) for p in params if p.requires_grad]
            steps[role] = min(observed) if observed else 0
        finite = all(torch.isfinite(p).all().item() and (p.grad is None or torch.isfinite(p.grad).all().item())
                     for params in self.role_parameters.values() for p in params)
        def finite_state(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite_state(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite_state(v) for v in value)
            return math.isfinite(value) if type(value) in (int, float) else True
        finite = finite and all(finite_state(optimizer.state_dict()) for optimizer in self.optimizers.values())
        if self.uses_recipe_prior:
            finite = finite and all(finite_state(model.state_dict()) for model in self.models.values())
        mechanisms = self.mechanism_audit.receipt()
        transport_requested = bool(self.recipe.kinetic_transport_weight or self.recipe.kinetic_transport_local_weight)
        transport_calls = self.transport_consumer.active_calls if hasattr(self, 'transport_consumer') else 0
        transport_exercised = not transport_requested or transport_calls == steps.get('generator', steps.get('prior', 0))
        return dict(all_finite=finite, optimizer_updates=steps,
                    hooks_exercised=not mechanism_blockers(mechanisms) and transport_exercised,
                    mechanism_audit=mechanisms,
                    component_transport_requested=transport_requested,
                    component_transport_active_calls=transport_calls,
                    unintended_rng_deviations=sum(a["unintended_rng_deviations"] for a in self.rng_audits))

    def optimizer_group_bindings(self):
        """Report construction and step controls without changing optimizer law."""
        rows = []
        for role, optimizer in self.optimizers.items():
            owned = optimizer.optimizers if isinstance(optimizer, _OptimizerBundle) else [optimizer]
            group_index = 0
            for public in owned:
                for group in public.param_groups:
                    direct = bool(self.direct_particle_ids) and {id(p) for p in group["params"]} == self.direct_particle_ids
                    latent = group.get("forge_role") == "prior" and not direct
                    response = getattr(public, "direct_response", None) if direct else None
                    rows.append({"optimizer_role": role, "group_index": group_index,
                        "optimizer": type(public).__name__, "counter_role": group.get("forge_role", role),
                        "representation": "direct_sample_coordinates" if direct else "latent_prior_locations" if latent else "network",
                        "base_lr": self.base_rates[id(optimizer)][group_index], "current_lr": group["lr"],
                        "base_betas": list(self.initial_group_betas[id(optimizer)][group_index]),
                        "current_group_betas": list(group["betas"]),
                        "lr_schedule": "prior" if group.get("forge_role") == "prior" else "network",
                        "prior_lr_mult_consumed": latent, "prior_betas_consumed": latent,
                        "beta2_schedule_declared": self.recipe.beta2_end is not None,
                        "group_betas_overridden_during_step": response is not None,
                        "direct_particle_response": {"installed": response is not None,
                            "step_betas": list(response.betas) if response is not None else None,
                            "gain_enabled": response.gain if response is not None else False,
                            "lr_gain_range": [1., 2.] if response is not None and response.gain else [1., 1.]}})
                    group_index += 1
        return rows

    def receipt(self):
        from .boundaries import ownership_receipt
        resolved_recipe = asdict(self.recipe)
        recipe = without_default_additions(resolved_recipe, fields=("optimizer_dense_profile",
            "reg_real_weight", "reg_fake_weight", "joint_reconstruction_weight",
            "conditional_energy_weight", "conditional_energy_samples",
            "masked_mixture_weight", "masked_transport_weight", "imputer_component_weight"))
        ownership = ownership_receipt(self.reference_candidate, self.task, resolved_recipe,
            self.protocol, self.context.initializer,
            extension_recipe_bindings=self.context.bindings["recipe"])
        # Validate the complete effective Recipe above, then preserve the
        # historical applied-packet shape for unselected optional additions.
        ownership["recipe_fields"] = {name: value for name, value in ownership["recipe_fields"].items()
                                      if name in recipe}
        return dict(execution_path="public_components", recipe=recipe,
                    initializer=self.context.initializer, initialization=deepcopy(self.context.initialization),
                    field_ownership=ownership,
                    **({"host_adaptation": self.host_adaptation} if self.host_adaptation else {}),
                    prior=self.context.prior_config, active_roles=sorted(self.role_parameters),
                    public_optimizers=[type(o).__name__ for opt in self.optimizers.values()
                                       for o in (opt.optimizers if isinstance(opt, _OptimizerBundle) else [opt])],
                    optimizer_group_bindings=self.optimizer_group_bindings(),
                    noise=self.noise.receipt(), rng=self.context.streams.manifest(), rng_audits=self.rng_audits,
                    training_schedules={"clock": "completed host updates; penalty observes completed critic updates",
                                        "horizon": self.recipe.total_steps,
                                        "optimizer_groups": deepcopy(self.schedule_observations),
                                        "penalty_coefficient": deepcopy(self.penalty.coefficient_observations)
                                        if self.penalty is not None else {}},
                    checkpoint_scope="component states; these hosts do not support continuation",
                    mechanism_applicability={"a2": ("disabled by latent_damping_max_rate=0" if self.recipe.latent_damping_max_rate == 0 else
                                               "installed on prior tables; activation is recorded in mechanism_audit" if any(n.startswith("prior") for n in self.models)
                                               else "not applicable: host has no latent-table optimizer"),
                                             "critic_guard": "disabled" if self.recipe.d_guard_ratio == 0 else
                                             "inactive before its declared minimum steps; see targeted component check" if self.task["execution"]["steps"] <= self.recipe.d_guard_min_steps else "installed; see measured activation counters"},
                    component_transport=(self.transport_consumer.state_dict()
                        if hasattr(self, 'transport_consumer') else None))

    def provenance_state(self):
        """Include standalone coordinates omitted by model/optimizer tables."""
        names = {id(parameter): f"{role}.{name}" for role, model in self.models.items()
                 for name, parameter in model.named_parameters()}
        direct = [parameter for parameters in self.role_parameters.values() for parameter in parameters
                  if id(parameter) in self.direct_particle_ids]
        names.update({id(parameter): f"direct_particles.{index}" for index, parameter in enumerate(direct)})
        roles = {role: [{"index": index, "name": names[id(parameter)],
                    "representation": "direct_sample_coordinates" if id(parameter) in self.direct_particle_ids
                                      else "latent_prior_locations" if role == "prior" else "network",
                    "requires_grad": parameter.requires_grad, "value": parameter.detach().clone()}
                       for index, parameter in enumerate(parameters)]
                 for role, parameters in self.role_parameters.items()}
        return dict(models={name: model.state_dict() for name, model in self.models.items()},
                    **self.recipe.prior_regularizer_checkpoint(self.prior_regularizer),
                    role_parameters=roles,
                    optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
                    streams=self.context.streams.state_dict(), applied=self.receipt(),
                    component_transport=(self.transport_consumer.state_dict()
                        if hasattr(self, 'transport_consumer') else None))


def run_behavior(request: dict, task: dict, output_dir: Path | str, device="cpu") -> dict:
    """Execute one original behavioral objective through the shared public binder."""
    from benchmarks.locked_shared import two_pole, trajectory
    from benchmarks.locked_shared.hosts import (ae_gan_hold, cover_leftover, mid_scale_identity,
                                               residual_student, unipolar, unused_token_hold)
    modules = {module.__name__.rsplit(".", 1)[-1]: module for module in (
        two_pole, trajectory, ae_gan_hold, cover_leftover, mid_scale_identity,
        residual_student, unipolar, unused_token_hold)}
    # These are deliberate zero-origin controls, not unspecified random layers.
    # Declare their constants through the public initializer registry once.
    for cls in (unipolar.FreeOriginResidual, unused_token_hold.SharedSlotStudent,
                mid_scale_identity.MidScaleResidual, cover_leftover._Residual):
        init.register(cls, lambda model: {name: init.KEEP for name, _ in model.named_parameters(recurse=False)})
    name = task["execution"].get("host", task["id"])
    if name not in modules:
        raise CapabilityError([f"unsupported behavioral host {name}"])
    if task["execution"].get("prior_applicability") == "not_sampled" and task["execution"]["prior"]["kind"] != "particle_cloud":
        raise CapabilityError(["a nonsampled parameter host cannot claim MoG sampling support"])
    module = modules[name]
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CapabilityError(["CUDA behavioral task requires an available GPU"])
    components = BehaviorComponents(request, task, device=device)
    seed = components.context.streams.seed
    steps = task["execution"]["steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("behavioral task needs a positive step budget")
    started = time.monotonic()
    previous_threads = torch.get_num_threads()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if task["execution"].get("horizon_diagnostic") is not None:
        from .two_pole_observer import TwoPoleObserver
        components.diagnostic = TwoPoleObserver(task, output_dir)
    try:
        torch.set_num_threads(1)
        # Retained fixture constructors call manual_seed, which also seeds every
        # visible GPU. Restore every affected caller stream, even on exceptions.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), torch.device(device), components.context.streams.fork("data", component="host", purpose="batches"), ExitStack() as stack:
            # Frozen tensor fixtures were created at module import on CPU.
            # Move their exact values alongside the host, without redrawing.
            for key, value in vars(module).copy().items():
                if isinstance(value, torch.Tensor):
                    stack.enter_context(patch.object(module, key, value.to(device)))
            stack.enter_context(patch.object(module, "checkpoint", components.checkpoint))
            stack.enter_context(patch.object(module, "schedule_optimizer", components.schedule_optimizer))
            if hasattr(module, "PROTOCOL"):
                overrides = {k: v for k, v in dict(steps=steps, seed=seed).items() if k in module.PROTOCOL}
                if components.uses_recipe_prior:
                    overrides.update({key: value for key, value in {
                        "particle_l2": components.recipe.prior_l2,
                        "vicreg_weight": components.recipe.prior_reg,
                        "vicreg_std": components.recipe.prior_reg_target_std}.items() if key in module.PROTOCOL})
                stack.enter_context(patch.dict(module.PROTOCOL, overrides))
            # These existing host scorers can run outside checkpoint callbacks.
            # Isolate every call, including AE's extra progress measurements.
            if name == "ae_gan_hold":
                original = module.evaluate
                def evaluate(*args, **kwargs):
                    if components.noise._evaluating:
                        return original(*args, **kwargs)
                    with components.noise.evaluation(steps):
                        return original(*args, **kwargs)
                stack.enter_context(patch.object(module, "evaluate", evaluate))
            kwargs = dict(noise_policy=components.noise, components=components)
            if name == "two_pole":
                stack.enter_context(patch.object(module, "TOY_STEPS", steps))
                stack.enter_context(patch.object(module, "TOY_SEED", seed))
                raw = module.train(**kwargs)
            elif name == "trajectory":
                raw = module.train(diagnostics=True, **kwargs)
            elif name == "residual_student":
                raw = module.train(**kwargs)
            elif name == "unipolar":
                raw = module.run_arm("locked_rpgan", steps=steps, seed=seed, **kwargs)
            elif name == "unused_token_hold":
                raw = module.train(module.UnusedHoldRecipe(name="forge", steps=steps, seed=seed), **kwargs)
            elif name == "ae_gan_hold":
                raw = module.train(module.HoldConfig(name="forge", steps=steps, seed=seed), **kwargs)
                raw.pop("cfg", None)
            elif name == "cover_leftover":
                raw = module.fit_cover_leftover(module.CoverRecipe(steps=steps, seed=seed), **kwargs)
            else:
                raw = module.run_arm("locked", steps=steps, seed=seed, **kwargs)
        live = raw.get("live", raw)
        checkpoint = components.provenance_state()
        torch.save(checkpoint, output_dir / "component-state.pt")
        metrics = {key: live[key] for key, _, _ in task["evaluation"]["thresholds"] if key in live}
        policy = executed_receipt(BEHAVIOR_POLICIES[name], eval_output_noise=POLICIES[BEHAVIOR_POLICIES[name]])
        result = dict(task_id=task["id"], evidence=dict(observations=components.observations, live=metrics,
                      scoring_weights="live", guards=components.guards(), **policy),
                      execution_path="public_components", device=str(device), applied=components.receipt(),
                      raw=raw, cost=dict(wall_seconds=time.monotonic()-started))
        if components.uses_recipe_prior:
            from .priors import prior_policy_receipt
            result["evidence"]["prior_policy"] = prior_policy_receipt(components.recipe,
                *(model for name, model in components.models.items() if name.startswith("prior")))
            for name, field in (("particle_l2", "prior_l2"), ("vicreg_weight", "prior_reg"),
                                ("vicreg_std", "prior_reg_target_std")):
                if name in raw:
                    raw[name] = getattr(components.recipe, field)
        outputs = getattr(components, "constraint_geometry_outputs", None)
        if outputs:
            path = output_dir / "constraint_geometry-scored-outputs.pt"
            torch.save(outputs, path)
            result["evidence"]["saved_observer_outputs"] = dict(path=path.name, bytes=path.stat().st_size,
                                                                  sha256=file_hash(path))
        result["evidence"]["provenance_checkpoint"] = save_provenance_checkpoint(
            output_dir, checkpoint,
            completed_steps=min(count for role, count in result["evidence"]["guards"]["optimizer_updates"].items()
                                if any(parameter.requires_grad for parameter in components.role_parameters.get(role, ()))))
        if components.diagnostic is not None:
            result["evidence"].update(diagnostic_observations=components.diagnostic_observations,
                                     horizon_diagnostic=components.diagnostic.finish())
        atomic_json(output_dir / "result.json", result)
        return result
    finally:
        if components.diagnostic is not None:
            components.diagnostic.close()
        torch.set_num_threads(previous_threads)
